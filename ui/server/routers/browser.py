"""数据浏览路由：股票列表、行情、K线、实时行情、总览、自选股。"""
from __future__ import annotations

import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from .. import app
from ..calendar_validation import InvalidCalendarError
from ..config import LIVE_DATA_DIR
from ..datadir import DataDir
from ..dependencies import get_data
from ..helpers import _last_non_nan, _prev_non_nan_nonzero, normalize_day_str, normalize_symbol, truthy_param
from ..schemas import WatchlistAddRequest
from ..stock_cache import load_stock_summary
from ..sync import get_data_health_snapshot
from ..tdx_quote import TDXQuote

_log = logging.getLogger(__name__)

router = APIRouter(tags=["browser"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _watchlist_path() -> Path:
    return LIVE_DATA_DIR / "watchlist" / "latest.json"


def _load_watchlist_file() -> dict:
    path = _watchlist_path()
    if not path.exists():
        return {"symbols": [], "updatedAt": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _log.warning("watchlist file is invalid, fallback to empty list", exc_info=True)
        return {"symbols": [], "updatedAt": None}

    symbols = data.get("symbols", [])
    if not isinstance(symbols, list):
        symbols = []
    normalized = []
    seen: set[str] = set()
    for symbol in symbols:
        code = normalize_symbol(symbol)
        if code and code not in seen:
            normalized.append(code)
            seen.add(code)
    return {"symbols": normalized, "updatedAt": data.get("updatedAt")}


def _save_watchlist_file(data: dict) -> None:
    path = _watchlist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _watchlist_response(raw: dict) -> dict:
    symbols = raw.get("symbols", [])
    return {"symbols": symbols, "updatedAt": raw.get("updatedAt"), "count": len(symbols)}


def _fetch_tdx_quotes(symbols: list[str]) -> dict:
    if not symbols or not app.tdx_quote:
        return {}
    return app.tdx_quote.fetch_quotes(symbols)


def _build_day_bar_from_minutes(minute_bars: list) -> Optional[dict]:
    if not minute_bars:
        return None
    trade_date = normalize_day_str(minute_bars[-1].get("date"))
    bars = [item for item in minute_bars if normalize_day_str(item.get("date")) == trade_date]
    if not trade_date or not bars:
        return None
    return {
        "date": trade_date,
        "open": round(float(bars[0]["open"]), 2),
        "high": round(max(float(item["high"]) for item in bars), 2),
        "low": round(min(float(item["low"]) for item in bars), 2),
        "close": round(float(bars[-1]["close"]), 2),
        "volume": int(sum(int(item.get("volume") or 0) for item in bars)),
        "realtime": True,
        "partial": True,
    }


def _merge_realtime_day_bar(kline_data, today_bar, start=None, end=None) -> list:
    if not today_bar:
        return list(kline_data or [])
    trade_date = today_bar["date"]
    if start and trade_date < str(start):
        return list(kline_data or [])
    if end and trade_date > str(end):
        return list(kline_data or [])
    result = list(kline_data or [])
    if not result:
        return [today_bar]
    last_date = normalize_day_str(result[-1].get("date"))
    if last_date == trade_date:
        merged = dict(result[-1])
        merged.update(today_bar)
        result[-1] = merged
        return result
    if last_date < trade_date:
        result.append(today_bar)
    return result


def _build_daily_quote(kline_data: list) -> dict:
    if not kline_data:
        return {}
    last = kline_data[-1]
    quote = {"close": last["close"], "change": 0, "changePct": "0", "lastClose": None}
    if len(kline_data) > 1:
        prev = kline_data[-2]["close"]
        quote["lastClose"] = prev
        if prev != 0:
            quote["change"] = round(last["close"] - prev, 2)
            quote["changePct"] = str(round((last["close"] - prev) / prev * 100, 2))
    return quote


def _with_realtime_day_bar(symbol, freq, start, end, kline_data):
    if freq != "day" or not app.tdx_quote:
        return list(kline_data or []), {"included": False}
    # 如果本地数据已经包含当日完整 K 线（自动同步完成后），跳过实时叠加
    if kline_data and normalize_day_str(kline_data[-1].get("date")) == str(_date.today()):
        # 保留本地完整数据，标记为已包含（来源为 local）
        return list(kline_data), {"included": True, "date": str(_date.today()), "source": "local", "partial": False}
    minute_bars = app.tdx_quote.get_today_kline(symbol)
    today_bar = _build_day_bar_from_minutes(minute_bars)
    if not today_bar:
        # TDX 不可用时，用东方财富实时行情构造当日 K 线
        today_bar = app.tdx_quote.fetch_today_day_bar_from_eastmoney(symbol)
    if not today_bar:
        return list(kline_data or []), {"included": False}
    merged = _merge_realtime_day_bar(kline_data, today_bar, start=start, end=end)
    included = bool(merged) and normalize_day_str(merged[-1].get("date")) == today_bar["date"]
    return merged, {
        "included": included,
        "date": today_bar["date"] if included else None,
        "source": "minute" if included else None,
        "partial": bool(today_bar.get("partial")) if included else False,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/browser/stocks")
def stocks(data: DataDir = Depends(get_data)):
    # 尝试预计算缓存（内存 → JSON 文件），可提速 10~50×
    cached = load_stock_summary()
    if cached is not None:
        return {"stocks": cached}

    # 降级：逐个读取 bin 文件（缓存尚未就绪时的首次加载）
    _log.warning("Stock summary cache miss, falling back to per-stock binary reads")
    names = data.get_names()
    instruments = data.get_instruments()
    result = []
    for sym, start, end in instruments:
        code6 = sym[2:] if len(sym) >= 3 and sym[:2] in ("SZ", "SH", "BJ") else sym
        name = names.get(code6, "")
        _, close_vals = data.read_field(sym, "close", "day")
        item = {"symbol": sym, "name": name, "startDate": start, "endDate": end}
        if close_vals is not None and len(close_vals) > 0:
            raw_c = _last_non_nan(close_vals)
            if raw_c is not None:
                c = round(raw_c, 2)
                item["close"] = c
                prev = _prev_non_nan_nonzero(close_vals, len(close_vals) - 1)
                if prev is not None:
                    item["change"] = round(c - prev, 2)
                    item["changePct"] = round((c - prev) / prev * 100, 2)
        _, vol_vals = data.read_field(sym, "volume", "day")
        if vol_vals is not None and len(vol_vals) > 0:
            raw_v = _last_non_nan(vol_vals)
            if raw_v is not None:
                item["volume"] = int(raw_v)
        result.append(item)

    # 后台异步构建缓存，下次访问就不慢了
    import threading
    from ..stock_cache import build_stock_summary
    threading.Thread(target=build_stock_summary, args=(data,), daemon=True).start()

    return {"stocks": result}


@router.get("/browser/quotes")
def quotes(
    symbols: str = Query("", description="逗号分隔的股票代码"),
    data: DataDir = Depends(get_data),
):
    if not symbols:
        return {"quotes": {}}
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return {"quotes": {}}

    rt = _fetch_tdx_quotes(symbol_list)
    result = {}
    for sym in symbol_list:
        normalized = normalize_symbol(sym)
        q = rt.get(sym.upper()) or rt.get(normalized)
        if q and q.get("price", 0) > 0:
            price = q["price"]
            last_close = q.get("lastClose", 0)
            chg = round(price - last_close, 2) if last_close > 0 else 0
            chg_pct = round(chg / last_close * 100, 2) if last_close > 0 else 0
            result[normalized] = {
                "close": round(price, 2),
                "change": chg,
                "changePct": chg_pct,
                "volume": q.get("vol", 0),
                "amount": q.get("amount", 0),
                "source": q.get("source", "tdx"),
            }
    if not result:
        for sym in symbol_list:
            _, close_vals = data.read_field(sym, "close", "day")
            if close_vals is None or len(close_vals) == 0:
                continue
            raw_c = _last_non_nan(close_vals)
            if raw_c is None:
                continue
            c = round(raw_c, 2)
            chg = 0
            chg_pct = 0.0
            prev = _prev_non_nan_nonzero(close_vals, len(close_vals) - 1)
            if prev is not None:
                chg = round(c - prev, 2)
                chg_pct = round((c - prev) / prev * 100, 2)
            vol = 0
            _, vol_vals = data.read_field(sym, "volume", "day")
            if vol_vals is not None and len(vol_vals) > 0:
                raw_v = _last_non_nan(vol_vals)
                if raw_v is not None:
                    vol = int(raw_v)
            result[normalize_symbol(sym)] = {
                "close": c,
                "change": chg,
                "changePct": chg_pct,
                "volume": vol,
                "source": "local",
            }
    return {"quotes": result}


@router.get("/browser/indices")
def indices():
    INDEX_SYMBOLS = [
        ("SH000001", "上证指数"),
        ("SZ399001", "深证成指"),
        ("SZ399006", "创业板指"),
        ("SH000688", "科创50"),
        ("SH000300", "沪深300"),
        ("SH000905", "中证500"),
    ]
    symbols = [s for s, _ in INDEX_SYMBOLS]
    rt = _fetch_tdx_quotes(symbols)
    result = []
    for sym, name in INDEX_SYMBOLS:
        q = rt.get(sym)
        if q and q.get("price", 0) > 0:
            price = q["price"]
            last_close = q.get("lastClose", 0)
            chg = round(price - last_close, 2) if last_close > 0 else 0
            chg_pct = round(chg / last_close * 100, 2) if last_close > 0 else 0
            result.append({
                "symbol": sym,
                "name": name,
                "price": round(price, 2),
                "change": chg,
                "changePct": chg_pct,
                "open": round(q.get("open", 0), 2),
                "high": round(q.get("high", 0), 2),
                "low": round(q.get("low", 0), 2),
                "vol": q.get("vol", 0),
                "amount": q.get("amount", 0),
                "lastClose": round(last_close, 2),
            })
        else:
            result.append({"symbol": sym, "name": name, "price": 0, "change": 0, "changePct": 0})
    return {"indices": result}


@router.get("/browser/kline/{symbol}")
def kline(
    symbol: str,
    freq: str = Query("day"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    includeRealtime: str = Query("0"),
    data: DataDir = Depends(get_data),
):
    if freq == "1d":
        freq = "day"
    symbol = normalize_symbol(symbol)
    include_realtime = truthy_param(includeRealtime)
    kline_data = data.get_kline(symbol, freq, start, end)
    realtime = {"included": False}
    if include_realtime:
        kline_data, realtime = _with_realtime_day_bar(symbol, freq, start, end, kline_data)
    if not kline_data:
        return {"kline": [], "quote": {}, "realtime": realtime}
    quote = _build_daily_quote(kline_data)
    return {"kline": kline_data, "quote": quote, "realtime": realtime}


@router.get("/realtime/quote/{symbol}")
def realtime_quote(symbol: str):
    if not app.tdx_quote:
        return {"ok": False, "quote": {}}
    q = app.tdx_quote.get_quote(symbol)
    if q:
        return {"ok": True, "quote": q}
    return {"ok": False, "quote": {}}


@router.get("/realtime/kline/{symbol}")
def realtime_kline(symbol: str):
    if not app.tdx_quote:
        return {"kline": [], "quote": {}}
    kline_data = app.tdx_quote.get_today_kline(symbol)
    if kline_data:
        last = kline_data[-1]
        quote = {"close": last["close"], "change": 0, "changePct": "0"}
        if len(kline_data) > 1:
            prev = kline_data[-2]["close"]
            if prev != 0:
                quote["change"] = round(last["close"] - prev, 2)
                quote["changePct"] = str(round((last["close"] - prev) / prev * 100, 2))
        return {"kline": kline_data, "quote": quote}
    return {"kline": [], "quote": {}}


@router.get("/overview")
def overview(data: DataDir = Depends(get_data)):
    instruments = data.get_instruments() if data else []
    names = data.get_names() if data else {}
    try:
        cal = data.read_calendar("day") if data else []
    except InvalidCalendarError:
        cal = []
    field_stats = []
    for sym, _, _ in instruments[:30]:
        _, close = data.read_field(sym, "close", "day") if data else (None, None)
        if close is not None and len(close) > 0:
            non_nan = sum(1 for v in close if not np.isnan(float(v)))
            pct = round(non_nan / len(close) * 100, 1)
        else:
            pct = 0
        code6 = sym[2:] if len(sym) >= 3 else sym
        field_stats.append({"symbol": sym, "name": names.get(code6, ""), "close": pct, "volume": pct})

    health = get_data_health_snapshot(data.data_dir) if data and data.data_dir else {
        "calendarLastDate": None,
        "effectiveLastDate": None,
        "marketEffectiveLastDate": None,
        "equityCoverageAtLastDate": 0.0,
        "equityCount": 0,
        "equityCoveredAtLastDate": 0,
        "calendarCoverage": 0.0,
        "calendarCoveredEquities": 0,
        "calendarHealthy": True,
        "calendarInvalidLineCount": 0,
        "sampleInvalidCalendarLines": [],
        "calendarDuplicateCount": 0,
        "calendarOrdered": True,
    }
    completeness = round(sum(s["close"] for s in field_stats) / len(field_stats), 1) if field_stats else 0
    return {
        "stockCount": len(instruments),
        "calendarDays": len(cal),
        "lastUpdate": health["effectiveLastDate"],
        "effectiveLastDate": health["effectiveLastDate"],
        "calendarLastDate": health["calendarLastDate"],
        "marketEffectiveLastDate": health["marketEffectiveLastDate"],
        "equityCoverageAtLastDate": health["equityCoverageAtLastDate"],
        "equityCoveredAtLastDate": health["equityCoveredAtLastDate"],
        "equityCount": health["equityCount"],
        "calendarCoverage": health["calendarCoverage"],
        "calendarCoveredEquities": health["calendarCoveredEquities"],
        "calendarHealthy": health.get("calendarHealthy", True),
        "calendarInvalidLineCount": health.get("calendarInvalidLineCount", 0),
        "sampleInvalidCalendarLines": health.get("sampleInvalidCalendarLines", []),
        "calendarDuplicateCount": health.get("calendarDuplicateCount", 0),
        "calendarOrdered": health.get("calendarOrdered", True),
        "completeness": completeness,
        "fieldStats": field_stats,
    }


@router.get("/watchlist")
def watchlist_get():
    return _watchlist_response(_load_watchlist_file())


@router.post("/watchlist")
def watchlist_add(body: WatchlistAddRequest):
    symbol = normalize_symbol(body.symbol)
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol 不能为空")
    raw = _load_watchlist_file()
    symbols = raw.get("symbols", [])
    if symbol not in symbols:
        symbols.append(symbol)
    raw["symbols"] = symbols
    raw["updatedAt"] = str(_date.today())
    _save_watchlist_file(raw)
    return {"ok": True, **_watchlist_response(raw)}


@router.delete("/watchlist/{symbol}")
def watchlist_remove(symbol: str):
    normalized = normalize_symbol(symbol)
    if not normalized:
        raise HTTPException(status_code=400, detail="symbol 不能为空")
    raw = _load_watchlist_file()
    raw["symbols"] = [item for item in raw.get("symbols", []) if item != normalized]
    raw["updatedAt"] = str(_date.today())
    _save_watchlist_file(raw)
    return {"ok": True, **_watchlist_response(raw)}
