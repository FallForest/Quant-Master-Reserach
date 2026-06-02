"""数据浏览：股票列表、行情、K线、实时行情、总览、自选股。"""
import json
import logging
from datetime import date as _date
from pathlib import Path

import numpy as np

from .. import app
from ..sync import get_data_health_snapshot

_log = logging.getLogger(__name__)

_LIVE_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "live_data"


def _normalize_symbol(symbol):
    s = str(symbol or "").strip().upper()
    if not s:
        return ""
    if s.startswith(("SH", "SZ", "BJ")):
        return s
    if s.startswith("6"):
        return "SH" + s
    if s.startswith(("4", "8", "9")):
        return "BJ" + s
    return "SZ" + s


def _watchlist_path():
    return _LIVE_DATA_DIR / "watchlist" / "latest.json"


def _load_watchlist_file():
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
    seen = set()
    for symbol in symbols:
        code = _normalize_symbol(symbol)
        if code and code not in seen:
            normalized.append(code)
            seen.add(code)
    return {
        "symbols": normalized,
        "updatedAt": data.get("updatedAt"),
    }


def _save_watchlist_file(data):
    path = _watchlist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _watchlist_response(raw):
    symbols = raw.get("symbols", [])
    return {
        "symbols": symbols,
        "updatedAt": raw.get("updatedAt"),
        "count": len(symbols),
    }


def _last_non_nan(arr, default=None):
    """从数组末尾向前扫描，返回最后一个非 NaN 的 float 值。"""
    for j in range(len(arr) - 1, -1, -1):
        v = float(arr[j])
        if not np.isnan(v):
            return v
    return default


def _prev_non_nan_nonzero(arr, before_idx):
    """从 before_idx-1 向前扫描，返回第一个非 NaN 且非零的 float 值。"""
    for j in range(before_idx - 1, -1, -1):
        v = float(arr[j])
        if not np.isnan(v) and v != 0:
            return v
    return None


def _fetch_tdx_quotes(symbols):
    """批量获取 TDX 实时行情。线程安全由 TDXQuote._call_lock 保证。"""
    if not symbols or not app.tdx_quote:
        return {}
    return app.tdx_quote.fetch_quotes(symbols)


def _normalize_day_str(value):
    return str(value or "")[:10]


def _truthy_param(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _build_day_bar_from_minutes(minute_bars):
    if not minute_bars:
        return None
    trade_date = _normalize_day_str(minute_bars[-1].get("date"))
    bars = [item for item in minute_bars if _normalize_day_str(item.get("date")) == trade_date]
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


def _merge_realtime_day_bar(kline_data, today_bar, start=None, end=None):
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

    last_date = _normalize_day_str(result[-1].get("date"))
    if last_date == trade_date:
        merged = dict(result[-1])
        merged.update(today_bar)
        result[-1] = merged
        return result
    if last_date < trade_date:
        result.append(today_bar)
    return result


def _build_daily_quote(kline_data):
    if not kline_data:
        return {}
    last = kline_data[-1]
    quote = {
        "close": last["close"],
        "change": 0,
        "changePct": "0",
        "lastClose": None,
    }
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

    minute_bars = app.tdx_quote.get_today_kline(symbol)
    today_bar = _build_day_bar_from_minutes(minute_bars)
    if not today_bar:
        return list(kline_data or []), {"included": False}

    merged = _merge_realtime_day_bar(kline_data, today_bar, start=start, end=end)
    included = bool(merged) and _normalize_day_str(merged[-1].get("date")) == today_bar["date"]
    return merged, {
        "included": included,
        "date": today_bar["date"] if included else None,
        "source": "minute" if included else None,
        "partial": bool(today_bar.get("partial")) if included else False,
    }


def stocks(rh):
    data = app.data
    names = data.get_names()
    instruments = data.get_instruments()
    result = []
    for sym, start, end in instruments:
        code6 = sym[2:] if len(sym) >= 3 and sym[:2] in ("SZ", "SH", "BJ") else sym
        name = names.get(code6, "")
        close_idx, close_vals = data.read_field(sym, "close", "day")
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
        vol_idx, vol_vals = data.read_field(sym, "volume", "day")
        if vol_vals is not None and len(vol_vals) > 0:
            raw_v = _last_non_nan(vol_vals)
            if raw_v is not None:
                item["volume"] = int(raw_v)
        result.append(item)

    # 初始加载只返回二进制文件数据，不拉 TDX（避免 3800 只股票批量请求卡死）
    # 实时行情由前端 1s 轮询 /api/browser/quotes 更新
    rh._json_response({"stocks": result})


def quotes(rh):
    """返回指定股票的实时行情，用于列表 1s 轮询刷新。

    前端通过 ?symbols=SH600519,SZ000001,... 传入当前页股票代码。
    TDX 不可用时回退到二进制文件数据。
    """
    params = rh._query_params()
    raw_symbols = params.get("symbols", "")
    if not raw_symbols:
        rh._json_response({"quotes": {}})
        return
    symbols = [s.strip() for s in raw_symbols.split(",") if s.strip()]
    if not symbols:
        rh._json_response({"quotes": {}})
        return

    data = app.data
    rt = _fetch_tdx_quotes(symbols)
    result = {}
    for sym in symbols:
        normalized = _normalize_symbol(sym)
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
    # TDX 不可用时回退到二进制文件数据
    if not result:
        for sym in symbols:
            close_idx, close_vals = data.read_field(sym, "close", "day")
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
            vol_idx, vol_vals = data.read_field(sym, "volume", "day")
            if vol_vals is not None and len(vol_vals) > 0:
                raw_v = _last_non_nan(vol_vals)
                if raw_v is not None:
                    vol = int(raw_v)
            result[_normalize_symbol(sym)] = {"close": c, "change": chg, "changePct": chg_pct, "volume": vol, "source": "local"}
    rh._json_response({"quotes": result})


def kline(rh, symbol):
    params = rh._query_params()
    freq = params.get("freq", "day")
    if freq == "1d":
        freq = "day"
    start = params.get("start")
    end = params.get("end")
    include_realtime = _truthy_param(params.get("includeRealtime"))
    kline_data = app.data.get_kline(symbol, freq, start, end)
    realtime = {"included": False}
    if include_realtime:
        kline_data, realtime = _with_realtime_day_bar(symbol, freq, start, end, kline_data)
    if not kline_data:
        rh._json_response({"kline": [], "quote": {}, "realtime": realtime})
        return
    quote = _build_daily_quote(kline_data)
    rh._json_response({"kline": kline_data, "quote": quote, "realtime": realtime})


def realtime_quote(rh, symbol):
    if not app.tdx_quote:
        return rh._json_response({"ok": False, "quote": {}})
    q = app.tdx_quote.get_quote(symbol)
    if q:
        rh._json_response({"ok": True, "quote": q})
    else:
        rh._json_response({"ok": False, "quote": {}})


def realtime_kline(rh, symbol):
    if not app.tdx_quote:
        return rh._json_response({"kline": [], "quote": {}})
    kline_data = app.tdx_quote.get_today_kline(symbol)
    if kline_data:
        last = kline_data[-1]
        quote = {"close": last["close"], "change": 0, "changePct": "0"}
        if len(kline_data) > 1:
            prev = kline_data[-2]["close"]
            if prev != 0:
                quote["change"] = round(last["close"] - prev, 2)
                quote["changePct"] = str(round((last["close"] - prev) / prev * 100, 2))
        rh._json_response({"kline": kline_data, "quote": quote})
    else:
        rh._json_response({"kline": [], "quote": {}})


def indices(rh):
    """返回主要大盘指数的实时行情。"""
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
    rh._json_response({"indices": result})


def overview(rh):
    data = app.data
    instruments = data.get_instruments() if data else []
    names = data.get_names() if data else {}
    cal = data.read_calendar("day") if data else []
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
    }
    completeness = round(sum(s["close"] for s in field_stats) / len(field_stats), 1) if field_stats else 0
    rh._json_response({
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
        "completeness": completeness,
        "fieldStats": field_stats,
    })


def watchlist(rh):
    rh._json_response(_watchlist_response(_load_watchlist_file()))



def watchlist_add(rh):
    try:
        body = rh._read_body()
    except Exception:
        return

    symbol = _normalize_symbol(body.get("symbol"))
    if not symbol:
        return rh._json_response({"error": "symbol 不能为空"}, status=400)

    raw = _load_watchlist_file()
    symbols = raw.get("symbols", [])
    if symbol not in symbols:
        symbols.append(symbol)
    raw["symbols"] = symbols
    raw["updatedAt"] = str(_date.today())
    _save_watchlist_file(raw)
    rh._json_response({"ok": True, **_watchlist_response(raw)})



def watchlist_remove(rh, symbol):
    normalized = _normalize_symbol(symbol)
    if not normalized:
        return rh._json_response({"error": "symbol 不能为空"}, status=400)

    raw = _load_watchlist_file()
    raw["symbols"] = [item for item in raw.get("symbols", []) if item != normalized]
    raw["updatedAt"] = str(_date.today())
    _save_watchlist_file(raw)
    rh._json_response({"ok": True, **_watchlist_response(raw)})
