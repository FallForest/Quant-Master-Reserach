"""数据浏览：股票列表、行情、K线、实时行情、总览。"""
import time

import numpy as np

from .. import app
from ..sync import _get_last_update_date


def _fetch_tdx_quotes(symbols):
    """批量获取 TDX 实时行情，返回 {symbol: {price, lastClose, ...}}。"""
    if not symbols or not app.tdx_quote:
        return {}
    pairs = []
    for sym in symbols:
        try:
            pairs.append(app.tdx_quote._parse_symbol(sym))
        except Exception:
            continue
    if not pairs:
        return {}

    from pytdx.hq import TdxHq_API
    result = {}
    try:
        api = None
        for ip, port in app.tdx_quote.TDX_HOSTS:
            try:
                api = TdxHq_API()
                api.connect(ip, port, time_out=5)
                break
            except Exception:
                continue
        if api is None:
            return {}
        try:
            # pytdx 每批最多约 80 只
            for i in range(0, len(pairs), 80):
                batch = pairs[i:i + 80]
                data = api.get_security_quotes(batch)
                if data:
                    for d in data:
                        code = d.get("code", "")
                        # 确定市场前缀
                        market = d.get("market", 1)
                        prefix = "SH" if market == 1 else "SZ"
                        sym_key = f"{prefix}{code}"
                        result[sym_key] = {
                            "price": d.get("price", 0),
                            "lastClose": d.get("last_close", 0),
                            "open": d.get("open", 0),
                            "high": d.get("high", 0),
                            "low": d.get("low", 0),
                            "vol": d.get("vol", 0),
                            "amount": d.get("amount", 0),
                        }
        finally:
            try:
                api.disconnect()
            except Exception:
                pass
    except Exception:
        pass
    return result


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
            c = None
            for j in range(len(close_vals) - 1, -1, -1):
                v = float(close_vals[j])
                if not np.isnan(v):
                    c = round(v, 2)
                    break
            if c is not None:
                item["close"] = c
                for j in range((len(close_vals) - 2) if c is not None else len(close_vals) - 1, -1, -1):
                    prev = float(close_vals[j])
                    if not np.isnan(prev) and prev != 0:
                        item["change"] = round(c - prev, 2)
                        item["changePct"] = round((c - prev) / prev * 100, 2)
                        break
        vol_idx, vol_vals = data.read_field(sym, "volume", "day")
        if vol_vals is not None and len(vol_vals) > 0:
            for j in range(len(vol_vals) - 1, -1, -1):
                v = float(vol_vals[j])
                if not np.isnan(v):
                    item["volume"] = int(v)
                    break
        result.append(item)

    # 用 TDX 实时行情覆盖价格数据
    all_symbols = [s["symbol"] for s in result]
    rt = _fetch_tdx_quotes(all_symbols)
    if rt:
        for item in result:
            q = rt.get(item["symbol"])
            if not q or q.get("price", 0) == 0:
                continue
            price = q["price"]
            last_close = q.get("lastClose", 0)
            item["close"] = round(price, 2)
            if last_close > 0:
                item["change"] = round(price - last_close, 2)
                item["changePct"] = round((price - last_close) / last_close * 100, 2)
            item["volume"] = q.get("vol", item.get("volume", 0))

    rh._json_response({"stocks": result})


def quotes(rh):
    """返回所有股票的实时行情，用于列表 3s 轮询刷新。TDX 不可用时回退到二进制文件。"""
    data = app.data
    instruments = data.get_instruments()
    symbols = [sym for sym, _, _ in instruments]
    rt = _fetch_tdx_quotes(symbols)
    result = {}
    for sym, _, _ in instruments:
        q = rt.get(sym)
        if q and q.get("price", 0) > 0:
            price = q["price"]
            last_close = q.get("lastClose", 0)
            chg = round(price - last_close, 2) if last_close > 0 else 0
            chg_pct = round(chg / last_close * 100, 2) if last_close > 0 else 0
            result[sym] = {
                "close": round(price, 2),
                "change": chg,
                "changePct": chg_pct,
                "volume": q.get("vol", 0),
                "amount": q.get("amount", 0),
            }
    # TDX 不可用时回退到二进制文件数据
    if not result:
        for sym, _, _ in instruments:
            close_idx, close_vals = data.read_field(sym, "close", "day")
            if close_vals is None or len(close_vals) == 0:
                continue
            c = None
            for j in range(len(close_vals) - 1, -1, -1):
                v = float(close_vals[j])
                if not np.isnan(v):
                    c = round(v, 2)
                    break
            if c is None:
                continue
            chg = 0
            chg_pct = 0.0
            for j in range(len(close_vals) - 2, -1, -1):
                prev = float(close_vals[j])
                if not np.isnan(prev) and prev != 0:
                    chg = round(c - prev, 2)
                    chg_pct = round((c - prev) / prev * 100, 2)
                    break
            vol = 0
            vol_idx, vol_vals = data.read_field(sym, "volume", "day")
            if vol_vals is not None and len(vol_vals) > 0:
                for j in range(len(vol_vals) - 1, -1, -1):
                    v = float(vol_vals[j])
                    if not np.isnan(v):
                        vol = int(v)
                        break
            result[sym] = {"close": c, "change": chg, "changePct": chg_pct, "volume": vol}
    rh._json_response({"quotes": result})


def kline(rh, symbol):
    params = rh._query_params()
    freq = params.get("freq", "day")
    if freq == "1d":
        freq = "day"
    start = params.get("start")
    end = params.get("end")
    kline_data = app.data.get_kline(symbol, freq, start, end)
    if not kline_data:
        rh._json_response({"kline": [], "quote": {}})
        return
    last = kline_data[-1]
    quote = {"close": last["close"], "change": 0, "changePct": "0"}
    if len(kline_data) > 1:
        prev = kline_data[-2]["close"]
        if prev != 0:
            quote["change"] = round(last["close"] - prev, 2)
            quote["changePct"] = round((last["close"] - prev) / prev * 100, 2)
    rh._json_response({"kline": kline_data, "quote": quote})


def realtime_quote(rh, symbol):
    q = app.tdx_quote.get_quote(symbol)
    if q:
        rh._json_response({"ok": True, "quote": q})
    else:
        rh._json_response({"ok": False, "quote": {}})


def realtime_kline(rh, symbol):
    kline_data = app.tdx_quote.get_today_kline(symbol)
    if kline_data:
        last = kline_data[-1]
        quote = {"close": last["close"], "change": 0, "changePct": "0"}
        if len(kline_data) > 1:
            prev = kline_data[-2]["close"]
            if prev != 0:
                quote["change"] = round(last["close"] - prev, 2)
                quote["changePct"] = round((last["close"] - prev) / prev * 100, 2)
        rh._json_response({"kline": kline_data, "quote": quote})
    else:
        rh._json_response({"kline": [], "quote": {}})


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
    last_update = None
    if data and data.data_dir:
        last_update = _get_last_update_date(data.data_dir)
    completeness = round(sum(s["close"] for s in field_stats) / len(field_stats), 1) if field_stats else 0
    rh._json_response({
        "stockCount": len(instruments),
        "calendarDays": len(cal),
        "lastUpdate": last_update,
        "completeness": completeness,
        "fieldStats": field_stats,
    })
