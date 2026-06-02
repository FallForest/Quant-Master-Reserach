"""我的持仓：读取 live_data 持仓文件，结合实时行情返回持仓详情。"""
import json
import logging
from datetime import date as _date
from pathlib import Path

from .. import app

logger = logging.getLogger(__name__)

# live_data 目录位于项目根目录下
_LIVE_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "live_data"

_DEFAULT_FEE_SETTINGS = {
    "stock_commission_rate": 0.0001,
    "etf_commission_rate": 0.00005,
    "stamp_duty_rate": 0.0005,
    "sh_transfer_fee_rate": 0.00001,
}


def _normalize_symbol(symbol):
    s = str(symbol or "").strip().upper()
    if not s:
        return ""
    if s.startswith(("SH", "SZ", "BJ")):
        return s
    if s.startswith("6"):
        return "SH" + s
    return "SZ" + s


def _is_etf(instrument):
    normalized = _normalize_symbol(instrument)
    code6 = normalized[2:] if len(normalized) > 2 else normalized
    return code6.startswith(("15", "16", "18", "50", "51", "56", "58"))


def _is_shanghai_market(instrument):
    normalized = _normalize_symbol(instrument)
    return normalized.startswith("SH")


def _fee_settings_from_raw(raw):
    return {
        "stock_commission_rate": float(raw.get("stock_commission_rate", _DEFAULT_FEE_SETTINGS["stock_commission_rate"])),
        "etf_commission_rate": float(raw.get("etf_commission_rate", _DEFAULT_FEE_SETTINGS["etf_commission_rate"])),
        "stamp_duty_rate": float(raw.get("stamp_duty_rate", _DEFAULT_FEE_SETTINGS["stamp_duty_rate"])),
        "sh_transfer_fee_rate": float(raw.get("sh_transfer_fee_rate", _DEFAULT_FEE_SETTINGS["sh_transfer_fee_rate"])),
    }


def _calc_trade_fee(instrument, amount, side, fee_settings):
    normalized_side = str(side or "").strip().lower()
    trade_amount = abs(float(amount or 0))
    if trade_amount <= 0:
        return {"commission": 0.0, "stampDuty": 0.0, "transferFee": 0.0, "total": 0.0}

    is_etf = _is_etf(instrument)
    commission_rate = fee_settings["etf_commission_rate"] if is_etf else fee_settings["stock_commission_rate"]
    commission = trade_amount * commission_rate
    stamp_duty = trade_amount * fee_settings["stamp_duty_rate"] if (not is_etf and normalized_side == "sell") else 0.0
    transfer_fee = trade_amount * fee_settings["sh_transfer_fee_rate"] if _is_shanghai_market(instrument) else 0.0
    total = commission + stamp_duty + transfer_fee
    return {
        "commission": round(commission, 2),
        "stampDuty": round(stamp_duty, 2),
        "transferFee": round(transfer_fee, 2),
        "total": round(total, 2),
    }


def _lookup_name(names, instrument):
    if not names:
        return instrument
    normalized = _normalize_symbol(instrument)
    code6 = normalized[2:] if len(normalized) > 2 else normalized
    return (
        names.get(instrument)
        or names.get(instrument.upper())
        or names.get(normalized)
        or names.get(normalized.lower())
        or names.get(code6)
        or instrument
    )


def _lookup_quote_price(quotes, instrument, fallback):
    if not quotes:
        return fallback
    normalized = _normalize_symbol(instrument)
    code6 = normalized[2:] if len(normalized) > 2 else normalized
    quote = (
        quotes.get(instrument)
        or quotes.get(instrument.upper())
        or quotes.get(normalized)
        or quotes.get(normalized.lower())
        or quotes.get(code6)
    )
    if not isinstance(quote, dict):
        return fallback
    price = quote.get("price", fallback)
    try:
        price = float(price)
    except (TypeError, ValueError):
        return fallback
    return price if price > 0 else fallback


def _lookup_local_close(data, instrument, fallback):
    if data is None:
        return fallback
    normalized = _normalize_symbol(instrument)
    candidates = [instrument, normalized]
    code6 = normalized[2:] if len(normalized) > 2 else normalized
    if code6 not in candidates:
        candidates.append(code6)
    for candidate in candidates:
        try:
            bars = data.get_kline(candidate, freq="day")
        except Exception:
            continue
        if not bars:
            continue
        close_price = bars[-1].get("close")
        try:
            close_price = float(close_price)
        except (TypeError, ValueError):
            continue
        if close_price > 0:
            return close_price
    return fallback


def _ensure_dirs():
    (_LIVE_DATA_DIR / "positions").mkdir(parents=True, exist_ok=True)


def _load_positions_file():
    """读取 live_data/positions/latest.json，不存在则返回默认空持仓。"""
    path = _LIVE_DATA_DIR / "positions" / "latest.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"cash": 1000000.0, "positions": {}, "date": None, "total_assets": 1000000.0}

    capital_amount = data.get("capital_amount")
    if capital_amount is None:
        capital_amount = data.get("total_assets", data.get("cash", 1000000.0))

    fee_settings = _fee_settings_from_raw(data)

    data.setdefault("cash", 1000000.0)
    data.setdefault("positions", {})
    data.setdefault("date", None)
    data["stock_commission_rate"] = fee_settings["stock_commission_rate"]
    data["etf_commission_rate"] = fee_settings["etf_commission_rate"]
    data["stamp_duty_rate"] = fee_settings["stamp_duty_rate"]
    data["sh_transfer_fee_rate"] = fee_settings["sh_transfer_fee_rate"]
    data["trading_fee_rate"] = fee_settings["stock_commission_rate"]
    data["capital_amount"] = float(capital_amount)
    return data


def _save_positions_file(data):
    """写入 live_data/positions/latest.json。"""
    _ensure_dirs()
    path = _LIVE_DATA_DIR / "positions" / "latest.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _enrich_positions(raw, names=None, quotes=None):
    """将原始持仓数据加工为前端需要的格式，附加实时行情和股票名称。"""
    data = app.data
    tdx = app.tdx_quote

    cash = raw.get("cash", 0)
    raw_positions = raw.get("positions", {})
    date = raw.get("date")
    fee_settings = _fee_settings_from_raw(raw)
    capital_amount = raw.get("capital_amount", raw.get("total_assets", cash))

    # 获取股票名称
    if names is None and data is not None:
        try:
            names = data.get_names()
        except Exception:
            names = {}
    elif names is None:
        names = {}

    # 获取实时行情
    instruments = list(raw_positions.keys())
    if quotes is None and tdx is not None and instruments:
        try:
            quotes = tdx.fetch_quotes(instruments)
        except Exception:
            quotes = {}
    elif quotes is None:
        quotes = {}

    positions = []
    total_market_value = 0

    for instrument, pos in raw_positions.items():
        shares = pos.get("shares", 0)
        cost_price = pos.get("price", 0)

        # 实时价格：优先实时行情，其次本地日线收盘价，最后退回持仓成本
        current_price = _lookup_quote_price(quotes, instrument, 0)
        if current_price <= 0:
            current_price = _lookup_local_close(data, instrument, cost_price)

        market_value = shares * current_price
        total_market_value += market_value
        pnl = (current_price - cost_price) * shares
        pnl_pct = ((current_price / cost_price) - 1) * 100 if cost_price > 0 else 0

        positions.append({
            "instrument": instrument,
            "name": _lookup_name(names, instrument),
            "shares": shares,
            "costPrice": round(cost_price, 4),
            "currentPrice": round(current_price, 4),
            "marketValue": round(market_value, 2),
            "pnl": round(pnl, 2),
            "pnlPct": round(pnl_pct, 2),
            "instrumentType": "ETF" if _is_etf(instrument) else "股票",
        })

    # 计算权重
    for p in positions:
        p["weight"] = round(p["marketValue"] / total_market_value * 100, 2) if total_market_value > 0 else 0

    # 按市值降序排列
    positions.sort(key=lambda x: x["marketValue"], reverse=True)

    total_assets = cash + total_market_value
    total_cost = sum(p["costPrice"] * p["shares"] for p in positions)
    total_pnl = total_market_value - total_cost
    total_pnl_pct = (total_market_value / total_cost - 1) * 100 if total_cost > 0 else 0

    return {
        "date": date,
        "cash": round(cash, 2),
        "capitalAmount": round(float(capital_amount), 2),
        "feeSettings": {
            "stockCommissionRate": round(fee_settings["stock_commission_rate"], 6),
            "etfCommissionRate": round(fee_settings["etf_commission_rate"], 6),
            "stampDutyRate": round(fee_settings["stamp_duty_rate"], 6),
            "shTransferFeeRate": round(fee_settings["sh_transfer_fee_rate"], 6),
            "stockCommissionLabel": "股票万1",
            "etfCommissionLabel": "ETF万0.5",
            "stampDutyLabel": "印花税买入不收，卖出万5",
            "shTransferFeeLabel": "沪市过户费买卖十万1，深市包含在佣金里",
        },
        "feeRuleSummary": "股票万1，ETF万0.5；印花税买入不收、卖出万5；沪市过户费买卖十万1，深市包含在佣金里",
        "totalAssets": round(total_assets, 2),
        "totalMarketValue": round(total_market_value, 2),
        "totalPnl": round(total_pnl, 2),
        "totalPnlPct": round(total_pnl_pct, 2),
        "positionCount": len(positions),
        "positions": positions,
    }


def current(rh):
    """GET /api/positions — 当前持仓概览。"""
    try:
        raw = _load_positions_file()
        result = _enrich_positions(raw)
        rh._json_response(result)
    except Exception as exc:
        logger.exception("positions error")
        rh._json_response({"error": str(exc)}, status=500)


def history(rh):
    """GET /api/positions/history — 历史委托记录。"""
    try:
        orders_dir = _LIVE_DATA_DIR / "orders"
        if not orders_dir.exists():
            return rh._json_response({"orders": []})

        params = rh._query_params()
        limit = int(params.get("limit", 30))

        files = sorted(orders_dir.glob("*.json"), reverse=True)[:limit]
        history = []
        for f in files:
            date_str = f.stem
            try:
                day_orders = json.loads(f.read_text(encoding="utf-8"))
                for o in day_orders:
                    o["date"] = date_str
                history.extend(day_orders)
            except Exception:
                continue

        rh._json_response({"orders": history})
    except Exception as exc:
        logger.exception("positions history error")
        rh._json_response({"error": str(exc)}, status=500)


def add_or_update(rh):
    """POST /api/positions — 添加或更新持仓。

    Body: {"instrument": "SH600011", "shares": 1000, "price": 8.50}
    如果 instrument 已存在，累加 shares 并按加权平均更新 price。
    """
    try:
        body = rh._read_body()
    except Exception:
        return

    instrument = (body.get("instrument") or "").strip().upper()
    shares = body.get("shares")
    price = body.get("price")

    if not instrument:
        return rh._json_response({"error": "instrument 不能为空"}, status=400)
    if shares is None or shares <= 0:
        return rh._json_response({"error": "shares 必须大于 0"}, status=400)
    if price is None or price <= 0:
        return rh._json_response({"error": "price 必须大于 0"}, status=400)

    shares = int(shares)
    price = float(price)

    raw = _load_positions_file()
    fee_settings = _fee_settings_from_raw(raw)
    trade_amount = shares * price
    fee_breakdown = _calc_trade_fee(instrument, trade_amount, "buy", fee_settings)
    positions = raw.get("positions", {})

    if instrument in positions:
        # 加仓：加权平均成本
        old = positions[instrument]
        old_shares = old.get("shares", 0)
        old_price = old.get("price", 0)
        new_shares = old_shares + shares
        new_price = (old_shares * old_price + shares * price + fee_breakdown["total"]) / new_shares if new_shares > 0 else price
        positions[instrument] = {"shares": new_shares, "price": round(new_price, 4)}
    else:
        positions[instrument] = {"shares": shares, "price": round((trade_amount + fee_breakdown["total"]) / shares, 4)}

    raw["positions"] = positions
    raw["date"] = str(_date.today())
    # 扣减现金（含费用）
    raw["cash"] = round(raw.get("cash", 0) - trade_amount - fee_breakdown["total"], 2)
    _save_positions_file(raw)

    result = _enrich_positions(raw)
    result["lastFeeBreakdown"] = fee_breakdown
    rh._json_response(result)


def update(rh, instrument):
    """POST /api/positions/{instrument} — 修改已有持仓的价格或数量。

    Body: {"shares": 1500, "price": 9.00}  (任选其一或全部)
    """
    instrument = instrument.strip().upper()
    if not instrument:
        return rh._json_response({"error": "instrument 不能为空"}, status=400)

    try:
        body = rh._read_body()
    except Exception:
        return

    raw = _load_positions_file()
    fee_settings = _fee_settings_from_raw(raw)
    positions = raw.get("positions", {})

    if instrument not in positions:
        return rh._json_response({"error": f"持仓 {instrument} 不存在"}, status=404)

    pos = positions[instrument]
    old_shares = pos.get("shares", 0)
    old_price = pos.get("price", 0)

    new_shares = body.get("shares")
    new_price = body.get("price")

    if new_shares is not None:
        new_shares = int(new_shares)
        if new_shares <= 0:
            return rh._json_response({"error": "shares 必须大于 0"}, status=400)
        pos["shares"] = new_shares

    if new_price is not None:
        new_price = float(new_price)
        if new_price <= 0:
            return rh._json_response({"error": "price 必须大于 0"}, status=400)
        pos["price"] = round(new_price, 4)

    # 现金差额补偿（按买入费规则估算成本差额）
    old_cost = old_shares * old_price
    new_cost = pos["shares"] * pos["price"]
    old_fee = _calc_trade_fee(instrument, old_cost, "buy", fee_settings)["total"] if old_cost > 0 else 0.0
    new_fee = _calc_trade_fee(instrument, new_cost, "buy", fee_settings)["total"] if new_cost > 0 else 0.0
    raw["cash"] = round(raw.get("cash", 0) + old_cost + old_fee - new_cost - new_fee, 2)

    raw["date"] = str(_date.today())
    _save_positions_file(raw)

    result = _enrich_positions(raw)
    result["lastFeeBreakdown"] = _calc_trade_fee(instrument, new_cost, "buy", fee_settings) if new_cost > 0 else {
        "commission": 0.0,
        "stampDuty": 0.0,
        "transferFee": 0.0,
        "total": 0.0,
    }
    rh._json_response(result)


def remove(rh, instrument):
    """DELETE /api/positions/{instrument} — 删除持仓。"""
    instrument = instrument.strip().upper()
    if not instrument:
        return rh._json_response({"error": "instrument 不能为空"}, status=400)

    raw = _load_positions_file()
    positions = raw.get("positions", {})

    if instrument not in positions:
        return rh._json_response({"error": f"持仓 {instrument} 不存在"}, status=404)

    # 卖出资金回笼（扣除卖出费用）
    pos = positions[instrument]
    fee_settings = _fee_settings_from_raw(raw)
    sell_amount = pos.get("shares", 0) * pos.get("price", 0)
    fee_breakdown = _calc_trade_fee(instrument, sell_amount, "sell", fee_settings)
    raw["cash"] = round(raw.get("cash", 0) + sell_amount - fee_breakdown["total"], 2)
    del positions[instrument]

    raw["date"] = str(_date.today())
    _save_positions_file(raw)

    result = _enrich_positions(raw)
    result["lastFeeBreakdown"] = fee_breakdown
    rh._json_response(result)


def set_cash(rh):
    """POST /api/positions/cash — 设置现金余额。

    Body: {"cash": 1000000}
    """
    try:
        body = rh._read_body()
    except Exception:
        return

    cash = body.get("cash")
    if cash is None or cash < 0:
        return rh._json_response({"error": "cash 必须 >= 0"}, status=400)

    raw = _load_positions_file()
    raw["cash"] = float(cash)
    raw["date"] = str(_date.today())
    _save_positions_file(raw)

    result = _enrich_positions(raw)
    rh._json_response(result)


def set_account(rh):
    """POST /api/positions/account — 设置账户信息。

    Body: {
        "capitalAmount": 1000000,
        "stockCommissionRate": 0.0001,
        "etfCommissionRate": 0.00005,
        "stampDutyRate": 0.0005,
        "shTransferFeeRate": 0.00001,
    }
    """
    try:
        body = rh._read_body()
    except Exception:
        return

    capital_amount = body.get("capitalAmount")
    if capital_amount is None:
        return rh._json_response({"error": "capitalAmount 不能为空"}, status=400)

    try:
        capital_amount = float(capital_amount)
    except (TypeError, ValueError):
        return rh._json_response({"error": "capitalAmount 必须为数字"}, status=400)

    if capital_amount < 0:
        return rh._json_response({"error": "capitalAmount 必须 >= 0"}, status=400)

    rate_fields = {
        "stockCommissionRate": "stock_commission_rate",
        "etfCommissionRate": "etf_commission_rate",
        "stampDutyRate": "stamp_duty_rate",
        "shTransferFeeRate": "sh_transfer_fee_rate",
    }

    raw = _load_positions_file()
    for api_key, raw_key in rate_fields.items():
        value = body.get(api_key, raw.get(raw_key, _DEFAULT_FEE_SETTINGS[raw_key]))
        try:
            value = float(value)
        except (TypeError, ValueError):
            return rh._json_response({"error": f"{api_key} 必须为数字"}, status=400)
        if value < 0:
            return rh._json_response({"error": f"{api_key} 必须 >= 0"}, status=400)
        raw[raw_key] = value

    raw["capital_amount"] = capital_amount
    raw["trading_fee_rate"] = raw["stock_commission_rate"]
    raw["date"] = str(_date.today())
    _save_positions_file(raw)

    result = _enrich_positions(raw)
    rh._json_response(result)
