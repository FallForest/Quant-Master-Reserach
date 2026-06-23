"""持仓服务层：文件 I/O、持仓丰富化（行情填充、计算）。"""

from __future__ import annotations

import json
from pathlib import Path

from .config import LIVE_DATA_DIR
from .helpers import (
    build_fee_labels,
    calc_trade_fee,
    fee_settings_from_raw,
    is_etf,
    lookup_local_close,
    lookup_name,
    lookup_quote_price,
)


def _ensure_dirs() -> None:
    (LIVE_DATA_DIR / "positions").mkdir(parents=True, exist_ok=True)


def _orders_dir() -> Path:
    return LIVE_DATA_DIR / "orders"


def load_positions_file() -> dict:
    path = LIVE_DATA_DIR / "positions" / "latest.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"cash": 1000000.0, "positions": {}, "date": None, "total_assets": 1000000.0}

    capital_amount = data.get("capital_amount")
    if capital_amount is None:
        capital_amount = data.get("total_assets", data.get("cash", 1000000.0))

    fee_settings = fee_settings_from_raw(data)

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


def save_positions_file(data: dict) -> None:
    _ensure_dirs()
    path = LIVE_DATA_DIR / "positions" / "latest.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_orders(limit: int = 30) -> list:
    orders_dir = _orders_dir()
    if not orders_dir.exists():
        return []
    files = sorted(orders_dir.glob("*.json"), reverse=True)[:limit]
    result = []
    fee_settings = None
    for f in files:
        date_str = f.stem
        try:
            day_orders = json.loads(f.read_text(encoding="utf-8"))
            for o in day_orders:
                o["date"] = date_str
                instrument = o.get("instrument") or o.get("stock_id") or o.get("symbol") or ""
                shares = abs(int(o.get("shares") or o.get("amount") or 0))
                price = float(o.get("price") or 0)
                side = str(o.get("side") or o.get("direction") or "buy").strip().lower()
                if instrument and shares > 0 and price > 0:
                    if fee_settings is None:
                        fee_settings = fee_settings_from_raw(load_positions_file())
                    trade_amount = shares * price
                    o["feeBreakdown"] = calc_trade_fee(instrument, trade_amount, side, fee_settings)
                else:
                    o["feeBreakdown"] = {"commission": 0, "stampDuty": 0, "transferFee": 0, "total": 0}
            result.extend(day_orders)
        except Exception:
            continue
    return result


def enrich_positions(raw: dict, data=None, tdx=None, names=None, quotes=None) -> dict:
    cash = raw.get("cash", 0)
    raw_positions = raw.get("positions", {})
    date = raw.get("date")
    fee_settings = fee_settings_from_raw(raw)
    capital_amount = raw.get("capital_amount", raw.get("total_assets", cash))

    if names is None and data is not None:
        try:
            names = data.get_names()
        except Exception:
            names = {}
    elif names is None:
        names = {}

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

        current_price = lookup_quote_price(quotes, instrument, 0)
        if current_price <= 0:
            current_price = lookup_local_close(data, instrument, cost_price)

        market_value = shares * current_price
        total_market_value += market_value
        pnl = (current_price - cost_price) * shares
        pnl_pct = ((current_price / cost_price) - 1) * 100 if cost_price > 0 else 0

        buy_fee = calc_trade_fee(instrument, cost_price * shares, "buy", fee_settings)
        positions.append({
            "instrument": instrument,
            "name": lookup_name(names, instrument),
            "shares": shares,
            "costPrice": round(cost_price, 4),
            "currentPrice": round(current_price, 4),
            "marketValue": round(market_value, 2),
            "pnl": round(pnl, 2),
            "pnlPct": round(pnl_pct, 2),
            "instrumentType": "ETF" if is_etf(instrument) else "股票",
            "buyFee": buy_fee,
        })

    for p in positions:
        p["weight"] = round(p["marketValue"] / total_market_value * 100, 2) if total_market_value > 0 else 0

    positions.sort(key=lambda x: x["marketValue"], reverse=True)

    total_assets = cash + total_market_value
    total_cost = sum(p["costPrice"] * p["shares"] for p in positions)
    total_pnl = total_market_value - total_cost
    total_pnl_pct = (total_market_value / total_cost - 1) * 100 if total_cost > 0 else 0

    fee_labels = build_fee_labels(fee_settings)

    result = {
        "date": date,
        "cash": round(cash, 2),
        "capitalAmount": round(float(capital_amount), 2),
        "feeSettings": {
            "stockCommissionRate": round(fee_settings["stock_commission_rate"], 6),
            "etfCommissionRate": round(fee_settings["etf_commission_rate"], 6),
            "stampDutyRate": round(fee_settings["stamp_duty_rate"], 6),
            "shTransferFeeRate": round(fee_settings["sh_transfer_fee_rate"], 6),
            "stockCommissionLabel": fee_labels["stockCommissionLabel"],
            "etfCommissionLabel": fee_labels["etfCommissionLabel"],
            "stampDutyLabel": fee_labels["stampDutyLabel"],
            "shTransferFeeLabel": fee_labels["shTransferFeeLabel"],
        },
        "feeRuleSummary": fee_labels["feeRuleSummary"],
        "totalAssets": round(total_assets, 2),
        "totalMarketValue": round(total_market_value, 2),
        "totalPnl": round(total_pnl, 2),
        "totalPnlPct": round(total_pnl_pct, 2),
        "positionCount": len(positions),
        "positions": positions,
    }
    return result
