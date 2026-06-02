"""Execution service for UI order preview and safe broker submission."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_master.contrib.broker.base import BrokerOrderDir, Position
from quant_master.contrib.broker.execution import A_SHARE_TRADE_UNIT, LiveOrderExecutor, LiveOrderRequest
from quant_master.contrib.broker.factory import create_broker

from .handlers import position

LIVE_DATA_DIR = Path(__file__).resolve().parent.parent / "live_data"
EXECUTION_DIR = LIVE_DATA_DIR / "executions"
SUPPORTED_BROKERS = ["paper", "tcdll", "tdx", "easytrader", "xiadan"]
DEFAULT_BROKER_KIND = "paper"
DEFAULT_DRY_RUN = True
DEFAULT_MAX_POSITION_RATIO = 1.0
LIVE_TRADING_ENABLED = False


def _ensure_dirs() -> None:
    EXECUTION_DIR.mkdir(parents=True, exist_ok=True)


def get_execution_config() -> dict[str, Any]:
    return {
        "defaultBrokerKind": DEFAULT_BROKER_KIND,
        "defaultDryRun": DEFAULT_DRY_RUN,
        "supportedBrokers": SUPPORTED_BROKERS,
        "liveTradingEnabled": LIVE_TRADING_ENABLED,
        "tradeUnit": A_SHARE_TRADE_UNIT,
        "riskDefaults": {
            "maxOrderValue": None,
            "maxPositionRatio": DEFAULT_MAX_POSITION_RATIO,
        },
    }


def _parse_side(side: str) -> BrokerOrderDir | None:
    normalized = str(side or "").strip().lower()
    if normalized == "buy":
        return BrokerOrderDir.BUY
    if normalized == "sell":
        return BrokerOrderDir.SELL
    return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _int_or_zero(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def _normalize_trade(trade: dict[str, Any]) -> dict[str, Any] | None:
    side = str(trade.get("side") or "").strip().lower()
    if side == "hold":
        return None

    stock_id = position._normalize_symbol(trade.get("stockId") or trade.get("instrument"))
    amount = abs(_int_or_zero(trade.get("amount", trade.get("deltaShares"))))
    price = float(trade.get("price", trade.get("currentPrice", 0)) or 0)
    direction = _parse_side(side)
    if direction is None:
        direction_label = side
    else:
        direction_label = "buy" if direction == BrokerOrderDir.BUY else "sell"

    order_value = round(price * amount, 2) if price > 0 and amount > 0 else 0.0
    request = {
        "stockId": stock_id,
        "name": trade.get("name") or stock_id,
        "side": direction_label,
        "price": round(price, 4) if price > 0 else price,
        "amount": amount,
        "orderValue": order_value,
        "source": trade.get("source") or "buffered-rebalance",
        "tradeDate": trade.get("tradeDate"),
        "valid": True,
        "validationError": "",
    }

    if not stock_id:
        request["valid"] = False
        request["validationError"] = "missing_stock_id"
    elif direction is None:
        request["valid"] = False
        request["validationError"] = "invalid_side"
    elif price <= 0:
        request["valid"] = False
        request["validationError"] = "invalid_price"
    elif amount <= 0:
        request["valid"] = False
        request["validationError"] = "invalid_amount"
    elif amount % A_SHARE_TRADE_UNIT != 0:
        request["valid"] = False
        request["validationError"] = f"amount_must_be_multiple_of_{A_SHARE_TRADE_UNIT}"

    return request


def build_order_preview(trades: list[dict[str, Any]], risk: dict[str, Any] | None = None) -> dict[str, Any]:
    orders = []
    risk = risk or {}
    max_order_value = _float_or_none(risk.get("maxOrderValue"))
    for trade in trades or []:
        normalized = _normalize_trade(trade)
        if normalized is None:
            continue
        if normalized["valid"] and max_order_value is not None and normalized["orderValue"] > max_order_value:
            normalized["valid"] = False
            normalized["validationError"] = "order_value_exceeds_limit"
        orders.append(normalized)

    buy_amount = round(sum(item["orderValue"] for item in orders if item["side"] == "buy"), 2)
    sell_amount = round(sum(item["orderValue"] for item in orders if item["side"] == "sell"), 2)
    valid_orders = sum(1 for item in orders if item["valid"])

    return {
        "orders": orders,
        "summary": {
            "totalOrders": len(orders),
            "validOrders": valid_orders,
            "invalidOrders": len(orders) - valid_orders,
            "buyAmount": buy_amount,
            "sellAmount": sell_amount,
        },
    }


def _paper_positions_from_file() -> tuple[float, list[Position]]:
    enriched = position._enrich_positions(position._load_positions_file())
    cash = float(enriched.get("cash", 0.0) or 0.0)
    items = []
    for pos in enriched.get("positions", []):
        shares = int(pos.get("shares", 0) or 0)
        price = float(pos.get("currentPrice") or pos.get("costPrice") or 0.0)
        items.append(
            Position(
                stock_id=position._normalize_symbol(pos.get("instrument")),
                volume=shares,
                available_volume=shares,
                cost_price=float(pos.get("costPrice", 0.0) or 0.0),
                current_price=price,
                market_value=round(shares * price, 2),
            )
        )
    return cash, items


def _create_executor(broker_kind: str, risk: dict[str, Any] | None) -> LiveOrderExecutor:
    risk = risk or {}
    broker_kind = (broker_kind or DEFAULT_BROKER_KIND).strip().lower()
    broker_kwargs: dict[str, Any] = {}
    if broker_kind == "paper":
        available_cash, positions_list = _paper_positions_from_file()
        broker_kwargs.update({"available_cash": available_cash, "positions": positions_list})
    broker = create_broker(broker_kind, **broker_kwargs)
    return LiveOrderExecutor(
        broker,
        max_order_value=_float_or_none(risk.get("maxOrderValue")),
        max_position_ratio=float(risk.get("maxPositionRatio", DEFAULT_MAX_POSITION_RATIO) or DEFAULT_MAX_POSITION_RATIO),
        trade_unit=A_SHARE_TRADE_UNIT,
        allow_sell_without_position_check=False,
        validate_account_state=True,
    )


def _to_live_requests(orders: list[dict[str, Any]]) -> list[LiveOrderRequest]:
    requests = []
    sells = []
    buys = []
    for order in orders:
        direction = _parse_side(order.get("side"))
        if direction is None:
            continue
        request = LiveOrderRequest(
            stock_id=position._normalize_symbol(order.get("stockId") or order.get("instrument")),
            price=float(order.get("price", 0) or 0),
            amount=int(order.get("amount", 0) or 0),
            direction=direction,
            note=str(order.get("note") or order.get("source") or "ui-execution"),
        )
        if direction == BrokerOrderDir.SELL:
            sells.append(request)
        else:
            buys.append(request)
    requests.extend(sells)
    requests.extend(buys)
    return requests


def _serialize_result(result) -> dict[str, Any]:
    broker_order = result.broker_order
    direction = result.request.direction
    side = "buy" if direction == BrokerOrderDir.BUY else "sell"
    payload = {
        "stockId": result.request.stock_id,
        "side": side,
        "price": round(float(result.request.price), 4),
        "amount": int(result.request.amount),
        "accepted": bool(result.accepted),
        "rejectionReason": result.rejection_reason,
        "postCheckStatus": result.post_check_status,
        "note": result.request.note,
        "orderId": broker_order.order_id if broker_order else None,
        "status": broker_order.status.value if broker_order else None,
        "dealAmount": int(broker_order.deal_amount) if broker_order else 0,
    }
    return payload


def _write_history(payload: dict[str, Any]) -> None:
    _ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    file_path = EXECUTION_DIR / f"{stamp}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def submit_orders(
    orders: list[dict[str, Any]],
    *,
    broker_kind: str | None,
    dry_run: bool | None,
    risk: dict[str, Any] | None,
    confirm: bool,
) -> dict[str, Any]:
    if not confirm:
        raise ValueError("confirm_required")

    dry_run = DEFAULT_DRY_RUN if dry_run is None else bool(dry_run)
    broker_kind = (broker_kind or DEFAULT_BROKER_KIND).strip().lower()
    if not dry_run and not LIVE_TRADING_ENABLED:
        raise PermissionError("live_trading_disabled")

    preview = build_order_preview(orders, risk=risk)
    invalid_orders = [item for item in preview["orders"] if not item["valid"]]
    if invalid_orders:
        return {
            "brokerKind": broker_kind,
            "dryRun": dry_run,
            "submittedAt": datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "total": len(preview["orders"]),
                "accepted": 0,
                "rejected": len(preview["orders"]),
            },
            "results": [
                {
                    "stockId": item["stockId"],
                    "side": item["side"],
                    "price": item["price"],
                    "amount": item["amount"],
                    "accepted": False,
                    "orderId": None,
                    "status": None,
                    "postCheckStatus": "not_submitted",
                    "rejectionReason": item["validationError"],
                    "note": item.get("source") or "ui-execution",
                    "dealAmount": 0,
                }
                for item in preview["orders"]
            ],
        }

    executor = _create_executor(broker_kind, risk)
    requests = _to_live_requests(preview["orders"])
    results = executor.submit_many(requests, dry_run=dry_run)
    serialized = [_serialize_result(item) for item in results]
    payload = {
        "brokerKind": broker_kind,
        "dryRun": dry_run,
        "submittedAt": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total": len(serialized),
            "accepted": sum(1 for item in serialized if item["accepted"]),
            "rejected": sum(1 for item in serialized if not item["accepted"]),
        },
        "results": serialized,
    }
    _write_history(payload)
    return payload


def load_history(limit: int = 30) -> dict[str, Any]:
    _ensure_dirs()
    history = []
    for path in sorted(EXECUTION_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["historyId"] = path.stem
        history.append(payload)
    return {"runs": history}
