#!/usr/bin/env python
"""Run validated live/paper orders through a configured broker.

Example
-------
python scripts/run_live_orders.py ^
  --broker paper ^
  --orders 000001,buy,12.34,100 600519,sell,1680.00,100 ^
  --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_master.contrib.broker import BrokerOrderDir, LiveOrderExecutor, LiveOrderRequest, create_broker


def parse_order(text: str) -> LiveOrderRequest:
    stock_id, direction, price, amount = [part.strip() for part in text.split(",")]
    direction_enum = BrokerOrderDir.BUY if direction.lower() == "buy" else BrokerOrderDir.SELL
    return LiveOrderRequest(
        stock_id=stock_id,
        direction=direction_enum,
        price=float(price),
        amount=int(amount),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit validated live orders")
    parser.add_argument("--broker", default="paper", help="paper, tcdll, tdx, easytrader, or xiadan")
    parser.add_argument("--host", help="Required when --broker tdx")
    parser.add_argument("--port", type=int, default=7708)
    parser.add_argument("--caller-path", help="Path to tdx_caller.exe when --broker tcdll")
    parser.add_argument("--xiadan-work-dir", help="Working directory containing the xiadan/Tc.dll runtime")
    parser.add_argument("--dll-dir", help="Directory used to resolve Tc.dll and its side-by-side DLLs")
    parser.add_argument("--enable-live", action="store_true", help="Allow live order submission for tcdll")
    parser.add_argument("--max-order-value", type=float, default=None)
    parser.add_argument("--max-position-ratio", type=float, default=1.0)
    parser.add_argument("--allow-sell-without-position-check", action="store_true")
    parser.add_argument(
        "--skip-account-check",
        action="store_true",
        help="Skip broker account/position queries before submission",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--orders",
        nargs="+",
        required=True,
        help="Space-separated orders in stock_id,direction,price,amount format",
    )
    return parser


def result_to_payload(item):
    return {
        "stock_id": item.request.stock_id,
        "direction": item.request.direction.name,
        "price": item.request.price,
        "amount": item.request.amount,
        "accepted": item.accepted,
        "rejection_reason": item.rejection_reason,
        "post_check_status": item.post_check_status,
        "order_id": None if item.broker_order is None else item.broker_order.order_id,
    }


def main() -> int:
    args = build_parser().parse_args()

    broker_kwargs = {}
    broker_name = args.broker.lower()
    if broker_name == "tdx":
        broker_kwargs["host"] = args.host
        broker_kwargs["port"] = args.port
    elif broker_name in {"tcdll", "tc_dll", "tc", "dll"}:
        if args.caller_path:
            broker_kwargs["caller_path"] = args.caller_path
        if args.xiadan_work_dir:
            broker_kwargs["xiadan_work_dir"] = args.xiadan_work_dir
        if args.dll_dir:
            broker_kwargs["dll_dir"] = args.dll_dir
        broker_kwargs["enable_live"] = args.enable_live

    broker = create_broker(args.broker, **broker_kwargs)
    if hasattr(broker, "connect"):
        broker.connect()

    executor = LiveOrderExecutor(
        broker,
        max_order_value=args.max_order_value,
        max_position_ratio=args.max_position_ratio,
        allow_sell_without_position_check=args.allow_sell_without_position_check,
        validate_account_state=not args.skip_account_check,
    )
    requests = [parse_order(text) for text in args.orders]
    results = executor.submit_many(requests, dry_run=args.dry_run)
    payload = [result_to_payload(item) for item in results]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(item.accepted for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
