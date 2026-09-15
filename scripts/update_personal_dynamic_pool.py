"""Update the personal point-in-time A-share training universe."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from quant_master.data.personal_dynamic_pool import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PROVIDER_URI,
    PersonalDynamicPoolConfig,
    update_personal_dynamic_pool,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--date", help="Update one exact trading date (YYYY-MM-DD).")
    parser.add_argument("--start-date", help="Inclusive range start date.")
    parser.add_argument("--end-date", help="Inclusive range end date.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=40.0)
    parser.add_argument("--min-history-days", type=int, default=120)
    parser.add_argument("--amount-lookback-days", type=int, default=20)
    parser.add_argument("--min-average-amount", type=float, default=50_000_000.0)
    parser.add_argument("--activity-lookback-days", type=int, default=20)
    parser.add_argument("--min-average-abs-return", type=float, default=0.008)
    parser.add_argument("--boards", nargs="+", choices=["main", "chinext", "star"], default=None)
    parser.add_argument("--min-pool-size", type=int, default=500)
    parser.add_argument("--max-pool-size", type=int, default=5500)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--kernels", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = PersonalDynamicPoolConfig(
        min_price=args.min_price,
        max_price=args.max_price,
        min_history_days=args.min_history_days,
        amount_lookback_days=args.amount_lookback_days,
        min_average_amount=args.min_average_amount,
        activity_lookback_days=args.activity_lookback_days,
        min_average_abs_return=args.min_average_abs_return,
        boards=tuple(args.boards or ("main", "chinext", "star")),
        min_pool_size=args.min_pool_size,
        max_pool_size=args.max_pool_size,
    )
    result = update_personal_dynamic_pool(
        provider_uri=args.provider_uri,
        update_date=args.date,
        start_date=args.start_date,
        end_date=args.end_date,
        output_name=args.output_name,
        config=config,
        chunk_size=args.chunk_size,
        kernels=args.kernels,
        dry_run=args.dry_run,
    )
    payload = asdict(result)
    payload["output_path"] = str(result.output_path)
    payload["metadata_path"] = str(result.metadata_path)
    payload["backup_paths"] = [str(path) for path in result.backup_paths]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
