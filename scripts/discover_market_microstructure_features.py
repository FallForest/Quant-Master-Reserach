"""Rank daily market-microstructure proxy candidates on train and validation periods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import quant_master
from discover_amount_flow_features import evaluate_features, select_low_redundancy_features
from quant_master.contrib.data.market_microstructure_handler import Alpha158MarketMicrostructureProxy
from quant_master.data import D


LABEL_EXPR = "Ref($close, -2)/Ref($close, -1)-1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default="~/.quant_master/quant_master_data/tdx_cn_data")
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--benchmark", default="SH000300")
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--valid-start", default="2021-01-01")
    parser.add_argument("--valid-end", default="2023-12-31")
    parser.add_argument("--min-stable-rank-ic", type=float, default=0.005)
    parser.add_argument("--max-abs-rank-corr", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    quant_master.init(provider_uri=args.provider_uri, region="cn", kernels=1)
    registry = Alpha158MarketMicrostructureProxy.get_market_microstructure_feature_registry(
        benchmark=args.benchmark
    )
    fields, names = registry.expand_columns()
    frame = D.features(
        D.instruments(args.market),
        fields + [LABEL_EXPR],
        start_time=args.train_start,
        end_time=args.valid_end,
        freq="day",
    )
    frame.columns = names + ["label"]
    rows = evaluate_features(
        frame,
        names,
        (args.train_start, args.train_end),
        (args.valid_start, args.valid_end),
    )
    recommended = select_low_redundancy_features(
        frame,
        rows,
        (args.valid_start, args.valid_end),
        min_stable_rank_ic=args.min_stable_rank_ic,
        max_abs_rank_corr=args.max_abs_rank_corr,
    )

    metadata = {
        spec.name: {
            "family": spec.family,
            "window": spec.window,
            "data_requirements": spec.data_requirements,
            "available_lag": spec.available_lag,
            "source": spec.source,
        }
        for spec in registry
    }
    payload = {
        "market": args.market,
        "benchmark": args.benchmark,
        "train": [args.train_start, args.train_end],
        "valid": [args.valid_start, args.valid_end],
        "selection_policy": "same Rank IC sign; rank by min(abs(train Rank IC), abs(valid Rank IC))",
        "redundancy_policy": {
            "min_stable_rank_ic": args.min_stable_rank_ic,
            "max_abs_rank_corr": args.max_abs_rank_corr,
        },
        "recommended_features": recommended,
        "metadata": metadata,
        "features": rows,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")

    print("Recommended low-redundancy features:")
    for item in recommended:
        print(
            f"  {item['feature']:<32} stable={item['stable_rank_ic']:.6f} "
            f"direction={item['direction']:<8} corr={item['closest_selected_rank_corr']:+.3f}"
        )
    print("All candidates:")
    for row in rows:
        print(
            f"{row['feature']:<32} stable={row['stable_rank_ic']:.6f} "
            f"train={row['train']['rank_ic']:+.6f} valid={row['valid']['rank_ic']:+.6f} "
            f"coverage={row['valid']['coverage']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
