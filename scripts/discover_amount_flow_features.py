"""Rank amount-flow feature candidates on train and validation periods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import quant_master
from quant_master.contrib.data.amount_flow_handler import Alpha158AmountFlow
from quant_master.data import D


LABEL_EXPR = "Ref($close, -2)/Ref($close, -1)-1"


def _datetime_level(index: pd.MultiIndex) -> int:
    if "datetime" in index.names:
        return index.names.index("datetime")
    for level in range(index.nlevels):
        values = index.get_level_values(level)
        if np.issubdtype(values.dtype, np.datetime64):
            return level
    raise ValueError("Feature frame has no datetime index level")


def _slice_dates(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    level = _datetime_level(frame.index)
    dates = pd.to_datetime(frame.index.get_level_values(level))
    return frame.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def _daily_correlations(panel: pd.DataFrame, feature: str) -> tuple[pd.Series, pd.Series]:
    values = panel[[feature, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    level = _datetime_level(values.index)
    grouped = values.groupby(level=level, sort=True)
    ic = grouped.apply(lambda day: day[feature].corr(day["label"]))
    rank_ic = grouped.apply(lambda day: day[feature].rank(pct=True).corr(day["label"].rank(pct=True)))
    return ic.dropna(), rank_ic.dropna()


def _summary(panel: pd.DataFrame, feature: str) -> dict:
    finite = panel[feature].replace([np.inf, -np.inf], np.nan)
    ic, rank_ic = _daily_correlations(panel, feature)
    rank_std = float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else float("nan")
    return {
        "coverage": float(finite.notna().mean()),
        "days": int(len(rank_ic)),
        "ic": float(ic.mean()) if len(ic) else float("nan"),
        "rank_ic": float(rank_ic.mean()) if len(rank_ic) else float("nan"),
        "rank_ic_ir": (
            float(np.sqrt(252.0) * rank_ic.mean() / rank_std)
            if np.isfinite(rank_std) and rank_std > 1e-12
            else float("nan")
        ),
    }


def evaluate_features(
    frame: pd.DataFrame,
    feature_names: list[str],
    train_range: tuple[str, str],
    valid_range: tuple[str, str],
) -> list[dict]:
    train = _slice_dates(frame, *train_range)
    valid = _slice_dates(frame, *valid_range)
    rows = []
    for feature in feature_names:
        train_stats = _summary(train, feature)
        valid_stats = _summary(valid, feature)
        same_sign = bool(
            np.isfinite(train_stats["rank_ic"])
            and np.isfinite(valid_stats["rank_ic"])
            and train_stats["rank_ic"] * valid_stats["rank_ic"] > 0
        )
        stable_score = (
            min(abs(train_stats["rank_ic"]), abs(valid_stats["rank_ic"]))
            if same_sign
            else 0.0
        )
        rows.append(
            {
                "feature": feature,
                "same_sign": same_sign,
                "stable_rank_ic": stable_score,
                "train": train_stats,
                "valid": valid_stats,
            }
        )
    return sorted(rows, key=lambda row: (row["stable_rank_ic"], abs(row["valid"]["rank_ic"])), reverse=True)


def select_low_redundancy_features(
    frame: pd.DataFrame,
    rows: list[dict],
    valid_range: tuple[str, str],
    min_stable_rank_ic: float = 0.005,
    max_abs_rank_corr: float = 0.8,
) -> list[dict]:
    eligible = [
        row
        for row in rows
        if row["same_sign"]
        and row["stable_rank_ic"] >= min_stable_rank_ic
        and row["valid"]["coverage"] >= 0.95
    ]
    if not eligible:
        return []

    valid = _slice_dates(frame, *valid_range)
    names = [row["feature"] for row in eligible]
    level = _datetime_level(valid.index)
    ranked = valid[names].replace([np.inf, -np.inf], np.nan).groupby(level=level).rank(pct=True)
    correlations = ranked.corr(method="pearson", min_periods=3)

    selected = []
    selected_names = []
    for row in eligible:
        correlations_to_selected = {
            name: float(correlations.loc[row["feature"], name]) for name in selected_names
        }
        finite_correlations = {
            name: value for name, value in correlations_to_selected.items() if np.isfinite(value)
        }
        closest_name = (
            max(finite_correlations, key=lambda name: abs(finite_correlations[name]))
            if finite_correlations
            else None
        )
        closest_correlation = finite_correlations.get(closest_name, 0.0)
        if closest_name is not None and abs(closest_correlation) >= max_abs_rank_corr:
            continue
        selected_names.append(row["feature"])
        selected.append(
            {
                "feature": row["feature"],
                "stable_rank_ic": row["stable_rank_ic"],
                "direction": "positive" if row["valid"]["rank_ic"] > 0 else "negative",
                "closest_selected_feature": closest_name,
                "closest_selected_rank_corr": closest_correlation,
            }
        )
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default="~/.quant_master/quant_master_data/tdx_cn_data")
    parser.add_argument("--market", default="csi300")
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
    fields, names = Alpha158AmountFlow.get_amount_flow_feature_config()
    query_fields = fields + [LABEL_EXPR]
    frame = D.features(
        D.instruments(args.market),
        query_fields,
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

    payload = {
        "market": args.market,
        "train": [args.train_start, args.train_end],
        "valid": [args.valid_start, args.valid_end],
        "selection_policy": "same Rank IC sign; rank by min(abs(train Rank IC), abs(valid Rank IC))",
        "redundancy_policy": {
            "min_stable_rank_ic": args.min_stable_rank_ic,
            "max_abs_rank_corr": args.max_abs_rank_corr,
        },
        "recommended_features": recommended,
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
            f"  {item['feature']:<34} stable={item['stable_rank_ic']:.6f} "
            f"direction={item['direction']:<8} corr={item['closest_selected_rank_corr']:+.3f}"
        )
    print("All candidates:")
    for row in rows:
        print(
            f"{row['feature']:<34} stable={row['stable_rank_ic']:.6f} "
            f"train={row['train']['rank_ic']:+.6f} valid={row['valid']['rank_ic']:+.6f} "
            f"coverage={row['valid']['coverage']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
