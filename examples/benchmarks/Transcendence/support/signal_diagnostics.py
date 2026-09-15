from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_ANN_SCALER = 252.0


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _first_numeric_column(frame: pd.DataFrame) -> pd.Series:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    preferred = [col for col in ("score", "label") if col in numeric.columns]
    return numeric[preferred[0] if preferred else numeric.columns[0]]


def coerce_signal_series(value: Any, *, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.copy()
    elif isinstance(value, pd.DataFrame):
        series = _first_numeric_column(value)
    else:
        series = pd.Series(value)
    series = pd.to_numeric(series, errors="coerce")
    series.name = name
    return series


def _date_index_values(index: pd.Index) -> pd.Series:
    if isinstance(index, pd.MultiIndex):
        return pd.Series(pd.to_datetime(index.get_level_values(0)), index=index)
    return pd.Series(pd.to_datetime(index), index=index)


def build_signal_panel(pred: Any, label: Any) -> pd.DataFrame:
    pred_s = coerce_signal_series(pred, name="pred")
    label_s = coerce_signal_series(label, name="label")
    panel = pd.concat([pred_s, label_s], axis=1)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
    if panel.empty:
        return panel
    panel["date"] = _date_index_values(panel.index)
    return panel


def daily_ic_frame(panel: pd.DataFrame, *, min_count: int = 20) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if panel.empty:
        return pd.DataFrame(columns=["date", "n", "ic", "rank_ic"])
    for date, group in panel.groupby("date", sort=True):
        if len(group) < min_count:
            continue
        ic = group["pred"].corr(group["label"])
        rank_ic = group["pred"].corr(group["label"], method="spearman")
        rows.append(
            {
                "date": pd.Timestamp(date),
                "n": int(len(group)),
                "ic": _safe_float(ic),
                "rank_ic": _safe_float(rank_ic),
            }
        )
    return pd.DataFrame(rows)


def _mean_ir(values: Iterable[Any], *, ann_scaler: float = DEFAULT_ANN_SCALER) -> Dict[str, Any]:
    series = pd.Series([v for v in values if _safe_float(v) is not None], dtype=float)
    if len(series) < 2:
        return {"mean": None, "ir": None, "days": int(len(series))}
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    return {
        "mean": _safe_float(mean),
        "ir": _safe_float(mean / (std + 1e-12) * math.sqrt(float(ann_scaler))),
        "days": int(len(series)),
    }


def summarize_daily_ic(daily: pd.DataFrame, *, ann_scaler: float = DEFAULT_ANN_SCALER) -> Dict[str, Any]:
    ic = _mean_ir(daily["ic"] if "ic" in daily else [], ann_scaler=ann_scaler)
    rank_ic = _mean_ir(daily["rank_ic"] if "rank_ic" in daily else [], ann_scaler=ann_scaler)
    result = {
        "ic": ic["mean"],
        "ic_ir": ic["ir"],
        "ic_days": ic["days"],
        "rank_ic": rank_ic["mean"],
        "rank_ic_ir": rank_ic["ir"],
        "rank_ic_days": rank_ic["days"],
    }
    if result["ic"] is None:
        result["ic_missing_reason"] = f"insufficient ic days: {ic['days']}"
    if result["rank_ic"] is None:
        result["rank_ic_missing_reason"] = f"insufficient rank_ic days: {rank_ic['days']}"
    return result


def summarize_by_year(daily: pd.DataFrame, *, ann_scaler: float = DEFAULT_ANN_SCALER) -> Dict[str, Any]:
    if daily.empty:
        return {}
    frame = daily.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year.astype(str)
    return {
        year: summarize_daily_ic(group, ann_scaler=ann_scaler)
        for year, group in frame.groupby("year", sort=True)
    }


def decile_diagnostics(panel: pd.DataFrame, *, deciles: int = 10, min_count: int = 20) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if panel.empty:
        return {"deciles": [], "top_bottom_spread": None, "monotonicity": None, "days": 0}

    for date, group in panel.groupby("date", sort=True):
        if len(group) < max(min_count, deciles):
            continue
        try:
            buckets = pd.qcut(group["pred"].rank(method="first"), deciles, labels=False, duplicates="drop")
        except ValueError:
            continue
        tmp = group.assign(decile=buckets.astype(float) + 1)
        by_decile = tmp.groupby("decile")["label"].mean()
        if len(by_decile) < 2:
            continue
        rows.extend(
            {"date": pd.Timestamp(date), "decile": int(decile), "mean_label": float(mean_label)}
            for decile, mean_label in by_decile.items()
            if math.isfinite(float(mean_label))
        )

    if not rows:
        return {"deciles": [], "top_bottom_spread": None, "monotonicity": None, "days": 0}

    decile_frame = pd.DataFrame(rows)
    mean_by_decile = decile_frame.groupby("decile")["mean_label"].mean().sort_index()
    low = mean_by_decile.iloc[0]
    high = mean_by_decile.iloc[-1]
    monotonicity = mean_by_decile.index.to_series().corr(mean_by_decile, method="spearman")
    return {
        "deciles": [
            {"decile": int(decile), "mean_label": _safe_float(value)}
            for decile, value in mean_by_decile.items()
        ],
        "top_bottom_spread": _safe_float(high - low),
        "monotonicity": _safe_float(monotonicity),
        "days": int(decile_frame["date"].nunique()),
    }


def evaluate_signal(
    pred: Any,
    label: Any,
    *,
    ann_scaler: float = DEFAULT_ANN_SCALER,
    min_count: int = 20,
    deciles: int = 10,
) -> Dict[str, Any]:
    panel = build_signal_panel(pred, label)
    daily = daily_ic_frame(panel, min_count=min_count)
    summary = summarize_daily_ic(daily, ann_scaler=ann_scaler)
    summary.update(
        {
            "rows": int(len(panel)),
            "unique_trade_days": int(panel["date"].nunique()) if "date" in panel else 0,
            "by_year": summarize_by_year(daily, ann_scaler=ann_scaler),
            "decile": decile_diagnostics(panel, deciles=deciles, min_count=min_count),
        }
    )
    if panel.empty:
        summary["error"] = "empty signal panel after aligning pred and label"
    return _jsonable(summary)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fp:
        return pickle.load(fp)


def _write_markdown(path: Path, metrics: Dict[str, Any], daily_csv: Optional[Path]) -> None:
    lines = [
        "# Pure Signal Diagnostics",
        "",
        f"- IC: `{metrics.get('ic')}`",
        f"- ICIR: `{metrics.get('ic_ir')}`",
        f"- Rank IC: `{metrics.get('rank_ic')}`",
        f"- Rank ICIR: `{metrics.get('rank_ic_ir')}`",
        f"- rows: `{metrics.get('rows')}`",
        f"- unique_trade_days: `{metrics.get('unique_trade_days')}`",
        f"- top_bottom_spread: `{metrics.get('decile', {}).get('top_bottom_spread')}`",
        f"- decile_monotonicity: `{metrics.get('decile', {}).get('monotonicity')}`",
    ]
    if daily_csv is not None:
        lines.append(f"- daily_csv: `{daily_csv}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pure prediction signal quality from pred.pkl and label.pkl.")
    parser.add_argument("--pred", required=True, help="Path to pred.pkl or a pickled Series/DataFrame.")
    parser.add_argument("--label", required=True, help="Path to label.pkl or a pickled Series/DataFrame.")
    parser.add_argument("--out-json", required=True, help="Output JSON path.")
    parser.add_argument("--daily-csv", default="", help="Optional daily IC CSV path.")
    parser.add_argument("--summary-md", default="", help="Optional Markdown summary path.")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--deciles", type=int, default=10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    pred = _load_pickle(Path(args.pred).expanduser().resolve())
    label = _load_pickle(Path(args.label).expanduser().resolve())
    panel = build_signal_panel(pred, label)
    daily = daily_ic_frame(panel, min_count=args.min_count)
    metrics = evaluate_signal(pred, label, min_count=args.min_count, deciles=args.deciles)

    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    daily_csv = Path(args.daily_csv).expanduser().resolve() if args.daily_csv else None
    if daily_csv is not None:
        daily_csv.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(daily_csv, index=False)

    if args.summary_md:
        summary_md = Path(args.summary_md).expanduser().resolve()
        summary_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(summary_md, metrics, daily_csv)
    print(json.dumps({"status": "ok", "out_json": str(out_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
