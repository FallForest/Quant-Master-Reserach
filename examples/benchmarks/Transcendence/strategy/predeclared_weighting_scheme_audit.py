#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import pickle
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import signal_portfolio_conversion_scan as conv
from quant_master.contrib.evaluate import risk_analysis


TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2026-04-30")
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
BASE_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
GRU45_RUN_IDS = (
    "7406e47063e9479cb34d300b9ed03bad",
    "1a085ff9b5a34f408a44ad74055fc5da",
    "773bd6d8413b4bb0b388a63a6b5b6a86",
)
GRU45_RUN_WEIGHTS = (0.40, 0.20, 0.40)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    topk: int
    hold_topk: int
    note: str


@dataclass(frozen=True)
class WeightSpec:
    scheme_id: str
    weight_mode: str
    score_power: float


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _dump_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _error_fields(exc: Exception) -> Dict[str, str]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)).strip()
    return {"error_type": type(exc).__name__, "error_message": str(exc), "error_traceback_tail": tb}


def _load_score_df(run_dir: Path) -> pd.DataFrame:
    return conv._as_score_df(conv._load_pickle(run_dir / "artifacts" / "pred.pkl")).sort_index()


def _date_values(df: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(df.index, pd.MultiIndex):
        return pd.to_datetime(df.index.get_level_values(0))
    return pd.to_datetime(df.index)


def _coverage(df: pd.DataFrame) -> Dict[str, Any]:
    dates = pd.DatetimeIndex(_date_values(df).normalize().unique()).sort_values()
    return {
        "rows": int(len(df)),
        "days": int(len(dates)),
        "start": str(dates.min().date()) if len(dates) else "",
        "end": str(dates.max().date()) if len(dates) else "",
    }


def _slice_df(df: pd.DataFrame, start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DataFrame:
    dates = _date_values(df)
    return df.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _rank_ensemble(
    tracking_dir: Path,
    run_ids: Sequence[str],
    weights: Sequence[float],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    cols = []
    for run_id in run_ids:
        run_dir = conv._find_run_dir(tracking_dir, run_id)
        pred = _slice_df(_load_score_df(run_dir), start, end)
        ranked = conv._cross_section_rank(pred["score"].astype(float))
        ranked.name = run_id
        cols.append(ranked)
    panel = pd.concat(cols, axis=1)
    w = pd.Series(weights, index=panel.columns, dtype=float)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    score = panel.mul(w, axis=1).fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return score.to_frame("score").sort_index()


def _candidate_family() -> List[WeightSpec]:
    return [
        WeightSpec("equal", "equal", 1.0),
        WeightSpec("score_power_0p5", "score", 0.5),
        WeightSpec("score_power_1p5", "score", 1.5),
        WeightSpec("score_power_2p0", "score", 2.0),
    ]


def _combo(source: SourceSpec, spec: WeightSpec) -> Dict[str, Any]:
    return {
        "family": "buffered_weight",
        "rebalance_mode": "daily",
        "rebalance_interval": 1,
        "topk": int(source.topk),
        "hold_topk": int(source.hold_topk),
        "weight_mode": str(spec.weight_mode),
        "score_power": float(spec.score_power),
    }


def _finite_report(report: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    required = ["return", "bench", "cost", "turnover"]
    missing = [c for c in required if c not in report.columns]
    if missing:
        raise KeyError(f"report missing required columns: {missing}")
    numeric = report[required].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    excess = numeric["return"] - numeric["bench"] - numeric["cost"]
    finite_excess = np.isfinite(excess.to_numpy(dtype=float))
    bad_rows = int(len(report) - int((finite_mask & finite_excess).sum()))
    if bad_rows:
        raise ValueError(f"non-finite report rows detected: {bad_rows} of {len(report)}")
    if report.empty:
        raise ValueError("empty report")
    return numeric, excess.astype(float)


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    if excess.empty:
        raise ValueError("no excess observations")
    if not np.isfinite(excess.to_numpy(dtype=float)).all():
        raise ValueError("non-finite excess observations")
    risk_df = risk_analysis(excess.astype(float).sort_index(), freq="1day")
    out = {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError(f"non-finite risk metrics: {out}")
    return out


def _metrics_from_report(report: pd.DataFrame) -> Dict[str, Any]:
    numeric, excess = _finite_report(report)
    risk = _metrics_from_excess(excess)
    turnover = float(numeric["turnover"].mean())
    if not math.isfinite(turnover):
        raise ValueError("non-finite turnover metric")
    return {
        "costed_annret": float(risk["annret"]),
        "costed_ir": float(risk["ir"]),
        "max_drawdown": float(risk["max_drawdown"]),
        "turnover": turnover,
        "rows": int(len(report)),
        "finite_rows": int(len(report)),
        "nonfinite_rows": 0,
        "excess": excess.rename("excess"),
    }


def _score_for_selection(metrics: Dict[str, Any]) -> float:
    score = (
        float(metrics["costed_ir"])
        + 0.20 * float(metrics["costed_annret"])
        - 0.20 * abs(float(metrics["max_drawdown"]))
        - 0.10 * float(metrics["turnover"])
    )
    if not math.isfinite(score):
        raise ValueError(f"non-finite selection score: {score}")
    return score


def _eval_scheme(
    *,
    source: SourceSpec,
    weight: WeightSpec,
    pred_df: pd.DataFrame,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Dict[str, Any]:
    port_cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = port_cfg["backtest"]
    backtest_cfg["start_time"] = str(pd.Timestamp(start_time).date())
    backtest_cfg["end_time"] = str(pd.Timestamp(end_time).date())
    executor_cfg = port_cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    pred_slice = conv._slice_pred(pred_df, pd.Timestamp(start_time), pd.Timestamp(end_time))
    if pred_slice.empty:
        raise ValueError(f"empty signal slice for {source.source_id}/{weight.scheme_id}: {start_time.date()}..{end_time.date()}")

    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    cache_key = (
        str(backtest_cfg["start_time"]),
        str(backtest_cfg["end_time"]),
        float(open_cost),
        float(close_cost),
        limit_threshold,
        deal_price,
    )
    if cache_key not in exchange_cache:
        exchange_cache[cache_key] = conv.get_exchange(
            freq=freq,
            start_time=backtest_cfg["start_time"],
            end_time=backtest_cfg["end_time"],
            deal_price=deal_price,
            limit_threshold=limit_threshold,
            open_cost=float(open_cost),
            close_cost=float(close_cost),
            min_cost=min_cost,
        )
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    strategy = conv._build_strategy_object(
        combo=_combo(source, weight),
        pred_df=pred_slice,
        base_strategy_kwargs=base_strategy_kwargs,
    )
    t0 = time.perf_counter()
    portfolio_metric_dict, _ = conv.run_backtest(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        strategy=strategy,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    report = conv._get_report_for_day_freq(portfolio_metric_dict).copy()
    report.index = pd.to_datetime(report.index)
    report = report.loc[(report.index >= pd.Timestamp(start_time)) & (report.index <= pd.Timestamp(end_time))].sort_index()
    metrics = _metrics_from_report(report)
    metrics["elapsed_sec"] = float(time.perf_counter() - t0)
    metrics["report"] = report
    return metrics


def _selection_windows() -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("2024H1", TEST_START, pd.Timestamp("2024-06-30"), "2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
        ("2024", TEST_START, pd.Timestamp("2024-12-31"), "2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("2024_2025", TEST_START, pd.Timestamp("2025-12-31"), "2026_ytd", pd.Timestamp("2026-01-01"), TEST_END),
    ]


def _apply_slices() -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    return [
        ("2024H1_default", TEST_START, pd.Timestamp("2024-06-30"), "fixed_predeclared_equal_no_prior_selection"),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"), "selected_by_2024H1"),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), "selected_by_2024"),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END, "selected_by_2024_2025"),
    ]


def _split_metrics(source_id: str, excess: pd.Series) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    split_defs = [
        ("test_full", TEST_START, TEST_END, "stitched_lockstep"),
        ("2024H1_default", TEST_START, pd.Timestamp("2024-06-30"), "fixed_predeclared_equal_no_prior_selection"),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"), "selected_by_2024H1"),
        ("2024", TEST_START, pd.Timestamp("2024-12-31"), "mixed_2024H1_default_2024H2_selected"),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), "selected_by_2024"),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END, "selected_by_2024_2025"),
    ]
    for split, start, end, selection_rule in split_defs:
        part = excess.loc[(excess.index >= start) & (excess.index <= end)]
        if part.empty:
            continue
        rows.append(
            {
                "source_id": source_id,
                "candidate_id": f"{source_id}_lockstep_selected",
                "split": split,
                "start": str(start.date()),
                "end": str(end.date()),
                "days": int(len(part)),
                "selection_rule": selection_rule,
                **_metrics_from_excess(part),
            }
        )
    return rows


def _audit_split_metrics(source_id: str, scheme_id: str, excess: pd.Series) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, start, end in [
        ("test_full", TEST_START, TEST_END),
        ("2024", TEST_START, pd.Timestamp("2024-12-31")),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END),
    ]:
        part = excess.loc[(excess.index >= start) & (excess.index <= end)]
        if part.empty:
            continue
        rows.append(
            {
                "source_id": source_id,
                "candidate_id": f"{source_id}_{scheme_id}",
                "split": split,
                "start": str(start.date()),
                "end": str(end.date()),
                "days": int(len(part)),
                "selection_rule": "full_fixed_candidate_audit_not_used_for_selection",
                **_metrics_from_excess(part),
            }
        )
    return rows


def _scheme_by_id(schemes: Sequence[WeightSpec]) -> Dict[str, WeightSpec]:
    return {s.scheme_id: s for s in schemes}


def _select_lockstep_for_source(
    *,
    source: SourceSpec,
    signal: pd.DataFrame,
    schemes: Sequence[WeightSpec],
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    candidate_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    selected_by_apply = {"2024H1_default": "equal"}
    for train_tag, train_start, train_end, apply_tag, apply_start, apply_end in _selection_windows():
        scored: List[Dict[str, Any]] = []
        for spec in schemes:
            row: Dict[str, Any] = {
                "source_id": source.source_id,
                "candidate_id": f"{source.source_id}_{spec.scheme_id}",
                "scheme_id": spec.scheme_id,
                "train_tag": train_tag,
                "apply_tag": apply_tag,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "apply_start": str(apply_start.date()),
                "apply_end": str(apply_end.date()),
                "ok": False,
                "strict_non_test_selected": True,
                "candidate_family_predeclared": True,
                **asdict(spec),
                **asdict(source),
            }
            try:
                metrics = _eval_scheme(
                    source=source,
                    weight=spec,
                    pred_df=signal,
                    base_port_cfg=base_port_cfg,
                    base_strategy_kwargs=base_strategy_kwargs,
                    open_cost=open_cost,
                    close_cost=close_cost,
                    start_time=train_start,
                    end_time=train_end,
                    exchange_cache=exchange_cache,
                )
                score = _score_for_selection(metrics)
                row.update(
                    {
                        "ok": True,
                        "selection_score": float(score),
                        "costed_ir": float(metrics["costed_ir"]),
                        "costed_annret": float(metrics["costed_annret"]),
                        "max_drawdown": float(metrics["max_drawdown"]),
                        "turnover": float(metrics["turnover"]),
                        "rows": int(metrics["rows"]),
                        "finite_rows": int(metrics["finite_rows"]),
                        "nonfinite_rows": int(metrics["nonfinite_rows"]),
                        "elapsed_sec": float(metrics["elapsed_sec"]),
                        "selection_metric": "costed_ir + 0.20*costed_annret - 0.20*abs(max_drawdown) - 0.10*turnover",
                        "error_type": "",
                        "error_message": "",
                    }
                )
                scored.append(row.copy())
            except Exception as exc:  # noqa: BLE001
                row.update(_error_fields(exc))
            candidate_rows.append(row)

        ranked = sorted(scored, key=lambda x: float(x["selection_score"]), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["selection_rank"] = rank
        if not ranked:
            selection_rows.append(
                {
                    "source_id": source.source_id,
                    "apply_tag": apply_tag,
                    "apply_start": str(apply_start.date()),
                    "apply_end": str(apply_end.date()),
                    "selected": False,
                    "selection_train_tag": train_tag,
                    "strict_non_test_selected": True,
                    "candidate_family_predeclared": True,
                    "error_type": "NoSelectableWeightingScheme",
                    "error_message": f"no finite candidate in {train_tag}",
                }
            )
            continue
        best = ranked[0]
        selected_by_apply[apply_tag] = str(best["scheme_id"])
        selection_rows.append(
            {
                "source_id": source.source_id,
                "apply_tag": apply_tag,
                "apply_start": str(apply_start.date()),
                "apply_end": str(apply_end.date()),
                "selected": True,
                "selection_train_tag": train_tag,
                "scheme_id": str(best["scheme_id"]),
                "candidate_id": str(best["candidate_id"]),
                "weight_mode": str(best["weight_mode"]),
                "score_power": float(best["score_power"]),
                "selection_score": float(best["selection_score"]),
                "train_ir": float(best["costed_ir"]),
                "train_annret": float(best["costed_annret"]),
                "train_mdd": float(best["max_drawdown"]),
                "train_turnover": float(best["turnover"]),
                "selection_rule": f"selected_by_{train_tag}",
                "strict_non_test_selected": True,
                "candidate_family_predeclared": True,
            }
        )
        exchange_cache.clear()
        gc.collect()

    selection_rows.insert(
        0,
        {
            "source_id": source.source_id,
            "apply_tag": "2024H1_default",
            "apply_start": str(TEST_START.date()),
            "apply_end": str(pd.Timestamp("2024-06-30").date()),
            "selected": True,
            "selection_train_tag": "fixed_predeclared_default_no_same_window_selection",
            "scheme_id": "equal",
            "candidate_id": f"{source.source_id}_equal",
            "weight_mode": "equal",
            "score_power": 1.0,
            "selection_score": "",
            "train_ir": "",
            "train_annret": "",
            "train_mdd": "",
            "train_turnover": "",
            "selection_rule": "fixed_predeclared_equal_no_prior_selection",
            "strict_non_test_selected": True,
            "candidate_family_predeclared": True,
        },
    )
    return candidate_rows, selection_rows, selected_by_apply


def _apply_lockstep_for_source(
    *,
    source: SourceSpec,
    signal: pd.DataFrame,
    selected_by_apply: Dict[str, str],
    schemes: Sequence[WeightSpec],
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame, pd.Series]:
    by_id = _scheme_by_id(schemes)
    apply_rows: List[Dict[str, Any]] = []
    report_parts: List[pd.DataFrame] = []
    excess_parts: List[pd.Series] = []
    for apply_tag, apply_start, apply_end, selection_rule in _apply_slices():
        scheme_id = selected_by_apply.get(apply_tag)
        row: Dict[str, Any] = {
            "source_id": source.source_id,
            "apply_tag": apply_tag,
            "start": str(apply_start.date()),
            "end": str(apply_end.date()),
            "selection_rule": selection_rule,
            "scheme_id": scheme_id or "",
            "candidate_id": f"{source.source_id}_{scheme_id}" if scheme_id else "",
            "ok": False,
            "strict_non_test_selected": True,
            "continuous_lockstep_backtest": False,
            "stitched_from_apply_window_backtests": True,
        }
        if not scheme_id or scheme_id not in by_id:
            row.update({"error_type": "MissingSelection", "error_message": f"no selected scheme for {apply_tag}"})
            apply_rows.append(row)
            continue
        spec = by_id[scheme_id]
        row.update(asdict(spec))
        try:
            metrics = _eval_scheme(
                source=source,
                weight=spec,
                pred_df=signal,
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=open_cost,
                close_cost=close_cost,
                start_time=apply_start,
                end_time=apply_end,
                exchange_cache=exchange_cache,
            )
            row.update(
                {
                    "ok": True,
                    "costed_ir": float(metrics["costed_ir"]),
                    "costed_annret": float(metrics["costed_annret"]),
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "turnover": float(metrics["turnover"]),
                    "rows": int(metrics["rows"]),
                    "finite_rows": int(metrics["finite_rows"]),
                    "nonfinite_rows": int(metrics["nonfinite_rows"]),
                    "elapsed_sec": float(metrics["elapsed_sec"]),
                    "error_type": "",
                    "error_message": "",
                }
            )
            report = metrics["report"].copy()
            report["source_id"] = source.source_id
            report["apply_tag"] = apply_tag
            report["scheme_id"] = scheme_id
            report["candidate_id"] = f"{source.source_id}_{scheme_id}"
            report_parts.append(report)
            excess_parts.append(metrics["excess"].rename(apply_tag))
        except Exception as exc:  # noqa: BLE001
            row.update(_error_fields(exc))
        apply_rows.append(row)
        exchange_cache.clear()
        gc.collect()

    stitched_report = pd.concat(report_parts).sort_index() if report_parts else pd.DataFrame()
    stitched_excess = pd.concat(excess_parts).sort_index() if excess_parts else pd.Series(dtype=float)
    split_rows = _split_metrics(source.source_id, stitched_excess) if not stitched_excess.empty else []
    return apply_rows, split_rows, stitched_report, stitched_excess


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict predeclared score-power weighting audit for base40/gru45.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=BASE_RUN_ID)
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--output-prefix", default="predeclared_weighting_scheme_audit")
    p.add_argument(
        "--run-full-fixed-audit",
        action="store_true",
        help="Also run audit-only full-window fixed-candidate backtests; not used for selection or hard-gate judgment.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    out_dir = SCRIPT_DIR
    candidate_csv = out_dir / f"{args.output_prefix}_candidate_train_metrics_{stamp}.csv"
    selections_csv = out_dir / f"{args.output_prefix}_selections_{stamp}.csv"
    apply_csv = out_dir / f"{args.output_prefix}_apply_metrics_{stamp}.csv"
    full_csv = out_dir / f"{args.output_prefix}_full_fixed_audit_{stamp}.csv"
    splits_csv = out_dir / f"{args.output_prefix}_split_metrics_{stamp}.csv"
    report_csv = out_dir / f"{args.output_prefix}_stitched_reports_{stamp}.csv"
    report_pkl = out_dir / f"{args.output_prefix}_stitched_reports_{stamp}.pkl"
    summary_json = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = out_dir / f"{args.output_prefix}_summary_{stamp}.md"

    tracking_dir = conv._parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, args.base_run_id)
    base_cfg = conv._load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(base_cfg)
    base_port_cfg = conv._extract_port_config(base_cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    base_signal = _load_score_df(base_dir)
    base_coverage = _coverage(base_signal)
    blend_start = min(pd.Timestamp(base_coverage["start"]), TEST_START)
    blend_end = max(pd.Timestamp(base_coverage["end"]), TEST_END)
    gru_signal = _rank_ensemble(tracking_dir, GRU45_RUN_IDS, GRU45_RUN_WEIGHTS, blend_start, blend_end)
    gru_coverage = _coverage(gru_signal)
    sources = [
        SourceSpec("base40", 40, 40, "fixed base run signal; topk=40, hold_topk=40; weighting scheme only varies"),
        SourceSpec("gru45", 45, 45, "fixed GRU45 rank ensemble signal; topk=45, hold_topk=45; weighting scheme only varies"),
    ]
    signals = {"base40": base_signal, "gru45": gru_signal}
    schemes = _candidate_family()
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}

    candidate_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    apply_rows: List[Dict[str, Any]] = []
    full_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    stitched_reports: List[pd.DataFrame] = []
    selected_by_source: Dict[str, Dict[str, str]] = {}
    stitched_metrics_by_source: Dict[str, Dict[str, Any]] = {}

    for source in sources:
        train_rows, select_rows, selected_by_apply = _select_lockstep_for_source(
            source=source,
            signal=signals[source.source_id],
            schemes=schemes,
            base_port_cfg=base_port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            exchange_cache=exchange_cache,
        )
        candidate_rows.extend(train_rows)
        selection_rows.extend(select_rows)
        selected_by_source[source.source_id] = selected_by_apply

        source_apply_rows, source_split_rows, stitched_report, stitched_excess = _apply_lockstep_for_source(
            source=source,
            signal=signals[source.source_id],
            selected_by_apply=selected_by_apply,
            schemes=schemes,
            base_port_cfg=base_port_cfg,
            base_strategy_kwargs=base_strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            exchange_cache=exchange_cache,
        )
        apply_rows.extend(source_apply_rows)
        split_rows.extend(source_split_rows)
        if not stitched_report.empty:
            stitched_reports.append(stitched_report)
        try:
            source_metrics = _metrics_from_excess(stitched_excess)
            stitched_metrics_by_source[source.source_id] = {
                "costed_annret": float(source_metrics["annret"]),
                "costed_ir": float(source_metrics["ir"]),
                "max_drawdown": float(source_metrics["max_drawdown"]),
                "days": int(len(stitched_excess)),
            }
        except Exception as exc:  # noqa: BLE001
            stitched_metrics_by_source[source.source_id] = {"error": f"{type(exc).__name__}: {exc}"}

        if args.run_full_fixed_audit:
            for scheme in schemes:
                row: Dict[str, Any] = {
                    "source_id": source.source_id,
                    "scheme_id": scheme.scheme_id,
                    "candidate_id": f"{source.source_id}_{scheme.scheme_id}",
                    "phase": "full_fixed_candidate_audit",
                    "selection_rule": "audit_only_not_used_for_selection",
                    "used_for_selection": False,
                    "start": str(TEST_START.date()),
                    "end": str(TEST_END.date()),
                    **asdict(source),
                    **asdict(scheme),
                }
                try:
                    metrics = _eval_scheme(
                        source=source,
                        weight=scheme,
                        pred_df=signals[source.source_id],
                        base_port_cfg=base_port_cfg,
                        base_strategy_kwargs=base_strategy_kwargs,
                        open_cost=float(args.open_cost),
                        close_cost=float(args.close_cost),
                        start_time=TEST_START,
                        end_time=TEST_END,
                        exchange_cache=exchange_cache,
                    )
                    row.update(
                        {
                            "costed_ir": float(metrics["costed_ir"]),
                            "costed_annret": float(metrics["costed_annret"]),
                            "max_drawdown": float(metrics["max_drawdown"]),
                            "turnover": float(metrics["turnover"]),
                            "rows": int(metrics["rows"]),
                            "finite_rows": int(metrics["finite_rows"]),
                            "nonfinite_rows": int(metrics["nonfinite_rows"]),
                            "elapsed_sec": float(metrics["elapsed_sec"]),
                            "error_type": "",
                            "error_message": "",
                        }
                    )
                    split_rows.extend(_audit_split_metrics(source.source_id, scheme.scheme_id, metrics["excess"]))
                except Exception as exc:  # noqa: BLE001
                    row.update(_error_fields(exc))
                full_rows.append(row)
                exchange_cache.clear()
                gc.collect()
        exchange_cache.clear()
        gc.collect()

    all_stitched = pd.concat(stitched_reports).sort_index() if stitched_reports else pd.DataFrame()
    if not all_stitched.empty:
        all_stitched.to_csv(report_csv)
        _dump_pickle(report_pkl, all_stitched)

    evaluation_complete_by_source = {
        s.source_id: bool(
            len([r for r in apply_rows if r.get("source_id") == s.source_id]) == 4
            and all(bool(r.get("ok")) for r in apply_rows if r.get("source_id") == s.source_id)
            and set(selected_by_source.get(s.source_id, {})).issuperset({"2024H1_default", "2024H2", "2025", "2026_ytd"})
        )
        for s in sources
    }
    strict_non_test_selected = bool(
        all(bool(r.get("strict_non_test_selected", False)) for r in selection_rows)
        and all(not bool(r.get("used_for_selection", False)) for r in full_rows)
        and all(evaluation_complete_by_source.values())
    )
    pass_sources = []
    for source_id, metrics in stitched_metrics_by_source.items():
        if (
            evaluation_complete_by_source.get(source_id, False)
            and strict_non_test_selected
            and float(metrics.get("costed_ir", float("nan"))) > HARD_GATE_IR
            and float(metrics.get("costed_annret", float("nan"))) > HARD_GATE_ANNRET
        ):
            pass_sources.append(source_id)
    hard_gate_pass = bool(pass_sources)
    verdict = "BREAKTHROUGH" if hard_gate_pass else "NO_GO"

    _write_csv(candidate_csv, candidate_rows)
    _write_csv(selections_csv, selection_rows)
    _write_csv(apply_csv, apply_rows)
    _write_csv(full_csv, full_rows)
    _write_csv(splits_csv, split_rows)

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "predeclared_weighting_scheme_audit",
        "verdict": verdict,
        "hard_gate_pass": hard_gate_pass,
        "pass_sources": pass_sources,
        "evaluation_complete_by_source": evaluation_complete_by_source,
        "strict_non_test_selected": strict_non_test_selected,
        "protocol": {
            "objective": "Change portfolio weighting scheme only for fixed reliable base40/gru45 signals.",
            "source_selection": "No source is selected by test metrics; base40 and gru45 are evaluated as separate fixed-source protocols.",
            "selection_rule": (
                "For each fixed source, 2024H1 uses fixed equal default; 2024H2 weighting is selected by 2024H1; "
                "2025 weighting is selected by 2024; 2026_ytd weighting is selected by 2024-2025."
            ),
            "candidate_family": [asdict(s) for s in schemes],
            "unsupported_candidates": {
                "inverse_rank": "not evaluated because existing supported BufferedTopkWeightStrategy only supports equal and score modes"
            },
            "strategy_execution": {
                "family": "buffered_weight",
                "rebalance_mode": "daily",
                "rebalance_interval": 1,
                "topk_hold_topk_by_source": {s.source_id: {"topk": s.topk, "hold_topk": s.hold_topk} for s in sources},
            },
            "finite_report_gate": "Fail closed on any non-finite return/bench/cost/turnover/excess row in reports used for metrics.",
            "full_fixed_candidate_audit": bool(args.run_full_fixed_audit),
            "selection_score": "costed_ir + 0.20*costed_annret - 0.20*abs(max_drawdown) - 0.10*turnover",
            "hard_gate": {
                "test_start": str(TEST_START.date()),
                "test_end": str(TEST_END.date()),
                "open_cost": float(args.open_cost),
                "close_cost": float(args.close_cost),
                "ir_gt": HARD_GATE_IR,
                "annret_gt": HARD_GATE_ANNRET,
            },
        },
        "inputs": {
            "tracking_dir": str(tracking_dir),
            "base_run_id": str(base_dir.name),
            "base_pred_pkl": str(base_dir / "artifacts" / "pred.pkl"),
            "gru45_run_ids": list(GRU45_RUN_IDS),
            "gru45_run_weights": list(GRU45_RUN_WEIGHTS),
        },
        "coverage": {"base40": base_coverage, "gru45": gru_coverage},
        "selected_by_source": selected_by_source,
        "stitched_metrics_by_source": stitched_metrics_by_source,
        "candidate_train_metrics": candidate_rows,
        "selections": selection_rows,
        "apply_metrics": apply_rows,
        "full_fixed_audit_metrics": full_rows,
        "split_metrics": split_rows,
        "counts": {
            "candidate_train_rows": len(candidate_rows),
            "selection_rows": len(selection_rows),
            "apply_rows": len(apply_rows),
            "full_fixed_audit_rows": len(full_rows),
            "split_rows": len(split_rows),
            "stitched_report_rows": int(len(all_stitched)),
        },
        "runtime_sec": float(time.perf_counter() - started),
        "artifacts": {
            "candidate_train_metrics_csv": str(candidate_csv),
            "selections_csv": str(selections_csv),
            "apply_metrics_csv": str(apply_csv),
            "full_fixed_audit_csv": str(full_csv),
            "split_metrics_csv": str(splits_csv),
            "stitched_report_csv": str(report_csv) if not all_stitched.empty else "",
            "stitched_report_pkl": str(report_pkl) if not all_stitched.empty else "",
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    _write_json(summary_json, summary)
    summary_md.write_text(
        "\n".join(
            [
                f"# Predeclared Weighting Scheme Audit {stamp}",
                "",
                f"Verdict: **{verdict}**",
                f"- Hard gate pass: `{hard_gate_pass}`",
                f"- Pass sources: `{json.dumps(pass_sources, ensure_ascii=False)}`",
                f"- Strict non-test-selected: `{strict_non_test_selected}`",
                f"- Evaluation complete by source: `{json.dumps(evaluation_complete_by_source, ensure_ascii=False)}`",
                f"- Stitched metrics by source: `{json.dumps(_jsonable(stitched_metrics_by_source), ensure_ascii=False)}`",
                f"- Selected by source: `{json.dumps(selected_by_source, ensure_ascii=False)}`",
                f"- Summary JSON: `{summary_json}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "verdict": verdict,
                "hard_gate_pass": hard_gate_pass,
                "pass_sources": pass_sources,
                "strict_non_test_selected": strict_non_test_selected,
                "evaluation_complete_by_source": evaluation_complete_by_source,
                "stitched_metrics_by_source": _jsonable(stitched_metrics_by_source),
                "selected_by_source": selected_by_source,
                "summary": str(summary_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

