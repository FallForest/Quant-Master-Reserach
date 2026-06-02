#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
GRU45_RUN_WEIGHTS = (0.4, 0.2, 0.4)
FACTOR_PRED_NAME = "factor_augmented_meta_candidate_pred_20260522T120515Z.pkl"
FACTOR_SUMMARY_NAME = "factor_augmented_meta_summary_20260522T120515Z.json"


@dataclass(frozen=True)
class FusionSpec:
    candidate_id: str
    factor_weight: float
    gru_weight: float
    base_weight: float


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _error_fields(exc: Exception) -> Dict[str, str]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)).strip()
    return {"error_type": type(exc).__name__, "error_message": str(exc), "error_traceback_tail": tb}


def _load_score_df(path: Path) -> pd.DataFrame:
    return conv._as_score_df(conv._load_pickle(path)).sort_index()


def _date_values(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        return pd.to_datetime(idx.get_level_values(0))
    return pd.to_datetime(idx)


def _coverage(df: pd.DataFrame) -> Dict[str, Any]:
    dates = pd.Index(_date_values(df).normalize().unique()).sort_values()
    return {
        "rows": int(len(df)),
        "days": int(len(dates)),
        "start": str(pd.Timestamp(dates.min()).date()) if len(dates) else None,
        "end": str(pd.Timestamp(dates.max()).date()) if len(dates) else None,
    }


def _slice_df(df: pd.DataFrame, start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DataFrame:
    dates = _date_values(df)
    return df.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _candidate_family() -> List[FusionSpec]:
    return [
        FusionSpec("factor_only", 1.0, 0.0, 0.0),
        FusionSpec("f80_g20", 0.8, 0.2, 0.0),
        FusionSpec("f70_g30", 0.7, 0.3, 0.0),
        FusionSpec("f80_g10_b10", 0.8, 0.1, 0.1),
        FusionSpec("f60_g20_b20", 0.6, 0.2, 0.2),
    ]


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
        pred = _slice_df(_load_score_df(run_dir / "artifacts" / "pred.pkl"), start, end)
        ranked = conv._cross_section_rank(pred["score"].astype(float))
        ranked.name = run_id
        cols.append(ranked)
    panel = pd.concat(cols, axis=1)
    w = pd.Series(weights, index=panel.columns, dtype=float)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    score = panel.mul(w, axis=1).fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return score.to_frame("score").sort_index()


def _build_rank_fusion(spec: FusionSpec, factor_df: pd.DataFrame, gru_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    panel = pd.concat(
        [
            conv._cross_section_rank(factor_df["score"].astype(float)).rename("factor"),
            conv._cross_section_rank(gru_df["score"].astype(float)).rename("gru"),
            conv._cross_section_rank(base_df["score"].astype(float)).rename("base"),
        ],
        axis=1,
    )
    weights = pd.Series(
        {"factor": float(spec.factor_weight), "gru": float(spec.gru_weight), "base": float(spec.base_weight)}
    )
    denom = panel.notna().mul(weights, axis=1).sum(axis=1)
    score = panel.mul(weights, axis=1).fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return score.to_frame("score").sort_index()


def _strategy_combo() -> Dict[str, Any]:
    return {
        "family": "buffered_weight",
        "rebalance_mode": "weekly",
        "rebalance_interval": 1,
        "topk": 55,
        "hold_topk": 85,
        "weight_mode": "equal",
        "score_power": 1.0,
    }


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    finite_excess = pd.to_numeric(excess, errors="coerce")
    finite_excess = finite_excess[
        finite_excess.notna() & (finite_excess != float("inf")) & (finite_excess != -float("inf"))
    ].astype(float)
    if finite_excess.empty:
        raise ValueError("no finite excess-return observations")
    risk_df = risk_analysis(finite_excess.sort_index(), freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _metrics_from_report(report: pd.DataFrame) -> Dict[str, Any]:
    required_cols = ["return", "bench", "cost"]
    missing_cols = [col for col in required_cols if col not in report.columns]
    if missing_cols:
        raise KeyError(f"report missing columns: {missing_cols}")
    excess = report["return"] - report["bench"] - report["cost"]
    numeric_excess = pd.to_numeric(excess, errors="coerce")
    finite_mask = numeric_excess.notna() & (numeric_excess != float("inf")) & (numeric_excess != -float("inf"))
    finite_excess = numeric_excess.loc[finite_mask].astype(float)
    if finite_excess.empty:
        raise ValueError("report has no finite net excess-return observations")
    risk_metrics = _metrics_from_excess(finite_excess)
    turnover = pd.to_numeric(report.get("turnover", pd.Series(index=report.index, dtype=float)), errors="coerce")
    finite_turnover = turnover.loc[finite_mask].dropna()
    return {
        "costed_annret": float(risk_metrics["annret"]),
        "costed_ir": float(risk_metrics["ir"]),
        "max_drawdown": float(risk_metrics["max_drawdown"]),
        "turnover": float(finite_turnover.mean()) if not finite_turnover.empty else float("nan"),
        "rows": int(len(report)),
        "finite_rows": int(len(finite_excess)),
        "nonfinite_rows": int(len(report) - len(finite_excess)),
        "excess": finite_excess,
    }


def _score_for_selection(metrics: Dict[str, Any]) -> float:
    return (
        float(metrics["costed_ir"])
        + 0.20 * float(metrics["costed_annret"])
        - 0.25 * float(metrics["turnover"])
        - 0.20 * abs(float(metrics["max_drawdown"]))
    )


def _eval_signal(
    *,
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

    pred_slice = conv._slice_pred(pred_df, start_time, end_time)
    if pred_slice.empty:
        raise ValueError(f"empty signal slice in {start_time.date()} ~ {end_time.date()}")

    strategy = conv._build_strategy_object(
        combo=_strategy_combo(),
        pred_df=pred_slice,
        base_strategy_kwargs=base_strategy_kwargs,
    )
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
    report = conv._get_report_for_day_freq(portfolio_metric_dict)
    report = report.loc[
        (pd.to_datetime(report.index) >= pd.Timestamp(start_time))
        & (pd.to_datetime(report.index) <= pd.Timestamp(end_time))
    ].copy()
    metrics = _metrics_from_report(report)
    return {
        "costed_annret": float(metrics["costed_annret"]),
        "costed_ir": float(metrics["costed_ir"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "turnover": float(metrics["turnover"]),
        "elapsed_sec": float(time.perf_counter() - t0),
        "rows": int(len(report)),
        "finite_rows": int(metrics["finite_rows"]),
        "nonfinite_rows": int(metrics["nonfinite_rows"]),
        "report": report,
        "excess": metrics["excess"],
    }


def _selection_windows() -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str, pd.Timestamp, pd.Timestamp]]:
    return [
        (
            "2024H1",
            TEST_START,
            pd.Timestamp("2024-06-30"),
            "2024H2",
            pd.Timestamp("2024-07-01"),
            pd.Timestamp("2024-12-31"),
        ),
        (
            "2024",
            TEST_START,
            pd.Timestamp("2024-12-31"),
            "2025",
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31"),
        ),
        (
            "2024_2025",
            TEST_START,
            pd.Timestamp("2025-12-31"),
            "2026_ytd",
            pd.Timestamp("2026-01-01"),
            TEST_END,
        ),
    ]


def _apply_slices() -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    return [
        ("2024H1_default", TEST_START, pd.Timestamp("2024-06-30"), "fixed_2024H1_predeclared_factor_only"),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"), "selected_by_2024H1"),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), "selected_by_2024"),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END, "selected_by_2024_2025"),
    ]


def _split_metrics(excess: pd.Series) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    split_defs = [
        ("test_full", TEST_START, TEST_END, "stitched_lockstep"),
        ("2024H1_default", TEST_START, pd.Timestamp("2024-06-30"), "fixed_2024H1_predeclared_factor_only"),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"), "selected_by_2024H1"),
        ("2024", TEST_START, pd.Timestamp("2024-12-31"), "mixed_2024H1_default_2024H2_selected"),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), "selected_by_2024"),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END, "selected_by_2024_2025"),
    ]
    for tag, start, end, selection_rule in split_defs:
        part = excess.loc[(excess.index >= start) & (excess.index <= end)]
        if part.empty:
            continue
        rows.append(
            {
                "candidate_id": "lockstep_selected",
                "split": tag,
                "start": str(start.date()),
                "end": str(end.date()),
                "selection_rule": selection_rule,
                "days": int(len(part)),
                **_metrics_from_excess(part),
            }
        )
    return rows


def _audit_split_metrics(candidate_id: str, excess: pd.Series) -> List[Dict[str, Any]]:
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
                "candidate_id": candidate_id,
                "split": split,
                "start": str(start.date()),
                "end": str(end.date()),
                "selection_rule": "full_fixed_candidate_audit_not_used_for_selection",
                "days": int(len(part)),
                **_metrics_from_excess(part),
            }
        )
    return rows


def _load_existing_selection(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("selections", [])
    selected: Dict[str, str] = {}
    for row in rows:
        if row.get("selected"):
            selected[str(row["apply_tag"])] = str(row["candidate_id"])
    return selected


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict signal-level factor/meta + gru45 + base40 rank fusion lockstep.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=BASE_RUN_ID)
    p.add_argument("--factor-pred-pkl", default="")
    p.add_argument("--factor-summary-json", default="")
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--output-prefix", default="factor_meta_gru_base_fusion_lockstep")
    p.add_argument("--skip-full-audit", action="store_true")
    p.add_argument("--verify-selection-json", default="", help="Rerun apply windows using frozen selections from a prior summary JSON.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    trans_dir = Path(__file__).resolve().parent
    candidate_csv = trans_dir / f"{args.output_prefix}_candidate_train_metrics_{stamp}.csv"
    selections_csv = trans_dir / f"{args.output_prefix}_selections_{stamp}.csv"
    full_csv = trans_dir / f"{args.output_prefix}_full_metrics_{stamp}.csv"
    splits_csv = trans_dir / f"{args.output_prefix}_split_metrics_{stamp}.csv"
    apply_csv = trans_dir / f"{args.output_prefix}_apply_metrics_{stamp}.csv"
    selected_report_csv = trans_dir / f"{args.output_prefix}_stitched_report_{stamp}.csv"
    selected_pred_pkl = trans_dir / f"{args.output_prefix}_selected_signal_{stamp}.pkl"
    summary_json = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = trans_dir / f"{args.output_prefix}_summary_{stamp}.md"

    factor_pred_path = Path(args.factor_pred_pkl) if args.factor_pred_pkl else trans_dir / FACTOR_PRED_NAME
    factor_summary_path = Path(args.factor_summary_json) if args.factor_summary_json else trans_dir / FACTOR_SUMMARY_NAME
    tracking_dir = conv._parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, args.base_run_id)
    base_cfg = conv._load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(base_cfg)
    base_port_cfg = conv._extract_port_config(base_cfg)
    base_strategy_kwargs = dict(base_port_cfg.get("strategy", {}).get("kwargs", {}))
    base_strategy_kwargs.pop("signal", None)

    factor_df = _load_score_df(factor_pred_path)
    base_df = _load_score_df(base_dir / "artifacts" / "pred.pkl")
    coverage_start = min(pd.Timestamp(_coverage(factor_df)["start"]), pd.Timestamp(_coverage(base_df)["start"]))
    coverage_end = max(pd.Timestamp(_coverage(factor_df)["end"]), pd.Timestamp(_coverage(base_df)["end"]))
    gru_df = _rank_ensemble(tracking_dir, GRU45_RUN_IDS, GRU45_RUN_WEIGHTS, coverage_start, coverage_end)

    specs = _candidate_family()
    signals = {spec.candidate_id: _build_rank_fusion(spec, factor_df, gru_df, base_df) for spec in specs}
    spec_by_id = {spec.candidate_id: spec for spec in specs}
    default_candidate_id = specs[0].candidate_id
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    candidate_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    selected_by_apply: Dict[str, str] = {"2024H1_default": default_candidate_id}

    verify_selection_json = Path(args.verify_selection_json) if args.verify_selection_json else None
    verification_mode = bool(verify_selection_json)
    if verification_mode:
        selected_by_apply = _load_existing_selection(verify_selection_json)
    else:
        for train_tag, train_start, train_end, apply_tag, apply_start, apply_end in _selection_windows():
            scored: List[Dict[str, Any]] = []
            for spec in specs:
                row: Dict[str, Any] = {
                    "train_tag": train_tag,
                    "apply_tag": apply_tag,
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                    "apply_start": str(apply_start.date()),
                    "apply_end": str(apply_end.date()),
                    "ok": False,
                    **asdict(spec),
                }
                try:
                    metrics = _eval_signal(
                        pred_df=signals[spec.candidate_id],
                        base_port_cfg=base_port_cfg,
                        base_strategy_kwargs=base_strategy_kwargs,
                        open_cost=float(args.open_cost),
                        close_cost=float(args.close_cost),
                        start_time=train_start,
                        end_time=train_end,
                        exchange_cache=exchange_cache,
                    )
                    selection_score = _score_for_selection(metrics)
                    row.update(
                        {
                            "ok": True,
                            "selection_score": float(selection_score),
                            "costed_ir": float(metrics["costed_ir"]),
                            "costed_annret": float(metrics["costed_annret"]),
                            "max_drawdown": float(metrics["max_drawdown"]),
                            "turnover": float(metrics["turnover"]),
                            "elapsed_sec": float(metrics["elapsed_sec"]),
                            "rows": int(metrics["rows"]),
                            "selection_metric": "ir + 0.20*annret - 0.25*turnover - 0.20*abs(max_drawdown)",
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
                        "apply_tag": apply_tag,
                        "selected": False,
                        "selection_train_tag": train_tag,
                        "error_type": "NoSelectableFusion",
                        "error_message": f"no valid fusion in {train_tag}",
                    }
                )
                continue
            best = ranked[0]
            selected_by_apply[apply_tag] = str(best["candidate_id"])
            selection_rows.append(
                {
                    "apply_tag": apply_tag,
                    "apply_start": str(apply_start.date()),
                    "apply_end": str(apply_end.date()),
                    "selected": True,
                    "selection_train_tag": train_tag,
                    "candidate_id": str(best["candidate_id"]),
                    "factor_weight": float(best["factor_weight"]),
                    "gru_weight": float(best["gru_weight"]),
                    "base_weight": float(best["base_weight"]),
                    "selection_score": float(best["selection_score"]),
                    "train_ir": float(best["costed_ir"]),
                    "train_annret": float(best["costed_annret"]),
                    "train_mdd": float(best["max_drawdown"]),
                    "train_turnover": float(best["turnover"]),
                    "strict_prior_window_selection": True,
                    "candidate_family_predeclared": True,
                }
            )

        selection_rows.insert(
            0,
            {
                "apply_tag": "2024H1_default",
                "apply_start": str(TEST_START.date()),
                "apply_end": str(pd.Timestamp("2024-06-30").date()),
                "selected": True,
                "selection_train_tag": "fixed_predeclared_default_no_same_window_selection",
                "candidate_id": default_candidate_id,
                "factor_weight": 1.0,
                "gru_weight": 0.0,
                "base_weight": 0.0,
                "selection_score": "",
                "train_ir": "",
                "train_annret": "",
                "train_mdd": "",
                "train_turnover": "",
                "strict_prior_window_selection": True,
                "candidate_family_predeclared": True,
            },
        )

    apply_rows: List[Dict[str, Any]] = []
    full_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    selected_signal_parts: List[pd.DataFrame] = []
    for apply_tag, apply_start, apply_end, selection_rule in _apply_slices():
        candidate_id = selected_by_apply.get(apply_tag)
        if not candidate_id or candidate_id not in signals:
            continue
        selected_signal_parts.append(_slice_df(signals[candidate_id], apply_start, apply_end))

    if selected_signal_parts:
        selected_signal = pd.concat(selected_signal_parts).sort_index()
        with selected_pred_pkl.open("wb") as f:
            pickle.dump(selected_signal, f)
    else:
        selected_signal = pd.DataFrame(columns=["score"])

    stitched_excess_parts: List[pd.Series] = []
    stitched_report_parts: List[pd.DataFrame] = []
    for apply_tag, apply_start, apply_end, selection_rule in _apply_slices():
        candidate_id = selected_by_apply.get(apply_tag)
        row: Dict[str, Any] = {
            "candidate_id": candidate_id or "",
            "apply_tag": apply_tag,
            "start": str(apply_start.date()),
            "end": str(apply_end.date()),
            "selection_rule": selection_rule,
            "ok": False,
            "continuous_lockstep_backtest": False,
            "stitched_from_apply_window_backtest": True,
        }
        if not candidate_id or candidate_id not in signals:
            row.update({"error_type": "MissingSelection", "error_message": f"no selected fusion for {apply_tag}"})
            apply_rows.append(row)
            continue
        spec = spec_by_id[candidate_id]
        row.update(asdict(spec))
        try:
            apply_eval = _eval_signal(
                pred_df=signals[candidate_id],
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
                start_time=apply_start,
                end_time=apply_end,
                exchange_cache=exchange_cache,
            )
            apply_report = apply_eval["report"]
            metrics = _metrics_from_report(apply_report)
            row.update(
                {
                    "ok": True,
                    "costed_ir": float(metrics["costed_ir"]),
                    "costed_annret": float(metrics["costed_annret"]),
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "turnover": float(metrics["turnover"]),
                    "elapsed_sec": float(apply_eval["elapsed_sec"]),
                    "rows": int(metrics["rows"]),
                    "finite_rows": int(metrics["finite_rows"]),
                    "nonfinite_rows": int(metrics["nonfinite_rows"]),
                    "error_type": "",
                    "error_message": "",
                }
            )
            apply_rows.append(row)
            tagged_report = apply_report.copy()
            tagged_report["apply_tag"] = apply_tag
            tagged_report["candidate_id"] = candidate_id
            stitched_report_parts.append(tagged_report)
            stitched_excess_parts.append(metrics["excess"].rename(apply_tag))
        except Exception as exc:  # noqa: BLE001
            row.update(_error_fields(exc))
            apply_rows.append(row)

    if stitched_report_parts:
        selected_full_report = pd.concat(stitched_report_parts).sort_index()
        selected_full_report.to_csv(selected_report_csv)
        stitched_report_rows = int(len(selected_full_report))
    else:
        stitched_report_rows = 0

    if stitched_excess_parts:
        stitched_excess = pd.concat(stitched_excess_parts).sort_index()
    else:
        stitched_excess = pd.Series(dtype=float)
    try:
        lockstep_metrics = _metrics_from_excess(stitched_excess)
    except Exception:
        lockstep_metrics = {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan")}

    evaluation_complete = all(bool(row.get("ok")) for row in apply_rows) and len(apply_rows) == 4
    full_rows.append(
        {
            "candidate_id": "lockstep_selected",
            "phase": "strict_lockstep_selected",
            "selection_rule": "2024H1 default; later windows selected only by prior windows",
            "start": str(TEST_START.date()),
            "end": str(TEST_END.date()),
            "days": int(len(stitched_excess)),
            "costed_annret": float(lockstep_metrics["annret"]),
            "costed_ir": float(lockstep_metrics["ir"]),
            "max_drawdown": float(lockstep_metrics["max_drawdown"]),
            "used_for_selection": False,
            "stitched_from_apply_window_backtests": True,
            "continuous_lockstep_backtest": False,
        }
    )
    if not stitched_excess.empty:
        split_rows.extend(_split_metrics(stitched_excess))

    if not args.skip_full_audit:
        for spec in specs:
            row: Dict[str, Any] = {
                "candidate_id": spec.candidate_id,
                "phase": "full_fixed_candidate_audit",
                "selection_rule": "audit_only_not_used_for_selection",
                "start": str(TEST_START.date()),
                "end": str(TEST_END.date()),
                "used_for_selection": False,
                **asdict(spec),
            }
            try:
                metrics = _eval_signal(
                    pred_df=signals[spec.candidate_id],
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
                        "days": int(metrics["rows"]),
                        "costed_annret": float(metrics["costed_annret"]),
                        "costed_ir": float(metrics["costed_ir"]),
                        "max_drawdown": float(metrics["max_drawdown"]),
                        "turnover": float(metrics["turnover"]),
                        "elapsed_sec": float(metrics["elapsed_sec"]),
                        "error_type": "",
                        "error_message": "",
                    }
                )
                split_rows.extend(_audit_split_metrics(spec.candidate_id, metrics["excess"]))
            except Exception as exc:  # noqa: BLE001
                row.update(_error_fields(exc))
            full_rows.append(row)

    strict_non_test_selected = bool(
        set(selected_by_apply).issuperset({"2024H1_default", "2024H2", "2025", "2026_ytd"})
        and all(row.get("selection_rule") != "selected_by_full_test" for row in apply_rows)
        and all(not bool(row.get("used_for_selection", False)) for row in full_rows)
    )
    hard_gate_pass = bool(
        evaluation_complete
        and strict_non_test_selected
        and float(lockstep_metrics["ir"]) > HARD_GATE_IR
        and float(lockstep_metrics["annret"]) > HARD_GATE_ANNRET
    )
    verdict = "BREAKTHROUGH" if hard_gate_pass else "NO_GO"

    _write_csv(candidate_csv, candidate_rows)
    _write_csv(selections_csv, selection_rows)
    _write_csv(apply_csv, apply_rows)
    _write_csv(full_csv, full_rows)
    _write_csv(splits_csv, split_rows)

    factor_summary = {}
    if factor_summary_path.exists():
        factor_summary = json.loads(factor_summary_path.read_text(encoding="utf-8"))

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "factor_meta_gru_base_fusion_lockstep",
        "verdict": verdict,
        "hard_gate_pass": hard_gate_pass,
        "evaluation_complete": evaluation_complete,
        "strict_non_test_selected": strict_non_test_selected,
        "verification_mode": verification_mode,
        "verify_selection_json": str(verify_selection_json) if verify_selection_json else "",
        "protocol": {
            "selection_rule": (
                "Tiny predeclared rank-fusion family. 2024H1 uses fixed factor_only default; "
                "2024H2 selected by 2024H1; 2025 selected by 2024; 2026_ytd selected by 2024-2025."
            ),
            "test_rule": "No fusion is selected using its apply window or full 2024-01-01..2026-04-30 metrics.",
            "rank_normalization": "Each source score is cross-sectionally ranked by date before weighted fusion.",
            "full_fixed_candidate_audit": not bool(args.skip_full_audit),
            "net_cost": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
            "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
            "strategy_execution": _strategy_combo(),
            "selection_score": "costed_ir + 0.20*costed_annret - 0.25*turnover - 0.20*abs(max_drawdown)",
        },
        "inputs": {
            "factor_candidate_pred_pkl": str(factor_pred_path),
            "factor_summary_json": str(factor_summary_path),
            "base_run_id": str(base_dir.name),
            "base_pred_pkl": str(base_dir / "artifacts" / "pred.pkl"),
            "gru45_run_ids": list(GRU45_RUN_IDS),
            "gru45_run_weights": list(GRU45_RUN_WEIGHTS),
            "factor_original_metrics": factor_summary.get("metrics", {}).get("meta_full", {}),
        },
        "coverage": {
            "factor_augmented_meta": _coverage(factor_df),
            "base40": _coverage(base_df),
            "gru45_rank_source": _coverage(gru_df),
        },
        "candidate_family": [asdict(spec) for spec in specs],
        "selected_by_apply": selected_by_apply,
        "selections": selection_rows,
        "candidate_train_metrics": candidate_rows,
        "apply_metrics": apply_rows,
        "full_metrics": full_rows,
        "split_metrics": split_rows,
        "stitched_metrics": lockstep_metrics,
        "counts": {
            "candidate_train_rows": len(candidate_rows),
            "selection_rows": len(selection_rows),
            "apply_rows": len(apply_rows),
            "full_rows": len(full_rows),
            "split_rows": len(split_rows),
            "stitched_observations": int(len(stitched_excess)),
            "stitched_report_rows": stitched_report_rows,
        },
        "runtime_sec": float(time.perf_counter() - started),
        "artifacts": {
            "candidate_train_metrics_csv": str(candidate_csv),
            "selections_csv": str(selections_csv),
            "apply_metrics_csv": str(apply_csv),
            "full_metrics_csv": str(full_csv),
            "split_metrics_csv": str(splits_csv),
            "stitched_report_csv": str(selected_report_csv) if stitched_report_rows else "",
            "selected_signal_pkl": str(selected_pred_pkl) if selected_signal_parts else "",
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    _write_json(summary_json, summary)
    summary_md.write_text(
        "\n".join(
            [
                f"# Factor Meta GRU Base Fusion Lockstep {stamp}",
                "",
                f"Verdict: **{verdict}**",
                f"- Hard gate pass: `{hard_gate_pass}`",
                f"- Strict non-test-selected: `{strict_non_test_selected}`",
                f"- Verification mode: `{verification_mode}`",
                f"- Full stitched metrics: `{json.dumps(lockstep_metrics, ensure_ascii=False)}`",
                f"- Selected fusions: `{json.dumps(selected_by_apply, ensure_ascii=False)}`",
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
                "strict_non_test_selected": strict_non_test_selected,
                "verification_mode": verification_mode,
                "stitched_metrics": lockstep_metrics,
                "selected_by_apply": selected_by_apply,
                "summary": str(summary_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

