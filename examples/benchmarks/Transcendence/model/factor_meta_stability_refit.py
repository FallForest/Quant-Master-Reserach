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

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import signal_portfolio_conversion_scan as conv
from quant_master.contrib.evaluate import risk_analysis


TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2026-04-30")
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
BASE_RUN_ID = "7406e47063e9479cb34d300b9ed03bad"
FACTOR_PRED_NAME = "factor_augmented_meta_candidate_pred_20260522T120515Z.pkl"
FACTOR_SUMMARY_NAME = "factor_augmented_meta_summary_20260522T120515Z.json"


@dataclass(frozen=True)
class TransformSpec:
    name: str
    smooth_window: int
    rank_winsor: bool = False
    blend_base_rank_weight: float = 0.0

    @property
    def transform_id(self) -> str:
        bits = [self.name, f"sw{self.smooth_window}"]
        if self.rank_winsor:
            bits.append("winsor")
        if self.blend_base_rank_weight:
            bits.append(f"base{self.blend_base_rank_weight:.2f}".replace(".", "p"))
        return "_".join(bits)


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


def _slice_df(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = _date_values(df)
    return df.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _cross_section_rank(score: pd.Series) -> pd.Series:
    idx = score.index
    if isinstance(idx, pd.MultiIndex):
        return score.groupby(level=0).rank(method="average", pct=True)
    return score.groupby(idx).rank(method="average", pct=True)


def _smooth_by_instrument(score: pd.Series, window: int) -> pd.Series:
    if int(window) <= 1:
        return score
    if not isinstance(score.index, pd.MultiIndex):
        return score.rolling(int(window), min_periods=1).mean()
    return score.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(int(window), min_periods=1).mean()
    )


def _rank_winsor_score(score: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    rank = _cross_section_rank(score.astype(float))
    return rank.clip(lower, upper)


def _blend_rank_scores(factor: pd.Series, base: pd.Series, base_weight: float) -> pd.Series:
    factor_rank = _cross_section_rank(factor.astype(float)).rename("factor")
    base_rank = _cross_section_rank(base.astype(float)).rename("base")
    panel = pd.concat([factor_rank, base_rank], axis=1)
    weights = pd.Series({"factor": 1.0 - float(base_weight), "base": float(base_weight)})
    denom = panel.notna().mul(weights, axis=1).sum(axis=1)
    return panel.mul(weights, axis=1).fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()


def _candidate_family() -> List[TransformSpec]:
    return [
        TransformSpec(name="raw_default", smooth_window=1),
        TransformSpec(name="smooth2", smooth_window=2),
        TransformSpec(name="smooth3", smooth_window=3),
        TransformSpec(name="smooth5", smooth_window=5),
        TransformSpec(name="winsor_rank", smooth_window=1, rank_winsor=True),
        TransformSpec(name="smooth3_base20", smooth_window=3, blend_base_rank_weight=0.20),
    ]


def _build_signal(spec: TransformSpec, factor_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    factor = factor_df["score"].astype(float)
    score = _smooth_by_instrument(factor, spec.smooth_window)
    if spec.rank_winsor:
        score = _rank_winsor_score(score)
    if spec.blend_base_rank_weight:
        base_score = base_df["score"].astype(float)
        score = _blend_rank_scores(score, base_score, spec.blend_base_rank_weight)
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
    risk_df = risk_analysis(excess.sort_index(), freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _score_for_selection(metrics: Dict[str, Any]) -> float:
    # IR is the missing gate; turnover is penalized to target the stability hypothesis.
    return (
        float(metrics["costed_ir"])
        + 0.20 * float(metrics["costed_annret"])
        - 0.35 * float(metrics["turnover"])
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
    annret, ir, maxdd, turnover = conv._calc_costed_metrics(report)
    excess = report["return"] - report["bench"] - report["cost"]
    return {
        "costed_annret": float(annret),
        "costed_ir": float(ir),
        "max_drawdown": float(maxdd),
        "turnover": float(turnover),
        "elapsed_sec": float(time.perf_counter() - t0),
        "rows": int(len(report)),
        "report": report,
        "excess": excess,
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
        ("2024H1_default", TEST_START, pd.Timestamp("2024-06-30"), "fixed_2024H1_default_raw"),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"), "selected_by_2024H1"),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), "selected_by_2024"),
        ("2026_ytd", pd.Timestamp("2026-01-01"), TEST_END, "selected_by_2024_2025"),
    ]


def _split_metrics(excess: pd.Series) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tag, start, end, selection_rule in _apply_slices():
        part = excess.loc[(excess.index >= start) & (excess.index <= end)]
        if part.empty:
            continue
        rows.append(
            {
                "split": tag,
                "start": str(start.date()),
                "end": str(end.date()),
                "selection_rule": selection_rule,
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
            selected[str(row["apply_tag"])] = str(row["transform_id"])
    return selected


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict prior-window stability refit for factor_augmented_meta.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=BASE_RUN_ID)
    p.add_argument("--factor-pred-pkl", default="")
    p.add_argument("--factor-summary-json", default="")
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--output-prefix", default="factor_meta_stability_refit")
    p.add_argument("--verify-selection-json", default="", help="Rerun apply windows using selections from a prior summary JSON.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    trans_dir = Path(__file__).resolve().parent
    candidate_csv = trans_dir / f"{args.output_prefix}_candidate_train_metrics_{stamp}.csv"
    selections_csv = trans_dir / f"{args.output_prefix}_selections_{stamp}.csv"
    apply_csv = trans_dir / f"{args.output_prefix}_apply_metrics_{stamp}.csv"
    splits_csv = trans_dir / f"{args.output_prefix}_split_metrics_{stamp}.csv"
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
    specs = _candidate_family()
    signals = {spec.transform_id: _build_signal(spec, factor_df, base_df) for spec in specs}
    spec_by_id = {spec.transform_id: spec for spec in specs}
    default_transform_id = specs[0].transform_id
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    candidate_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    selected_by_apply: Dict[str, str] = {"2024H1_default": default_transform_id}

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
                    "transform_id": spec.transform_id,
                    **asdict(spec),
                }
                try:
                    metrics = _eval_signal(
                        pred_df=signals[spec.transform_id],
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
                        "error_type": "NoSelectableTransform",
                        "error_message": f"no valid transform in {train_tag}",
                    }
                )
                continue
            best = ranked[0]
            selected_by_apply[apply_tag] = str(best["transform_id"])
            selection_rows.append(
                {
                    "apply_tag": apply_tag,
                    "apply_start": str(apply_start.date()),
                    "apply_end": str(apply_end.date()),
                    "selected": True,
                    "selection_train_tag": train_tag,
                    "transform_id": str(best["transform_id"]),
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
                "transform_id": default_transform_id,
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
    excess_parts: List[pd.Series] = []
    report_parts: List[pd.DataFrame] = []
    selected_signal_parts: List[pd.DataFrame] = []
    for apply_tag, apply_start, apply_end, selection_rule in _apply_slices():
        transform_id = selected_by_apply.get(apply_tag)
        row: Dict[str, Any] = {
            "apply_tag": apply_tag,
            "start": str(apply_start.date()),
            "end": str(apply_end.date()),
            "selection_rule": selection_rule,
            "ok": False,
            "transform_id": transform_id or "",
        }
        if not transform_id or transform_id not in signals:
            row.update({"error_type": "MissingSelection", "error_message": f"no selected transform for {apply_tag}"})
            apply_rows.append(row)
            continue
        spec = spec_by_id[transform_id]
        row.update(asdict(spec))
        try:
            metrics = _eval_signal(
                pred_df=signals[transform_id],
                base_port_cfg=base_port_cfg,
                base_strategy_kwargs=base_strategy_kwargs,
                open_cost=float(args.open_cost),
                close_cost=float(args.close_cost),
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
                    "elapsed_sec": float(metrics["elapsed_sec"]),
                    "rows": int(metrics["rows"]),
                    "error_type": "",
                    "error_message": "",
                }
            )
            excess_parts.append(metrics["excess"])
            report_parts.append(metrics["report"])
            selected_signal_parts.append(_slice_df(signals[transform_id], apply_start, apply_end))
        except Exception as exc:  # noqa: BLE001
            row.update(_error_fields(exc))
        apply_rows.append(row)

    evaluation_complete = len(excess_parts) == 4 and all(bool(row.get("ok")) for row in apply_rows)
    if excess_parts:
        stitched_excess = pd.concat(excess_parts).sort_index()
        full_metrics = _metrics_from_excess(stitched_excess)
        split_rows = _split_metrics(stitched_excess)
    else:
        stitched_excess = pd.Series(dtype=float)
        full_metrics = {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan")}
        split_rows = []

    stitched_report_rows = 0
    if report_parts:
        stitched_report = pd.concat(report_parts).sort_index()
        stitched_report_rows = int(len(stitched_report))
        stitched_report.to_csv(selected_report_csv)
    if selected_signal_parts:
        selected_signal = pd.concat(selected_signal_parts).sort_index()
        with selected_pred_pkl.open("wb") as f:
            pickle.dump(selected_signal, f)

    strict_non_test_selected = bool(
        set(selected_by_apply).issuperset({"2024H1_default", "2024H2", "2025", "2026_ytd"})
        and all(row.get("selection_rule") != "selected_by_full_test" for row in apply_rows)
    )
    hard_gate_pass = bool(
        evaluation_complete
        and strict_non_test_selected
        and float(full_metrics["ir"]) > HARD_GATE_IR
        and float(full_metrics["annret"]) > HARD_GATE_ANNRET
    )
    verdict = "BREAKTHROUGH" if hard_gate_pass else "NO_GO"

    _write_csv(candidate_csv, candidate_rows)
    _write_csv(selections_csv, selection_rows)
    _write_csv(apply_csv, apply_rows)
    _write_csv(splits_csv, split_rows)

    factor_summary = {}
    if factor_summary_path.exists():
        factor_summary = json.loads(factor_summary_path.read_text(encoding="utf-8"))

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "factor_meta_stability_refit",
        "verdict": verdict,
        "hard_gate_pass": hard_gate_pass,
        "evaluation_complete": evaluation_complete,
        "strict_non_test_selected": strict_non_test_selected,
        "verification_mode": verification_mode,
        "verify_selection_json": str(verify_selection_json) if verify_selection_json else "",
        "protocol": {
            "selection_rule": (
                "Small predeclared transform family. 2024H1 uses fixed raw default; "
                "2024H2 selected by 2024H1; 2025 selected by 2024; 2026_ytd selected by 2024-2025."
            ),
            "test_rule": "No transform is selected using its apply window or full 2024-01-01..2026-04-30 metrics.",
            "net_cost": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
            "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
            "strategy_execution": _strategy_combo(),
            "selection_score": "costed_ir + 0.20*costed_annret - 0.35*turnover - 0.20*abs(max_drawdown)",
        },
        "inputs": {
            "factor_candidate_pred_pkl": str(factor_pred_path),
            "factor_summary_json": str(factor_summary_path),
            "base_run_id": str(base_dir.name),
            "base_pred_pkl": str(base_dir / "artifacts" / "pred.pkl"),
            "factor_original_metrics": factor_summary.get("metrics", {}).get("meta_full", {}),
        },
        "coverage": {
            "factor_augmented_meta": _coverage(factor_df),
            "base40": _coverage(base_df),
        },
        "candidate_family": [asdict(spec) | {"transform_id": spec.transform_id} for spec in specs],
        "selected_by_apply": selected_by_apply,
        "selections": selection_rows,
        "apply_metrics": apply_rows,
        "stitched_metrics": full_metrics,
        "split_metrics": split_rows,
        "counts": {
            "candidate_train_rows": len(candidate_rows),
            "selection_rows": len(selection_rows),
            "apply_rows": len(apply_rows),
            "stitched_observations": int(len(stitched_excess)),
            "stitched_report_rows": stitched_report_rows,
        },
        "runtime_sec": float(time.perf_counter() - started),
        "artifacts": {
            "candidate_train_metrics_csv": str(candidate_csv),
            "selections_csv": str(selections_csv),
            "apply_metrics_csv": str(apply_csv),
            "split_metrics_csv": str(splits_csv),
            "stitched_report_csv": str(selected_report_csv) if report_parts else "",
            "selected_signal_pkl": str(selected_pred_pkl) if selected_signal_parts else "",
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    _write_json(summary_json, summary)
    summary_md.write_text(
        "\n".join(
            [
                f"# Factor Meta Stability Refit {stamp}",
                "",
                f"Verdict: **{verdict}**",
                f"- Hard gate pass: `{hard_gate_pass}`",
                f"- Strict non-test-selected: `{strict_non_test_selected}`",
                f"- Verification mode: `{verification_mode}`",
                f"- Full stitched metrics: `{json.dumps(full_metrics, ensure_ascii=False)}`",
                f"- Selected transforms: `{json.dumps(selected_by_apply, ensure_ascii=False)}`",
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
                "stitched_metrics": full_metrics,
                "selected_by_apply": selected_by_apply,
                "summary": str(summary_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
