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


HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27


@dataclass(frozen=True)
class Combo:
    transform: str
    blend_weight: float
    family: str
    topk: int
    n_drop: int
    hold_topk: int


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _metrics_from_excess(excess: pd.Series) -> Dict[str, float]:
    risk_df = risk_analysis(excess.sort_index(), freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
    }


def _candidate_grid(grid_mode: str) -> List[Combo]:
    if grid_mode == "tiny":
        transforms = ("raw",)
        blend_weights = (0.2,)
        topk_ndrop = ((40, 2),)
        hold_topks = (65,)
        families = ("buffered_weight",)
    elif grid_mode == "smoke":
        transforms = ("raw", "winsor_rank")
        blend_weights = (0.2,)
        topk_ndrop = ((40, 2), (45, 4))
        hold_topks = (65,)
        families = ("buffered_weight", "topk_dropout")
    elif grid_mode == "topkdrop_smoke":
        transforms = ("raw", "winsor_rank")
        blend_weights = (0.2,)
        topk_ndrop = ((40, 2), (45, 4), (50, 5), (55, 0))
        hold_topks = ()
        families = ("topk_dropout",)
    elif grid_mode == "topkdrop_fullish":
        transforms = ("raw", "inverted", "winsor_rank", "smooth3", "smooth5")
        blend_weights = (0.1, 0.2, 0.3, 0.4)
        topk_ndrop = ((30, 1), (35, 2), (40, 2), (45, 4), (50, 5), (55, 0), (60, 6), (70, 7))
        hold_topks = ()
        families = ("topk_dropout",)
    elif grid_mode == "full":
        transforms = ("raw", "inverted", "winsor_rank", "smooth3")
        blend_weights = (0.2, 0.3)
        topk_ndrop = ((40, 2), (45, 4), (50, 5), (55, 0))
        hold_topks = (65, 75, 85)
        families = ("buffered_weight", "topk_dropout")
    else:
        raise ValueError(f"unsupported grid_mode={grid_mode}")

    combos = []
    for transform in transforms:
        for blend_weight in blend_weights:
            for topk, n_drop in topk_ndrop:
                if "topk_dropout" in families:
                    combos.append(
                        Combo(
                            transform=transform,
                            blend_weight=blend_weight,
                            family="topk_dropout",
                            topk=topk,
                            n_drop=n_drop,
                            hold_topk=topk,
                        )
                    )
                if "buffered_weight" in families:
                    for hold_topk in hold_topks:
                        combos.append(
                            Combo(
                                transform=transform,
                                blend_weight=blend_weight,
                                family="buffered_weight",
                                topk=topk,
                                n_drop=n_drop,
                                hold_topk=hold_topk,
                            )
                        )
    return combos


def _combo_to_strategy(combo: Combo) -> Dict[str, Any]:
    out = {
        "family": combo.family,
        "rebalance_mode": "weekly",
        "rebalance_interval": 1,
        "topk": int(combo.topk),
        "n_drop": int(combo.n_drop),
        "hold_topk": int(combo.hold_topk),
        "weight_mode": "equal",
        "score_power": 1.0,
    }
    if combo.family == "topk_dropout":
        out["hold_topk"] = int(combo.topk)
    return out


def _error_category(exc: Exception) -> str:
    msg = str(exc)
    if "NoneType" in msg and "float" in msg:
        return "missing_deal_price_or_market_data"
    if "not enough" in msg and "require" in msg:
        return "position_share_drift"
    if "empty signal slice" in msg:
        return "empty_signal_slice"
    return "backtest_exception"


def _error_fields(exc: Exception) -> Dict[str, str]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)).strip()
    return {
        "error_type": type(exc).__name__,
        "error_category": _error_category(exc),
        "error_message": str(exc),
        "error_traceback_tail": tb,
    }


def _rank_train_candidates(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda x: float(x["selection_score"]), reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["train_rank"] = i
    return ranked


def _eval_combo_period_with_series(
    *,
    combo: Dict[str, Any],
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
        raise ValueError(f"empty signal slice in {start_time} ~ {end_time}")

    strategy_obj = conv._build_strategy_object(combo=combo, pred_df=pred_slice, base_strategy_kwargs=base_strategy_kwargs)
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    min_cost = float(exchange_kwargs.get("min_cost", 5))
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
        strategy=strategy_obj,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    report_df = conv._get_report_for_day_freq(portfolio_metric_dict)
    annret, ir, maxdd, turnover = conv._calc_costed_metrics(report_df)
    excess = report_df["return"] - report_df["bench"] - report_df["cost"]
    return {
        "costed_annret": float(annret),
        "costed_ir": float(ir),
        "max_drawdown": float(maxdd),
        "turnover": float(turnover),
        "elapsed_sec": float(time.perf_counter() - t0),
        "report_df": report_df,
        "excess_series": excess,
    }


def _slice_dates(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("2024H1_train", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-06-30")),
        ("2024H2_apply", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
        ("2024_train", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31")),
        ("2025_apply", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("up_to_2025_train", pd.Timestamp("2024-01-02"), pd.Timestamp("2025-12-31")),
        ("2026_apply", pd.Timestamp("2026-01-01"), min(pd.Timestamp("2026-04-30"), end)),
    ]


def _year_rows(excess: pd.Series) -> List[Dict[str, Any]]:
    rows = []
    for tag, st, ed in [
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("2026_ytd", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-30")),
    ]:
        s = excess.loc[(excess.index >= st) & (excess.index <= ed)]
        if len(s):
            rows.append({"split": tag, **_metrics_from_excess(s)})
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict lockstep validation for signal conversion candidates.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=conv.SOTA_RUN_ID)
    p.add_argument("--signal-runs", default="gru_bcbecf55:bcbecf55,metalabel_top_bottom_29864:29864,metalabel_rank_4a98:4a98")
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--grid-mode", choices=["tiny", "smoke", "topkdrop_smoke", "topkdrop_fullish", "full"], default="full")
    p.add_argument("--max-candidates", type=int, default=0, help="Cap evaluated train candidates per lockstep segment; 0 means no cap.")
    p.add_argument("--max-apply-plans", type=int, default=0, help="Cap lockstep train/apply segments for smoke runs; 0 means all.")
    p.add_argument("--checkpoint-every", type=int, default=10, help="Write checkpoint after this many eval rows; 0 disables interval writes.")
    p.add_argument("--output-prefix", default="strict_conversion_lockstep")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    trans_dir = Path(__file__).resolve().parent
    summary_json = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    summary_md = trans_dir / f"{args.output_prefix}_summary_{stamp}.md"
    checkpoint_json = trans_dir / f"{args.output_prefix}_checkpoint_{stamp}.json"
    selections_csv = trans_dir / f"{args.output_prefix}_selections_{stamp}.csv"
    splits_csv = trans_dir / f"{args.output_prefix}_splits_{stamp}.csv"
    eval_csv = trans_dir / f"{args.output_prefix}_eval_{stamp}.csv"
    error_csv = trans_dir / f"{args.output_prefix}_errors_{stamp}.csv"

    tracking_dir = conv._parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, args.base_run_id)
    wf_cfg = conv._load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(wf_cfg)
    port_cfg = conv._extract_port_config(wf_cfg)
    strategy_kwargs = dict(port_cfg.get("strategy", {}).get("kwargs", {}))
    strategy_kwargs.pop("signal", None)

    base_pred = conv._as_score_df(conv._load_pickle(base_dir / "artifacts" / "pred.pkl"))
    coverage_dates = conv._pred_date_values(base_pred)
    coverage_start = pd.Timestamp(coverage_dates.min())
    coverage_end = pd.Timestamp(coverage_dates.max())

    signal_specs = []
    for item in args.signal_runs.split(","):
        if not item.strip():
            continue
        key, token = item.split(":", 1)
        run_dir = conv._find_run_dir(tracking_dir, token.strip())
        pred = conv._as_score_df(conv._load_pickle(run_dir / "artifacts" / "pred.pkl"))
        transforms = conv._build_transforms(pred)
        signal_specs.append((key.strip(), run_dir.name, transforms))

    grid = _candidate_grid(args.grid_mode)
    slices = _slice_dates(coverage_start, coverage_end)
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    selection_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    apply_excess_parts: List[pd.Series] = []
    apply_reports: List[pd.DataFrame] = []

    apply_plan = [
        ("2024H1_train", "2024H2_apply"),
        ("2024_train", "2025_apply"),
        ("up_to_2025_train", "2026_apply"),
    ]
    if int(args.max_apply_plans) > 0:
        apply_plan = apply_plan[: int(args.max_apply_plans)]
    slice_map = {name: (st, ed) for name, st, ed in slices}

    def write_checkpoint(current: Dict[str, Any] | None = None) -> None:
        error_rows = [row for row in eval_rows if not bool(row.get("ok", False))]
        checkpoint = {
            "timestamp_utc": _now_utc(),
            "task": "strict_conversion_lockstep",
            "status": "running",
            "current": current or {},
            "controls": {
                "grid_mode": args.grid_mode,
                "max_candidates": int(args.max_candidates),
                "max_apply_plans": int(args.max_apply_plans),
                "checkpoint_every": int(args.checkpoint_every),
            },
            "counts": {
                "eval_rows": len(eval_rows),
                "error_rows": len(error_rows),
                "selection_rows": len(selection_rows),
                "apply_success_windows": len(apply_excess_parts),
            },
            "selection_rows": selection_rows,
            "error_rows": error_rows,
            "artifacts": {
                "checkpoint_json": str(checkpoint_json),
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
                "selections_csv": str(selections_csv),
                "splits_csv": str(splits_csv),
                "eval_csv": str(eval_csv),
                "error_csv": str(error_csv),
            },
            "runtime_sec": float(time.perf_counter() - started),
        }
        _write_json(checkpoint_json, checkpoint)

    write_checkpoint({"stage": "initialized"})

    for train_tag, apply_tag in apply_plan:
        train_start, train_end = slice_map[train_tag]
        apply_start, apply_end = slice_map[apply_tag]
        train_valid: List[Dict[str, Any]] = []
        train_candidates_evaluated = 0
        train_candidates_possible = len(signal_specs) * len(grid)
        for sig_key, run_id, transforms in signal_specs:
            if int(args.max_candidates) > 0 and train_candidates_evaluated >= int(args.max_candidates):
                break
            for combo in grid:
                if int(args.max_candidates) > 0 and train_candidates_evaluated >= int(args.max_candidates):
                    break
                trans_df = transforms[combo.transform]
                signal_df = conv._blend_rank_scores(base_pred, trans_df, 1.0 - combo.blend_weight, combo.blend_weight)
                row: Dict[str, Any] = {
                    "phase": "train_selection",
                    "ok": False,
                    "sig_key": sig_key,
                    "run_id": run_id,
                    "train_tag": train_tag,
                    "apply_tag": apply_tag,
                    "start": str(train_start.date()),
                    "end": str(train_end.date()),
                    **asdict(combo),
                }
                train_candidates_evaluated += 1
                candidate_started = time.perf_counter()
                try:
                    ev = _eval_combo_period_with_series(
                        combo=_combo_to_strategy(combo),
                        pred_df=signal_df,
                        base_port_cfg=port_cfg,
                        base_strategy_kwargs=strategy_kwargs,
                        open_cost=float(args.open_cost),
                        close_cost=float(args.close_cost),
                        start_time=train_start,
                        end_time=train_end,
                        exchange_cache=exchange_cache,
                    )
                    score = float(ev["costed_ir"]) + 0.35 * float(ev["costed_annret"]) - 0.25 * abs(float(ev["max_drawdown"]))
                    row.update(
                        {
                            "ok": True,
                            "selection_score": score,
                            "costed_ir": float(ev["costed_ir"]),
                            "costed_annret": float(ev["costed_annret"]),
                            "max_drawdown": float(ev["max_drawdown"]),
                            "turnover": float(ev["turnover"]),
                            "elapsed_sec": float(ev.get("elapsed_sec", time.perf_counter() - candidate_started)),
                            "error_type": "",
                            "error_message": "",
                        }
                    )
                    rec = {
                        "sig_key": sig_key,
                        "run_id": run_id,
                        "train_tag": train_tag,
                        "apply_tag": apply_tag,
                        "selection_score": score,
                        "train_ir": float(ev["costed_ir"]),
                        "train_annret": float(ev["costed_annret"]),
                        "train_mdd": float(ev["max_drawdown"]),
                        "train_turnover": float(ev["turnover"]),
                        **asdict(combo),
                    }
                    train_valid.append(rec)
                except Exception as exc:  # noqa: BLE001
                    row.update({"elapsed_sec": float(time.perf_counter() - candidate_started), **_error_fields(exc)})
                eval_rows.append(row)
                if int(args.checkpoint_every) > 0 and len(eval_rows) % int(args.checkpoint_every) == 0:
                    write_checkpoint({"stage": "train_selection", "train_tag": train_tag, "apply_tag": apply_tag})
        ranked_train = _rank_train_candidates(train_valid)
        if not ranked_train:
            selection_rows.append(
                {
                    "train_tag": train_tag,
                    "apply_tag": apply_tag,
                    "selected": False,
                    "apply_ok": False,
                    "train_candidates_possible": train_candidates_possible,
                    "train_candidates_evaluated": train_candidates_evaluated,
                    "train_candidates_skipped_by_cap": max(0, train_candidates_possible - train_candidates_evaluated),
                    "error_type": "NoSelectableCombo",
                    "error_message": f"no selectable combo for {train_tag}",
                }
            )
            write_checkpoint({"stage": "no_selectable_combo", "train_tag": train_tag, "apply_tag": apply_tag})
            continue

        best: Dict[str, Any] | None = None
        apply_failures = 0
        for candidate in ranked_train:
            sig_key = str(candidate["sig_key"])
            transforms = next(x[2] for x in signal_specs if x[0] == sig_key)
            combo = Combo(
                transform=str(candidate["transform"]),
                blend_weight=float(candidate["blend_weight"]),
                family=str(candidate["family"]),
                topk=int(candidate["topk"]),
                n_drop=int(candidate["n_drop"]),
                hold_topk=int(candidate["hold_topk"]),
            )
            signal_df = conv._blend_rank_scores(base_pred, transforms[combo.transform], 1.0 - combo.blend_weight, combo.blend_weight)
            apply_row: Dict[str, Any] = {
                "phase": "apply_selected",
                "ok": False,
                "sig_key": sig_key,
                "run_id": str(candidate["run_id"]),
                "train_tag": train_tag,
                "apply_tag": apply_tag,
                "train_rank": int(candidate["train_rank"]),
                "selection_score": float(candidate["selection_score"]),
                "start": str(apply_start.date()),
                "end": str(apply_end.date()),
                **asdict(combo),
            }
            candidate_started = time.perf_counter()
            try:
                apply_ev = _eval_combo_period_with_series(
                    combo=_combo_to_strategy(combo),
                    pred_df=signal_df,
                    base_port_cfg=port_cfg,
                    base_strategy_kwargs=strategy_kwargs,
                    open_cost=float(args.open_cost),
                    close_cost=float(args.close_cost),
                    start_time=apply_start,
                    end_time=apply_end,
                    exchange_cache=exchange_cache,
                )
                apply_row.update(
                    {
                        "ok": True,
                        "costed_ir": float(apply_ev["costed_ir"]),
                        "costed_annret": float(apply_ev["costed_annret"]),
                        "max_drawdown": float(apply_ev["max_drawdown"]),
                        "turnover": float(apply_ev["turnover"]),
                        "elapsed_sec": float(apply_ev.get("elapsed_sec", time.perf_counter() - candidate_started)),
                        "error_type": "",
                        "error_category": "",
                        "error_message": "",
                        "error_traceback_tail": "",
                    }
                )
                best = dict(candidate)
                best.update(
                    {
                        "selected": True,
                        "apply_ok": True,
                        "apply_start": str(apply_start.date()),
                        "apply_end": str(apply_end.date()),
                        "apply_ir": float(apply_ev["costed_ir"]),
                        "apply_annret": float(apply_ev["costed_annret"]),
                        "apply_mdd": float(apply_ev["max_drawdown"]),
                        "apply_turnover": float(apply_ev["turnover"]),
                        "apply_failures_before_success": apply_failures,
                        "train_valid_candidates": len(ranked_train),
                        "train_candidates_possible": train_candidates_possible,
                        "train_candidates_evaluated": train_candidates_evaluated,
                        "train_candidates_skipped_by_cap": max(0, train_candidates_possible - train_candidates_evaluated),
                        "error_type": "",
                        "error_category": "",
                        "error_message": "",
                        "error_traceback_tail": "",
                    }
                )
                apply_excess_parts.append(apply_ev["excess_series"])
                apply_reports.append(apply_ev["report_df"])
                eval_rows.append(apply_row)
                break
            except Exception as exc:  # noqa: BLE001
                apply_failures += 1
                apply_row.update({"elapsed_sec": float(time.perf_counter() - candidate_started), **_error_fields(exc)})
                eval_rows.append(apply_row)
                if int(args.checkpoint_every) > 0 and len(eval_rows) % int(args.checkpoint_every) == 0:
                    write_checkpoint({"stage": "apply_retry", "train_tag": train_tag, "apply_tag": apply_tag})

        if best is None:
            first = ranked_train[0]
            best = dict(first)
            best.update(
                {
                    "selected": True,
                    "apply_ok": False,
                    "apply_start": str(apply_start.date()),
                    "apply_end": str(apply_end.date()),
                    "apply_failures_before_success": apply_failures,
                    "train_valid_candidates": len(ranked_train),
                    "train_candidates_possible": train_candidates_possible,
                    "train_candidates_evaluated": train_candidates_evaluated,
                    "train_candidates_skipped_by_cap": max(0, train_candidates_possible - train_candidates_evaluated),
                    "error_type": "AllApplyCandidatesFailed",
                    "error_category": "all_apply_candidates_failed",
                    "error_message": f"all {len(ranked_train)} train-valid candidates failed in {apply_tag}",
                    "error_traceback_tail": "",
                }
            )
        selection_rows.append(best)
        write_checkpoint({"stage": "apply_complete", "train_tag": train_tag, "apply_tag": apply_tag})

    evaluation_complete = len(apply_excess_parts) == len(apply_plan) and len(apply_plan) == 3
    if apply_excess_parts:
        stitched_excess = pd.concat(apply_excess_parts).sort_index()
        stitched_metrics = _metrics_from_excess(stitched_excess)
        split_rows = _year_rows(stitched_excess)
    else:
        stitched_metrics = {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan")}
        split_rows = []
    diagnostic_apply_fallback_used = any(int(row.get("apply_failures_before_success", 0) or 0) > 0 for row in selection_rows)
    hard_pass = bool(
        evaluation_complete
        and not diagnostic_apply_fallback_used
        and stitched_metrics["ir"] > HARD_GATE_IR
        and stitched_metrics["annret"] > HARD_GATE_ANNRET
    )
    verdict = "BREAKTHROUGH" if hard_pass else "NO_GO"

    error_rows = [row for row in eval_rows if not bool(row.get("ok", False))]
    _write_csv(eval_csv, eval_rows)
    _write_csv(error_csv, error_rows)
    _write_csv(selections_csv, selection_rows)
    _write_csv(splits_csv, split_rows)
    summary = {
        "timestamp_utc": _now_utc(),
        "task": "strict_conversion_lockstep",
        "verdict": verdict,
        "hard_gate_pass": hard_pass,
        "evaluation_complete": evaluation_complete,
        "diagnostic_apply_fallback_used": diagnostic_apply_fallback_used,
        "protocol": "Select signal conversion on prior period only, then apply to next period; stitched apply windows are 2024H2, 2025, 2026_ytd.",
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        "controls": {
            "grid_mode": args.grid_mode,
            "max_candidates": int(args.max_candidates),
            "max_apply_plans": int(args.max_apply_plans),
            "checkpoint_every": int(args.checkpoint_every),
        },
        "counts": {
            "eval_rows": len(eval_rows),
            "error_rows": len(error_rows),
            "selection_rows": len(selection_rows),
            "apply_success_windows": len(apply_excess_parts),
            "apply_plan_windows": len(apply_plan),
        },
        "stitched_metrics": stitched_metrics,
        "split_metrics": split_rows,
        "selection_rows": selection_rows,
        "error_rows": error_rows,
        "runtime_sec": float(time.perf_counter() - started),
        "artifacts": {
            "checkpoint_json": str(checkpoint_json),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "selections_csv": str(selections_csv),
            "splits_csv": str(splits_csv),
            "eval_csv": str(eval_csv),
            "error_csv": str(error_csv),
        },
    }
    _write_json(summary_json, summary)
    md = [
        f"# Strict Conversion Lockstep {stamp}",
        "",
        f"Verdict: **{verdict}**",
        f"- Stitched metrics: `{json.dumps(stitched_metrics, ensure_ascii=False)}`",
        f"- Hard gate pass: `{hard_pass}`",
        f"- Evaluation complete: `{evaluation_complete}`",
        f"- Diagnostic apply fallback used: `{diagnostic_apply_fallback_used}`",
        f"- Eval rows/errors: `{len(eval_rows)}/{len(error_rows)}`",
    ]
    summary_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    checkpoint = {
        "timestamp_utc": _now_utc(),
        "task": "strict_conversion_lockstep",
        "status": "complete",
        "verdict": verdict,
        "hard_gate_pass": hard_pass,
        "evaluation_complete": evaluation_complete,
        "diagnostic_apply_fallback_used": diagnostic_apply_fallback_used,
        "counts": summary["counts"],
        "error_rows": error_rows,
        "artifacts": summary["artifacts"],
        "runtime_sec": summary["runtime_sec"],
    }
    _write_json(checkpoint_json, checkpoint)
    print(json.dumps({"verdict": verdict, "hard_gate_pass": hard_pass, "stitched_metrics": stitched_metrics, "summary": str(summary_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
