#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import faulthandler
import json
import pickle
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import signal_portfolio_conversion_scan as conv
from quant_master.contrib.evaluate import risk_analysis


TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
DEFAULT_BASE_RUN = "7406e47063e9479cb34d300b9ed03bad"

RUN_ALIAS = {
    "7406": DEFAULT_BASE_RUN,
    "7406e470": DEFAULT_BASE_RUN,
    "773": "773bd6d8413b4bb0b388a63a6b5b6a86",
    "773bd6d": "773bd6d8413b4bb0b388a63a6b5b6a86",
    "bc641": "bc641cef654441d2bf0c7008e6c90458",
    "1a085": "1a085ff9b5a34f408a44ad74055fc5da",
}


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    signal_kind: str
    run_ids: Tuple[str, ...]
    weights: Tuple[float, ...]
    topk: int
    n_drop: int
    note: str = ""


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


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _resolve_run(token: str) -> str:
    t = str(token).strip()
    return RUN_ALIAS.get(t, t)


def _daily_excess(report: pd.DataFrame) -> pd.Series:
    return (report["return"].astype(float) - report["bench"].astype(float) - report["cost"].astype(float)).rename("excess")


def _metrics_from_excess(excess: pd.Series, turnover: pd.Series | None = None) -> Dict[str, float]:
    s = excess.dropna().sort_index()
    if s.empty:
        return {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
    risk_df = risk_analysis(s, freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(turnover.reindex(s.index).mean()) if turnover is not None else float("nan"),
    }


def _slice_report(report: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    idx = pd.to_datetime(report.index)
    return report.loc[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))].copy()


def _rank_pct(score: pd.Series) -> pd.Series:
    if isinstance(score.index, pd.MultiIndex):
        return score.groupby(level=0).rank(method="average", pct=True)
    return score.groupby(score.index).rank(method="average", pct=True)


def _load_score_df(tracking_dir: Path, run_id: str, start: str, end: str) -> pd.DataFrame:
    run_dir = conv._find_run_dir(tracking_dir, _resolve_run(run_id))
    pred = conv._as_score_df(_load_pickle(run_dir / "artifacts" / "pred.pkl"))
    return conv._slice_pred(pred, pd.Timestamp(start), pd.Timestamp(end))


def _rank_ensemble_signal(
    tracking_dir: Path,
    run_ids: Sequence[str],
    weights: Sequence[float],
    start: str,
    end: str,
) -> pd.DataFrame:
    cols = []
    for run_id in run_ids:
        pred = _load_score_df(tracking_dir, run_id, start, end)
        s = _rank_pct(pred["score"].astype(float))
        s.name = _resolve_run(run_id)
        cols.append(s)
    panel = pd.concat(cols, axis=1)
    w = pd.Series(weights, index=panel.columns, dtype=float)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    blend = panel.mul(w, axis=1).fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return blend.to_frame("score")


def _signal_for_action(tracking_dir: Path, action: ActionSpec, base_pred: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if action.signal_kind == "base":
        return conv._slice_pred(base_pred, pd.Timestamp(start), pd.Timestamp(end))
    if action.signal_kind == "rank_ensemble":
        return _rank_ensemble_signal(tracking_dir, action.run_ids, action.weights, start, end)
    raise ValueError(f"unsupported signal_kind={action.signal_kind}")


def _action_combo(action: ActionSpec) -> Dict[str, Any]:
    return {
        "family": "topk_dropout",
        "rebalance_mode": "daily",
        "rebalance_interval": 1,
        "topk": int(action.topk),
        "n_drop": int(action.n_drop),
        "hold_topk": int(action.topk),
    }


def _eval_action_report(
    *,
    action: ActionSpec,
    pred_df: pd.DataFrame,
    base_port_cfg: Dict[str, Any],
    base_strategy_kwargs: Dict[str, Any],
    open_cost: float,
    close_cost: float,
    start: str,
    end: str,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_port_cfg)
    backtest_cfg = cfg["backtest"]
    backtest_cfg["start_time"] = str(pd.Timestamp(start).date())
    backtest_cfg["end_time"] = str(pd.Timestamp(end).date())
    executor_cfg = cfg.get(
        "executor",
        {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
    )
    pred_slice = conv._slice_pred(pred_df, pd.Timestamp(start), pd.Timestamp(end))
    if pred_slice.empty:
        raise ValueError(f"empty signal slice for {action.action_id}: {start}..{end}")

    exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
    deal_price = str(exchange_kwargs.get("deal_price", "close"))
    limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
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
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    t0 = time.perf_counter()
    strategy = conv._build_strategy_object(
        combo=_action_combo(action),
        pred_df=pred_slice,
        base_strategy_kwargs=base_strategy_kwargs,
    )
    pm, _ = conv.run_backtest(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        strategy=strategy,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark", "SH000300"),
        account=backtest_cfg.get("account", 100000000),
        exchange_kwargs=exchange_kwargs,
        pos_type=backtest_cfg.get("pos_type", "Position"),
    )
    report = conv._get_report_for_day_freq(pm)
    excess = _daily_excess(report)
    metrics = _metrics_from_excess(excess, report["turnover"].astype(float))
    metrics["elapsed_sec"] = float(time.perf_counter() - t0)
    return {"report": report, "excess": excess, "metrics": metrics}


def _load_action_report_artifact(tracking_dir: Path, action: ActionSpec, start: str, end: str) -> Dict[str, Any]:
    if action.signal_kind != "base":
        raise ValueError("artifact mode is only exact for base actions; use --action-source backtest for ensembles")
    run_dir = conv._find_run_dir(tracking_dir, _resolve_run(action.run_ids[0]))
    report_path = run_dir / "artifacts" / "portfolio_analysis" / "report_normal_1day.pkl"
    if not report_path.exists():
        raise FileNotFoundError(f"report artifact not found for {action.action_id}: {report_path}")
    report = _slice_report(_load_pickle(report_path), start, end)
    if report.empty:
        raise ValueError(f"empty report artifact for {action.action_id}: {start}..{end}")
    excess = _daily_excess(report)
    return {
        "report": report,
        "excess": excess,
        "metrics": _metrics_from_excess(excess, report["turnover"].astype(float)),
        "artifact_report_path": str(report_path),
    }


def _build_actions(include_base50: bool) -> List[ActionSpec]:
    actions = [
        ActionSpec("base45", "base", (DEFAULT_BASE_RUN,), (1.0,), 45, 4, "stable SOTA anchor"),
        ActionSpec("base40", "base", (DEFAULT_BASE_RUN,), (1.0,), 40, 2, "lower-turnover/high-AnnRet anchor"),
        ActionSpec(
            "rank50",
            "rank_ensemble",
            (DEFAULT_BASE_RUN, RUN_ALIAS["773"], RUN_ALIAS["bc641"]),
            (0.60, 0.20, 0.20),
            50,
            5,
            "predeclared rank ensemble",
        ),
        ActionSpec(
            "gru45",
            "rank_ensemble",
            (DEFAULT_BASE_RUN, RUN_ALIAS["1a085"], RUN_ALIAS["773"]),
            (0.40, 0.20, 0.40),
            45,
            4,
            "predeclared GRU/rank blend",
        ),
    ]
    if include_base50:
        actions.append(ActionSpec("base50", "base", (DEFAULT_BASE_RUN,), (1.0,), 50, 5, "optional fixed topk sensitivity"))
    return actions


def _scoring_functions(metrics: Dict[str, float], split_rows: Sequence[Dict[str, Any]] | None = None) -> Dict[str, float]:
    ir = float(metrics.get("ir", float("nan")))
    ann = float(metrics.get("annret", float("nan")))
    mdd = abs(float(metrics.get("max_drawdown", float("nan"))))
    turnover = float(metrics.get("turnover", float("nan")))
    vals = [ir, ann, mdd, turnover]
    if not all(np.isfinite(x) for x in vals):
        return {"ir_focus": -1e9, "ann_mdd": -1e9, "stability": -1e9}

    rows = list(split_rows or [])
    split_ir = [float(r["ir"]) for r in rows if np.isfinite(float(r.get("ir", float("nan"))))]
    split_ann = [float(r["annret"]) for r in rows if np.isfinite(float(r.get("annret", float("nan"))))]
    ir_std = float(np.std(split_ir, ddof=0)) if len(split_ir) >= 2 else 0.0
    ann_std = float(np.std(split_ann, ddof=0)) if len(split_ann) >= 2 else 0.0
    min_ann = min(split_ann) if split_ann else ann
    neg_ann_penalty = abs(min(0.0, min_ann))
    return {
        "ir_focus": float(ir + 0.15 * ann - 0.12 * mdd - 0.12 * turnover),
        "ann_mdd": float(ann + 0.24 * ir - 0.85 * mdd - 0.18 * turnover),
        "stability": float(ir + 0.40 * ann - 0.30 * mdd - 0.16 * turnover - 0.45 * ir_std - 0.20 * ann_std - 0.60 * neg_ann_penalty),
    }


def _subsplit_rows(report: pd.DataFrame, start: str, end: str, freq: str = "Q") -> List[Dict[str, Any]]:
    rep = _slice_report(report, start, end)
    if rep.empty:
        return []
    rows: List[Dict[str, Any]] = []
    periods = pd.PeriodIndex(pd.to_datetime(rep.index), freq=freq)
    for per in sorted(periods.unique()):
        part = rep.loc[periods == per]
        if len(part) < 5:
            continue
        rows.append(
            {
                "split": str(per),
                "start": str(pd.Timestamp(part.index.min()).date()),
                "end": str(pd.Timestamp(part.index.max()).date()),
                **_metrics_from_excess(_daily_excess(part), part["turnover"].astype(float)),
            }
        )
    return rows


def _evaluate_action_on_window(
    action_id: str,
    report: pd.DataFrame,
    start: str,
    end: str,
) -> Dict[str, Any]:
    rep = _slice_report(report, start, end)
    metrics = _metrics_from_excess(_daily_excess(rep), rep["turnover"].astype(float)) if not rep.empty else {}
    split_rows = _subsplit_rows(rep, start, end, freq="Q") if not rep.empty else []
    scores = _scoring_functions(metrics, split_rows)
    return {
        "action_id": action_id,
        "start": start,
        "end": end,
        **metrics,
        **{f"score_{k}": v for k, v in scores.items()},
        "subsplits_json": json.dumps(split_rows, ensure_ascii=False),
    }


def _select_action(
    *,
    action_reports: Dict[str, pd.DataFrame],
    candidate_actions: Sequence[str],
    train_start: str,
    train_end: str,
    score_name: str,
    min_train_ir: float,
    min_train_annret: float,
    fallback_action: str,
) -> Tuple[str, List[Dict[str, Any]], str]:
    rows = []
    for aid in candidate_actions:
        row = _evaluate_action_on_window(aid, action_reports[aid], train_start, train_end)
        ok = (
            np.isfinite(float(row.get("ir", float("nan"))))
            and np.isfinite(float(row.get("annret", float("nan"))))
            and float(row["ir"]) >= float(min_train_ir)
            and float(row["annret"]) >= float(min_train_annret)
        )
        row["train_gate_ok"] = bool(ok)
        row["train_gate_reason"] = "" if ok else f"requires ir>={min_train_ir}, annret>={min_train_annret}"
        rows.append(row)

    passed = [r for r in rows if bool(r["train_gate_ok"])]
    pool = passed if passed else rows
    if not pool:
        return fallback_action, rows, "empty_pool"
    score_col = f"score_{score_name}"
    chosen = sorted(
        pool,
        key=lambda r: (
            float(r.get(score_col, -1e9)),
            float(r.get("ir", -1e9)),
            float(r.get("annret", -1e9)),
            str(r.get("action_id", "")),
        ),
        reverse=True,
    )[0]
    reason = f"max_{score_name}_on_past_train"
    if not passed:
        reason += "_without_min_gate"
    return str(chosen["action_id"]), rows, reason


def _stitch_periods(action_reports: Dict[str, pd.DataFrame], plan_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    parts = []
    for row in plan_rows:
        aid = str(row["selected_action"])
        part = _slice_report(action_reports[aid], row["apply_start"], row["apply_end"])
        if part.empty:
            continue
        part = part.copy()
        part["selected_action"] = aid
        part["selection_reason"] = str(row["selection_reason"])
        part["apply_tag"] = str(row["apply_tag"])
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


def _year_rows(stitched: pd.DataFrame, start: str, end: str) -> List[Dict[str, Any]]:
    rows = []
    for year in (2024, 2025, 2026):
        ys = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(start))
        ye = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(end))
        if ys > ye:
            continue
        rep = _slice_report(stitched, ys, ye)
        if rep.empty:
            continue
        tag = f"{year}_ytd" if ye < pd.Timestamp(f"{year}-12-31") else str(year)
        rows.append(
            {
                "split": tag,
                "start": str(ys.date()),
                "end": str(ye.date()),
                **_metrics_from_excess(_daily_excess(rep), rep["turnover"].astype(float)),
                "selected_counts": json.dumps({str(k): int(v) for k, v in rep["selected_action"].value_counts().to_dict().items()}),
            }
        )
    return rows


def _checkpoint(path: Path, *, stage: str, started: float, extra: Dict[str, Any]) -> None:
    _write_json(
        path,
        {
            "timestamp_utc": _now_utc(),
            "stage": stage,
            "runtime_sec": float(time.perf_counter() - started),
            "extra": extra,
        },
    )


def _error_fields(exc: Exception) -> Dict[str, str]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)).strip()
    return {"error_type": type(exc).__name__, "error_message": str(exc), "traceback_tail": tb}


def _primary_plan(
    *,
    action_reports: Dict[str, pd.DataFrame],
    candidate_actions: Sequence[str],
    fixed_2024_action: str,
    primary_score: str,
    train_min_ir: float,
    train_min_annret: float,
    start_date: str,
    end_date: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    plan: List[Dict[str, Any]] = []
    selection_audit: List[Dict[str, Any]] = []
    if pd.Timestamp(start_date) <= pd.Timestamp("2024-12-31"):
        plan.append(
            {
                "apply_tag": "2024_fixed_predeclared",
                "train_start": "",
                "train_end": "",
                "apply_start": str(max(pd.Timestamp(start_date), pd.Timestamp("2024-01-01")).date()),
                "apply_end": str(min(pd.Timestamp(end_date), pd.Timestamp("2024-12-31")).date()),
                "selected_action": fixed_2024_action,
                "selection_reason": "fixed_2024_predeclared_no_test_selection",
                "score_name": primary_score,
            }
        )

    if pd.Timestamp(end_date) >= pd.Timestamp("2025-01-01"):
        selected, rows, reason = _select_action(
            action_reports=action_reports,
            candidate_actions=candidate_actions,
            train_start="2024-01-01",
            train_end="2024-12-31",
            score_name=primary_score,
            min_train_ir=train_min_ir,
            min_train_annret=train_min_annret,
            fallback_action=fixed_2024_action,
        )
        for r in rows:
            r.update({"apply_tag": "2025", "score_name": primary_score})
            selection_audit.append(r)
        plan.append(
            {
                "apply_tag": "2025",
                "train_start": "2024-01-01",
                "train_end": "2024-12-31",
                "apply_start": "2025-01-01",
                "apply_end": str(min(pd.Timestamp(end_date), pd.Timestamp("2025-12-31")).date()),
                "selected_action": selected,
                "selection_reason": reason,
                "score_name": primary_score,
            }
        )

    if pd.Timestamp(end_date) >= pd.Timestamp("2026-01-01"):
        selected, rows, reason = _select_action(
            action_reports=action_reports,
            candidate_actions=candidate_actions,
            train_start="2024-01-01",
            train_end="2025-12-31",
            score_name=primary_score,
            min_train_ir=train_min_ir,
            min_train_annret=train_min_annret,
            fallback_action=fixed_2024_action,
        )
        for r in rows:
            r.update({"apply_tag": "2026_ytd", "score_name": primary_score})
            selection_audit.append(r)
        plan.append(
            {
                "apply_tag": "2026_ytd",
                "train_start": "2024-01-01",
                "train_end": "2025-12-31",
                "apply_start": "2026-01-01",
                "apply_end": str(min(pd.Timestamp(end_date), pd.Timestamp("2026-04-30")).date()),
                "selected_action": selected,
                "selection_reason": reason,
                "score_name": primary_score,
            }
        )
    return plan, selection_audit


def _profile_audit(
    *,
    action_reports: Dict[str, pd.DataFrame],
    candidate_actions: Sequence[str],
    fixed_2024_actions: Sequence[str],
    score_names: Sequence[str],
    train_min_ir: float,
    train_min_annret: float,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    rows = []
    for fixed in fixed_2024_actions:
        for score in score_names:
            plan, _ = _primary_plan(
                action_reports=action_reports,
                candidate_actions=candidate_actions,
                fixed_2024_action=fixed,
                primary_score=score,
                train_min_ir=train_min_ir,
                train_min_annret=train_min_annret,
                start_date=start_date,
                end_date=end_date,
            )
            stitched = _stitch_periods(action_reports, plan)
            metrics = (
                _metrics_from_excess(_daily_excess(stitched), stitched["turnover"].astype(float))
                if not stitched.empty
                else {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
            )
            rows.append(
                {
                    "fixed_2024_action": fixed,
                    "score_name": score,
                    "hard_gate_pass": bool(metrics["ir"] > HARD_GATE_IR and metrics["annret"] > HARD_GATE_ANNRET),
                    **metrics,
                    "plan_json": json.dumps(plan, ensure_ascii=False),
                    "selection_counts": json.dumps(
                        {str(k): int(v) for k, v in stitched["selected_action"].value_counts().to_dict().items()}
                    )
                    if not stitched.empty
                    else "{}",
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict 2024-train / 2025-2026 past-only gate over predeclared actions.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=DEFAULT_BASE_RUN)
    p.add_argument("--start-date", default=TEST_START)
    p.add_argument("--end-date", default=TEST_END)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--fixed-2024-action", default="base40", choices=["base45", "base40", "rank50", "gru45", "base50"])
    p.add_argument("--primary-score", default="ann_mdd", choices=["ir_focus", "ann_mdd", "stability"])
    p.add_argument("--train-min-ir", type=float, default=0.0)
    p.add_argument("--train-min-annret", type=float, default=0.0)
    p.add_argument("--include-base50", action="store_true")
    p.add_argument(
        "--action-source",
        choices=["backtest", "artifacts"],
        default="backtest",
        help="artifacts is exact only for base actions and cannot complete if rank ensembles are required.",
    )
    p.add_argument("--output-prefix", default="robust_2024_train_2025_2026_gate")
    return p


def main() -> int:
    faulthandler.enable()
    args = build_parser().parse_args()
    started = time.perf_counter()
    trans_dir = Path(__file__).resolve().parent
    stamp = _stamp()
    summary_json = trans_dir / f"{args.output_prefix}_summary_{stamp}.json"
    checkpoint_json = trans_dir / f"{args.output_prefix}_checkpoint_{stamp}.json"
    actions_csv = trans_dir / f"{args.output_prefix}_actions_{stamp}.csv"
    selection_csv = trans_dir / f"{args.output_prefix}_selection_audit_{stamp}.csv"
    plan_csv = trans_dir / f"{args.output_prefix}_plan_{stamp}.csv"
    splits_csv = trans_dir / f"{args.output_prefix}_splits_{stamp}.csv"
    profiles_csv = trans_dir / f"{args.output_prefix}_profiles_{stamp}.csv"
    errors_csv = trans_dir / f"{args.output_prefix}_errors_{stamp}.csv"

    tracking_dir = conv._parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, _resolve_run(args.base_run_id))
    cfg = conv._load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(cfg)
    port_cfg = conv._extract_port_config(cfg)
    strategy_kwargs = dict(port_cfg.get("strategy", {}).get("kwargs", {}))
    strategy_kwargs.pop("signal", None)

    start_date = str(pd.Timestamp(args.start_date).date())
    end_date = str(pd.Timestamp(args.end_date).date())
    base_pred = conv._as_score_df(_load_pickle(base_dir / "artifacts" / "pred.pkl"))
    base_pred = conv._slice_pred(base_pred, pd.Timestamp(start_date), pd.Timestamp(end_date))

    actions = _build_actions(include_base50=bool(args.include_base50))
    action_ids = [a.action_id for a in actions]
    if args.fixed_2024_action not in action_ids:
        raise ValueError(f"fixed action {args.fixed_2024_action} not in action universe {action_ids}; add --include-base50 if needed")

    _checkpoint(
        checkpoint_json,
        stage="initialized",
        started=started,
        extra={
            "action_ids": action_ids,
            "start_date": start_date,
            "end_date": end_date,
            "action_source": args.action_source,
            "primary_score": args.primary_score,
            "fixed_2024_action": args.fixed_2024_action,
        },
    )

    action_reports: Dict[str, pd.DataFrame] = {}
    action_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}

    for action in actions:
        try:
            _checkpoint(
                checkpoint_json,
                stage="action_start",
                started=started,
                extra={"action_id": action.action_id, "completed_actions": list(action_reports.keys())},
            )
            if args.action_source == "artifacts":
                ev = _load_action_report_artifact(tracking_dir, action, start_date, end_date)
            else:
                sig = _signal_for_action(tracking_dir, action, base_pred, start_date, end_date)
                ev = _eval_action_report(
                    action=action,
                    pred_df=sig,
                    base_port_cfg=port_cfg,
                    base_strategy_kwargs=strategy_kwargs,
                    open_cost=float(args.open_cost),
                    close_cost=float(args.close_cost),
                    start=start_date,
                    end=end_date,
                    exchange_cache=exchange_cache,
                )
            action_reports[action.action_id] = ev["report"]
            year_json = json.dumps(_year_rows(ev["report"].assign(selected_action=action.action_id), start_date, end_date), ensure_ascii=False)
            action_rows.append(
                {
                    **asdict(action),
                    **ev["metrics"],
                    "ok": True,
                    "action_source": args.action_source,
                    "year_metrics_json": year_json,
                    "artifact_report_path": str(ev.get("artifact_report_path", "")),
                }
            )
            _checkpoint(
                checkpoint_json,
                stage="action_complete",
                started=started,
                extra={"action_id": action.action_id, "metrics": ev["metrics"], "exchange_cache_keys": len(exchange_cache)},
            )
        except Exception as exc:  # noqa: BLE001
            fields = {"action_id": action.action_id, **_error_fields(exc)}
            error_rows.append(fields)
            action_rows.append({**asdict(action), "ok": False, "action_source": args.action_source, **fields})
            _checkpoint(checkpoint_json, stage="action_error", started=started, extra=fields)

    required = {"base45", "base40", "rank50", "gru45"}
    missing = sorted(required.difference(action_reports))
    if missing:
        _write_csv(actions_csv, action_rows)
        _write_csv(errors_csv, error_rows)
        raise RuntimeError(f"missing required action reports: {missing}")

    candidate_actions = [aid for aid in action_ids if aid in action_reports]
    primary_plan, selection_audit = _primary_plan(
        action_reports=action_reports,
        candidate_actions=candidate_actions,
        fixed_2024_action=str(args.fixed_2024_action),
        primary_score=str(args.primary_score),
        train_min_ir=float(args.train_min_ir),
        train_min_annret=float(args.train_min_annret),
        start_date=start_date,
        end_date=end_date,
    )
    stitched = _stitch_periods(action_reports, primary_plan)
    stitched_metrics = (
        _metrics_from_excess(_daily_excess(stitched), stitched["turnover"].astype(float))
        if not stitched.empty
        else {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
    )
    split_rows = _year_rows(stitched, start_date, end_date) if not stitched.empty else []

    profile_rows = _profile_audit(
        action_reports=action_reports,
        candidate_actions=candidate_actions,
        fixed_2024_actions=[a for a in ["base45", "base40", "rank50", "gru45", "base50"] if a in candidate_actions],
        score_names=["ir_focus", "ann_mdd", "stability"],
        train_min_ir=float(args.train_min_ir),
        train_min_annret=float(args.train_min_annret),
        start_date=start_date,
        end_date=end_date,
    )

    evaluation_complete = bool(
        start_date == TEST_START
        and end_date == TEST_END
        and float(args.open_cost) == 0.0005
        and float(args.close_cost) == 0.0015
        and args.action_source == "backtest"
    )
    hard_gate_pass = bool(
        evaluation_complete
        and stitched_metrics["ir"] > HARD_GATE_IR
        and stitched_metrics["annret"] > HARD_GATE_ANNRET
    )
    verdict = "BREAKTHROUGH" if hard_gate_pass else "NO_GO"

    _write_csv(actions_csv, action_rows)
    _write_csv(selection_csv, selection_audit)
    _write_csv(plan_csv, primary_plan)
    _write_csv(splits_csv, split_rows)
    _write_csv(profiles_csv, profile_rows)
    _write_csv(errors_csv, error_rows)

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "robust_2024_train_2025_2026_gate",
        "verdict": verdict,
        "hard_gate_pass": hard_gate_pass,
        "evaluation_complete": evaluation_complete,
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "scope": {
            "start_date": start_date,
            "end_date": end_date,
            "action_source": args.action_source,
            "fixed_2024_action": args.fixed_2024_action,
            "primary_score": args.primary_score,
            "train_min_ir": float(args.train_min_ir),
            "train_min_annret": float(args.train_min_annret),
            "candidate_actions": candidate_actions,
        },
        "strict_leakage_controls": [
            "2024 action is fixed by command-line protocol before evaluation.",
            "2025 action is selected using 2024-01-01..2024-12-31 action metrics only.",
            "2026 action is selected using 2024-01-01..2025-12-31 action metrics only.",
            "Action universe and topk/n_drop/run weights are fixed constants in this script.",
            "Scoring profiles are transparent; hard_gate_pass is based only on the predeclared primary_score, not the profile audit.",
            "Full-period pass is disabled unless dates/costs/action_source match the hard gate convention.",
        ],
        "primary_plan": primary_plan,
        "stitched_metrics": stitched_metrics,
        "split_metrics": split_rows,
        "action_metrics": action_rows,
        "selection_audit_rows": selection_audit,
        "profile_audit_rows": profile_rows,
        "selection_counts": {str(k): int(v) for k, v in stitched["selected_action"].value_counts().to_dict().items()}
        if not stitched.empty
        else {},
        "errors": error_rows,
        "artifacts": {
            "summary_json": str(summary_json),
            "checkpoint_json": str(checkpoint_json),
            "actions_csv": str(actions_csv),
            "selection_audit_csv": str(selection_csv),
            "plan_csv": str(plan_csv),
            "splits_csv": str(splits_csv),
            "profiles_csv": str(profiles_csv),
            "errors_csv": str(errors_csv),
        },
        "runtime_sec": float(time.perf_counter() - started),
    }
    _write_json(summary_json, summary)
    _checkpoint(
        checkpoint_json,
        stage="complete",
        started=started,
        extra={
            "summary_json": str(summary_json),
            "verdict": verdict,
            "hard_gate_pass": hard_gate_pass,
            "stitched_metrics": stitched_metrics,
        },
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "hard_gate_pass": hard_gate_pass,
                "evaluation_complete": evaluation_complete,
                "primary_plan": primary_plan,
                "stitched_metrics": stitched_metrics,
                "summary": str(summary_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
