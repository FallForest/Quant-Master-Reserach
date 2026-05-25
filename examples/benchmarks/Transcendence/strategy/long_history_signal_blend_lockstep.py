#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
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


TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
HARD_GATE_IR = 2.90
HARD_GATE_ANNRET = 0.27
DEFAULT_BASE_RUN = "7406e47063e9479cb34d300b9ed03bad"
DEFAULT_LONG_HISTORY_PRED = "long_history_retrain_candidate_pred_20260522T134241Z.pkl"
DEFAULT_LONG_HISTORY_SUMMARY = "long_history_retrain_summary_20260522T134241Z.json"


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    long_history_weight: float
    base_weight: float
    topk: int
    n_drop: int
    note: str


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


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _date_bounds(df: pd.DataFrame) -> Tuple[str, str, int]:
    idx = pd.to_datetime(df.index.get_level_values(0) if isinstance(df.index, pd.MultiIndex) else df.index)
    dates = pd.DatetimeIndex(idx.normalize().unique()).sort_values()
    return str(dates.min().date()), str(dates.max().date()), int(len(dates))


def _combo(action: ActionSpec) -> Dict[str, Any]:
    return {
        "family": "topk_dropout",
        "rebalance_mode": "daily",
        "rebalance_interval": 1,
        "topk": int(action.topk),
        "n_drop": int(action.n_drop),
        "hold_topk": int(action.topk),
    }


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
        combo=_combo(action),
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


def _build_actions(weights: Sequence[float], topk: int, n_drop: int) -> List[ActionSpec]:
    actions: List[ActionSpec] = []
    for w in weights:
        weight = float(w)
        if weight < 0.0 or weight > 1.0:
            raise ValueError(f"blend weight must be in [0,1], got {weight}")
        label = "base40" if weight == 0.0 else f"lh{int(round(weight * 1000)):03d}"
        actions.append(
            ActionSpec(
                action_id=label,
                long_history_weight=weight,
                base_weight=1.0 - weight,
                topk=int(topk),
                n_drop=int(n_drop),
                note=(
                    "base40 control"
                    if weight == 0.0
                    else f"rank-score blend: base40_weight={1.0 - weight:.3f}, long_history_weight={weight:.3f}"
                ),
            )
        )
    if not any(a.long_history_weight == 0.0 for a in actions):
        raise ValueError("weights must include 0.0 base40 control")
    return actions


def _build_signals(base_pred: pd.DataFrame, long_history_pred: pd.DataFrame, actions: Sequence[ActionSpec]) -> Dict[str, pd.DataFrame]:
    signals: Dict[str, pd.DataFrame] = {}
    for action in actions:
        if action.long_history_weight == 0.0:
            signals[action.action_id] = base_pred.sort_index()
        else:
            signals[action.action_id] = conv._blend_rank_scores(
                base_pred,
                long_history_pred,
                action.base_weight,
                action.long_history_weight,
            ).sort_index()
    return signals


def _evaluate_window(action_id: str, report: pd.DataFrame, start: str, end: str) -> Dict[str, Any]:
    rep = _slice_report(report, start, end)
    metrics = _metrics_from_excess(_daily_excess(rep), rep["turnover"].astype(float)) if not rep.empty else {}
    return {
        "action_id": action_id,
        "start": start,
        "end": end,
        "days": int(len(rep)),
        **metrics,
    }


def _select_action(
    *,
    action_reports: Dict[str, pd.DataFrame],
    candidate_actions: Sequence[str],
    train_start: str,
    train_end: str,
    min_train_days: int,
    min_train_ir: float,
    min_train_annret: float,
    fallback_action: str,
) -> Tuple[str, List[Dict[str, Any]], str]:
    rows = []
    for aid in candidate_actions:
        row = _evaluate_window(aid, action_reports[aid], train_start, train_end)
        finite = np.isfinite(float(row.get("ir", float("nan")))) and np.isfinite(float(row.get("annret", float("nan"))))
        gate_ok = (
            finite
            and int(row.get("days", 0)) >= int(min_train_days)
            and float(row["ir"]) >= float(min_train_ir)
            and float(row["annret"]) >= float(min_train_annret)
        )
        row["train_gate_ok"] = bool(gate_ok)
        row["train_gate_reason"] = "" if gate_ok else f"requires days>={min_train_days}, ir>={min_train_ir}, annret>={min_train_annret}"
        rows.append(row)

    pool = [r for r in rows if bool(r["train_gate_ok"])]
    if not pool:
        return fallback_action, rows, "fallback_no_prior_candidate_met_train_gate"
    chosen = sorted(
        pool,
        key=lambda r: (
            float(r.get("ir", -1e9)),
            float(r.get("annret", -1e9)),
            -abs(float(r.get("max_drawdown", 1e9))),
            str(r.get("action_id", "")),
        ),
        reverse=True,
    )[0]
    return str(chosen["action_id"]), rows, "max_prior_window_ir_tiebreak_annret"


def _selection_plan(
    *,
    action_reports: Dict[str, pd.DataFrame],
    candidate_actions: Sequence[str],
    start_date: str,
    end_date: str,
    fixed_first_action: str,
    min_train_days: int,
    min_train_ir: float,
    min_train_annret: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    plan: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    windows = [
        ("2024H1_fixed", "", "", "2024-01-01", "2024-06-30", fixed_first_action, "fixed_predeclared_2024H1_no_prior_intra2024_window"),
        ("2024H2", "2024-01-01", "2024-06-30", "2024-07-01", "2024-12-31", None, ""),
        ("2025", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31", None, ""),
        ("2026_ytd", "2024-01-01", "2025-12-31", "2026-01-01", "2026-04-30", None, ""),
    ]
    for tag, train_start, train_end, apply_start, apply_end, fixed, fixed_reason in windows:
        app_start = max(start_ts, pd.Timestamp(apply_start))
        app_end = min(end_ts, pd.Timestamp(apply_end))
        if app_start > app_end:
            continue
        if fixed is not None:
            selected = fixed
            reason = fixed_reason
        else:
            selected, rows, reason = _select_action(
                action_reports=action_reports,
                candidate_actions=candidate_actions,
                train_start=train_start,
                train_end=train_end,
                min_train_days=min_train_days,
                min_train_ir=min_train_ir,
                min_train_annret=min_train_annret,
                fallback_action=fixed_first_action,
            )
            for row in rows:
                row.update({"apply_tag": tag, "selection_rule": "prior_completed_window_only"})
                audit.append(row)
        plan.append(
            {
                "apply_tag": tag,
                "train_start": train_start,
                "train_end": train_end,
                "apply_start": str(app_start.date()),
                "apply_end": str(app_end.date()),
                "selected_action": selected,
                "selection_reason": reason,
            }
        )
    return plan, audit


def _stitch(action_reports: Dict[str, pd.DataFrame], plan: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    parts = []
    for row in plan:
        aid = str(row["selected_action"])
        part = _slice_report(action_reports[aid], row["apply_start"], row["apply_end"])
        if part.empty:
            continue
        part = part.copy()
        part["selected_action"] = aid
        part["apply_tag"] = str(row["apply_tag"])
        part["selection_reason"] = str(row["selection_reason"])
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


def _split_rows(stitched: pd.DataFrame, start: str, end: str) -> List[Dict[str, Any]]:
    rows = []
    specs = [
        ("full", start, end),
        ("2024H1", "2024-01-01", "2024-06-30"),
        ("2024H2", "2024-07-01", "2024-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", "2026-04-30"),
    ]
    for split, s0, e0 in specs:
        s = max(pd.Timestamp(start), pd.Timestamp(s0))
        e = min(pd.Timestamp(end), pd.Timestamp(e0))
        if s > e:
            continue
        rep = _slice_report(stitched, s, e)
        if rep.empty:
            continue
        rows.append(
            {
                "split": split,
                "start": str(s.date()),
                "end": str(e.date()),
                "days": int(len(rep)),
                **_metrics_from_excess(_daily_excess(rep), rep["turnover"].astype(float)),
                "selected_counts": json.dumps({str(k): int(v) for k, v in rep["selected_action"].value_counts().to_dict().items()}),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict lockstep blend of base40 and long-history signal.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=DEFAULT_BASE_RUN)
    p.add_argument("--long-history-pred", default=DEFAULT_LONG_HISTORY_PRED)
    p.add_argument("--long-history-summary", default=DEFAULT_LONG_HISTORY_SUMMARY)
    p.add_argument("--start-date", default=TEST_START)
    p.add_argument("--end-date", default=TEST_END)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--weights", default="0,0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.25")
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--n-drop", type=int, default=2)
    p.add_argument("--fixed-first-action", default="base40")
    p.add_argument("--min-train-days", type=int, default=20)
    p.add_argument("--min-train-ir", type=float, default=-99.0)
    p.add_argument("--min-train-annret", type=float, default=-99.0)
    p.add_argument("--max-actions", type=int, default=0, help="Bounded debug run: evaluate only the first N actions; 0 means all.")
    p.add_argument("--output-prefix", default="long_history_signal_blend_lockstep")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    out_dir = Path(__file__).resolve().parent
    summary_json = out_dir / f"{args.output_prefix}_summary_{stamp}.json"
    actions_csv = out_dir / f"{args.output_prefix}_actions_{stamp}.csv"
    selection_csv = out_dir / f"{args.output_prefix}_selection_{stamp}.csv"
    plan_csv = out_dir / f"{args.output_prefix}_plan_{stamp}.csv"
    splits_csv = out_dir / f"{args.output_prefix}_splits_{stamp}.csv"

    start_date = str(pd.Timestamp(args.start_date).date())
    end_date = str(pd.Timestamp(args.end_date).date())
    tracking_dir = conv._parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, args.base_run_id)
    cfg = conv._load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(cfg)
    port_cfg = conv._extract_port_config(cfg)
    strategy_kwargs = dict(port_cfg.get("strategy", {}).get("kwargs", {}))
    strategy_kwargs.pop("signal", None)

    base_raw = conv._as_score_df(_load_pickle(base_dir / "artifacts" / "pred.pkl"))
    long_history_path = (out_dir / str(args.long_history_pred)).resolve()
    long_history_raw = conv._as_score_df(_load_pickle(long_history_path))
    base_pred = conv._slice_pred(base_raw, pd.Timestamp(start_date), pd.Timestamp(end_date)).sort_index()
    long_history_pred = conv._slice_pred(long_history_raw, pd.Timestamp(start_date), pd.Timestamp(end_date)).sort_index()

    weights = [float(x.strip()) for x in str(args.weights).split(",") if x.strip()]
    actions = _build_actions(weights, topk=int(args.topk), n_drop=int(args.n_drop))
    if args.max_actions and int(args.max_actions) > 0:
        actions = actions[: int(args.max_actions)]
    action_ids = [a.action_id for a in actions]
    if args.fixed_first_action not in action_ids:
        raise ValueError(f"fixed first action {args.fixed_first_action} not available in action set {action_ids}")

    signals = _build_signals(base_pred, long_history_pred, actions)
    action_reports: Dict[str, pd.DataFrame] = {}
    action_rows: List[Dict[str, Any]] = []
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}

    for action in actions:
        sig_start, sig_end, sig_days = _date_bounds(signals[action.action_id])
        ev = _eval_action_report(
            action=action,
            pred_df=signals[action.action_id],
            base_port_cfg=port_cfg,
            base_strategy_kwargs=strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start=start_date,
            end=end_date,
            exchange_cache=exchange_cache,
        )
        action_reports[action.action_id] = ev["report"]
        action_rows.append(
            {
                **asdict(action),
                "ok": True,
                "admissible_for_selection": "prior-window slices only; full-period row is diagnostic/non-selection",
                "signal_start": sig_start,
                "signal_end": sig_end,
                "signal_days": sig_days,
                **ev["metrics"],
            }
        )

    plan, selection_rows = _selection_plan(
        action_reports=action_reports,
        candidate_actions=action_ids,
        start_date=start_date,
        end_date=end_date,
        fixed_first_action=str(args.fixed_first_action),
        min_train_days=int(args.min_train_days),
        min_train_ir=float(args.min_train_ir),
        min_train_annret=float(args.min_train_annret),
    )
    stitched = _stitch(action_reports, plan)
    full_metrics = (
        _metrics_from_excess(_daily_excess(stitched), stitched["turnover"].astype(float))
        if not stitched.empty
        else {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
    )
    split_rows = _split_rows(stitched, start_date, end_date)

    evaluation_complete = bool(
        start_date == TEST_START
        and end_date == TEST_END
        and float(args.open_cost) == 0.0005
        and float(args.close_cost) == 0.0015
        and int(args.max_actions) == 0
        and int(args.topk) == 40
        and int(args.n_drop) == 2
        and 0.0 in weights
        and Path(args.long_history_pred).name == DEFAULT_LONG_HISTORY_PRED
    )
    hard_gate_pass = bool(evaluation_complete and full_metrics["ir"] > HARD_GATE_IR and full_metrics["annret"] > HARD_GATE_ANNRET)

    long_history_summary_path = (out_dir / str(args.long_history_summary)).resolve()
    long_history_summary: Dict[str, Any] = {}
    if long_history_summary_path.exists():
        raw_summary = _load_json(long_history_summary_path)
        long_history_summary = {
            "summary_json": str(long_history_summary_path),
            "prediction": raw_summary.get("prediction", {}),
            "valid_best_non_test": raw_summary.get("metrics", {}).get("valid_best_non_test", {}),
            "test_with_valid_selection": raw_summary.get("metrics", {}).get("test_with_valid_selection", {}),
            "hard_gate": raw_summary.get("hard_gate", {}),
        }

    _write_csv(actions_csv, action_rows)
    _write_csv(selection_csv, selection_rows)
    _write_csv(plan_csv, plan)
    _write_csv(splits_csv, split_rows)
    summary = {
        "timestamp_utc": _now_utc(),
        "task": "long_history_signal_blend_lockstep",
        "hard_gate_pass": hard_gate_pass,
        "evaluation_complete": evaluation_complete,
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET, "scope": f"net-cost {TEST_START}..{TEST_END} stitched lockstep"},
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "scope": {
            "start_date": start_date,
            "end_date": end_date,
            "fixed_first_action": args.fixed_first_action,
            "candidate_actions": action_ids,
            "weights": weights,
            "topk": int(args.topk),
            "n_drop": int(args.n_drop),
            "max_actions": int(args.max_actions),
        },
        "selection_protocol": {
            "mode": "intra_2024_forward",
            "reason": "long-history saved predictions cover the test period only; no pre-2024 prediction panel was available for pre-test validation weight selection",
            "admissible_metrics": "real net-cost backtest excess return only",
            "non_admissible_proxy": False,
            "rules": [
                "No blend weight is selected from the full 2024-2026 target result.",
                "2024H1 is fixed to a predeclared action because no prior intra-2024 window exists.",
                "2024H2 selection uses only 2024-01-01..2024-06-30 action report slices.",
                "2025 selection uses only 2024-01-01..2024-12-31 action report slices.",
                "2026 selection uses only 2024-01-01..2025-12-31 action report slices.",
                "Action full-period metrics are recorded for audit/diagnostic comparison and are non-selection evidence.",
            ],
        },
        "source_artifacts": {
            "base_run_id": args.base_run_id,
            "base_run_dir": str(base_dir),
            "long_history_pred": str(long_history_path),
            "long_history_summary": long_history_summary,
            "base_signal_bounds": dict(zip(("start", "end", "days"), _date_bounds(base_pred))),
            "long_history_signal_bounds": dict(zip(("start", "end", "days"), _date_bounds(long_history_pred))),
        },
        "action_definitions": action_rows,
        "selection_plan": plan,
        "selection_rows": selection_rows,
        "full_metrics": full_metrics,
        "split_metrics": split_rows,
        "selection_counts": {str(k): int(v) for k, v in stitched["selected_action"].value_counts().to_dict().items()}
        if not stitched.empty
        else {},
        "artifacts": {
            "summary_json": str(summary_json),
            "actions_csv": str(actions_csv),
            "selection_csv": str(selection_csv),
            "plan_csv": str(plan_csv),
            "splits_csv": str(splits_csv),
        },
        "runtime_sec": float(time.perf_counter() - started),
    }
    _write_json(summary_json, summary)
    print(
        json.dumps(
            {
                "hard_gate_pass": hard_gate_pass,
                "evaluation_complete": evaluation_complete,
                "full_metrics": full_metrics,
                "selection_plan": plan,
                "summary_json": str(summary_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
