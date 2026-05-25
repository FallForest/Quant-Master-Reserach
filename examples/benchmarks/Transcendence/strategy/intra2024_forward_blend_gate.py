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
GRU45_RUN_IDS = (
    DEFAULT_BASE_RUN,
    "1a085ff9b5a34f408a44ad74055fc5da",
    "773bd6d8413b4bb0b388a63a6b5b6a86",
)
GRU45_RUN_WEIGHTS = (0.40, 0.20, 0.40)


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    signal_kind: str
    run_ids: Tuple[str, ...]
    weights: Tuple[float, ...]
    topk: int
    n_drop: int
    blend_gru_weight: float | None = None
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


def _date_bounds(df: pd.DataFrame) -> Tuple[str, str, int]:
    idx = pd.to_datetime(df.index.get_level_values(0) if isinstance(df.index, pd.MultiIndex) else df.index)
    dates = pd.DatetimeIndex(idx.normalize().unique()).sort_values()
    return str(dates.min().date()), str(dates.max().date()), int(len(dates))


def _rank_pct(score: pd.Series) -> pd.Series:
    if isinstance(score.index, pd.MultiIndex):
        return score.groupby(level=0).rank(method="average", pct=True)
    return score.groupby(score.index).rank(method="average", pct=True)


def _rank_ensemble_signal(
    tracking_dir: Path,
    run_ids: Sequence[str],
    weights: Sequence[float],
    start: str,
    end: str,
) -> pd.DataFrame:
    cols = []
    for run_id in run_ids:
        run_dir = conv._find_run_dir(tracking_dir, run_id)
        pred = conv._as_score_df(_load_pickle(run_dir / "artifacts" / "pred.pkl"))
        pred = conv._slice_pred(pred, pd.Timestamp(start), pd.Timestamp(end))
        ranked = _rank_pct(pred["score"].astype(float))
        ranked.name = run_id
        cols.append(ranked)
    panel = pd.concat(cols, axis=1)
    w = pd.Series(weights, index=panel.columns, dtype=float)
    denom = panel.notna().mul(w, axis=1).sum(axis=1)
    score = panel.mul(w, axis=1).fillna(0.0).sum(axis=1).div(denom.where(denom > 0)).dropna()
    return score.to_frame("score").sort_index()


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


def _build_actions() -> List[ActionSpec]:
    actions = [
        ActionSpec("base40", "base", (DEFAULT_BASE_RUN,), (1.0,), 40, 2, None, "fixed base signal topk=40,n_drop=2"),
        ActionSpec("gru45", "gru_rank", GRU45_RUN_IDS, GRU45_RUN_WEIGHTS, 45, 4, 1.0, "fixed GRU/rank signal topk=45,n_drop=4"),
    ]
    for w in (0.25, 0.50, 0.75):
        label = f"{int(round(w * 100)):02d}"
        actions.append(
            ActionSpec(
                f"blend{label}",
                "base_gru_blend",
                GRU45_RUN_IDS,
                (1.0 - w, w),
                40,
                2,
                w,
                f"fixed rank-signal blend: base_weight={1.0 - w:.2f}, gru_weight={w:.2f}; topk=40,n_drop=2",
            )
        )
    return actions


def _build_signals(tracking_dir: Path, base_pred: pd.DataFrame, start: str, end: str) -> Dict[str, pd.DataFrame]:
    base = conv._slice_pred(base_pred, pd.Timestamp(start), pd.Timestamp(end)).sort_index()
    gru = _rank_ensemble_signal(tracking_dir, GRU45_RUN_IDS, GRU45_RUN_WEIGHTS, start, end)
    return {
        "base40": base,
        "gru45": gru,
        "blend25": conv._blend_rank_scores(base, gru, 0.75, 0.25).sort_index(),
        "blend50": conv._blend_rank_scores(base, gru, 0.50, 0.50).sort_index(),
        "blend75": conv._blend_rank_scores(base, gru, 0.25, 0.75).sort_index(),
    }


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
    fixed_2024h1_action: str,
    min_train_days: int,
    min_train_ir: float,
    min_train_annret: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    plan: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    windows = [
        ("2024H1_fixed", "", "", "2024-01-01", "2024-06-30", fixed_2024h1_action, "fixed_predeclared_2024H1_no_test_selection"),
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
                fallback_action=fixed_2024h1_action,
            )
            for row in rows:
                row.update({"apply_tag": tag, "selection_rule": "prior_window_only"})
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
    p = argparse.ArgumentParser(description="Strict intra-2024 forward selection over base40/gru45/fixed blends.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=DEFAULT_BASE_RUN)
    p.add_argument("--start-date", default=TEST_START)
    p.add_argument("--end-date", default=TEST_END)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--fixed-2024h1-action", default="base40", choices=["base40", "gru45", "blend25", "blend50", "blend75"])
    p.add_argument("--min-train-days", type=int, default=20)
    p.add_argument("--min-train-ir", type=float, default=-99.0)
    p.add_argument("--min-train-annret", type=float, default=-99.0)
    p.add_argument("--max-actions", type=int, default=0, help="Bounded debug run: evaluate only the first N actions; 0 means all.")
    p.add_argument("--output-prefix", default="intra2024_forward_blend_gate")
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

    base_pred = conv._as_score_df(_load_pickle(base_dir / "artifacts" / "pred.pkl"))
    actions = _build_actions()
    if args.max_actions and int(args.max_actions) > 0:
        actions = actions[: int(args.max_actions)]
    action_ids = [a.action_id for a in actions]
    if args.fixed_2024h1_action not in action_ids:
        raise ValueError(f"fixed 2024H1 action {args.fixed_2024h1_action} not available in bounded action set {action_ids}")

    signals = _build_signals(tracking_dir, base_pred, start_date, end_date)
    action_reports: Dict[str, pd.DataFrame] = {}
    action_rows: List[Dict[str, Any]] = []
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}

    for action in actions:
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
                "signal_start": _date_bounds(signals[action.action_id])[0],
                "signal_end": _date_bounds(signals[action.action_id])[1],
                "signal_days": _date_bounds(signals[action.action_id])[2],
                **ev["metrics"],
            }
        )

    plan, selection_rows = _selection_plan(
        action_reports=action_reports,
        candidate_actions=action_ids,
        start_date=start_date,
        end_date=end_date,
        fixed_2024h1_action=str(args.fixed_2024h1_action),
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
        and set(action_ids) == {"base40", "gru45", "blend25", "blend50", "blend75"}
    )
    hard_gate_pass = bool(evaluation_complete and full_metrics["ir"] > HARD_GATE_IR and full_metrics["annret"] > HARD_GATE_ANNRET)

    _write_csv(actions_csv, action_rows)
    _write_csv(selection_csv, selection_rows)
    _write_csv(plan_csv, plan)
    _write_csv(splits_csv, split_rows)
    summary = {
        "timestamp_utc": _now_utc(),
        "task": "intra2024_forward_blend_gate",
        "hard_gate_pass": hard_gate_pass,
        "evaluation_complete": evaluation_complete,
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "scope": {
            "start_date": start_date,
            "end_date": end_date,
            "fixed_2024h1_action": args.fixed_2024h1_action,
            "candidate_actions": action_ids,
            "max_actions": int(args.max_actions),
        },
        "strict_protocol": [
            "No selection uses the full 2024-2026 target period.",
            "2024H1 is fixed to a predeclared action because no prior intra-2024 window exists inside the target.",
            "2024H2 selection uses only 2024-01-01..2024-06-30 action report slices.",
            "2025 selection uses only 2024-01-01..2024-12-31 action report slices.",
            "2026 selection uses only 2024-01-01..2025-12-31 action report slices.",
            "Action signals, blend weights, topk/n_drop, costs, and hard gate are constants or CLI inputs recorded in artifacts.",
        ],
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
