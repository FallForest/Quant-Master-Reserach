#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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


@dataclass(frozen=True)
class DiscoveredRun:
    run_id: str
    exp_id: str
    pred_path: str
    model_class: str
    dataset_class: str
    instruments: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    pred_start: str
    pred_end: str
    pred_days: int
    full_test_coverage: bool
    valid_l2: float | None
    train_l2: float | None
    source_rank_icir: float | None
    source_test_ir: float | None
    source_test_annret: float | None
    source_topk: int | None
    source_n_drop: int | None


@dataclass(frozen=True)
class SignalDiagnostic:
    run_id: str
    rank_corr_mean: float
    rank_corr_median: float
    rank_corr_abs_mean: float
    rank_corr_days: int
    top40_overlap_mean: float
    top40_overlap_median: float
    common_rows: int
    diagnostic_scope: str
    diagnostic_used_for_shortlist: bool
    diagnostic_uses_returns_or_labels: bool


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    signal_kind: str
    run_ids: Tuple[str, ...]
    weights: Tuple[float, ...]
    topk: int
    n_drop: int
    base_weight: float
    candidate_weight: float
    shortlist_rank: int
    shortlist_reason: str


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


def _parse_metric_file(metric_path: Path) -> float | None:
    if not metric_path.exists():
        return None
    text = metric_path.read_text(encoding="utf-8", errors="ignore").strip().split()
    if len(text) < 2:
        return None
    try:
        return float(text[1])
    except ValueError:
        return None


def _extract_dataset_meta(config: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str]:
    task = config.get("task", {}) if isinstance(config, dict) else {}
    model_class = ""
    dataset_class = ""
    instruments = ""
    valid_start = ""
    valid_end = ""
    test_start = ""
    test_end = ""
    if not isinstance(task, dict):
        return model_class, dataset_class, instruments, valid_start, valid_end, test_start, test_end

    model_cfg = task.get("model", {})
    if isinstance(model_cfg, dict):
        model_class = str(model_cfg.get("class", ""))

    dataset_cfg = task.get("dataset", {})
    if isinstance(dataset_cfg, dict):
        kwargs = dataset_cfg.get("kwargs", {})
        if isinstance(kwargs, dict):
            segments = kwargs.get("segments", {})
            if isinstance(segments, dict):
                valid = segments.get("valid", [])
                test = segments.get("test", [])
                if isinstance(valid, Sequence) and len(valid) == 2:
                    valid_start = str(valid[0])
                    valid_end = str(valid[1])
                if isinstance(test, Sequence) and len(test) == 2:
                    test_start = str(test[0])
                    test_end = str(test[1])
            handler = kwargs.get("handler", {})
            if isinstance(handler, dict):
                dataset_class = str(handler.get("class", ""))
                hkwargs = handler.get("kwargs", {})
                if isinstance(hkwargs, dict):
                    instruments = str(hkwargs.get("instruments", ""))
    return model_class, dataset_class, instruments, valid_start, valid_end, test_start, test_end


def _extract_exec_params(port_cfg: Dict[str, Any]) -> Tuple[int | None, int | None]:
    strat = port_cfg.get("strategy", {})
    kwargs = strat.get("kwargs", {}) if isinstance(strat, dict) else {}
    topk = kwargs.get("topk")
    n_drop = kwargs.get("n_drop")
    return (int(topk) if topk is not None else None, int(n_drop) if n_drop is not None else None)


def _pred_dates(pred_df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pred_df.index
    if isinstance(idx, pd.MultiIndex):
        vals = pd.to_datetime(idx.get_level_values(0))
    else:
        vals = pd.to_datetime(idx)
    return pd.DatetimeIndex(vals.normalize().unique()).sort_values()


def _date_bounds(pred_df: pd.DataFrame) -> Tuple[str, str, int]:
    dates = _pred_dates(pred_df)
    if len(dates) == 0:
        return "", "", 0
    return str(dates.min().date()), str(dates.max().date()), int(len(dates))


def _discover_runs(tracking_dir: Path, start: str, end: str, max_runs: int) -> List[DiscoveredRun]:
    rows: List[DiscoveredRun] = []
    run_dirs = [p for p in sorted(tracking_dir.glob("*/*")) if p.is_dir() and len(p.name) == 32]
    if max_runs > 0:
        run_dirs = run_dirs[:max_runs]

    for run_dir in run_dirs:
        pred_path = run_dir / "artifacts" / "pred.pkl"
        cfg_path = run_dir / "artifacts" / "config"
        if not pred_path.exists() or not cfg_path.exists():
            continue
        try:
            config = conv._load_config(cfg_path)
            port_cfg = conv._extract_port_config(config)
            pred_df = conv._as_score_df(_load_pickle(pred_path))
        except Exception:
            continue
        pred_start, pred_end, pred_days = _date_bounds(pred_df)
        if not pred_start or not pred_end:
            continue
        full_coverage = bool(pd.Timestamp(pred_start) <= pd.Timestamp(start) + pd.Timedelta(days=1) and pd.Timestamp(pred_end) >= pd.Timestamp(end))
        model_class, dataset_class, instruments, valid_start, valid_end, test_start, test_end = _extract_dataset_meta(config)
        topk, n_drop = _extract_exec_params(port_cfg)
        metric_dir = run_dir / "metrics"
        rows.append(
            DiscoveredRun(
                run_id=run_dir.name,
                exp_id=run_dir.parent.name,
                pred_path=str(pred_path),
                model_class=model_class,
                dataset_class=dataset_class,
                instruments=instruments,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
                pred_start=pred_start,
                pred_end=pred_end,
                pred_days=pred_days,
                full_test_coverage=full_coverage,
                valid_l2=_parse_metric_file(metric_dir / "l2.valid"),
                train_l2=_parse_metric_file(metric_dir / "l2.train"),
                source_rank_icir=_parse_metric_file(metric_dir / "Rank ICIR"),
                source_test_ir=_parse_metric_file(metric_dir / "1day.excess_return_with_cost.information_ratio"),
                source_test_annret=_parse_metric_file(metric_dir / "1day.excess_return_with_cost.annualized_return"),
                source_topk=topk,
                source_n_drop=n_drop,
            )
        )
    return rows


def _rank_pct(score: pd.Series) -> pd.Series:
    if isinstance(score.index, pd.MultiIndex):
        return score.groupby(level=0).rank(method="average", pct=True)
    return score.groupby(score.index).rank(method="average", pct=True)


def _daily_rank_corr(base: pd.Series, cand: pd.Series) -> pd.Series:
    panel = pd.concat([base.rename("base"), cand.rename("cand")], axis=1, join="inner").dropna()
    if panel.empty:
        return pd.Series(dtype=float)
    values: Dict[pd.Timestamp, float] = {}
    for dt, grp in panel.groupby(level=0):
        if len(grp) < 5:
            continue
        corr = grp["base"].corr(grp["cand"], method="spearman")
        if pd.notna(corr):
            values[pd.Timestamp(dt)] = float(corr)
    return pd.Series(values, dtype=float).sort_index()


def _topk_overlap(base: pd.Series, cand: pd.Series, topk: int) -> pd.Series:
    panel = pd.concat([base.rename("base"), cand.rename("cand")], axis=1, join="inner").dropna()
    if panel.empty:
        return pd.Series(dtype=float)
    values: Dict[pd.Timestamp, float] = {}
    for dt, grp in panel.groupby(level=0):
        k = min(int(topk), len(grp))
        if k <= 0:
            continue
        base_top = set(grp.nlargest(k, "base").index.get_level_values(-1).astype(str))
        cand_top = set(grp.nlargest(k, "cand").index.get_level_values(-1).astype(str))
        values[pd.Timestamp(dt)] = float(len(base_top.intersection(cand_top)) / max(1, k))
    return pd.Series(values, dtype=float).sort_index()


def _signal_diagnostics(base_pred: pd.DataFrame, cand_pred: pd.DataFrame, run_id: str, start: str, end: str) -> SignalDiagnostic:
    base_slice = conv._slice_pred(base_pred, pd.Timestamp(start), pd.Timestamp(end))
    cand_slice = conv._slice_pred(cand_pred, pd.Timestamp(start), pd.Timestamp(end))
    base_rank = _rank_pct(base_slice["score"].astype(float))
    cand_rank = _rank_pct(cand_slice["score"].astype(float))
    corr = _daily_rank_corr(base_rank, cand_rank)
    overlap = _topk_overlap(base_rank, cand_rank, topk=40)
    common = pd.concat([base_rank.rename("base"), cand_rank.rename("cand")], axis=1, join="inner").dropna()
    return SignalDiagnostic(
        run_id=run_id,
        rank_corr_mean=float(corr.mean()) if not corr.empty else float("nan"),
        rank_corr_median=float(corr.median()) if not corr.empty else float("nan"),
        rank_corr_abs_mean=float(corr.abs().mean()) if not corr.empty else float("nan"),
        rank_corr_days=int(len(corr)),
        top40_overlap_mean=float(overlap.mean()) if not overlap.empty else float("nan"),
        top40_overlap_median=float(overlap.median()) if not overlap.empty else float("nan"),
        common_rows=int(len(common)),
        diagnostic_scope=f"{start}..{end}",
        diagnostic_used_for_shortlist=True,
        diagnostic_uses_returns_or_labels=False,
    )


def _safe_float(value: float | None, default: float) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _select_shortlist(
    discovered: Sequence[DiscoveredRun],
    diagnostics: Dict[str, SignalDiagnostic],
    base_run_id: str,
    max_shortlist: int,
    max_abs_corr: float,
    min_corr_days: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_run = {r.run_id: r for r in discovered}
    for run in discovered:
        if run.run_id == base_run_id or not run.full_test_coverage:
            continue
        diag = diagnostics.get(run.run_id)
        if diag is None:
            continue
        corr_abs = _safe_float(diag.rank_corr_abs_mean, 9.0)
        corr_mean = _safe_float(diag.rank_corr_mean, 9.0)
        valid_l2 = _safe_float(run.valid_l2, 9.0)
        corr_gate = bool(corr_abs <= float(max_abs_corr) and diag.rank_corr_days >= int(min_corr_days))
        rows.append(
            {
                "run_id": run.run_id,
                "selected": False,
                "shortlist_rank": 0,
                "selection_score": float(-corr_abs - valid_l2 * 1e-4 + (0.01 if corr_gate else 0.0)),
                "corr_gate": corr_gate,
                "rank_corr_abs_mean": corr_abs,
                "rank_corr_mean": corr_mean,
                "rank_corr_days": int(diag.rank_corr_days),
                "top40_overlap_mean": diag.top40_overlap_mean,
                "valid_l2": run.valid_l2,
                "source_topk": run.source_topk,
                "source_n_drop": run.source_n_drop,
                "model_class": run.model_class,
                "dataset_class": run.dataset_class,
                "instruments": run.instruments,
                "shortlist_rule": "full-test non-performance diagnostics only: low abs rank corr vs base, then pre-2024 valid_l2 tie-break",
                "performance_used_for_shortlist": False,
            }
        )
    ranked = sorted(
        rows,
        key=lambda r: (
            bool(r["corr_gate"]),
            -float(r["rank_corr_abs_mean"]),
            -_safe_float(by_run[str(r["run_id"])].valid_l2, 9.0),
            str(r["run_id"]),
        ),
        reverse=True,
    )
    selected = ranked[: max(0, int(max_shortlist))]
    selected_ids = {str(r["run_id"]) for r in selected}
    out = []
    rank = 0
    for row in ranked:
        row = dict(row)
        if str(row["run_id"]) in selected_ids:
            rank += 1
            row["selected"] = True
            row["shortlist_rank"] = rank
        out.append(row)
    return out


def _combo(action: ActionSpec) -> Dict[str, Any]:
    return {
        "family": "topk_dropout",
        "rebalance_mode": "daily",
        "rebalance_interval": 1,
        "topk": int(action.topk),
        "n_drop": int(action.n_drop),
        "hold_topk": int(action.topk),
    }


def _blend_rank_signal(base_df: pd.DataFrame, cand_df: pd.DataFrame, base_weight: float, cand_weight: float) -> pd.DataFrame:
    return conv._blend_rank_scores(base_df, cand_df, float(base_weight), float(cand_weight)).sort_index()


def _build_actions(
    shortlisted_rows: Sequence[Dict[str, Any]],
    discovered_by_id: Dict[str, DiscoveredRun],
    base_run_id: str,
    blend_weights: Sequence[float],
    include_raw_candidate: bool,
) -> List[ActionSpec]:
    actions = [
        ActionSpec(
            action_id="base40",
            signal_kind="base",
            run_ids=(base_run_id,),
            weights=(1.0,),
            topk=40,
            n_drop=2,
            base_weight=1.0,
            candidate_weight=0.0,
            shortlist_rank=0,
            shortlist_reason="predeclared base40 control",
        )
    ]
    for row in shortlisted_rows:
        if not row.get("selected"):
            continue
        run_id = str(row["run_id"])
        rank = int(row["shortlist_rank"])
        run = discovered_by_id[run_id]
        cand_topk = int(run.source_topk or 40)
        cand_ndrop = int(run.source_n_drop or 2)
        if include_raw_candidate:
            actions.append(
                ActionSpec(
                    action_id=f"cand{rank:02d}_raw_tk{cand_topk}_nd{cand_ndrop}",
                    signal_kind="candidate_raw",
                    run_ids=(run_id,),
                    weights=(1.0,),
                    topk=cand_topk,
                    n_drop=cand_ndrop,
                    base_weight=0.0,
                    candidate_weight=1.0,
                    shortlist_rank=rank,
                    shortlist_reason=str(row["shortlist_rule"]),
                )
            )
        for w in blend_weights:
            pct = int(round(float(w) * 100))
            actions.append(
                ActionSpec(
                    action_id=f"cand{rank:02d}_blend{pct:02d}",
                    signal_kind="base_candidate_rank_blend",
                    run_ids=(base_run_id, run_id),
                    weights=(1.0 - float(w), float(w)),
                    topk=40,
                    n_drop=2,
                    base_weight=1.0 - float(w),
                    candidate_weight=float(w),
                    shortlist_rank=rank,
                    shortlist_reason=str(row["shortlist_rule"]),
                )
            )
    return actions


def _daily_excess(report: pd.DataFrame) -> pd.Series:
    return (report["return"].astype(float) - report["bench"].astype(float) - report["cost"].astype(float)).rename("excess")


def _metrics_from_report(report: pd.DataFrame) -> Dict[str, float]:
    if report.empty:
        return {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
    excess = _daily_excess(report).dropna().sort_index()
    if excess.empty:
        return {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
    risk_df = risk_analysis(excess, freq="1day")
    return {
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(report["turnover"].astype(float).mean()),
    }


def _slice_report(report: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    idx = pd.to_datetime(report.index)
    return report.loc[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))].copy()


def _run_backtest_report(
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
) -> Tuple[pd.DataFrame, float]:
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
    exchange_kwargs["open_cost"] = float(open_cost)
    exchange_kwargs["close_cost"] = float(close_cost)
    exchange_kwargs["exchange"] = exchange_cache[cache_key]

    strategy = conv._build_strategy_object(
        combo=_combo(action),
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
    elapsed = float(time.perf_counter() - t0)
    return conv._get_report_for_day_freq(portfolio_metric_dict).copy().sort_index(), elapsed


def _evaluate_window(action_id: str, report: pd.DataFrame, start: str, end: str) -> Dict[str, Any]:
    sliced = _slice_report(report, start, end)
    return {"action_id": action_id, "start": start, "end": end, "days": int(len(sliced)), **_metrics_from_report(sliced)}


def _select_action(
    *,
    action_reports: Dict[str, pd.DataFrame],
    candidate_actions: Sequence[str],
    train_start: str,
    train_end: str,
    min_train_days: int,
    min_train_annret: float,
    fallback_action: str,
) -> Tuple[str, List[Dict[str, Any]], str]:
    rows = []
    for action_id in candidate_actions:
        row = _evaluate_window(action_id, action_reports[action_id], train_start, train_end)
        finite = bool(np.isfinite(float(row["ir"])) and np.isfinite(float(row["annret"])))
        gate_ok = bool(finite and int(row["days"]) >= int(min_train_days) and float(row["annret"]) >= float(min_train_annret))
        row["train_gate_ok"] = gate_ok
        row["train_gate_reason"] = "" if gate_ok else f"requires days>={min_train_days} and annret>={min_train_annret}"
        rows.append(row)
    pool = [r for r in rows if bool(r["train_gate_ok"])]
    if not pool:
        return fallback_action, rows, "fallback_no_prior_action_met_gate"
    best = sorted(
        pool,
        key=lambda r: (
            float(r.get("annret", -1e9)),
            float(r.get("ir", -1e9)),
            -abs(float(r.get("max_drawdown", -1e9))),
            str(r.get("action_id", "")),
        ),
        reverse=True,
    )[0]
    return str(best["action_id"]), rows, "max_prior_annret_tiebreak_ir"


def _selection_plan(
    *,
    action_reports: Dict[str, pd.DataFrame],
    candidate_actions: Sequence[str],
    start_date: str,
    end_date: str,
    fixed_first_action: str,
    min_train_days: int,
    min_train_annret: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    plan: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    windows = [
        ("2024H1_fixed", "", "", "2024-01-01", "2024-06-30", fixed_first_action, "fixed_predeclared_no_target_performance_selection"),
        ("2024H2", "2024-01-01", "2024-06-30", "2024-07-01", "2024-12-31", None, ""),
        ("2025", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31", None, ""),
        ("2026_ytd", "2024-01-01", "2025-12-31", "2026-01-01", "2026-04-30", None, ""),
    ]
    for tag, train_start, train_end, apply_start, apply_end, fixed, reason in windows:
        app_start = max(start_ts, pd.Timestamp(apply_start))
        app_end = min(end_ts, pd.Timestamp(apply_end))
        if app_start > app_end:
            continue
        if fixed is not None:
            selected = fixed
        else:
            selected, rows, reason = _select_action(
                action_reports=action_reports,
                candidate_actions=candidate_actions,
                train_start=train_start,
                train_end=train_end,
                min_train_days=min_train_days,
                min_train_annret=min_train_annret,
                fallback_action=fixed_first_action,
            )
            for row in rows:
                row.update({"apply_tag": tag, "selection_rule": "strict_prior_window_performance_only"})
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


def _stitch_reports(action_reports: Dict[str, pd.DataFrame], plan: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    parts = []
    for row in plan:
        action_id = str(row["selected_action"])
        part = _slice_report(action_reports[action_id], row["apply_start"], row["apply_end"])
        if part.empty:
            continue
        part = part.copy()
        part["selected_action"] = action_id
        part["apply_tag"] = str(row["apply_tag"])
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


def _split_rows(stitched: pd.DataFrame, start: str, end: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
                **_metrics_from_report(rep),
                "selected_counts": json.dumps({str(k): int(v) for k, v in rep["selected_action"].value_counts().to_dict().items()}),
            }
        )
    return rows


def _parse_blend_weights(text: str) -> List[float]:
    vals = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        val = float(raw)
        if val <= 0.0 or val >= 1.0:
            raise ValueError(f"blend weights must be in (0,1): {val}")
        vals.append(val)
    return vals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bounded low-correlation signal search with strict forward performance selection.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=DEFAULT_BASE_RUN)
    p.add_argument("--start-date", default=TEST_START)
    p.add_argument("--end-date", default=TEST_END)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--max-discovery-runs", type=int, default=0, help="0 scans all mlruns prediction artifacts.")
    p.add_argument("--max-shortlist", type=int, default=3)
    p.add_argument("--max-abs-corr", type=float, default=0.92)
    p.add_argument("--min-corr-days", type=int, default=120)
    p.add_argument("--blend-weights", default="0.20,0.35,0.50")
    p.add_argument("--include-raw-candidate", action="store_true")
    p.add_argument("--fixed-first-action", default="base40")
    p.add_argument("--min-train-days", type=int, default=20)
    p.add_argument("--min-train-annret", type=float, default=-99.0)
    p.add_argument("--output-prefix", default="high_return_low_corr_signal_search")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    stamp = _stamp()
    out_dir = Path(__file__).resolve().parent
    discovery_csv = out_dir / f"{args.output_prefix}_discovered_runs_{stamp}.csv"
    diagnostics_csv = out_dir / f"{args.output_prefix}_diagnostics_{stamp}.csv"
    shortlist_csv = out_dir / f"{args.output_prefix}_shortlist_{stamp}.csv"
    actions_csv = out_dir / f"{args.output_prefix}_actions_{stamp}.csv"
    selection_csv = out_dir / f"{args.output_prefix}_selection_{stamp}.csv"
    plan_csv = out_dir / f"{args.output_prefix}_plan_{stamp}.csv"
    splits_csv = out_dir / f"{args.output_prefix}_splits_{stamp}.csv"
    summary_json = out_dir / f"{args.output_prefix}_summary_{stamp}.json"

    start_date = str(pd.Timestamp(args.start_date).date())
    end_date = str(pd.Timestamp(args.end_date).date())
    tracking_dir = conv._parse_tracking_dir(args.tracking_uri)
    base_dir = conv._find_run_dir(tracking_dir, args.base_run_id)
    cfg = conv._load_config(base_dir / "artifacts" / "config")
    conv._init_quant_master(cfg)
    port_cfg = conv._extract_port_config(cfg)
    strategy_kwargs = dict(port_cfg.get("strategy", {}).get("kwargs", {}))
    strategy_kwargs.pop("signal", None)

    base_pred = conv._as_score_df(_load_pickle(base_dir / "artifacts" / "pred.pkl")).sort_index()
    discovered = _discover_runs(tracking_dir, start_date, end_date, int(args.max_discovery_runs))
    discovered_by_id = {r.run_id: r for r in discovered}
    if args.base_run_id not in discovered_by_id:
        raise RuntimeError(f"base run not discovered: {args.base_run_id}")

    diagnostics: Dict[str, SignalDiagnostic] = {}
    for run in discovered:
        if not run.full_test_coverage or run.run_id == args.base_run_id:
            continue
        try:
            cand_pred = conv._as_score_df(_load_pickle(Path(run.pred_path))).sort_index()
            diagnostics[run.run_id] = _signal_diagnostics(base_pred, cand_pred, run.run_id, start_date, end_date)
        except Exception:
            continue

    shortlist_rows = _select_shortlist(
        discovered=discovered,
        diagnostics=diagnostics,
        base_run_id=args.base_run_id,
        max_shortlist=int(args.max_shortlist),
        max_abs_corr=float(args.max_abs_corr),
        min_corr_days=int(args.min_corr_days),
    )
    selected_shortlist = [r for r in shortlist_rows if bool(r.get("selected"))]
    blend_weights = _parse_blend_weights(args.blend_weights)
    actions = _build_actions(
        shortlisted_rows=selected_shortlist,
        discovered_by_id=discovered_by_id,
        base_run_id=args.base_run_id,
        blend_weights=blend_weights,
        include_raw_candidate=bool(args.include_raw_candidate),
    )
    action_ids = [a.action_id for a in actions]
    if args.fixed_first_action not in action_ids:
        raise ValueError(f"fixed-first-action {args.fixed_first_action} not in available action set {action_ids}")

    signal_cache: Dict[str, pd.DataFrame] = {"base40": conv._slice_pred(base_pred, pd.Timestamp(start_date), pd.Timestamp(end_date)).sort_index()}
    for row in selected_shortlist:
        run_id = str(row["run_id"])
        cand_pred = conv._as_score_df(_load_pickle(Path(discovered_by_id[run_id].pred_path))).sort_index()
        cand_pred = conv._slice_pred(cand_pred, pd.Timestamp(start_date), pd.Timestamp(end_date)).sort_index()
        for action in actions:
            if run_id not in action.run_ids:
                continue
            if action.signal_kind == "candidate_raw":
                signal_cache[action.action_id] = cand_pred
            elif action.signal_kind == "base_candidate_rank_blend":
                signal_cache[action.action_id] = _blend_rank_signal(base_pred, cand_pred, action.base_weight, action.candidate_weight)

    action_reports: Dict[str, pd.DataFrame] = {}
    action_rows: List[Dict[str, Any]] = []
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    for action in actions:
        report, elapsed = _run_backtest_report(
            action=action,
            pred_df=signal_cache[action.action_id],
            base_port_cfg=port_cfg,
            base_strategy_kwargs=strategy_kwargs,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
            start=start_date,
            end=end_date,
            exchange_cache=exchange_cache,
        )
        action_reports[action.action_id] = report
        sig_start, sig_end, sig_days = _date_bounds(signal_cache[action.action_id])
        action_rows.append(
            {
                **asdict(action),
                "signal_start": sig_start,
                "signal_end": sig_end,
                "signal_days": sig_days,
                **_metrics_from_report(report),
                "elapsed_sec": elapsed,
                "full_window_metrics_used_for_selection": False,
            }
        )

    plan, selection_rows = _selection_plan(
        action_reports=action_reports,
        candidate_actions=action_ids,
        start_date=start_date,
        end_date=end_date,
        fixed_first_action=str(args.fixed_first_action),
        min_train_days=int(args.min_train_days),
        min_train_annret=float(args.min_train_annret),
    )
    stitched = _stitch_reports(action_reports, plan)
    full_metrics = _metrics_from_report(stitched)
    split_rows = _split_rows(stitched, start_date, end_date)
    evaluation_complete = bool(
        start_date == TEST_START
        and end_date == TEST_END
        and float(args.open_cost) == 0.0005
        and float(args.close_cost) == 0.0015
    )
    hard_gate_pass = bool(evaluation_complete and full_metrics["ir"] > HARD_GATE_IR and full_metrics["annret"] > HARD_GATE_ANNRET)

    discovery_rows = [asdict(r) for r in discovered]
    diagnostics_rows = [asdict(v) for v in diagnostics.values()]
    _write_csv(discovery_csv, discovery_rows)
    _write_csv(diagnostics_csv, diagnostics_rows)
    _write_csv(shortlist_csv, shortlist_rows)
    _write_csv(actions_csv, action_rows)
    _write_csv(selection_csv, selection_rows)
    _write_csv(plan_csv, plan)
    _write_csv(splits_csv, split_rows)

    summary = {
        "timestamp_utc": _now_utc(),
        "task": "high_return_low_corr_signal_search",
        "hard_gate_pass": hard_gate_pass,
        "evaluation_complete": evaluation_complete,
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "scope": {
            "start_date": start_date,
            "end_date": end_date,
            "base_run_id": args.base_run_id,
            "max_discovery_runs": int(args.max_discovery_runs),
            "max_shortlist": int(args.max_shortlist),
            "blend_weights": blend_weights,
            "include_raw_candidate": bool(args.include_raw_candidate),
            "fixed_first_action": args.fixed_first_action,
        },
        "leakage_controls": {
            "shortlist_selection": "Uses full-window prediction-only diagnostics vs base: rank correlation and top40 overlap. No returns, labels, reports, or test performance are used.",
            "first_window": "2024H1 uses fixed predeclared action.",
            "forward_selection": "2024H2 uses only 2024H1 action reports; 2025 uses only 2024 action reports; 2026_ytd uses only 2024-2025 action reports.",
            "full_metrics_role": "Full-window action metrics and final stitched metrics are evaluation artifacts only and are not used to choose the shortlist or selected action plan.",
        },
        "discovery_counts": {
            "discovered_runs": int(len(discovered)),
            "full_coverage_runs": int(sum(1 for r in discovered if r.full_test_coverage)),
            "diagnosed_runs": int(len(diagnostics)),
            "shortlisted_runs": int(len(selected_shortlist)),
            "actions_evaluated": int(len(actions)),
        },
        "shortlisted_candidates": selected_shortlist,
        "action_definitions": action_rows,
        "selection_plan": plan,
        "selection_rows": selection_rows,
        "full_metrics": full_metrics,
        "split_metrics": split_rows,
        "selection_counts": {str(k): int(v) for k, v in stitched["selected_action"].value_counts().to_dict().items()}
        if not stitched.empty
        else {},
        "artifacts": {
            "discovery_csv": str(discovery_csv),
            "diagnostics_csv": str(diagnostics_csv),
            "shortlist_csv": str(shortlist_csv),
            "actions_csv": str(actions_csv),
            "selection_csv": str(selection_csv),
            "plan_csv": str(plan_csv),
            "splits_csv": str(splits_csv),
            "summary_json": str(summary_json),
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
                "selection_counts": summary["selection_counts"],
                "summary_json": str(summary_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
