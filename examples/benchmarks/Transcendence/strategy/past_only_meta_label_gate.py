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
    fallback_of: str = ""


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


def _daily_excess(report: pd.DataFrame) -> pd.Series:
    return (report["return"].astype(float) - report["bench"].astype(float) - report["cost"].astype(float)).rename("excess")


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
    if action.signal_kind == "fallback_anchor":
        return conv._slice_pred(base_pred, pd.Timestamp(start), pd.Timestamp(end))
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
    artifact_note = "loaded exact run portfolio report artifact"
    if action.signal_kind == "base" or action.signal_kind == "fallback_anchor":
        run_id = action.run_ids[0]
    elif action.signal_kind == "rank_ensemble":
        # Artifact mode is a smoke/diagnostic fallback; use the leading predeclared run's real report.
        run_id = action.run_ids[0]
        artifact_note = "diagnostic fallback: rank-ensemble action uses leading predeclared run report artifact"
    else:
        raise ValueError(f"unsupported action artifact kind={action.signal_kind}")
    run_dir = conv._find_run_dir(tracking_dir, _resolve_run(run_id))
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
        "artifact_note": artifact_note,
        "artifact_run_id": _resolve_run(run_id),
        "artifact_report_path": str(report_path),
    }


def _build_actions() -> List[ActionSpec]:
    return [
        ActionSpec("base45", "base", (DEFAULT_BASE_RUN,), (1.0,), 45, 4),
        ActionSpec("base40", "base", (DEFAULT_BASE_RUN,), (1.0,), 40, 2),
        ActionSpec("rank50", "rank_ensemble", (DEFAULT_BASE_RUN, RUN_ALIAS["773"], RUN_ALIAS["bc641"]), (0.60, 0.20, 0.20), 50, 5),
        ActionSpec("gru45", "rank_ensemble", (DEFAULT_BASE_RUN, RUN_ALIAS["1a085"], RUN_ALIAS["773"]), (0.40, 0.20, 0.40), 45, 4),
        ActionSpec("fallback", "fallback_anchor", (DEFAULT_BASE_RUN,), (1.0,), 45, 4, fallback_of="base45"),
    ]


def _build_periods(max_periods: int) -> List[Dict[str, str]]:
    periods = [
        {
            "period": "2024H2",
            "train_start": "2024-01-02",
            "train_end": "2024-06-30",
            "apply_start": "2024-07-01",
            "apply_end": "2024-12-31",
        },
        {
            "period": "2025H1",
            "train_start": "2024-01-02",
            "train_end": "2024-12-31",
            "apply_start": "2025-01-01",
            "apply_end": "2025-06-30",
        },
        {
            "period": "2025H2",
            "train_start": "2024-07-01",
            "train_end": "2025-06-30",
            "apply_start": "2025-07-01",
            "apply_end": "2025-12-31",
        },
        {
            "period": "2026YTD",
            "train_start": "2025-01-01",
            "train_end": "2025-12-31",
            "apply_start": "2026-01-01",
            "apply_end": "2026-04-30",
        },
    ]
    if max_periods > 0:
        return periods[:max_periods]
    return periods


def _clip_periods(periods: Iterable[Dict[str, str]], start: str, end: str) -> List[Dict[str, str]]:
    out = []
    st = pd.Timestamp(start)
    ed = pd.Timestamp(end)
    for p in periods:
        q = dict(p)
        q["train_start"] = str(max(pd.Timestamp(q["train_start"]), st).date())
        q["train_end"] = str(min(pd.Timestamp(q["train_end"]), ed).date())
        q["apply_start"] = str(max(pd.Timestamp(q["apply_start"]), st).date())
        q["apply_end"] = str(min(pd.Timestamp(q["apply_end"]), ed).date())
        if pd.Timestamp(q["train_start"]) <= pd.Timestamp(q["train_end"]) and pd.Timestamp(q["apply_start"]) <= pd.Timestamp(q["apply_end"]):
            out.append(q)
    return out


def _action_features(reports: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for action_id, report in reports.items():
        rep = report.reindex(dates)
        ex = _daily_excess(rep)
        feat = pd.DataFrame(index=dates)
        feat[f"{action_id}__ret5"] = ex.rolling(5, min_periods=3).mean() * 252.0
        feat[f"{action_id}__ret20"] = ex.rolling(20, min_periods=8).mean() * 252.0
        feat[f"{action_id}__vol20"] = ex.rolling(20, min_periods=8).std(ddof=0) * np.sqrt(252.0)
        feat[f"{action_id}__ir20"] = feat[f"{action_id}__ret20"].div(feat[f"{action_id}__vol20"].replace(0.0, np.nan))
        feat[f"{action_id}__turn5"] = rep["turnover"].astype(float).rolling(5, min_periods=3).mean()
        frames.append(feat.shift(1))
    return pd.concat(frames, axis=1)


def _score_action(action_id: str, feat_row: pd.Series, params: Dict[str, float]) -> float:
    p = str(action_id)
    ret5 = float(feat_row.get(f"{p}__ret5", 0.0) or 0.0)
    ret20 = float(feat_row.get(f"{p}__ret20", 0.0) or 0.0)
    ir20 = float(feat_row.get(f"{p}__ir20", 0.0) or 0.0)
    vol20 = float(feat_row.get(f"{p}__vol20", 0.0) or 0.0)
    turn5 = float(feat_row.get(f"{p}__turn5", 0.0) or 0.0)
    return (
        float(params["w_ret5"]) * ret5
        + float(params["w_ret20"]) * ret20
        + float(params["w_ir20"]) * ir20
        - float(params["w_vol20"]) * vol20
        - float(params["w_turn5"]) * turn5
    )


def _select_actions_for_dates(
    *,
    features: pd.DataFrame,
    candidate_actions: Sequence[str],
    params: Dict[str, float],
    dates: pd.DatetimeIndex,
    warmup_days: int,
    margin: float,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    selected = pd.Series(index=dates, dtype=object)
    rows: List[Dict[str, Any]] = []
    prev = "fallback"
    for i, dt in enumerate(dates):
        feat = features.loc[dt] if dt in features.index else pd.Series(dtype=float)
        if i < int(warmup_days) or feat.dropna().empty:
            sid = "fallback"
            reason = "warmup_or_missing_features"
            scores = {a: float("nan") for a in candidate_actions}
        else:
            scores = {a: _score_action(a, feat, params) for a in candidate_actions}
            candidate = max(scores, key=scores.get)
            prev_score = scores.get(prev, -1e18)
            cand_score = scores[candidate]
            sid = prev if prev in scores and cand_score < prev_score + float(margin) else candidate
            reason = "hysteresis_hold" if sid == prev and candidate != prev else "past_only_score"
        selected.loc[dt] = sid
        rows.append(
            {
                "date": str(pd.Timestamp(dt).date()),
                "selected_action": sid,
                "previous_action": prev,
                "reason": reason,
                **{f"score_{k}": float(v) for k, v in scores.items()},
            }
        )
        prev = sid
    return selected, rows


def _stitch_reports(reports: Dict[str, pd.DataFrame], selected: pd.Series) -> pd.DataFrame:
    rows = []
    for dt, action_id in selected.items():
        source = "base45" if action_id == "fallback" else str(action_id)
        report = reports[source]
        if dt in report.index:
            rec = report.loc[dt].copy()
            rec["selected_action"] = str(action_id)
            rows.append((dt, rec))
    if not rows:
        return pd.DataFrame()
    idx, vals = zip(*rows)
    return pd.DataFrame(list(vals), index=pd.DatetimeIndex(idx)).sort_index()


def _param_grid(mode: str) -> List[Dict[str, float]]:
    if mode == "smoke":
        return [
            {"w_ret5": 0.60, "w_ret20": 0.90, "w_ir20": 0.20, "w_vol20": 0.35, "w_turn5": 0.20, "margin": 0.01},
            {"w_ret5": 0.80, "w_ret20": 0.70, "w_ir20": 0.30, "w_vol20": 0.45, "w_turn5": 0.25, "margin": 0.02},
        ]
    return [
        {"w_ret5": 0.50, "w_ret20": 1.00, "w_ir20": 0.15, "w_vol20": 0.25, "w_turn5": 0.15, "margin": 0.00},
        {"w_ret5": 0.60, "w_ret20": 0.90, "w_ir20": 0.20, "w_vol20": 0.35, "w_turn5": 0.20, "margin": 0.01},
        {"w_ret5": 0.80, "w_ret20": 0.70, "w_ir20": 0.30, "w_vol20": 0.45, "w_turn5": 0.25, "margin": 0.02},
        {"w_ret5": 1.00, "w_ret20": 0.50, "w_ir20": 0.30, "w_vol20": 0.55, "w_turn5": 0.30, "margin": 0.03},
    ]


def _blocked_cv_dates(train_dates: pd.DatetimeIndex, valid_days: int, max_folds: int) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    uniq = pd.DatetimeIndex(sorted(pd.Index(train_dates).unique()))
    if len(uniq) < int(valid_days) * 2:
        return []
    folds: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    end = len(uniq)
    for _ in range(int(max_folds)):
        v_start = end - int(valid_days)
        if v_start < int(valid_days):
            break
        folds.append((pd.DatetimeIndex(uniq[:v_start]), pd.DatetimeIndex(uniq[v_start:end])))
        end = v_start
    return list(reversed(folds))


def _select_params_prior_only(
    *,
    reports: Dict[str, pd.DataFrame],
    features: pd.DataFrame,
    candidate_actions: Sequence[str],
    train_start: str,
    train_end: str,
    params_grid: Sequence[Dict[str, float]],
    warmup_days: int,
    cv_valid_days: int,
    cv_max_folds: int,
) -> Dict[str, Any]:
    train_idx = pd.date_range(train_start, train_end, freq="D")
    train_idx = pd.DatetimeIndex([d for d in train_idx if d in features.index])
    folds = _blocked_cv_dates(train_idx, valid_days=cv_valid_days, max_folds=cv_max_folds)
    if not folds:
        p0 = dict(params_grid[0])
        return {"params": p0, "cv_score": float("nan"), "cv_folds": 0, "fold_scores": []}

    best: Dict[str, Any] | None = None
    for params in params_grid:
        scores = []
        for _, va_dates in folds:
            selected, _ = _select_actions_for_dates(
                features=features,
                candidate_actions=candidate_actions,
                params=params,
                dates=va_dates,
                warmup_days=warmup_days,
                margin=float(params["margin"]),
            )
            stitched = _stitch_reports(reports, selected)
            if stitched.empty:
                continue
            excess = _daily_excess(stitched)
            met = _metrics_from_excess(excess, stitched["turnover"].astype(float))
            score = float(met["ir"]) + 0.35 * float(met["annret"]) - 0.20 * abs(float(met["max_drawdown"]))
            if np.isfinite(score):
                scores.append(score)
        if not scores:
            continue
        row = {"params": dict(params), "cv_score": float(np.mean(scores)), "cv_folds": len(scores), "fold_scores": scores}
        if best is None or float(row["cv_score"]) > float(best["cv_score"]):
            best = row
    if best is None:
        p0 = dict(params_grid[0])
        return {"params": p0, "cv_score": float("nan"), "cv_folds": 0, "fold_scores": []}
    return best


def _year_rows(stitched: pd.DataFrame, start: str, end: str) -> List[Dict[str, Any]]:
    rows = []
    for year in (2024, 2025, 2026):
        ys = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(start))
        ye = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(end))
        if ys > ye:
            continue
        tag = f"{year}_ytd" if ye < pd.Timestamp(f"{year}-12-31") else str(year)
        rep = _slice_report(stitched, ys, ye)
        if rep.empty:
            continue
        rows.append({"split": tag, "start": str(ys.date()), "end": str(ye.date()), **_metrics_from_excess(_daily_excess(rep), rep["turnover"].astype(float))})
    return rows


def _error_fields(exc: Exception) -> Dict[str, str]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)).strip()
    return {"error_type": type(exc).__name__, "error_message": str(exc), "traceback_tail": tb}


def _checkpoint(
    path: Path,
    *,
    stage: str,
    started: float,
    action_rows: Sequence[Dict[str, Any]],
    error_rows: Sequence[Dict[str, Any]],
    extra: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "timestamp_utc": _now_utc(),
        "stage": stage,
        "runtime_sec": float(time.perf_counter() - started),
        "counts": {"actions": len(action_rows), "errors": len(error_rows)},
        "last_action": action_rows[-1] if action_rows else None,
        "errors": list(error_rows),
        "extra": extra or {},
    }
    _write_json(path, payload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Past-only meta-label gate over predeclared Transcendence alternatives.")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--base-run-id", default=DEFAULT_BASE_RUN)
    p.add_argument("--start-date", default=TEST_START)
    p.add_argument("--end-date", default=TEST_END)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--max-periods", type=int, default=0, help="Cap walk-forward apply periods; 0 means all.")
    p.add_argument("--warmup-days", type=int, default=20)
    p.add_argument("--cv-valid-days", type=int, default=42)
    p.add_argument("--cv-max-folds", type=int, default=3)
    p.add_argument(
        "--action-source",
        choices=["backtest", "artifacts"],
        default="backtest",
        help="backtest runs real portfolio evaluation; artifacts reuses existing report_normal_1day.pkl files for bounded smoke diagnostics.",
    )
    p.add_argument("--output-prefix", default="past_only_meta_label_gate")
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
    selections_csv = trans_dir / f"{args.output_prefix}_selections_{stamp}.csv"
    split_csv = trans_dir / f"{args.output_prefix}_splits_{stamp}.csv"
    period_csv = trans_dir / f"{args.output_prefix}_periods_{stamp}.csv"
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
    periods = _clip_periods(_build_periods(int(args.max_periods)), start_date, end_date)
    if args.mode == "smoke" and int(args.max_periods) == 0:
        periods = periods[:1]
    if not periods:
        raise RuntimeError("no walk-forward periods overlap requested start/end dates")
    action_start = str(min(pd.Timestamp(p["train_start"]) for p in periods).date())
    action_end = str(max(pd.Timestamp(p["apply_end"]) for p in periods).date())

    base_pred = conv._as_score_df(_load_pickle(base_dir / "artifacts" / "pred.pkl"))
    base_pred = conv._slice_pred(base_pred, pd.Timestamp(action_start), pd.Timestamp(action_end))
    actions = _build_actions()
    action_reports: Dict[str, pd.DataFrame] = {}
    action_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
    _checkpoint(
        checkpoint_json,
        stage="initialized",
        started=started,
        action_rows=action_rows,
        error_rows=error_rows,
        extra={
            "action_source": args.action_source,
            "action_coverage_start": action_start,
            "action_coverage_end": action_end,
            "note": "If process exits non-zero without summary, this checkpoint shows the last completed stage.",
        },
    )

    for action in actions:
        source_action = "base45" if action.action_id == "fallback" else action.action_id
        if action.action_id == "fallback" and "base45" in action_reports:
            action_reports[action.action_id] = action_reports["base45"]
            base_met = _metrics_from_excess(_daily_excess(action_reports["fallback"]), action_reports["fallback"]["turnover"].astype(float))
            action_rows.append(
                {
                    **asdict(action),
                    **base_met,
                    "source_action": source_action,
                    "action_source": args.action_source,
                    "artifact_note": "copied base45 action report",
                    "ok": True,
                }
            )
            _checkpoint(
                checkpoint_json,
                stage="action_complete",
                started=started,
                action_rows=action_rows,
                error_rows=error_rows,
                extra={"action_id": action.action_id, "source_action": source_action, "copied_from": "base45"},
            )
            continue
        try:
            _checkpoint(
                checkpoint_json,
                stage="action_start",
                started=started,
                action_rows=action_rows,
                error_rows=error_rows,
                extra={
                    "action_id": action.action_id,
                    "source_action": source_action,
                    "action_source": args.action_source,
                    "exchange_cache_keys": len(exchange_cache),
                },
            )
            if args.action_source == "artifacts":
                ev = _load_action_report_artifact(tracking_dir, action, action_start, action_end)
            else:
                sig = _signal_for_action(tracking_dir, action, base_pred, action_start, action_end)
                ev = _eval_action_report(
                    action=action,
                    pred_df=sig,
                    base_port_cfg=port_cfg,
                    base_strategy_kwargs=strategy_kwargs,
                    open_cost=float(args.open_cost),
                    close_cost=float(args.close_cost),
                    start=action_start,
                    end=action_end,
                    exchange_cache=exchange_cache,
                )
            action_reports[action.action_id] = ev["report"]
            action_rows.append(
                {
                    **asdict(action),
                    **ev["metrics"],
                    "source_action": source_action,
                    "action_source": args.action_source,
                    "artifact_note": str(ev.get("artifact_note", "")),
                    "artifact_run_id": str(ev.get("artifact_run_id", "")),
                    "artifact_report_path": str(ev.get("artifact_report_path", "")),
                    "ok": True,
                }
            )
            _checkpoint(
                checkpoint_json,
                stage="action_complete",
                started=started,
                action_rows=action_rows,
                error_rows=error_rows,
                extra={
                    "action_id": action.action_id,
                    "source_action": source_action,
                    "action_source": args.action_source,
                    "exchange_cache_keys": len(exchange_cache),
                },
            )
        except Exception as exc:  # noqa: BLE001
            fields = {"action_id": action.action_id, **_error_fields(exc)}
            error_rows.append(fields)
            action_rows.append({**asdict(action), "source_action": source_action, "action_source": args.action_source, "ok": False, **fields})
            _checkpoint(
                checkpoint_json,
                stage="action_error",
                started=started,
                action_rows=action_rows,
                error_rows=error_rows,
                extra={"action_id": action.action_id, "source_action": source_action, "action_source": args.action_source},
            )

    required = {"base45", "base40", "rank50", "gru45", "fallback"}
    missing = sorted(required.difference(action_reports))
    if missing:
        raise RuntimeError(f"missing required action reports: {missing}")

    dates = pd.DatetimeIndex(action_reports["base45"].index)
    features = _action_features(action_reports, dates)

    candidate_actions = ["base45", "base40", "rank50", "gru45", "fallback"]
    params_grid = _param_grid(args.mode)
    period_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    stitched_parts: List[pd.DataFrame] = []

    for period in periods:
        apply_dates = pd.DatetimeIndex([d for d in dates if pd.Timestamp(period["apply_start"]) <= d <= pd.Timestamp(period["apply_end"])])
        sel = _select_params_prior_only(
            reports=action_reports,
            features=features,
            candidate_actions=candidate_actions,
            train_start=period["train_start"],
            train_end=period["train_end"],
            params_grid=params_grid,
            warmup_days=int(args.warmup_days),
            cv_valid_days=int(args.cv_valid_days),
            cv_max_folds=int(args.cv_max_folds),
        )
        selected, sel_rows = _select_actions_for_dates(
            features=features,
            candidate_actions=candidate_actions,
            params=sel["params"],
            dates=apply_dates,
            warmup_days=0,
            margin=float(sel["params"]["margin"]),
        )
        stitched = _stitch_reports(action_reports, selected)
        met = _metrics_from_excess(_daily_excess(stitched), stitched["turnover"].astype(float)) if not stitched.empty else {}
        counts = selected.value_counts().to_dict()
        period_rows.append(
            {
                **period,
                **met,
                "cv_score": float(sel["cv_score"]),
                "cv_folds": int(sel["cv_folds"]),
                "selected_counts": json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False),
                "params": json.dumps(sel["params"], ensure_ascii=False),
            }
        )
        for row in sel_rows:
            row.update(
                {
                    "period": period["period"],
                    "train_start": period["train_start"],
                    "train_end": period["train_end"],
                    "apply_start": period["apply_start"],
                    "apply_end": period["apply_end"],
                    "cv_score": float(sel["cv_score"]),
                    "cv_folds": int(sel["cv_folds"]),
                    "params": json.dumps(sel["params"], ensure_ascii=False),
                }
            )
            selection_rows.append(row)
        stitched_parts.append(stitched)

    stitched_full = pd.concat(stitched_parts).sort_index() if stitched_parts else pd.DataFrame()
    full_metrics = (
        _metrics_from_excess(_daily_excess(stitched_full), stitched_full["turnover"].astype(float))
        if not stitched_full.empty
        else {"annret": float("nan"), "ir": float("nan"), "max_drawdown": float("nan"), "turnover": float("nan")}
    )
    split_rows = _year_rows(stitched_full, start_date, end_date) if not stitched_full.empty else []
    evaluation_complete = bool(periods) and len(stitched_parts) == len(periods) and int(args.max_periods) == 0 and args.mode == "full" and args.action_source == "backtest"
    hard_gate_pass = bool(evaluation_complete and full_metrics["ir"] > HARD_GATE_IR and full_metrics["annret"] > HARD_GATE_ANNRET)
    verdict = "BREAKTHROUGH" if hard_gate_pass else "NO_GO"

    _write_csv(actions_csv, action_rows)
    _write_csv(selections_csv, selection_rows)
    _write_csv(split_csv, split_rows)
    _write_csv(period_csv, period_rows)
    _write_csv(errors_csv, error_rows)
    summary = {
        "timestamp_utc": _now_utc(),
        "task": "past_only_meta_label_gate",
        "verdict": verdict,
        "hard_gate_pass": hard_gate_pass,
        "evaluation_complete": evaluation_complete,
        "hard_gate": {"ir_gt": HARD_GATE_IR, "annret_gt": HARD_GATE_ANNRET},
        "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
        "scope": {
            "start_date": start_date,
            "end_date": end_date,
            "mode": args.mode,
            "max_periods": int(args.max_periods),
            "action_source": args.action_source,
            "action_coverage_start": action_start,
            "action_coverage_end": action_end,
        },
        "strict_leakage_controls": [
            "Action universe is predeclared: base45, base40, rank50, gru45, fallback.",
            "No test-period conversion grid search; topk/n_drop/run weights are fixed constants in the script.",
            "Gate features are rolling realized action report diagnostics shifted by one trading row before selection.",
            "Per-period gate hyperparameters are selected only by blocked CV inside the prior train window.",
            "Apply windows are selected lockstep after training: 2024H2, 2025H1, 2025H2, 2026YTD.",
            "Artifact action-source mode is marked non-complete and cannot pass hard_gate_pass.",
        ],
        "actions": action_rows,
        "period_metrics": period_rows,
        "stitched_metrics": full_metrics,
        "split_metrics": split_rows,
        "selection_counts": {str(k): int(v) for k, v in pd.Series([r["selected_action"] for r in selection_rows]).value_counts().to_dict().items()},
        "errors": error_rows,
        "artifacts": {
            "summary_json": str(summary_json),
            "checkpoint_json": str(checkpoint_json),
            "actions_csv": str(actions_csv),
            "selections_csv": str(selections_csv),
            "splits_csv": str(split_csv),
            "periods_csv": str(period_csv),
            "errors_csv": str(errors_csv),
        },
        "runtime_sec": float(time.perf_counter() - started),
    }
    _write_json(summary_json, summary)
    _checkpoint(
        checkpoint_json,
        stage="complete",
        started=started,
        action_rows=action_rows,
        error_rows=error_rows,
        extra={
            "summary_json": str(summary_json),
            "verdict": verdict,
            "hard_gate_pass": hard_gate_pass,
            "evaluation_complete": evaluation_complete,
        },
    )
    print(json.dumps({"verdict": verdict, "hard_gate_pass": hard_gate_pass, "evaluation_complete": evaluation_complete, "stitched_metrics": full_metrics, "summary": str(summary_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
