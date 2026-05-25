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
THIS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import quant_master
import pre2024_train_new_model_lockstep as base
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy


EXPERIMENT_NAME = "cost-aware holding survival label"
RAW_START = "2019-01-01"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END = "2023-12-31"
BASE_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change")
PROMOTE_IR = 2.30
PROMOTE_ANNRET = 0.16
PROMOTE_TURNOVER = 0.16


@dataclass(frozen=True)
class CandidateMetric:
    candidate_id: str
    model_family: str
    target_mode: str
    horizon_days: int
    alpha: float
    feature_count: int
    fit_sec: float
    train_rank_ic: float
    train_rank_ic_ir: float
    valid_rank_ic: float
    valid_rank_ic_ir: float


@dataclass(frozen=True)
class BacktestMetric:
    split: str
    candidate_id: str
    topk: int
    n_drop: int
    annret: float
    ir: float
    max_drawdown: float
    turnover: float
    elapsed_sec: float
    row_count: int
    finite_rows: int
    nonfinite_rows: int
    error: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(_json_sanitize(list(rows)))


def _mask(index: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(index)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


def _cs_rank_pct(s: pd.Series) -> pd.Series:
    return s.groupby(level=0, sort=False).rank(method="average", pct=True)


def _cs_z(s: pd.Series, clip: float = 6.0) -> pd.Series:
    mu = s.groupby(level=0, sort=False).transform("mean")
    sd = s.groupby(level=0, sort=False).transform("std")
    return ((s - mu) / (sd + 1e-12)).clip(-clip, clip).fillna(0.0)


def _count_calendar_rows(provider_uri: Path, start: str, end: str) -> int:
    cal_path = provider_uri / "calendars" / "day.txt"
    vals = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    idx = pd.to_datetime(pd.Index(vals))
    return int(((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))).sum())


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-8] = 1.0
    xz = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    yz = np.nan_to_num(y - np.nanmean(y), nan=0.0, posinf=0.0, neginf=0.0)
    xtx = xz.T @ xz
    coef = np.linalg.solve(xtx + np.eye(xtx.shape[0], dtype=np.float64) * float(alpha), xz.T @ yz)
    return coef.astype(np.float64), mu.astype(np.float64), sd.astype(np.float64), float(time.perf_counter() - t0)


def _predict_ridge(x: np.ndarray, coef: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    xz = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return xz @ coef


def _daily_rank_ic_series(pred: pd.Series, label: pd.Series) -> pd.Series:
    panel = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    vals: List[Tuple[pd.Timestamp, float]] = []
    for dt, g in panel.groupby(level=0, sort=False):
        if len(g) < 20:
            continue
        corr = g["pred"].corr(g["label"], method="spearman")
        if pd.notna(corr):
            vals.append((pd.Timestamp(dt), float(corr)))
    if not vals:
        return pd.Series(dtype=float)
    return pd.Series({dt: val for dt, val in vals}, dtype=float).sort_index()


def _mean_and_ir(s: pd.Series) -> Tuple[float, float]:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return float("nan"), float("nan")
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    return mean, float(mean / (std + 1e-12) * np.sqrt(252.0))


def _resolve_workflow_config(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = THIS_DIR / path
    return path.resolve()


def _active_sample_mask(index: pd.MultiIndex, start: str, end: str, horizon: int) -> pd.Series:
    dates = pd.DatetimeIndex(sorted(pd.unique(index.get_level_values(0))))
    exit_map: Dict[pd.Timestamp, pd.Timestamp] = {}
    for i, dt in enumerate(dates):
        exit_pos = i + int(horizon)
        if exit_pos < len(dates):
            exit_map[pd.Timestamp(dt)] = pd.Timestamp(dates[exit_pos])
    date_level = pd.to_datetime(index.get_level_values(0))
    exit_dates = pd.Series(date_level.map(exit_map), index=index)
    return (date_level >= pd.Timestamp(start)) & (date_level <= pd.Timestamp(end)) & (exit_dates <= pd.Timestamp(end))


def _build_cost_survival_labels(
    panel_raw: pd.DataFrame,
    feature_index: pd.MultiIndex,
    horizons: Sequence[int],
    open_cost: float,
    close_cost: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = panel_raw.copy()
    for col in BASE_FIELDS:
        p[col] = pd.to_numeric(p[col], errors="coerce").astype(float)
    factor = p["factor"].replace(0.0, np.nan).fillna(1.0)
    close = (p["close"] * factor).replace([np.inf, -np.inf], np.nan)
    labels = pd.DataFrame(index=feature_index)
    meta: Dict[str, Any] = {}
    round_trip_cost = float(open_cost) + float(close_cost)

    for horizon in horizons:
        h = int(horizon)
        entry = close.groupby(level=1, sort=False).shift(-1)
        exit_ = close.groupby(level=1, sort=False).shift(-h)
        raw_ret = exit_ / (entry + 1e-12) - 1.0
        mkt_ret = raw_ret.groupby(level=0, sort=False).transform("mean")
        excess = (raw_ret - mkt_ret).reindex(feature_index)
        churn_penalty = round_trip_cost * (10.0 / float(h)) * 0.25
        net_survival = (excess - round_trip_cost - churn_penalty).replace([np.inf, -np.inf], np.nan)

        train_mask = _active_sample_mask(feature_index, TRAIN_START, TRAIN_END, h)
        train_vals = net_survival.loc[train_mask].dropna()
        if train_vals.empty:
            raise RuntimeError(f"no train values for survival horizon {h}")
        q25, q60, q75 = train_vals.quantile([0.25, 0.60, 0.75]).astype(float).tolist()
        scale = float((q75 - q25) / 1.349)
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(train_vals.std(ddof=1))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        normalized = np.tanh((net_survival - float(q60)) / scale)
        target_col = f"survival_{h}d_trainfit_rank"
        raw_col = f"survival_{h}d_net_excess"
        labels[raw_col] = net_survival
        labels[target_col] = (_cs_rank_pct(pd.Series(normalized, index=feature_index)) - 0.5) * 2.0
        labels.loc[~_active_sample_mask(feature_index, TRAIN_START, VALID_END, h), target_col] = np.nan
        meta[target_col] = {
            "horizon_days": h,
            "round_trip_cost": round_trip_cost,
            "churn_penalty": churn_penalty,
            "train_q60_threshold": float(q60),
            "train_iqr_scale": float(scale),
            "train_sample_count": int(len(train_vals)),
            "label_formula": "tanh((forward_excess_return - round_trip_cost - churn_penalty - train_q60) / train_iqr_scale), cross-section ranked",
            "train_exit_guard": f"training samples require horizon exit <= {TRAIN_END}",
            "smoke_future_guard": f"smoke panel ends at {VALID_END}; late-2023 labels without in-window exits are NaN",
        }
    return labels.replace([np.inf, -np.inf], np.nan), meta


def _select_feature_cols(feature_cols: Sequence[str]) -> List[str]:
    preferred = [
        "rev_1__rank",
        "rev_5__rank",
        "mom_10__rank",
        "mom_20__rank",
        "mom_60__rank",
        "mom_spread_5_20__rank",
        "intraday__rank",
        "overnight__rank",
        "vwap_gap__rank",
        "vol_comp_10_60__rank",
        "vol_exp_10_20__rank",
        "liq_amount_z20__rank",
        "liq_amount_shock_5__rank",
        "vp_div_20__rank",
        "price_pos20__rank",
        "dd60__rank",
        "recovery20__rank",
        "mn_excess_ret20__rank",
        "mn_market_neutral_mom20__rank",
        "inv_vol20__rank",
    ]
    present = [c for c in preferred if c in feature_cols]
    if len(present) >= 8:
        return present
    return [c for c in feature_cols if c.endswith("__rank")][:24]


def _make_predictions(
    dataset: pd.DataFrame,
    train_mask_by_target: Dict[str, pd.Series],
    valid_mask: np.ndarray,
    feature_cols: Sequence[str],
    target_modes: Sequence[str],
    alpha_grid: Sequence[float],
    target_meta: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, pd.Series]]]:
    valid_df = dataset.loc[valid_mask]
    x_valid = valid_df[list(feature_cols)].astype(np.float64).values
    candidate_rows: List[Dict[str, Any]] = []
    predictions: Dict[str, Dict[str, pd.Series]] = {}
    for target in target_modes:
        train_mask = train_mask_by_target[target]
        train_df = dataset.loc[train_mask].dropna(subset=list(feature_cols) + [target])
        if train_df.empty:
            raise RuntimeError(f"empty train split for target={target}")
        x_train = train_df[list(feature_cols)].astype(np.float64).values
        y_train = train_df[target].astype(np.float64).values
        for alpha in alpha_grid:
            candidate_id = f"ridge_costsurv_{target}_a{float(alpha):g}"
            coef, mu, sd, fit_sec = _fit_ridge(x_train, y_train, float(alpha))
            pred_train = _cs_z(pd.Series(_predict_ridge(x_train, coef, mu, sd), index=train_df.index, name="score"))
            pred_valid = _cs_z(pd.Series(_predict_ridge(x_valid, coef, mu, sd), index=valid_df.index, name="score"))
            train_ic_s = _daily_rank_ic_series(pred_train, train_df[target])
            valid_ic_s = _daily_rank_ic_series(pred_valid, valid_df[target])
            train_ic, train_ic_ir = _mean_and_ir(train_ic_s)
            valid_ic, valid_ic_ir = _mean_and_ir(valid_ic_s)
            candidate_rows.append(
                asdict(
                    CandidateMetric(
                        candidate_id=candidate_id,
                        model_family="closed_form_ridge",
                        target_mode=target,
                        horizon_days=int(target_meta[target]["horizon_days"]),
                        alpha=float(alpha),
                        feature_count=int(len(feature_cols)),
                        fit_sec=fit_sec,
                        train_rank_ic=train_ic,
                        train_rank_ic_ir=train_ic_ir,
                        valid_rank_ic=valid_ic,
                        valid_rank_ic_ir=valid_ic_ir,
                    )
                )
            )
            predictions[candidate_id] = {"train": pred_train, "valid": pred_valid}
    return candidate_rows, predictions


def _get_report_for_day_freq(portfolio_metric_dict: Dict[str, Any]) -> pd.DataFrame:
    if "1day" in portfolio_metric_dict:
        return portfolio_metric_dict["1day"][0]
    if "day" in portfolio_metric_dict:
        return portfolio_metric_dict["day"][0]
    return portfolio_metric_dict[next(iter(portfolio_metric_dict.keys()))][0]


def _run_backtest_with_report(
    signal_df: pd.DataFrame,
    split_name: str,
    candidate_id: str,
    start_time: str,
    end_time: str,
    topk: int,
    n_drop: int,
    port_cfg_template: Dict[str, Any],
    benchmark: str,
    open_cost: float,
    close_cost: float,
    exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any],
) -> Tuple[BacktestMetric, pd.DataFrame]:
    t0 = time.perf_counter()
    try:
        cfg = copy.deepcopy(port_cfg_template)
        bcfg = cfg["backtest"]
        bcfg["start_time"] = str(pd.Timestamp(start_time).date())
        bcfg["end_time"] = str(pd.Timestamp(end_time).date())
        executor_cfg = cfg.get(
            "executor",
            {
                "class": "SimulatorExecutor",
                "module_path": "quant_master.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
            },
        )
        pred_slice = signal_df.loc[_mask(signal_df.index.get_level_values(0), start_time, end_time)].copy()
        if pred_slice.empty:
            raise ValueError(f"empty signal slice for {candidate_id}: {start_time}..{end_time}")

        exchange_kwargs = dict(bcfg.get("exchange_kwargs", {}))
        exchange_kwargs["open_cost"] = float(open_cost)
        exchange_kwargs["close_cost"] = float(close_cost)
        freq = str(executor_cfg.get("kwargs", {}).get("time_per_step", "day"))
        deal_price = str(exchange_kwargs.get("deal_price", "close"))
        limit_threshold = float(exchange_kwargs.get("limit_threshold", 0.095))
        min_cost = float(exchange_kwargs.get("min_cost", 5))
        cache_key = (
            str(bcfg["start_time"]),
            str(bcfg["end_time"]),
            float(open_cost),
            float(close_cost),
            limit_threshold,
            deal_price,
        )
        if cache_key not in exchange_cache:
            exchange_cache[cache_key] = get_exchange(
                freq=freq,
                start_time=bcfg["start_time"],
                end_time=bcfg["end_time"],
                deal_price=deal_price,
                limit_threshold=limit_threshold,
                open_cost=float(open_cost),
                close_cost=float(close_cost),
                min_cost=min_cost,
            )
        exchange_kwargs["exchange"] = exchange_cache[cache_key]

        strategy = TopkDropoutStrategy(
            signal=pred_slice,
            topk=int(topk),
            n_drop=int(n_drop),
            method_sell="bottom",
            method_buy="top",
            hold_thresh=1,
            only_tradable=False,
            forbid_all_trade_at_limit=True,
        )
        pm, _ = run_backtest(
            start_time=bcfg["start_time"],
            end_time=bcfg["end_time"],
            strategy=strategy,
            executor=executor_cfg,
            benchmark=benchmark,
            account=bcfg.get("account", 100000000),
            exchange_kwargs=exchange_kwargs,
            pos_type=bcfg.get("pos_type", "Position"),
        )
        report = _get_report_for_day_freq(pm).sort_index()
        if report.empty:
            raise ValueError(f"empty report for {candidate_id}: {start_time}..{end_time}")
        excess = (report["return"].astype(float) - report["bench"].astype(float) - report["cost"].astype(float)).rename("excess")
        risk_df = risk_analysis(excess, freq="1day")
        annret = float(risk_df.loc["annualized_return", "risk"])
        ir = float(risk_df.loc["information_ratio", "risk"])
        max_drawdown = float(risk_df.loc["max_drawdown", "risk"])
        turnover = float(report["turnover"].astype(float).mean())
        finite_rows = int(np.isfinite(report[["return", "bench", "cost", "turnover"]].to_numpy(dtype=float)).all(axis=1).sum())
        row_count = int(len(report))
        metric = BacktestMetric(
            split=split_name,
            candidate_id=candidate_id,
            topk=int(topk),
            n_drop=int(n_drop),
            annret=annret,
            ir=ir,
            max_drawdown=max_drawdown,
            turnover=turnover,
            elapsed_sec=float(time.perf_counter() - t0),
            row_count=row_count,
            finite_rows=finite_rows,
            nonfinite_rows=int(row_count - finite_rows),
            error="",
        )
        return metric, report
    except Exception as exc:  # noqa: BLE001
        metric = BacktestMetric(
            split=split_name,
            candidate_id=candidate_id,
            topk=int(topk),
            n_drop=int(n_drop),
            annret=float("nan"),
            ir=float("nan"),
            max_drawdown=float("nan"),
            turnover=float("nan"),
            elapsed_sec=float(time.perf_counter() - t0),
            row_count=0,
            finite_rows=0,
            nonfinite_rows=0,
            error=f"{type(exc).__name__}: {exc}",
        )
        return metric, pd.DataFrame()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cost-aware holding survival label quick-smoke.")
    p.add_argument("--mode", choices=["smoke"], default="smoke")
    p.add_argument("--provider-uri", default=".qmData/cn_data")
    p.add_argument("--market", default="csi300")
    p.add_argument(
        "--workflow-config",
        default="workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml",
    )
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--alpha-grid", default="10")
    p.add_argument("--topk-grid", default="35,40")
    p.add_argument("--ndrop-grid", default="2,3")
    p.add_argument("--min-names-per-day", type=int, default=40)
    p.add_argument("--output-prefix", default="cost_survival_label_pre2024")
    return p


def _artifact_paths(output_prefix: str, stamp: str) -> Dict[str, Path]:
    return {
        "summary_json": THIS_DIR / f"{output_prefix}_summary_{stamp}.json",
        "summary_md": THIS_DIR / f"{output_prefix}_summary_{stamp}.md",
        "candidates_csv": THIS_DIR / f"{output_prefix}_candidates_{stamp}.csv",
        "validation_backtests_csv": THIS_DIR / f"{output_prefix}_validation_backtests_{stamp}.csv",
        "selected_valid_report_csv": THIS_DIR / f"{output_prefix}_selected_valid_report_{stamp}.csv",
        "selected_valid_signal_csv": THIS_DIR / f"{output_prefix}_selected_valid_signal_{stamp}.csv",
    }


def _sort_metric_key(row: Dict[str, Any]) -> Tuple[float, float]:
    ir = _safe_float(row.get("ir"))
    annret = _safe_float(row.get("annret"))
    return (ir if np.isfinite(ir) else -1e9, annret if np.isfinite(annret) else -1e9)


def main() -> int:
    args = build_parser().parse_args()
    t0_all = time.perf_counter()
    stamp = _stamp()
    paths = _artifact_paths(str(args.output_prefix), stamp)
    provider_uri = Path(args.provider_uri).expanduser().resolve()
    mode = str(args.mode)
    data_end = VALID_END
    summary: Dict[str, Any] = {
        "scan_time_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "experiment_name": EXPERIMENT_NAME,
        "mode": mode,
        "status": "started",
        "blocker": "",
        "artifacts": {k: str(v) for k, v in paths.items()},
        "leakage_guardrails": {
            "train_window": [TRAIN_START, TRAIN_END],
            "selection_window": [VALID_START, VALID_END],
            "smoke_load_end": VALID_END,
            "smoke_evaluates_2024_2026": False,
            "full_mode_available": False,
        },
    }

    try:
        quant_master.init(provider_uri=str(provider_uri), region="cn")
        wf_cfg = base._load_config(_resolve_workflow_config(str(args.workflow_config)))
        port_cfg = base._extract_port_config(wf_cfg)
        benchmark = str(wf_cfg.get("benchmark", "SH000300"))

        panel_raw, _coverage_df = base._build_panel(provider_uri, str(args.market), RAW_START, data_end, BASE_FIELDS)
        feature_df, all_feature_cols = base._build_features_and_targets(panel_raw)
        feature_cols = _select_feature_cols(all_feature_cols)
        labels_df, label_meta = _build_cost_survival_labels(
            panel_raw,
            feature_df.index,
            horizons=(5, 10),
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        dataset = pd.concat([feature_df[feature_cols], labels_df], axis=1).replace([np.inf, -np.inf], np.nan)
        day_counts = dataset.groupby(level=0)[feature_cols[0]].count()
        good_days = day_counts[day_counts >= int(args.min_names_per_day)].index
        dataset = dataset.loc[dataset.index.get_level_values(0).isin(good_days)].copy()

        dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
        valid_mask = _mask(dt_idx, VALID_START, VALID_END)
        if not valid_mask.any():
            raise RuntimeError("empty 2023 validation split")
        target_modes = [c for c in label_meta]
        train_mask_by_target = {
            target: _active_sample_mask(dataset.index, TRAIN_START, TRAIN_END, int(label_meta[target]["horizon_days"]))
            for target in target_modes
        }
        alpha_grid = [float(x) for x in str(args.alpha_grid).split(",") if x.strip()]
        topk_grid = [int(x) for x in str(args.topk_grid).split(",") if x.strip()]
        ndrop_grid = [int(x) for x in str(args.ndrop_grid).split(",") if x.strip()]
        combos = [(topk, ndrop) for topk in topk_grid for ndrop in ndrop_grid if ndrop < topk]
        if not combos:
            raise RuntimeError("no valid topk/n_drop combinations")

        candidate_rows, predictions = _make_predictions(
            dataset=dataset,
            train_mask_by_target=train_mask_by_target,
            valid_mask=valid_mask,
            feature_cols=feature_cols,
            target_modes=target_modes,
            alpha_grid=alpha_grid,
            target_meta=label_meta,
        )
        expected_valid_rows = _count_calendar_rows(provider_uri, VALID_START, VALID_END)
        exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
        valid_bt_rows: List[Dict[str, Any]] = []
        valid_reports: Dict[Tuple[str, int, int], pd.DataFrame] = {}
        finite_gate_errors: List[str] = []

        for cand in candidate_rows:
            cid = str(cand["candidate_id"])
            sig = predictions[cid]["valid"].rename("score").to_frame("score")
            for topk, ndrop in combos:
                metric, report = _run_backtest_with_report(
                    sig,
                    "valid_2023_selection_only",
                    cid,
                    VALID_START,
                    VALID_END,
                    topk,
                    ndrop,
                    port_cfg,
                    benchmark,
                    float(args.open_cost),
                    float(args.close_cost),
                    exchange_cache,
                )
                row = asdict(metric)
                valid_bt_rows.append(row)
                if report is not None and not report.empty:
                    valid_reports[(cid, int(topk), int(ndrop))] = report
                if metric.error or metric.row_count != expected_valid_rows or metric.finite_rows != expected_valid_rows:
                    finite_gate_errors.append(
                        f"{cid} topk={topk} n_drop={ndrop}: error={metric.error!r} rows={metric.row_count} finite={metric.finite_rows} expected={expected_valid_rows}"
                    )

        selectable = [
            r
            for r in valid_bt_rows
            if not r["error"]
            and int(r["row_count"]) == expected_valid_rows
            and int(r["finite_rows"]) == expected_valid_rows
            and np.isfinite(_safe_float(r["ir"]))
            and np.isfinite(_safe_float(r["annret"]))
        ]
        if not selectable:
            raise RuntimeError("no validation portfolio passed finite-row and finite-metric gates")
        selected = sorted(selectable, key=_sort_metric_key, reverse=True)[0]
        selected_candidate = next(r for r in candidate_rows if r["candidate_id"] == selected["candidate_id"])
        selected_key = (str(selected["candidate_id"]), int(selected["topk"]), int(selected["n_drop"]))
        selected_report = valid_reports[selected_key]
        selected_signal = predictions[str(selected["candidate_id"])]["valid"].rename("score").to_frame("score").sort_index()

        promotion_pass = bool(
            _safe_float(selected["ir"]) >= PROMOTE_IR
            and _safe_float(selected["annret"]) >= PROMOTE_ANNRET
            and _safe_float(selected["turnover"]) <= PROMOTE_TURNOVER
            and not finite_gate_errors
        )
        verdict = "GO_FOR_EXPLICIT_FULL" if promotion_pass else "NO_GO"
        status = "ok" if promotion_pass else "gate_failed"
        if finite_gate_errors:
            status = "failed_finite_gate"
            verdict = "NO_GO"

        _write_csv(paths["candidates_csv"], candidate_rows)
        _write_csv(paths["validation_backtests_csv"], valid_bt_rows)
        selected_report.to_csv(paths["selected_valid_report_csv"])
        selected_signal.reset_index().to_csv(paths["selected_valid_signal_csv"], index=False)

        summary.update(
            {
                "status": status,
                "verdict": verdict,
                "provider_uri": str(provider_uri),
                "market": str(args.market),
                "benchmark": benchmark,
                "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
                "protocol": {
                    "raw_start": RAW_START,
                    "data_end_loaded": data_end,
                    "train": [TRAIN_START, TRAIN_END],
                    "validation_selection_only": [VALID_START, VALID_END],
                    "smoke_no_2024_2026_load_or_eval": mode == "smoke" and data_end == VALID_END,
                    "full_mode": "not implemented in this smoke-only worker artifact",
                    "model": "closed-form ridge only",
                    "candidate_labels": label_meta,
                    "feature_count": int(len(feature_cols)),
                    "features": list(feature_cols),
                    "alpha_grid": alpha_grid,
                    "portfolio_grid": [{"topk": topk, "n_drop": ndrop} for topk, ndrop in combos],
                    "selection_rule": "2023 net-cost information ratio, tie by annualized return",
                    "promotion_gate": {
                        "costed_ir_min": PROMOTE_IR,
                        "costed_annret_min": PROMOTE_ANNRET,
                        "turnover_max": PROMOTE_TURNOVER,
                    },
                },
                "selected_candidate": {
                    **selected_candidate,
                    "topk": int(selected["topk"]),
                    "n_drop": int(selected["n_drop"]),
                },
                "validation_metrics": {
                    "costed_ir": selected["ir"],
                    "costed_annret": selected["annret"],
                    "max_drawdown": selected["max_drawdown"],
                    "turnover": selected["turnover"],
                    "row_count": selected["row_count"],
                    "finite_rows": selected["finite_rows"],
                    "nonfinite_rows": selected["nonfinite_rows"],
                    "expected_validation_trading_rows": expected_valid_rows,
                },
                "finite_row_check": {
                    "passed": not finite_gate_errors,
                    "expected_validation_trading_rows": expected_valid_rows,
                    "fail_closed_errors": finite_gate_errors,
                    "rule": "every validation report must have row_count == finite_rows == validation trading rows",
                },
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )
        _write_json(paths["summary_json"], summary)
        paths["summary_md"].write_text(
            "\n".join(
                [
                    f"# Cost-Aware Holding Survival Label ({stamp})",
                    "",
                    f"- status: `{status}`",
                    f"- verdict: `{verdict}`",
                    f"- mode: `{mode}`",
                    f"- smoke_no_2024_2026_load_or_eval: `{mode == 'smoke' and data_end == VALID_END}`",
                    f"- selected_candidate: `{selected['candidate_id']}`",
                    f"- selected_rule: `topk={int(selected['topk'])}, n_drop={int(selected['n_drop'])}`",
                    f"- 2023 costed IR / AnnRet / turnover: `{_safe_float(selected['ir']):.6f}` / `{_safe_float(selected['annret']):.6f}` / `{_safe_float(selected['turnover']):.6f}`",
                    f"- finite_rows: `{int(selected['finite_rows'])}` / `{expected_valid_rows}`",
                    f"- validation_nonfinite_rows: `{int(selected['nonfinite_rows'])}`",
                    f"- fail_closed_finite_gate_passed: `{not finite_gate_errors}`",
                    f"- promotion_gate: `IR >= {PROMOTE_IR}, AnnRet >= {PROMOTE_ANNRET}, turnover <= {PROMOTE_TURNOVER}`",
                    f"- runtime_sec: `{summary['runtime_sec_total']:.3f}`",
                    f"- summary_json: `{paths['summary_json']}`",
                ]
            ),
            encoding="utf-8",
        )
        return 0 if promotion_pass and not finite_gate_errors else 2
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "verdict": "NO_GO",
                "blocker": f"{type(exc).__name__}: {exc}",
                "runtime_sec_total": float(time.perf_counter() - t0_all),
            }
        )
        _write_json(paths["summary_json"], summary)
        paths["summary_md"].write_text(
            "\n".join(
                [
                    f"# Cost-Aware Holding Survival Label ({stamp})",
                    "",
                    "- status: `failed`",
                    "- verdict: `NO_GO`",
                    f"- blocker: `{type(exc).__name__}: {exc}`",
                    f"- summary_json: `{paths['summary_json']}`",
                ]
            ),
            encoding="utf-8",
        )
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"summary_json={paths['summary_json']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
