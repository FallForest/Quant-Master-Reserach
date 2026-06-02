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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = THIS_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parents[2]
for _path in (REPO_ROOT, BENCHMARK_ROOT, BENCHMARK_ROOT / "model", BENCHMARK_ROOT / "strategy", THIS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import quant_master
from quant_master.config import resolve_provider_uri
import pre2024_train_new_model_lockstep as base
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy


EXPERIMENT_NAME = "market-state conditional target"
RAW_START = "2019-01-01"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
BASE_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change")
INVESTIGATE_IR = 2.0
PROMOTE_IR = 2.30
PROMOTE_ANNRET = 0.16
FULL_HARD_IR = 2.9
FULL_HARD_ANNRET = 0.27
FULL_HARD_MAX_DRAWDOWN_ABS = 0.25


@dataclass(frozen=True)
class CandidateMetric:
    candidate_id: str
    model_family: str
    target_mode: str
    state_policy: str
    alpha: float
    bull_weight: float
    bear_weight: float
    volatile_weight: float
    feature_count: int
    train_sample_count: int
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


def _fit_weighted_ridge(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    w = np.nan_to_num(sample_weight.astype(np.float64), nan=1.0, posinf=1.0, neginf=1.0).clip(0.05, 10.0)
    w = w / (float(np.nanmean(w)) + 1e-12)
    mu = np.average(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), axis=0, weights=w)
    centered = np.nan_to_num(x - mu, nan=0.0, posinf=0.0, neginf=0.0)
    sd = np.sqrt(np.average(centered * centered, axis=0, weights=w))
    sd[sd < 1e-8] = 1.0
    xz = centered / sd
    y_clean = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    yz = y_clean - float(np.average(y_clean, weights=w))
    xw = xz * np.sqrt(w)[:, None]
    yw = yz * np.sqrt(w)
    xtx = xw.T @ xw
    coef = np.linalg.solve(xtx + np.eye(xtx.shape[0], dtype=np.float64) * float(alpha), xw.T @ yw)
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
    if path.is_absolute():
        return path.resolve()

    search_roots = [Path.cwd(), THIS_DIR, *THIS_DIR.parents]
    seen = set()
    for root in search_roots:
        root_resolved = root.resolve()
        if root_resolved in seen:
            continue
        seen.add(root_resolved)
        candidate = root_resolved / path
        if candidate.exists():
            return candidate.resolve()
    return (THIS_DIR.parent / path).resolve()


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


def _build_market_state_table(
    panel_raw: pd.DataFrame,
    feature_index: pd.MultiIndex,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = panel_raw.copy()
    for col in BASE_FIELDS:
        p[col] = pd.to_numeric(p[col], errors="coerce").astype(float)
    factor = p["factor"].replace(0.0, np.nan).fillna(1.0)
    close = (p["close"] * factor).replace([np.inf, -np.inf], np.nan).sort_index()
    ret1 = close.groupby(level=1, sort=False).pct_change()
    daily_mean = ret1.groupby(level=0, sort=False).mean()
    daily_width = (ret1.groupby(level=0, sort=False).apply(lambda s: float((s > 0).mean())) - 0.5) * 2.0
    state_raw = pd.DataFrame(
        {
            "mkt_ret20": daily_mean.rolling(20, min_periods=10).sum(),
            "mkt_ret60": daily_mean.rolling(60, min_periods=20).sum(),
            "mkt_vol20": daily_mean.rolling(20, min_periods=10).std(),
            "mkt_width20": daily_width.rolling(20, min_periods=10).mean(),
        }
    ).shift(1)
    train_state = state_raw.loc[(state_raw.index >= pd.Timestamp(TRAIN_START)) & (state_raw.index <= pd.Timestamp(TRAIN_END))]
    thresholds = {
        "ret20_q60": float(train_state["mkt_ret20"].quantile(0.60)),
        "ret20_q40": float(train_state["mkt_ret20"].quantile(0.40)),
        "ret60_q60": float(train_state["mkt_ret60"].quantile(0.60)),
        "width_q60": float(train_state["mkt_width20"].quantile(0.60)),
        "width_q40": float(train_state["mkt_width20"].quantile(0.40)),
        "vol_q70": float(train_state["mkt_vol20"].quantile(0.70)),
    }
    state_raw["bull"] = (
        (state_raw["mkt_ret20"] >= thresholds["ret20_q60"])
        & (state_raw["mkt_ret60"] >= thresholds["ret60_q60"])
        & (state_raw["mkt_width20"] >= thresholds["width_q60"])
    ).astype(float)
    state_raw["bear"] = (
        (state_raw["mkt_ret20"] <= thresholds["ret20_q40"])
        & (state_raw["mkt_width20"] <= thresholds["width_q40"])
    ).astype(float)
    state_raw["volatile"] = (state_raw["mkt_vol20"] >= thresholds["vol_q70"]).astype(float)
    state_by_sample = pd.DataFrame(index=feature_index)
    sample_dates = pd.DatetimeIndex(feature_index.get_level_values(0))
    for col in state_raw.columns:
        state_by_sample[f"state_{col}"] = state_raw[col].reindex(sample_dates).to_numpy(dtype=float)
    meta = {
        "definition": "daily universe mean return, breadth, 20/60-day return and volatility shifted by one trading day",
        "shift_rule": "all state variables use shift(1), so sample t only sees information available before t",
        "threshold_fit_window": [TRAIN_START, TRAIN_END],
        "thresholds": thresholds,
        "state_columns": list(state_by_sample.columns),
    }
    return state_by_sample.replace([np.inf, -np.inf], np.nan), meta


def _forward_return(
    close: pd.Series,
    horizon: int,
    open_cost: float,
    close_cost: float,
) -> pd.Series:
    entry = close.groupby(level=1, sort=False).shift(-1)
    exit_ = close.groupby(level=1, sort=False).shift(-int(horizon))
    return exit_ / (entry + 1e-12) - 1.0 - float(open_cost) - float(close_cost)


def _build_market_state_targets(
    panel_raw: pd.DataFrame,
    feature_index: pd.MultiIndex,
    state_df: pd.DataFrame,
    policies: Sequence[str],
    open_cost: float,
    close_cost: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = panel_raw.copy()
    for col in BASE_FIELDS:
        p[col] = pd.to_numeric(p[col], errors="coerce").astype(float)
    factor = p["factor"].replace(0.0, np.nan).fillna(1.0)
    close = (p["close"] * factor).replace([np.inf, -np.inf], np.nan).sort_index()
    labels = pd.DataFrame(index=feature_index)
    meta: Dict[str, Any] = {}
    ret5 = _forward_return(close, 5, open_cost, close_cost).reindex(feature_index)
    ret10 = _forward_return(close, 10, open_cost, close_cost).reindex(feature_index)
    ret20 = _forward_return(close, 20, open_cost, close_cost).reindex(feature_index)
    market_ret5 = ret5.groupby(level=0, sort=False).transform("mean")
    market_ret10 = ret10.groupby(level=0, sort=False).transform("mean")
    market_ret20 = ret20.groupby(level=0, sort=False).transform("mean")
    rel5 = ret5 - market_ret5
    rel10 = ret10 - market_ret10
    rel20 = ret20 - market_ret20
    entry = close.groupby(level=1, sort=False).shift(-1)
    path_returns_10 = pd.concat(
        [close.groupby(level=1, sort=False).shift(-step) / (entry + 1e-12) - 1.0 for step in range(1, 11)],
        axis=1,
    )
    adverse_dd10 = (-path_returns_10.min(axis=1, skipna=False)).clip(lower=0.0).reindex(feature_index)
    dd_quality10 = (rel10 - 0.5 * adverse_dd10).replace([np.inf, -np.inf], np.nan)
    defensive = (0.70 * rel5 + 0.30 * _cs_rank_pct(-pd.Series(state_df["state_volatile"].to_numpy(), index=feature_index)).fillna(0.5)).replace(
        [np.inf, -np.inf], np.nan
    )

    for policy in policies:
        if policy == "bull20_bear5_vol_def":
            raw = rel10.copy()
            raw = raw.where(state_df["state_bull"] < 0.5, rel20)
            raw = raw.where(state_df["state_bear"] < 0.5, rel5)
            raw = raw.where(state_df["state_volatile"] < 0.5, defensive)
            horizon_guard = 20
            formula = "neutral 10d excess; bull state switches to 20d excess; bear state switches to 5d excess; volatile state uses defensive 5d/state target"
        elif policy == "bull10_bear_def_vol5":
            raw = rel10.copy()
            raw = raw.where(state_df["state_bull"] < 0.5, ret10 - market_ret10)
            raw = raw.where(state_df["state_bear"] < 0.5, defensive)
            raw = raw.where(state_df["state_volatile"] < 0.5, rel5)
            horizon_guard = 10
            formula = "neutral/bull 10d excess; bear state defensive; volatile state 5d excess"
        elif policy == "bull20_bear_dd10_vol_dd10":
            raw = rel10.copy()
            raw = raw.where(state_df["state_bull"] < 0.5, rel20)
            raw = raw.where(state_df["state_bear"] < 0.5, dd_quality10)
            raw = raw.where(state_df["state_volatile"] < 0.5, dd_quality10)
            horizon_guard = 20
            formula = "neutral 10d excess; bull state 20d excess; bear/volatile states use 10d excess minus 0.5x adverse path drawdown"
        elif policy == "bull10_bear_dd10_vol_def":
            raw = rel10.copy()
            raw = raw.where(state_df["state_bull"] < 0.5, rel10)
            raw = raw.where(state_df["state_bear"] < 0.5, dd_quality10)
            raw = raw.where(state_df["state_volatile"] < 0.5, defensive)
            horizon_guard = 10
            formula = "neutral/bull 10d excess; bear state 10d path-risk quality; volatile state defensive 5d/state target"
        else:
            raise ValueError(f"unknown market-state policy: {policy}")

        train_mask = _active_sample_mask(feature_index, TRAIN_START, TRAIN_END, horizon_guard)
        active_mask = _active_sample_mask(feature_index, TRAIN_START, VALID_END, horizon_guard)
        train_vals = raw.loc[train_mask].dropna()
        if train_vals.empty:
            raise RuntimeError(f"no train values for market-state target {policy}")
        q25, q50, q75 = train_vals.quantile([0.25, 0.50, 0.75]).astype(float).tolist()
        scale = float((q75 - q25) / 1.349)
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(train_vals.std(ddof=1))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        target_col = f"mstate_{policy}_trainfit_rank"
        normalized = np.tanh((raw - float(q50)) / scale)
        labels[target_col] = (_cs_rank_pct(pd.Series(normalized, index=feature_index)) - 0.5) * 2.0
        labels.loc[~active_mask, target_col] = np.nan
        meta[target_col] = {
            "policy": policy,
            "max_horizon_days": int(horizon_guard),
            "train_median_threshold": float(q50),
            "train_iqr_scale": float(scale),
            "train_sample_count": int(len(train_vals)),
            "label_formula": formula,
            "state_input_rule": "state columns are pre-shifted by one trading day and thresholds are fit on train only",
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
    test_mask: Optional[np.ndarray],
    feature_cols: Sequence[str],
    target_modes: Sequence[str],
    alpha_grid: Sequence[float],
    weight_grid: Sequence[str],
    target_meta: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, pd.Series]]]:
    valid_df = dataset.loc[valid_mask]
    x_valid = valid_df[list(feature_cols)].astype(np.float64).values
    test_df = dataset.loc[test_mask] if test_mask is not None else None
    x_test = test_df[list(feature_cols)].astype(np.float64).values if test_df is not None else None
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
            for weight_mode in weight_grid:
                weights = _sample_weights(train_df, weight_mode)
                candidate_id = f"ridge_mstate_{target}_w{weight_mode}_a{float(alpha):g}"
                coef, mu, sd, fit_sec = _fit_weighted_ridge(x_train, y_train, weights, float(alpha))
                pred_train = _cs_z(pd.Series(_predict_ridge(x_train, coef, mu, sd), index=train_df.index, name="score"))
                pred_valid = _cs_z(pd.Series(_predict_ridge(x_valid, coef, mu, sd), index=valid_df.index, name="score"))
                pred_test = (
                    _cs_z(pd.Series(_predict_ridge(x_test, coef, mu, sd), index=test_df.index, name="score"))
                    if x_test is not None and test_df is not None
                    else None
                )
                train_ic_s = _daily_rank_ic_series(pred_train, train_df[target])
                valid_ic_s = _daily_rank_ic_series(pred_valid, valid_df[target])
                train_ic, train_ic_ir = _mean_and_ir(train_ic_s)
                valid_ic, valid_ic_ir = _mean_and_ir(valid_ic_s)
                w_meta = _weight_meta(weight_mode)
                candidate_rows.append(
                    asdict(
                        CandidateMetric(
                            candidate_id=candidate_id,
                            model_family="closed_form_weighted_ridge",
                            target_mode=target,
                            state_policy=str(target_meta[target]["policy"]),
                            alpha=float(alpha),
                            bull_weight=float(w_meta["bull_weight"]),
                            bear_weight=float(w_meta["bear_weight"]),
                            volatile_weight=float(w_meta["volatile_weight"]),
                            feature_count=int(len(feature_cols)),
                            train_sample_count=int(len(train_df)),
                            fit_sec=fit_sec,
                            train_rank_ic=train_ic,
                            train_rank_ic_ir=train_ic_ir,
                            valid_rank_ic=valid_ic,
                            valid_rank_ic_ir=valid_ic_ir,
                        )
                    )
                )
                pred_map: Dict[str, pd.Series] = {"train": pred_train, "valid": pred_valid}
                if pred_test is not None:
                    pred_map["test"] = pred_test
                predictions[candidate_id] = pred_map
    return candidate_rows, predictions


def _weight_meta(weight_mode: str) -> Dict[str, float]:
    if weight_mode == "flat":
        return {"bull_weight": 1.0, "bear_weight": 1.0, "volatile_weight": 1.0}
    if weight_mode == "defensive":
        return {"bull_weight": 0.85, "bear_weight": 1.35, "volatile_weight": 1.25}
    if weight_mode == "trend":
        return {"bull_weight": 1.25, "bear_weight": 0.90, "volatile_weight": 1.10}
    raise ValueError(f"unknown weight_mode: {weight_mode}")


def _sample_weights(df: pd.DataFrame, weight_mode: str) -> np.ndarray:
    meta = _weight_meta(weight_mode)
    w = np.ones(len(df), dtype=np.float64)
    if "state_bull" in df:
        w *= np.where(df["state_bull"].to_numpy(dtype=float) >= 0.5, float(meta["bull_weight"]), 1.0)
    if "state_bear" in df:
        w *= np.where(df["state_bear"].to_numpy(dtype=float) >= 0.5, float(meta["bear_weight"]), 1.0)
    if "state_volatile" in df:
        w *= np.where(df["state_volatile"].to_numpy(dtype=float) >= 0.5, float(meta["volatile_weight"]), 1.0)
    return w


def _get_report_for_day_freq(portfolio_metric_dict: Dict[str, Any]) -> pd.DataFrame:
    if "1day" in portfolio_metric_dict:
        return portfolio_metric_dict["1day"][0]
    if "day" in portfolio_metric_dict:
        return portfolio_metric_dict["day"][0]
    return portfolio_metric_dict[next(iter(portfolio_metric_dict.keys()))][0]


def _metrics_from_report(report: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    if report.empty:
        return {
            "split": split_name,
            "start": None,
            "end": None,
            "annret": float("nan"),
            "ir": float("nan"),
            "max_drawdown": float("nan"),
            "turnover": float("nan"),
            "row_count": 0,
            "finite_rows": 0,
            "nonfinite_rows": 0,
        }
    excess = (report["return"].astype(float) - report["bench"].astype(float) - report["cost"].astype(float)).rename("excess")
    risk_df = risk_analysis(excess, freq="1day")
    finite_cols = report[["return", "bench", "cost", "turnover"]].to_numpy(dtype=float)
    finite_rows = int(np.isfinite(finite_cols).all(axis=1).sum())
    return {
        "split": split_name,
        "start": str(pd.Timestamp(report.index.min()).date()),
        "end": str(pd.Timestamp(report.index.max()).date()),
        "annret": float(risk_df.loc["annualized_return", "risk"]),
        "ir": float(risk_df.loc["information_ratio", "risk"]),
        "max_drawdown": float(risk_df.loc["max_drawdown", "risk"]),
        "turnover": float(report["turnover"].astype(float).mean()),
        "row_count": int(len(report)),
        "finite_rows": finite_rows,
        "nonfinite_rows": int(len(report) - finite_rows),
    }


def _slice_report_metrics(report: pd.DataFrame) -> List[Dict[str, Any]]:
    specs = [
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", TEST_END),
    ]
    rows: List[Dict[str, Any]] = []
    for name, start, end in specs:
        sliced = report.loc[_mask(report.index, start, end)]
        if not sliced.empty:
            rows.append(_metrics_from_report(sliced, name))
    return rows


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
    p = argparse.ArgumentParser(description="Market-state conditional target quick-smoke / strict full eval.")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--provider-uri", default="~/.quant_master/quant_master_data/tdx_cn_data")
    p.add_argument("--market", default="csi300")
    p.add_argument(
        "--workflow-config",
        default="workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml",
    )
    p.add_argument("--open-cost", type=float, default=0.0001)
    p.add_argument("--close-cost", type=float, default=0.0006)
    p.add_argument("--policy-grid", default="bull20_bear5_vol_def,bull10_bear_def_vol5,bull20_bear_dd10_vol_dd10,bull10_bear_dd10_vol_def")
    p.add_argument("--weight-grid", default="flat,defensive,trend")
    p.add_argument("--alpha-grid", default="10")
    p.add_argument("--topk-grid", default="40")
    p.add_argument("--ndrop-grid", default="2,3")
    p.add_argument("--min-names-per-day", type=int, default=40)
    p.add_argument("--output-prefix", default="market_state_target_pre2024")
    return p


def _artifact_paths(output_prefix: str, stamp: str) -> Dict[str, Path]:
    return {
        "summary_json": THIS_DIR / f"{output_prefix}_summary_{stamp}.json",
        "summary_md": THIS_DIR / f"{output_prefix}_summary_{stamp}.md",
        "candidates_csv": THIS_DIR / f"{output_prefix}_candidates_{stamp}.csv",
        "validation_backtests_csv": THIS_DIR / f"{output_prefix}_validation_backtests_{stamp}.csv",
        "selected_valid_report_csv": THIS_DIR / f"{output_prefix}_selected_valid_report_{stamp}.csv",
        "selected_valid_signal_csv": THIS_DIR / f"{output_prefix}_selected_valid_signal_{stamp}.csv",
        "selected_test_report_csv": THIS_DIR / f"{output_prefix}_selected_test_report_{stamp}.csv",
        "selected_test_signal_csv": THIS_DIR / f"{output_prefix}_selected_test_signal_{stamp}.csv",
        "test_slices_csv": THIS_DIR / f"{output_prefix}_test_slices_{stamp}.csv",
    }


def _sort_metric_key(row: Dict[str, Any]) -> Tuple[float, float]:
    ir = _safe_float(row.get("ir"))
    annret = _safe_float(row.get("annret"))
    return (ir if np.isfinite(ir) else -1e9, annret if np.isfinite(annret) else -1e9)


def _selection_status(ir: float, annret: float, finite_gate_pass: bool) -> Tuple[str, str]:
    if not finite_gate_pass:
        return "failed_finite_gate", "NO_GO"
    if np.isfinite(ir) and np.isfinite(annret) and ir >= PROMOTE_IR and annret >= PROMOTE_ANNRET:
        return "promotion_passed", "PROMOTE"
    if np.isfinite(ir) and ir >= INVESTIGATE_IR:
        return "investigate", "INVESTIGATE"
    return "gate_failed", "NO_GO"


def main() -> int:
    args = build_parser().parse_args()
    t0_all = time.perf_counter()
    stamp = _stamp()
    paths = _artifact_paths(str(args.output_prefix), stamp)
    provider_uri = Path(resolve_provider_uri(args.provider_uri, base_dir=REPO_ROOT))
    mode = str(args.mode)
    data_end = TEST_END if mode == "full" else VALID_END
    summary: Dict[str, Any] = {
        "scan_time_utc": _now_utc(),
        "script": str(Path(__file__).resolve()),
        "experiment_name": EXPERIMENT_NAME,
        "mode": mode,
        "status": "started",
        "verdict": "NO_GO",
        "blocker": "",
        "artifacts": {k: str(v) for k, v in paths.items()},
        "leakage_guardrails": {
            "raw_lookback_start": RAW_START,
            "train_window": [TRAIN_START, TRAIN_END],
            "selection_window": [VALID_START, VALID_END],
            "smoke_load_end": VALID_END,
            "smoke_evaluates_2024_2026": False,
            "full_mode_available": True,
            "full_test_window": [TEST_START, TEST_END],
            "full_selection_uses_test_metrics": False,
            "full_mode_guard": "candidate and portfolio selection are locked from 2023 validation only; 2024-2026 is evaluated once after selection",
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
        policy_grid = [x.strip() for x in str(args.policy_grid).split(",") if x.strip()]
        weight_grid = [x.strip() for x in str(args.weight_grid).split(",") if x.strip()]
        state_df, state_meta = _build_market_state_table(panel_raw, feature_df.index)
        labels_df, label_meta = _build_market_state_targets(
            panel_raw,
            feature_df.index,
            state_df=state_df,
            policies=policy_grid,
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        state_cols = list(state_df.columns)
        dataset = pd.concat([feature_df[feature_cols], state_df, labels_df], axis=1).replace([np.inf, -np.inf], np.nan)
        day_counts = dataset.groupby(level=0)[feature_cols[0]].count()
        good_days = day_counts[day_counts >= int(args.min_names_per_day)].index
        dataset = dataset.loc[dataset.index.get_level_values(0).isin(good_days)].copy()

        dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
        valid_mask = _mask(dt_idx, VALID_START, VALID_END)
        test_mask = _mask(dt_idx, TEST_START, TEST_END) if mode == "full" else None
        if not valid_mask.any():
            raise RuntimeError("empty 2023 validation split")
        if mode == "full" and (test_mask is None or not test_mask.any()):
            raise RuntimeError("empty 2024-2026 test split")
        target_modes = [c for c in label_meta]
        train_mask_by_target = {
            target: _active_sample_mask(dataset.index, TRAIN_START, TRAIN_END, int(label_meta[target]["max_horizon_days"]))
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
            test_mask=test_mask,
            feature_cols=feature_cols,
            target_modes=target_modes,
            alpha_grid=alpha_grid,
            weight_grid=weight_grid,
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

        _write_csv(paths["candidates_csv"], candidate_rows)
        _write_csv(paths["validation_backtests_csv"], valid_bt_rows)
        selected_report.to_csv(paths["selected_valid_report_csv"])
        selected_signal.reset_index().to_csv(paths["selected_valid_signal_csv"], index=False)

        expected_test_rows = 0
        test_metric: Optional[Dict[str, Any]] = None
        test_slice_rows: List[Dict[str, Any]] = []
        test_finite_gate_errors: List[str] = []
        if mode == "full":
            expected_test_rows = _count_calendar_rows(provider_uri, TEST_START, TEST_END)
            selected_test_signal = predictions[str(selected["candidate_id"])]["test"].rename("score").to_frame("score").sort_index()
            test_bt_metric, test_report = _run_backtest_with_report(
                selected_test_signal,
                "test_2024_2026_one_shot",
                str(selected["candidate_id"]),
                TEST_START,
                TEST_END,
                int(selected["topk"]),
                int(selected["n_drop"]),
                port_cfg,
                benchmark,
                float(args.open_cost),
                float(args.close_cost),
                exchange_cache,
            )
            test_metric = asdict(test_bt_metric)
            if test_report.empty:
                test_finite_gate_errors.append(f"empty test report: {test_metric['error']}")
            else:
                test_report.to_csv(paths["selected_test_report_csv"])
                selected_test_signal.reset_index().to_csv(paths["selected_test_signal_csv"], index=False)
                test_slice_rows = _slice_report_metrics(test_report)
                _write_csv(paths["test_slices_csv"], test_slice_rows)
            if (
                test_bt_metric.error
                or test_bt_metric.row_count != expected_test_rows
                or test_bt_metric.finite_rows != expected_test_rows
                or test_bt_metric.nonfinite_rows != 0
            ):
                test_finite_gate_errors.append(
                    f"{selected['candidate_id']} test: error={test_bt_metric.error!r} rows={test_bt_metric.row_count} finite={test_bt_metric.finite_rows} expected={expected_test_rows}"
                )

        finite_gate_pass = bool(not finite_gate_errors)
        status, verdict = _selection_status(
            _safe_float(selected["ir"]),
            _safe_float(selected["annret"]),
            finite_gate_pass,
        )
        full_hard_gate = None
        if mode == "full":
            if test_metric is None:
                raise RuntimeError("full mode did not produce test metrics")
            full_hard_gate_pass = bool(
                not test_finite_gate_errors
                and _safe_float(test_metric["ir"]) > FULL_HARD_IR
                and _safe_float(test_metric["annret"]) > FULL_HARD_ANNRET
                and abs(_safe_float(test_metric["max_drawdown"])) <= FULL_HARD_MAX_DRAWDOWN_ABS
                and int(test_metric["finite_rows"]) == int(expected_test_rows)
                and int(test_metric["nonfinite_rows"]) == 0
            )
            full_hard_gate = {
                "passed": full_hard_gate_pass,
                "finite_rows_required": expected_test_rows,
                "finite_rows_actual": test_metric["finite_rows"],
                "nonfinite_rows_required": 0,
                "nonfinite_rows_actual": test_metric["nonfinite_rows"],
                "costed_ir_required_gt": FULL_HARD_IR,
                "costed_ir_actual": test_metric["ir"],
                "costed_annret_required_gt": FULL_HARD_ANNRET,
                "costed_annret_actual": test_metric["annret"],
                "max_drawdown_abs_required_lte": FULL_HARD_MAX_DRAWDOWN_ABS,
                "max_drawdown_actual": test_metric["max_drawdown"],
                "selection_locked_from_validation_only": True,
                "selection_candidate_id": str(selected["candidate_id"]),
                "selection_topk": int(selected["topk"]),
                "selection_n_drop": int(selected["n_drop"]),
                "fail_closed_errors": test_finite_gate_errors,
            }
            status = "full_hard_gate_passed" if full_hard_gate_pass else "full_hard_gate_failed"
            verdict = "FULL_HARD_GATE_PASS" if full_hard_gate_pass else "NO_GO"

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
                    "full_mode": "locked one-shot final test only" if mode == "full" else "available via --mode full",
                    "test_window_if_full": [TEST_START, TEST_END],
                    "full_does_not_tune_using_2024_2026": True,
                    "model": "closed-form weighted ridge",
                    "market_state": state_meta,
                    "candidate_targets": label_meta,
                    "feature_count": int(len(feature_cols)),
                    "features": list(feature_cols),
                    "state_columns": state_cols,
                    "policy_grid": policy_grid,
                    "weight_grid": weight_grid,
                    "alpha_grid": alpha_grid,
                    "portfolio_grid": [{"topk": topk, "n_drop": ndrop} for topk, ndrop in combos],
                    "selection_rule": "2023 net-cost information ratio, tie by annualized return",
                    "acceptance": {
                        "investigate_costed_ir_min": INVESTIGATE_IR,
                        "promotion_costed_ir_min": PROMOTE_IR,
                        "promotion_annret_min": PROMOTE_ANNRET,
                    },
                    "promotion_gate": {
                        "costed_ir_min": PROMOTE_IR,
                        "costed_annret_min": PROMOTE_ANNRET,
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
                "test_metrics": None
                if mode != "full"
                else {
                    "costed_ir": test_metric["ir"] if test_metric else None,
                    "costed_annret": test_metric["annret"] if test_metric else None,
                    "max_drawdown": test_metric["max_drawdown"] if test_metric else None,
                    "turnover": test_metric["turnover"] if test_metric else None,
                    "row_count": test_metric["row_count"] if test_metric else None,
                    "finite_rows": test_metric["finite_rows"] if test_metric else None,
                    "nonfinite_rows": test_metric["nonfinite_rows"] if test_metric else None,
                    "expected_test_trading_rows": expected_test_rows,
                    "yearly_slices": test_slice_rows,
                },
                "yearly_slices": test_slice_rows if mode == "full" else [],
                "full_hard_gate": full_hard_gate,
                "finite_row_check": {
                    "passed": finite_gate_pass,
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
                    f"# Market-State Conditional Target ({stamp})",
                    "",
                    f"- status: `{status}`",
                    f"- verdict: `{verdict}`",
                    f"- mode: `{mode}`",
                    f"- smoke_no_2024_2026_load_or_eval: `{mode == 'smoke' and data_end == VALID_END}`",
                    f"- selected_candidate: `{selected['candidate_id']}`",
                    f"- selected_rule: `topk={int(selected['topk'])}, n_drop={int(selected['n_drop'])}`",
                    f"- 2023 costed IR / AnnRet / turnover: `{_safe_float(selected['ir']):.6f}` / `{_safe_float(selected['annret']):.6f}` / `{_safe_float(selected['turnover']):.6f}`",
                    *(
                        [
                            f"- 2024-2026 costed IR / AnnRet / turnover: `{_safe_float(test_metric['ir']):.6f}` / `{_safe_float(test_metric['annret']):.6f}` / `{_safe_float(test_metric['turnover']):.6f}`",
                            f"- 2024-2026 max_drawdown: `{_safe_float(test_metric['max_drawdown']):.6f}`",
                            f"- 2024-2026 finite_rows: `{int(test_metric['finite_rows'])}` / `{expected_test_rows}`",
                            f"- 2024-2026 nonfinite_rows: `{int(test_metric['nonfinite_rows'])}` / `0`",
                            f"- 2024-2026 hard-gate thresholds (IR, AnnRet, |MDD|): `> {FULL_HARD_IR}` / `> {FULL_HARD_ANNRET}` / `<= {FULL_HARD_MAX_DRAWDOWN_ABS}`",
                            f"- full_hard_gate_passed: `{bool(full_hard_gate and full_hard_gate['passed'])}`",
                        ]
                        if mode == "full" and test_metric is not None
                        else []
                    ),
                    f"- finite_rows: `{int(selected['finite_rows'])}` / `{expected_valid_rows}`",
                    f"- validation_nonfinite_rows: `{int(selected['nonfinite_rows'])}`",
                    f"- fail_closed_finite_gate_passed: `{finite_gate_pass}`",
                    f"- investigate_gate: `IR >= {INVESTIGATE_IR}`",
                    f"- promotion_gate: `IR >= {PROMOTE_IR}, AnnRet >= {PROMOTE_ANNRET}`",
                    f"- runtime_sec: `{summary['runtime_sec_total']:.3f}`",
                    f"- summary_json: `{paths['summary_json']}`",
                ]
            ),
            encoding="utf-8",
        )
        if mode == "full":
            return 0 if verdict == "FULL_HARD_GATE_PASS" else 2
        return 0 if verdict == "PROMOTE" else 2
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
                    f"# Market-State Conditional Target ({stamp})",
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

