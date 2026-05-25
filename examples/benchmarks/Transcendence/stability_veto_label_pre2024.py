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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_master
from quant_master.backtest import backtest as run_backtest
from quant_master.backtest import get_exchange
from quant_master.contrib.evaluate import risk_analysis
from quant_master.contrib.strategy.signal_strategy import TopkDropoutStrategy


EXPERIMENT_NAME = "pre-2023 stability-veto label family"
RAW_START = "2019-01-01"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2026-04-30"
TRAIN_YEARS = ("2020", "2021", "2022")
BASE_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change")
INVESTIGATE_IR = 1.80
PROMOTE_IR = 2.30
PROMOTE_ANNRET = 0.16
FULL_HARD_IR = 2.90
FULL_HARD_ANNRET = 0.27
FULL_HARD_MAX_DRAWDOWN_ABS = 0.25
LOCKED_CANDIDATE_ID = "ridge_stab_vol_scaled_net_excess_10d_rank_trend_quality_rank10_a10"
LOCKED_LABEL_NAME = "vol_scaled_net_excess_10d_rank"
LOCKED_FEATURE_SET = "trend_quality_rank10"
LOCKED_ALPHA = 10.0
LOCKED_TOPK = 40
LOCKED_N_DROP = 3


@dataclass(frozen=True)
class CandidateMetric:
    candidate_id: str
    model_family: str
    label_name: str
    feature_set: str
    alpha: float
    horizon_days: int
    feature_count: int
    train_sample_count: int
    fit_sec: float
    train_rank_ic: float
    train_rank_ic_ir: float
    train_rank_ic_2020: float
    train_rank_ic_2021: float
    train_rank_ic_2022: float
    stability_veto_pass: bool
    stability_veto_reason: str
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


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return _load_pickle(path)


def _extract_port_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(config.get("port_analysis_config"), dict):
        return copy.deepcopy(config["port_analysis_config"])
    task_cfg = config.get("task", {})
    for rec in task_cfg.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            rec_cfg = rec.get("kwargs", {}).get("config")
            if isinstance(rec_cfg, dict):
                return copy.deepcopy(rec_cfg)
    raise KeyError("cannot find port_analysis_config")


def _resolve_workflow_config(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = THIS_DIR / path
    return path.resolve()


def _read_calendar(provider_uri: Path) -> pd.DatetimeIndex:
    cal_path = provider_uri / "calendars" / "day.txt"
    vals = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return pd.to_datetime(pd.Index(vals))


def _count_calendar_rows(provider_uri: Path, start: str, end: str) -> int:
    idx = _read_calendar(provider_uri)
    return int(((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))).sum())


def _parse_instrument_intervals(inst_path: Path) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    out: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        out.setdefault(parts[0].lower(), []).append((pd.Timestamp(parts[1]), pd.Timestamp(parts[2])))
    return out


def _read_feature_bin(path: Path, n_cal: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    raw = np.fromfile(path, dtype="<f4")
    if raw.size <= 1:
        return None
    start_idx = int(raw[0])
    vals = raw[1:]
    arr = np.full(n_cal, np.nan, dtype=np.float32)
    s = max(0, start_idx)
    e = min(n_cal, start_idx + vals.size)
    if s < e:
        arr[s:e] = vals[(s - start_idx) : (e - start_idx)]
    return arr


def _interval_active_mask(dates: pd.DatetimeIndex, intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]]) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    for st, ed in intervals:
        mask |= (dates >= st) & (dates <= ed)
    return mask


def _build_panel(
    provider_uri: Path,
    market: str,
    raw_start: str,
    end_date: str,
    fields: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cal = _read_calendar(provider_uri)
    raw_mask = (cal >= pd.Timestamp(raw_start)) & (cal <= pd.Timestamp(end_date))
    if not raw_mask.any():
        raise RuntimeError(f"calendar has no overlap with {raw_start}..{end_date}")
    cal_sub = cal[raw_mask]
    start_pos = int(np.flatnonzero(raw_mask)[0])
    end_pos = int(np.flatnonzero(raw_mask)[-1])
    intervals = _parse_instrument_intervals(provider_uri / "instruments" / f"{market.lower()}.txt")

    frames: List[pd.DataFrame] = []
    coverage: List[Dict[str, Any]] = []
    for inst in sorted(intervals):
        feat_dir = provider_uri / "features" / inst
        arr_map: Dict[str, np.ndarray] = {}
        missing: List[str] = []
        for field in fields:
            arr = _read_feature_bin(feat_dir / f"{field}.day.bin", len(cal))
            if arr is None:
                arr = np.full(len(cal), np.nan, dtype=np.float32)
                missing.append(field)
            arr_map[field] = arr[start_pos : end_pos + 1]
        active = _interval_active_mask(cal_sub, intervals[inst])
        if not active.any():
            continue
        df_i = pd.DataFrame({k: v for k, v in arr_map.items()}, index=cal_sub)
        df_i = df_i.loc[active]
        df_i["instrument"] = inst.upper()
        frames.append(df_i)
        idx = cal_sub[active]
        coverage.append(
            {
                "instrument": inst.upper(),
                "first_date": str(idx.min().date()),
                "last_date": str(idx.max().date()),
                "rows_active": int(active.sum()),
                "rows_train": int(((idx >= pd.Timestamp(TRAIN_START)) & (idx <= pd.Timestamp(TRAIN_END))).sum()),
                "rows_valid": int(((idx >= pd.Timestamp(VALID_START)) & (idx <= pd.Timestamp(VALID_END))).sum()),
                "rows_all_fields_nonnull": int(
                    np.isfinite(np.column_stack([arr_map[f][active] for f in fields])).all(axis=1).sum()
                ),
                "missing_fields": ";".join(missing),
            }
        )
    if not frames:
        raise RuntimeError("no rows constructed from local feature bins")
    panel = pd.concat(frames, axis=0)
    panel.index.name = "datetime"
    panel = panel.reset_index().set_index(["datetime", "instrument"]).sort_index()
    return panel, pd.DataFrame(coverage).sort_values("instrument").reset_index(drop=True)


def _by_inst_pct(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level=1, sort=False).pct_change(w, fill_method=None)


def _by_inst_shift(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level=1, sort=False).shift(w)


def _by_inst_roll_mean(s: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    return s.groupby(level=1, sort=False).rolling(w, min_periods=minp or max(2, w // 3)).mean().reset_index(level=0, drop=True)


def _by_inst_roll_std(s: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    return s.groupby(level=1, sort=False).rolling(w, min_periods=minp or max(3, w // 3)).std().reset_index(level=0, drop=True)


def _by_inst_roll_min(s: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    return s.groupby(level=1, sort=False).rolling(w, min_periods=minp or max(2, w // 3)).min().reset_index(level=0, drop=True)


def _by_inst_roll_max(s: pd.Series, w: int, minp: Optional[int] = None) -> pd.Series:
    return s.groupby(level=1, sort=False).rolling(w, min_periods=minp or max(2, w // 3)).max().reset_index(level=0, drop=True)


def _cs_rank_pct(s: pd.Series) -> pd.Series:
    return s.groupby(level=0, sort=False).rank(method="average", pct=True)


def _cs_z(s: pd.Series, clip: float = 6.0) -> pd.Series:
    mu = s.groupby(level=0, sort=False).transform("mean")
    sd = s.groupby(level=0, sort=False).transform("std")
    return ((s - mu) / (sd + 1e-12)).clip(-clip, clip).fillna(0.0)


def _mask(index: pd.Index, start: str, end: str) -> np.ndarray:
    dt = pd.to_datetime(index)
    return (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))


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


def _build_features(panel: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    p = panel.copy()
    for col in BASE_FIELDS:
        p[col] = pd.to_numeric(p[col], errors="coerce").astype(float)

    factor = p["factor"].replace(0.0, np.nan).fillna(1.0)
    close = p["close"] * factor
    open_ = p["open"] * factor
    high = p["high"] * factor
    low = p["low"] * factor
    vwap = p["vwap"] * factor
    volume = p["volume"].clip(lower=0.0)
    amount = p["amount"].clip(lower=0.0)

    ret1 = _by_inst_pct(close, 1)
    ret2 = _by_inst_pct(close, 2)
    ret5 = _by_inst_pct(close, 5)
    ret10 = _by_inst_pct(close, 10)
    ret20 = _by_inst_pct(close, 20)
    ret60 = _by_inst_pct(close, 60)
    log_vol = np.log1p(volume)
    log_amt = np.log1p(amount)
    vol_chg1 = _by_inst_pct(volume.replace(0.0, np.nan), 1)
    vol_chg5 = _by_inst_pct(volume.replace(0.0, np.nan), 5)
    amt_chg5 = _by_inst_pct(amount.replace(0.0, np.nan), 5)
    intraday = close / (open_ + 1e-12) - 1.0
    overnight = open_ / (_by_inst_shift(close, 1) + 1e-12) - 1.0
    hl_range = (high - low) / (_by_inst_shift(close, 1).abs() + 1e-12)
    vwap_gap = close / (vwap + 1e-12) - 1.0
    vol10 = _by_inst_roll_std(ret1, 10)
    vol20 = _by_inst_roll_std(ret1, 20)
    vol60 = _by_inst_roll_std(ret1, 60)
    roll_min20 = _by_inst_roll_min(close, 20)
    roll_max20 = _by_inst_roll_max(close, 20)
    roll_max60 = _by_inst_roll_max(close, 60)
    recovery20 = close / (roll_min20 + 1e-12) - 1.0
    dd60 = close / (roll_max60 + 1e-12) - 1.0
    price_pos20 = (close - roll_min20) / (roll_max20 - roll_min20 + 1e-12)
    mkt_ret = ret1.groupby(level=0).mean()
    mkt_ret_s = pd.Series(mkt_ret.reindex(ret1.index.get_level_values(0)).values, index=ret1.index)
    market_mom20 = _by_inst_roll_mean(mkt_ret_s, 20)

    raw_features: Dict[str, pd.Series] = {
        "rev_1": -ret1,
        "rev_5": -ret5,
        "mom_10": ret10,
        "mom_20": ret20,
        "mom_60": ret60,
        "mom_spread_5_20": ret5 - ret20,
        "ret2": ret2,
        "intraday": intraday,
        "overnight": overnight,
        "vwap_gap": vwap_gap,
        "hl_range": hl_range,
        "vol_comp_10_60": -(vol10 / (vol60 + 1e-12)),
        "vol_exp_10_20": vol10 / (vol20 + 1e-12),
        "liq_volume_z20": (log_vol - _by_inst_roll_mean(log_vol, 20)) / (_by_inst_roll_std(log_vol, 20) + 1e-12),
        "liq_amount_z20": (log_amt - _by_inst_roll_mean(log_amt, 20)) / (_by_inst_roll_std(log_amt, 20) + 1e-12),
        "liq_volume_shock_5": vol_chg5,
        "liq_amount_shock_5": amt_chg5,
        "vp_div_20": _by_inst_roll_mean(ret1, 20) - _by_inst_roll_mean(vol_chg1, 20),
        "price_pos20": price_pos20,
        "dd60": dd60,
        "recovery20": recovery20,
        "mn_excess_ret20": _by_inst_roll_mean(ret1 - mkt_ret_s, 20),
        "mn_market_neutral_mom20": ret20 - market_mom20,
        "raw_change": p["change"],
        "vol20": vol20,
        "inv_vol20": 1.0 / (vol20 + 1e-4),
    }

    out = pd.DataFrame(index=p.index)
    all_rank_cols: List[str] = []
    for name, series in raw_features.items():
        clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[f"{name}__z"] = _cs_z(clean)
        out[f"{name}__rank"] = (_cs_rank_pct(clean).fillna(0.5) - 0.5) * 2.0
        all_rank_cols.append(f"{name}__rank")

    feature_sets = {
        "core_rank12": [
            "rev_1__rank",
            "rev_5__rank",
            "mom_10__rank",
            "mom_20__rank",
            "mom_spread_5_20__rank",
            "intraday__rank",
            "overnight__rank",
            "vwap_gap__rank",
            "vol_comp_10_60__rank",
            "liq_amount_z20__rank",
            "price_pos20__rank",
            "mn_excess_ret20__rank",
        ],
        "defensive_rank10": [
            "rev_1__rank",
            "rev_5__rank",
            "vol_comp_10_60__rank",
            "vol_exp_10_20__rank",
            "liq_volume_z20__rank",
            "liq_amount_z20__rank",
            "liq_amount_shock_5__rank",
            "dd60__rank",
            "recovery20__rank",
            "inv_vol20__rank",
        ],
        "trend_quality_rank10": [
            "mom_10__rank",
            "mom_20__rank",
            "mom_60__rank",
            "mom_spread_5_20__rank",
            "intraday__rank",
            "vwap_gap__rank",
            "vp_div_20__rank",
            "price_pos20__rank",
            "mn_market_neutral_mom20__rank",
            "raw_change__rank",
        ],
    }
    feature_sets = {name: [c for c in cols if c in out.columns] for name, cols in feature_sets.items()}
    if any(len(cols) < 6 for cols in feature_sets.values()):
        feature_sets["fallback_rank24"] = [c for c in all_rank_cols if c in out.columns][:24]
    return out.replace([np.inf, -np.inf], np.nan), feature_sets


def _build_stability_labels(
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
        excess = (raw_ret - mkt_ret).reindex(feature_index).replace([np.inf, -np.inf], np.nan)
        vol20 = _by_inst_roll_std(_by_inst_pct(close, 1), 20).reindex(feature_index).replace([np.inf, -np.inf], np.nan)

        label_defs = {
            f"net_excess_{h}d_rank": excess - round_trip_cost,
            f"vol_scaled_net_excess_{h}d_rank": (excess - round_trip_cost) / (vol20 + 1e-4),
        }
        active_mask = _active_sample_mask(feature_index, TRAIN_START, VALID_END, h)
        train_mask = _active_sample_mask(feature_index, TRAIN_START, TRAIN_END, h)
        for label_name, raw_label in label_defs.items():
            raw_label = pd.to_numeric(raw_label, errors="coerce").replace([np.inf, -np.inf], np.nan)
            labels[label_name] = (_cs_rank_pct(raw_label) - 0.5) * 2.0
            labels.loc[~active_mask, label_name] = np.nan
            train_vals = raw_label.loc[train_mask].dropna()
            meta[label_name] = {
                "horizon_days": h,
                "round_trip_cost": round_trip_cost,
                "train_sample_count": int(len(train_vals)),
                "label_formula": label_name,
                "train_exit_guard": f"training samples require horizon exit <= {TRAIN_END}",
                "smoke_future_guard": f"smoke panel ends at {VALID_END}; late-2023 labels without in-window exits are NaN",
            }
    return labels.replace([np.inf, -np.inf], np.nan), meta


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


def _year_ic_map(pred: pd.Series, label: pd.Series) -> Dict[str, float]:
    daily = _daily_rank_ic_series(pred, label)
    out: Dict[str, float] = {}
    for year in TRAIN_YEARS:
        year_s = daily.loc[(daily.index >= pd.Timestamp(f"{year}-01-01")) & (daily.index <= pd.Timestamp(f"{year}-12-31"))]
        out[year] = float(year_s.mean()) if len(year_s) else float("nan")
    return out


def _make_predictions(
    dataset: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    label_meta: Dict[str, Any],
    alpha_grid: Sequence[float],
    min_train_samples: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, pd.Series]]]:
    candidate_rows: List[Dict[str, Any]] = []
    predictions: Dict[str, Dict[str, pd.Series]] = {}
    dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
    valid_mask = _mask(dt_idx, VALID_START, VALID_END)
    valid_df = dataset.loc[valid_mask]

    for label_name, meta in label_meta.items():
        h = int(meta["horizon_days"])
        train_mask = _active_sample_mask(dataset.index, TRAIN_START, TRAIN_END, h)
        for feature_set, feature_cols in feature_sets.items():
            if len(feature_cols) < 2:
                continue
            train_df = dataset.loc[train_mask].dropna(subset=list(feature_cols) + [label_name])
            if len(train_df) < int(min_train_samples):
                reason = f"train_sample_count {len(train_df)} < min_train_samples {min_train_samples}"
                for alpha in alpha_grid:
                    cid = f"ridge_stab_{label_name}_{feature_set}_a{float(alpha):g}"
                    candidate_rows.append(
                        asdict(
                            CandidateMetric(
                                candidate_id=cid,
                                model_family="closed_form_ridge",
                                label_name=label_name,
                                feature_set=feature_set,
                                alpha=float(alpha),
                                horizon_days=h,
                                feature_count=int(len(feature_cols)),
                                train_sample_count=int(len(train_df)),
                                fit_sec=0.0,
                                train_rank_ic=float("nan"),
                                train_rank_ic_ir=float("nan"),
                                train_rank_ic_2020=float("nan"),
                                train_rank_ic_2021=float("nan"),
                                train_rank_ic_2022=float("nan"),
                                stability_veto_pass=False,
                                stability_veto_reason=reason,
                                valid_rank_ic=float("nan"),
                                valid_rank_ic_ir=float("nan"),
                            )
                        )
                    )
                continue

            x_train = train_df[list(feature_cols)].astype(np.float64).values
            y_train = train_df[label_name].astype(np.float64).values
            valid_for_pred = valid_df.dropna(subset=list(feature_cols))
            x_valid = valid_for_pred[list(feature_cols)].astype(np.float64).values
            for alpha in alpha_grid:
                cid = f"ridge_stab_{label_name}_{feature_set}_a{float(alpha):g}"
                coef, mu, sd, fit_sec = _fit_ridge(x_train, y_train, float(alpha))
                pred_train = _cs_z(pd.Series(_predict_ridge(x_train, coef, mu, sd), index=train_df.index, name="score"))
                pred_valid_part = _cs_z(
                    pd.Series(_predict_ridge(x_valid, coef, mu, sd), index=valid_for_pred.index, name="score")
                )
                pred_valid = pd.Series(index=valid_df.index, dtype=float, name="score")
                pred_valid.loc[pred_valid_part.index] = pred_valid_part
                train_ic_s = _daily_rank_ic_series(pred_train, train_df[label_name])
                valid_ic_s = _daily_rank_ic_series(pred_valid, valid_df[label_name])
                train_ic, train_ic_ir = _mean_and_ir(train_ic_s)
                valid_ic, valid_ic_ir = _mean_and_ir(valid_ic_s)
                year_ic = _year_ic_map(pred_train, train_df[label_name])
                year_vals = [year_ic[y] for y in TRAIN_YEARS]
                stability_pass = bool(all(np.isfinite(v) and v >= 0.0 for v in year_vals))
                if stability_pass:
                    reason = "pass: train-year rank IC is nonnegative for 2020, 2021, and 2022"
                else:
                    reason = "veto: one or more train-year rank IC values are negative or nonfinite"
                candidate_rows.append(
                    asdict(
                        CandidateMetric(
                            candidate_id=cid,
                            model_family="closed_form_ridge",
                            label_name=label_name,
                            feature_set=feature_set,
                            alpha=float(alpha),
                            horizon_days=h,
                            feature_count=int(len(feature_cols)),
                            train_sample_count=int(len(train_df)),
                            fit_sec=fit_sec,
                            train_rank_ic=train_ic,
                            train_rank_ic_ir=train_ic_ir,
                            train_rank_ic_2020=year_ic["2020"],
                            train_rank_ic_2021=year_ic["2021"],
                            train_rank_ic_2022=year_ic["2022"],
                            stability_veto_pass=stability_pass,
                            stability_veto_reason=reason,
                            valid_rank_ic=valid_ic,
                            valid_rank_ic_ir=valid_ic_ir,
                        )
                    )
                )
                if stability_pass:
                    predictions[cid] = {"train": pred_train, "valid": pred_valid}
    return candidate_rows, predictions


def _make_locked_full_prediction(
    dataset: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    label_meta: Dict[str, Any],
    locked_candidate_id: str,
) -> Tuple[Dict[str, Any], pd.Series, pd.Series]:
    feature_cols = feature_sets[LOCKED_FEATURE_SET]
    h = int(label_meta[LOCKED_LABEL_NAME]["horizon_days"])
    train_mask = _active_sample_mask(dataset.index, TRAIN_START, TRAIN_END, h)
    dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
    valid_mask = _mask(dt_idx, VALID_START, VALID_END)
    test_mask = _mask(dt_idx, TEST_START, TEST_END)
    train_df = dataset.loc[train_mask].dropna(subset=list(feature_cols) + [LOCKED_LABEL_NAME])
    valid_df = dataset.loc[valid_mask].dropna(subset=list(feature_cols))
    test_df = dataset.loc[test_mask].dropna(subset=list(feature_cols))
    if train_df.empty:
        raise RuntimeError(f"empty locked train split for {locked_candidate_id}")
    if valid_df.empty:
        raise RuntimeError(f"empty locked validation signal split for {locked_candidate_id}")
    if test_df.empty:
        raise RuntimeError(f"empty locked test signal split for {locked_candidate_id}")

    x_train = train_df[list(feature_cols)].astype(np.float64).values
    y_train = train_df[LOCKED_LABEL_NAME].astype(np.float64).values
    coef, mu, sd, fit_sec = _fit_ridge(x_train, y_train, LOCKED_ALPHA)
    pred_train = _cs_z(pd.Series(_predict_ridge(x_train, coef, mu, sd), index=train_df.index, name="score"))
    pred_valid = _cs_z(
        pd.Series(
            _predict_ridge(valid_df[list(feature_cols)].astype(np.float64).values, coef, mu, sd),
            index=valid_df.index,
            name="score",
        )
    )
    pred_test = _cs_z(
        pd.Series(
            _predict_ridge(test_df[list(feature_cols)].astype(np.float64).values, coef, mu, sd),
            index=test_df.index,
            name="score",
        )
    )
    train_ic_s = _daily_rank_ic_series(pred_train, train_df[LOCKED_LABEL_NAME])
    valid_label = dataset.loc[valid_mask, LOCKED_LABEL_NAME]
    valid_ic_s = _daily_rank_ic_series(pred_valid, valid_label)
    train_ic, train_ic_ir = _mean_and_ir(train_ic_s)
    valid_ic, valid_ic_ir = _mean_and_ir(valid_ic_s)
    year_ic = _year_ic_map(pred_train, train_df[LOCKED_LABEL_NAME])
    year_vals = [year_ic[y] for y in TRAIN_YEARS]
    stability_pass = bool(all(np.isfinite(v) and v >= 0.0 for v in year_vals))
    candidate = asdict(
        CandidateMetric(
            candidate_id=locked_candidate_id,
            model_family="closed_form_ridge",
            label_name=LOCKED_LABEL_NAME,
            feature_set=LOCKED_FEATURE_SET,
            alpha=LOCKED_ALPHA,
            horizon_days=h,
            feature_count=int(len(feature_cols)),
            train_sample_count=int(len(train_df)),
            fit_sec=fit_sec,
            train_rank_ic=train_ic,
            train_rank_ic_ir=train_ic_ir,
            train_rank_ic_2020=year_ic["2020"],
            train_rank_ic_2021=year_ic["2021"],
            train_rank_ic_2022=year_ic["2022"],
            stability_veto_pass=stability_pass,
            stability_veto_reason=(
                "pass: train-year rank IC is nonnegative for 2020, 2021, and 2022"
                if stability_pass
                else "veto: one or more train-year rank IC values are negative or nonfinite"
            ),
            valid_rank_ic=valid_ic,
            valid_rank_ic_ir=valid_ic_ir,
        )
    )
    return candidate, pred_valid, pred_test


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
        finite_cols = report[["return", "bench", "cost", "turnover"]].to_numpy(dtype=float)
        finite_rows = int(np.isfinite(finite_cols).all(axis=1).sum())
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


def _metrics_from_report(report: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    if report.empty:
        return {
            "split": split_name,
            "start": "",
            "end": "",
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-2023 stability-veto label family quick-smoke.")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
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
    p.add_argument("--min-train-samples", type=int, default=50000)
    p.add_argument("--output-prefix", default="stability_veto_label_pre2024")
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
    provider_uri = Path(args.provider_uri).expanduser().resolve()
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
            "train_stability_window": [TRAIN_START, TRAIN_END],
            "train_stability_years": list(TRAIN_YEARS),
            "selection_window": [VALID_START, VALID_END],
            "smoke_load_end": VALID_END,
            "smoke_evaluates_2024_2026": False,
            "full_mode_available": True,
            "full_test_window": [TEST_START, TEST_END],
            "full_locked_candidate_id": LOCKED_CANDIDATE_ID,
        },
    }

    try:
        quant_master.init(provider_uri=str(provider_uri), region="cn")
        wf_cfg = _load_config(_resolve_workflow_config(str(args.workflow_config)))
        port_cfg = _extract_port_config(wf_cfg)
        benchmark = str(wf_cfg.get("benchmark", "SH000300"))

        panel_raw, coverage_df = _build_panel(provider_uri, str(args.market), RAW_START, data_end, BASE_FIELDS)
        feature_df, feature_sets = _build_features(panel_raw)
        labels_df, label_meta = _build_stability_labels(
            panel_raw,
            feature_df.index,
            horizons=(5, 10),
            open_cost=float(args.open_cost),
            close_cost=float(args.close_cost),
        )
        all_feature_cols = sorted({col for cols in feature_sets.values() for col in cols})
        dataset = pd.concat([feature_df[all_feature_cols], labels_df], axis=1).replace([np.inf, -np.inf], np.nan)
        day_counts = dataset.groupby(level=0)[all_feature_cols[0]].count()
        good_days = day_counts[day_counts >= int(args.min_names_per_day)].index
        dataset = dataset.loc[dataset.index.get_level_values(0).isin(good_days)].copy()

        dt_idx = pd.to_datetime(dataset.index.get_level_values(0))
        if not _mask(dt_idx, VALID_START, VALID_END).any():
            raise RuntimeError("empty 2023 validation split")
        alpha_grid = [float(x) for x in str(args.alpha_grid).split(",") if x.strip()]
        topk_grid = [int(x) for x in str(args.topk_grid).split(",") if x.strip()]
        ndrop_grid = [int(x) for x in str(args.ndrop_grid).split(",") if x.strip()]
        if mode == "full":
            alpha_grid = [LOCKED_ALPHA]
            topk_grid = [LOCKED_TOPK]
            ndrop_grid = [LOCKED_N_DROP]
            feature_sets = {LOCKED_FEATURE_SET: feature_sets[LOCKED_FEATURE_SET]}
            label_meta = {LOCKED_LABEL_NAME: label_meta[LOCKED_LABEL_NAME]}
        combos = [(topk, ndrop) for topk in topk_grid for ndrop in ndrop_grid if ndrop < topk]
        if not combos:
            raise RuntimeError("no valid topk/n_drop combinations")

        candidate_rows, predictions = _make_predictions(
            dataset=dataset,
            feature_sets=feature_sets,
            label_meta=label_meta,
            alpha_grid=alpha_grid,
            min_train_samples=int(args.min_train_samples),
        )
        survivors = [r for r in candidate_rows if bool(r.get("stability_veto_pass"))]
        _write_csv(paths["candidates_csv"], candidate_rows)

        expected_valid_rows = _count_calendar_rows(provider_uri, VALID_START, VALID_END)
        valid_bt_rows: List[Dict[str, Any]] = []
        valid_reports: Dict[Tuple[str, int, int], pd.DataFrame] = {}
        finite_gate_errors: List[str] = []
        selected: Optional[Dict[str, Any]] = None
        selected_candidate: Optional[Dict[str, Any]] = None
        selected_report = pd.DataFrame()
        selected_signal = pd.DataFrame()
        test_metric: Optional[Dict[str, Any]] = None
        test_report = pd.DataFrame()
        test_signal = pd.DataFrame()
        test_slice_rows: List[Dict[str, Any]] = []

        if not survivors:
            status, verdict = "no_stability_survivors", "NO_GO"
            summary.update(
                {
                    "status": status,
                    "verdict": verdict,
                    "blocker": "no candidates survived the pre-2023 stability veto",
                    "provider_uri": str(provider_uri),
                    "market": str(args.market),
                    "benchmark": benchmark,
                    "costs": {"open_cost": float(args.open_cost), "close_cost": float(args.close_cost)},
                    "candidate_counts": {
                        "total": int(len(candidate_rows)),
                        "stability_survivors": 0,
                        "vetoed": int(len(candidate_rows)),
                    },
                    "stability_veto_diagnostics": {
                        "rule": "candidate train-year rank IC must be finite and nonnegative in 2020, 2021, and 2022",
                        "selected_candidate_year_ics_nonnegative": False,
                        "no_survivor_note": "No 2023 portfolio validation was run because the pre-2023 gate failed closed.",
                    },
                    "finite_row_check": {
                        "passed": False,
                        "expected_validation_trading_rows": expected_valid_rows,
                        "fail_closed_errors": ["no validation reports: no candidates survived stability veto"],
                    },
                    "runtime_sec_total": float(time.perf_counter() - t0_all),
                }
            )
            _write_csv(paths["validation_backtests_csv"], valid_bt_rows)
            _write_json(paths["summary_json"], summary)
            paths["summary_md"].write_text(
                "\n".join(
                    [
                        f"# Pre-2023 Stability-Veto Label Family ({stamp})",
                        "",
                        "- status: `no_stability_survivors`",
                        "- verdict: `NO_GO`",
                        "- blocker: `no candidates survived the pre-2023 stability veto`",
                        f"- candidates_total: `{len(candidate_rows)}`",
                        f"- summary_json: `{paths['summary_json']}`",
                    ]
                ),
                encoding="utf-8",
            )
            return 2

        exchange_cache: Dict[Tuple[str, str, float, float, float, str], Any] = {}
        for cand in survivors:
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

        _write_csv(paths["validation_backtests_csv"], valid_bt_rows)
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
        if mode == "full":
            locked_ok = (
                str(selected["candidate_id"]) == LOCKED_CANDIDATE_ID
                and int(selected["topk"]) == LOCKED_TOPK
                and int(selected["n_drop"]) == LOCKED_N_DROP
            )
            if not locked_ok:
                raise RuntimeError(f"full mode selected unexpected portfolio: {selected}")
        selected_candidate = next(r for r in candidate_rows if r["candidate_id"] == selected["candidate_id"])
        selected_key = (str(selected["candidate_id"]), int(selected["topk"]), int(selected["n_drop"]))
        selected_report = valid_reports[selected_key]
        selected_signal = predictions[str(selected["candidate_id"])]["valid"].rename("score").to_frame("score").sort_index()
        selected_report.to_csv(paths["selected_valid_report_csv"])
        selected_signal.reset_index().to_csv(paths["selected_valid_signal_csv"], index=False)

        expected_test_rows = 0
        test_finite_gate_errors: List[str] = []
        if mode == "full":
            expected_test_rows = _count_calendar_rows(provider_uri, TEST_START, TEST_END)
            _, _locked_valid_signal, locked_test_pred = _make_locked_full_prediction(
                dataset=dataset,
                feature_sets=feature_sets,
                label_meta=label_meta,
                locked_candidate_id=LOCKED_CANDIDATE_ID,
            )
            test_signal = locked_test_pred.rename("score").to_frame("score").sort_index()
            test_bt_metric, test_report = _run_backtest_with_report(
                test_signal,
                "test_2024_2026_one_shot",
                LOCKED_CANDIDATE_ID,
                TEST_START,
                TEST_END,
                LOCKED_TOPK,
                LOCKED_N_DROP,
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
                test_signal.reset_index().to_csv(paths["selected_test_signal_csv"], index=False)
                test_slice_rows = _slice_report_metrics(test_report)
                _write_csv(paths["test_slices_csv"], test_slice_rows)
            if (
                test_bt_metric.error
                or test_bt_metric.row_count != expected_test_rows
                or test_bt_metric.finite_rows != expected_test_rows
                or test_bt_metric.nonfinite_rows != 0
            ):
                test_finite_gate_errors.append(
                    f"{LOCKED_CANDIDATE_ID} test: error={test_bt_metric.error!r} rows={test_bt_metric.row_count} finite={test_bt_metric.finite_rows} expected={expected_test_rows}"
                )

        selected_year_ics = [
            _safe_float(selected_candidate["train_rank_ic_2020"]),
            _safe_float(selected_candidate["train_rank_ic_2021"]),
            _safe_float(selected_candidate["train_rank_ic_2022"]),
        ]
        selected_stability_ok = bool(all(np.isfinite(v) and v >= 0.0 for v in selected_year_ics))
        finite_gate_pass = bool(not finite_gate_errors and selected_stability_ok)
        status, verdict = _selection_status(_safe_float(selected["ir"]), _safe_float(selected["annret"]), finite_gate_pass)
        full_hard_gate = None
        if mode == "full":
            if test_metric is None:
                raise RuntimeError("full mode did not produce test metrics")
            max_dd_abs = abs(_safe_float(test_metric["max_drawdown"]))
            full_hard_gate_pass = bool(
                not test_finite_gate_errors
                and selected_stability_ok
                and _safe_float(test_metric["ir"]) > FULL_HARD_IR
                and _safe_float(test_metric["annret"]) > FULL_HARD_ANNRET
                and np.isfinite(max_dd_abs)
                and max_dd_abs <= FULL_HARD_MAX_DRAWDOWN_ABS
            )
            full_hard_gate = {
                "passed": full_hard_gate_pass,
                "finite_rows_required": expected_test_rows,
                "finite_rows_actual": test_metric["finite_rows"],
                "nonfinite_rows": test_metric["nonfinite_rows"],
                "costed_ir_required_gt": FULL_HARD_IR,
                "costed_annret_required_gt": FULL_HARD_ANNRET,
                "max_drawdown_abs_required_lte": FULL_HARD_MAX_DRAWDOWN_ABS,
                "max_drawdown_abs_actual": max_dd_abs,
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
                    "raw_start_for_past_features_only": RAW_START,
                    "data_end_loaded": data_end,
                    "train_and_stability": [TRAIN_START, TRAIN_END],
                    "train_stability_years": list(TRAIN_YEARS),
                    "validation_selection_only": [VALID_START, VALID_END],
                    "smoke_no_2024_2026_load_or_eval": mode == "smoke" and data_end == VALID_END,
                    "full_mode": "locked one-shot final test only" if mode == "full" else "available via --mode full",
                    "test_window_if_full": [TEST_START, TEST_END],
                    "full_does_not_tune_using_2024_2026": mode == "full",
                    "full_locked_candidate": {
                        "candidate_id": LOCKED_CANDIDATE_ID,
                        "label_name": LOCKED_LABEL_NAME,
                        "feature_set": LOCKED_FEATURE_SET,
                        "alpha": LOCKED_ALPHA,
                        "topk": LOCKED_TOPK,
                        "n_drop": LOCKED_N_DROP,
                    },
                    "model": "closed-form ridge only",
                    "candidate_labels": label_meta,
                    "feature_sets": feature_sets,
                    "alpha_grid": alpha_grid,
                    "portfolio_grid": [{"topk": topk, "n_drop": ndrop} for topk, ndrop in combos],
                    "selection_rule": "2023 net-cost information ratio, tie by annualized return",
                    "investigate_gate": {"costed_ir_min": INVESTIGATE_IR},
                    "promotion_gate": {"costed_ir_min": PROMOTE_IR, "costed_annret_min": PROMOTE_ANNRET},
                },
                "dataset_shape": {
                    "rows_after_good_day_filter": int(len(dataset)),
                    "coverage_rows": int(len(coverage_df)),
                    "good_days": int(len(good_days)),
                },
                "candidate_counts": {
                    "total": int(len(candidate_rows)),
                    "stability_survivors": int(len(survivors)),
                    "vetoed": int(len(candidate_rows) - len(survivors)),
                    "validation_backtests": int(len(valid_bt_rows)),
                },
                "stability_veto_diagnostics": {
                    "rule": "candidate train-year rank IC must be finite and nonnegative in 2020, 2021, and 2022",
                    "candidate_list_artifact": str(paths["candidates_csv"]),
                    "selected_candidate_year_ics": {
                        "2020": selected_candidate["train_rank_ic_2020"],
                        "2021": selected_candidate["train_rank_ic_2021"],
                        "2022": selected_candidate["train_rank_ic_2022"],
                    },
                    "selected_candidate_year_ics_nonnegative": selected_stability_ok,
                    "survivor_candidate_ids": [str(r["candidate_id"]) for r in survivors],
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
                    f"# Pre-2023 Stability-Veto Label Family ({stamp})",
                    "",
                    f"- status: `{status}`",
                    f"- verdict: `{verdict}`",
                    f"- mode: `{mode}`",
                    f"- smoke_no_2024_2026_load_or_eval: `{mode == 'smoke' and data_end == VALID_END}`",
                    f"- full_locked_candidate: `{LOCKED_CANDIDATE_ID if mode == 'full' else 'n/a'}`",
                    f"- candidates_total/survivors: `{len(candidate_rows)}` / `{len(survivors)}`",
                    f"- selected_candidate: `{selected['candidate_id']}`",
                    f"- selected_rule: `topk={int(selected['topk'])}, n_drop={int(selected['n_drop'])}`",
                    f"- selected_train_year_rank_ic: `2020={selected_year_ics[0]:.6f}, 2021={selected_year_ics[1]:.6f}, 2022={selected_year_ics[2]:.6f}`",
                    f"- selected_train_year_rank_ic_nonnegative: `{selected_stability_ok}`",
                    f"- 2023 costed IR / AnnRet / turnover: `{_safe_float(selected['ir']):.6f}` / `{_safe_float(selected['annret']):.6f}` / `{_safe_float(selected['turnover']):.6f}`",
                    *(
                        [
                            f"- 2024-2026 costed IR / AnnRet / turnover: `{_safe_float(test_metric['ir']):.6f}` / `{_safe_float(test_metric['annret']):.6f}` / `{_safe_float(test_metric['turnover']):.6f}`",
                            f"- 2024-2026 max_drawdown: `{_safe_float(test_metric['max_drawdown']):.6f}`",
                            f"- 2024-2026 finite_rows: `{int(test_metric['finite_rows'])}` / `{expected_test_rows}`",
                            f"- 2024-2026 nonfinite_rows: `{int(test_metric['nonfinite_rows'])}`",
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
        return 0 if verdict in {"PROMOTE", "INVESTIGATE"} else 2
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
                    f"# Pre-2023 Stability-Veto Label Family ({stamp})",
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
