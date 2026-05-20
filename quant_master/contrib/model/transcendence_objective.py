from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def build_objective_label_frame(
    label_df: pd.DataFrame,
    *,
    base_horizon: int,
    mode: str = "raw",
    horizon_days: Optional[Sequence[int]] = None,
    horizon_weights: Optional[Sequence[float]] = None,
    market_relative: bool = False,
    vol_adjust: bool = False,
    vol_window: int = 20,
    vol_floor: float = 1e-4,
    rank_power: float = 1.0,
    decile: float = 0.1,
    decile_scale: float = 0.0,
    clip: Optional[float] = None,
) -> pd.DataFrame:
    if label_df.empty:
        return label_df.copy()

    mode = str(mode or "raw").strip().lower()
    series = label_df.iloc[:, 0].astype(float)
    horizons = _normalize_horizons(base_horizon, horizon_days)
    weights = _normalize_weights(horizons, horizon_weights)

    target = pd.Series(0.0, index=series.index, dtype=float)
    for h, w in zip(horizons, weights):
        target = target + float(w) * _forward_window_mean_by_instrument(series, h)

    if market_relative:
        target = _cross_sectional_center(target)
    if vol_adjust:
        target = _instrument_vol_adjust(target, window=vol_window, floor=vol_floor)

    if mode in {"rank", "rank_decile_spread"}:
        rank_pct = _cross_sectional_rank_pct(target)
        rank_center = rank_pct - 0.5
        if rank_power != 1.0:
            rank_center = np.sign(rank_center) * np.power(np.abs(rank_center), float(rank_power))
        target = rank_center

    if mode in {"decile_spread", "rank_decile_spread"} and decile_scale > 0:
        rank_pct = _cross_sectional_rank_pct(target)
        q = min(max(float(decile), 1e-4), 0.49)
        top_flag = (rank_pct >= 1.0 - q).astype(float)
        bot_flag = (rank_pct <= q).astype(float)
        target = target + float(decile_scale) * (top_flag - bot_flag)

    target = target.replace([np.inf, -np.inf], np.nan)
    if clip is not None and float(clip) > 0:
        target = target.clip(-float(clip), float(clip))
    target = target.fillna(0.0)

    out = label_df.copy()
    source_col = out.columns[0]
    out.loc[:, source_col] = target.to_numpy(dtype=out[source_col].dtype, na_value=np.nan)
    return out


def _normalize_horizons(base_horizon: int, horizon_days: Optional[Sequence[int]]) -> Sequence[int]:
    if not horizon_days:
        return [max(int(base_horizon), 1)]
    uniq = []
    for h in horizon_days:
        h_int = int(h)
        if h_int <= 0:
            continue
        if h_int not in uniq:
            uniq.append(h_int)
    return uniq or [max(int(base_horizon), 1)]


def _normalize_weights(horizons: Sequence[int], weights: Optional[Sequence[float]]) -> np.ndarray:
    if not horizons:
        return np.array([1.0], dtype=float)
    if not weights:
        return np.ones(len(horizons), dtype=float) / float(len(horizons))
    arr = np.array([float(w) for w in weights[: len(horizons)]], dtype=float)
    if len(arr) < len(horizons):
        arr = np.pad(arr, (0, len(horizons) - len(arr)), constant_values=0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return np.ones(len(horizons), dtype=float) / float(len(horizons))
    return arr / total


def _forward_window_mean_by_instrument(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    if window <= 1:
        return series
    if not isinstance(series.index, pd.MultiIndex):
        return _forward_window_mean(series, window)
    inst_level = "instrument" if "instrument" in series.index.names else series.index.names[-1]
    return series.groupby(level=inst_level, group_keys=False).apply(lambda x: _forward_window_mean(x, window))


def _cross_sectional_center(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.MultiIndex):
        return series - float(series.mean())
    date_level = "datetime" if "datetime" in series.index.names else series.index.names[0]
    return series - series.groupby(level=date_level).transform("mean")


def _instrument_vol_adjust(series: pd.Series, window: int, floor: float) -> pd.Series:
    floor = max(float(floor), 1e-8)
    window = max(int(window), 2)
    if not isinstance(series.index, pd.MultiIndex):
        vol = series.shift(1).rolling(window=window, min_periods=max(3, window // 3)).std()
        vol = vol.fillna(float(series.std()) if np.isfinite(series.std()) else 1.0).clip(lower=floor)
        return series / vol
    inst_level = "instrument" if "instrument" in series.index.names else series.index.names[-1]
    vol = series.groupby(level=inst_level, group_keys=False).apply(
        lambda x: x.shift(1).rolling(window=window, min_periods=max(3, window // 3)).std()
    )
    fallback = float(series.std()) if np.isfinite(series.std()) and float(series.std()) > 0 else 1.0
    vol = vol.fillna(fallback).clip(lower=floor)
    return series / vol


def _cross_sectional_rank_pct(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.MultiIndex):
        return series.rank(pct=True)
    date_level = "datetime" if "datetime" in series.index.names else series.index.names[0]
    return series.groupby(level=date_level).rank(pct=True)


def _forward_window_mean(series: pd.Series, window: int) -> pd.Series:
    values = series.values.astype(float)
    n = len(values)
    if n == 0 or window <= 1:
        return series

    values_for_sum = np.nan_to_num(values, nan=0.0)
    valid = np.isfinite(values).astype(float)
    csum = np.cumsum(np.insert(values_for_sum, 0, 0.0))
    cvalid = np.cumsum(np.insert(valid, 0, 0.0))
    out = np.empty(n, dtype=float)

    for i in range(n):
        j = min(i + window, n)
        denom = cvalid[j] - cvalid[i]
        if denom <= 0:
            out[i] = np.nan
        else:
            out[i] = (csum[j] - csum[i]) / denom

    return pd.Series(out, index=series.index).fillna(series)
