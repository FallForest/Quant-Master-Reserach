# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd


HoldingsLike = Optional[Union[pd.Series, Mapping, Iterable]]


def transform_scores_for_cost(
    scores: pd.Series,
    previous_holdings: HoldingsLike = None,
    volatility: Optional[pd.Series] = None,
    previous_scores: Optional[pd.Series] = None,
    *,
    previous_holding_boost: float = 0.0,
    turnover_penalty: float = 0.0,
    volatility_penalty: float = 0.0,
    smoothing_alpha: float = 0.0,
    normalize_scores: bool = True,
    use_holding_weight: bool = False,
) -> pd.Series:
    """
    Apply a cost/turnover-aware score transform for top-k style selection.

    This utility only uses information available at decision time:
    current scores, previous holdings/scores, and optional contemporaneous
    (or lagged) volatility proxy. It does not consume future returns/labels.

    Parameters
    ----------
    scores
        Current cross-sectional scores.
    previous_holdings
        Holdings from previous rebalance step. Can be a Series (weight or
        binary), a dict-like mapping, or an iterable of held instruments.
    volatility
        Optional volatility proxy aligned by index. Penalty is applied using
        cross-sectional rank (higher vol -> larger penalty).
    previous_scores
        Optional previous-step scores for temporal smoothing.
    previous_holding_boost
        Additive boost for previously held instruments.
    turnover_penalty
        Additive penalty for instruments not held in previous holdings.
    volatility_penalty
        Penalty coefficient applied to ranked volatility.
    smoothing_alpha
        Blend factor in [0, 1]: new = (1-a) * score + a * previous_score.
    normalize_scores
        Whether to rank-normalize scores in each cross section.
    use_holding_weight
        If True and previous holdings have numeric weight, boost scales with
        relative holding weight; otherwise uses binary held/not-held.
    """
    if not isinstance(scores, pd.Series):
        raise TypeError("scores must be a pandas Series")
    if not 0.0 <= float(smoothing_alpha) <= 1.0:
        raise ValueError("smoothing_alpha must be in [0, 1]")

    base = _to_numeric_series(scores).copy()
    if base.isna().all():
        return base.fillna(0.0)
    if base.isna().any():
        base = base.fillna(float(base.min()))
    if normalize_scores:
        base = _cross_section_rank(base)

    if previous_scores is not None and smoothing_alpha > 0:
        prev = _to_numeric_series(previous_scores).reindex(base.index)
        prev = prev.fillna(float(base.mean()))
        if normalize_scores:
            prev = _cross_section_rank(prev)
        alpha = float(smoothing_alpha)
        base = (1.0 - alpha) * base + alpha * prev

    holding_signal = _holding_signal(previous_holdings, base.index, use_holding_weight=use_holding_weight)
    if previous_holding_boost:
        base = base + float(previous_holding_boost) * holding_signal
    if turnover_penalty:
        base = base - float(turnover_penalty) * (1.0 - holding_signal)

    if volatility is not None and volatility_penalty:
        vol = _to_numeric_series(volatility).reindex(base.index)
        vol = vol.abs()
        if vol.isna().all():
            vol = pd.Series(0.0, index=base.index)
        else:
            vol = vol.fillna(float(vol.median()))
        vol_rank = _cross_section_rank(vol)
        base = base - float(volatility_penalty) * vol_rank

    return base.astype(float)


def select_buffered_topk(
    scores: pd.Series,
    topk: int,
    previous_holdings: HoldingsLike = None,
    *,
    rank_buffer: int = 0,
) -> pd.Index:
    """
    Select top-k names with a past-only turnover buffer.

    Previously held instruments are retained when their current score rank is
    within ``topk + rank_buffer``. This keeps the original score order inside
    retained and replacement buckets, and uses only current scores plus
    previous holdings available at decision time.
    """
    if not isinstance(scores, pd.Series):
        raise TypeError("scores must be a pandas Series")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if rank_buffer < 0:
        raise ValueError("rank_buffer must be non-negative")

    ranked = _finite_scores(scores).sort_values(ascending=False, kind="mergesort")
    if ranked.empty:
        return pd.Index([])

    topk = min(int(topk), len(ranked))
    if rank_buffer == 0 or previous_holdings is None:
        return pd.Index(ranked.iloc[:topk].index)

    holding_signal = _holding_signal(previous_holdings, ranked.index, use_holding_weight=False)
    rank_limit = min(len(ranked), topk + int(rank_buffer))

    selected = []
    selected_set = set()
    for instrument in ranked.iloc[:rank_limit].index:
        if holding_signal.get(instrument, 0.0) <= 0 or instrument in selected_set:
            continue
        selected.append(instrument)
        selected_set.add(instrument)
        if len(selected) >= topk:
            break

    for instrument in ranked.index:
        if len(selected) >= topk:
            break
        if instrument in selected_set:
            continue
        selected.append(instrument)
        selected_set.add(instrument)

    return pd.Index([instrument for instrument in ranked.index if instrument in selected_set][:topk])


def transform_scores_with_rank_buffer(
    scores: pd.Series,
    previous_holdings: HoldingsLike = None,
    *,
    topk: int,
    rank_buffer: int = 0,
) -> pd.Series:
    """
    Return finite rank scores after applying ``select_buffered_topk``.

    The output preserves the input index and converts the buffered selection
    into rank-like scores: selected names rank above non-selected names, while
    original score order is preserved inside each bucket. With
    ``rank_buffer=0``, the full cross-sectional rank order is unchanged.
    """
    if not isinstance(scores, pd.Series):
        raise TypeError("scores must be a pandas Series")

    clean = _finite_scores(scores)
    if clean.empty:
        return clean.astype(float)

    ranked = clean.sort_values(ascending=False, kind="mergesort")
    selected = select_buffered_topk(
        clean,
        topk=topk,
        previous_holdings=previous_holdings,
        rank_buffer=rank_buffer,
    )
    selected_set = set(selected.tolist())
    selected_order = [instrument for instrument in ranked.index if instrument in selected_set]
    other_order = [instrument for instrument in ranked.index if instrument not in selected_set]
    buffered_order = selected_order + other_order

    adjusted = pd.Series(index=clean.index, dtype=float)
    for rank, instrument in enumerate(buffered_order):
        adjusted.loc[instrument] = float(len(buffered_order) - rank)
    return adjusted.astype(float)


def rerank_topk_with_turnover_limit(
    adjusted_scores: pd.Series,
    topk: int,
    previous_holdings: HoldingsLike = None,
    max_new_names: Optional[int] = None,
) -> pd.Index:
    """
    Select top-k names with optional cap on newly-entered names.

    Parameters
    ----------
    adjusted_scores
        Score after transformation. Higher is better.
    topk
        Number of names to select.
    previous_holdings
        Previous holdings used to define "new" names.
    max_new_names
        Maximum number of names not in previous_holdings.
        If None, no turnover cap is enforced.
    """
    if not isinstance(adjusted_scores, pd.Series):
        raise TypeError("adjusted_scores must be a pandas Series")
    if topk <= 0:
        raise ValueError("topk must be positive")

    scores = _to_numeric_series(adjusted_scores).dropna()
    if len(scores) <= topk and max_new_names is None:
        return scores.sort_values(ascending=False).index
    if max_new_names is None:
        return scores.nlargest(min(topk, len(scores))).index

    max_new = int(max_new_names)
    if max_new < 0:
        raise ValueError("max_new_names must be non-negative")

    holding_signal = _holding_signal(previous_holdings, scores.index, use_holding_weight=False)
    is_held = holding_signal > 0
    ranked = scores.sort_values(ascending=False)

    selected = []
    held_ranked = ranked[is_held.reindex(ranked.index).fillna(False)]
    keep_quota = max(0, topk - max_new)
    if keep_quota > 0 and len(held_ranked) > 0:
        selected.extend(held_ranked.iloc[:keep_quota].index.tolist())

    new_used = 0
    selected_set = set(selected)
    for instrument in ranked.index:
        if instrument in selected_set:
            continue
        current_is_new = not bool(is_held.get(instrument, False))
        if current_is_new and new_used >= max_new:
            continue
        selected.append(instrument)
        selected_set.add(instrument)
        if current_is_new:
            new_used += 1
        if len(selected) >= topk:
            break

    if len(selected) < topk:
        for instrument in ranked.index:
            if instrument not in selected_set:
                selected.append(instrument)
                selected_set.add(instrument)
                if len(selected) >= topk:
                    break

    return pd.Index(selected[:topk])


def estimate_turnover(selected: Union[pd.Index, Iterable], previous_holdings: HoldingsLike = None) -> float:
    """
    Estimate one-step name turnover ratio in [0, 1].
    """
    selected_set = set(selected if not isinstance(selected, pd.Index) else selected.tolist())
    prev_signal = _holding_signal(previous_holdings, pd.Index(list(selected_set)), use_holding_weight=False)
    prev_set = set(prev_signal[prev_signal > 0].index.tolist())
    if not selected_set:
        return 0.0
    overlap = len(selected_set.intersection(prev_set))
    return float(1.0 - overlap / max(len(selected_set), 1))


def _to_numeric_series(values: pd.Series) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("Expected one score column when DataFrame is passed.")
        values = values.iloc[:, 0]
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric), pd.NA)


def _finite_scores(values: pd.Series) -> pd.Series:
    clean = _to_numeric_series(values).copy()
    if clean.empty:
        return clean.astype(float)
    if clean.isna().all():
        return pd.Series(0.0, index=clean.index, dtype=float)
    return clean.fillna(float(clean.min(skipna=True))).astype(float)


def _cross_section_rank(values: pd.Series) -> pd.Series:
    if not isinstance(values.index, pd.MultiIndex):
        return values.rank(pct=True).fillna(0.0)
    date_level = "datetime" if "datetime" in values.index.names else values.index.names[0]
    return values.groupby(level=date_level).rank(pct=True).fillna(0.0)


def _holding_signal(previous_holdings: HoldingsLike, index: pd.Index, use_holding_weight: bool) -> pd.Series:
    if previous_holdings is None:
        return pd.Series(0.0, index=index)

    if isinstance(previous_holdings, pd.Series):
        raw = pd.to_numeric(previous_holdings, errors="coerce")
    elif isinstance(previous_holdings, Mapping):
        raw = pd.to_numeric(pd.Series(dict(previous_holdings)), errors="coerce")
    else:
        raw = pd.Series(1.0, index=pd.Index(list(previous_holdings)))

    raw = raw[raw > 0]
    if raw.empty:
        return pd.Series(0.0, index=index)

    if use_holding_weight:
        max_weight = float(raw.max())
        if max_weight <= 0:
            signal = pd.Series(0.0, index=raw.index)
        else:
            signal = (raw / max_weight).clip(lower=0.0, upper=1.0)
    else:
        signal = pd.Series(1.0, index=raw.index)

    return signal.reindex(index).fillna(0.0).astype(float)
