# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Union

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
    return pd.to_numeric(values, errors="coerce")


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
