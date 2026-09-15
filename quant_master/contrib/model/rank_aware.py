# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


RANK_TARGET_MODE = "cs_rank"


def normalize_target_mode(target_mode: str) -> str:
    mode = str(target_mode or "raw").strip().lower()
    aliases = {"rank": RANK_TARGET_MODE, "rank_ic": RANK_TARGET_MODE, "cross_sectional_rank": RANK_TARGET_MODE}
    mode = aliases.get(mode, mode)
    if mode not in {"raw", RANK_TARGET_MODE}:
        raise ValueError("target_mode must be 'raw' or 'cs_rank'.")
    return mode


def single_label_series(label: pd.DataFrame) -> pd.Series:
    if not isinstance(label, pd.DataFrame) or label.shape[1] != 1:
        raise ValueError("LightGBM only supports single-label training.")
    return label.iloc[:, 0].astype(float)


def prepare_tree_target(label: pd.DataFrame, target_mode: str) -> pd.Series:
    target = single_label_series(label)
    if normalize_target_mode(target_mode) == "raw":
        return target
    if not isinstance(target.index, pd.MultiIndex):
        raise ValueError("target_mode='cs_rank' requires a MultiIndex with a datetime level.")
    date_level = "datetime" if "datetime" in target.index.names else target.index.names[0]
    grouped = target.groupby(level=date_level, sort=False)
    ranked = grouped.rank(method="average")
    counts = grouped.transform("count").astype(float)
    denominator = (counts - 1.0).where(counts > 1.0, 1.0)
    centered = (ranked - (counts + 1.0) / 2.0) / denominator
    return centered.where(counts > 1.0, 0.0)


class DailyRankICEval:
    """LightGBM evaluation callable for mean daily cross-sectional Spearman IC."""

    def __init__(self, index_by_dataset_id: Dict[int, pd.Index]):
        self.index_by_dataset_id = dict(index_by_dataset_id)

    def __call__(self, predictions, dataset):
        index = self.index_by_dataset_id.get(id(dataset))
        if index is None:
            raise ValueError("No row index registered for the LightGBM evaluation dataset.")
        if len(index) != len(predictions):
            raise ValueError("LightGBM predictions and registered evaluation index have different lengths.")
        score = pd.Series(np.asarray(predictions, dtype=float), index=index)
        label = pd.Series(np.asarray(dataset.get_label(), dtype=float), index=index)
        rank_ic = mean_daily_rank_ic(score, label)
        return "rank_ic", rank_ic, True


def mean_daily_rank_ic(score: pd.Series, label: pd.Series) -> float:
    if not isinstance(score.index, pd.MultiIndex):
        raise ValueError("Daily RankIC requires a MultiIndex with a datetime level.")
    date_level = "datetime" if "datetime" in score.index.names else score.index.names[0]
    frame = pd.DataFrame({"score": score, "label": label.reindex(score.index)}).replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame.dropna()
    if frame.empty:
        return 0.0

    grouped = frame.groupby(level=date_level, sort=False)
    score_rank = grouped["score"].rank(method="average")
    label_rank = grouped["label"].rank(method="average")
    score_centered = score_rank - score_rank.groupby(level=date_level, sort=False).transform("mean")
    label_centered = label_rank - label_rank.groupby(level=date_level, sort=False).transform("mean")
    numerator = (score_centered * label_centered).groupby(level=date_level, sort=False).sum()
    score_norm = score_centered.pow(2).groupby(level=date_level, sort=False).sum()
    label_norm = label_centered.pow(2).groupby(level=date_level, sort=False).sum()
    denominator = np.sqrt(score_norm * label_norm)
    counts = grouped.size()
    daily_ic = (numerator / denominator).where((counts >= 2) & (denominator > 0)).dropna()
    return float(daily_ic.mean()) if not daily_ic.empty else 0.0
