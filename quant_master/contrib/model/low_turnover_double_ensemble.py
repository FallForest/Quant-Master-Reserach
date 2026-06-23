# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, Optional, Sequence, Text, Union

import numpy as np
import pandas as pd

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from .double_ensemble import DEnsembleModel


class LowTurnoverDEnsembleModel(Model):
    """DoubleEnsemble with validation-selected previous-holding score boost."""

    def __init__(
        self,
        double_ensemble_kwargs: Optional[Dict] = None,
        topk: int = 50,
        turnover_penalty: float = 0.0007,
        turnover_boost_grid: Optional[Sequence[float]] = None,
        use_rank_score: bool = True,
        random_state: int = 42,
    ):
        if topk <= 0:
            raise ValueError("topk must be positive.")

        self.logger = get_module_logger("LowTurnoverDEnsembleModel")
        self.topk = topk
        self.turnover_penalty = turnover_penalty
        self.turnover_boost_grid = list(turnover_boost_grid or [0.0, 0.01, 0.02, 0.03, 0.05])
        self.use_rank_score = use_rank_score
        self.random_state = random_state

        self.double_ensemble_kwargs = double_ensemble_kwargs or {}
        self.double_ensemble_kwargs.setdefault("random_state", random_state)
        self.model = DEnsembleModel(**self.double_ensemble_kwargs)
        self.turnover_boost = 0.0

    def fit(self, dataset: DatasetH):
        self.model.fit(dataset)
        valid_score = self._score(self.model.predict(dataset, "valid"))
        valid_label = dataset.prepare("valid", col_set=["label"], data_key=DataHandlerLP.DK_L)["label"]
        valid_score, valid_label = self._align_prediction_and_label(valid_score, valid_label)
        label = pd.Series(self._squeeze_label(valid_label), index=valid_label.index)
        self.turnover_boost = self._learn_turnover_boost(valid_score, label)
        self.logger.info("Turnover boost: %s", self.turnover_boost)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        score = self._score(self.model.predict(dataset, segment))
        if self.turnover_boost > 0:
            score = self._apply_turnover_boost(score, self.turnover_boost)
        return score

    def _learn_turnover_boost(self, score: pd.Series, label: pd.Series) -> float:
        best_boost = 0.0
        best_objective = None
        for boost in self.turnover_boost_grid:
            adjusted = self._apply_turnover_boost(score, float(boost)) if boost > 0 else score
            objective = self._topk_objective(adjusted, label)
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_boost = float(boost)
        return best_boost

    def _topk_objective(self, score: pd.Series, label: pd.Series) -> float:
        date_level = self._date_level(score.index)
        inst_level = self._instrument_level(score.index)

        daily_returns = []
        daily_turnovers = []
        prev_inst = None
        for _, daily_score in score.groupby(level=date_level, sort=True):
            daily_label = label.loc[daily_score.index]
            selected_index = self._topk_index(daily_score)
            daily_returns.append(float(daily_label.loc[selected_index].mean()))
            selected_inst = set(selected_index.get_level_values(inst_level))
            if prev_inst is not None:
                overlap = len(prev_inst.intersection(selected_inst))
                daily_turnovers.append(1.0 - overlap / max(len(selected_inst), 1))
            prev_inst = selected_inst

        if not daily_returns:
            return float("-inf")
        turnover = float(np.mean(daily_turnovers)) if daily_turnovers else 0.0
        return float(np.mean(daily_returns) - self.turnover_penalty * turnover)

    def _apply_turnover_boost(self, score: pd.Series, boost: float) -> pd.Series:
        date_level = self._date_level(score.index)
        inst_level = self._instrument_level(score.index)
        adjusted_parts = []
        prev_inst = set()
        for _, daily_score in score.groupby(level=date_level, sort=True):
            adjusted = daily_score.copy()
            if prev_inst:
                inst_values = adjusted.index.get_level_values(inst_level)
                adjusted.loc[inst_values.isin(prev_inst)] += boost
            selected_index = self._topk_index(adjusted)
            prev_inst = set(selected_index.get_level_values(inst_level))
            adjusted_parts.append(adjusted)
        if not adjusted_parts:
            return score.copy()
        return pd.concat(adjusted_parts).sort_index()

    def _score(self, score: pd.Series) -> pd.Series:
        score = score.fillna(0.0)
        if not self.use_rank_score:
            return score
        date_level = self._date_level(score.index)
        return score.groupby(level=date_level).rank(pct=True).fillna(0.0)

    def _topk_index(self, score: pd.Series) -> pd.Index:
        if len(score) <= self.topk:
            return score.index
        values = score.values
        selected_pos = np.argpartition(values, -self.topk)[-self.topk :]
        return score.index[selected_pos]

    @staticmethod
    def _align_prediction_and_label(score: pd.Series, label_df: pd.DataFrame):
        common_index = score.index.intersection(label_df.index)
        if len(common_index) == 0:
            raise ValueError("Prediction index and label index do not overlap.")
        return score.loc[common_index], label_df.loc[common_index]

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return np.squeeze(label)
        raise ValueError("LowTurnoverDEnsembleModel only supports single-label training.")

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("LowTurnoverDEnsembleModel requires a MultiIndex prediction index.")
        return "datetime" if "datetime" in index.names else index.names[0]

    @staticmethod
    def _instrument_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("LowTurnoverDEnsembleModel requires a MultiIndex prediction index.")
        return "instrument" if "instrument" in index.names else index.names[-1]
