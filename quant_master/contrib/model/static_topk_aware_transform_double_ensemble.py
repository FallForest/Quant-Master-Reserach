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


class StaticTopKAwareTransformDEnsembleModel(Model):
    """DoubleEnsemble with a static validation-selected top-k-aware score transform."""

    def __init__(
        self,
        double_ensemble_kwargs: Optional[Dict] = None,
        topk: int = 50,
        turnover_penalty: float = 0.0007,
        gamma_grid: Optional[Sequence[float]] = None,
        uplift_grid: Optional[Sequence[float]] = None,
        tau_grid: Optional[Sequence[float]] = None,
        use_rank_score: bool = True,
        random_state: int = 42,
        eps: float = 1e-6,
    ):
        if topk <= 0:
            raise ValueError("topk must be positive.")

        self.logger = get_module_logger("StaticTopKAwareTransformDEnsembleModel")
        self.topk = int(topk)
        self.turnover_penalty = float(turnover_penalty)
        self.gamma_grid = list(gamma_grid or [0.8, 1.0, 1.2, 1.4])
        self.uplift_grid = list(uplift_grid or [0.0, 0.01, 0.02, 0.03])
        self.tau_grid = list(tau_grid or [0.01, 0.02, 0.05])
        self.use_rank_score = bool(use_rank_score)
        self.random_state = int(random_state)
        self.eps = float(eps)

        self.double_ensemble_kwargs = double_ensemble_kwargs or {}
        self.double_ensemble_kwargs.setdefault("random_state", random_state)
        self.model = DEnsembleModel(**self.double_ensemble_kwargs)

        self.best_gamma = 1.0
        self.best_uplift = 0.0
        self.best_tau = 0.02

    def fit(self, dataset: DatasetH):
        self.model.fit(dataset)
        valid_score = self.model.predict(dataset, "valid")
        valid_label = dataset.prepare("valid", col_set=["label"], data_key=DataHandlerLP.DK_L)["label"]

        valid_score, valid_label = self._align_prediction_and_label(valid_score, valid_label)
        label = pd.Series(self._squeeze_label(valid_label), index=valid_label.index)

        if self.use_rank_score:
            valid_score = self._cross_sectional_rank(valid_score)

        self.best_gamma, self.best_uplift, self.best_tau = self._search_transform_params(valid_score, label)
        self.logger.info(
            "Selected transform params: gamma=%s uplift=%s tau=%s",
            self.best_gamma,
            self.best_uplift,
            self.best_tau,
        )

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        score = self.model.predict(dataset, segment).fillna(0.0)
        if self.use_rank_score:
            score = self._cross_sectional_rank(score)

        return self._apply_transform(score, self.best_gamma, self.best_uplift, self.best_tau)

    def _search_transform_params(self, score: pd.Series, label: pd.Series):
        best_obj = None
        best = (1.0, 0.0, 0.02)

        for gamma in self.gamma_grid:
            for uplift in self.uplift_grid:
                for tau in self.tau_grid:
                    transformed = self._apply_transform(score, float(gamma), float(uplift), float(tau))
                    obj = self._topk_objective(transformed, label)
                    if best_obj is None or obj > best_obj:
                        best_obj = obj
                        best = (float(gamma), float(uplift), float(tau))
        return best

    def _apply_transform(self, score: pd.Series, gamma: float, uplift: float, tau: float) -> pd.Series:
        date_level = self._date_level(score.index)
        parts = []
        safe_tau = max(float(tau), self.eps)

        for _, daily_score in score.groupby(level=date_level, sort=True):
            s = daily_score.fillna(0.0).clip(lower=self.eps, upper=1.0 - self.eps)
            base = s.pow(gamma)

            if uplift > 0.0 and len(base) > 0:
                k = min(self.topk, len(base))
                threshold = float(np.partition(base.values, -k)[-k])
                edge = 1.0 / (1.0 + np.exp(-(base.values - threshold) / safe_tau))
                base = pd.Series(base.values + uplift * edge, index=base.index)

            parts.append(base)

        if not parts:
            return score.copy()
        return pd.concat(parts).sort_index()

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

    def _topk_index(self, score: pd.Series) -> pd.Index:
        if len(score) <= self.topk:
            return score.index
        selected_pos = np.argpartition(score.values, -self.topk)[-self.topk :]
        return score.index[selected_pos]

    @staticmethod
    def _align_prediction_and_label(score: pd.Series, label_df: pd.DataFrame):
        common_index = score.index.intersection(label_df.index)
        if len(common_index) == 0:
            raise ValueError("Prediction index and label index do not overlap.")
        return score.loc[common_index], label_df.loc[common_index]

    @staticmethod
    def _cross_sectional_rank(values: pd.Series) -> pd.Series:
        if not isinstance(values.index, pd.MultiIndex):
            return values.rank(pct=True).fillna(0.0)
        date_level = StaticTopKAwareTransformDEnsembleModel._date_level(values.index)
        return values.groupby(level=date_level).rank(pct=True).fillna(0.0)

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return np.squeeze(label)
        raise ValueError("StaticTopKAwareTransformDEnsembleModel only supports single-label training.")

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("StaticTopKAwareTransformDEnsembleModel requires a MultiIndex prediction index.")
        return "datetime" if "datetime" in index.names else index.names[0]

    @staticmethod
    def _instrument_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("StaticTopKAwareTransformDEnsembleModel requires a MultiIndex prediction index.")
        return "instrument" if "instrument" in index.names else index.names[-1]
