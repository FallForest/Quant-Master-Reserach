# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Text, Union

import numpy as np
import pandas as pd

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from .double_ensemble import DEnsembleModel
from .gbdt import LGBModel
from .linear import LinearModel


class DynamicMetaEnsembleModel(Model):
    """Regime-gated blend of DoubleEnsemble, LightGBM, and Linear models.

    The gate is learned only from validation-period labels. Test-time regimes are
    inferred from the base models' cross-sectional prediction shape, so no future
    labels or benchmark returns are required.
    """

    def __init__(
        self,
        double_ensemble_kwargs: Optional[Dict] = None,
        lightgbm_kwargs: Optional[Dict] = None,
        linear_kwargs: Optional[Dict] = None,
        topk: int = 50,
        num_regimes: int = 3,
        search_step: float = 0.1,
        min_weight: float = 1e-6,
        turnover_penalty: float = 0.0007,
        turnover_boost_grid: Optional[Sequence[float]] = None,
        use_rank_score: bool = True,
        random_state: int = 42,
    ):
        if topk <= 0:
            raise ValueError("topk must be positive.")
        if num_regimes <= 0:
            raise ValueError("num_regimes must be positive.")
        if not 0 < search_step <= 1:
            raise ValueError("search_step must be in (0, 1].")

        self.logger = get_module_logger("DynamicMetaEnsembleModel")
        self.topk = topk
        self.num_regimes = num_regimes
        self.search_step = search_step
        self.min_weight = min_weight
        self.turnover_penalty = turnover_penalty
        self.turnover_boost_grid = list(turnover_boost_grid or [0.0])
        self.use_rank_score = use_rank_score
        self.random_state = random_state

        self.double_ensemble_kwargs = double_ensemble_kwargs or {}
        self.lightgbm_kwargs = lightgbm_kwargs or {}
        self.linear_kwargs = linear_kwargs or {}
        self.double_ensemble_kwargs.setdefault("random_state", random_state)
        self.lightgbm_kwargs.setdefault("seed", random_state)
        self.lightgbm_kwargs.setdefault("feature_fraction_seed", random_state)
        self.lightgbm_kwargs.setdefault("bagging_seed", random_state)
        self.lightgbm_kwargs.setdefault("data_random_seed", random_state)

        self.double_ensemble_model = DEnsembleModel(**self.double_ensemble_kwargs)
        self.lightgbm_model = LGBModel(**self.lightgbm_kwargs)
        self.linear_model = LinearModel(**self.linear_kwargs)

        self.model_order = ["double_ensemble", "lightgbm", "linear"]
        self.global_weights: Dict[str, float] = {}
        self.regime_weights: Dict[int, Dict[str, float]] = {}
        self.regime_thresholds: List[float] = []
        self.turnover_boost: float = 0.0

    def fit(self, dataset: DatasetH):
        self.logger.info("Training DoubleEnsemble sub-model.")
        self.double_ensemble_model.fit(dataset)

        self.logger.info("Training LightGBM sub-model.")
        self.lightgbm_model.fit(dataset)

        self.logger.info("Training Linear sub-model.")
        self.linear_model.fit(dataset)

        valid_pred_raw = self._predict_frame(dataset, "valid")
        valid_label = dataset.prepare("valid", col_set=["label"], data_key=DataHandlerLP.DK_L)["label"]
        valid_pred_raw, valid_label = self._align_prediction_and_label(valid_pred_raw, valid_label)
        label = self._squeeze_label(valid_label)
        label_s = pd.Series(label, index=valid_pred_raw.index, name="label")

        valid_pred = self._prepare_prediction_scores(valid_pred_raw)
        self.global_weights = self._learn_weights(valid_pred, label_s)
        self.regime_thresholds = self._fit_regime_thresholds(valid_pred_raw)
        regimes = self._assign_row_regimes(valid_pred_raw)
        self.regime_weights = self._learn_regime_weights(valid_pred, label_s, regimes)
        self.turnover_boost = self._learn_turnover_boost(valid_pred, label_s, regimes)

        self.logger.info("Global weights: %s", self.global_weights)
        self.logger.info("Regime thresholds: %s", self.regime_thresholds)
        self.logger.info("Regime weights: %s", self.regime_weights)
        self.logger.info("Turnover boost: %s", self.turnover_boost)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.global_weights:
            raise ValueError("model is not fitted yet!")
        pred_raw = self._predict_frame(dataset, segment)
        pred_scores = self._prepare_prediction_scores(pred_raw)
        regimes = self._assign_row_regimes(pred_raw)
        score = self._blend_by_regime(pred_scores, regimes)
        if self.turnover_boost > 0:
            score = self._apply_turnover_boost(score, self.turnover_boost)
        return score

    def _predict_frame(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.DataFrame:
        preds = {
            "double_ensemble": self.double_ensemble_model.predict(dataset, segment),
            "lightgbm": self.lightgbm_model.predict(dataset, segment),
            "linear": self.linear_model.predict(dataset, segment),
        }
        return pd.DataFrame(preds)

    def _prepare_prediction_scores(self, pred_frame: pd.DataFrame) -> pd.DataFrame:
        if self.use_rank_score:
            return self._cross_sectional_rank_score(pred_frame)
        return pred_frame.fillna(0.0)

    def _learn_regime_weights(
        self, pred_frame: pd.DataFrame, label: pd.Series, regimes: pd.Series
    ) -> Dict[int, Dict[str, float]]:
        result = {}
        for regime in range(self.num_regimes):
            mask = regimes == regime
            if mask.sum() < self.topk:
                result[regime] = self.global_weights
                continue
            result[regime] = self._learn_weights(pred_frame.loc[mask], label.loc[mask])
        return result

    def _learn_weights(self, pred_frame: pd.DataFrame, label: pd.Series) -> Dict[str, float]:
        best_score = None
        best_weights = None
        for weights in self._weight_grid():
            score = self._topk_objective(pred_frame, label, weights)
            if best_score is None or score > best_score:
                best_score = score
                best_weights = weights

        if best_weights is None:
            return {name: 1.0 / len(self.model_order) for name in self.model_order}

        best_weights = np.clip(best_weights, self.min_weight, None)
        best_weights = best_weights / best_weights.sum()
        return {name: float(weight) for name, weight in zip(self.model_order, best_weights)}

    def _learn_turnover_boost(self, pred_frame: pd.DataFrame, label: pd.Series, regimes: pd.Series) -> float:
        if len(self.turnover_boost_grid) == 1:
            return float(self.turnover_boost_grid[0])

        base_score = self._blend_by_regime(pred_frame, regimes)
        best_boost = 0.0
        best_score = None
        for boost in self.turnover_boost_grid:
            adjusted = self._apply_turnover_boost(base_score, float(boost)) if boost > 0 else base_score
            score = self._topk_series_objective(adjusted, label)
            if best_score is None or score > best_score:
                best_score = score
                best_boost = float(boost)
        return best_boost

    def _blend_by_regime(self, pred_frame: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        result = pd.Series(np.zeros(len(pred_frame), dtype=float), index=pred_frame.index)
        for regime in range(self.num_regimes):
            mask = regimes == regime
            if not mask.any():
                continue
            weights = self.regime_weights.get(regime, self.global_weights)
            result.loc[mask] = self._weighted_sum(pred_frame.loc[mask], weights)
        return result

    def _fit_regime_thresholds(self, pred_frame: pd.DataFrame) -> List[float]:
        daily_values = self._daily_regime_values(pred_frame)
        if daily_values.empty or self.num_regimes == 1:
            return []
        quantiles = np.linspace(0, 1, self.num_regimes + 1)[1:-1]
        thresholds = daily_values.quantile(quantiles).dropna().values
        return sorted(float(value) for value in np.unique(thresholds))

    def _assign_row_regimes(self, pred_frame: pd.DataFrame) -> pd.Series:
        if not self.regime_thresholds:
            return pd.Series(np.zeros(len(pred_frame), dtype=int), index=pred_frame.index)
        daily_values = self._daily_regime_values(pred_frame)
        daily_regimes = pd.Series(
            np.searchsorted(self.regime_thresholds, daily_values.values, side="right"),
            index=daily_values.index,
            dtype=int,
        ).clip(0, self.num_regimes - 1)
        date_level = self._date_level(pred_frame.index)
        row_dates = pred_frame.index.get_level_values(date_level)
        return pd.Series(daily_regimes.reindex(row_dates).fillna(0).values.astype(int), index=pred_frame.index)

    def _daily_regime_values(self, pred_frame: pd.DataFrame) -> pd.Series:
        date_level = self._date_level(pred_frame.index)
        pred_z = self._cross_sectional_zscore(pred_frame)
        equal_score = pred_z.mean(axis=1)
        signal_spread = equal_score.groupby(level=date_level).std().fillna(0.0)
        disagreement = pred_z.std(axis=1).groupby(level=date_level).mean().fillna(0.0)
        return (signal_spread + disagreement).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _weight_grid(self):
        grid = np.arange(0.0, 1.0 + 1e-9, self.search_step)
        for w1 in grid:
            for w2 in grid:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                yield np.array([w1, w2, max(0.0, w3)], dtype=float)

    def _topk_objective(self, pred_frame: pd.DataFrame, label: pd.Series, weights: np.ndarray) -> float:
        score = pd.Series(pred_frame.values @ weights, index=pred_frame.index)
        return self._topk_series_objective(score, label)

    def _topk_series_objective(self, score: pd.Series, label: pd.Series) -> float:
        common_index = score.index.intersection(label.index)
        score = score.loc[common_index]
        label = label.loc[common_index]
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

    def _topk_index(self, score: pd.Series) -> pd.Index:
        if len(score) <= self.topk:
            return score.index
        values = score.values
        selected_pos = np.argpartition(values, -self.topk)[-self.topk :]
        return score.index[selected_pos]

    def _weighted_sum(self, pred_frame: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
        weight_array = np.array([weights[name] for name in self.model_order], dtype=float)
        return pred_frame[self.model_order].values @ weight_array

    @staticmethod
    def _align_prediction_and_label(pred_frame: pd.DataFrame, label_df: pd.DataFrame):
        common_index = pred_frame.index.intersection(label_df.index)
        if len(common_index) == 0:
            raise ValueError("Prediction index and label index do not overlap.")
        return pred_frame.loc[common_index], label_df.loc[common_index]

    @staticmethod
    def _cross_sectional_rank_score(values: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(values.index, pd.MultiIndex):
            return values.rank(pct=True).fillna(0.0)
        date_level = DynamicMetaEnsembleModel._date_level(values.index)
        return values.groupby(level=date_level).rank(pct=True).fillna(0.0)

    @staticmethod
    def _cross_sectional_zscore(values: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(values.index, pd.MultiIndex):
            centered = values - values.mean()
            return (centered / (values.std() + 1e-12)).fillna(0.0)
        date_level = DynamicMetaEnsembleModel._date_level(values.index)
        grouped = values.groupby(level=date_level)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        return ((values - mean) / (std + 1e-12)).fillna(0.0)

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return np.squeeze(label)
        raise ValueError("DynamicMetaEnsembleModel only supports single-label training.")

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("DynamicMetaEnsembleModel requires a MultiIndex prediction index.")
        return "datetime" if "datetime" in index.names else index.names[0]

    @staticmethod
    def _instrument_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("DynamicMetaEnsembleModel requires a MultiIndex prediction index.")
        return "instrument" if "instrument" in index.names else index.names[-1]
