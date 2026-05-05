# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, Optional, Text, Union

import numpy as np
import pandas as pd

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from .double_ensemble import DEnsembleModel
from .gbdt import LGBModel
from .linear import LinearModel


class MetaEnsembleModel(Model):
    """Blend DEnsemble, LightGBM and Linear models with validation-learned weights."""

    def __init__(
        self,
        double_ensemble_kwargs: Optional[Dict] = None,
        lightgbm_kwargs: Optional[Dict] = None,
        linear_kwargs: Optional[Dict] = None,
        blend_method: str = "icir",
        min_weight: float = 1e-6,
        search_step: float = 0.05,
        topk: int = 50,
        use_rank_score: bool = True,
        random_state: int = 42,
    ):
        if blend_method not in {"icir", "lstsq", "topk"}:
            raise ValueError(f"Unsupported blend_method: {blend_method}")

        self.logger = get_module_logger("MetaEnsembleModel")
        self.blend_method = blend_method
        self.min_weight = min_weight
        self.search_step = search_step
        self.topk = topk
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
        self.model_weights: Dict[str, float] = {}

    def fit(self, dataset: DatasetH):
        self.logger.info("Training DoubleEnsemble sub-model.")
        self.double_ensemble_model.fit(dataset)

        self.logger.info("Training LightGBM sub-model.")
        self.lightgbm_model.fit(dataset)

        self.logger.info("Training Linear sub-model.")
        self.linear_model.fit(dataset)

        valid_pred = self._predict_valid(dataset)
        valid_label = dataset.prepare("valid", col_set=["label"], data_key=DataHandlerLP.DK_L)["label"]
        valid_pred, valid_label = self._align_prediction_and_label(valid_pred, valid_label)
        label = self._squeeze_label(valid_label)

        if self.blend_method == "icir":
            self.model_weights = self._learn_icir_weights(valid_pred, label)
        elif self.blend_method == "topk":
            self.model_weights = self._learn_topk_weights(valid_pred, label)
        else:
            self.model_weights = self._learn_lstsq_weights(valid_pred, label)
        self.logger.info("Model weights: %s", self.model_weights)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.model_weights:
            raise ValueError("model is not fitted yet!")
        pred_frame = self._predict_frame(dataset, segment)
        if self.use_rank_score:
            pred_frame = self._cross_sectional_rank_score(pred_frame)
        ensemble = np.zeros(len(pred_frame), dtype=float)
        for model_name in self.model_order:
            ensemble += pred_frame[model_name].values * self.model_weights[model_name]
        return pd.Series(ensemble, index=pred_frame.index)

    def _predict_valid(self, dataset: DatasetH) -> pd.DataFrame:
        return self._predict_frame(dataset, "valid")

    def _predict_frame(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.DataFrame:
        preds = {
            "double_ensemble": self.double_ensemble_model.predict(dataset, segment),
            "lightgbm": self.lightgbm_model.predict(dataset, segment),
            "linear": self.linear_model.predict(dataset, segment),
        }
        return pd.DataFrame(preds)

    @staticmethod
    def _align_prediction_and_label(pred_frame: pd.DataFrame, label_df: pd.DataFrame):
        common_index = pred_frame.index.intersection(label_df.index)
        if len(common_index) == 0:
            raise ValueError("Prediction index and label index do not overlap.")
        return pred_frame.loc[common_index], label_df.loc[common_index]

    def _learn_icir_weights(self, pred_frame: pd.DataFrame, label: np.ndarray) -> Dict[str, float]:
        label_s = pd.Series(label, index=pred_frame.index)
        icir_scores = {}
        for model_name in self.model_order:
            daily_ic = self._daily_rank_ic(pred_frame[model_name], label_s)
            mean = daily_ic.mean()
            std = daily_ic.std()
            icir = float(mean / std) if std and np.isfinite(std) and std > 0 else 0.0
            icir_scores[model_name] = max(icir, 0.0)

        if sum(icir_scores.values()) <= 0:
            return {name: 1.0 / len(self.model_order) for name in self.model_order}

        total = sum(icir_scores.values())
        return {name: score / total for name, score in icir_scores.items()}

    def _learn_lstsq_weights(self, pred_frame: pd.DataFrame, label: np.ndarray) -> Dict[str, float]:
        pred_norm = self._cross_sectional_zscore(pred_frame)
        label_norm = self._cross_sectional_zscore(pd.Series(label, index=pred_frame.index))
        design = pred_norm.values
        target = label_norm.values
        mask = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
        design = design[mask]
        target = target[mask]
        if design.size == 0:
            return {name: 1.0 / len(self.model_order) for name in self.model_order}
        weights, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        weights = np.clip(weights, self.min_weight, None)
        weights = weights / weights.sum()
        return {name: float(weight) for name, weight in zip(self.model_order, weights)}

    def _learn_topk_weights(self, pred_frame: pd.DataFrame, label: np.ndarray) -> Dict[str, float]:
        working_pred = self._cross_sectional_rank_score(pred_frame) if self.use_rank_score else pred_frame.copy()
        label_s = pd.Series(label, index=working_pred.index, name="label")
        grid = np.arange(0.0, 1.0 + 1e-9, self.search_step)
        best_score = None
        best_weights = None

        for w1 in grid:
            for w2 in grid:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                w3 = max(0.0, w3)
                weights = np.array([w1, w2, w3], dtype=float)
                score = self._topk_label_mean(working_pred, label_s, weights)
                if best_score is None or score > best_score:
                    best_score = score
                    best_weights = weights

        if best_weights is None:
            return {name: 1.0 / len(self.model_order) for name in self.model_order}

        best_weights = np.clip(best_weights, self.min_weight, None)
        best_weights = best_weights / best_weights.sum()
        return {name: float(weight) for name, weight in zip(self.model_order, best_weights)}

    def _topk_label_mean(self, pred_frame: pd.DataFrame, label: pd.Series, weights: np.ndarray) -> float:
        tmp = pred_frame.copy()
        tmp["label"] = label
        tmp["score"] = pred_frame.values @ weights
        date_level = "datetime" if "datetime" in tmp.index.names else tmp.index.names[0]
        daily_topk = tmp.groupby(level=date_level, group_keys=False).apply(
            lambda df: df.nlargest(self.topk, columns="score")["label"].mean()
        )
        return float(daily_topk.mean())

    @staticmethod
    def _daily_rank_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
        df = pd.DataFrame({"pred": pred, "label": label}).dropna()
        date_level = "datetime" if "datetime" in df.index.names else df.index.names[0]
        return df.groupby(level=date_level).apply(
            lambda x: x["pred"].rank(pct=True).corr(x["label"].rank(pct=True))
        ).replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _cross_sectional_zscore(values: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(values.index, pd.MultiIndex):
            centered = values - values.mean()
            return (centered / (values.std() + 1e-12)).fillna(0.0)
        date_level = "datetime" if "datetime" in values.index.names else values.index.names[0]
        grouped = values.groupby(level=date_level)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        return ((values - mean) / (std + 1e-12)).fillna(0.0)

    @staticmethod
    def _cross_sectional_rank_score(values: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(values.index, pd.MultiIndex):
            return values.rank(pct=True).fillna(0.0)
        date_level = "datetime" if "datetime" in values.index.names else values.index.names[0]
        return values.groupby(level=date_level).rank(pct=True).fillna(0.0)

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return np.squeeze(label)
        raise ValueError("MetaEnsembleModel only supports single-label training.")
