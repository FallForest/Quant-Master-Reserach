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


class MultiSeedDEnsembleModel(Model):
    """Bag multiple DoubleEnsemble models trained with different random seeds."""

    def __init__(
        self,
        seed_list: Optional[Sequence[int]] = None,
        double_ensemble_kwargs: Optional[Dict] = None,
        blend_method: str = "icir",
        use_rank_score: bool = True,
        min_weight: float = 1e-6,
        topk: int = 50,
        search_step: float = 0.2,
    ):
        seed_list = (0, 1, 2) if seed_list is None else seed_list
        if not seed_list:
            raise ValueError("seed_list must not be empty.")
        if blend_method not in {"equal", "icir", "topk"}:
            raise ValueError(f"Unsupported blend_method: {blend_method}")

        self.logger = get_module_logger("MultiSeedDEnsembleModel")
        self.seed_list = [int(seed) for seed in seed_list]
        self.double_ensemble_kwargs = dict(double_ensemble_kwargs or {})
        self.blend_method = blend_method
        self.use_rank_score = use_rank_score
        self.min_weight = min_weight
        self.topk = topk
        self.search_step = search_step

        self.models: List[DEnsembleModel] = []
        self.model_names = [f"seed_{seed}" for seed in self.seed_list]
        self.model_weights: Dict[str, float] = {}

    def fit(self, dataset: DatasetH):
        self.models = []
        for seed in self.seed_list:
            kwargs = dict(self.double_ensemble_kwargs)
            kwargs["random_state"] = seed
            model = DEnsembleModel(**kwargs)
            self.logger.info("Training DoubleEnsemble seed=%s", seed)
            model.fit(dataset)
            self.models.append(model)

        valid_pred = self._predict_frame(dataset, "valid")
        valid_label = dataset.prepare("valid", col_set=["label"], data_key=DataHandlerLP.DK_L)["label"]
        valid_pred, valid_label = self._align_prediction_and_label(valid_pred, valid_label)
        label = pd.Series(self._squeeze_label(valid_label), index=valid_label.index)
        if self.use_rank_score:
            valid_pred = self._cross_sectional_rank_score(valid_pred)

        if self.blend_method == "equal":
            weight = 1.0 / len(self.model_names)
            self.model_weights = {name: weight for name in self.model_names}
        elif self.blend_method == "icir":
            self.model_weights = self._learn_icir_weights(valid_pred, label)
        else:
            self.model_weights = self._learn_topk_weights(valid_pred, label)
        self.logger.info("Model weights: %s", self.model_weights)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.models:
            raise ValueError("model is not fitted yet!")
        pred_frame = self._predict_frame(dataset, segment)
        if self.use_rank_score:
            pred_frame = self._cross_sectional_rank_score(pred_frame)
        pred = np.zeros(len(pred_frame), dtype=float)
        for name in self.model_names:
            pred += pred_frame[name].values * self.model_weights[name]
        return pd.Series(pred, index=pred_frame.index)

    def _predict_frame(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.DataFrame:
        return pd.DataFrame(
            {name: model.predict(dataset, segment) for name, model in zip(self.model_names, self.models)}
        )

    def _learn_icir_weights(self, pred_frame: pd.DataFrame, label: pd.Series) -> Dict[str, float]:
        icir_scores = {}
        for model_name in self.model_names:
            daily_ic = self._daily_rank_ic(pred_frame[model_name], label)
            mean = daily_ic.mean()
            std = daily_ic.std()
            icir = float(mean / std) if std and np.isfinite(std) and std > 0 else 0.0
            icir_scores[model_name] = max(icir, 0.0)

        if sum(icir_scores.values()) <= 0:
            weight = 1.0 / len(self.model_names)
            return {name: weight for name in self.model_names}

        total = sum(icir_scores.values())
        return {name: score / total for name, score in icir_scores.items()}

    def _learn_topk_weights(self, pred_frame: pd.DataFrame, label: pd.Series) -> Dict[str, float]:
        if len(self.model_names) > 3:
            raise ValueError("topk blend_method currently supports at most 3 seeds.")

        best_score = None
        best_weights = None
        for weights in self._weight_grid(len(self.model_names)):
            score = self._topk_label_mean(pred_frame, label, weights)
            if best_score is None or score > best_score:
                best_score = score
                best_weights = weights

        if best_weights is None:
            weight = 1.0 / len(self.model_names)
            return {name: weight for name in self.model_names}

        best_weights = np.clip(best_weights, self.min_weight, None)
        best_weights = best_weights / best_weights.sum()
        return {name: float(weight) for name, weight in zip(self.model_names, best_weights)}

    def _weight_grid(self, n_models: int):
        grid = np.arange(0.0, 1.0 + 1e-9, self.search_step)
        if n_models == 1:
            yield np.array([1.0], dtype=float)
            return
        if n_models == 2:
            for w1 in grid:
                yield np.array([w1, max(0.0, 1.0 - w1)], dtype=float)
            return
        for w1 in grid:
            for w2 in grid:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                yield np.array([w1, w2, max(0.0, w3)], dtype=float)

    def _topk_label_mean(self, pred_frame: pd.DataFrame, label: pd.Series, weights: np.ndarray) -> float:
        tmp = pred_frame.copy()
        tmp["label"] = label
        tmp["score"] = pred_frame.values @ weights
        date_level = self._date_level(tmp.index)
        daily_topk = tmp.groupby(level=date_level, group_keys=False).apply(
            lambda df: df.nlargest(self.topk, columns="score")["label"].mean()
        )
        return float(daily_topk.mean())

    @staticmethod
    def _daily_rank_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
        df = pd.DataFrame({"pred": pred, "label": label}).dropna()
        date_level = MultiSeedDEnsembleModel._date_level(df.index)
        return (
            df.groupby(level=date_level)
            .apply(lambda x: x["pred"].rank(pct=True).corr(x["label"].rank(pct=True)))
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

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
        date_level = MultiSeedDEnsembleModel._date_level(values.index)
        return values.groupby(level=date_level).rank(pct=True).fillna(0.0)

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return np.squeeze(label)
        raise ValueError("MultiSeedDEnsembleModel only supports single-label training.")

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("MultiSeedDEnsembleModel requires a MultiIndex index.")
        return "datetime" if "datetime" in index.names else index.names[0]
