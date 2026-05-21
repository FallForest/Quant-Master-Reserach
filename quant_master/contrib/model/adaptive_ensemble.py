# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Text, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model


@dataclass
class _FeatureStats:
    median: pd.Series
    mean: pd.Series
    std: pd.Series


class AdaptiveEnsembleModel(Model):
    """Validation-weighted heterogeneous ensemble for tabular alpha modeling."""

    def __init__(
        self,
        loss: str = "mse",
        enable_lightgbm: bool = True,
        enable_extratrees: bool = True,
        enable_ridge: bool = True,
        lgb_params: Optional[Dict] = None,
        lgb_num_boost_round: int = 600,
        lgb_early_stopping_rounds: int = 50,
        extratrees_params: Optional[Dict] = None,
        ridge_alpha: float = 2.0,
        min_weight: float = 1e-4,
    ):
        if loss != "mse":
            raise NotImplementedError("AdaptiveEnsembleModel only supports mse loss.")

        self.loss = loss
        self.enable_lightgbm = enable_lightgbm
        self.enable_extratrees = enable_extratrees
        self.enable_ridge = enable_ridge
        self.lgb_params = {
            "objective": loss,
            "verbosity": -1,
            "feature_fraction": 0.8879,
            "bagging_fraction": 0.8789,
            "bagging_freq": 1,
            "learning_rate": 0.03,
            "num_leaves": 255,
            "max_depth": 10,
            "min_data_in_leaf": 120,
            "lambda_l1": 1.0,
            "lambda_l2": 20.0,
            "num_threads": 20,
        }
        if lgb_params:
            self.lgb_params.update(lgb_params)
        self.lgb_num_boost_round = lgb_num_boost_round
        self.lgb_early_stopping_rounds = lgb_early_stopping_rounds
        self.extratrees_params = {
            "n_estimators": 240,
            "max_depth": 12,
            "min_samples_leaf": 80,
            "max_features": 0.35,
            "n_jobs": 1,
            "random_state": 42,
        }
        if extratrees_params:
            self.extratrees_params.update(extratrees_params)
        self.ridge_alpha = ridge_alpha
        self.min_weight = min_weight

        self.logger = get_module_logger("AdaptiveEnsembleModel")
        self.model_order: List[str] = []
        self.model_weights: Dict[str, float] = {}
        self.models: Dict[str, object] = {}
        self.feature_stats: Optional[_FeatureStats] = None

    def fit(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train = df_train["feature"].astype(float)
        y_train = self._squeeze_label(df_train["label"])
        x_valid = df_valid["feature"].astype(float)
        y_valid = self._squeeze_label(df_valid["label"])

        self.feature_stats = self._build_feature_stats(x_train)
        x_train_filled = self._fill_features(x_train)
        x_valid_filled = self._fill_features(x_valid)

        self.models = {}
        self.model_order = []

        if self.enable_lightgbm:
            self.logger.info("Training LightGBM sub-model.")
            self.models["lightgbm"] = self._fit_lightgbm(x_train_filled, y_train, x_valid_filled, y_valid)
            self.model_order.append("lightgbm")

        if self.enable_extratrees:
            self.logger.info("Training ExtraTrees sub-model.")
            extra_trees = ExtraTreesRegressor(**self.extratrees_params)
            extra_trees.fit(x_train_filled.values, y_train)
            self.models["extratrees"] = extra_trees
            self.model_order.append("extratrees")

        if self.enable_ridge:
            self.logger.info("Training Ridge sub-model.")
            x_train_scaled = self._scale_features(x_train_filled)
            ridge = Ridge(alpha=self.ridge_alpha)
            ridge.fit(x_train_scaled.values, y_train)
            self.models["ridge"] = ridge
            self.model_order.append("ridge")

        if not self.model_order:
            raise ValueError("No sub-model is enabled.")

        valid_pred_frame = self._predict_base_models(x_valid)
        self.model_weights = self._learn_weights(valid_pred_frame, y_valid, valid_pred_frame.index)
        self.logger.info("Model weights: %s", self.model_weights)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.models:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I).astype(float)
        pred_frame = self._predict_base_models(x_test)
        ensemble_pred = np.zeros(len(pred_frame), dtype=float)
        for model_name in self.model_order:
            ensemble_pred += pred_frame[model_name].values * self.model_weights[model_name]
        return pd.Series(ensemble_pred, index=pred_frame.index)

    def _fit_lightgbm(self, x_train, y_train, x_valid, y_valid):
        dtrain = lgb.Dataset(x_train.values, label=y_train, free_raw_data=False)
        dvalid = lgb.Dataset(x_valid.values, label=y_valid, free_raw_data=False)
        return lgb.train(
            self.lgb_params,
            dtrain,
            num_boost_round=self.lgb_num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.lgb_early_stopping_rounds),
                lgb.log_evaluation(50),
            ],
        )

    def _predict_base_models(self, features: pd.DataFrame) -> pd.DataFrame:
        filled = self._fill_features(features)
        result = {}
        for model_name in self.model_order:
            if model_name == "lightgbm":
                result[model_name] = self.models[model_name].predict(filled.values)
            elif model_name == "extratrees":
                result[model_name] = self.models[model_name].predict(filled.values)
            elif model_name == "ridge":
                scaled = self._scale_features(filled)
                result[model_name] = self.models[model_name].predict(scaled.values)
            else:
                raise ValueError(f"Unknown sub-model: {model_name}")
        return pd.DataFrame(result, index=features.index)

    def _learn_weights(self, pred_frame: pd.DataFrame, label: np.ndarray, index: pd.Index) -> Dict[str, float]:
        pred_norm = self._cross_sectional_zscore(pred_frame, index)
        label_norm = self._cross_sectional_zscore(pd.Series(label, index=index), index)

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

    def _build_feature_stats(self, x_train: pd.DataFrame) -> _FeatureStats:
        median = x_train.median()
        filled = x_train.fillna(median)
        mean = filled.mean()
        std = filled.std().replace(0, 1.0).fillna(1.0)
        return _FeatureStats(median=median, mean=mean, std=std)

    def _fill_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_stats is None:
            raise ValueError("feature statistics are not initialized")
        return features.fillna(self.feature_stats.median)

    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_stats is None:
            raise ValueError("feature statistics are not initialized")
        return (features - self.feature_stats.mean) / self.feature_stats.std

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return np.squeeze(label)
        raise ValueError("AdaptiveEnsembleModel only supports single-label training.")

    @staticmethod
    def _cross_sectional_zscore(
        values: Union[pd.DataFrame, pd.Series], index: pd.Index
    ) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(index, pd.MultiIndex):
            centered = values - values.mean()
            scaled = centered / (values.std() + 1e-12)
            return scaled.fillna(0.0)

        date_level = "datetime" if "datetime" in index.names else index.names[0]
        grouped = values.groupby(level=date_level)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        scaled = (values - mean) / (std + 1e-12)
        return scaled.fillna(0.0)
