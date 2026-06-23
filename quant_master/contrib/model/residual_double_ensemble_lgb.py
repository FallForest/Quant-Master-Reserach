# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, Optional, Sequence, Text, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from .double_ensemble import DEnsembleModel


class ResidualDEnsembleLGBModel(Model):
    """DoubleEnsemble plus a LightGBM residual model."""

    def __init__(
        self,
        double_ensemble_kwargs: Optional[Dict] = None,
        residual_lgb_params: Optional[Dict] = None,
        residual_num_boost_round: int = 500,
        residual_early_stopping_rounds: int = 50,
        residual_weight_grid: Optional[Sequence[float]] = None,
        topk: int = 50,
        turnover_penalty: float = 0.0007,
        use_rank_score: bool = True,
        random_state: int = 42,
    ):
        if topk <= 0:
            raise ValueError("topk must be positive.")

        self.logger = get_module_logger("ResidualDEnsembleLGBModel")
        self.double_ensemble_kwargs = double_ensemble_kwargs or {}
        self.double_ensemble_kwargs.setdefault("random_state", random_state)
        self.residual_lgb_params = {
            "objective": "mse",
            "verbosity": -1,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "max_depth": 8,
            "min_data_in_leaf": 180,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l1": 20.0,
            "lambda_l2": 200.0,
            "num_threads": 20,
            "seed": random_state,
            "feature_fraction_seed": random_state,
            "bagging_seed": random_state,
            "data_random_seed": random_state,
        }
        if residual_lgb_params:
            self.residual_lgb_params.update(residual_lgb_params)
        self.residual_num_boost_round = residual_num_boost_round
        self.residual_early_stopping_rounds = residual_early_stopping_rounds
        self.residual_weight_grid = list(residual_weight_grid or [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        self.topk = topk
        self.turnover_penalty = turnover_penalty
        self.use_rank_score = use_rank_score
        self.random_state = random_state

        self.de_model = DEnsembleModel(**self.double_ensemble_kwargs)
        self.residual_model = None
        self.residual_weight = 0.0

    def fit(self, dataset: DatasetH):
        self.logger.info("Training DoubleEnsemble base model.")
        self.de_model.fit(dataset)

        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        base_train = self.de_model.predict(dataset, "train").loc[df_train.index]
        base_valid = self.de_model.predict(dataset, "valid").loc[df_valid.index]
        y_train = self._squeeze_label(df_train["label"]) - base_train.values
        y_valid = self._squeeze_label(df_valid["label"]) - base_valid.values

        self.logger.info("Training residual LightGBM model.")
        dtrain = lgb.Dataset(df_train["feature"].values, label=y_train, free_raw_data=False)
        dvalid = lgb.Dataset(df_valid["feature"].values, label=y_valid, free_raw_data=False)
        self.residual_model = lgb.train(
            self.residual_lgb_params,
            dtrain,
            num_boost_round=self.residual_num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.residual_early_stopping_rounds),
                lgb.log_evaluation(50),
            ],
        )

        residual_valid = pd.Series(self.residual_model.predict(df_valid["feature"].values), index=df_valid.index)
        label_valid = pd.Series(self._squeeze_label(df_valid["label"]), index=df_valid.index)
        self.residual_weight = self._learn_residual_weight(base_valid, residual_valid, label_valid)
        self.logger.info("Residual weight: %s", self.residual_weight)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.residual_model is None:
            raise ValueError("model is not fitted yet!")
        base = self.de_model.predict(dataset, segment)
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        residual = pd.Series(self.residual_model.predict(features.values), index=features.index)
        return self._combine(base, residual, self.residual_weight)

    def _learn_residual_weight(self, base: pd.Series, residual: pd.Series, label: pd.Series) -> float:
        best_weight = 0.0
        best_objective = None
        for weight in self.residual_weight_grid:
            pred = self._combine(base, residual, float(weight))
            objective = self._topk_objective(pred, label)
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_weight = float(weight)
        return best_weight

    def _combine(self, base: pd.Series, residual: pd.Series, weight: float) -> pd.Series:
        if self.use_rank_score:
            base_score = self._rank_score(base)
            residual_score = self._rank_score(residual)
            return base_score + weight * residual_score
        return base + weight * residual

    def _topk_objective(self, score: pd.Series, label: pd.Series) -> float:
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

    def _rank_score(self, score: pd.Series) -> pd.Series:
        date_level = self._date_level(score.index)
        return score.groupby(level=date_level).rank(pct=True).fillna(0.0)

    def _topk_index(self, score: pd.Series) -> pd.Index:
        if len(score) <= self.topk:
            return score.index
        values = score.values
        selected_pos = np.argpartition(values, -self.topk)[-self.topk :]
        return score.index[selected_pos]

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return np.squeeze(label)
        raise ValueError("ResidualDEnsembleLGBModel only supports single-label training.")

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("ResidualDEnsembleLGBModel requires a MultiIndex prediction index.")
        return "datetime" if "datetime" in index.names else index.names[0]

    @staticmethod
    def _instrument_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("ResidualDEnsembleLGBModel requires a MultiIndex prediction index.")
        return "instrument" if "instrument" in index.names else index.names[-1]
