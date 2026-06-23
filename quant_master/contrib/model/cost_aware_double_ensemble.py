# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, Text, Union

import numpy as np
import pandas as pd

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from .double_ensemble import DEnsembleModel


class CostAwareDEnsembleModel(DEnsembleModel):
    """DoubleEnsemble variant with top-k-aware sample reweighting and validation scoring."""

    def __init__(
        self,
        *args,
        topk: int = 50,
        turnover_penalty: float = 0.0007,
        label_rank_weight: float = 0.5,
        stability_weight: float = 0.3,
        trajectory_weight: float = 1.0,
        use_rank_score: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if topk <= 0:
            raise ValueError("topk must be positive.")
        self.topk = topk
        self.turnover_penalty = turnover_penalty
        self.label_rank_weight = label_rank_weight
        self.stability_weight = stability_weight
        self.trajectory_weight = trajectory_weight
        self.use_rank_score = use_rank_score
        self.logger = get_module_logger("CostAwareDEnsembleModel")

        self._train_rank_score = None
        self._train_stability_score = None

    def fit(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        n_samples, _ = x_train.shape
        weights = pd.Series(np.ones(n_samples, dtype=float), index=x_train.index)
        features = x_train.columns
        pred_sub_train = pd.DataFrame(np.zeros((n_samples, self.num_models), dtype=float), index=x_train.index)
        pred_sub_valid = pd.DataFrame(
            np.zeros((df_valid["feature"].shape[0], self.num_models), dtype=float), index=df_valid.index
        )

        self._prepare_cost_aware_scores(y_train)
        self.ensemble = []
        self.sub_features = []

        for k in range(self.num_models):
            self.sub_features.append(features)
            self.logger.info("Training sub-model: ({}/{})".format(k + 1, self.num_models))
            model_k = self.train_submodel(df_train, df_valid, weights, features)
            self.ensemble.append(model_k)

            pred_k_train = self.predict_sub(model_k, df_train, features)
            pred_k_valid = self.predict_sub(model_k, df_valid, features)
            pred_sub_train.iloc[:, k] = pred_k_train
            pred_sub_valid.iloc[:, k] = pred_k_valid

            if k + 1 == self.num_models:
                break

            self.logger.info("Retrieving loss curve and loss values...")
            loss_curve = self.retrieve_loss_curve(model_k, df_train, features)
            pred_ensemble = (pred_sub_train.iloc[:, : k + 1] * np.asarray(self.sub_weights[0 : k + 1])).sum(
                axis=1
            ) / np.sum(self.sub_weights[0 : k + 1])
            loss_values = pd.Series(
                self.get_loss(y_train.values.squeeze(), pred_ensemble.values), index=pred_ensemble.index
            )

            if self.enable_sr:
                self.logger.info("Cost-aware sample re-weighting...")
                weights = self.sample_reweight(loss_curve, loss_values, k + 1)

            if self.enable_fs:
                self.logger.info("Feature selection...")
                features = self.feature_selection(df_train, loss_values)

        valid_label = pd.Series(self._squeeze_label(df_valid["label"]), index=df_valid.index)
        self.sub_weights = self._learn_submodel_weights(pred_sub_valid, valid_label)
        self.logger.info("Validation-learned sub-model weights: %s", self.sub_weights)

    def sample_reweight(self, loss_curve, loss_values, k_th):
        loss_curve_norm = loss_curve.rank(axis=0, pct=True)
        loss_values_norm = (-loss_values).rank(pct=True)

        _, num_rounds = loss_curve.shape
        part = np.maximum(int(num_rounds * 0.1), 1)
        l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
        l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)

        trajectory_part = pd.Series((l_end / l_start).rank(pct=True).values, index=loss_values.index)
        trajectory_score = self.alpha1 * loss_values_norm + self.alpha2 * trajectory_part
        rank_score = self._train_rank_score.reindex(loss_values.index).fillna(0.5)
        stability_score = self._train_stability_score.reindex(loss_values.index).fillna(0.5)

        h_value = (
            self.trajectory_weight * trajectory_score
            + self.label_rank_weight * rank_score
            + self.stability_weight * stability_score
        )
        score_frame = pd.DataFrame({"h_value": h_value})
        score_frame["bins"] = pd.cut(score_frame["h_value"], self.bins_sr)
        h_avg = score_frame.groupby("bins", group_keys=False, observed=False)["h_value"].mean()
        weights = pd.Series(np.zeros(len(score_frame), dtype=float), index=score_frame.index)
        for b in h_avg.index:
            weights[score_frame["bins"] == b] = 1.0 / (self.decay**k_th * h_avg[b] + 0.1)
        return weights

    def _prepare_cost_aware_scores(self, label_df: pd.DataFrame):
        label = pd.Series(self._squeeze_label(label_df), index=label_df.index)
        date_level = self._date_level(label.index)
        instrument_level = self._instrument_level(label.index)

        self._train_rank_score = label.groupby(level=date_level).rank(pct=True).fillna(0.5)

        rank_df = self._train_rank_score.to_frame("rank_score")
        rank_df["rank_diff"] = (
            rank_df.groupby(level=instrument_level, group_keys=False)["rank_score"].diff().abs().fillna(0.0)
        )
        rank_df["stability_score"] = 1.0 - rank_df.groupby(level=date_level)["rank_diff"].rank(pct=True).fillna(0.5)
        self._train_stability_score = rank_df["stability_score"]

    def _learn_submodel_weights(self, pred_sub_valid: pd.DataFrame, valid_label: pd.Series):
        objective_scores = []
        for col in pred_sub_valid.columns:
            pred = pred_sub_valid[col]
            if self.use_rank_score:
                pred = self._cross_sectional_rank_score(pred)
            objective_scores.append(max(self._topk_cost_objective(pred, valid_label), 0.0))

        total = float(np.sum(objective_scores))
        if total <= 0:
            return [1.0] * self.num_models
        return [score / total for score in objective_scores]

    def _topk_cost_objective(self, pred: pd.Series, label: pd.Series) -> float:
        df = pd.DataFrame({"pred": pred, "label": label}).dropna()
        if df.empty:
            return float("-inf")

        date_level = self._date_level(df.index)
        instrument_level = self._instrument_level(df.index)
        daily_returns = []
        daily_turnovers = []
        prev_selected = None

        for _, daily_df in df.groupby(level=date_level, sort=True):
            topk_df = daily_df.nlargest(self.topk, columns="pred")
            daily_returns.append(float(topk_df["label"].mean()))
            selected = set(topk_df.index.get_level_values(instrument_level))
            if prev_selected is not None:
                overlap = len(prev_selected.intersection(selected))
                daily_turnovers.append(1.0 - overlap / max(len(selected), 1))
            prev_selected = selected

        turnover = float(np.mean(daily_turnovers)) if daily_turnovers else 0.0
        return float(np.mean(daily_returns) - self.turnover_penalty * turnover)

    @staticmethod
    def _cross_sectional_rank_score(values: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(values.index, pd.MultiIndex):
            return values.rank(pct=True).fillna(0.0)
        date_level = CostAwareDEnsembleModel._date_level(values.index)
        return values.groupby(level=date_level).rank(pct=True).fillna(0.0)

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return np.squeeze(label)
        raise ValueError("CostAwareDEnsembleModel only supports single-label training.")

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("CostAwareDEnsembleModel requires a MultiIndex index.")
        return "datetime" if "datetime" in index.names else index.names[0]

    @staticmethod
    def _instrument_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("CostAwareDEnsembleModel requires a MultiIndex index.")
        return "instrument" if "instrument" in index.names else index.names[-1]
