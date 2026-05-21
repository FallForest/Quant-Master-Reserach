# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, Optional, Text, Union

import numpy as np
import pandas as pd

from ...data.dataset.handler import DataHandler, DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from .gbdt import LGBModel
from .pytorch_cn_lstm_ts import CnLSTM


class TreeCnLstmRLModel(Model):
    """LightGBM + CnLSTM blended by a validation-driven RL bandit."""

    def __init__(
        self,
        tree_kwargs: Optional[Dict] = None,
        cn_lstm_kwargs: Optional[Dict] = None,
        rl_lr: float = 0.15,
        rl_epochs: int = 200,
        entropy_coef: float = 0.01,
        reward_method: str = "rank_ic",
        topk: int = 50,
        use_rank_score: bool = True,
        fixed_weights: Optional[Dict[str, float]] = None,
        parallel_submodels: Optional[bool] = None,
        random_state: int = 42,
    ):
        if reward_method not in {"rank_ic", "topk"}:
            raise ValueError(f"Unsupported reward_method: {reward_method}")

        self.logger = get_module_logger("TreeCnLstmRLModel")
        self.tree_kwargs = tree_kwargs or {}
        self.cn_lstm_kwargs = cn_lstm_kwargs or {}
        self.rl_lr = rl_lr
        self.rl_epochs = rl_epochs
        self.entropy_coef = entropy_coef
        self.reward_method = reward_method
        self.topk = topk
        self.use_rank_score = use_rank_score
        self.fixed_weights = fixed_weights
        if parallel_submodels is None:
            parallel_submodels = os.name != "nt"
        self.parallel_submodels = parallel_submodels
        self.random_state = random_state

        self.tree_kwargs.setdefault("seed", random_state)
        self.tree_kwargs.setdefault("feature_fraction_seed", random_state)
        self.tree_kwargs.setdefault("bagging_seed", random_state)
        self.tree_kwargs.setdefault("data_random_seed", random_state)
        self.cn_lstm_kwargs.setdefault("seed", random_state)

        self.tree_model = LGBModel(**self.tree_kwargs)
        self.cn_lstm_model = CnLSTM(**self.cn_lstm_kwargs)
        self.model_order = ["tree", "cn_lstm"]
        self.model_weights: Dict[str, float] = {}
        self.fitted = False

    def fit(self, dataset):
        if self.parallel_submodels:
            self.logger.info("Training tree and CnLSTM sub-models in parallel.")
            with ThreadPoolExecutor(max_workers=2) as executor:
                tree_future = executor.submit(self.tree_model.fit, _LastStepDataset(dataset))
                cn_lstm_future = executor.submit(self.cn_lstm_model.fit, dataset)
                tree_future.result()
                cn_lstm_future.result()
        else:
            self.logger.info("Training tree and CnLSTM sub-models sequentially.")
            self.tree_model.fit(_LastStepDataset(dataset))
            self.cn_lstm_model.fit(dataset)

        valid_pred = self._predict_frame(dataset, "valid")
        valid_label = self._prepare_label(dataset, "valid")
        valid_pred, valid_label = self._align_prediction_and_label(valid_pred, valid_label)
        if self.use_rank_score:
            valid_pred = self._cross_sectional_rank_score(valid_pred)

        rewards = self._calc_rewards(valid_pred, valid_label)
        if self.fixed_weights is None:
            self.model_weights = self._learn_rl_weights(rewards)
        else:
            self.model_weights = self._normalize_fixed_weights(self.fixed_weights)
        self.logger.info("Model rewards: %s", rewards)
        self.logger.info("Model weights: %s", self.model_weights)
        self.fitted = True

    def predict(self, dataset, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        pred_frame = self._predict_frame(dataset, segment)
        if self.use_rank_score:
            pred_frame = self._cross_sectional_rank_score(pred_frame)

        pred = np.zeros(len(pred_frame), dtype=float)
        for model_name in self.model_order:
            pred += pred_frame[model_name].values * self.model_weights[model_name]
        return pd.Series(pred, index=pred_frame.index)

    def _predict_frame(self, dataset, segment: Union[Text, slice]) -> pd.DataFrame:
        preds = {
            "tree": self.tree_model.predict(_LastStepDataset(dataset), segment),
            "cn_lstm": self.cn_lstm_model.predict(dataset, segment),
        }
        return pd.DataFrame(preds)

    def _prepare_label(self, dataset, segment: Union[Text, slice]) -> pd.Series:
        label_df = _LastStepDataset(dataset).prepare(segment, col_set=["label"], data_key=DataHandlerLP.DK_L)["label"]
        label = self._squeeze_label(label_df)
        return pd.Series(label, index=label_df.index)

    def _calc_rewards(self, pred_frame: pd.DataFrame, label: pd.Series) -> Dict[str, float]:
        rewards = {}
        for model_name in self.model_order:
            if self.reward_method == "rank_ic":
                daily_ic = self._daily_rank_ic(pred_frame[model_name], label)
                reward = daily_ic.mean()
            else:
                reward = self._topk_label_mean(pred_frame[model_name], label, self.topk)
            rewards[model_name] = 0.0 if not np.isfinite(reward) else float(reward)
        return rewards

    def _learn_rl_weights(self, rewards: Dict[str, float]) -> Dict[str, float]:
        reward_vec = np.array([rewards[name] for name in self.model_order], dtype=float)
        rng = np.random.default_rng(self.random_state)
        logits = rng.normal(loc=0.0, scale=1e-3, size=len(self.model_order))
        baseline = 0.0

        for _ in range(self.rl_epochs):
            probs = self._softmax(logits)
            action = rng.choice(len(self.model_order), p=probs)
            reward = reward_vec[action]
            baseline = 0.9 * baseline + 0.1 * reward
            advantage = reward - baseline

            grad = -probs
            grad[action] += 1.0
            entropy_grad = -probs * (np.log(probs + 1e-12) + 1.0)
            logits += self.rl_lr * (advantage * grad + self.entropy_coef * entropy_grad)

        weights = self._softmax(logits)
        return {name: float(weight) for name, weight in zip(self.model_order, weights)}

    def _normalize_fixed_weights(self, fixed_weights: Dict[str, float]) -> Dict[str, float]:
        missing = set(self.model_order) - set(fixed_weights)
        if missing:
            raise ValueError(f"fixed_weights is missing weights for: {sorted(missing)}")

        weights = np.array([fixed_weights[name] for name in self.model_order], dtype=float)
        if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("fixed_weights must be non-negative finite values with positive sum.")
        weights = weights / weights.sum()
        return {name: float(weight) for name, weight in zip(self.model_order, weights)}

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    @staticmethod
    def _align_prediction_and_label(pred_frame: pd.DataFrame, label: pd.Series):
        common_index = pred_frame.index.intersection(label.index)
        if len(common_index) == 0:
            raise ValueError("Prediction index and label index do not overlap.")
        return pred_frame.loc[common_index], label.loc[common_index]

    @staticmethod
    def _daily_rank_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
        df = pd.DataFrame({"pred": pred, "label": label}).dropna()
        date_level = "datetime" if "datetime" in df.index.names else df.index.names[0]
        return df.groupby(level=date_level).apply(
            lambda x: x["pred"].rank(pct=True).corr(x["label"].rank(pct=True))
        ).replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _topk_label_mean(pred: pd.Series, label: pd.Series, topk: int = 50) -> float:
        df = pd.DataFrame({"pred": pred, "label": label}).dropna()
        date_level = "datetime" if "datetime" in df.index.names else df.index.names[0]
        daily_topk = df.groupby(level=date_level, group_keys=False).apply(
            lambda x: x.nlargest(topk, "pred")["label"].mean()
        )
        return float(daily_topk.mean())

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
        raise ValueError("TreeCnLstmRLModel only supports single-label training.")


class _LastStepDataset:
    """Adapter that exposes TSDatasetH handler data as tabular frames."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.segments = dataset.segments

    def prepare(
        self,
        segments: Union[Text, slice],
        col_set=DataHandler.CS_ALL,
        data_key=DataHandlerLP.DK_I,
        **kwargs,
    ) -> pd.DataFrame:
        if isinstance(segments, str) and segments in self.segments:
            segments = self.segments[segments]
        if not isinstance(segments, slice):
            segments = slice(*segments)
        return self.dataset.handler.fetch(segments, col_set=col_set, data_key=data_key, **kwargs)
