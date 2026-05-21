# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, Iterable, Optional, Text, Union

import numpy as np
import pandas as pd

from ...data.dataset.handler import DataHandler, DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from .double_ensemble import DEnsembleModel
from .pytorch_cn_lstm_ts import CnLSTM


class DoubleEnsembleResidualCnLstmModel(Model):
    """DoubleEnsemble base model plus a CnLSTM trained on validation-selected residual alpha."""

    def __init__(
        self,
        double_ensemble_kwargs: Optional[Dict] = None,
        cn_lstm_kwargs: Optional[Dict] = None,
        residual_weight_grid: Optional[Iterable[float]] = None,
        stack_metric: str = "topk_turnover",
        topk: int = 50,
        turnover_penalty: float = 0.0,
        min_residual_improvement: float = 0.0,
        use_rank_score: bool = False,
        random_state: int = 42,
    ):
        if stack_metric not in {"rank_ic", "topk", "topk_turnover"}:
            raise ValueError(f"Unsupported stack_metric: {stack_metric}")

        self.logger = get_module_logger("DoubleEnsembleResidualCnLstmModel")
        self.double_ensemble_kwargs = double_ensemble_kwargs or {}
        self.cn_lstm_kwargs = cn_lstm_kwargs or {}
        self.stack_metric = stack_metric
        self.topk = topk
        self.turnover_penalty = turnover_penalty
        self.min_residual_improvement = min_residual_improvement
        self.use_rank_score = use_rank_score
        self.random_state = random_state

        if residual_weight_grid is None:
            residual_weight_grid = np.linspace(-1.0, 2.0, 31)
        self.residual_weight_grid = np.array(list(residual_weight_grid), dtype=float)
        if self.residual_weight_grid.size == 0:
            raise ValueError("residual_weight_grid must contain at least one value.")

        self.double_ensemble_kwargs.setdefault("random_state", random_state)
        self.cn_lstm_kwargs.setdefault("seed", random_state)

        self.base_model = DEnsembleModel(**self.double_ensemble_kwargs)
        self.residual_model = CnLSTM(**self.cn_lstm_kwargs)
        self.residual_weight = 0.0
        self.fitted = False

    def fit(self, dataset):
        tab_dataset = _LastStepDataset(dataset)
        self.logger.info("Training DoubleEnsemble base model.")
        self.base_model.fit(tab_dataset)

        train_label = self._prepare_label(tab_dataset, "train")
        valid_label = self._prepare_label(tab_dataset, "valid")
        base_train = self.base_model.predict(tab_dataset, "train").sort_index()
        base_valid = self.base_model.predict(tab_dataset, "valid").sort_index()

        train_residual = self._align_label_and_pred(train_label, base_train)
        valid_residual = self._align_label_and_pred(valid_label, base_valid)
        residual_dataset = _ResidualTSDataset(
            dataset,
            residuals={
                "train": train_residual,
                "valid": valid_residual,
            },
        )

        self.logger.info("Training CnLSTM residual model.")
        self.residual_model.fit(residual_dataset)

        residual_valid = self.residual_model.predict(residual_dataset, "valid").sort_index()
        valid_frame = pd.DataFrame(
            {
                "base": base_valid,
                "residual": residual_valid,
                "label": valid_label,
            }
        ).dropna()
        if valid_frame.empty:
            raise ValueError("Validation predictions are empty after alignment.")

        self.residual_weight = self._select_residual_weight(valid_frame)
        self.logger.info("Selected residual weight: %.6f", self.residual_weight)
        self.fitted = True

    def predict(self, dataset, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        tab_dataset = _LastStepDataset(dataset)
        base_pred = self.base_model.predict(tab_dataset, segment).sort_index()
        residual_pred = self.residual_model.predict(dataset, segment).sort_index()

        pred_frame = pd.DataFrame({"base": base_pred, "residual": residual_pred}).dropna()
        pred = pred_frame["base"] + self.residual_weight * pred_frame["residual"]
        return pred.sort_index()

    def _select_residual_weight(self, valid_frame: pd.DataFrame) -> float:
        base_score = valid_frame["base"]
        if self.use_rank_score:
            base_score = self._cross_sectional_rank_score(base_score)
        base_reward = self._calc_reward(base_score, valid_frame["label"])

        best_weight = 0.0
        best_reward = base_reward

        for weight in self.residual_weight_grid:
            if weight == 0.0:
                continue
            score = valid_frame["base"] + weight * valid_frame["residual"]
            if self.use_rank_score:
                score = self._cross_sectional_rank_score(score)
            reward = self._calc_reward(score, valid_frame["label"])
            if reward > best_reward:
                best_reward = reward
                best_weight = weight

        if best_reward < base_reward + self.min_residual_improvement:
            self.logger.info(
                "Residual rejected: base reward %.8f, best stack reward %.8f, min improvement %.8f",
                base_reward,
                best_reward,
                self.min_residual_improvement,
            )
            return 0.0

        self.logger.info("Base validation reward: %.8f", base_reward)
        self.logger.info("Best validation stack reward: %.8f", best_reward)
        return float(best_weight)

    def _calc_reward(self, pred: pd.Series, label: pd.Series) -> float:
        if self.stack_metric == "rank_ic":
            reward = self._daily_rank_ic(pred, label).mean()
        elif self.stack_metric == "topk":
            reward = self._daily_topk_label(pred, label).mean()
        else:
            reward = self._daily_topk_label(pred, label).mean() - self.turnover_penalty * self._topk_turnover(pred)
        return 0.0 if not np.isfinite(reward) else float(reward)

    def _daily_topk_label(self, pred: pd.Series, label: pd.Series) -> pd.Series:
        df = pd.DataFrame({"pred": pred, "label": label}).dropna()
        date_level = "datetime" if "datetime" in df.index.names else df.index.names[0]
        return df.groupby(level=date_level, group_keys=False).apply(
            lambda x: x.nlargest(self.topk, "pred")["label"].mean()
        )

    def _topk_turnover(self, pred: pd.Series) -> float:
        df = pred.dropna().rename("pred").to_frame()
        date_level = "datetime" if "datetime" in df.index.names else df.index.names[0]
        selections = []
        for _, daily_pred in df.groupby(level=date_level):
            selections.append(set(daily_pred.nlargest(self.topk, "pred").index.get_level_values("instrument")))
        if len(selections) <= 1:
            return 0.0

        turnovers = []
        for prev, cur in zip(selections[:-1], selections[1:]):
            if not prev and not cur:
                turnovers.append(0.0)
            else:
                turnovers.append(1.0 - len(prev & cur) / max(len(cur), 1))
        return float(np.mean(turnovers))

    @staticmethod
    def _prepare_label(tab_dataset: "_LastStepDataset", segment: Union[Text, slice]) -> pd.Series:
        label_df = tab_dataset.prepare(segment, col_set=["label"], data_key=DataHandlerLP.DK_L)["label"]
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return pd.Series(np.squeeze(label), index=label_df.index).sort_index()
        raise ValueError("DoubleEnsembleResidualCnLstmModel only supports single-label training.")

    @staticmethod
    def _align_label_and_pred(label: pd.Series, pred: pd.Series) -> pd.Series:
        frame = pd.DataFrame({"label": label, "pred": pred}).dropna()
        if frame.empty:
            raise ValueError("Label and base prediction index do not overlap.")
        return (frame["label"] - frame["pred"]).sort_index()

    @staticmethod
    def _daily_rank_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
        df = pd.DataFrame({"pred": pred, "label": label}).dropna()
        date_level = "datetime" if "datetime" in df.index.names else df.index.names[0]
        return df.groupby(level=date_level).apply(
            lambda x: x["pred"].rank(pct=True).corr(x["label"].rank(pct=True))
        ).replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _cross_sectional_rank_score(values: pd.Series) -> pd.Series:
        if not isinstance(values.index, pd.MultiIndex):
            return values.rank(pct=True).fillna(0.0)
        date_level = "datetime" if "datetime" in values.index.names else values.index.names[0]
        return values.groupby(level=date_level).rank(pct=True).fillna(0.0)


class _ResidualTSDataset:
    def __init__(self, dataset, residuals: Dict[str, pd.Series]):
        self.dataset = dataset
        self.segments = dataset.segments
        self.residuals = {key: value.sort_index() for key, value in residuals.items()}

    def prepare(
        self,
        segments: Union[Text, slice],
        col_set=DataHandler.CS_ALL,
        data_key=DataHandlerLP.DK_I,
        **kwargs,
    ):
        sampler = self.dataset.prepare(segments, col_set=col_set, data_key=data_key, **kwargs)
        if isinstance(segments, str) and segments in self.residuals and _uses_label(col_set):
            self._replace_label(sampler, self.residuals[segments])
        return sampler

    @staticmethod
    def _replace_label(sampler, residual: pd.Series):
        residual = residual.dropna()
        if residual.empty:
            return

        positions = sampler.idx_df.stack().reindex(residual.index).dropna()
        if positions.empty:
            return
        sampler.data_arr[positions.astype(np.int64).values, -1] = residual.loc[positions.index].values


class _LastStepDataset:
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
        if isinstance(segments, (list, tuple)) and all(seg in self.segments for seg in segments):
            return [self.prepare(seg, col_set=col_set, data_key=data_key, **kwargs) for seg in segments]
        if isinstance(segments, str) and segments in self.segments:
            segments = self.segments[segments]
        if not isinstance(segments, slice):
            segments = slice(*segments)
        return self.dataset.handler.fetch(segments, col_set=col_set, data_key=data_key, **kwargs)


def _uses_label(col_set) -> bool:
    if col_set == DataHandler.CS_ALL:
        return True
    if isinstance(col_set, str):
        return col_set == "label"
    return "label" in col_set
