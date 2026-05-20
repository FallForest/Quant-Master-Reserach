# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, List, Optional, Text, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import ModelFT
from ...model.interpret.base import LightGBMFInt
from ...workflow import R


class TopKMetaLabelModel(ModelFT, LightGBMFInt):
    """LightGBM with train/valid-only cross-sectional meta labels."""

    SUPPORTED_MODES = {"top_bottom", "top_only", "rank"}
    SUPPORTED_LOSSES = {"auto", "mse", "binary"}

    def __init__(
        self,
        target_mode: str = "top_bottom",
        top_quantile: float = 0.1,
        loss: str = "auto",
        early_stopping_rounds: int = 80,
        num_boost_round: int = 600,
        **kwargs,
    ):
        mode = str(target_mode or "top_bottom").strip().lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported target_mode={target_mode}. Expected one of {sorted(self.SUPPORTED_MODES)}.")
        if not 0.0 < float(top_quantile) < 0.5:
            raise ValueError("top_quantile must be in (0, 0.5).")

        loss_name = str(loss or "auto").strip().lower()
        if loss_name not in self.SUPPORTED_LOSSES:
            raise ValueError(f"Unsupported loss={loss}. Expected one of {sorted(self.SUPPORTED_LOSSES)}.")
        objective = self._resolve_objective(mode, loss_name)

        self.logger = get_module_logger("TopKMetaLabelModel")
        self.target_mode = mode
        self.top_quantile = float(top_quantile)
        self.loss = loss_name
        self.params = {"objective": objective, "verbosity": -1}
        self.params.update(kwargs)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.num_boost_round = int(num_boost_round)
        self.model = None

    @staticmethod
    def _resolve_objective(mode: str, loss_name: str) -> str:
        if loss_name != "auto":
            return "regression" if loss_name == "mse" else "binary"
        return "binary" if mode == "top_only" else "regression"

    def _prepare_data(self, dataset: DatasetH) -> List[Tuple[lgb.Dataset, str]]:
        ds_l: List[Tuple[lgb.Dataset, str]] = []
        assert "train" in dataset.segments
        for key in ["train", "valid"]:
            if key not in dataset.segments:
                continue
            df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            if df.empty:
                raise ValueError(f"Empty data from dataset segment={key}, please check dataset config.")
            x, y_df = df["feature"], df["label"]
            y_raw = self._squeeze_label(y_df)
            y_target = self._build_meta_target(pd.Series(y_raw, index=y_df.index, dtype=float))
            ds_l.append((lgb.Dataset(x.values, label=y_target.values, free_raw_data=False), key))
        return ds_l

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose_eval: int = 20,
        evals_result: Optional[Dict] = None,
        reweighter=None,
        **kwargs,
    ):
        if evals_result is None:
            evals_result = {}
        ds_l = self._prepare_data(dataset)
        ds, names = list(zip(*ds_l))
        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds),
            lgb.log_evaluation(period=verbose_eval),
            lgb.record_evaluation(evals_result),
        ]
        self.model = lgb.train(
            self.params,
            ds[0],
            num_boost_round=self.num_boost_round if num_boost_round is None else int(num_boost_round),
            valid_sets=ds,
            valid_names=names,
            callbacks=callbacks,
            **kwargs,
        )
        self._log_evals(evals_result, names)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        pred = self.model.predict(x_test.values)
        return pd.Series(pred, index=x_test.index)

    def finetune(self, dataset: DatasetH, num_boost_round: int = 10, verbose_eval: int = 20, reweighter=None):
        ds_l = self._prepare_data(dataset)
        dtrain, _ = ds_l[0]
        callbacks = [lgb.log_evaluation(period=verbose_eval)]
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=int(num_boost_round),
            init_model=self.model,
            valid_sets=[dtrain],
            valid_names=["train"],
            callbacks=callbacks,
        )

    def _build_meta_target(self, label_s: pd.Series) -> pd.Series:
        label_s = label_s.replace([np.inf, -np.inf], np.nan)
        rank_pct = self._cross_sectional_rank_pct(label_s)
        q = self.top_quantile
        if self.target_mode == "rank":
            target = (rank_pct - 0.5) * 2.0
            return target.fillna(0.0)
        if self.target_mode == "top_only":
            target = (rank_pct >= 1.0 - q).astype(float)
            return target.fillna(0.0)
        top = (rank_pct >= 1.0 - q).astype(float)
        bottom = (rank_pct <= q).astype(float)
        target = top - bottom
        return target.fillna(0.0)

    @staticmethod
    def _cross_sectional_rank_pct(values: pd.Series) -> pd.Series:
        if not isinstance(values.index, pd.MultiIndex):
            return values.rank(pct=True)
        date_level = "datetime" if "datetime" in values.index.names else values.index.names[0]
        return values.groupby(level=date_level).rank(pct=True)

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        arr = label_df.values
        if arr.ndim == 2 and arr.shape[1] == 1:
            return np.squeeze(arr)
        raise ValueError("TopKMetaLabelModel only supports single-label training.")

    @staticmethod
    def _log_evals(evals_result: Dict, names: Tuple[str, ...]) -> None:
        for name in names:
            section = evals_result.get(name, {})
            for metric_name, values in section.items():
                log_name = f"{metric_name}.{name}".replace("@", "_")
                for epoch, metric_value in enumerate(values):
                    R.log_metrics(**{log_name: metric_value}, step=epoch)
