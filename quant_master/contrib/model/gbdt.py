# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import lightgbm as lgb
from typing import List, Text, Tuple, Union
from ...model.base import ModelFT
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import LightGBMFInt
from ...data.dataset.weight import Reweighter
from quant_master.workflow import R
from .rank_aware import DailyRankICEval, RANK_TARGET_MODE, normalize_target_mode, prepare_tree_target


class LGBModel(ModelFT, LightGBMFInt):
    """LightGBM Model"""

    def __init__(
        self,
        loss="mse",
        early_stopping_rounds=50,
        num_boost_round=1000,
        target_mode="raw",
        rank_eval=None,
        **kwargs,
    ):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        self.target_mode = normalize_target_mode(target_mode)
        if self.target_mode == RANK_TARGET_MODE and loss != "mse":
            raise ValueError("target_mode='cs_rank' requires loss='mse'.")
        self.rank_eval = self.target_mode == RANK_TARGET_MODE if rank_eval is None else bool(rank_eval)
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        if self.rank_eval:
            self.params.setdefault("metric", "None")
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None

    def _prepare_data(self, dataset: DatasetH, reweighter=None) -> List[Tuple[lgb.Dataset, str]]:
        """
        The motivation of current version is to make validation optional
        - train segment is necessary;
        """
        ds_l = []
        self._index_by_dataset_id = {}
        assert "train" in dataset.segments
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
                if df.empty:
                    raise ValueError("Empty data from dataset, please check your dataset config.")
                x = df["feature"]
                y = prepare_tree_target(df["label"], self.target_mode)

                if reweighter is None:
                    w = None
                elif isinstance(reweighter, Reweighter):
                    w = reweighter.reweight(df)
                else:
                    raise ValueError("Unsupported reweighter type.")
                ds = lgb.Dataset(x.values, label=y.to_numpy(), weight=w, free_raw_data=False)
                self._index_by_dataset_id[id(ds)] = df.index
                ds_l.append((ds, key))
        return ds_l

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=None,
        early_stopping_rounds=None,
        verbose_eval=20,
        evals_result=None,
        reweighter=None,
        **kwargs,
    ):
        if evals_result is None:
            evals_result = {}  # in case of unsafety of Python default values
        ds_l = self._prepare_data(dataset, reweighter)
        ds, names = list(zip(*ds_l))
        eval_ds, eval_names = ds, names
        if self.rank_eval and len(ds) > 1:
            eval_ds, eval_names = ds[1:], names[1:]
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds
        )
        # NOTE: if you encounter error here. Please upgrade your lightgbm
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        train_kwargs = dict(kwargs)
        if self.rank_eval and "feval" not in train_kwargs:
            train_kwargs["feval"] = DailyRankICEval(self._index_by_dataset_id)
        self.model = lgb.train(
            self.params,
            ds[0],  # training dataset
            num_boost_round=self.num_boost_round if num_boost_round is None else num_boost_round,
            valid_sets=eval_ds,
            valid_names=eval_names,
            callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback],
            **train_kwargs,
        )
        for k in eval_names:
            for key, val in evals_result[k].items():
                name = f"{key}.{k}"
                for epoch, m in enumerate(val):
                    R.log_metrics(**{name.replace("@", "_"): m}, step=epoch)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20, reweighter=None):
        """
        finetune model

        Parameters
        ----------
        dataset : DatasetH
            dataset for finetuning
        num_boost_round : int
            number of round to finetune model
        verbose_eval : int
            verbose level
        """
        # Based on existing model and finetune by train more rounds
        ds_l = self._prepare_data(dataset, reweighter)
        dtrain, _ = ds_l[0]

        if dtrain.construct().num_data() == 0:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=self.model,
            valid_sets=[dtrain],
            valid_names=["train"],
            callbacks=[verbose_eval_callback],
            feval=DailyRankICEval(self._index_by_dataset_id) if self.rank_eval else None,
        )
