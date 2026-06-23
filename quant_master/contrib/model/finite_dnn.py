# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy
from typing import Optional, Sequence, Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from .pytorch_nn import Net


class FiniteDNNModelPytorch(Model):
    """Small PyTorch tabular model with finite-value guards.

    This is intended as a low-risk deep auxiliary for tree anchors. It keeps
    feature cleanup local to the neural branch so shared workflow handlers do
    not have to change the CPU model's data contract.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        layers: Sequence[int] = (64, 16),
        output_dim: int = 1,
        lr: float = 0.001,
        max_steps: int = 32,
        batch_size: int = 4096,
        early_stop_rounds: int = 6,
        eval_steps: int = 2,
        optimizer: str = "adam",
        loss: str = "mse",
        GPU: int = 0,
        seed: Optional[int] = None,
        weight_decay: float = 0.0,
        feature_fill_value: float = 0.0,
        feature_daily_zscore: bool = True,
        label_rank: bool = True,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if early_stop_rounds <= 0:
            raise ValueError("early_stop_rounds must be positive.")
        if eval_steps <= 0:
            raise ValueError("eval_steps must be positive.")
        if loss != "mse":
            raise NotImplementedError("FiniteDNNModelPytorch currently supports only mse loss.")

        self.logger = get_module_logger("FiniteDNNModelPytorch")
        self.input_dim = None if input_dim is None else int(input_dim)
        self.layers = tuple(int(x) for x in layers)
        self.output_dim = int(output_dim)
        self.lr = float(lr)
        self.max_steps = int(max_steps)
        self.batch_size = int(batch_size)
        self.early_stop_rounds = int(early_stop_rounds)
        self.eval_steps = int(eval_steps)
        self.optimizer = str(optimizer).lower()
        self.loss = str(loss).lower()
        self.seed = seed
        self.weight_decay = float(weight_decay)
        self.feature_fill_value = float(feature_fill_value)
        self.feature_daily_zscore = bool(feature_daily_zscore)
        self.label_rank = bool(label_rank)
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.dnn_model: Optional[nn.Module] = None
        self.train_optimizer = None
        self.fitted = False

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        if self.input_dim is not None:
            self._build_network(self.input_dim)

        self.logger.info(
            "FiniteDNNModelPytorch settings:"
            "\ninput_dim : %s"
            "\nlayers : %s"
            "\nmax_steps : %s"
            "\nbatch_size : %s"
            "\nearly_stop_rounds : %s"
            "\neval_steps : %s"
            "\noptimizer : %s"
            "\ndevice : %s"
            "\nuse_GPU : %s",
            self.input_dim,
            self.layers,
            self.max_steps,
            self.batch_size,
            self.early_stop_rounds,
            self.eval_steps,
            self.optimizer,
            self.device,
            self.use_gpu,
        )

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def _build_network(self, input_dim: int) -> None:
        self.input_dim = int(input_dim)
        self.dnn_model = Net(input_dim=self.input_dim, output_dim=self.output_dim, layers=self.layers).to(self.device)
        if self.optimizer == "adam":
            self.train_optimizer = optim.Adam(self.dnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "gd":
            self.train_optimizer = optim.SGD(self.dnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(f"optimizer {self.optimizer} is not supported.")

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            return None
        return "datetime" if "datetime" in index.names else index.names[0]

    def _clean_feature_frame(self, feature: pd.DataFrame) -> pd.DataFrame:
        x = feature.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if self.feature_daily_zscore and isinstance(x.index, pd.MultiIndex):
            date_level = self._date_level(x.index)

            def _zscore(group: pd.DataFrame) -> pd.DataFrame:
                std = group.std(axis=0).replace(0.0, np.nan)
                return (group - group.mean(axis=0)) / std

            x = x.groupby(level=date_level, group_keys=False).apply(_zscore)
        return x.replace([np.inf, -np.inf], np.nan).fillna(self.feature_fill_value).astype("float32")

    def _clean_label_series(self, label: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if isinstance(label, pd.DataFrame):
            y = label.iloc[:, 0]
        else:
            y = label
        y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if self.label_rank and isinstance(y.index, pd.MultiIndex):
            date_level = self._date_level(y.index)
            y = y.groupby(level=date_level).rank(pct=True) - 0.5
        return y.astype("float32")

    def _prepare_xy(self, dataset: DatasetH, segment: Union[Text, slice]):
        df = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        x = self._clean_feature_frame(df["feature"])
        y = self._clean_label_series(df["label"])
        common = x.index.intersection(y.index)
        x = x.loc[common]
        y = y.loc[common]
        mask = np.isfinite(y.values)
        x = x.loc[mask]
        y = y.loc[mask]
        if x.empty:
            raise ValueError(f"No finite rows for FiniteDNNModelPytorch segment {segment}.")
        return x, y

    def _prepare_x(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.DataFrame:
        x = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return self._clean_feature_frame(x)

    def fit(self, dataset: DatasetH, evals_result=dict(), verbose=True, **kwargs):
        x_train, y_train = self._prepare_xy(dataset, "train")
        x_valid, y_valid = self._prepare_xy(dataset, "valid") if "valid" in dataset.segments else (None, None)
        if self.input_dim is None:
            self._build_network(x_train.shape[1])
        if x_train.shape[1] != self.input_dim:
            raise ValueError(f"input_dim={self.input_dim} does not match feature width={x_train.shape[1]}.")

        x_train_t = torch.from_numpy(x_train.values).float().to(self.device)
        y_train_t = torch.from_numpy(y_train.values.reshape(-1, 1)).float().to(self.device)
        x_valid_t = torch.from_numpy(x_valid.values).float().to(self.device) if x_valid is not None else None
        y_valid_t = torch.from_numpy(y_valid.values.reshape(-1, 1)).float().to(self.device) if y_valid is not None else None

        rng = np.random.default_rng(self.seed)
        best_loss = float("inf")
        best_state = copy.deepcopy(self.dnn_model.state_dict())
        stop_rounds = 0
        train_n = len(y_train_t)
        loss_fn = nn.MSELoss()
        evals_result.setdefault("train", [])
        if x_valid_t is not None:
            evals_result.setdefault("valid", [])

        for step in range(1, self.max_steps + 1):
            self.dnn_model.train()
            replace = train_n < self.batch_size
            choice = rng.choice(train_n, size=min(self.batch_size, train_n), replace=replace)
            choice_t = torch.as_tensor(choice, dtype=torch.long, device=self.device)
            pred = self.dnn_model(x_train_t.index_select(0, choice_t))
            loss = loss_fn(pred, y_train_t.index_select(0, choice_t))
            if not torch.isfinite(loss):
                raise ValueError("FiniteDNNModelPytorch encountered non-finite training loss.")
            self.train_optimizer.zero_grad()
            loss.backward()
            self.train_optimizer.step()

            if step % self.eval_steps == 0 or step == self.max_steps:
                self.dnn_model.eval()
                with torch.no_grad():
                    train_eval = loss_fn(self.dnn_model(x_train_t), y_train_t).item()
                    valid_eval = (
                        loss_fn(self.dnn_model(x_valid_t), y_valid_t).item() if x_valid_t is not None else train_eval
                    )
                evals_result["train"].append(float(train_eval))
                if x_valid_t is not None:
                    evals_result["valid"].append(float(valid_eval))
                if verbose:
                    self.logger.info("[Step %d]: train_loss %.6f, valid_loss %.6f", step, train_eval, valid_eval)
                if valid_eval < best_loss:
                    best_loss = float(valid_eval)
                    best_state = copy.deepcopy(self.dnn_model.state_dict())
                    stop_rounds = 0
                else:
                    stop_rounds += 1
                if stop_rounds >= self.early_stop_rounds:
                    break

        self.dnn_model.load_state_dict(best_state)
        self.fitted = True
        if self.use_gpu:
            torch.cuda.empty_cache()

    def _nn_predict(self, x: pd.DataFrame) -> np.ndarray:
        data = torch.from_numpy(x.values).float().to(self.device)
        preds = []
        self.dnn_model.eval()
        with torch.no_grad():
            for i in range(0, len(data), 8192):
                preds.append(self.dnn_model(data[i : i + 8192]).detach().reshape(-1).cpu().numpy())
        return np.concatenate(preds)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x = self._prepare_x(dataset, segment)
        return pd.Series(self._nn_predict(x), index=x.index)
