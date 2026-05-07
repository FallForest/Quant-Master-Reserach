# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function

import copy
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.weight import Reweighter
from ...log import get_module_logger
from ...model.base import Model
from ...model.utils import ConcatDataset
from ...utils import get_or_create_path
from .pytorch_utils import count_parameters


class CnLSTM(Model):
    """CNN + LSTM time-series model for Qlib TSDatasetH."""

    def __init__(
        self,
        d_feat=158,
        conv_channels=64,
        kernel_size=3,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        prefetch_factor=2,
        persistent_workers=None,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        self.logger = get_module_logger("CnLSTM")
        self.logger.info("CnLSTM pytorch version...")

        self.d_feat = d_feat
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = self._resolve_num_workers(n_jobs)
        self.prefetch_factor = prefetch_factor
        if persistent_workers is None:
            persistent_workers = os.name != "nt"
        self.persistent_workers = bool(persistent_workers)
        self.seed = seed

        self.logger.info(
            "CnLSTM parameters setting:"
            "\nd_feat : {}"
            "\nconv_channels : {}"
            "\nkernel_size : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nprefetch_factor : {}"
            "\npersistent_workers : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                conv_channels,
                kernel_size,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                self.optimizer,
                loss,
                self.device,
                self.n_jobs,
                prefetch_factor,
                self.persistent_workers,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        if self.use_gpu:
            torch.backends.cudnn.benchmark = True

        self.model = CnLSTMModel(
            d_feat=self.d_feat,
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.logger.info("model:\n{:}".format(self.model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.model)))

        if self.optimizer == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    @staticmethod
    def _resolve_num_workers(n_jobs: int) -> int:
        if n_jobs <= 0:
            return 0
        if os.name == "nt":
            return min(n_jobs, 4)
        return n_jobs

    def mse(self, pred, label, weight):
        return torch.mean(weight * (pred - label) ** 2)

    def loss_fn(self, pred, label, weight):
        mask = ~torch.isnan(label)
        if weight is None:
            weight = torch.ones_like(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask], weight=None)
        raise ValueError("unknown metric `%s`" % self.metric)

    def _data_loader(self, data, weight=None, shuffle=False, drop_last=False):
        if weight is not None:
            data = ConcatDataset(data, weight)
        loader_kwargs = dict(
            dataset=data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.n_jobs,
            drop_last=drop_last,
            pin_memory=self.use_gpu,
        )
        if self.n_jobs > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**loader_kwargs)

    def train_epoch(self, data_loader):
        self.model.train()

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device, non_blocking=self.use_gpu)
            label = data[:, -1, -1].to(self.device, non_blocking=self.use_gpu)
            weight = weight.to(self.device, non_blocking=self.use_gpu)

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label, weight)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.model.eval()

        scores = []
        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device, non_blocking=self.use_gpu)
            label = data[:, -1, -1].to(self.device, non_blocking=self.use_gpu)
            weight = weight.to(self.device, non_blocking=self.use_gpu)

            with torch.no_grad():
                pred = self.model(feature.float())
                score = self.metric_fn(pred, label)

            scores.append(score.item())

        return np.mean(scores)

    def fit(self, dataset, evals_result=None, save_path=None, reweighter=None):
        if evals_result is None:
            evals_result = {}

        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = self._data_loader(dl_train, wl_train, shuffle=True, drop_last=True)
        valid_loader = self._data_loader(dl_valid, wl_valid, shuffle=False, drop_last=True)
        save_path = get_or_create_path(save_path)

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = copy.deepcopy(self.model.state_dict())
        evals_result["train"] = []
        evals_result["valid"] = []

        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            val_score = self.test_epoch(valid_loader)
            self.logger.info("valid %.6f" % val_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = self._data_loader(dl_test, shuffle=False, drop_last=False)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device, non_blocking=self.use_gpu)
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index()).sort_index()


class CnLSTMModel(nn.Module):
    def __init__(self, d_feat=158, conv_channels=64, kernel_size=3, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(d_feat, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze(-1)
