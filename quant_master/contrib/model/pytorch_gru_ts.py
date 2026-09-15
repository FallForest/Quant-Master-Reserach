# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from .pytorch_utils import count_parameters, deepcopy_state_dict
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter


class DailyBatchSampler(Sampler):
    """Yield complete trading-day cross-sections for rank-aware training."""

    def __init__(self, data_source, shuffle=False):
        index = data_source.get_index()
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("RankIC loss and metric require a MultiIndex with a datetime level.")
        date_level = "datetime" if "datetime" in index.names else index.names[0]
        dates = index.get_level_values(date_level)
        positions = pd.Series(np.arange(len(index)), index=dates)
        self.batches = [group.to_numpy().tolist() for _, group in positions.groupby(level=0, sort=False)]
        self.shuffle = shuffle

    def __iter__(self):
        order = np.arange(len(self.batches))
        if self.shuffle:
            np.random.shuffle(order)
        for idx in order:
            yield self.batches[idx]

    def __len__(self):
        return len(self.batches)


class GRU(Model):
    """GRU Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        rank_loss_weight=0.5,
        min_rank_samples=3,
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GRU")
        self.logger.info("GRU pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
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
        self.rank_loss_weight = float(rank_loss_weight)
        self.min_rank_samples = int(min_rank_samples)
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed
        if not 0.0 <= self.rank_loss_weight <= 1.0:
            raise ValueError("rank_loss_weight must be between 0 and 1.")
        if self.min_rank_samples < 2:
            raise ValueError("min_rank_samples must be at least 2.")

        self.logger.info(
            "GRU parameters setting:"
            "\nd_feat : {}"
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
            "\nrank_loss_weight : {}"
            "\nmin_rank_samples : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                rank_loss_weight,
                min_rank_samples,
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.GRU_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.logger.info("model:\n{:}".format(self.GRU_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.GRU_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.GRU_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.GRU_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.GRU_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    @property
    def uses_rank_loss(self):
        return self.loss in {"rank_ic", "mse_rank_ic"}

    @property
    def uses_rank_metric(self):
        return self.metric in {"rank_ic", "rankic"}

    def rank_ic_loss(self, pred, label):
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        mask = torch.isfinite(pred) & torch.isfinite(label)
        pred = pred[mask]
        label = label[mask]
        if pred.numel() < self.min_rank_samples:
            return pred.sum() * 0.0
        pred = pred - pred.mean()
        label = label - label.mean()
        denominator = torch.linalg.vector_norm(pred) * torch.linalg.vector_norm(label)
        if denominator.detach().item() <= torch.finfo(pred.dtype).eps:
            return pred.sum() * 0.0
        return 1.0 - torch.sum(pred * label) / denominator

    def loss_fn(self, pred, label, weight=None):
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        mask = ~torch.isnan(label)

        if weight is None:
            weight = torch.ones_like(label)
        else:
            weight = weight.reshape(-1)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])
        if self.loss == "rank_ic":
            return self.rank_ic_loss(pred, label)
        if self.loss == "mse_rank_ic":
            mse_loss = self.mse(pred[mask], label[mask], weight[mask])
            rank_loss = self.rank_ic_loss(pred, label)
            return (1.0 - self.rank_loss_weight) * mse_loss + self.rank_loss_weight * rank_loss

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred, label)

        raise ValueError("unknown metric `%s`" % self.metric)

    def batch_rank_ic(self, pred, label):
        values = pd.DataFrame(
            {
                "pred": pred.detach().cpu().numpy().reshape(-1),
                "label": label.detach().cpu().numpy().reshape(-1),
            }
        ).replace([np.inf, -np.inf], np.nan)
        values = values.dropna()
        if (
            len(values) < self.min_rank_samples
            or values["pred"].nunique() < 2
            or values["label"].nunique() < 2
        ):
            return np.nan
        return float(values["pred"].corr(values["label"], method="spearman"))

    def train_epoch(self, data_loader):
        self.GRU_model.train()

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.GRU_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GRU_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.GRU_model.eval()

        scores = []
        losses = []

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())

                if self.uses_rank_metric:
                    score = self.batch_rank_ic(pred, label)
                    if np.isfinite(score):
                        scores.append(score)
                else:
                    score = self.metric_fn(pred, label)
                    scores.append(score.item())

        return np.mean(losses), np.mean(scores) if scores else float("-inf")

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        if self.uses_rank_loss:
            train_loader = DataLoader(
                ConcatDataset(dl_train, wl_train),
                batch_sampler=DailyBatchSampler(dl_train, shuffle=True),
                num_workers=self.n_jobs,
            )
        else:
            train_loader = DataLoader(
                ConcatDataset(dl_train, wl_train),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.n_jobs,
                drop_last=True,
            )

        if self.uses_rank_metric and not self.uses_rank_loss:
            train_eval_loader = DataLoader(
                ConcatDataset(dl_train, wl_train),
                batch_sampler=DailyBatchSampler(dl_train, shuffle=False),
                num_workers=self.n_jobs,
            )
        else:
            train_eval_loader = train_loader

        if self.uses_rank_loss or self.uses_rank_metric:
            valid_loader = DataLoader(
                ConcatDataset(dl_valid, wl_valid),
                batch_sampler=DailyBatchSampler(dl_valid, shuffle=False),
                num_workers=self.n_jobs,
            )
        else:
            valid_loader = DataLoader(
                ConcatDataset(dl_valid, wl_valid),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.n_jobs,
                drop_last=True,
            )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        best_param = deepcopy_state_dict(self.GRU_model)
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_eval_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = deepcopy_state_dict(self.GRU_model)
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.GRU_model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
