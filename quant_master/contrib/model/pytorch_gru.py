# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function
from typing import Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from quant_master.workflow import R

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from ...utils import get_or_create_path
from .pytorch_utils import count_parameters, deepcopy_state_dict


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
            "\nvisible_GPU : {}"
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
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.gru_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.logger.info("model:\n{:}".format(self.gru_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.gru_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.gru_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.gru_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.gru_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    @property
    def uses_rank_loss(self):
        return self.loss in {"rank_ic", "mse_rank_ic"}

    def rank_ic_loss(self, pred, label):
        """Differentiable daily cross-sectional correlation loss.

        The validation metric remains actual Spearman RankIC. Pearson correlation is
        used here because sorting/ranking is not differentiable.
        """

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

    def loss_fn(self, pred, label):
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        if self.loss == "rank_ic":
            return self.rank_ic_loss(pred, label)
        if self.loss == "mse_rank_ic":
            mse_loss = self.mse(pred[mask], label[mask])
            rank_loss = self.rank_ic_loss(pred, label)
            return (1.0 - self.rank_loss_weight) * mse_loss + self.rank_loss_weight * rank_loss

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred, label)

        raise ValueError("unknown metric `%s`" % self.metric)

    def _daily_batch_indices(self, index, shuffle=False):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("RankIC loss and metric require a MultiIndex with a datetime level.")
        date_level = "datetime" if "datetime" in index.names else index.names[0]
        dates = index.get_level_values(date_level)
        positions = pd.Series(np.arange(len(index)), index=dates)
        batches = [group.to_numpy() for _, group in positions.groupby(level=0, sort=False)]
        if shuffle:
            np.random.shuffle(batches)
        return batches

    def _batch_indices(self, index, shuffle=False):
        if self.uses_rank_loss:
            return self._daily_batch_indices(index, shuffle=shuffle)
        indices = np.arange(len(index))
        if shuffle:
            np.random.shuffle(indices)
        return [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

    def daily_rank_ic(self, pred, label, index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("RankIC metric requires a MultiIndex with a datetime level.")
        date_level = "datetime" if "datetime" in index.names else index.names[0]
        values = pd.DataFrame({"pred": pred, "label": label}, index=index).replace([np.inf, -np.inf], np.nan)
        rank_ic = []
        for _, daily in values.groupby(level=date_level, sort=False):
            daily = daily.dropna()
            if (
                len(daily) < self.min_rank_samples
                or daily["pred"].nunique() < 2
                or daily["label"].nunique() < 2
            ):
                continue
            value = daily["pred"].corr(daily["label"], method="spearman")
            if np.isfinite(value):
                rank_ic.append(float(value))
        return float(np.mean(rank_ic)) if rank_ic else float("-inf")

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.gru_model.train()

        for indices in self._batch_indices(x_train.index, shuffle=True):
            feature = torch.from_numpy(x_train_values[indices]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices]).float().to(self.device)

            pred = self.gru_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.gru_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.gru_model.eval()

        scores = []
        losses = []
        pred_values = np.full(len(x_values), np.nan, dtype=float)
        label_values = np.full(len(y_values), np.nan, dtype=float)

        for indices in self._batch_indices(data_x.index, shuffle=False):
            feature = torch.from_numpy(x_values[indices]).float().to(self.device)
            label = torch.from_numpy(y_values[indices]).float().to(self.device)

            with torch.no_grad():
                pred = self.gru_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                pred_values[indices] = pred.detach().cpu().numpy()
                label_values[indices] = label.detach().cpu().numpy()

                if self.metric not in {"rank_ic", "rankic"}:
                    score = self.metric_fn(pred, label)
                    scores.append(score.item())

        if self.metric in {"rank_ic", "rankic"}:
            score = self.daily_rank_ic(pred_values, label_values, data_x.index)
        else:
            score = np.mean(scores)
        return np.mean(losses), score

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        # prepare training and validation data
        dfs = {
            k: dataset.prepare(
                k,
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            for k in ["train", "valid"]
            if k in dataset.segments
        }
        df_train, df_valid = dfs.get("train", pd.DataFrame()), dfs.get("valid", pd.DataFrame())

        # check if training data is empty
        if df_train.empty:
            raise ValueError("Empty training data from dataset, please check your dataset config.")

        df_train = df_train.dropna()
        x_train, y_train = df_train["feature"], df_train["label"]

        # check if validation data is provided
        if not df_valid.empty:
            df_valid = df_valid.dropna()
            x_valid, y_valid = df_valid["feature"], df_valid["label"]
        else:
            x_valid, y_valid = None, None

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

        best_param = deepcopy_state_dict(self.gru_model)
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            evals_result["train"].append(train_score)

            # evaluate on validation data if provided
            if x_valid is not None and y_valid is not None:
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
                evals_result["valid"].append(val_score)

                if val_score > best_score:
                    best_score = val_score
                    stop_steps = 0
                    best_epoch = step
                    best_param = deepcopy_state_dict(self.gru_model)
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        self.logger.info("early stop")
                        break
            elif train_score > best_score:
                best_score = train_score
                best_epoch = step
                best_param = deepcopy_state_dict(self.gru_model)

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.gru_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        # Logging
        rec = R.get_recorder()
        for k, v_l in evals_result.items():
            for i, v in enumerate(v_l):
                rec.log_metrics(step=i, **{k: v})

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.gru_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                pred = self.gru_model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


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
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
