# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function

import copy
import math
from contextlib import contextmanager
from typing import Text, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from ...utils import get_or_create_path


class TFTModel(Model):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 2048,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        use_amp=True,
        **kwargs,
    ):
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.use_amp = bool(use_amp and self.device.type == "cuda" and hasattr(torch.cuda, "amp"))
        self.logger = get_module_logger("TFTModel")
        self.logger.info(
            "Temporal Fusion Transformer:"
            "\nbatch_size : {}"
            "\ndevice : {}"
            "\nuse_amp : {}".format(self.batch_size, self.device, self.use_amp)
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = TemporalFusionTransformer(d_feat, d_model, nhead, num_layers, dropout)
        if self.optimizer == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif self.optimizer == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.scaler = torch.cuda.amp.GradScaler(enabled=True) if self.use_amp else None
        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            self.train_optimizer.zero_grad()
            with self.autocast():
                pred = self.model(feature)
                loss = self.loss_fn(pred, label)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.train_optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.scaler.step(self.train_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                with self.autocast():
                    pred = self.model(feature)
                    loss = self.loss_fn(pred, label)
                    score = self.metric_fn(pred, label)
                losses.append(loss.item())
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

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
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
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

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.model.eval()
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
                with self.autocast():
                    pred = self.model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)

    @contextmanager
    def autocast(self):
        if self.use_amp:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.fc2(self.dropout(self.elu(self.fc1(x))))
        x = x * torch.sigmoid(self.gate(x))
        return self.norm(x + residual)


class GateAddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(GateAddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, residual):
        x = self.dropout(x)
        x = x * torch.sigmoid(self.gate(x))
        return self.norm(x + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_feat, d_model, dropout=0.1):
        super(VariableSelectionNetwork, self).__init__()
        self.weight_layer = nn.Linear(d_feat, d_feat)
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.grn = GatedResidualNetwork(d_model, dropout)

    def forward(self, x):
        weights = torch.softmax(self.weight_layer(x), dim=-1)
        x = self.feature_layer(x * weights)
        return self.grn(x)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, d_feat=6, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TemporalFusionTransformer, self).__init__()
        self.d_feat = d_feat
        self.variable_selection = VariableSelectionNetwork(d_feat, d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.lstm_gate = GateAddNorm(d_model, dropout)
        self.pos_encoder = PositionalEncoding(d_model)
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.batch_first_attention = True
        except TypeError:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
            )
            self.batch_first_attention = False
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_gate = GateAddNorm(d_model, dropout)
        self.position_grn = GatedResidualNetwork(d_model, dropout)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        selected = self.variable_selection(src)
        lstm_out, _ = self.lstm(selected)
        temporal = self.lstm_gate(lstm_out, selected)
        attention_input = self.pos_encoder(temporal)
        if self.batch_first_attention:
            attended = self.attention(attention_input)
        else:
            attended = self.attention(attention_input.transpose(1, 0)).transpose(1, 0)
        fused = self.attention_gate(attended, temporal)
        output = self.position_grn(fused[:, -1, :])
        return self.decoder(output).squeeze(-1)
