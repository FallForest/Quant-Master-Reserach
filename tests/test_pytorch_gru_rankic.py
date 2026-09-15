from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from quant_master.contrib.model.pytorch_gru import GRU
from quant_master.contrib.model.pytorch_gru_ts import DailyBatchSampler
from quant_master.contrib.model.pytorch_gru_ts import GRU as TSGRU


def _model(**kwargs) -> GRU:
    return GRU(d_feat=1, hidden_size=2, num_layers=1, n_epochs=1, GPU=-1, **kwargs)


def _ts_model(**kwargs) -> TSGRU:
    return TSGRU(d_feat=1, hidden_size=2, num_layers=1, n_epochs=1, GPU=-1, n_jobs=0, **kwargs)


def _panel_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            ("2024-01-02", "A"),
            ("2024-01-03", "A"),
            ("2024-01-02", "B"),
            ("2024-01-03", "B"),
            ("2024-01-02", "C"),
            ("2024-01-03", "C"),
        ],
        names=["datetime", "instrument"],
    )


def test_rank_ic_loss_prefers_cross_sectionally_aligned_predictions() -> None:
    model = _model(loss="rank_ic")
    label = torch.tensor([-1.0, 0.0, 2.0])
    aligned = torch.tensor([-2.0, 0.0, 1.0], requires_grad=True)
    reversed_pred = -aligned.detach()

    aligned_loss = model.loss_fn(aligned, label)
    reversed_loss = model.loss_fn(reversed_pred, label)
    aligned_loss.backward()

    assert aligned_loss.item() < reversed_loss.item()
    assert torch.isfinite(aligned.grad).all()


def test_rank_loss_batches_keep_each_trading_day_together() -> None:
    model = _model(loss="rank_ic")

    batches = model._batch_indices(_panel_index(), shuffle=False)

    assert [batch.tolist() for batch in batches] == [[0, 2, 4], [1, 3, 5]]


class _IndexSource:
    def get_index(self) -> pd.MultiIndex:
        return _panel_index()


def test_ts_daily_batch_sampler_keeps_each_trading_day_together() -> None:
    sampler = DailyBatchSampler(_IndexSource(), shuffle=False)

    assert list(sampler) == [[0, 2, 4], [1, 3, 5]]
    assert len(sampler) == 2


def test_daily_rank_ic_is_equal_weighted_across_valid_days() -> None:
    model = _model(metric="rank_ic", min_rank_samples=3)
    index = _panel_index()
    label = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    pred = np.array([1.0, 3.0, 2.0, 2.0, 3.0, 1.0])

    score = model.daily_rank_ic(pred, label, index)

    assert score == pytest.approx(0.0)


class _FirstFeature(nn.Module):
    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature[:, 0]


def test_rank_ic_metric_scores_full_daily_cross_sections_across_inference_batches() -> None:
    model = _model(metric="rank_ic", loss="mse", batch_size=2)
    model.gru_model = _FirstFeature()
    index = _panel_index()
    feature = pd.DataFrame({"feature": [1.0, 3.0, 2.0, 2.0, 3.0, 1.0]}, index=index)
    label = pd.DataFrame({"label": [1.0, 3.0, 2.0, 2.0, 3.0, 1.0]}, index=index)

    _, score = model.test_epoch(feature, label)

    assert score == pytest.approx(1.0)


def test_default_loss_keeps_mse_behavior() -> None:
    model = _model()
    pred = torch.tensor([1.0, 3.0])
    label = torch.tensor([0.0, 1.0])

    assert not model.uses_rank_loss
    assert model.loss_fn(pred, label).item() == pytest.approx(2.5)


def test_training_without_valid_keeps_trained_parameters(tmp_path, monkeypatch) -> None:
    model = _model()
    index = pd.MultiIndex.from_tuples(
        [("2020-01-02", "A"), ("2020-01-02", "B")], names=["datetime", "instrument"]
    )
    columns = pd.MultiIndex.from_tuples([("feature", "F0"), ("label", "LABEL0")])
    frame = pd.DataFrame([[1.0, 0.1], [2.0, 0.2]], index=index, columns=columns)
    dataset = SimpleNamespace(segments={"train": ("2020-01-02", "2020-01-02")})
    dataset.prepare = lambda *args, **kwargs: frame

    def train_epoch(*args, **kwargs):
        with torch.no_grad():
            for parameter in model.gru_model.parameters():
                parameter.fill_(1.0)

    model.train_epoch = train_epoch
    model.test_epoch = lambda *args, **kwargs: (0.0, 1.0)
    monkeypatch.setattr(
        "quant_master.contrib.model.pytorch_gru.R",
        SimpleNamespace(get_recorder=lambda: SimpleNamespace(log_metrics=lambda **kwargs: None)),
    )

    model.fit(dataset, save_path=tmp_path / "gru.bin")

    assert all(torch.all(parameter == 1.0) for parameter in model.gru_model.parameters())


def test_ts_rank_loss_and_metric_use_daily_cross_section() -> None:
    model = _ts_model(loss="rank_ic", metric="rank_ic")
    label = torch.tensor([-1.0, 0.0, 2.0])
    aligned = torch.tensor([-2.0, 0.0, 1.0], requires_grad=True)

    loss = model.loss_fn(aligned, label)
    score = model.batch_rank_ic(aligned, label)
    loss.backward()

    assert loss.item() < 1.0
    assert score == pytest.approx(1.0)
    assert torch.isfinite(aligned.grad).all()


def test_ts_default_loss_keeps_weighted_mse_behavior() -> None:
    model = _ts_model()
    pred = torch.tensor([1.0, 3.0])
    label = torch.tensor([0.0, 1.0])
    weight = torch.tensor([2.0, 0.5])

    assert not model.uses_rank_loss
    assert not model.uses_rank_metric
    assert model.loss_fn(pred, label, weight).item() == pytest.approx(2.0)


@pytest.mark.parametrize("kwargs", [{"rank_loss_weight": -0.1}, {"rank_loss_weight": 1.1}, {"min_rank_samples": 1}])
def test_rank_hyperparameters_are_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _model(**kwargs)
