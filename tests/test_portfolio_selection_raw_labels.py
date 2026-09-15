import pandas as pd

from quant_master.contrib.model.transcendence_hybrid import TranscendenceHybridModel
from quant_master.contrib.model.transcendence_signal_ensemble import TranscendenceSignalEnsembleModel
from quant_master.data.dataset.handler import DataHandlerLP


class _LabelDataset:
    def __init__(self):
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2022-01-03"]), ["a", "b"]], names=["datetime", "instrument"]
        )
        self.frames = {
            DataHandlerLP.DK_R: pd.DataFrame({("label", "LABEL0"): [0.01, -0.02]}, index=index),
            DataHandlerLP.DK_L: pd.DataFrame({("label", "LABEL0"): [1.0, -1.0]}, index=index),
        }
        self.requested_keys = []

    def prepare(self, segment, col_set, data_key):
        self.requested_keys.append(data_key)
        return self.frames[data_key]


def test_signal_ensemble_portfolio_labels_use_raw_returns():
    dataset = _LabelDataset()
    model = TranscendenceSignalEnsembleModel()

    label = model._prepare_label_series(dataset, "valid")

    assert dataset.requested_keys == [DataHandlerLP.DK_R]
    assert label.tolist() == [0.01, -0.02]


def test_hybrid_separates_portfolio_and_training_label_keys():
    dataset = _LabelDataset()
    model = TranscendenceHybridModel()

    portfolio_label = model._prepare_label_series(dataset, "valid", data_key=DataHandlerLP.DK_R)
    training_label = model._prepare_label_series(dataset, "valid", data_key=DataHandlerLP.DK_L)

    assert dataset.requested_keys == [DataHandlerLP.DK_R, DataHandlerLP.DK_L]
    assert portfolio_label.tolist() == [0.01, -0.02]
    assert training_label.tolist() == [1.0, -1.0]
