# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Text, Union

import pandas as pd

from ...data.dataset import DatasetH
from ...log import get_module_logger
from ...model.base import Model


class PretrainedSignalModel(Model):
    """Serve a precomputed signal series/dataframe as a Qlib model."""

    def __init__(self, signal_path: str, score_column: str = "score"):
        self.signal_path = signal_path
        self.score_column = score_column
        self.logger = get_module_logger("PretrainedSignalModel")
        self.prediction: Optional[pd.Series] = None

    def fit(self, dataset: DatasetH):
        signal = pickle.load(Path(self.signal_path).open("rb"))
        if isinstance(signal, pd.DataFrame):
            if self.score_column not in signal.columns:
                raise ValueError(f"Column `{self.score_column}` not found in pretrained signal.")
            signal = signal[self.score_column]
        elif not isinstance(signal, pd.Series):
            raise TypeError("Pretrained signal must be a pandas Series or DataFrame.")
        self.prediction = signal.sort_index()
        self.logger.info("Loaded pretrained signal from %s", self.signal_path)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.prediction is None:
            raise ValueError("model is not fitted yet!")
        return self.prediction.copy()
