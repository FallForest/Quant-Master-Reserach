# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import pandas as pd

from .base import BaseStrategySelector


class SeriesStrategySelector(BaseStrategySelector):
    def __init__(
        self,
        signal: pd.Series,
        mapping: dict | None = None,
        default_strategy: str | None = None,
        default_family: str | None = None,
        default_variant: str | None = None,
        reason: str = "series_selector",
    ):
        if not isinstance(signal, pd.Series):
            raise TypeError("series selector requires `signal` to be a pandas Series")
        self.signal = signal
        self.mapping = mapping
        self.default_strategy = default_strategy
        self.default_family = default_family
        self.default_variant = default_variant
        self.reason = reason

    @staticmethod
    def _normalize_label(label):
        if isinstance(label, pd.DataFrame):
            if label.shape[1] != 1:
                raise ValueError("selector dataframe must contain exactly one column")
            label = label.iloc[:, 0]
        if isinstance(label, pd.Series):
            raise TypeError("series object should be resolved before label normalization")
        if pd.isna(label):
            return None
        return label

    @staticmethod
    def _lookup_series_value(series: pd.Series, trade_start_time: pd.Timestamp):
        ts = pd.Timestamp(trade_start_time)
        candidates = [ts, ts.normalize(), ts.date()]
        for key in candidates:
            try:
                return series.loc[key]
            except KeyError:
                continue
        return None

    def _resolve_label(self, trade_start_time: pd.Timestamp):
        label = self._lookup_series_value(self.signal, trade_start_time)
        if isinstance(label, pd.Series):
            label = label.iloc[0]
        label = self._normalize_label(label)
        if label is None:
            return None
        if self.mapping is not None:
            label = self.mapping.get(label, label)
        return label

    def select(self, trade_start_time, context=None):
        resolved = self._resolve_label(pd.Timestamp(trade_start_time))
        selection = {
            "reason": self.reason,
            "selector_name": self.__class__.__name__,
        }
        if resolved is None:
            if self.default_strategy is not None:
                selection["strategy_key"] = self.default_strategy
            if self.default_family is not None:
                selection["family"] = self.default_family
            if self.default_variant is not None:
                selection["variant"] = self.default_variant
            return selection
        if isinstance(resolved, dict):
            selection.update(resolved)
            return selection
        if isinstance(resolved, tuple) and len(resolved) == 2:
            selection["family"], selection["variant"] = resolved
            return selection
        selection["strategy_key"] = resolved
        return selection
