# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStrategySelector(ABC):
    @abstractmethod
    def select(self, trade_start_time, context=None):
        raise NotImplementedError("select is not implemented")
