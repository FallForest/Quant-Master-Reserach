# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from .base import BaseStrategySelector


class FixedStrategySelector(BaseStrategySelector):
    def __init__(self, strategy_key=None, family=None, variant=None, reason=None):
        self.strategy_key = strategy_key
        self.family = family
        self.variant = variant
        self.reason = reason or "fixed_selector"

    def select(self, trade_start_time, context=None):
        return {
            "strategy_key": self.strategy_key,
            "family": self.family,
            "variant": self.variant,
            "reason": self.reason,
            "selector_name": self.__class__.__name__,
        }
