# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from abc import ABC
from typing import Dict, List, Text, Union

import pandas as pd

from quant_master.backtest.decision import TradeDecisionWO
from quant_master.contrib.strategy.selectors import BaseStrategySelector, FixedStrategySelector, SeriesStrategySelector
from quant_master.strategy.base import BaseStrategy
from quant_master.utils import init_instance_by_config


class BaseStrategyRouter(BaseStrategy, ABC):
    def __init__(
        self,
        *,
        strategies: Dict[Text, Union[BaseStrategy, dict]] | None = None,
        strategy_families: Dict[Text, dict] | None = None,
        selector=None,
        default_strategy: Text | None = None,
        default_family: Text | None = None,
        default_variant: Text | None = None,
        fallback_policy: dict | None = None,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        super().__init__(level_infra=level_infra, common_infra=common_infra, trade_exchange=trade_exchange, **kwargs)
        self._strategy_registry: Dict[Text, dict] = {}
        self._family_defaults: Dict[Text, Text] = {}
        self._default_selection = None
        self.fallback_policy = {
            "on_selector_missing": "use_default",
            "on_selector_invalid": "use_default",
        }
        if fallback_policy is not None:
            self.fallback_policy.update(fallback_policy)
        self._init_strategy_registry(
            strategies=strategies,
            strategy_families=strategy_families,
            default_strategy=default_strategy,
            default_family=default_family,
            default_variant=default_variant,
        )
        self.selector = self._init_selector(selector)
        self.selected_strategy_history: List[dict] = []
        self.route_history = self.selected_strategy_history

    def _init_strategy_registry(
        self,
        *,
        strategies: Dict[Text, Union[BaseStrategy, dict]] | None,
        strategy_families: Dict[Text, dict] | None,
        default_strategy: Text | None,
        default_family: Text | None,
        default_variant: Text | None,
    ) -> None:
        if strategies is not None and strategy_families is not None:
            raise ValueError("use either `strategies` or `strategy_families`, not both")
        if strategy_families:
            self._register_strategy_families(strategy_families)
            self._default_selection = self._resolve_default_family_variant(default_family, default_variant)
        elif strategies:
            self._register_legacy_strategies(strategies)
            self._default_selection = self._resolve_default_strategy(default_strategy)
        else:
            raise ValueError("`strategies` or `strategy_families` must not be empty")

    def _register_legacy_strategies(self, strategies: Dict[Text, Union[BaseStrategy, dict]]) -> None:
        for name, strategy in strategies.items():
            self._strategy_registry[name] = {
                "strategy": init_instance_by_config(strategy, accept_types=BaseStrategy),
                "strategy_key": name,
                "family": name,
                "variant": "default",
            }
            self._family_defaults[name] = "default"

    def _register_strategy_families(self, strategy_families: Dict[Text, dict]) -> None:
        for family_name, family_conf in strategy_families.items():
            variants = family_conf.get("variants")
            if not variants:
                raise ValueError(f"strategy family `{family_name}` must define non-empty `variants`")
            default_variant = family_conf.get("default_variant") or next(iter(variants))
            if default_variant not in variants:
                raise KeyError(f"default variant `{default_variant}` is not defined for family `{family_name}`")
            self._family_defaults[family_name] = default_variant
            for variant_name, strategy in variants.items():
                strategy_key = f"{family_name}.{variant_name}"
                self._strategy_registry[strategy_key] = {
                    "strategy": init_instance_by_config(strategy, accept_types=BaseStrategy),
                    "strategy_key": strategy_key,
                    "family": family_name,
                    "variant": variant_name,
                }

    def _resolve_default_strategy(self, default_strategy: Text | None) -> dict:
        default_key = default_strategy or next(iter(self._strategy_registry))
        if default_key not in self._strategy_registry:
            raise KeyError(f"default strategy `{default_key}` is not defined")
        registry_item = self._strategy_registry[default_key]
        return {
            "strategy_key": registry_item["strategy_key"],
            "family": registry_item["family"],
            "variant": registry_item["variant"],
        }

    def _resolve_default_family_variant(self, default_family: Text | None, default_variant: Text | None) -> dict:
        family = default_family or next(iter(self._family_defaults))
        if family not in self._family_defaults:
            raise KeyError(f"default strategy family `{family}` is not defined")
        variant = default_variant or self._family_defaults[family]
        strategy_key = f"{family}.{variant}"
        if strategy_key not in self._strategy_registry:
            raise KeyError(f"default strategy `{strategy_key}` is not defined")
        return {
            "strategy_key": strategy_key,
            "family": family,
            "variant": variant,
        }

    def _init_selector(self, selector):
        if selector is None:
            return FixedStrategySelector(strategy_key=self._default_selection["strategy_key"], reason="default_selector")
        if isinstance(selector, BaseStrategySelector):
            return selector
        if isinstance(selector, dict) and "class" in selector:
            return init_instance_by_config(selector, accept_types=BaseStrategySelector)
        if isinstance(selector, str):
            return FixedStrategySelector(strategy_key=selector)
        if isinstance(selector, pd.Series):
            return SeriesStrategySelector(signal=selector)
        if isinstance(selector, dict):
            selector_type = selector.get("type", "fixed")
            if selector_type == "fixed":
                return FixedStrategySelector(
                    strategy_key=selector.get("strategy"),
                    family=selector.get("family"),
                    variant=selector.get("variant"),
                    reason=selector.get("reason", "fixed_selector"),
                )
            if selector_type == "series":
                return SeriesStrategySelector(
                    signal=selector.get("signal", selector.get("series")),
                    mapping=selector.get("mapping"),
                    default_strategy=selector.get("default_strategy"),
                    default_family=selector.get("default_family"),
                    default_variant=selector.get("default_variant"),
                )
            raise NotImplementedError(f"selector type `{selector_type}` is not supported")
        raise TypeError(f"unsupported selector type: {type(selector)}")

    def _get_trade_start_time(self) -> pd.Timestamp:
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, _ = self.trade_calendar.get_step_time(trade_step)
        return pd.Timestamp(trade_start_time)

    def _normalize_selection_result(self, selection_result, trade_start_time: pd.Timestamp) -> dict:
        if selection_result is None:
            return {}
        if isinstance(selection_result, str):
            return {
                "strategy_key": selection_result,
                "reason": "selector_returned_string",
                "selection_date": pd.Timestamp(trade_start_time),
            }
        if not isinstance(selection_result, dict):
            raise TypeError(f"selector returned unsupported result type: {type(selection_result)}")
        normalized = dict(selection_result)
        normalized.setdefault("selection_date", pd.Timestamp(trade_start_time))
        normalized.setdefault("selector_name", self.selector.__class__.__name__)
        return normalized

    def _use_default_for_missing(self) -> bool:
        return self.fallback_policy.get("on_selector_missing", "use_default") == "use_default"

    def _use_default_for_invalid(self) -> bool:
        return self.fallback_policy.get("on_selector_invalid", "use_default") == "use_default"

    def _resolve_selection(self, selection_result: dict) -> dict:
        fallback_used = False
        candidate = dict(selection_result)
        if not candidate.get("strategy_key") and not candidate.get("family"):
            if not self._use_default_for_missing():
                raise ValueError("selector did not return strategy selection")
            candidate.update(self._default_selection)
            fallback_used = True
        if candidate.get("strategy_key") in self._strategy_registry:
            registry_item = self._strategy_registry[candidate["strategy_key"]]
        else:
            family = candidate.get("family")
            variant = candidate.get("variant")
            if family and variant is None and family in self._family_defaults:
                variant = self._family_defaults[family]
            strategy_key = candidate.get("strategy_key") or (
                f"{family}.{variant}" if family is not None and variant is not None else None
            )
            if strategy_key in self._strategy_registry:
                registry_item = self._strategy_registry[strategy_key]
            elif candidate.get("strategy_key") in self._family_defaults:
                family = candidate["strategy_key"]
                variant = self._family_defaults[family]
                registry_item = self._strategy_registry[f"{family}.{variant}"]
            else:
                if not self._use_default_for_invalid():
                    raise KeyError(f"selected strategy `{candidate}` is not defined")
                fallback_used = True
                registry_item = self._strategy_registry[self._default_selection["strategy_key"]]
        resolved = {
            "strategy_key": registry_item["strategy_key"],
            "strategy_family": registry_item["family"],
            "strategy_variant": registry_item["variant"],
            "reason": candidate.get("reason"),
            "selector_name": candidate.get("selector_name", self.selector.__class__.__name__),
            "fallback_used": fallback_used,
        }
        features = candidate.get("features")
        if features is not None:
            resolved["features"] = features
        return resolved

    def _record_selected_strategy(self, trade_start_time: pd.Timestamp, selection: dict) -> None:
        record = {
            "datetime": pd.Timestamp(trade_start_time),
            "strategy_key": selection["strategy_key"],
            "strategy_family": selection["strategy_family"],
            "strategy_variant": selection["strategy_variant"],
            "selector_name": selection.get("selector_name"),
            "reason": selection.get("reason"),
            "fallback_used": selection.get("fallback_used", False),
        }
        if "features" in selection:
            record["features"] = selection["features"]
        self.selected_strategy_history.append(record)

    def get_route_history(self) -> pd.DataFrame:
        if not self.selected_strategy_history:
            return pd.DataFrame(
                columns=[
                    "strategy_key",
                    "strategy_family",
                    "strategy_variant",
                    "selector_name",
                    "reason",
                    "fallback_used",
                ]
            )
        route_df = pd.DataFrame(self.selected_strategy_history).drop_duplicates(subset=["datetime"], keep="last")
        return route_df.set_index("datetime").sort_index()

    def get_route_summary(self) -> pd.DataFrame:
        route_df = self.get_route_history()
        if route_df.empty:
            return pd.DataFrame(columns=["selection_count"])
        summary = (
            route_df.groupby(["strategy_family", "strategy_variant", "strategy_key"])
            .size()
            .rename("selection_count")
            .to_frame()
            .sort_values("selection_count", ascending=False)
        )
        return summary

    def _get_strategy(self, strategy_key: Text) -> BaseStrategy:
        registry_item = self._strategy_registry[strategy_key]
        strategy = registry_item["strategy"]
        strategy.reset(
            level_infra=getattr(self, "level_infra", None),
            common_infra=getattr(self, "common_infra", None),
            outer_trade_decision=getattr(self, "outer_trade_decision", None),
        )
        return strategy

    def generate_trade_decision(self, execute_result=None):
        if not hasattr(self, "level_infra"):
            return TradeDecisionWO([], self)
        trade_start_time = self._get_trade_start_time()
        selection_result = self.selector.select(
            trade_start_time=trade_start_time,
            context={"execute_result": execute_result, "router": self},
        )
        normalized = self._normalize_selection_result(selection_result, trade_start_time)
        resolved = self._resolve_selection(normalized)
        self._record_selected_strategy(trade_start_time=trade_start_time, selection=resolved)
        return self._get_strategy(resolved["strategy_key"]).generate_trade_decision(execute_result=execute_result)


class DailyRebalanceRouterStrategy(BaseStrategyRouter):
    pass
