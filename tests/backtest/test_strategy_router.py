import pandas as pd
import pytest

from quant_master.backtest.utils import CommonInfrastructure, LevelInfrastructure, TradeCalendarManager
from quant_master.contrib.strategy.router_strategy import DailyRebalanceRouterStrategy
from quant_master.strategy.base import BaseStrategy
from quant_master.tests import TestAutoData


class DummyAccount:
    def __init__(self):
        self.current_position = object()


class DummyStrategy(BaseStrategy):
    def __init__(self, strategy_name, **kwargs):
        super().__init__(**kwargs)
        self.strategy_name = strategy_name
        self.calls = []

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, _ = self.trade_calendar.get_step_time(trade_step)
        record = {
            "strategy_name": self.strategy_name,
            "trade_start_time": pd.Timestamp(trade_start_time),
            "execute_result": execute_result,
            "level_infra_is_router": self.level_infra.get("trade_calendar") is not None,
            "common_infra_is_router": self.common_infra.get("trade_account") is not None,
        }
        self.calls.append(record)
        return record


def create_router(selector, default_strategy="first"):
    TestAutoData.setUpClass(enable_1d_type="simple")
    trade_calendar = TradeCalendarManager(freq="day", start_time="2005-01-04", end_time="2005-01-06")
    common_infra = CommonInfrastructure(trade_account=DummyAccount(), trade_exchange=object())
    level_infra = LevelInfrastructure(trade_calendar=trade_calendar, common_infra=common_infra, executor=object())
    router = DailyRebalanceRouterStrategy(
        strategies={
            "first": DummyStrategy(strategy_name="first"),
            "second": DummyStrategy(strategy_name="second"),
        },
        selector=selector,
        default_strategy=default_strategy,
    )
    router.reset(common_infra=common_infra, level_infra=level_infra)
    return router, trade_calendar


def create_family_router(selector=None):
    TestAutoData.setUpClass(enable_1d_type="simple")
    trade_calendar = TradeCalendarManager(freq="day", start_time="2005-01-04", end_time="2005-01-06")
    common_infra = CommonInfrastructure(trade_account=DummyAccount(), trade_exchange=object())
    level_infra = LevelInfrastructure(trade_calendar=trade_calendar, common_infra=common_infra, executor=object())
    router = DailyRebalanceRouterStrategy(
        selector=selector,
        default_family="topk",
        default_variant="aggressive",
        strategy_families={
            "topk": {
                "default_variant": "aggressive",
                "variants": {
                    "aggressive": DummyStrategy(strategy_name="topk-aggressive"),
                    "defensive": DummyStrategy(strategy_name="topk-defensive"),
                },
            },
            "buffered": {
                "default_variant": "base",
                "variants": {
                    "base": DummyStrategy(strategy_name="buffered-base"),
                },
            },
        },
    )
    router.reset(common_infra=common_infra, level_infra=level_infra)
    return router, trade_calendar


def test_fixed_selector_routes_to_expected_strategy():
    router, _ = create_router({"type": "fixed", "strategy": "second"})

    result = router.generate_trade_decision(execute_result=["ok"])

    assert result["strategy_name"] == "second"
    assert result["execute_result"] == ["ok"]
    assert router.selected_strategy_history[0]["strategy_key"] == "second"


def test_series_selector_switches_by_trade_date():
    series = pd.Series(
        ["second", "first", "second"],
        index=pd.to_datetime(["2005-01-04", "2005-01-05", "2005-01-06"]),
    )
    router, trade_calendar = create_router({"type": "series", "signal": series})

    results = []
    for _ in range(3):
        results.append(router.generate_trade_decision())
        trade_calendar.step()

    assert [item["strategy_name"] for item in results] == ["second", "first", "second"]
    route_history = router.get_route_history()
    assert list(route_history["strategy_key"]) == ["second", "first", "second"]


def test_series_selector_falls_back_to_default_strategy_when_missing():
    series = pd.Series(["second"], index=pd.to_datetime(["2005-01-04"]))
    router, trade_calendar = create_router({"type": "series", "signal": series}, default_strategy="first")

    first_day = router.generate_trade_decision()
    trade_calendar.step()
    second_day = router.generate_trade_decision()

    assert first_day["strategy_name"] == "second"
    assert second_day["strategy_name"] == "first"
    route_history = router.get_route_history()
    assert route_history.iloc[-1]["fallback_used"]


def test_series_selector_mapping_and_infra_propagation():
    series = pd.Series([1], index=pd.to_datetime(["2005-01-04"]))
    router, _ = create_router({"type": "series", "signal": series, "mapping": {1: "second"}})

    result = router.generate_trade_decision()

    assert result["strategy_name"] == "second"
    assert result["level_infra_is_router"] is True
    assert result["common_infra_is_router"] is True


def test_strategy_family_selector_uses_family_and_variant_metadata():
    series = pd.Series(
        [
            {"family": "buffered", "variant": "base", "reason": "cost_regime", "features": {"cost": 0.8}},
            {"family": "topk", "variant": "defensive", "reason": "risk_regime"},
        ],
        index=pd.to_datetime(["2005-01-04", "2005-01-05"]),
    )
    router, trade_calendar = create_family_router(selector={"type": "series", "signal": series})

    first_day = router.generate_trade_decision()
    trade_calendar.step()
    second_day = router.generate_trade_decision()

    assert first_day["strategy_name"] == "buffered-base"
    assert second_day["strategy_name"] == "topk-defensive"
    route_history = router.get_route_history()
    assert list(route_history["strategy_family"]) == ["buffered", "topk"]
    assert list(route_history["strategy_variant"]) == ["base", "defensive"]
    assert route_history.iloc[0]["reason"] == "cost_regime"
    assert route_history.iloc[0]["features"] == {"cost": 0.8}


def test_strategy_family_summary_counts_selected_variants():
    series = pd.Series(
        [
            {"family": "buffered", "variant": "base"},
            {"family": "topk", "variant": "defensive"},
            {"family": "buffered", "variant": "base"},
        ],
        index=pd.to_datetime(["2005-01-04", "2005-01-05", "2005-01-06"]),
    )
    router, trade_calendar = create_family_router(selector={"type": "series", "signal": series})

    for _ in range(3):
        router.generate_trade_decision()
        trade_calendar.step()

    summary = router.get_route_summary()
    assert summary.loc[("buffered", "base", "buffered.base"), "selection_count"] == 2
    assert summary.loc[("topk", "defensive", "topk.defensive"), "selection_count"] == 1


def test_invalid_family_selection_falls_back_to_default_family_variant():
    series = pd.Series([{"family": "unknown", "variant": "base"}], index=pd.to_datetime(["2005-01-04"]))
    router, _ = create_family_router(selector={"type": "series", "signal": series})

    result = router.generate_trade_decision()

    assert result["strategy_name"] == "topk-aggressive"
    route_history = router.get_route_history()
    assert bool(route_history.iloc[0]["fallback_used"]) is True
    assert route_history.iloc[0]["strategy_key"] == "topk.aggressive"


def test_strategy_families_require_variants():
    with pytest.raises(ValueError, match="variants"):
        DailyRebalanceRouterStrategy(strategy_families={"topk": {"variants": {}}})
