import re

import pytest

from quant_master.contrib.data.market_microstructure_handler import Alpha158MarketMicrostructureProxy
from quant_master.config import C
from quant_master.data.data import LocalExpressionProvider
from quant_master.data.ops import register_all_ops


RAW_FIELD = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
FUTURE_REF = re.compile(r"Ref\s*\([^)]*,\s*-", re.IGNORECASE)
ALLOWED_RAW_FIELDS = {"open", "high", "low", "close", "volume", "amount"}


def test_handler_appends_ten_tdx_microstructure_proxies():
    fields, names = Alpha158MarketMicrostructureProxy(init_data=False).get_feature_config()

    assert len(fields) == len(names) == 168
    assert "$vwap" not in " ".join(fields).lower()
    assert len(set(names)) == len(names)
    assert all(name.startswith("MSP_") for name in names[158:])


def test_proxy_registry_exposes_family_metadata_and_ablation():
    registry = Alpha158MarketMicrostructureProxy.get_market_microstructure_feature_registry()

    assert len(registry) == 10
    assert registry.family_counts() == {
        "roll_spread": 1,
        "corwin_schultz_spread": 1,
        "parkinson_volatility": 1,
        "garman_klass_volatility": 1,
        "rogers_satchell_volatility": 1,
        "price_staleness": 1,
        "price_limit_distance": 2,
        "return_dependence": 1,
        "jump_risk": 1,
    }
    assert len(registry.families) == 9
    assert all(spec.source == "tdx_daily_proxy" for spec in registry)
    assert all(spec.available_lag == 0 for spec in registry)

    fields, names = Alpha158MarketMicrostructureProxy.get_market_microstructure_feature_config(
        exclude_feature_families=("parkinson_volatility", "price_limit_distance")
    )
    assert len(fields) == len(names) == 7
    assert not {"MSP_PARKINSON20", "MSP_UPLIMIT10_DIST"}.intersection(names)

    handler = Alpha158MarketMicrostructureProxy(feature_families="garman_klass_volatility", init_data=False)
    _, handler_names = handler.get_feature_config()
    assert handler_names[158:] == ["MSP_GARMAN_KLASS20"]


def test_proxy_expressions_are_causal_and_only_use_tdx_daily_fields():
    fields, _ = Alpha158MarketMicrostructureProxy.get_market_microstructure_feature_config()

    assert all(FUTURE_REF.search(expression) is None for expression in fields)
    assert set().union(*(set(RAW_FIELD.findall(expression)) for expression in fields)) <= ALLOWED_RAW_FIELDS
    forbidden = ("$bid", "$ask", "orderbook", "order_book", "$limit", "$pit")
    assert all(token not in expression.lower() for expression in fields for token in forbidden)


def test_benchmark_features_are_explicitly_optional():
    _, names_without_benchmark = Alpha158MarketMicrostructureProxy.get_market_microstructure_feature_config()
    fields, names = Alpha158MarketMicrostructureProxy.get_market_microstructure_feature_config(benchmark="SH000300")

    assert "MSP_MARKET_BETA20" not in names_without_benchmark
    assert "MSP_IDIO_VOL20" not in names_without_benchmark
    assert len(fields) == len(names) == 12
    assert names[-2:] == ["MSP_MARKET_BETA20", "MSP_IDIO_VOL20"]
    assert all("ChangeInstrument('SH000300'" in expression for expression in fields[-2:])

    with pytest.raises(ValueError, match="unsupported characters"):
        Alpha158MarketMicrostructureProxy.get_market_microstructure_feature_config(benchmark="bad'instrument")


def test_proxy_columns_can_be_selected_by_screened_feature_name():
    recommended = (
        "MSP_IDIO_VOL20",
        "MSP_CORWIN_SCHULTZ20",
        "MSP_UPLIMIT10_DIST",
        "MSP_ROLL_SPREAD20",
        "MSP_DOWNLIMIT10_DIST",
    )
    _, names = Alpha158MarketMicrostructureProxy.get_market_microstructure_feature_config(
        benchmark="SH000300", feature_names=recommended
    )

    assert names == [
        "MSP_ROLL_SPREAD20",
        "MSP_CORWIN_SCHULTZ20",
        "MSP_UPLIMIT10_DIST",
        "MSP_DOWNLIMIT10_DIST",
        "MSP_IDIO_VOL20",
    ]

    handler = Alpha158MarketMicrostructureProxy(
        benchmark="SH000300", feature_names=recommended, init_data=False
    )
    _, handler_names = handler.get_feature_config()
    assert len(handler_names) == 163
    assert handler_names[-5:] == names

    with pytest.raises(ValueError, match="unknown feature name.*MSP_UNKNOWN"):
        Alpha158MarketMicrostructureProxy.get_market_microstructure_feature_config(
            benchmark="SH000300", feature_names=("MSP_UNKNOWN",)
        )


def test_all_proxy_expressions_parse_with_registered_operators():
    fields, _ = Alpha158MarketMicrostructureProxy.get_market_microstructure_feature_config(benchmark="SH000300")
    register_all_ops(C)
    parser = LocalExpressionProvider()

    parsed = [parser.get_expression_instance(expression) for expression in fields]

    assert len(parsed) == 12
    assert all(str(expression) for expression in parsed)
