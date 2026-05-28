import re

from quant_master.contrib.data.handler import Alpha158
from quant_master.contrib.data.liquidity_state_handler import Alpha158LiquidityState


def test_alpha158_liquidity_state_handler_instantiates_without_loading_data():
    handler = Alpha158LiquidityState(
        instruments="csi300",
        start_time="2012-01-01",
        end_time="2026-04-30",
        fit_start_time="2012-01-01",
        fit_end_time="2020-12-31",
        init_data=False,
    )

    assert isinstance(handler, Alpha158)


def test_alpha158_liquidity_state_appends_expected_feature_names():
    base_fields, base_names = Alpha158(init_data=False).get_feature_config()
    fields, names = Alpha158LiquidityState(init_data=False).get_feature_config()
    extra_fields, extra_names = Alpha158LiquidityState.get_liquidity_state_feature_config()

    assert len(extra_names) == 17
    assert len(fields) == len(base_fields) + len(extra_fields)
    assert len(names) == len(base_names) + len(extra_names)
    assert all(name.startswith(Alpha158LiquidityState.EXTRA_FEATURE_PREFIX) for name in extra_names)
    assert extra_names == names[-len(extra_names) :]
    assert "LS_AMIHUD_20" in names
    assert "LS_DVOL_REL20" in names
    assert "LS_RET_VOLUME_CORR20" in names


def test_alpha158_liquidity_state_features_do_not_use_obvious_future_references():
    fields, _ = Alpha158LiquidityState.get_liquidity_state_feature_config()
    future_patterns = [
        re.compile(r"Ref\s*\([^)]*,\s*-", re.IGNORECASE),
        re.compile(r"shift\s*\(\s*-", re.IGNORECASE),
        re.compile(r"LABEL", re.IGNORECASE),
    ]

    for expr in fields:
        for pattern in future_patterns:
            assert pattern.search(expr) is None, expr
