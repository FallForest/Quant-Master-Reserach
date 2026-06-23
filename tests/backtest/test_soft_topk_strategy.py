import pandas as pd
import pytest
from quant_master.contrib.strategy.cost_control import SoftTopkStrategy


class MockPosition:
    def __init__(self, weights):
        self.weights = weights

    def get_stock_weight_dict(self, only_stock=True):
        return self.weights


def create_test_strategy(
    topk=2,
    risk_degree=1.0,
    impact_limit=1.0,
    rank_buffer=0,
    max_new_names=None,
    dynamic_risk_gate=None,
):
    strat = SoftTopkStrategy.__new__(SoftTopkStrategy)
    strat.topk = topk
    strat.risk_degree = risk_degree
    strat.trade_impact_limit = impact_limit
    strat.selection_rank_buffer = rank_buffer
    strat.selection_max_new_names = max_new_names
    strat.dynamic_risk_gate = dynamic_risk_gate
    strat._dynamic_risk_scale = 1.0
    strat._dynamic_equity_history = []
    strat._last_effective_risk_degree = risk_degree
    return strat


def test_soft_topk_logic():
    # Initial: A=0.8, B=0.2 (Total=1.0). Target Risk=0.95.
    # Scores: A and B are low, C and D are topk.
    scores = pd.Series({"C": 0.9, "D": 0.8, "A": 0.1, "B": 0.1})
    current_pos = MockPosition({"A": 0.8, "B": 0.2})

    topk = 2
    risk_degree = 0.95
    impact_limit = 0.1  # Max change per step

    # 1. With impact limit: Expect deterministic sell and limited buy
    strat_i = create_test_strategy(topk=topk, risk_degree=risk_degree, impact_limit=impact_limit)
    res_i = strat_i.generate_target_weight_position(scores, current_pos, None, None)

    # A should be exactly 0.8 - 0.1 = 0.7
    assert abs(res_i["A"] - 0.7) < 1e-8
    # B should be exactly 0.2 - 0.1 = 0.1
    assert abs(res_i["B"] - 0.1) < 1e-8
    # Total sells = 0.2 released. New budget = 0.2 + (0.95 - 1.0) = 0.15.
    # C and D share 0.15 -> 0.075 each.
    assert abs(res_i["C"] - 0.075) < 1e-8
    assert abs(res_i["D"] - 0.075) < 1e-8

    # 2. Without impact limit: Expect full liquidation and full target fill
    strat_c = create_test_strategy(topk=topk, risk_degree=risk_degree, impact_limit=1.0)
    res_c = strat_c.generate_target_weight_position(scores, current_pos, None, None)

    # A, B not in topk -> Liquidated
    assert "A" not in res_c and "B" not in res_c
    # C, D should reach ideal_per_stock (0.95/2 = 0.475)
    assert abs(res_c["C"] - 0.475) < 1e-8
    assert abs(res_c["D"] - 0.475) < 1e-8


def test_soft_topk_selection_rank_buffer_is_opt_in():
    scores = pd.Series({"A": 0.99, "B": 0.98, "C": 0.97, "D": 0.1})
    current_pos = MockPosition({"C": 0.5, "D": 0.5})

    default_strat = create_test_strategy(topk=2)
    default_res = default_strat.generate_target_weight_position(scores, current_pos, None, None)

    buffered_strat = create_test_strategy(topk=2, rank_buffer=1)
    buffered_res = buffered_strat.generate_target_weight_position(scores, current_pos, None, None)

    assert set(default_res) == {"A", "B"}
    assert set(buffered_res) == {"A", "C"}
    assert abs(buffered_res["C"] - 0.5) < 1e-8


def test_soft_topk_accepts_single_column_dataframe_signal():
    scores = pd.DataFrame({"score": {"A": 0.99, "B": 0.98, "C": 0.97}})
    current_pos = MockPosition({})

    strat = create_test_strategy(topk=2, risk_degree=0.9)
    res = strat.generate_target_weight_position(scores, current_pos, None, None)

    assert set(res) == {"A", "B"}
    assert abs(res["A"] - 0.45) < 1e-8
    assert abs(res["B"] - 0.45) < 1e-8


def test_soft_topk_selection_max_new_names_none_preserves_default_behavior():
    scores = pd.Series({"A": 0.99, "B": 0.98, "C": 0.97, "D": 0.96})
    current_pos = MockPosition({"C": 0.5, "D": 0.5})

    default_strat = create_test_strategy(topk=3, max_new_names=None)
    explicit_none_strat = create_test_strategy(topk=3, max_new_names=None)

    default_res = default_strat.generate_target_weight_position(scores, current_pos, None, None)
    explicit_none_res = explicit_none_strat.generate_target_weight_position(scores, current_pos, None, None)

    assert explicit_none_res == default_res
    assert set(explicit_none_res) == {"A", "B", "C"}


def test_soft_topk_selection_max_new_names_none_cold_start_fills_topk():
    scores = pd.Series({"A": 0.99, "B": 0.98, "C": 0.97})
    current_pos = MockPosition({})

    strat = create_test_strategy(topk=2, risk_degree=0.9, max_new_names=None)
    res = strat.generate_target_weight_position(scores, current_pos, None, None)

    assert set(res) == {"A", "B"}
    assert abs(res["A"] - 0.45) < 1e-8
    assert abs(res["B"] - 0.45) < 1e-8


def test_soft_topk_selection_max_new_names_one_cold_start_fills_one_name():
    scores = pd.Series({"A": 0.99, "B": 0.98, "C": 0.97})
    current_pos = MockPosition({})

    strat = create_test_strategy(topk=2, risk_degree=0.9, max_new_names=1)
    res = strat.generate_target_weight_position(scores, current_pos, None, None)

    assert res == {"A": 0.45}


def test_soft_topk_selection_max_new_names_zero_cold_start_returns_no_holdings():
    scores = pd.Series({"A": 0.99, "B": 0.98, "C": 0.97})
    current_pos = MockPosition({})

    strat = create_test_strategy(topk=2, risk_degree=0.9, max_new_names=0)
    res = strat.generate_target_weight_position(scores, current_pos, None, None)

    assert res == {}


def test_soft_topk_selection_max_new_names_caps_entrants_and_keeps_holdings():
    scores = pd.Series({"A": 0.99, "B": 0.98, "C": 0.97, "D": 0.96, "E": 0.95})
    current_pos = MockPosition({"C": 0.5, "D": 0.5})

    strat = create_test_strategy(topk=3, max_new_names=1)
    res = strat.generate_target_weight_position(scores, current_pos, None, None)

    new_names = set(res) - set(current_pos.weights)
    assert len(new_names) <= 1
    assert set(res) == {"A", "C", "D"}
    assert abs(res["A"] - (1.0 / 3.0)) < 1e-8
    assert abs(res["C"] - (1.0 / 3.0)) < 1e-8
    assert abs(res["D"] - (1.0 / 3.0)) < 1e-8


def test_soft_topk_selection_max_new_names_combines_with_rank_buffer():
    scores = pd.Series({"A": 0.99, "B": 0.98, "C": 0.97, "D": 0.96, "E": 0.1})
    current_pos = MockPosition({"C": 1.0 / 3.0, "D": 1.0 / 3.0, "E": 1.0 / 3.0})

    strat = create_test_strategy(topk=3, rank_buffer=1, max_new_names=1)
    res = strat.generate_target_weight_position(scores, current_pos, None, None)

    assert set(res) == {"A", "C", "D"}
    assert "B" not in res
    assert len(set(res) - set(current_pos.weights)) == 1


def test_soft_topk_selection_max_new_names_rejects_negative_values():
    with pytest.raises(ValueError, match="selection_max_new_names must be non-negative"):
        SoftTopkStrategy(selection_max_new_names=-1)


def test_soft_topk_dynamic_risk_gate_disabled_preserves_behavior():
    scores = pd.Series({"A": 0.99, "B": 0.98, "C": 0.1})
    current_pos = MockPosition({})

    baseline = create_test_strategy(topk=2, risk_degree=0.9)
    disabled = create_test_strategy(
        topk=2,
        risk_degree=0.9,
        dynamic_risk_gate={
            "enabled": False,
            "mode": "drawdown",
            "lookback": 20,
            "drawdown_threshold": 0.08,
            "full_clamp_threshold": 0.16,
            "min_risk_degree": 0.45,
        },
    )

    assert disabled.generate_target_weight_position(scores, current_pos, None, None) == baseline.generate_target_weight_position(
        scores, current_pos, None, None
    )
    assert disabled.get_risk_degree() == pytest.approx(0.9)


def test_soft_topk_dynamic_risk_gate_threshold_crossing_lowers_effective_risk():
    scores = pd.Series({"A": 0.99, "B": 0.98})
    gate = {
        "enabled": True,
        "mode": "drawdown",
        "lookback": 20,
        "drawdown_threshold": 0.08,
        "full_clamp_threshold": 0.16,
        "min_risk_degree": 0.45,
        "recovery_rate": 0.10,
        "decay_rate": 1.0,
    }
    strat = create_test_strategy(topk=2, risk_degree=1.0, dynamic_risk_gate=gate)

    peak_pos = MockPosition({})
    peak_pos.position = {"now_account_value": 1000.0}
    res = strat.generate_target_weight_position(scores, peak_pos, None, None)

    assert res["A"] == pytest.approx(0.5)
    assert strat.get_risk_degree() == pytest.approx(1.0)

    falling_pos = MockPosition({})
    falling_pos.position = {"now_account_value": 900.0}
    res = strat.generate_target_weight_position(scores, falling_pos, None, None)

    assert strat.get_risk_degree() == pytest.approx(0.8625)
    assert res["A"] == pytest.approx(strat.get_risk_degree() / 2.0)
    assert res["B"] == pytest.approx(strat.get_risk_degree() / 2.0)


def test_soft_topk_dynamic_risk_gate_full_clamp_and_gradual_recovery():
    scores = pd.Series({"A": 0.99, "B": 0.98})
    gate = {
        "enabled": True,
        "mode": "drawdown",
        "lookback": 20,
        "drawdown_threshold": 0.08,
        "full_clamp_threshold": 0.16,
        "min_risk_degree": 0.45,
        "recovery_rate": 0.10,
        "decay_rate": 1.0,
    }
    strat = create_test_strategy(topk=2, risk_degree=1.0, dynamic_risk_gate=gate)

    peak_pos = MockPosition({})
    peak_pos.position = {"now_account_value": 1000.0}
    clamp_pos = MockPosition({})
    clamp_pos.position = {"now_account_value": 800.0}
    recovered_pos = MockPosition({})
    recovered_pos.position = {"now_account_value": 1000.0}

    strat.generate_target_weight_position(scores, peak_pos, None, None)
    clamped = strat.generate_target_weight_position(scores, clamp_pos, None, None)
    assert strat.get_risk_degree() == pytest.approx(0.45)
    assert clamped["A"] == pytest.approx(0.225)

    recovering = strat.generate_target_weight_position(scores, recovered_pos, None, None)
    assert strat.get_risk_degree() == pytest.approx(0.505)
    assert recovering["A"] == pytest.approx(strat.get_risk_degree() / 2.0)


if __name__ == "__main__":
    pytest.main([__file__])
