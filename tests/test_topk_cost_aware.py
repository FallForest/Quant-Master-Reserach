import pandas as pd

from quant_master.contrib.strategy.topk_cost_aware import (
    estimate_turnover,
    rerank_topk_with_turnover_limit,
    transform_scores_for_cost,
)


def test_transform_scores_for_cost_holding_boost_and_turnover_penalty():
    scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.9})
    previous_holdings = pd.Series({"A": 1.0, "B": 1.0})

    adjusted = transform_scores_for_cost(
        scores=scores,
        previous_holdings=previous_holdings,
        previous_holding_boost=0.6,
        turnover_penalty=0.3,
        normalize_scores=True,
    )

    # A and B are boosted/penalized less than new name C; ranking can change.
    ranked = adjusted.sort_values(ascending=False).index.tolist()
    assert ranked[0] in {"A", "B"}


def test_transform_scores_for_cost_volatility_penalty():
    scores = pd.Series({"A": 0.9, "B": 0.8, "C": 0.7})
    vol = pd.Series({"A": 0.9, "B": 0.1, "C": 0.1})

    adjusted = transform_scores_for_cost(
        scores=scores,
        volatility=vol,
        volatility_penalty=0.8,
        normalize_scores=True,
    )

    # A has highest volatility and should lose enough rank edge.
    assert adjusted["A"] < adjusted["B"]


def test_transform_scores_for_cost_smoothing_uses_previous_scores_only():
    scores = pd.Series({"A": 0.1, "B": 0.9})
    prev_scores = pd.Series({"A": 0.9, "B": 0.1})

    adjusted = transform_scores_for_cost(
        scores=scores,
        previous_scores=prev_scores,
        smoothing_alpha=1.0,
        normalize_scores=False,
    )

    # alpha=1 means fully previous scores (no future leakage inputs).
    assert adjusted["A"] == prev_scores["A"]
    assert adjusted["B"] == prev_scores["B"]


def test_rerank_topk_with_turnover_limit():
    adjusted = pd.Series({"A": 0.99, "B": 0.98, "C": 0.97, "D": 0.96})
    previous_holdings = ["C", "D"]

    selected = rerank_topk_with_turnover_limit(
        adjusted_scores=adjusted,
        topk=2,
        previous_holdings=previous_holdings,
        max_new_names=1,
    )

    # At most one newly-entered name is allowed.
    new_count = len(set(selected.tolist()) - set(previous_holdings))
    assert new_count <= 1

    turnover = estimate_turnover(selected, previous_holdings=previous_holdings)
    assert 0.0 <= turnover <= 1.0
