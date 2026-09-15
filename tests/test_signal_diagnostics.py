from __future__ import annotations

import math

import numpy as np
import pandas as pd

from examples.benchmarks.Transcendence.support.signal_diagnostics import (
    build_signal_panel,
    daily_ic_frame,
    decile_diagnostics,
    evaluate_signal,
)


def _panel(days: int = 4, names: int = 30) -> tuple[pd.Series, pd.Series]:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=days, freq="D"), [f"s{i:03d}" for i in range(names)]],
        names=["datetime", "instrument"],
    )
    values = np.tile(np.arange(names, dtype=float), days)
    pred = pd.Series(values, index=index, name="score")
    label = pd.Series(values * 0.01, index=index, name="label")
    return pred, label


def test_evaluate_signal_reports_ic_rank_ic_year_and_decile() -> None:
    pred, label = _panel()

    metrics = evaluate_signal(pred, label, min_count=20, deciles=10)

    assert math.isclose(metrics["ic"], 1.0)
    assert math.isclose(metrics["rank_ic"], 1.0)
    assert metrics["ic_days"] == 4
    assert metrics["rank_ic_days"] == 4
    assert "2024" in metrics["by_year"]
    assert metrics["decile"]["days"] == 4
    assert metrics["decile"]["top_bottom_spread"] > 0
    assert math.isclose(metrics["decile"]["monotonicity"], 1.0)


def test_daily_ic_filters_nonfinite_and_small_groups() -> None:
    pred, label = _panel(days=2, names=10)
    pred.iloc[0] = np.inf
    panel = build_signal_panel(pred, label)

    daily = daily_ic_frame(panel, min_count=20)

    assert daily.empty


def test_decile_diagnostics_handles_empty_input() -> None:
    empty = pd.DataFrame(columns=["pred", "label", "date"])

    metrics = decile_diagnostics(empty)

    assert metrics == {"deciles": [], "top_bottom_spread": None, "monotonicity": None, "days": 0}
