from __future__ import annotations

import math
from typing import Any

import pandas as pd

from examples.benchmarks.Transcendence.model import alpha158alpha360_regime_horizon_run as runner


class _Recorder:
    def __init__(self, objects: dict[str, Any]) -> None:
        self.objects = objects

    def load_object(self, name: str) -> Any:
        if name not in self.objects:
            raise FileNotFoundError(name)
        return self.objects[name]


def _perfect_panel_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2024-01-02", "2024-01-03"]),
            [f"S{i:03d}" for i in range(20)],
        ],
        names=["datetime", "instrument"],
    )
    values = list(range(20)) * 2
    return (
        pd.DataFrame({"score": values}, index=index),
        pd.DataFrame({"label": values}, index=index),
    )


def test_signal_metrics_computes_ic_and_rank_ic_from_saved_signal_artifacts() -> None:
    pred, label = _perfect_panel_frames()

    metrics = runner._signal_metrics(_Recorder({"pred.pkl": pred, "label.pkl": label}))

    assert math.isclose(metrics["ic"], 1.0)
    assert math.isclose(metrics["rank_ic"], 1.0)
    assert metrics["ic_days"] == 2
    assert metrics["rank_ic_days"] == 2
    assert "error" not in metrics


def test_apply_validation_metrics_promotes_standard_top_level_fields() -> None:
    pred, label = _perfect_panel_frames()
    metrics = runner._signal_metrics(_Recorder({"pred.pkl": pred, "label.pkl": label}))
    summary: dict[str, Any] = {}

    runner._apply_validation_metrics(summary, metrics)

    assert summary["validation_metrics"] == metrics
    assert math.isclose(summary["ic"], 1.0)
    assert math.isclose(summary["rank_ic"], 1.0)
    assert "ic_missing_reason" not in summary
    assert "rank_ic_missing_reason" not in summary


def test_apply_validation_metrics_records_missing_reason_without_fabricating_values() -> None:
    summary: dict[str, Any] = {}
    metrics = {"error": "missing pred or label"}

    runner._apply_validation_metrics(summary, metrics)

    assert summary["validation_metrics"] == metrics
    assert summary["ic"] is None
    assert summary["rank_ic"] is None
    assert summary["ic_missing_reason"] == "missing pred or label"
    assert summary["rank_ic_missing_reason"] == "missing pred or label"
