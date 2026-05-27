from __future__ import annotations

import math
import pickle

import pandas as pd

from examples.benchmarks.Transcendence import regime_switch_stability_eval as regime_eval


def _dump_pickle(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _panel_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2024-01-02", "2024-01-03"]),
            [f"S{i:03d}" for i in range(4)],
        ],
        names=["datetime", "instrument"],
    )
    values = [1.0, 2.0, 3.0, 4.0] * 2
    return (
        pd.DataFrame({"score": values}, index=index),
        pd.DataFrame({"label": values}, index=index),
    )


def test_baseline_signal_metrics_prefers_existing_sig_analysis_artifacts(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    _dump_pickle(artifacts / "sig_analysis" / "ic.pkl", pd.Series([0.03, 0.05, float("nan")]))
    _dump_pickle(artifacts / "sig_analysis" / "ric.pkl", pd.Series([0.04, 0.06]))
    pred, label = _panel_frames()
    _dump_pickle(artifacts / "pred.pkl", pred)
    _dump_pickle(artifacts / "label.pkl", label)

    metrics = regime_eval._baseline_signal_metrics_from_artifacts(artifacts)

    assert math.isclose(metrics["ic"], 0.04)
    assert math.isclose(metrics["rank_ic"], 0.05)
    assert metrics["source"] == "artifacts.sig_analysis.ic.pkl+ric.pkl"
    assert metrics["ic_days"] == 2
    assert metrics["rank_ic_days"] == 2
    assert "reason" not in metrics


def test_baseline_signal_metrics_falls_back_to_daily_pred_label(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    pred, label = _panel_frames()
    _dump_pickle(artifacts / "pred.pkl", pred)
    _dump_pickle(artifacts / "label.pkl", label)

    metrics = regime_eval._baseline_signal_metrics_from_artifacts(artifacts)

    assert math.isclose(metrics["ic"], 1.0)
    assert math.isclose(metrics["rank_ic"], 1.0)
    assert metrics["source"] == "artifacts.pred.pkl+label.pkl"
    assert metrics["fallback_reason"] == "missing sig_analysis/ic.pkl or sig_analysis/ric.pkl"
    assert metrics["ic_days"] == 2
    assert metrics["rank_ic_days"] == 2


def test_baseline_signal_metrics_fails_closed_when_artifacts_missing(tmp_path) -> None:
    metrics = regime_eval._baseline_signal_metrics_from_artifacts(tmp_path / "artifacts")

    assert metrics["ic"] is None
    assert metrics["rank_ic"] is None
    assert metrics["source"] == "unavailable"
    assert "missing" in metrics["reason"]
