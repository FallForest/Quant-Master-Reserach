import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

from examples.benchmarks.Transcendence.model.run_purged_oof_stacking import (
    _assert_held_out_gate,
    _format_diagnostics,
    _get_recorder,
    _held_out_test_diagnostics,
    _load_config,
    _run_preflight,
)


def test_format_diagnostics_serializes_dates_as_iso_strings():
    formatted = _format_diagnostics({"fold_end": date(2023, 12, 29), "rank_ic": 0.05})

    assert json.loads(formatted) == {"fold_end": "2023-12-29", "rank_ic": 0.05}


def test_config_loader_merges_relative_base_config(tmp_path):
    (tmp_path / "base.yaml").write_text("task:\n  model:\n    kwargs:\n      alpha: 1\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "BASE_CONFIG_PATH: base.yaml\ntask:\n  model:\n    kwargs:\n      beta: 2\n",
        encoding="utf-8",
    )

    config = _load_config(tmp_path / "child.yaml")

    assert config["task"]["model"]["kwargs"] == {"alpha": 1, "beta": 2}


def test_personal_under40_config_applies_same_causal_pool_to_all_datasets():
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "benchmarks"
        / "Transcendence"
        / "configs"
        / "workflows"
        / "ensemble"
        / "workflow_config_purged_oof_stacking_Alpha168_tdx_personal_under40_v1.yaml"
    )

    config = _load_config(config_path)
    preflight = config["data_preflight"]
    final_handler = config["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
    fold_handler = config["task"]["model"]["kwargs"]["fold_dataset_config"]["kwargs"]["handler"]["kwargs"]
    expression = preflight["filter_pipe"][0]["rule_expression"]

    assert preflight["market"] == "personal_a_share_pit_v1"
    assert final_handler["instruments"] == preflight["market"] == fold_handler["instruments"]
    assert final_handler["filter_pipe"] == preflight["filter_pipe"] == fold_handler["filter_pipe"]
    assert "$close/($factor+1e-12)" in expression
    assert "Ge($close/($factor+1e-12),2)" in expression
    assert "Le($close/($factor+1e-12),40)" in expression
    assert "Ref($amount,1)" in expression
    assert "$open" not in expression
    assert "Ref($close,-" not in expression
    assert config["universe_contract"]["signal_time"] == "after_close"
    assert config["universe_contract"]["execution_price_cap_applies_to"] == "buy_only"


def test_get_recorder_recovers_existing_run_without_training(monkeypatch):
    recovered = object()
    calls = {}

    def fake_get_recorder(**kwargs):
        calls.update(kwargs)
        return recovered

    monkeypatch.setattr(
        "examples.benchmarks.Transcendence.model.run_purged_oof_stacking.R",
        SimpleNamespace(get_recorder=fake_get_recorder),
    )
    monkeypatch.setattr(
        "examples.benchmarks.Transcendence.model.run_purged_oof_stacking.task_train",
        lambda *args, **kwargs: pytest.fail("recovery must not retrain"),
    )

    result = _get_recorder({"experiment_name": "experiment"}, recorder_id="run-id")

    assert result is recovered
    assert calls == {"recorder_id": "run-id", "experiment_name": "experiment"}


def test_held_out_gate_rejects_incomplete_daily_coverage():
    diagnostics = {
        "overall": {"rank_ic": 0.06, "rank_icir": 0.5},
        "yearly": {"2024": {"rank_ic": 0.06}},
        "coverage": {"min_daily_rows": 901},
    }
    gate = {
        "min_rank_ic": 0.05,
        "min_rank_icir": 0.4,
        "min_yearly_rank_ic": 0.02,
        "min_daily_rows": 950,
    }

    with pytest.raises(ValueError, match="minimum daily rows 901 < 950"):
        _assert_held_out_gate(diagnostics, gate)


def test_held_out_diagnostics_slice_configured_segment_and_report_coverage():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "A"),
        ],
        names=["datetime", "instrument"],
    )
    objects = {
        "pred.pkl": pd.DataFrame({"score": range(len(index))}, index=index),
        "label.pkl": pd.DataFrame({"label": range(len(index))}, index=index),
    }
    recorder = SimpleNamespace(load_object=lambda name: objects[name])
    model = SimpleNamespace(
        _rank_ic_summary=lambda prediction, label: {
            "days": prediction.index.get_level_values("datetime").nunique()
        }
    )

    diagnostics = _held_out_test_diagnostics(
        model,
        recorder,
        segment=("2024-01-02", "2024-01-03"),
    )

    assert diagnostics["overall"]["days"] == 2
    assert diagnostics["coverage"] == {
        "min_daily_rows": 2,
        "median_daily_rows": 2.0,
        "max_daily_rows": 2,
        "min_daily_rows_date": "2024-01-02",
    }


def test_preflight_reports_structural_and_dynamic_eligible_counts(monkeypatch, tmp_path):
    for symbol in ("a", "b"):
        (tmp_path / "features" / symbol).mkdir(parents=True)

    class FakeData:
        @staticmethod
        def instruments(market, filter_pipe=None):
            return {"market": market, "filtered": bool(filter_pipe)}

        @staticmethod
        def list_instruments(instruments, **kwargs):
            return ["A", "B"] if instruments["filtered"] else ["A", "B", "C"]

        @staticmethod
        def features(*args, **kwargs):
            return pd.DataFrame({"feature": [1.0, 2.0]})

    class FakeHandler:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def get_feature_config():
            return ["$close"], ["feature"]

    monkeypatch.setattr("examples.benchmarks.Transcendence.model.run_purged_oof_stacking.D", FakeData)
    monkeypatch.setattr(
        "examples.benchmarks.Transcendence.model.run_purged_oof_stacking.Alpha158AmountFlow", FakeHandler
    )
    config = {
        "quant_master_init": {"provider_uri": str(tmp_path)},
        "data_preflight": {
            "market": "personal_a_share_pit_v1",
            "filter_pipe": [{"filter_type": "ExpressionDFilter", "rule_expression": "rule"}],
            "min_universe_size": 3,
            "max_universe_size": 3,
            "min_eligible_size": 2,
            "max_eligible_size": 2,
            "min_eligible_ratio": 0.6,
            "max_eligible_ratio": 0.7,
            "min_feature_dir_coverage": 1.0,
            "anchor_dates": ["2024-01-02"],
            "feature_set": "alpha158_amount_flow",
            "feature_sample_start": "2023-01-01",
            "feature_sample_end": "2024-01-02",
            "feature_sample_size": 2,
            "expected_feature_columns": 1,
        },
    }

    diagnostics = _run_preflight(config)

    assert diagnostics["anchor_counts"] == {"2024-01-02": 3}
    assert diagnostics["eligible_anchor_counts"] == {"2024-01-02": 2}
    assert diagnostics["eligible_anchor_ratios"]["2024-01-02"] == pytest.approx(2 / 3)
    assert diagnostics["feature_dir_coverage"] == {"2024-01-02": 1.0}
