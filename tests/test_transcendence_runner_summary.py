from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
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


def _base_config() -> dict[str, Any]:
    return {
        "quant_master_init": {"provider_uri": "mock_data"},
        "data_handler_config": {
            "start_time": "2018-01-01",
            "end_time": "2024-06-30",
            "fit_start_time": "2019-01-01",
            "fit_end_time": "2021-12-31",
        },
        "port_analysis_config": {
            "strategy": {"kwargs": {"topk": 45, "n_drop": 4}},
            "backtest": {
                "start_time": "2024-02-01",
                "end_time": "2024-06-30",
                "exchange_kwargs": {},
            },
        },
        "task": {
            "model": {
                "kwargs": {
                    "search_step": 0.1,
                    "memory_boost_grid": [0.0, 0.005],
                    "turnover_penalty_grid": [0.0, 0.0005, 0.001],
                    "risk_penalty_grid": [0.0, 0.05],
                    "robust_rank_blend_grid": [0.0, 0.15, 0.3],
                    "prediction_shrinkage_grid": [1.0, 0.9, 0.8],
                    "regime_consensus_quantiles": [0.4, 0.5, 0.6],
                    "regime_disagreement_quantiles": [0.4, 0.5, 0.6],
                    "min_regime_samples": 240,
                    "horizon_model_specs": [
                        {
                            "name": "de_h1",
                            "model_type": "double_ensemble",
                            "model_kwargs": {
                                "num_models": 3,
                                "epochs": 28,
                                "enable_sr": True,
                                "enable_fs": True,
                                "sub_weights": [1, 1, 1],
                                "num_threads": 8,
                            },
                        },
                        {
                            "name": "de_h5",
                            "model_type": "double_ensemble",
                            "model_kwargs": {
                                "num_models": 2,
                                "epochs": 12,
                                "enable_sr": True,
                                "enable_fs": True,
                                "sub_weights": [1, 1],
                                "num_threads": 6,
                            },
                        },
                    ],
                }
            },
            "dataset": {
                "kwargs": {
                    "handler": {
                        "kwargs": {
                            "start_time": "2018-01-01",
                            "end_time": "2024-06-30",
                            "fit_start_time": "2019-01-01",
                            "fit_end_time": "2021-12-31",
                        }
                    },
                    "segments": {
                        "train": ["2019-01-01", "2021-12-31"],
                        "valid": ["2022-01-01", "2022-12-31"],
                        "test": ["2024-02-01", "2024-06-30"],
                    },
                }
            },
        },
    }


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


def test_metrics_from_report_uses_index_aligned_finite_mask() -> None:
    report = pd.DataFrame(
        {
            "return": [0.0, 0.01, np.nan, 0.02],
            "bench": [0.0, 0.001, 0.002, 0.003],
            "cost": [0.0, 0.0001, 0.0002, 0.0003],
            "turnover": [0.0, 0.5, 0.6, 0.7],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )

    metrics = runner._metrics_from_report(report)

    assert metrics["rows"] == 4
    assert metrics["finite_rows"] == 3
    assert metrics["nonfinite_rows"] == 1
    assert math.isclose(metrics["turnover"], (0.0 + 0.5 + 0.7) / 3.0)
    assert math.isfinite(metrics["costed_annret"])


def test_summary_metadata_jsonable_handles_dates_timestamps_and_numpy_scalars() -> None:
    summary = {
        "status": "ok",
        "runner_metadata": {
            "as_of_date": date(2026, 5, 27),
            "created_at": datetime(2026, 5, 27, 12, 34, 56, tzinfo=timezone.utc),
            "rebalance_time": pd.Timestamp("2026-05-27T15:00:00+08:00"),
            "rows": np.int64(562),
            "costed_ir": np.float64(3.14),
            "hard_gate_pass": np.bool_(True),
        },
    }

    payload = json.loads(json.dumps(runner._jsonable(summary), ensure_ascii=False))

    assert payload["runner_metadata"] == {
        "as_of_date": "2026-05-27",
        "created_at": "2026-05-27T12:34:56+00:00",
        "rebalance_time": "2026-05-27T15:00:00+08:00",
        "rows": 562,
        "costed_ir": 3.14,
        "hard_gate_pass": True,
    }


def test_preserve_config_windows_keeps_yaml_segments_and_records_metadata() -> None:
    cfg = runner._apply_common_overrides(_base_config(), mode="medium", preserve_config_windows=True)

    expected_segments = {
        "train": ["2019-01-01", "2021-12-31"],
        "valid": ["2022-01-01", "2022-12-31"],
        "test": ["2024-02-01", "2024-06-30"],
    }
    assert cfg["task"]["dataset"]["kwargs"]["segments"] == expected_segments
    assert cfg["port_analysis_config"]["backtest"]["start_time"] == "2024-02-01"
    assert cfg["port_analysis_config"]["backtest"]["end_time"] == "2024-06-30"
    assert cfg["runner_metadata"]["preserve_config_windows"] is True
    assert cfg["runner_metadata"]["actual_segments"] == expected_segments


def test_common_overrides_falls_back_to_default_costs_when_yaml_costs_missing() -> None:
    cfg = runner._apply_common_overrides(_base_config(), mode="medium", preserve_config_windows=True)
    exchange_kwargs = cfg["port_analysis_config"]["backtest"]["exchange_kwargs"]

    assert exchange_kwargs["open_cost"] == runner.OPEN_COST
    assert exchange_kwargs["close_cost"] == runner.CLOSE_COST
    assert cfg["runner_metadata"]["protocol"]["open_cost"] == runner.OPEN_COST
    assert cfg["runner_metadata"]["protocol"]["close_cost"] == runner.CLOSE_COST


def test_common_overrides_preserves_yaml_costs_in_run_config_and_metadata() -> None:
    base_config = _base_config()
    base_config["port_analysis_config"]["backtest"]["exchange_kwargs"] = {
        "open_cost": 0.00012,
        "close_cost": 0.00045,
    }

    cfg = runner._apply_common_overrides(base_config, mode="medium", preserve_config_windows=True)
    exchange_kwargs = cfg["port_analysis_config"]["backtest"]["exchange_kwargs"]

    assert exchange_kwargs["open_cost"] == 0.00012
    assert exchange_kwargs["close_cost"] == 0.00045
    assert cfg["runner_metadata"]["protocol"]["open_cost"] == 0.00012
    assert cfg["runner_metadata"]["protocol"]["close_cost"] == 0.00045


def test_medium_preserve_config_windows_keeps_candidate_selection_grids() -> None:
    cfg = runner._apply_common_overrides(_base_config(), mode="medium", preserve_config_windows=True)
    model_kwargs = cfg["task"]["model"]["kwargs"]
    decisions = {
        decision["key"]: decision
        for decision in cfg["runner_metadata"]["selection_grid_decisions"]
    }

    assert model_kwargs["search_step"] == 0.1
    assert model_kwargs["memory_boost_grid"] == [0.0, 0.005]
    assert model_kwargs["turnover_penalty_grid"] == [0.0, 0.0005, 0.001]
    assert model_kwargs["risk_penalty_grid"] == [0.0, 0.05]
    assert model_kwargs["robust_rank_blend_grid"] == [0.0, 0.15, 0.3]
    assert model_kwargs["prediction_shrinkage_grid"] == [1.0, 0.9, 0.8]
    assert set(decisions) == {
        "search_step",
        "memory_boost_grid",
        "turnover_penalty_grid",
        "risk_penalty_grid",
        "robust_rank_blend_grid",
        "prediction_shrinkage_grid",
    }
    assert all(decision["action"] == "preserved" for decision in decisions.values())
    assert decisions["memory_boost_grid"]["value"] == [0.0, 0.005]
    assert not any(
        change["path"] in {
            "task.model.kwargs.search_step",
            "task.model.kwargs.memory_boost_grid",
            "task.model.kwargs.turnover_penalty_grid",
            "task.model.kwargs.risk_penalty_grid",
            "task.model.kwargs.robust_rank_blend_grid",
            "task.model.kwargs.prediction_shrinkage_grid",
        }
        for change in cfg["runner_metadata"]["budget_overrides"]
    )


def test_default_windows_still_override_yaml_segments_for_compatibility() -> None:
    cfg = runner._apply_common_overrides(_base_config(), mode="full")

    assert cfg["task"]["dataset"]["kwargs"]["segments"] == {
        "train": runner.TRAIN_RANGE,
        "valid": runner.VALID_RANGE,
        "test": runner.TEST_RANGE,
    }
    assert cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["fit_start_time"] == runner.TRAIN_RANGE[0]
    assert cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["fit_end_time"] == runner.TRAIN_RANGE[1]
    assert cfg["runner_metadata"]["preserve_config_windows"] is False


def test_medium_uses_full_test_window_with_low_budget_overrides() -> None:
    cfg = runner._apply_common_overrides(_base_config(), mode="medium")
    model_kwargs = cfg["task"]["model"]["kwargs"]
    specs = model_kwargs["horizon_model_specs"]

    assert cfg["task"]["dataset"]["kwargs"]["segments"]["test"] == runner.TEST_RANGE
    assert cfg["port_analysis_config"]["backtest"]["start_time"] == runner.TEST_RANGE[0]
    assert cfg["port_analysis_config"]["backtest"]["end_time"] == runner.TEST_RANGE[1]
    assert model_kwargs["search_step"] == 0.5
    assert model_kwargs["memory_boost_grid"] == [0.0]
    assert model_kwargs["turnover_penalty_grid"] == [0.0, 0.0005, 0.001]
    assert model_kwargs["risk_penalty_grid"] == [0.0, 0.05]
    assert model_kwargs["robust_rank_blend_grid"] == [0.0, 0.15, 0.3]
    assert model_kwargs["prediction_shrinkage_grid"] == [1.0, 0.9, 0.8]
    assert model_kwargs["regime_consensus_quantiles"] == [0.5]
    assert model_kwargs["regime_disagreement_quantiles"] == [0.5]
    assert len(specs) == 2
    for spec in specs:
        mk = spec["model_kwargs"]
        assert mk["num_models"] == 1
        assert mk["epochs"] <= 8
        assert mk["enable_sr"] is False
        assert mk["enable_fs"] is False
    assert cfg["runner_metadata"]["mode"] == "medium"
    assert cfg["runner_metadata"]["actual_test"] == runner.TEST_RANGE
    assert cfg["runner_metadata"]["budget_overrides"]
    decisions = {
        decision["key"]: decision
        for decision in cfg["runner_metadata"]["selection_grid_decisions"]
    }
    assert set(decisions) == {
        "search_step",
        "memory_boost_grid",
        "turnover_penalty_grid",
        "risk_penalty_grid",
        "robust_rank_blend_grid",
        "prediction_shrinkage_grid",
    }
    assert decisions["search_step"]["action"] == "overridden"
    assert decisions["search_step"]["old"] == 0.1
    assert decisions["search_step"]["new"] == 0.5
    assert decisions["memory_boost_grid"]["action"] == "overridden"
    assert decisions["memory_boost_grid"]["old"] == [0.0, 0.005]
    assert decisions["memory_boost_grid"]["new"] == [0.0]
    assert decisions["turnover_penalty_grid"]["action"] == "preserved"
    assert decisions["risk_penalty_grid"]["action"] == "preserved"
    assert decisions["robust_rank_blend_grid"]["action"] == "preserved"
    assert decisions["prediction_shrinkage_grid"]["action"] == "preserved"


def test_smoke_uses_single_point_final_control_grids() -> None:
    cfg = runner._apply_common_overrides(_base_config(), mode="smoke", preserve_config_windows=True)
    model_kwargs = cfg["task"]["model"]["kwargs"]
    decisions = {
        decision["key"]: decision
        for decision in cfg["runner_metadata"]["selection_grid_decisions"]
    }

    assert model_kwargs["search_step"] == 0.5
    assert model_kwargs["memory_boost_grid"] == [0.0]
    assert model_kwargs["robust_rank_blend_grid"] == [0.0]
    assert model_kwargs["prediction_shrinkage_grid"] == [1.0]
    assert decisions["robust_rank_blend_grid"]["action"] == "overridden"
    assert decisions["robust_rank_blend_grid"]["old"] == [0.0, 0.15, 0.3]
    assert decisions["robust_rank_blend_grid"]["new"] == [0.0]
    assert decisions["prediction_shrinkage_grid"]["action"] == "overridden"
    assert decisions["prediction_shrinkage_grid"]["old"] == [1.0, 0.9, 0.8]
    assert decisions["prediction_shrinkage_grid"]["new"] == [1.0]


def test_smoke_does_not_inject_n_drop_into_weight_based_strategies() -> None:
    base_config = _base_config()
    base_config["port_analysis_config"]["strategy"] = {
        "class": "SoftTopkStrategy",
        "module_path": "quant_master.contrib.strategy",
        "kwargs": {
            "topk": 3,
            "risk_degree": 1.0,
            "trade_impact_limit": 0.3333333333,
        },
    }

    cfg = runner._apply_common_overrides(base_config, mode="smoke", preserve_config_windows=True)
    strategy_kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]

    assert strategy_kwargs["topk"] == 3
    assert "n_drop" not in strategy_kwargs
    assert not any(
        change["path"] == "port_analysis_config.strategy.kwargs.n_drop"
        for change in cfg["runner_metadata"]["budget_overrides"]
    )
