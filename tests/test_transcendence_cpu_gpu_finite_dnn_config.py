from __future__ import annotations

from pathlib import Path

import yaml

from examples.benchmarks.Transcendence.model.alpha158alpha360_regime_horizon_run import (
    _apply_common_overrides,
    _load_config as _load_runner_config,
)
from quant_master.contrib.model.finite_dnn import FiniteDNNModelPytorch
from quant_master.contrib.model.regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel
from quant_master.contrib.model.transcendence_signal_ensemble import TranscendenceSignalEnsembleModel
from quant_master.utils import init_instance_by_config


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "transcendence"
    / "workflow_config_transcendence_cpu_gpu_finite_dnn_smallfund_Alpha158_2026_csi300.yaml"
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _model_config(candidate: dict) -> dict:
    return candidate["task"]["model"]


def _base_specs(candidate: dict) -> list[dict]:
    return _model_config(candidate)["kwargs"]["base_learner_specs"]


def _iter_mapping_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from _iter_mapping_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_mapping_keys(value)


def test_finite_dnn_hybrid_keeps_smallfund_execution_shape():
    candidate = _load_config(CANDIDATE_CONFIG)
    strategy_kwargs = candidate["port_analysis_config"]["strategy"]["kwargs"]
    backtest_config = candidate["port_analysis_config"]["backtest"]
    exchange_kwargs = backtest_config["exchange_kwargs"]

    assert backtest_config["account"] == 13000
    assert backtest_config["benchmark"] == "SH000300"
    assert exchange_kwargs["open_cost"] == 0.0001
    assert exchange_kwargs["close_cost"] == 0.0006
    assert exchange_kwargs["min_cost"] == 0
    assert strategy_kwargs["topk"] == 3
    assert strategy_kwargs["n_drop"] == 1


def test_finite_dnn_hybrid_has_cpu_anchor_and_local_clean_deep_aux():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_config = _model_config(candidate)
    model_kwargs = model_config["kwargs"]
    specs = _base_specs(candidate)
    cpu_anchor = specs[0]
    deep_aux = specs[1]
    cpu_kwargs = cpu_anchor["model_kwargs"]
    deep_kwargs = deep_aux["model_kwargs"]

    assert model_config["class"] == "TranscendenceSignalEnsembleModel"
    assert [spec["name"] for spec in specs] == ["regime_horizon_cpu_anchor", "finite_dnn_aux_cpu_safe"]
    assert cpu_anchor["model_type"] == "regime_horizon"
    assert model_kwargs["search_step"] == 0.5
    assert model_kwargs["max_random_weight_candidates"] == 8
    assert model_kwargs["refine_top_weight_candidates"] == 4
    assert model_kwargs["volatility_penalty_grid"] == [0.02]
    assert model_kwargs["max_drawdown_penalty"] == 0.4
    assert deep_aux["class"] == "FiniteDNNModelPytorch"
    assert deep_aux["module_path"] == "quant_master.contrib.model.finite_dnn"
    assert deep_kwargs["input_dim"] == 158
    assert deep_kwargs["layers"] == [32, 8]
    assert deep_kwargs["GPU"] == -1
    assert deep_kwargs["max_steps"] == 8
    assert deep_kwargs["early_stop_rounds"] == 3
    assert deep_kwargs["weight_decay"] == 0.0003
    assert deep_kwargs["feature_daily_zscore"] is True
    assert deep_kwargs["label_rank"] is True
    assert cpu_kwargs["weight_constraints"] == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.8,
        "max_aux_weight": 0.2,
        "model_max_weights": {"lgb_h5": 0.2},
    }
    assert model_kwargs["manual_weight_candidates"] == [
        {"regime_horizon_cpu_anchor": 1.0, "finite_dnn_aux_cpu_safe": 0.0},
        {"regime_horizon_cpu_anchor": 0.97, "finite_dnn_aux_cpu_safe": 0.03},
        {"regime_horizon_cpu_anchor": 0.94, "finite_dnn_aux_cpu_safe": 0.06},
        {"regime_horizon_cpu_anchor": 0.9, "finite_dnn_aux_cpu_safe": 0.1},
    ]
    assert {"test_scan", "test_scan_grid", "test_only", "test_only_leakage"}.isdisjoint(
        set(_iter_mapping_keys(candidate))
    )


def test_finite_dnn_hybrid_learners_build_without_data():
    candidate = _load_config(CANDIDATE_CONFIG)
    model = init_instance_by_config(_model_config(candidate))

    assert isinstance(model, TranscendenceSignalEnsembleModel)
    cpu_anchor = model._build_model(model.specs[0])
    deep_aux = model._build_model(model.specs[1])

    assert isinstance(cpu_anchor, RegimeHorizonCostEnsembleModel)
    assert isinstance(deep_aux, FiniteDNNModelPytorch)
    assert deep_aux.device.type == "cpu"
    assert deep_aux.max_steps == 8
    assert deep_aux.early_stop_rounds == 3
    assert cpu_anchor.weight_constraints == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.8,
        "max_aux_weight": 0.2,
        "model_max_weights": {"lgb_h5": 0.2},
    }


def test_signal_ensemble_manual_weight_candidates_are_validated_and_normalized():
    model = TranscendenceSignalEnsembleModel(
        base_learner_specs=[
            {"name": "cpu", "model_type": "linear"},
            {"name": "dnn", "model_type": "linear"},
        ],
        manual_weight_candidates=[{"cpu": 9.0, "dnn": 1.0}, [0.8, 0.2]],
    )
    model.model_order = ["cpu", "dnn"]

    candidates = list(model._manual_weight_candidates(2))

    assert candidates[0].tolist() == [0.9, 0.1]
    assert candidates[1].tolist() == [0.8, 0.2]


def test_signal_ensemble_manual_weight_candidates_reject_invalid_shape():
    model = TranscendenceSignalEnsembleModel(
        base_learner_specs=[
            {"name": "cpu", "model_type": "linear"},
            {"name": "dnn", "model_type": "linear"},
        ],
        manual_weight_candidates=[[1.0, 0.0, 0.0]],
    )
    model.model_order = ["cpu", "dnn"]

    try:
        list(model._manual_weight_candidates(2))
    except ValueError as exc:
        assert "2 weights" in str(exc)
    else:
        raise AssertionError("invalid manual weight candidate shape was accepted")


def test_finite_dnn_smoke_overrides_keep_nested_budgeting_valid():
    candidate = _load_runner_config(CANDIDATE_CONFIG)
    run_config = _apply_common_overrides(candidate, mode="smoke", preserve_config_windows=True)
    model_kwargs = _model_config(run_config)["kwargs"]
    cpu_anchor_kwargs = model_kwargs["base_learner_specs"][0]["model_kwargs"]
    deep_kwargs = model_kwargs["base_learner_specs"][1]["model_kwargs"]

    assert {"horizon_model_specs", "robust_rank_blend_grid", "prediction_shrinkage_grid"}.isdisjoint(model_kwargs)
    assert model_kwargs["search_step"] == 0.5
    assert cpu_anchor_kwargs["min_regime_samples"] == 120
    assert cpu_anchor_kwargs["horizon_model_specs"][0]["model_kwargs"]["num_models"] == 1
    assert cpu_anchor_kwargs["horizon_model_specs"][1]["model_kwargs"]["num_boost_round"] == 80
    assert deep_kwargs["max_steps"] == 8
    assert run_config["port_analysis_config"]["backtest"]["account"] == 13000
    assert run_config["port_analysis_config"]["backtest"]["exchange_kwargs"]["open_cost"] == 0.0001
    assert run_config["port_analysis_config"]["backtest"]["exchange_kwargs"]["close_cost"] == 0.0006
    assert run_config["port_analysis_config"]["backtest"]["exchange_kwargs"]["min_cost"] == 0

    model = init_instance_by_config(_model_config(run_config))
    assert isinstance(model._build_model(model.specs[0]), RegimeHorizonCostEnsembleModel)
    assert isinstance(model._build_model(model.specs[1]), FiniteDNNModelPytorch)
