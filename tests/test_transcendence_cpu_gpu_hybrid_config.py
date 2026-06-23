from __future__ import annotations

from pathlib import Path

import yaml

from quant_master.contrib.model.pytorch_nn import DNNModelPytorch
from quant_master.contrib.model.regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel
from quant_master.contrib.model.transcendence_signal_ensemble import TranscendenceSignalEnsembleModel
from quant_master.utils import init_instance_by_config

from examples.benchmarks.Transcendence.model.alpha158alpha360_regime_horizon_run import (
    _apply_common_overrides,
    _load_config as _load_runner_config,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "transcendence"
    / "workflow_config_transcendence_cpu_gpu_hybrid_smallfund_Alpha158_2026_csi300.yaml"
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _iter_mapping_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from _iter_mapping_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_mapping_keys(value)


def _model_config(candidate: dict) -> dict:
    return candidate["task"]["model"]


def _base_specs(candidate: dict) -> list[dict]:
    return _model_config(candidate)["kwargs"]["base_learner_specs"]


def test_cpu_gpu_hybrid_keeps_smallfund_execution_shape():
    candidate = _load_config(CANDIDATE_CONFIG)
    strategy_kwargs = candidate["port_analysis_config"]["strategy"]["kwargs"]
    backtest_config = candidate["port_analysis_config"]["backtest"]
    exchange_kwargs = backtest_config["exchange_kwargs"]
    model_kwargs = _model_config(candidate)["kwargs"]

    assert backtest_config["account"] == 13000
    assert exchange_kwargs["open_cost"] == 0.0001
    assert exchange_kwargs["close_cost"] == 0.0006
    assert exchange_kwargs["min_cost"] == 0
    assert backtest_config["benchmark"] == "SH000300"
    assert strategy_kwargs["topk"] == 3
    assert strategy_kwargs["n_drop"] == 1
    assert model_kwargs["topk_grid"] == [3]
    assert model_kwargs["n_drop_grid"] == [1]


def test_cpu_gpu_hybrid_has_cpu_anchor_and_deep_auxiliary():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_config = _model_config(candidate)
    model_kwargs = model_config["kwargs"]
    specs = _base_specs(candidate)
    cpu_anchor = specs[0]
    deep_aux = specs[1]
    cpu_kwargs = cpu_anchor["model_kwargs"]

    assert model_config["class"] == "TranscendenceSignalEnsembleModel"
    assert model_config["module_path"] == "quant_master.contrib.model.transcendence_signal_ensemble"
    assert [spec["name"] for spec in specs] == ["regime_horizon_cpu_anchor", "deep_dnn_aux_cpu_safe"]
    assert model_kwargs["max_random_weight_candidates"] == 2
    assert model_kwargs["volatility_penalty_grid"] == [0.02]
    assert model_kwargs["max_drawdown_penalty"] == 0.35
    assert model_kwargs["manual_weight_candidates"] == [
        {"regime_horizon_cpu_anchor": 1.0, "deep_dnn_aux_cpu_safe": 0.0},
        {"regime_horizon_cpu_anchor": 0.97, "deep_dnn_aux_cpu_safe": 0.03},
        {"regime_horizon_cpu_anchor": 0.94, "deep_dnn_aux_cpu_safe": 0.06},
    ]
    assert cpu_anchor["model_type"] == "regime_horizon"
    assert cpu_kwargs["horizon_model_specs"][0]["model_type"] == "double_ensemble"
    assert cpu_kwargs["horizon_model_specs"][1]["model_type"] == "lightgbm"
    assert cpu_kwargs["weight_constraints"] == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.8,
        "max_aux_weight": 0.2,
        "model_max_weights": {"lgb_h5": 0.2},
    }
    # These constraints are scoped to the inner CPU anchor only; the outer ensemble
    # does not expose a hard cap for the DNN auxiliary in this config-only candidate.
    assert "weight_constraints" not in model_kwargs
    assert "max_aux_weight" not in model_kwargs
    assert deep_aux["class"] == "DNNModelPytorch"
    assert deep_aux["module_path"] == "quant_master.contrib.model.pytorch_nn"


def test_cpu_gpu_hybrid_deep_branch_is_cpu_safe_and_tiny():
    candidate = _load_config(CANDIDATE_CONFIG)
    deep_kwargs = _base_specs(candidate)[1]["model_kwargs"]

    assert deep_kwargs["pt_model_kwargs"]["input_dim"] == 157
    assert deep_kwargs["pt_model_kwargs"]["layers"] == [8]
    assert deep_kwargs["GPU"] == -1
    assert deep_kwargs["max_steps"] == 2
    assert deep_kwargs["eval_steps"] == 1
    assert deep_kwargs["early_stop_rounds"] == 1
    assert deep_kwargs["weight_decay"] == 0.0003
    assert {"test_scan", "test_scan_grid", "test_only", "test_only_leakage"}.isdisjoint(
        set(_iter_mapping_keys(candidate))
    )


def test_cpu_gpu_hybrid_handler_uses_flat_dnn_processors():
    candidate = _load_config(CANDIDATE_CONFIG)
    handler_kwargs = candidate["data_handler_config"]

    assert handler_kwargs["process_type"] == "independent"
    assert handler_kwargs["infer_processors"] == [
        {"class": "DropCol", "kwargs": {"col_list": ["VWAP0"]}},
        {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},
    ]
    assert handler_kwargs["learn_processors"] == [
        {"class": "DropCol", "kwargs": {"col_list": ["VWAP0"]}},
        {"class": "DropnaProcessor", "kwargs": {"fields_group": "feature"}},
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ]


def test_cpu_gpu_hybrid_outer_model_kwargs_instantiate_without_data():
    candidate = _load_config(CANDIDATE_CONFIG)
    model = init_instance_by_config(_model_config(candidate))

    assert isinstance(model, TranscendenceSignalEnsembleModel)
    assert model.topk_grid == [3]
    assert model.n_drop_grid == [1]
    assert model.memory_boost_grid == [0.0]
    assert model.turnover_penalty_grid == [0.0007]
    assert model.volatility_penalty_grid == [0.02]
    assert [spec.name for spec in model.specs] == ["regime_horizon_cpu_anchor", "deep_dnn_aux_cpu_safe"]


def test_cpu_gpu_hybrid_inner_learners_build_without_data():
    candidate = _load_config(CANDIDATE_CONFIG)
    model = init_instance_by_config(_model_config(candidate))

    cpu_anchor = model._build_model(model.specs[0])
    deep_aux = model._build_model(model.specs[1])

    assert isinstance(cpu_anchor, RegimeHorizonCostEnsembleModel)
    assert cpu_anchor.weight_constraints == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.8,
        "max_aux_weight": 0.2,
        "model_max_weights": {"lgb_h5": 0.2},
    }
    assert isinstance(deep_aux, DNNModelPytorch)
    assert deep_aux.device.type == "cpu"
    assert deep_aux.max_steps == 2
    assert deep_aux.eval_steps == 1
    assert deep_aux.early_stop_rounds == 1


def test_cpu_gpu_hybrid_smoke_overrides_keep_outer_kwargs_valid_and_nested_learners_build():
    candidate = _load_runner_config(CANDIDATE_CONFIG)
    run_config = _apply_common_overrides(candidate, mode="smoke", preserve_config_windows=True)
    model_kwargs = _model_config(run_config)["kwargs"]
    cpu_anchor_kwargs = model_kwargs["base_learner_specs"][0]["model_kwargs"]

    assert {"horizon_model_specs", "robust_rank_blend_grid", "prediction_shrinkage_grid"}.isdisjoint(model_kwargs)
    assert {"regime_consensus_quantiles", "regime_disagreement_quantiles", "min_regime_samples"}.isdisjoint(
        model_kwargs
    )
    assert model_kwargs["search_step"] == 0.5
    assert model_kwargs["memory_boost_grid"] == [0.0]
    assert cpu_anchor_kwargs["min_regime_samples"] == 120
    assert cpu_anchor_kwargs["robust_rank_blend_grid"] == [0.0]
    assert cpu_anchor_kwargs["prediction_shrinkage_grid"] == [1.0]
    assert cpu_anchor_kwargs["horizon_model_specs"][0]["model_kwargs"]["num_models"] == 1
    assert cpu_anchor_kwargs["horizon_model_specs"][0]["model_kwargs"]["epochs"] == 4
    assert cpu_anchor_kwargs["horizon_model_specs"][1]["model_kwargs"]["num_boost_round"] == 80
    assert run_config["port_analysis_config"]["backtest"]["account"] == 13000
    assert run_config["port_analysis_config"]["backtest"]["exchange_kwargs"]["open_cost"] == 0.0001
    assert run_config["port_analysis_config"]["backtest"]["exchange_kwargs"]["close_cost"] == 0.0006
    assert run_config["port_analysis_config"]["backtest"]["exchange_kwargs"]["min_cost"] == 0

    model = init_instance_by_config(_model_config(run_config))
    assert isinstance(model._build_model(model.specs[0]), RegimeHorizonCostEnsembleModel)
    deep_aux = model._build_model(model.specs[1])
    assert isinstance(deep_aux, DNNModelPytorch)
    assert deep_aux.device.type == "cpu"
    assert deep_aux.max_steps == 2
