from __future__ import annotations

from pathlib import Path
from datetime import date

import yaml

from quant_master.contrib.model.finite_dnn import FiniteDNNModelPytorch
from quant_master.contrib.model.regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel
from quant_master.contrib.model.transcendence_signal_ensemble import TranscendenceSignalEnsembleModel
from quant_master.utils import init_instance_by_config


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "transcendence"
    / "workflow_config_transcendence_cpu_gpu_finite_dnn_smallfund_Alpha158_2026_csi300.yaml"
)
CANDIDATE_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "transcendence"
    / "workflow_config_transcendence_cpu_gpu_lowweight_rerank_smallfund_Alpha158_2026_csi300.yaml"
)
RISKCLAMP_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "transcendence"
    / "workflow_config_transcendence_cpu_gpu_lowweight_rerank_riskclamp_smallfund_Alpha158_2026_csi300.yaml"
)
BALANCED_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "transcendence"
    / "workflow_config_transcendence_cpu_gpu_lowweight_rerank_balanced_smallfund_Alpha158_2026_csi300.yaml"
)
MIDRISK_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "transcendence"
    / "workflow_config_transcendence_cpu_gpu_lowweight_rerank_midrisk_smallfund_Alpha158_2026_csi300.yaml"
)
DYNAMICGATE_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "transcendence"
    / "workflow_config_transcendence_cpu_gpu_lowweight_rerank_dynamicgate_smallfund_Alpha158_2026_csi300.yaml"
)
DYNAMICGATE_TIGHT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "transcendence"
    / "workflow_config_transcendence_cpu_gpu_lowweight_rerank_dynamicgate_tight_smallfund_Alpha158_2026_csi300.yaml"
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _model_config(candidate: dict) -> dict:
    return candidate["task"]["model"]


def _model_kwargs(candidate: dict) -> dict:
    return _model_config(candidate)["kwargs"]


def _base_specs(candidate: dict) -> list[dict]:
    return _model_kwargs(candidate)["base_learner_specs"]


def test_lowweight_rerank_yaml_keeps_smallfund_windows_and_cpu_safe_dnn():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = _model_kwargs(candidate)
    specs = _base_specs(candidate)
    dnn_kwargs = specs[1]["model_kwargs"]
    segments = candidate["task"]["dataset"]["kwargs"]["segments"]

    assert candidate["data_handler_config"]["start_time"] == date(2008, 1, 1)
    assert candidate["data_handler_config"]["fit_end_time"] == date(2018, 12, 31)
    assert segments["train"] == [date(2008, 1, 1), date(2018, 12, 31)]
    assert segments["valid"] == [date(2019, 1, 1), date(2023, 12, 31)]
    assert segments["test"] == [date(2024, 1, 1), date(2026, 4, 30)]
    assert _model_config(candidate)["class"] == "TranscendenceSignalEnsembleModel"
    assert _model_config(candidate)["module_path"] == "quant_master.contrib.model.transcendence_signal_ensemble"
    assert [spec["name"] for spec in specs] == ["regime_horizon_cpu_anchor", "finite_dnn_aux_cpu_safe"]
    assert specs[1]["class"] == "FiniteDNNModelPytorch"
    assert specs[1]["module_path"] == "quant_master.contrib.model.finite_dnn"
    assert dnn_kwargs["GPU"] == -1
    assert dnn_kwargs["input_dim"] == 158
    assert dnn_kwargs["layers"] == [32, 8]
    assert dnn_kwargs["max_steps"] == 8
    assert dnn_kwargs["label_rank"] is True
    assert model_kwargs["max_random_weight_candidates"] == 1
    assert model_kwargs["refine_top_weight_candidates"] == 1
    assert model_kwargs["weight_constraints"] == {"anchor_index": 0, "max_aux_weight": 0.05}


def test_lowweight_rerank_limits_dnn_and_horizon_auxiliary_weights():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = _model_kwargs(candidate)
    anchor_kwargs = _base_specs(candidate)[0]["model_kwargs"]
    weight_constraints = anchor_kwargs["weight_constraints"]

    dnn_weights = [
        weights["finite_dnn_aux_cpu_safe"]
        for weights in model_kwargs["manual_weight_candidates"]
    ]

    assert dnn_weights == [0.0, 0.02, 0.05]
    assert max(dnn_weights) <= 0.05
    assert weight_constraints == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.9,
        "max_aux_weight": 0.1,
        "model_max_weights": {"lgb_h5": 0.1},
    }


def test_lowweight_rerank_tightens_turnover_drawdown_and_preserves_rerank_execution():
    baseline = _load_config(BASELINE_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)
    baseline_kwargs = _model_kwargs(baseline)
    candidate_kwargs = _model_kwargs(candidate)
    baseline_anchor_kwargs = _base_specs(baseline)[0]["model_kwargs"]
    candidate_anchor_kwargs = _base_specs(candidate)[0]["model_kwargs"]
    strategy_config = candidate["port_analysis_config"]["strategy"]

    assert strategy_config["class"] == "SoftTopkStrategy"
    assert strategy_config["module_path"] == "quant_master.contrib.strategy"
    assert strategy_config["kwargs"]["topk"] == 3
    assert strategy_config["kwargs"]["selection_rank_buffer"] == 1
    assert strategy_config["kwargs"]["selection_max_new_names"] == 1
    assert candidate_kwargs["turnover_penalty_grid"] == [0.0012, 0.0018, 0.0025, 0.0035]
    assert min(candidate_kwargs["turnover_penalty_grid"]) > max(baseline_kwargs["turnover_penalty_grid"])
    assert candidate_anchor_kwargs["turnover_penalty_grid"] == [0.0012, 0.0018, 0.0025, 0.0035]
    assert min(candidate_anchor_kwargs["turnover_penalty_grid"]) >= min(baseline_anchor_kwargs["turnover_penalty_grid"])
    assert candidate_kwargs["max_drawdown_penalty"] > baseline_kwargs["max_drawdown_penalty"]
    assert candidate_kwargs["volatility_penalty_grid"] == [0.03]
    assert candidate_anchor_kwargs["memory_boost_grid"] == [0.0]
    assert candidate_anchor_kwargs["risk_penalty_grid"] == [0.08, 0.12, 0.16]


def test_lowweight_rerank_learners_build_without_data():
    candidate = _load_config(CANDIDATE_CONFIG)
    model = init_instance_by_config(_model_config(candidate))

    assert isinstance(model, TranscendenceSignalEnsembleModel)
    assert model.turnover_penalty_grid == [0.0012, 0.0018, 0.0025, 0.0035]
    assert model.max_drawdown_penalty == 0.6
    assert model.max_random_weight_candidates == 1
    assert model.weight_constraints == {"anchor_index": 0, "max_aux_weight": 0.05}

    cpu_anchor = model._build_model(model.specs[0])
    deep_aux = model._build_model(model.specs[1])

    assert isinstance(cpu_anchor, RegimeHorizonCostEnsembleModel)
    assert isinstance(deep_aux, FiniteDNNModelPytorch)
    assert deep_aux.device.type == "cpu"
    assert cpu_anchor.weight_constraints == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.9,
        "max_aux_weight": 0.1,
        "model_max_weights": {"lgb_h5": 0.1},
    }


def test_lowweight_rerank_actual_weight_candidates_keep_dnn_auxiliary():
    candidate = _load_config(CANDIDATE_CONFIG)
    model = init_instance_by_config(_model_config(candidate))
    model.model_order = [spec.name for spec in model.specs]

    generated_weights = [
        dict(zip(model.model_order, weights.tolist()))
        for weights in model._weight_candidates(len(model.model_order))
    ]
    dnn_weights = [weights["finite_dnn_aux_cpu_safe"] for weights in generated_weights]

    assert max(dnn_weights) <= 0.05


def test_signal_ensemble_unconstrained_weight_candidates_keep_auto_candidates():
    model = TranscendenceSignalEnsembleModel(
        base_learner_specs=[
            {"name": "cpu", "model_type": "linear"},
            {"name": "dnn", "model_type": "linear"},
        ],
        search_step=0.5,
        max_random_weight_candidates=1,
        manual_weight_candidates=[{"cpu": 0.95, "dnn": 0.05}],
    )
    model.model_order = ["cpu", "dnn"]

    candidates = [weights.tolist() for weights in model._weight_candidates(2)]

    assert [0.5, 0.5] in candidates
    assert [0.0, 1.0] in candidates
    assert [1.0, 0.0] in candidates


def test_riskclamp_variant_parses_and_keeps_dnn_auxiliary_weight_low():
    candidate = _load_config(RISKCLAMP_CONFIG)
    model_kwargs = _model_kwargs(candidate)
    model = init_instance_by_config(_model_config(candidate))
    model.model_order = [spec.name for spec in model.specs]

    generated_weights = [
        dict(zip(model.model_order, weights.tolist()))
        for weights in model._weight_candidates(len(model.model_order))
    ]
    dnn_weights = [weights["finite_dnn_aux_cpu_safe"] for weights in generated_weights]

    assert _model_config(candidate)["class"] == "TranscendenceSignalEnsembleModel"
    assert model_kwargs["weight_constraints"] == {"anchor_index": 0, "max_aux_weight": 0.05}
    assert max(dnn_weights) <= 0.05
    assert model_kwargs["manual_weight_candidates"] == [
        {"regime_horizon_cpu_anchor": 1.0, "finite_dnn_aux_cpu_safe": 0.0},
        {"regime_horizon_cpu_anchor": 0.98, "finite_dnn_aux_cpu_safe": 0.02},
        {"regime_horizon_cpu_anchor": 0.95, "finite_dnn_aux_cpu_safe": 0.05},
    ]


def test_riskclamp_variant_is_more_conservative_than_lowweight_rerank():
    lowweight = _load_config(CANDIDATE_CONFIG)
    riskclamp = _load_config(RISKCLAMP_CONFIG)
    lowweight_kwargs = _model_kwargs(lowweight)
    riskclamp_kwargs = _model_kwargs(riskclamp)
    lowweight_anchor = _base_specs(lowweight)[0]["model_kwargs"]
    riskclamp_anchor = _base_specs(riskclamp)[0]["model_kwargs"]
    lowweight_strategy = lowweight["port_analysis_config"]["strategy"]["kwargs"]
    riskclamp_strategy = riskclamp["port_analysis_config"]["strategy"]["kwargs"]

    assert riskclamp_strategy["risk_degree"] < lowweight_strategy["risk_degree"]
    assert riskclamp_strategy["trade_impact_limit"] < lowweight_strategy["trade_impact_limit"]
    assert riskclamp_strategy["selection_rank_buffer"] > lowweight_strategy["selection_rank_buffer"]
    assert riskclamp_strategy["selection_max_new_names"] == lowweight_strategy["selection_max_new_names"] == 1
    assert min(riskclamp_kwargs["turnover_penalty_grid"]) > min(lowweight_kwargs["turnover_penalty_grid"])
    assert min(riskclamp_kwargs["volatility_penalty_grid"]) > min(lowweight_kwargs["volatility_penalty_grid"])
    assert riskclamp_kwargs["max_drawdown_penalty"] > lowweight_kwargs["max_drawdown_penalty"]
    assert riskclamp_kwargs["annret_weight"] < lowweight_kwargs["annret_weight"]
    assert min(riskclamp_anchor["risk_penalty_grid"]) > min(lowweight_anchor["risk_penalty_grid"])
    assert riskclamp_anchor["weight_constraints"] == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.95,
        "max_aux_weight": 0.05,
        "model_max_weights": {"lgb_h5": 0.05},
    }


def test_balanced_variant_parses_and_keeps_dnn_auxiliary_weight_low():
    candidate = _load_config(BALANCED_CONFIG)
    model_kwargs = _model_kwargs(candidate)
    model = init_instance_by_config(_model_config(candidate))
    model.model_order = [spec.name for spec in model.specs]

    generated_weights = [
        dict(zip(model.model_order, weights.tolist()))
        for weights in model._weight_candidates(len(model.model_order))
    ]
    dnn_weights = [weights["finite_dnn_aux_cpu_safe"] for weights in generated_weights]

    assert _model_config(candidate)["class"] == "TranscendenceSignalEnsembleModel"
    assert model_kwargs["weight_constraints"] == {"anchor_index": 0, "max_aux_weight": 0.05}
    assert max(dnn_weights) <= 0.05
    assert model_kwargs["manual_weight_candidates"] == [
        {"regime_horizon_cpu_anchor": 1.0, "finite_dnn_aux_cpu_safe": 0.0},
        {"regime_horizon_cpu_anchor": 0.98, "finite_dnn_aux_cpu_safe": 0.02},
        {"regime_horizon_cpu_anchor": 0.95, "finite_dnn_aux_cpu_safe": 0.05},
    ]


def test_balanced_variant_sits_between_lowweight_and_riskclamp_controls():
    lowweight = _load_config(CANDIDATE_CONFIG)
    riskclamp = _load_config(RISKCLAMP_CONFIG)
    balanced = _load_config(BALANCED_CONFIG)
    lowweight_kwargs = _model_kwargs(lowweight)
    riskclamp_kwargs = _model_kwargs(riskclamp)
    balanced_kwargs = _model_kwargs(balanced)
    lowweight_anchor = _base_specs(lowweight)[0]["model_kwargs"]
    riskclamp_anchor = _base_specs(riskclamp)[0]["model_kwargs"]
    balanced_anchor = _base_specs(balanced)[0]["model_kwargs"]
    lowweight_strategy = lowweight["port_analysis_config"]["strategy"]["kwargs"]
    riskclamp_strategy = riskclamp["port_analysis_config"]["strategy"]["kwargs"]
    balanced_strategy = balanced["port_analysis_config"]["strategy"]["kwargs"]

    assert balanced_strategy["topk"] == lowweight_strategy["topk"] == riskclamp_strategy["topk"] == 3
    assert riskclamp_strategy["risk_degree"] < balanced_strategy["risk_degree"] < lowweight_strategy["risk_degree"]
    assert (
        riskclamp_strategy["trade_impact_limit"]
        < balanced_strategy["trade_impact_limit"]
        < lowweight_strategy["trade_impact_limit"]
    )
    assert balanced_strategy["selection_rank_buffer"] == lowweight_strategy["selection_rank_buffer"] == 1
    assert balanced_strategy["selection_max_new_names"] == 1
    assert lowweight_kwargs["max_drawdown_penalty"] < balanced_kwargs["max_drawdown_penalty"] < riskclamp_kwargs["max_drawdown_penalty"]
    assert (
        min(lowweight_kwargs["volatility_penalty_grid"])
        < min(balanced_kwargs["volatility_penalty_grid"])
        < min(riskclamp_kwargs["volatility_penalty_grid"])
    )
    assert lowweight_kwargs["annret_weight"] > balanced_kwargs["annret_weight"] > riskclamp_kwargs["annret_weight"]
    assert min(balanced_anchor["risk_penalty_grid"]) == min(lowweight_anchor["risk_penalty_grid"])
    assert max(lowweight_anchor["risk_penalty_grid"]) < max(balanced_anchor["risk_penalty_grid"]) < max(riskclamp_anchor["risk_penalty_grid"])
    assert balanced_anchor["weight_constraints"] == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.92,
        "max_aux_weight": 0.08,
        "model_max_weights": {"lgb_h5": 0.08},
    }


def test_midrisk_variant_parses_and_keeps_actual_dnn_auxiliary_weight_low():
    candidate = _load_config(MIDRISK_CONFIG)
    model_kwargs = _model_kwargs(candidate)
    model = init_instance_by_config(_model_config(candidate))
    model.model_order = [spec.name for spec in model.specs]

    generated_weights = [
        dict(zip(model.model_order, weights.tolist()))
        for weights in model._weight_candidates(len(model.model_order))
    ]
    dnn_weights = [weights["finite_dnn_aux_cpu_safe"] for weights in generated_weights]

    assert _model_config(candidate)["class"] == "TranscendenceSignalEnsembleModel"
    assert _model_config(candidate)["module_path"] == "quant_master.contrib.model.transcendence_signal_ensemble"
    assert model_kwargs["weight_constraints"] == {"anchor_index": 0, "max_aux_weight": 0.05}
    assert model.weight_constraints == {"anchor_index": 0, "max_aux_weight": 0.05}
    assert max(dnn_weights) <= 0.05
    assert model_kwargs["manual_weight_candidates"] == [
        {"regime_horizon_cpu_anchor": 1.0, "finite_dnn_aux_cpu_safe": 0.0},
        {"regime_horizon_cpu_anchor": 0.98, "finite_dnn_aux_cpu_safe": 0.02},
        {"regime_horizon_cpu_anchor": 0.95, "finite_dnn_aux_cpu_safe": 0.05},
    ]


def test_midrisk_variant_uses_static_midpoint_risk_controls():
    balanced = _load_config(BALANCED_CONFIG)
    riskclamp = _load_config(RISKCLAMP_CONFIG)
    midrisk = _load_config(MIDRISK_CONFIG)
    balanced_kwargs = _model_kwargs(balanced)
    riskclamp_kwargs = _model_kwargs(riskclamp)
    midrisk_kwargs = _model_kwargs(midrisk)
    balanced_anchor = _base_specs(balanced)[0]["model_kwargs"]
    riskclamp_anchor = _base_specs(riskclamp)[0]["model_kwargs"]
    midrisk_anchor = _base_specs(midrisk)[0]["model_kwargs"]
    balanced_strategy = balanced["port_analysis_config"]["strategy"]["kwargs"]
    riskclamp_strategy = riskclamp["port_analysis_config"]["strategy"]["kwargs"]
    midrisk_strategy = midrisk["port_analysis_config"]["strategy"]["kwargs"]

    assert midrisk_strategy["topk"] == balanced_strategy["topk"] == riskclamp_strategy["topk"] == 3
    assert riskclamp_strategy["risk_degree"] < midrisk_strategy["risk_degree"] < balanced_strategy["risk_degree"]
    assert midrisk_strategy["risk_degree"] == 0.85
    assert (
        riskclamp_strategy["trade_impact_limit"]
        < midrisk_strategy["trade_impact_limit"]
        < balanced_strategy["trade_impact_limit"]
    )
    assert 0.18 <= midrisk_strategy["trade_impact_limit"] <= 0.2
    assert midrisk_strategy["selection_rank_buffer"] == balanced_strategy["selection_rank_buffer"] == 1
    assert midrisk_strategy["selection_max_new_names"] == 1
    assert balanced_kwargs["max_drawdown_penalty"] < midrisk_kwargs["max_drawdown_penalty"] < riskclamp_kwargs["max_drawdown_penalty"]
    assert midrisk_kwargs["max_drawdown_penalty"] == 1.0
    assert balanced_kwargs["annret_weight"] > midrisk_kwargs["annret_weight"] > riskclamp_kwargs["annret_weight"]
    assert midrisk_kwargs["turnover_penalty_grid"] == [0.0018, 0.0025, 0.0035]
    assert max(midrisk_kwargs["turnover_penalty_grid"]) < max(riskclamp_kwargs["turnover_penalty_grid"])
    assert midrisk_anchor["turnover_penalty_grid"] == [0.0018, 0.0025, 0.0035]
    assert midrisk_anchor["risk_penalty_grid"] == [0.12, 0.16, 0.22]
    assert (
        min(balanced_anchor["risk_penalty_grid"])
        < min(midrisk_anchor["risk_penalty_grid"])
        == min(riskclamp_anchor["risk_penalty_grid"])
    )
    assert max(balanced_anchor["risk_penalty_grid"]) == max(midrisk_anchor["risk_penalty_grid"])
    assert max(midrisk_anchor["risk_penalty_grid"]) < max(riskclamp_anchor["risk_penalty_grid"])
    assert midrisk_anchor["weight_constraints"] == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.94,
        "max_aux_weight": 0.06,
        "model_max_weights": {"lgb_h5": 0.06},
    }
    assert (
        balanced_anchor["weight_constraints"]["min_anchor_weight"]
        < midrisk_anchor["weight_constraints"]["min_anchor_weight"]
        < riskclamp_anchor["weight_constraints"]["min_anchor_weight"]
    )


def test_dynamicgate_variant_parses_and_keeps_auxiliary_weight_low():
    candidate = _load_config(DYNAMICGATE_CONFIG)
    model_kwargs = _model_kwargs(candidate)
    anchor_kwargs = _base_specs(candidate)[0]["model_kwargs"]
    strategy_kwargs = candidate["port_analysis_config"]["strategy"]["kwargs"]
    model = init_instance_by_config(_model_config(candidate))
    model.model_order = [spec.name for spec in model.specs]

    generated_weights = [
        dict(zip(model.model_order, weights.tolist()))
        for weights in model._weight_candidates(len(model.model_order))
    ]
    dnn_weights = [weights["finite_dnn_aux_cpu_safe"] for weights in generated_weights]
    gate = strategy_kwargs["dynamic_risk_gate"]

    assert _model_config(candidate)["class"] == "TranscendenceSignalEnsembleModel"
    assert strategy_kwargs["risk_degree"] >= 0.95
    assert gate == {
        "enabled": True,
        "mode": "drawdown",
        "lookback": 20,
        "drawdown_threshold": 0.08,
        "full_clamp_threshold": 0.16,
        "min_risk_degree": 0.45,
        "recovery_rate": 0.10,
        "decay_rate": 1.0,
    }
    assert model_kwargs["weight_constraints"] == {"anchor_index": 0, "max_aux_weight": 0.05}
    assert max(dnn_weights) <= 0.05
    assert anchor_kwargs["weight_constraints"] == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.9,
        "max_aux_weight": 0.05,
        "model_max_weights": {"lgb_h5": 0.05},
    }


def test_dynamicgate_tight_variant_modestly_tightens_gate_only():
    base = _load_config(DYNAMICGATE_CONFIG)
    candidate = _load_config(DYNAMICGATE_TIGHT_CONFIG)
    model_kwargs = _model_kwargs(candidate)
    anchor_kwargs = _base_specs(candidate)[0]["model_kwargs"]
    base_strategy = base["port_analysis_config"]["strategy"]["kwargs"]
    strategy_kwargs = candidate["port_analysis_config"]["strategy"]["kwargs"]
    base_gate = base_strategy["dynamic_risk_gate"]
    gate = strategy_kwargs["dynamic_risk_gate"]

    assert _model_config(candidate)["class"] == "TranscendenceSignalEnsembleModel"
    assert strategy_kwargs["risk_degree"] == base_strategy["risk_degree"] >= 0.95
    assert strategy_kwargs["trade_impact_limit"] == base_strategy["trade_impact_limit"]
    assert gate["enabled"] is True
    assert gate["mode"] == "drawdown"
    assert gate["lookback"] == base_gate["lookback"] == 20
    assert gate["drawdown_threshold"] == 0.07
    assert gate["full_clamp_threshold"] == 0.15
    assert gate["min_risk_degree"] == base_gate["min_risk_degree"] == 0.45
    assert gate["recovery_rate"] == 0.08
    assert gate["decay_rate"] == base_gate["decay_rate"] == 1.0
    assert gate["drawdown_threshold"] < base_gate["drawdown_threshold"]
    assert gate["full_clamp_threshold"] < base_gate["full_clamp_threshold"]
    assert gate["recovery_rate"] < base_gate["recovery_rate"]
    assert model_kwargs["weight_constraints"] == {"anchor_index": 0, "max_aux_weight": 0.05}
    assert anchor_kwargs["weight_constraints"]["max_aux_weight"] <= 0.05
    assert max(anchor_kwargs["weight_constraints"]["model_max_weights"].values()) <= 0.05
