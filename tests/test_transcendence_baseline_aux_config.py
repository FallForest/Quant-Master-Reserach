from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from quant_master.contrib.model.regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel


ROOT = Path(__file__).resolve().parents[1]
BASE_7406_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
)
BASELINE_EXEC_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "workflow_config_regime_horizon_de_only_baseline_exec_Alpha158_2026_csi300.yaml"
)
CANDIDATE_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "workflow_config_regime_horizon_de_baseline_aux_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _without_horizon_model_specs(config: dict) -> dict:
    copied = deepcopy(config)
    copied["task"]["model"]["kwargs"].pop("horizon_model_specs", None)
    return copied


def _specs(config: dict) -> list[dict]:
    return config["task"]["model"]["kwargs"]["horizon_model_specs"]


def test_baseline_aux_candidate_is_7406_lockstep_except_horizon_model_specs():
    base = _load_config(BASE_7406_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)

    assert _without_horizon_model_specs(candidate) == _without_horizon_model_specs(base)


def test_baseline_aux_candidate_keeps_only_expected_h1_specs():
    base = _load_config(BASE_7406_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)
    candidate_specs = _specs(candidate)

    assert [spec["name"] for spec in candidate_specs] == ["de_h1", "de_h1_baseline_aux"]
    assert [spec["horizon"] for spec in candidate_specs] == [1, 1]
    assert candidate_specs[0] == _specs(base)[0]


def test_baseline_aux_spec_uses_baseline_exec_training_params_only():
    baseline_exec = _load_config(BASELINE_EXEC_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)

    expected_aux = deepcopy(_specs(baseline_exec)[0])
    expected_aux["name"] = "de_h1_baseline_aux"

    assert _specs(candidate)[1] == expected_aux


def test_baseline_aux_candidate_does_not_import_baseline_exec_strategy_params():
    base = _load_config(BASE_7406_CONFIG)
    baseline_exec = _load_config(BASELINE_EXEC_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)

    candidate_model_kwargs = candidate["task"]["model"]["kwargs"]
    base_model_kwargs = base["task"]["model"]["kwargs"]
    baseline_exec_model_kwargs = baseline_exec["task"]["model"]["kwargs"]

    assert candidate_model_kwargs["topk"] == base_model_kwargs["topk"] == 45
    assert candidate_model_kwargs["topk"] != baseline_exec_model_kwargs["topk"]
    assert candidate_model_kwargs["turnover_penalty"] == base_model_kwargs["turnover_penalty"] == 0.00005
    assert candidate_model_kwargs["memory_boost_grid"] == base_model_kwargs["memory_boost_grid"] == [0.0, 0.005]
    assert candidate["port_analysis_config"] == base["port_analysis_config"]
    assert candidate["port_analysis_config"] != baseline_exec["port_analysis_config"]


def test_baseline_aux_candidate_has_no_test_scan_fields():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    assert "test_scan" not in model_kwargs
    assert "test_scan_grid" not in model_kwargs
    assert "test_scan_fields" not in model_kwargs


def test_baseline_aux_candidate_model_kwargs_instantiate():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert [spec.name for spec in model.model_specs] == ["de_h1", "de_h1_baseline_aux"]
    assert [spec.horizon for spec in model.model_specs] == [1, 1]
