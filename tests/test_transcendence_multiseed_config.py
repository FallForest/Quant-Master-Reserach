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
MULTISEED_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "workflow_config_regime_horizon_de_only_multiseed_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
)

EXPECTED_SPEC_SEEDS = {
    "de_h1": 42,
    "de_h1_seed57": 57,
    "de_h1_seed7": 7,
}


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _without_horizon_model_specs(config: dict) -> dict:
    copied = deepcopy(config)
    copied["task"]["model"]["kwargs"].pop("horizon_model_specs", None)
    return copied


def test_multiseed_candidate_is_7406_lockstep_except_horizon_model_specs():
    base = _load_config(BASE_7406_CONFIG)
    candidate = _load_config(MULTISEED_CONFIG)

    assert _without_horizon_model_specs(candidate) == _without_horizon_model_specs(base)


def test_multiseed_candidate_has_three_h1_specs_with_expected_seeds():
    base = _load_config(BASE_7406_CONFIG)
    candidate = _load_config(MULTISEED_CONFIG)
    base_specs = base["task"]["model"]["kwargs"]["horizon_model_specs"]
    candidate_specs = candidate["task"]["model"]["kwargs"]["horizon_model_specs"]

    assert [spec["name"] for spec in candidate_specs] == list(EXPECTED_SPEC_SEEDS)
    assert candidate_specs[0] == base_specs[0]

    base_spec_template = deepcopy(base_specs[0])
    for spec in candidate_specs:
        expected_seed = EXPECTED_SPEC_SEEDS[spec["name"]]
        assert spec["horizon"] == 1
        assert spec["model_kwargs"]["random_state"] == expected_seed

        comparable = deepcopy(spec)
        comparable["name"] = base_spec_template["name"]
        comparable["model_kwargs"]["random_state"] = base_spec_template["model_kwargs"]["random_state"]
        assert comparable == base_spec_template


def test_multiseed_candidate_has_no_test_scan_fields():
    candidate = _load_config(MULTISEED_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    assert "test_scan" not in model_kwargs
    assert "test_scan_grid" not in model_kwargs
    assert "test_scan_fields" not in model_kwargs


def test_multiseed_candidate_model_kwargs_instantiate():
    candidate = _load_config(MULTISEED_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert [spec.name for spec in model.model_specs] == list(EXPECTED_SPEC_SEEDS)
    assert [spec.horizon for spec in model.model_specs] == [1, 1, 1]
    assert [
        spec.model_kwargs["random_state"] for spec in model.model_specs
    ] == list(EXPECTED_SPEC_SEEDS.values())
