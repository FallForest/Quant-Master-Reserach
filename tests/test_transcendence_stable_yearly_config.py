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
STABLE_YEARLY_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "workflow_config_regime_horizon_de_only_stable_yearly_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
)

STABLE_YEARLY_KWARGS = {
    "selection_objective": "stable_yearly",
    "yearly_stability_penalty": 0.5,
    "min_valid_years": 3,
}


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _without_stable_yearly_kwargs(config: dict) -> dict:
    copied = deepcopy(config)
    model_kwargs = copied["task"]["model"]["kwargs"]
    for key in STABLE_YEARLY_KWARGS:
        model_kwargs.pop(key, None)
    return copied


def test_stable_yearly_candidate_is_7406_lockstep_except_stable_yearly_kwargs():
    base = _load_config(BASE_7406_CONFIG)
    candidate = _load_config(STABLE_YEARLY_CONFIG)

    assert _without_stable_yearly_kwargs(candidate) == base


def test_stable_yearly_candidate_has_only_expected_stable_yearly_kwargs():
    base = _load_config(BASE_7406_CONFIG)
    candidate = _load_config(STABLE_YEARLY_CONFIG)
    base_kwargs = base["task"]["model"]["kwargs"]
    candidate_kwargs = candidate["task"]["model"]["kwargs"]

    added_keys = set(candidate_kwargs) - set(base_kwargs)

    assert added_keys == set(STABLE_YEARLY_KWARGS)
    for key, value in STABLE_YEARLY_KWARGS.items():
        assert candidate_kwargs[key] == value
    assert "min_yearly_objective" not in candidate_kwargs


def test_stable_yearly_candidate_has_no_test_scan_fields():
    candidate = _load_config(STABLE_YEARLY_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    assert "test_scan" not in model_kwargs
    assert "test_scan_grid" not in model_kwargs
    assert "test_scan_fields" not in model_kwargs


def test_stable_yearly_candidate_model_kwargs_instantiate():
    candidate = _load_config(STABLE_YEARLY_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.selection_objective == "stable_yearly"
    assert model.yearly_stability_penalty == 0.5
    assert model.min_valid_years == 3
    assert model.min_yearly_objective is None
    assert [spec.name for spec in model.model_specs] == ["de_h1"]
    assert [spec.horizon for spec in model.model_specs] == [1]
