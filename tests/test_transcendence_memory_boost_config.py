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
CANDIDATE_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "workflow_config_regime_horizon_de_only_memory_boost_lockstep_Alpha158_2026_csi300.yaml"
)

EXPECTED_MEMORY_BOOST_GRID = [0.0, 0.0025, 0.005, 0.01]


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _without_memory_boost_grid(config: dict) -> dict:
    copied = deepcopy(config)
    copied["task"]["model"]["kwargs"].pop("memory_boost_grid", None)
    return copied


def test_memory_boost_candidate_is_7406_lockstep_except_memory_grid():
    base = _load_config(BASE_7406_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)

    assert _without_memory_boost_grid(candidate) == _without_memory_boost_grid(base)
    assert candidate["task"]["model"]["kwargs"]["memory_boost_grid"] == EXPECTED_MEMORY_BOOST_GRID


def test_memory_boost_candidate_has_no_penalty_grid_extension():
    base = _load_config(BASE_7406_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)
    base_kwargs = base["task"]["model"]["kwargs"]
    candidate_kwargs = candidate["task"]["model"]["kwargs"]

    assert "turnover_penalty_grid" not in candidate_kwargs
    assert "risk_penalty_grid" not in candidate_kwargs
    assert candidate_kwargs["turnover_penalty"] == base_kwargs["turnover_penalty"]
    assert float(candidate_kwargs["turnover_penalty"]) == 5e-05
    assert candidate_kwargs["risk_penalty"] == base_kwargs["risk_penalty"] == 0.0


def test_memory_boost_candidate_model_kwargs_instantiate():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.memory_boost_grid == EXPECTED_MEMORY_BOOST_GRID
    assert model.turnover_penalty_grid == []
    assert model.risk_penalty_grid == []
    assert model.turnover_penalty == 5e-05
    assert model.risk_penalty == 0.0
