from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from quant_master.contrib.model.regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
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
    / "workflow_config_regime_horizon_de_only_penalty_grid_lockstep_Alpha158_2026_csi300.yaml"
)

EXPECTED_GRIDS = {
    "turnover_penalty_grid": [0.0, 0.000025, 0.00005, 0.0001],
    "risk_penalty_grid": [0.0, 0.02, 0.05],
    "memory_boost_grid": [0.0, 0.0025, 0.005, 0.01],
}
GRID_KEYS = set(EXPECTED_GRIDS)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _without_grid_params(config: dict) -> dict:
    copied = deepcopy(config)
    kwargs = copied["task"]["model"]["kwargs"]
    for key in GRID_KEYS:
        kwargs.pop(key, None)
    return copied


def test_penalty_grid_candidate_is_lockstep_except_valid_only_grids():
    base = _load_config(BASE_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)

    assert _without_grid_params(candidate) == _without_grid_params(base)
    assert candidate["task"]["model"]["kwargs"]["turnover_penalty_grid"] == EXPECTED_GRIDS["turnover_penalty_grid"]
    assert candidate["task"]["model"]["kwargs"]["risk_penalty_grid"] == EXPECTED_GRIDS["risk_penalty_grid"]
    assert candidate["task"]["model"]["kwargs"]["memory_boost_grid"] == EXPECTED_GRIDS["memory_boost_grid"]


def test_penalty_grid_candidate_model_kwargs_instantiate():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.turnover_penalty_grid == EXPECTED_GRIDS["turnover_penalty_grid"]
    assert model.risk_penalty_grid == EXPECTED_GRIDS["risk_penalty_grid"]
    assert model.memory_boost_grid == EXPECTED_GRIDS["memory_boost_grid"]
    assert model.turnover_penalty == 0.00005
    assert model.risk_penalty == 0.0
