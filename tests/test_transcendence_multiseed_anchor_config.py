from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from quant_master.contrib.model.regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel


ROOT = Path(__file__).resolve().parents[1]
MULTISEED_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "workflow_config_regime_horizon_de_only_multiseed_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
)
ANCHOR_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "workflow_config_regime_horizon_de_only_multiseed_anchor80_aux20_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
)

EXPECTED_WEIGHT_CONSTRAINTS = {
    "anchor_model": "de_h1",
    "min_anchor_weight": 0.8,
    "max_aux_weight": 0.2,
    "model_max_weights": {
        "de_h1_seed57": 0.2,
        "de_h1_seed7": 0.2,
    },
}


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _without_weight_constraints(config: dict) -> dict:
    copied = deepcopy(config)
    copied["task"]["model"]["kwargs"].pop("weight_constraints", None)
    return copied


def test_multiseed_anchor_candidate_is_multiseed_lockstep_except_constraints():
    multiseed = _load_config(MULTISEED_CONFIG)
    candidate = _load_config(ANCHOR_CONFIG)

    assert _without_weight_constraints(candidate) == _without_weight_constraints(multiseed)


def test_multiseed_anchor_candidate_has_expected_weight_constraints():
    candidate = _load_config(ANCHOR_CONFIG)

    assert candidate["task"]["model"]["kwargs"]["weight_constraints"] == EXPECTED_WEIGHT_CONSTRAINTS


def test_multiseed_anchor_candidate_model_kwargs_instantiate_with_constraints():
    candidate = _load_config(ANCHOR_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert isinstance(model, RegimeHorizonCostEnsembleModel)
    assert model.weight_constraints == EXPECTED_WEIGHT_CONSTRAINTS
    assert [spec.name for spec in model.model_specs] == ["de_h1", "de_h1_seed57", "de_h1_seed7"]
