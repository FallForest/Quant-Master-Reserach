from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from quant_master.contrib.model.regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel
from quant_master.utils import init_instance_by_config


ROOT = Path(__file__).resolve().parents[1]
TRANS_DIR = ROOT / "examples" / "benchmarks" / "Transcendence"
BASE_7406_CONFIG = (
    TRANS_DIR
    / "workflow_config_regime_horizon_de_only_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
)
REFERENCE_DE_LGB_CONFIG = (
    TRANS_DIR
    / "workflow_config_regime_horizon_de_lgb_rank_preserving_cost_exec_topk45_drop4_Alpha158_2026_csi300.yaml"
)
CANDIDATE_CONFIG = (
    TRANS_DIR
    / "workflow_config_regime_horizon_de_lgb_relaxed_rank_preserving_cost_exec_Alpha158_2026_csi300.yaml"
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


def _model_kwargs(config: dict) -> dict:
    return config["task"]["model"]["kwargs"]


def test_de_lgb_relaxed_candidate_is_7406_lockstep_except_horizon_model_specs():
    base = _load_config(BASE_7406_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)

    assert _without_horizon_model_specs(candidate) == _without_horizon_model_specs(base)


def test_de_lgb_relaxed_candidate_keeps_de_h1_and_adds_lgb_h5_only():
    base = _load_config(BASE_7406_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)
    candidate_specs = _specs(candidate)

    assert [spec["name"] for spec in candidate_specs] == ["de_h1", "lgb_h5"]
    assert [spec["model_type"] for spec in candidate_specs] == ["double_ensemble", "lightgbm"]
    assert [spec["horizon"] for spec in candidate_specs] == [1, 5]
    assert candidate_specs[0] == _specs(base)[0]


def test_de_lgb_relaxed_lgb_h5_matches_reference_lgb_spec():
    reference = _load_config(REFERENCE_DE_LGB_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)
    reference_lgb = _specs(reference)[1]
    candidate_lgb = _specs(candidate)[1]

    assert candidate_lgb["name"] == "lgb_h5"
    assert candidate_lgb == reference_lgb


def test_de_lgb_relaxed_keeps_7406_search_penalty_and_relaxed_monotonic_settings():
    base = _load_config(BASE_7406_CONFIG)
    reference = _load_config(REFERENCE_DE_LGB_CONFIG)
    candidate = _load_config(CANDIDATE_CONFIG)
    base_kwargs = _model_kwargs(base)
    reference_kwargs = _model_kwargs(reference)
    candidate_kwargs = _model_kwargs(candidate)

    assert candidate_kwargs["topk"] == base_kwargs["topk"] == 45
    assert candidate_kwargs["search_step"] == base_kwargs["search_step"] == 0.1
    assert candidate_kwargs["turnover_penalty"] == base_kwargs["turnover_penalty"] == 0.00005
    assert candidate_kwargs["risk_penalty"] == base_kwargs["risk_penalty"] == 0.0
    assert candidate_kwargs["memory_boost_grid"] == base_kwargs["memory_boost_grid"] == [0.0, 0.005]
    assert candidate_kwargs["enforce_horizon_monotonic"] is False
    assert reference_kwargs["enforce_horizon_monotonic"] is True
    assert "monotonic_direction" not in candidate_kwargs
    assert "regime_consensus_quantiles" not in candidate_kwargs
    assert "regime_disagreement_quantiles" not in candidate_kwargs
    assert "min_regime_samples" not in candidate_kwargs


def test_de_lgb_relaxed_candidate_has_no_test_scan_fields():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = _model_kwargs(candidate)

    assert "test_scan" not in model_kwargs
    assert "test_scan_grid" not in model_kwargs
    assert "test_scan_fields" not in model_kwargs


def test_de_lgb_relaxed_candidate_model_kwargs_instantiate():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = _model_kwargs(candidate)

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert [spec.name for spec in model.model_specs] == ["de_h1", "lgb_h5"]
    assert [spec.model_type for spec in model.model_specs] == ["double_ensemble", "lightgbm"]
    assert [spec.horizon for spec in model.model_specs] == [1, 5]
    assert model.enforce_horizon_monotonic is False


def test_de_lgb_relaxed_yaml_model_import_smoke():
    candidate = _load_config(CANDIDATE_CONFIG)

    model = init_instance_by_config(candidate["task"]["model"])

    assert isinstance(model, RegimeHorizonCostEnsembleModel)
    assert [spec.name for spec in model.model_specs] == ["de_h1", "lgb_h5"]
