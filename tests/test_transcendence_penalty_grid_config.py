from __future__ import annotations

from pathlib import Path

import yaml

from quant_master.contrib.model.regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_penalty_grid_lockstep_Alpha158_2026_csi300.yaml"
)

EXPECTED_GRIDS = {
    "turnover_penalty_grid": [0.0, 0.000175, 0.00035, 0.0007],
    "risk_penalty_grid": [0.0, 0.02, 0.05],
    "memory_boost_grid": [0.0, 0.0025, 0.005, 0.01],
}


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_penalty_grid_candidate_keeps_legacy_execution_shape_and_grids():
    candidate = _load_config(CANDIDATE_CONFIG)
    strategy_kwargs = candidate["port_analysis_config"]["strategy"]["kwargs"]
    backtest_config = candidate["port_analysis_config"]["backtest"]
    exchange_kwargs = backtest_config["exchange_kwargs"]
    model_kwargs = candidate["task"]["model"]["kwargs"]

    assert backtest_config["account"] == 100000000
    assert exchange_kwargs["open_cost"] == 0.0001
    assert exchange_kwargs["close_cost"] == 0.0006
    assert exchange_kwargs["min_cost"] == 0
    assert strategy_kwargs["topk"] == 45
    assert strategy_kwargs["n_drop"] == 4
    assert model_kwargs["topk"] == 45
    assert model_kwargs["turnover_penalty_grid"] == EXPECTED_GRIDS["turnover_penalty_grid"]
    assert model_kwargs["risk_penalty_grid"] == EXPECTED_GRIDS["risk_penalty_grid"]
    assert model_kwargs["memory_boost_grid"] == EXPECTED_GRIDS["memory_boost_grid"]


def test_penalty_grid_candidate_model_kwargs_instantiate():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.turnover_penalty_grid == EXPECTED_GRIDS["turnover_penalty_grid"]
    assert model.risk_penalty_grid == EXPECTED_GRIDS["risk_penalty_grid"]
    assert model.memory_boost_grid == EXPECTED_GRIDS["memory_boost_grid"]
    assert model.turnover_penalty == 0.0007
    assert model.risk_penalty == 0.0
