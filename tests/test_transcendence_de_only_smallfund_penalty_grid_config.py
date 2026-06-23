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
    / "workflow_config_regime_horizon_de_only_smallfund_penalty_grid_Alpha158_2026_csi300.yaml"
)

EXPECTED_GRIDS = {
    "turnover_penalty_grid": [0.0003, 0.0007, 0.0012, 0.0018],
    "risk_penalty_grid": [0.0, 0.02, 0.05],
    "memory_boost_grid": [0.0, 0.0025, 0.005, 0.01],
}


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_de_only_smallfund_candidate_keeps_cash_cost_and_execution_shape():
    candidate = _load_config(CANDIDATE_CONFIG)
    strategy_kwargs = candidate["port_analysis_config"]["strategy"]["kwargs"]
    backtest_config = candidate["port_analysis_config"]["backtest"]
    exchange_kwargs = backtest_config["exchange_kwargs"]
    model_kwargs = candidate["task"]["model"]["kwargs"]

    assert backtest_config["account"] == 13000
    assert exchange_kwargs["open_cost"] == 0.0001
    assert exchange_kwargs["close_cost"] == 0.0006
    assert exchange_kwargs["min_cost"] == 0
    assert strategy_kwargs["topk"] == 3
    assert strategy_kwargs["n_drop"] == 1
    assert model_kwargs["topk"] == 3


def test_de_only_smallfund_candidate_is_de_only_with_cost_grid():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]
    specs = model_kwargs["horizon_model_specs"]

    assert [spec["name"] for spec in specs] == ["de_h1"]
    assert [spec["model_type"] for spec in specs] == ["double_ensemble"]
    assert [spec["horizon"] for spec in specs] == [1]
    assert model_kwargs["use_rank_score"] is False
    assert model_kwargs["zscore_clip"] == 100.0
    assert model_kwargs["neutralize_daily_mean"] is False
    for key, expected in EXPECTED_GRIDS.items():
        assert model_kwargs[key] == expected


def test_de_only_smallfund_candidate_model_kwargs_instantiate():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 3
    assert model.turnover_penalty == 0.0007
    assert model.risk_penalty == 0.0
    assert model.turnover_penalty_grid == EXPECTED_GRIDS["turnover_penalty_grid"]
    assert model.risk_penalty_grid == EXPECTED_GRIDS["risk_penalty_grid"]
    assert model.memory_boost_grid == EXPECTED_GRIDS["memory_boost_grid"]
