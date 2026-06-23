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
    / "workflow_config_regime_horizon_smallfund_rank_shrink_Alpha158Alpha360_2026_csi300.yaml"
)

EXPECTED_GRIDS = {
    "turnover_penalty_grid": [0.0003, 0.0007, 0.0012, 0.0018],
    "risk_penalty_grid": [0.05, 0.08, 0.12],
    "memory_boost_grid": [0.0, 0.01, 0.02],
    "robust_rank_blend_grid": [0.0, 0.15, 0.3],
    "prediction_shrinkage_grid": [1.0, 0.9, 0.8],
}


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_smallfund_candidate_keeps_account_costs_and_topk_executable():
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


def test_smallfund_candidate_grids_and_horizons_match_design():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]
    specs = model_kwargs["horizon_model_specs"]

    assert [spec["name"] for spec in specs] == ["de_h1", "lgb_h5"]
    assert [spec["horizon"] for spec in specs] == [1, 5]
    assert str(candidate["data_handler_config"]["start_time"]) == "2008-01-01"
    assert [str(v) for v in candidate["task"]["dataset"]["kwargs"]["segments"]["train"]] == [
        "2008-01-01",
        "2018-12-31",
    ]
    for key, expected in EXPECTED_GRIDS.items():
        assert model_kwargs[key] == expected


def test_smallfund_candidate_model_kwargs_instantiate():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 3
    assert model.turnover_penalty == 0.0007
    assert model.risk_penalty == 0.08
    assert model.turnover_penalty_grid == EXPECTED_GRIDS["turnover_penalty_grid"]
    assert model.risk_penalty_grid == EXPECTED_GRIDS["risk_penalty_grid"]
    assert model.memory_boost_grid == EXPECTED_GRIDS["memory_boost_grid"]
    assert model.robust_rank_blend_grid == EXPECTED_GRIDS["robust_rank_blend_grid"]
    assert model.prediction_shrinkage_grid == EXPECTED_GRIDS["prediction_shrinkage_grid"]
