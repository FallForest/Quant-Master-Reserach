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
    / "workflow_config_regime_horizon_de_only_smallfund_lowturnover_Alpha158_2026_csi300.yaml"
)
HOLD3_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold3_Alpha158_2026_csi300.yaml"
)
NO_MEMORY_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold2_nomemory_Alpha158_2026_csi300.yaml"
)
STABLEBUDGET_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold2_stablebudget_Alpha158_2026_csi300.yaml"
)
STABLE_RISK_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold2_stable_risk_Alpha158_2026_csi300.yaml"
)
H1H5_ANCHOR_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold2_h1h5_anchor_Alpha158_2026_csi300.yaml"
)
H1H5_ANCHOR_STABLE_RISK_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold2_h1h5_anchor_stable_risk_Alpha158_2026_csi300.yaml"
)
H1H5_ANCHOR_TOPK2_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold2_h1h5_anchor_topk2_Alpha158_2026_csi300.yaml"
)
H1H5_ANCHOR_OBJECTIVE_RANKDECILE_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold2_h1h5_anchor_objective_rankdecile_Alpha158_2026_csi300.yaml"
)
H1H5_ANCHOR_AUX15_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold2_h1h5_anchor_aux15_Alpha158_2026_csi300.yaml"
)
H1H5_ANCHOR_NDROP0_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold2_h1h5_anchor_ndrop0_Alpha158_2026_csi300.yaml"
)
H1H5_ANCHOR_STABLE_FLOOR_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold2_h1h5_anchor_stable_floor_Alpha158_2026_csi300.yaml"
)
HOLD3_H1H5_ANCHOR_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_hold3_h1h5_anchor_Alpha158_2026_csi300.yaml"
)
SOFTTOPK_H1H5_ANCHOR_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "regime_horizon"
    / "variants"
    / "workflow_config_regime_horizon_de_only_smallfund_softtopk_h1h5_anchor_Alpha158_2026_csi300.yaml"
)

EXPECTED_GRIDS = {
    "turnover_penalty_grid": [0.0012, 0.0018, 0.0025, 0.0035],
    "risk_penalty_grid": [0.0, 0.02, 0.05],
    "memory_boost_grid": [0.0, 0.0025, 0.005, 0.01],
}


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_lowturnover_candidate_keeps_cash_cost_and_execution_shape():
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
    assert strategy_kwargs["hold_thresh"] == 2
    assert model_kwargs["topk"] == 3


def test_lowturnover_candidate_is_de_only_with_stronger_turnover_grid():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]
    specs = model_kwargs["horizon_model_specs"]

    assert [spec["name"] for spec in specs] == ["de_h1"]
    assert [spec["model_type"] for spec in specs] == ["double_ensemble"]
    assert [spec["horizon"] for spec in specs] == [1]
    assert model_kwargs["turnover_penalty"] == 0.0018
    assert model_kwargs["use_rank_score"] is False
    assert model_kwargs["zscore_clip"] == 100.0
    assert model_kwargs["neutralize_daily_mean"] is False
    for key, expected in EXPECTED_GRIDS.items():
        assert model_kwargs[key] == expected


def test_lowturnover_candidate_model_kwargs_instantiate():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 3
    assert model.turnover_penalty == 0.0018
    assert model.risk_penalty == 0.0
    assert model.turnover_penalty_grid == EXPECTED_GRIDS["turnover_penalty_grid"]
    assert model.risk_penalty_grid == EXPECTED_GRIDS["risk_penalty_grid"]
    assert model.memory_boost_grid == EXPECTED_GRIDS["memory_boost_grid"]


def test_hold3_ablation_only_changes_minimum_holding_threshold():
    hold2 = _load_config(CANDIDATE_CONFIG)
    hold3 = _load_config(HOLD3_CONFIG)

    hold2_strategy = hold2["port_analysis_config"]["strategy"]["kwargs"]
    hold3_strategy = hold3["port_analysis_config"]["strategy"]["kwargs"]

    assert hold2_strategy["hold_thresh"] == 2
    assert hold3_strategy["hold_thresh"] == 3
    for key in ("topk", "n_drop"):
        assert hold2_strategy[key] == hold3_strategy[key]
    assert hold2["port_analysis_config"]["backtest"] == hold3["port_analysis_config"]["backtest"]
    assert hold2["task"]["model"]["kwargs"] == hold3["task"]["model"]["kwargs"]
    assert hold2["task"]["dataset"]["kwargs"]["segments"] == hold3["task"]["dataset"]["kwargs"]["segments"]


def test_nomemory_ablation_only_disables_model_memory_boost_grid():
    hold2 = _load_config(CANDIDATE_CONFIG)
    nomemory = _load_config(NO_MEMORY_CONFIG)

    hold2_model_kwargs = hold2["task"]["model"]["kwargs"]
    nomemory_model_kwargs = nomemory["task"]["model"]["kwargs"]
    expected_nomemory_kwargs = dict(hold2_model_kwargs)
    expected_nomemory_kwargs["memory_boost_grid"] = [0.0]

    assert nomemory["port_analysis_config"] == hold2["port_analysis_config"]
    assert nomemory_model_kwargs == expected_nomemory_kwargs
    assert nomemory["task"]["dataset"]["kwargs"]["segments"] == hold2["task"]["dataset"]["kwargs"]["segments"]


def test_stablebudget_candidate_locks_smoke_stable_training_shape():
    candidate = _load_config(STABLEBUDGET_CONFIG)
    strategy_kwargs = candidate["port_analysis_config"]["strategy"]["kwargs"]
    backtest_config = candidate["port_analysis_config"]["backtest"]
    exchange_kwargs = backtest_config["exchange_kwargs"]
    model_kwargs = candidate["task"]["model"]["kwargs"]
    de_kwargs = model_kwargs["horizon_model_specs"][0]["model_kwargs"]

    assert backtest_config["account"] == 13000
    assert exchange_kwargs["open_cost"] == 0.0001
    assert exchange_kwargs["close_cost"] == 0.0006
    assert exchange_kwargs["min_cost"] == 0
    assert strategy_kwargs["topk"] == 3
    assert strategy_kwargs["n_drop"] == 1
    assert strategy_kwargs["hold_thresh"] == 2
    assert model_kwargs["topk"] == 3
    assert model_kwargs["memory_boost_grid"] == [0.0]
    assert model_kwargs["turnover_penalty_grid"] == EXPECTED_GRIDS["turnover_penalty_grid"]
    assert model_kwargs["risk_penalty_grid"] == EXPECTED_GRIDS["risk_penalty_grid"]
    assert de_kwargs["num_models"] == 1
    assert de_kwargs["epochs"] == 4
    assert de_kwargs["enable_sr"] is False
    assert de_kwargs["enable_fs"] is False
    assert de_kwargs["sub_weights"] == [1]
    assert de_kwargs["num_threads"] == 4


def test_stablebudget_candidate_model_kwargs_instantiate():
    candidate = _load_config(STABLEBUDGET_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 3
    assert model.turnover_penalty == 0.0018
    assert model.risk_penalty == 0.0
    assert model.memory_boost_grid == [0.0]
    assert model.turnover_penalty_grid == EXPECTED_GRIDS["turnover_penalty_grid"]
    assert model.risk_penalty_grid == EXPECTED_GRIDS["risk_penalty_grid"]


def test_stable_risk_candidate_keeps_stable_training_shape_with_risk_selection():
    stable = _load_config(STABLEBUDGET_CONFIG)
    risk = _load_config(STABLE_RISK_CONFIG)

    stable_kwargs = stable["task"]["model"]["kwargs"]
    risk_kwargs = risk["task"]["model"]["kwargs"]
    expected_risk_kwargs = dict(stable_kwargs)
    expected_risk_kwargs["risk_penalty"] = 0.02
    expected_risk_kwargs["risk_penalty_grid"] = [0.0, 0.02, 0.05, 0.08, 0.12]
    expected_risk_kwargs["selection_objective"] = "stable_yearly"
    expected_risk_kwargs["yearly_stability_penalty"] = 0.5
    expected_risk_kwargs["min_valid_years"] = 3

    assert risk["port_analysis_config"] == stable["port_analysis_config"]
    assert risk_kwargs == expected_risk_kwargs
    assert risk["task"]["dataset"]["kwargs"]["segments"] == stable["task"]["dataset"]["kwargs"]["segments"]


def test_stable_risk_candidate_model_kwargs_instantiate():
    candidate = _load_config(STABLE_RISK_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 3
    assert model.turnover_penalty == 0.0018
    assert model.risk_penalty == 0.02
    assert model.memory_boost_grid == [0.0]
    assert model.turnover_penalty_grid == EXPECTED_GRIDS["turnover_penalty_grid"]
    assert model.risk_penalty_grid == [0.0, 0.02, 0.05, 0.08, 0.12]
    assert model.selection_objective == "stable_yearly"
    assert model.yearly_stability_penalty == 0.5
    assert model.min_valid_years == 3


def test_h1h5_anchor_candidate_only_adds_bounded_auxiliary_horizon():
    stable = _load_config(STABLEBUDGET_CONFIG)
    h1h5 = _load_config(H1H5_ANCHOR_CONFIG)

    stable_kwargs = stable["task"]["model"]["kwargs"]
    h1h5_kwargs = h1h5["task"]["model"]["kwargs"]
    specs = h1h5_kwargs["horizon_model_specs"]

    assert h1h5["port_analysis_config"] == stable["port_analysis_config"]
    assert h1h5["task"]["dataset"]["kwargs"]["segments"] == stable["task"]["dataset"]["kwargs"]["segments"]
    assert [spec["name"] for spec in specs] == ["de_h1", "de_h5"]
    assert [spec["model_type"] for spec in specs] == ["double_ensemble", "double_ensemble"]
    assert [spec["horizon"] for spec in specs] == [1, 5]
    assert specs[0]["model_kwargs"] == stable_kwargs["horizon_model_specs"][0]["model_kwargs"]
    assert specs[1]["model_kwargs"]["random_state"] == 57
    for key in (
        "topk",
        "search_step",
        "turnover_penalty",
        "turnover_penalty_grid",
        "risk_penalty",
        "risk_penalty_grid",
        "memory_boost_grid",
        "use_rank_score",
        "zscore_clip",
        "neutralize_daily_mean",
        "enforce_horizon_monotonic",
        "random_state",
    ):
        assert h1h5_kwargs[key] == stable_kwargs[key]
    assert h1h5_kwargs["weight_constraints"] == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.7,
        "max_aux_weight": 0.3,
    }


def test_h1h5_anchor_candidate_model_kwargs_instantiate():
    candidate = _load_config(H1H5_ANCHOR_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 3
    assert [spec.name for spec in model.model_specs] == ["de_h1", "de_h5"]
    assert [spec.horizon for spec in model.model_specs] == [1, 5]
    assert model.weight_constraints == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.7,
        "max_aux_weight": 0.3,
    }


def test_h1h5_anchor_stable_risk_only_changes_validation_selection():
    anchor = _load_config(H1H5_ANCHOR_CONFIG)
    stable_risk = _load_config(H1H5_ANCHOR_STABLE_RISK_CONFIG)

    anchor_kwargs = anchor["task"]["model"]["kwargs"]
    stable_risk_kwargs = stable_risk["task"]["model"]["kwargs"]
    expected_kwargs = dict(anchor_kwargs)
    expected_kwargs["risk_penalty"] = 0.02
    expected_kwargs["risk_penalty_grid"] = [0.0, 0.02, 0.05, 0.08, 0.12]
    expected_kwargs["selection_objective"] = "stable_yearly"
    expected_kwargs["yearly_stability_penalty"] = 0.5
    expected_kwargs["min_valid_years"] = 3

    assert stable_risk["port_analysis_config"] == anchor["port_analysis_config"]
    assert stable_risk["task"]["dataset"] == anchor["task"]["dataset"]
    assert stable_risk_kwargs == expected_kwargs
    assert stable_risk["task"]["record"][-1]["kwargs"]["config"] == stable_risk["port_analysis_config"]


def test_h1h5_anchor_stable_risk_model_kwargs_instantiate():
    candidate = _load_config(H1H5_ANCHOR_STABLE_RISK_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 3
    assert [spec.name for spec in model.model_specs] == ["de_h1", "de_h5"]
    assert model.risk_penalty == 0.02
    assert model.risk_penalty_grid == [0.0, 0.02, 0.05, 0.08, 0.12]
    assert model.selection_objective == "stable_yearly"
    assert model.yearly_stability_penalty == 0.5
    assert model.min_valid_years == 3
    assert model.weight_constraints == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.7,
        "max_aux_weight": 0.3,
    }


def test_h1h5_anchor_topk2_candidate_only_tightens_position_count():
    anchor = _load_config(H1H5_ANCHOR_CONFIG)
    topk2 = _load_config(H1H5_ANCHOR_TOPK2_CONFIG)

    anchor_strategy = anchor["port_analysis_config"]["strategy"]["kwargs"]
    topk2_strategy = topk2["port_analysis_config"]["strategy"]["kwargs"]
    anchor_kwargs = anchor["task"]["model"]["kwargs"]
    topk2_kwargs = topk2["task"]["model"]["kwargs"]
    expected_kwargs = dict(anchor_kwargs)
    expected_kwargs["topk"] = 2

    assert topk2["port_analysis_config"]["backtest"] == anchor["port_analysis_config"]["backtest"]
    assert topk2["task"]["dataset"] == anchor["task"]["dataset"]
    assert topk2_strategy["topk"] == 2
    assert topk2_strategy["n_drop"] == anchor_strategy["n_drop"]
    assert topk2_strategy["hold_thresh"] == anchor_strategy["hold_thresh"]
    assert topk2_kwargs == expected_kwargs
    assert topk2["task"]["record"][-1]["kwargs"]["config"] == topk2["port_analysis_config"]


def test_h1h5_anchor_topk2_model_kwargs_instantiate():
    candidate = _load_config(H1H5_ANCHOR_TOPK2_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 2
    assert [spec.name for spec in model.model_specs] == ["de_h1", "de_h5"]
    assert model.weight_constraints == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.7,
        "max_aux_weight": 0.3,
    }


def test_h1h5_anchor_objective_rankdecile_only_changes_training_label():
    anchor = _load_config(H1H5_ANCHOR_CONFIG)
    objective = _load_config(H1H5_ANCHOR_OBJECTIVE_RANKDECILE_CONFIG)

    anchor_kwargs = anchor["task"]["model"]["kwargs"]
    objective_kwargs = objective["task"]["model"]["kwargs"]
    expected_kwargs = dict(anchor_kwargs)
    expected_kwargs.update(
        {
            "objective_label_mode": "rank_decile_spread",
            "objective_horizon_days": [1, 5, 10],
            "objective_horizon_weights": [0.6, 0.3, 0.1],
            "objective_market_relative": True,
            "objective_vol_adjust": True,
            "objective_vol_window": 20,
            "objective_vol_floor": 0.0001,
            "objective_rank_power": 1.0,
            "objective_decile": 0.1,
            "objective_decile_scale": 0.25,
            "objective_clip": 6.0,
        }
    )

    assert objective["port_analysis_config"] == anchor["port_analysis_config"]
    assert objective["task"]["dataset"] == anchor["task"]["dataset"]
    assert objective_kwargs == expected_kwargs
    assert objective["task"]["record"][-1]["kwargs"]["config"] == objective["port_analysis_config"]


def test_h1h5_anchor_objective_rankdecile_model_kwargs_instantiate():
    candidate = _load_config(H1H5_ANCHOR_OBJECTIVE_RANKDECILE_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 3
    assert [spec.name for spec in model.model_specs] == ["de_h1", "de_h5"]
    assert model.objective_enabled is True
    assert model.objective_label_mode == "rank_decile_spread"
    assert model.objective_horizon_days == [1, 5, 10]
    assert model.objective_horizon_weights == [0.6, 0.3, 0.1]
    assert model.objective_market_relative is True
    assert model.objective_vol_adjust is True
    assert model.objective_decile_scale == 0.25
    assert model.weight_constraints == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.7,
        "max_aux_weight": 0.3,
    }


def test_h1h5_anchor_aux15_candidate_only_tightens_auxiliary_weight_cap():
    anchor = _load_config(H1H5_ANCHOR_CONFIG)
    aux15 = _load_config(H1H5_ANCHOR_AUX15_CONFIG)

    anchor_kwargs = anchor["task"]["model"]["kwargs"]
    aux15_kwargs = aux15["task"]["model"]["kwargs"]
    expected_kwargs = dict(anchor_kwargs)
    expected_kwargs["weight_constraints"] = {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.85,
        "max_aux_weight": 0.15,
    }

    assert aux15["port_analysis_config"] == anchor["port_analysis_config"]
    assert aux15["task"]["dataset"] == anchor["task"]["dataset"]
    assert aux15_kwargs == expected_kwargs
    assert aux15["task"]["record"][-1]["kwargs"]["config"] == aux15["port_analysis_config"]


def test_h1h5_anchor_aux15_model_kwargs_instantiate():
    candidate = _load_config(H1H5_ANCHOR_AUX15_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 3
    assert [spec.name for spec in model.model_specs] == ["de_h1", "de_h5"]
    assert model.weight_constraints == {
        "anchor_model": "de_h1",
        "min_anchor_weight": 0.85,
        "max_aux_weight": 0.15,
    }


def test_h1h5_anchor_ndrop0_candidate_only_disables_forced_dropout():
    anchor = _load_config(H1H5_ANCHOR_CONFIG)
    ndrop0 = _load_config(H1H5_ANCHOR_NDROP0_CONFIG)

    anchor_strategy = anchor["port_analysis_config"]["strategy"]["kwargs"]
    ndrop0_strategy = ndrop0["port_analysis_config"]["strategy"]["kwargs"]

    assert ndrop0["port_analysis_config"]["backtest"] == anchor["port_analysis_config"]["backtest"]
    assert ndrop0["task"]["model"] == anchor["task"]["model"]
    assert ndrop0["task"]["dataset"] == anchor["task"]["dataset"]
    assert ndrop0_strategy["topk"] == anchor_strategy["topk"]
    assert ndrop0_strategy["n_drop"] == 0
    assert ndrop0_strategy["hold_thresh"] == anchor_strategy["hold_thresh"]
    assert ndrop0["task"]["record"][-1]["kwargs"]["config"] == ndrop0["port_analysis_config"]


def test_h1h5_anchor_stable_floor_only_changes_validation_selection():
    anchor = _load_config(H1H5_ANCHOR_CONFIG)
    stable_floor = _load_config(H1H5_ANCHOR_STABLE_FLOOR_CONFIG)

    anchor_kwargs = anchor["task"]["model"]["kwargs"]
    stable_floor_kwargs = stable_floor["task"]["model"]["kwargs"]
    expected_kwargs = dict(anchor_kwargs)
    expected_kwargs["selection_objective"] = "stable_yearly"
    expected_kwargs["yearly_stability_penalty"] = 0.25
    expected_kwargs["min_yearly_objective"] = 0.0
    expected_kwargs["min_valid_years"] = 3

    assert stable_floor["port_analysis_config"] == anchor["port_analysis_config"]
    assert stable_floor["task"]["dataset"] == anchor["task"]["dataset"]
    assert stable_floor_kwargs == expected_kwargs
    assert stable_floor["task"]["record"][-1]["kwargs"]["config"] == stable_floor["port_analysis_config"]


def test_h1h5_anchor_stable_floor_model_kwargs_instantiate():
    candidate = _load_config(H1H5_ANCHOR_STABLE_FLOOR_CONFIG)
    model_kwargs = candidate["task"]["model"]["kwargs"]

    model = RegimeHorizonCostEnsembleModel(**model_kwargs)

    assert model.topk == 3
    assert [spec.name for spec in model.model_specs] == ["de_h1", "de_h5"]
    assert model.selection_objective == "stable_yearly"
    assert model.yearly_stability_penalty == 0.25
    assert model.min_yearly_objective == 0.0
    assert model.min_valid_years == 3


def test_hold3_h1h5_anchor_ablation_only_changes_minimum_holding_threshold():
    hold2 = _load_config(H1H5_ANCHOR_CONFIG)
    hold3 = _load_config(HOLD3_H1H5_ANCHOR_CONFIG)

    hold2_strategy = hold2["port_analysis_config"]["strategy"]["kwargs"]
    hold3_strategy = hold3["port_analysis_config"]["strategy"]["kwargs"]

    assert hold2_strategy["hold_thresh"] == 2
    assert hold3_strategy["hold_thresh"] == 3
    for key in ("topk", "n_drop"):
        assert hold2_strategy[key] == hold3_strategy[key]
    assert hold2["port_analysis_config"]["backtest"] == hold3["port_analysis_config"]["backtest"]
    assert hold2["task"]["model"] == hold3["task"]["model"]
    assert hold2["task"]["dataset"] == hold3["task"]["dataset"]
    assert hold3["task"]["record"][-1]["kwargs"]["config"] == hold3["port_analysis_config"]


def test_softtopk_h1h5_anchor_only_changes_execution_strategy():
    hold2 = _load_config(H1H5_ANCHOR_CONFIG)
    soft = _load_config(SOFTTOPK_H1H5_ANCHOR_CONFIG)

    strategy = soft["port_analysis_config"]["strategy"]
    strategy_kwargs = strategy["kwargs"]

    assert soft["port_analysis_config"]["backtest"] == hold2["port_analysis_config"]["backtest"]
    assert soft["task"]["model"] == hold2["task"]["model"]
    assert soft["task"]["dataset"] == hold2["task"]["dataset"]
    assert strategy["class"] == "SoftTopkStrategy"
    assert strategy["module_path"] == "quant_master.contrib.strategy"
    assert strategy_kwargs == {
        "signal": "<PRED>",
        "topk": 3,
        "risk_degree": 1.0,
        "trade_impact_limit": 0.25,
        "selection_rank_buffer": 1,
        "selection_max_new_names": 1,
    }
    assert soft["task"]["record"][-1]["kwargs"]["config"] == soft["port_analysis_config"]
