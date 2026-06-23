from __future__ import annotations

from pathlib import Path

import yaml

from examples.benchmarks.Transcendence.model.alpha158alpha360_regime_horizon_run import (
    _apply_common_overrides,
    _load_config as _load_runner_config,
)
from quant_master.contrib.model.regime_finite_dnn_residual import RegimeFiniteDNNResidualModel
from quant_master.utils import init_instance_by_config


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Transcendence"
    / "configs"
    / "workflows"
    / "transcendence"
    / "workflow_config_transcendence_cpu_gpu_residual_finite_dnn_smallfund_Alpha158_2026_csi300.yaml"
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _model_config(candidate: dict) -> dict:
    return candidate["task"]["model"]


def test_residual_finite_dnn_keeps_smallfund_execution_shape():
    candidate = _load_config(CANDIDATE_CONFIG)
    backtest_config = candidate["port_analysis_config"]["backtest"]
    exchange_kwargs = backtest_config["exchange_kwargs"]
    strategy_kwargs = candidate["port_analysis_config"]["strategy"]["kwargs"]

    assert backtest_config["account"] == 13000
    assert backtest_config["benchmark"] == "SH000300"
    assert exchange_kwargs["open_cost"] == 0.0001
    assert exchange_kwargs["close_cost"] == 0.0006
    assert exchange_kwargs["min_cost"] == 0
    assert strategy_kwargs["topk"] == 3
    assert strategy_kwargs["n_drop"] == 1


def test_residual_finite_dnn_model_builds_without_data():
    candidate = _load_config(CANDIDATE_CONFIG)
    model_kwargs = _model_config(candidate)["kwargs"]

    model = init_instance_by_config(_model_config(candidate))

    assert isinstance(model, RegimeFiniteDNNResidualModel)
    assert model.topk == 3
    assert model.n_drop == 1
    assert model.residual_weight_grid == [0.0, 0.03, 0.05, 0.08, 0.1, 0.15]
    assert model.residual_model.device.type == "cpu"
    assert model.residual_model.input_dim == 158
    assert model.residual_model.label_rank is False
    assert model.residual_clip == 0.2
    assert model.restrict_residual_to_anchor_pool is True
    assert model_kwargs["anchor_kwargs"]["weight_constraints"]["anchor_model"] == "de_h1"


def test_residual_rank_combine_only_adjusts_anchor_candidate_pool_and_clips_aux():
    import pandas as pd

    model = RegimeFiniteDNNResidualModel(
        anchor_kwargs={"horizon_model_specs": [{"name": "dummy", "model_type": "linear", "horizon": 1}]},
        topk=1,
        candidate_pool=2,
        residual_weight_grid=[0.0],
        residual_clip=0.1,
    )
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-02")], ["a", "b", "c", "d"]], names=["datetime", "instrument"]
    )
    anchor = pd.Series([4.0, 3.0, 2.0, 1.0], index=index)
    residual = pd.Series([0.3, -0.3, 0.3, -0.3], index=index)

    combined = model._combine(anchor, residual, weight=1.0)
    anchor_rank = model._cross_sectional_rank_score(anchor)

    assert combined.loc[(pd.Timestamp("2024-01-02"), "a")] == anchor_rank.loc[(pd.Timestamp("2024-01-02"), "a")] + 0.1
    assert combined.loc[(pd.Timestamp("2024-01-02"), "b")] == anchor_rank.loc[(pd.Timestamp("2024-01-02"), "b")] - 0.1
    assert combined.loc[(pd.Timestamp("2024-01-02"), "c")] == anchor_rank.loc[(pd.Timestamp("2024-01-02"), "c")]
    assert combined.loc[(pd.Timestamp("2024-01-02"), "d")] == anchor_rank.loc[(pd.Timestamp("2024-01-02"), "d")]


def test_residual_rank_transform_clips_cross_sectional_aux_signal():
    import pandas as pd

    model = RegimeFiniteDNNResidualModel(
        anchor_kwargs={"horizon_model_specs": [{"name": "dummy", "model_type": "linear", "horizon": 1}]},
        residual_weight_grid=[0.0],
        residual_clip=0.1,
    )
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-02")], ["a", "b", "c", "d"]], names=["datetime", "instrument"]
    )
    pred = pd.Series([10.0, 5.0, -5.0, -10.0], index=index)

    transformed = model._transform_aux_prediction(pred)

    assert transformed.max() == 0.1
    assert transformed.min() == -0.1


def test_residual_finite_dnn_smoke_overrides_budget_nested_anchor_only():
    candidate = _load_runner_config(CANDIDATE_CONFIG)
    run_config = _apply_common_overrides(candidate, mode="smoke", preserve_config_windows=True)
    model_kwargs = _model_config(run_config)["kwargs"]
    anchor_kwargs = model_kwargs["anchor_kwargs"]

    assert "horizon_model_specs" not in model_kwargs
    assert anchor_kwargs["min_regime_samples"] == 120
    assert anchor_kwargs["robust_rank_blend_grid"] == [0.0]
    assert anchor_kwargs["prediction_shrinkage_grid"] == [1.0]
    assert anchor_kwargs["horizon_model_specs"][0]["model_kwargs"]["num_models"] == 1
    assert anchor_kwargs["horizon_model_specs"][0]["model_kwargs"]["epochs"] == 4
    assert anchor_kwargs["horizon_model_specs"][1]["model_kwargs"]["num_boost_round"] == 80
    assert run_config["port_analysis_config"]["backtest"]["account"] == 13000
    assert run_config["port_analysis_config"]["backtest"]["exchange_kwargs"]["open_cost"] == 0.0001
    assert run_config["port_analysis_config"]["backtest"]["exchange_kwargs"]["close_cost"] == 0.0006
    assert run_config["port_analysis_config"]["backtest"]["exchange_kwargs"]["min_cost"] == 0

    model = init_instance_by_config(_model_config(run_config))
    assert isinstance(model, RegimeFiniteDNNResidualModel)
