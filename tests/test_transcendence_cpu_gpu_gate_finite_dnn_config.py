from __future__ import annotations

from pathlib import Path

import pandas as pd
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
    / "workflow_config_transcendence_cpu_gpu_gate_finite_dnn_smallfund_Alpha158_2026_csi300.yaml"
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _model_config(candidate: dict) -> dict:
    return candidate["task"]["model"]


def test_gate_finite_dnn_keeps_smallfund_execution_shape_and_builds():
    candidate = _load_config(CANDIDATE_CONFIG)
    backtest_config = candidate["port_analysis_config"]["backtest"]
    exchange_kwargs = backtest_config["exchange_kwargs"]
    model = init_instance_by_config(_model_config(candidate))

    assert backtest_config["account"] == 13000
    assert backtest_config["benchmark"] == "SH000300"
    assert exchange_kwargs["open_cost"] == 0.0001
    assert exchange_kwargs["close_cost"] == 0.0006
    assert exchange_kwargs["min_cost"] == 0
    assert isinstance(model, RegimeFiniteDNNResidualModel)
    assert model.target_mode == "topk_gate"
    assert model.candidate_pool == 12
    assert model.residual_weight_grid == [0.0, 0.01, 0.02, 0.03, 0.05]
    assert model.residual_model.device.type == "cpu"
    assert model.residual_clip == 0.2
    assert model.restrict_residual_to_anchor_pool is True


def test_gate_target_only_marks_anchor_candidate_pool():
    model = RegimeFiniteDNNResidualModel(
        anchor_kwargs={"horizon_model_specs": [{"name": "dummy", "model_type": "linear", "horizon": 1}]},
        target_mode="topk_gate",
        topk=1,
        candidate_pool=2,
        residual_weight_grid=[0.0],
    )
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-02")], ["a", "b", "c", "d"]], names=["datetime", "instrument"]
    )
    label = pd.Series([0.4, 0.3, 0.9, 0.1], index=index)
    anchor = pd.Series([4.0, 3.0, 2.0, 1.0], index=index)

    target = model._build_topk_gate_target(label, anchor)

    assert target.loc[(pd.Timestamp("2024-01-02"), "a")] == 1.0
    assert target.loc[(pd.Timestamp("2024-01-02"), "b")] == 0.0
    assert target.loc[(pd.Timestamp("2024-01-02"), "c")] == 0.0
    assert target.loc[(pd.Timestamp("2024-01-02"), "d")] == 0.0


def test_gate_smoke_overrides_budget_nested_anchor_only():
    candidate = _load_runner_config(CANDIDATE_CONFIG)
    run_config = _apply_common_overrides(candidate, mode="smoke", preserve_config_windows=True)
    model_kwargs = _model_config(run_config)["kwargs"]
    anchor_kwargs = model_kwargs["anchor_kwargs"]

    assert "horizon_model_specs" not in model_kwargs
    assert anchor_kwargs["min_regime_samples"] == 120
    assert anchor_kwargs["horizon_model_specs"][0]["model_kwargs"]["num_models"] == 1
    assert anchor_kwargs["horizon_model_specs"][1]["model_kwargs"]["num_boost_round"] == 80
    assert model_kwargs["target_mode"] == "topk_gate"
    assert model_kwargs["candidate_pool"] == 12

    model = init_instance_by_config(_model_config(run_config))
    assert isinstance(model, RegimeFiniteDNNResidualModel)
