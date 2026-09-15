# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_master.contrib.model.double_ensemble import DEnsembleModel
from quant_master.contrib.model.gbdt import LGBModel
from quant_master.contrib.model.rank_aware import DailyRankICEval, prepare_tree_target
from quant_master.contrib.model.xgboost import XGBModel
from quant_master.data.dataset.handler import DataHandlerLP


def _frame(dates):
    instruments = ["a", "b", "c", "d", "e"]
    index = pd.MultiIndex.from_product([pd.to_datetime(dates), instruments], names=["datetime", "instrument"])
    cross_section = np.tile(np.arange(len(instruments), dtype=float), len(dates))
    date_offset = np.repeat(np.arange(len(dates), dtype=float) * 100.0, len(instruments))
    columns = pd.MultiIndex.from_tuples(
        [("feature", "rank_feature"), ("feature", "noise"), ("label", "LABEL0")]
    )
    values = np.column_stack([cross_section, np.sin(np.arange(len(index))), cross_section + date_offset])
    return pd.DataFrame(values, index=index, columns=columns)


class _Dataset:
    segments = {"train": ("2022-01-03", "2022-01-10"), "valid": ("2022-01-11", "2022-01-14")}

    def __init__(self):
        self.frames = {
            "train": _frame(pd.date_range("2022-01-03", periods=6)),
            "valid": _frame(pd.date_range("2022-01-11", periods=4)),
        }

    def prepare(self, segments, col_set="__all", data_key=DataHandlerLP.DK_L):
        if isinstance(segments, (list, tuple)):
            return [self.prepare(segment, col_set=col_set, data_key=data_key) for segment in segments]
        frame = self.frames[segments]
        if col_set == "feature":
            return frame["feature"]
        return frame


def test_cross_sectional_rank_target_removes_daily_level_and_scale():
    frame = _frame(["2022-01-03", "2022-01-04"])

    target = prepare_tree_target(frame["label"], "cs_rank")

    expected = np.tile([-0.5, -0.25, 0.0, 0.25, 0.5], 2)
    np.testing.assert_allclose(target.to_numpy(), expected)
    np.testing.assert_allclose(target.groupby(level="datetime").mean().to_numpy(), [0.0, 0.0])


def test_daily_rank_ic_eval_reports_perfect_cross_sectional_ordering():
    frame = _frame(["2022-01-03", "2022-01-04"])
    model = DEnsembleModel(num_models=1, target_mode="cs_rank")
    dtrain, _ = model._prepare_data_gbm(frame, frame, np.ones(len(frame)), frame["feature"].columns)
    dtrain.construct()
    evaluator = DailyRankICEval({id(dtrain): frame.index})

    name, value, higher_is_better = evaluator(frame[("feature", "rank_feature")].to_numpy(), dtrain)

    assert name == "rank_ic"
    assert np.isclose(value, 1.0)
    assert higher_is_better is True


def test_double_ensemble_rank_mode_transforms_lightgbm_labels_and_metric():
    frame = _frame(["2022-01-03", "2022-01-04"])
    model = DEnsembleModel(num_models=1, target_mode="rank_ic")

    dtrain, _ = model._prepare_data_gbm(frame, frame, np.ones(len(frame)), frame["feature"].columns)
    labels = dtrain.construct().get_label()

    np.testing.assert_allclose(labels, np.tile([-0.5, -0.25, 0.0, 0.25, 0.5], 2))
    assert model.target_mode == "cs_rank"
    assert model.rank_eval is True
    assert model.params["metric"] == "None"


def test_lgb_model_raw_mode_remains_backward_compatible():
    dataset = _Dataset()
    model = LGBModel()

    datasets = model._prepare_data(dataset)
    labels = datasets[0][0].construct().get_label()

    np.testing.assert_allclose(labels, dataset.frames["train"][("label", "LABEL0")].to_numpy())
    assert model.target_mode == "raw"
    assert model.rank_eval is False
    assert "metric" not in model.params


def test_double_ensemble_rank_mode_runs_small_lightgbm_fit():
    model = DEnsembleModel(
        num_models=1,
        epochs=4,
        early_stopping_rounds=2,
        target_mode="cs_rank",
        min_data_in_leaf=2,
        verbosity=-1,
        random_state=7,
    )

    model.fit(_Dataset())

    assert len(model.ensemble) == 1
    assert model.ensemble[0].num_trees() >= 1


def test_lgb_model_rank_mode_runs_small_lightgbm_fit():
    model = LGBModel(
        num_boost_round=4,
        early_stopping_rounds=2,
        target_mode="cs_rank",
        min_data_in_leaf=2,
        verbosity=-1,
    )

    with patch("quant_master.contrib.model.gbdt.R.log_metrics", create=True):
        model.fit(_Dataset(), verbose_eval=0)

    assert model.model is not None
    assert model.model.num_trees() >= 1


def test_xgb_model_rank_mode_runs_small_fit():
    model = XGBModel(
        target_mode="cs_rank",
        objective="reg:squarederror",
        tree_method="hist",
        device="cpu",
        max_depth=2,
        nthread=1,
    )

    model.fit(_Dataset(), num_boost_round=4, early_stopping_rounds=2, verbose_eval=False)

    assert model.model is not None
    assert model.target_mode == "cs_rank"
