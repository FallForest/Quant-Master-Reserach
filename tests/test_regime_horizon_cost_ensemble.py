# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from quant_master.contrib.model.regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel


class TestRegimeHorizonCostEnsembleControls(unittest.TestCase):
    def _model(self, **kwargs):
        defaults = {
            "horizon_model_specs": [
                {
                    "name": "lin_h1",
                    "model_type": "linear",
                    "horizon": 1,
                    "model_kwargs": {"estimator": "ridge", "alpha": 1.0},
                }
            ]
        }
        defaults.update(kwargs)
        return RegimeHorizonCostEnsembleModel(**defaults)

    def test_default_final_score_controls_are_identity(self):
        model = self._model()
        score = pd.Series([0.3, -0.2, 1.1], index=pd.Index(["a", "b", "c"]))

        controlled = model._apply_final_score_controls(score)

        pd.testing.assert_series_equal(controlled, score.astype(float))

    def test_rank_blend_preserves_cross_sectional_order(self):
        model = self._model(robust_rank_blend=1.0)
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2022-01-03", "2022-01-04"]), ["a", "b", "c"]],
            names=["datetime", "instrument"],
        )
        score = pd.Series([0.1, 0.3, -0.2, 1.0, -0.5, 0.2], index=index)

        controlled = model._apply_final_score_controls(score)

        for dt, daily_score in score.groupby(level="datetime"):
            daily_controlled = controlled.loc[dt]
            self.assertEqual(
                list(daily_controlled.sort_values().index),
                list(daily_score.droplevel("datetime").sort_values().index),
            )

    def test_final_score_control_grid_validation(self):
        with self.assertRaises(ValueError):
            self._model(robust_rank_blend_grid=[0.0, 1.1])
        with self.assertRaises(ValueError):
            self._model(prediction_shrinkage_grid=[np.nan])

    def test_default_final_score_controls_are_not_grid_optimized(self):
        model = self._model(topk=1, memory_boost_grid=[0.0])
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2022-01-03", "2022-01-04"]), ["a", "b", "c"]],
            names=["datetime", "instrument"],
        )
        score = pd.Series([0.1, 0.3, -0.2, 1.0, -0.5, 0.2], index=index)
        label = pd.Series([0.0, 1.0, -1.0, 1.0, -1.0, 0.0], index=index)

        selected = model._select_final_score_control(score, label)

        self.assertEqual(list(model._iter_final_score_controls()), [(0.0, 1.0)])
        self.assertEqual(selected["robust_rank_blend"], 0.0)
        self.assertEqual(selected["prediction_shrinkage"], 1.0)

    def test_final_score_grid_falls_back_when_rank_ic_deteriorates(self):
        model = self._model(
            robust_rank_blend_grid=[0.0],
            prediction_shrinkage_grid=[1.0],
            topk=1,
            memory_boost_grid=[0.0],
        )
        baseline = {
            "objective": 0.10,
            "rank_ic": 0.20,
            "memory_boost": 0.0,
            "robust_rank_blend": 0.0,
            "prediction_shrinkage": 1.0,
            "is_identity_control": True,
        }
        candidate = {
            "objective": 0.11,
            "rank_ic": 0.19,
            "memory_boost": 0.0,
            "robust_rank_blend": 0.3,
            "prediction_shrinkage": 1.0,
            "is_identity_control": False,
        }

        self.assertFalse(model._final_score_control_beats(candidate, baseline, baseline))

    def test_final_score_grid_tie_break_prefers_identity(self):
        model = self._model(
            robust_rank_blend_grid=[0.2, 0.0],
            prediction_shrinkage_grid=[1.0],
            topk=1,
            memory_boost_grid=[0.0],
        )
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2022-01-03", "2022-01-04"]), ["a", "b", "c"]],
            names=["datetime", "instrument"],
        )
        score = pd.Series([0.1, 0.3, -0.2, 1.0, -0.5, 0.2], index=index)
        label = pd.Series([0.0, 1.0, -1.0, 1.0, -1.0, 0.0], index=index)

        selected = model._select_final_score_control(score, label)

        self.assertTrue(selected["is_identity_control"])
        self.assertEqual(selected["robust_rank_blend"], 0.0)
        self.assertEqual(selected["prediction_shrinkage"], 1.0)

    def test_final_score_grid_appends_identity_fallback(self):
        model = self._model(robust_rank_blend_grid=[0.25], prediction_shrinkage_grid=[0.8])

        controls = list(model._iter_final_score_controls())

        self.assertIn((0.25, 0.8), controls)
        self.assertIn((0.0, 1.0), controls)


if __name__ == "__main__":
    unittest.main()
