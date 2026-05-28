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

    def _three_model(self, **kwargs):
        defaults = {
            "horizon_model_specs": [
                {
                    "name": "de_h1",
                    "model_type": "linear",
                    "horizon": 1,
                    "model_kwargs": {"estimator": "ridge", "alpha": 1.0},
                },
                {
                    "name": "de_h1_seed57",
                    "model_type": "linear",
                    "horizon": 1,
                    "model_kwargs": {"estimator": "ridge", "alpha": 1.0},
                },
                {
                    "name": "de_h1_seed7",
                    "model_type": "linear",
                    "horizon": 1,
                    "model_kwargs": {"estimator": "ridge", "alpha": 1.0},
                },
            ],
            "topk": 1,
            "search_step": 0.1,
            "turnover_penalty": 0.0,
            "risk_penalty": 0.0,
            "use_rank_score": False,
            "neutralize_daily_mean": False,
            "enforce_horizon_monotonic": False,
            "zscore_clip": 100.0,
        }
        defaults.update(kwargs)
        return RegimeHorizonCostEnsembleModel(**defaults)

    def _weight_learning_data(self):
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2022-01-03", "2022-01-04"]), ["a", "b", "c"]],
            names=["datetime", "instrument"],
        )
        pred_frame = pd.DataFrame(
            {
                "de_h1": [0.3, 0.2, 0.1, 0.3, 0.2, 0.1],
                "de_h1_seed57": [0.1, 1.0, 0.0, 0.1, 1.0, 0.0],
                "de_h1_seed7": [0.0, 0.1, 1.0, 0.0, 0.1, 1.0],
            },
            index=index,
        )
        label = pd.Series([0.0, 1.0, -1.0, 0.0, 1.0, -1.0], index=index)
        return pred_frame, label

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

    def test_weight_learning_without_constraints_does_not_apply_anchor_limits(self):
        model = self._three_model()
        pred_frame, label = self._weight_learning_data()

        weights = model._learn_weights(pred_frame, label)

        self.assertEqual(model.weight_constraints, {})
        self.assertLess(weights["de_h1"], 0.8)
        self.assertGreater(weights["de_h1_seed57"] + weights["de_h1_seed7"], 0.2)

    def test_anchor_min_weight_constraint_is_enforced(self):
        model = self._three_model(
            weight_constraints={"anchor_model": "de_h1", "min_anchor_weight": 0.8}
        )
        pred_frame, label = self._weight_learning_data()

        weights = model._learn_weights(pred_frame, label)

        self.assertGreaterEqual(weights["de_h1"], 0.8 - 1e-12)
        self.assertTrue(model._check_weight_constraints(weights))

    def test_max_aux_weight_constraint_is_enforced(self):
        model = self._three_model(
            weight_constraints={"anchor_model": "de_h1", "max_aux_weight": 0.2}
        )
        pred_frame, label = self._weight_learning_data()

        weights = model._learn_weights(pred_frame, label)

        aux_weight = weights["de_h1_seed57"] + weights["de_h1_seed7"]
        self.assertLessEqual(aux_weight, 0.2 + 1e-12)
        self.assertTrue(model._check_weight_constraints(weights))

    def test_model_max_weights_constraint_is_enforced(self):
        model = self._three_model(
            weight_constraints={"model_max_weights": {"de_h1_seed57": 0.2}}
        )
        pred_frame, label = self._weight_learning_data()

        weights = model._learn_weights(pred_frame, label)

        self.assertLessEqual(weights["de_h1_seed57"], 0.2 + 1e-12)
        self.assertTrue(model._check_weight_constraints(weights))

    def test_invalid_anchor_model_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "anchor_model 'missing' is not in model_specs"):
            self._three_model(weight_constraints={"anchor_model": "missing", "min_anchor_weight": 0.8})

    def test_infeasible_weight_constraints_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "weight_constraints are infeasible"):
            self._three_model(
                weight_constraints={
                    "anchor_model": "de_h1",
                    "min_anchor_weight": 0.8,
                    "model_max_weights": {"de_h1": 0.7},
                }
            )

    def test_regime_low_sample_fallback_uses_current_global_weights(self):
        model = self._three_model(min_regime_samples=100)
        pred_frame, label = self._weight_learning_data()
        regimes = pd.Series([0] * len(pred_frame), index=pred_frame.index)
        global_weights = {"de_h1": 0.8, "de_h1_seed57": 0.1, "de_h1_seed7": 0.1}

        regime_weights = model._learn_regime_weights(pred_frame, label, regimes, global_weights)

        self.assertEqual(regime_weights[0], global_weights)


if __name__ == "__main__":
    unittest.main()
