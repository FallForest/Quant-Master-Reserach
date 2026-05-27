# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Text, Union

import numpy as np
import pandas as pd

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from ..strategy.topk_cost_aware import transform_scores_for_cost
from .double_ensemble import DEnsembleModel
from .gbdt import LGBModel
from .linear import LinearModel
from .transcendence_objective import build_objective_label_frame


@dataclass
class _ModelSpec:
    name: str
    model_type: str
    horizon: int
    model_kwargs: Dict


class RegimeHorizonCostEnsembleModel(Model):
    """Regime-aware, multi-horizon, cost-aware ensemble.

    Design goals:
    1) Multi-horizon label alignment: each base learner is trained on a horizon-smoothed
       label built from train/valid only.
    2) Regime-aware blending: validation-period prediction structure defines regimes.
    3) Cost-aware objective: validation weight search targets top-k return minus turnover
       and return-volatility penalties.
    4) Monotonic/risk controls: enforce horizon monotonic weights and score clipping.
    """

    _FINAL_CONTROL_EPS = 1e-12

    def __init__(
        self,
        horizon_model_specs: Optional[Sequence[Dict]] = None,
        double_ensemble_kwargs: Optional[Dict] = None,
        lightgbm_kwargs: Optional[Dict] = None,
        linear_kwargs: Optional[Dict] = None,
        horizon_days: Optional[Sequence[int]] = None,
        topk: int = 50,
        search_step: float = 0.1,
        turnover_penalty: Optional[float] = None,
        turnover_penalty_grid: Optional[Sequence[float]] = None,
        risk_penalty: Optional[float] = None,
        risk_penalty_grid: Optional[Sequence[float]] = None,
        cost_weight_grid: Optional[Sequence[float]] = None,
        memory_boost_grid: Optional[Sequence[float]] = None,
        use_rank_score: bool = True,
        zscore_clip: float = 3.0,
        neutralize_daily_mean: bool = True,
        robust_rank_blend: Optional[float] = None,
        robust_rank_blend_grid: Optional[Sequence[float]] = None,
        prediction_shrinkage: Optional[float] = None,
        prediction_shrinkage_grid: Optional[Sequence[float]] = None,
        enforce_horizon_monotonic: bool = True,
        monotonic_direction: str = "decreasing",
        regime_consensus_quantiles: Optional[Sequence[float]] = None,
        regime_disagreement_quantiles: Optional[Sequence[float]] = None,
        min_regime_samples: int = 600,
        random_state: int = 42,
        primary_feature_set: Optional[str] = None,
        secondary_feature_set: Optional[str] = None,
        secondary_handler: Optional[Dict] = None,
        feature_blend_mode: Optional[str] = None,
        feature_weight_grid: Optional[Sequence[Sequence[float]]] = None,
        horizon_weight_decay_grid: Optional[Sequence[float]] = None,
        regime_lookback_windows: Optional[Sequence[int]] = None,
        regime_feature_windows: Optional[Sequence[int]] = None,
        regime_count_grid: Optional[Sequence[int]] = None,
        rolling_train_years: Optional[Sequence[int]] = None,
        rolling_valid_months: Optional[Sequence[int]] = None,
        n_drop: Optional[int] = None,
        slippage_bps_grid: Optional[Sequence[float]] = None,
        early_prune_rounds: Optional[int] = None,
        cv_folds: Optional[int] = None,
        transformer_kwargs: Optional[Dict] = None,
        num_threads: Optional[int] = None,
        quick_smoke: Optional[bool] = None,
        objective_label_mode: str = "raw",
        objective_horizon_days: Optional[Sequence[int]] = None,
        objective_horizon_weights: Optional[Sequence[float]] = None,
        objective_market_relative: bool = False,
        objective_vol_adjust: bool = False,
        objective_vol_window: int = 20,
        objective_vol_floor: float = 1e-4,
        objective_rank_power: float = 1.0,
        objective_decile: float = 0.1,
        objective_decile_scale: float = 0.0,
        objective_clip: Optional[float] = None,
        **kwargs,
    ):
        if topk <= 0:
            raise ValueError("topk must be positive.")
        if not 0 < search_step <= 1:
            raise ValueError("search_step must be in (0, 1].")
        if zscore_clip <= 0:
            raise ValueError("zscore_clip must be positive.")
        if monotonic_direction not in {"decreasing", "increasing"}:
            raise ValueError("monotonic_direction must be 'decreasing' or 'increasing'.")

        self.logger = get_module_logger("RegimeHorizonCostEnsembleModel")
        self.topk = topk
        self.search_step = search_step
        self.turnover_penalty_grid = self._normalize_float_grid(turnover_penalty_grid)
        self.risk_penalty_grid = self._normalize_float_grid(risk_penalty_grid)
        self.cost_weight_grid = self._normalize_float_grid(cost_weight_grid)
        self.turnover_penalty = (
            float(turnover_penalty)
            if turnover_penalty is not None
            else (self.turnover_penalty_grid[0] if self.turnover_penalty_grid else 0.0002)
        )
        self.risk_penalty = (
            float(risk_penalty)
            if risk_penalty is not None
            else (
                self.risk_penalty_grid[0]
                if self.risk_penalty_grid
                else (self.cost_weight_grid[0] if self.cost_weight_grid else 0.05)
            )
        )
        self.memory_boost_grid = list(memory_boost_grid or [0.0, 0.01, 0.02, 0.03])
        self.use_rank_score = use_rank_score
        self.zscore_clip = zscore_clip
        self.neutralize_daily_mean = neutralize_daily_mean
        self._final_score_control_grid_opt_in = (
            robust_rank_blend is not None
            or robust_rank_blend_grid is not None
            or prediction_shrinkage is not None
            or prediction_shrinkage_grid is not None
        )
        rank_blend_values = robust_rank_blend_grid
        if rank_blend_values is None:
            rank_blend_values = [0.0 if robust_rank_blend is None else robust_rank_blend]
        self.robust_rank_blend_grid = self._normalize_bounded_float_grid(
            rank_blend_values, "robust_rank_blend_grid", lower=0.0, upper=1.0
        )
        shrinkage_values = prediction_shrinkage_grid
        if shrinkage_values is None:
            shrinkage_values = [1.0 if prediction_shrinkage is None else prediction_shrinkage]
        self.prediction_shrinkage_grid = self._normalize_bounded_float_grid(
            shrinkage_values, "prediction_shrinkage_grid", lower=0.0, upper=1.0
        )
        self.robust_rank_blend = self.robust_rank_blend_grid[0]
        self.prediction_shrinkage = self.prediction_shrinkage_grid[0]
        self.enforce_horizon_monotonic = enforce_horizon_monotonic
        self.monotonic_direction = monotonic_direction
        if regime_count_grid and (regime_consensus_quantiles is None and regime_disagreement_quantiles is None):
            regime_count = max(int(regime_count_grid[0]), 1)
            self.regime_consensus_quantiles = self._uniform_quantiles(regime_count)
            self.regime_disagreement_quantiles = []
        else:
            self.regime_consensus_quantiles = list(regime_consensus_quantiles or [0.33, 0.67])
            self.regime_disagreement_quantiles = list(regime_disagreement_quantiles or [0.5])
        self.min_regime_samples = min_regime_samples
        self.random_state = random_state
        self.objective_label_mode = str(objective_label_mode or "raw").strip().lower()
        self.objective_horizon_days = list(objective_horizon_days) if objective_horizon_days else None
        self.objective_horizon_weights = list(objective_horizon_weights) if objective_horizon_weights else None
        self.objective_market_relative = bool(objective_market_relative)
        self.objective_vol_adjust = bool(objective_vol_adjust)
        self.objective_vol_window = int(objective_vol_window)
        self.objective_vol_floor = float(objective_vol_floor)
        self.objective_rank_power = float(objective_rank_power)
        self.objective_decile = float(objective_decile)
        self.objective_decile_scale = float(objective_decile_scale)
        self.objective_clip = None if objective_clip is None else float(objective_clip)
        self.objective_enabled = (
            self.objective_label_mode != "raw"
            or self.objective_horizon_days is not None
            or self.objective_market_relative
            or self.objective_vol_adjust
            or abs(self.objective_rank_power - 1.0) > 1e-12
            or self.objective_decile_scale > 0
            or self.objective_clip is not None
        )

        self.model_specs = self._build_model_specs(
            self._build_specs_from_compat_config(
                horizon_model_specs=horizon_model_specs,
                horizon_days=horizon_days,
                double_ensemble_kwargs=double_ensemble_kwargs,
                lightgbm_kwargs=lightgbm_kwargs,
                linear_kwargs=linear_kwargs,
            )
        )
        self.models: Dict[str, Model] = {}
        self.global_weights: Dict[str, float] = {}
        self.regime_weights: Dict[int, Dict[str, float]] = {}
        self.regime_consensus_thresholds: List[float] = []
        self.regime_disagreement_thresholds: List[float] = []
        self.num_regimes: int = 1
        self.memory_boost: float = 0.0
        self.fitted = False
        self.unused_config_keys: List[str] = []

        reserved_but_unused = {
            "primary_feature_set": primary_feature_set,
            "secondary_feature_set": secondary_feature_set,
            "secondary_handler": secondary_handler,
            "feature_blend_mode": feature_blend_mode,
            "feature_weight_grid": feature_weight_grid,
            "horizon_weight_decay_grid": horizon_weight_decay_grid,
            "regime_lookback_windows": regime_lookback_windows,
            "regime_feature_windows": regime_feature_windows,
            "rolling_train_years": rolling_train_years,
            "rolling_valid_months": rolling_valid_months,
            "n_drop": n_drop,
            "slippage_bps_grid": slippage_bps_grid,
            "early_prune_rounds": early_prune_rounds,
            "cv_folds": cv_folds,
            "transformer_kwargs": transformer_kwargs,
            "num_threads": num_threads,
            "quick_smoke": quick_smoke,
        }
        self.unused_config_keys.extend([k for k, v in reserved_but_unused.items() if v is not None])
        self.unused_config_keys.extend(sorted(kwargs.keys()))
        self.unused_config_keys = sorted(set(self.unused_config_keys))
        if self.unused_config_keys:
            self.logger.warning(
                "Unused model config keys (accepted for workflow compatibility, not consumed yet): %s",
                self.unused_config_keys,
            )

    def fit(self, dataset: DatasetH):
        self.models = {}
        for spec in self.model_specs:
            self.logger.info("Training %s with horizon=%s", spec.name, spec.horizon)
            model = self._build_model(spec)
            horizon_dataset = _HorizonLabelDataset(
                dataset,
                horizon=spec.horizon,
                objective_kwargs=self._objective_kwargs() if self.objective_enabled else None,
            )
            model.fit(horizon_dataset)
            self.models[spec.name] = model

        valid_pred_raw = self._predict_frame(dataset, "valid")
        valid_label_df = dataset.prepare("valid", col_set=["label"], data_key=DataHandlerLP.DK_L)["label"]
        if self.objective_enabled:
            valid_label_df = build_objective_label_frame(
                _HorizonLabelDataset._to_label_frame(valid_label_df),
                base_horizon=self._objective_valid_horizon(),
                **self._objective_kwargs(),
            )
        valid_pred_raw, valid_label_df = self._align_prediction_and_label(valid_pred_raw, valid_label_df)
        valid_label = pd.Series(self._squeeze_label(valid_label_df), index=valid_label_df.index)
        valid_pred = self._prepare_prediction_scores(valid_pred_raw)

        self._fit_regime_thresholds(valid_pred_raw)
        regimes = self._assign_row_regimes(valid_pred_raw)
        self._fit_penalty_and_weights(valid_pred, valid_label, regimes)

        self.logger.info("Global weights: %s", self.global_weights)
        self.logger.info("Regime weights: %s", self.regime_weights)
        self.logger.info("Regime thresholds (consensus): %s", self.regime_consensus_thresholds)
        self.logger.info("Regime thresholds (disagreement): %s", self.regime_disagreement_thresholds)
        self.logger.info(
            "Selected penalties: turnover_penalty=%s, risk_penalty=%s",
            self.turnover_penalty,
            self.risk_penalty,
        )
        self.logger.info(
            "Final score controls: robust_rank_blend=%s, prediction_shrinkage=%s",
            self.robust_rank_blend,
            self.prediction_shrinkage,
        )
        self.logger.info("Memory boost: %s", self.memory_boost)
        self.fitted = True

    def _fit_penalty_and_weights(self, valid_pred: pd.DataFrame, valid_label: pd.Series, regimes: pd.Series):
        best_state = None
        for turnover_penalty, risk_penalty in self._iter_penalty_pairs():
            self.turnover_penalty = turnover_penalty
            self.risk_penalty = risk_penalty
            global_weights = self._learn_weights(valid_pred, valid_label)
            regime_weights = self._learn_regime_weights(valid_pred, valid_label, regimes)
            blended = self._blend_by_regime_with_weights(valid_pred, regimes, regime_weights, global_weights)
            blended = self._apply_risk_controls(blended)
            selected_control = self._select_final_score_control(blended, valid_label)

            if self._is_better_penalty_state(selected_control, best_state):
                best_state = {
                    "objective": float(selected_control["objective"]),
                    "rank_ic": float(selected_control["rank_ic"]),
                    "is_identity_control": bool(selected_control["is_identity_control"]),
                    "turnover_penalty": float(turnover_penalty),
                    "risk_penalty": float(risk_penalty),
                    "global_weights": global_weights,
                    "regime_weights": regime_weights,
                    "memory_boost": float(selected_control["memory_boost"]),
                    "robust_rank_blend": float(selected_control["robust_rank_blend"]),
                    "prediction_shrinkage": float(selected_control["prediction_shrinkage"]),
                }

        if best_state is None:
            raise ValueError("Failed to fit penalty/weight state from validation data.")
        self.turnover_penalty = best_state["turnover_penalty"]
        self.risk_penalty = best_state["risk_penalty"]
        self.global_weights = best_state["global_weights"]
        self.regime_weights = best_state["regime_weights"]
        self.memory_boost = best_state["memory_boost"]
        self.robust_rank_blend = best_state["robust_rank_blend"]
        self.prediction_shrinkage = best_state["prediction_shrinkage"]

    def _objective_valid_horizon(self) -> int:
        if not self.model_specs:
            return 1
        return max(int(spec.horizon) for spec in self.model_specs)

    def _objective_kwargs(self) -> Dict:
        return {
            "mode": self.objective_label_mode,
            "horizon_days": self.objective_horizon_days,
            "horizon_weights": self.objective_horizon_weights,
            "market_relative": self.objective_market_relative,
            "vol_adjust": self.objective_vol_adjust,
            "vol_window": self.objective_vol_window,
            "vol_floor": self.objective_vol_floor,
            "rank_power": self.objective_rank_power,
            "decile": self.objective_decile,
            "decile_scale": self.objective_decile_scale,
            "clip": self.objective_clip,
        }

    def _iter_penalty_pairs(self):
        turnover_grid = self.turnover_penalty_grid or [self.turnover_penalty]
        risk_grid = self.risk_penalty_grid or [self.risk_penalty]
        for turnover_penalty in turnover_grid:
            for risk_penalty in risk_grid:
                yield float(turnover_penalty), float(risk_penalty)

    def _select_final_score_control(self, score: pd.Series, label: pd.Series) -> Dict[str, float]:
        baseline = self._evaluate_final_score_control(
            score,
            label,
            robust_rank_blend=0.0,
            prediction_shrinkage=1.0,
        )
        best = dict(baseline)
        for rank_blend, prediction_shrinkage in self._iter_final_score_controls():
            candidate = self._evaluate_final_score_control(
                score,
                label,
                robust_rank_blend=rank_blend,
                prediction_shrinkage=prediction_shrinkage,
            )
            if self._final_score_control_beats(candidate, best, baseline):
                best = candidate
        return best

    def _evaluate_final_score_control(
        self,
        score: pd.Series,
        label: pd.Series,
        robust_rank_blend: float,
        prediction_shrinkage: float,
    ) -> Dict[str, float]:
        controlled = self._apply_final_score_controls(
            score,
            robust_rank_blend=robust_rank_blend,
            prediction_shrinkage=prediction_shrinkage,
        )
        memory_boost = self._learn_memory_boost(controlled, label)
        adjusted = self._apply_turnover_boost(controlled, memory_boost) if memory_boost > 0 else controlled
        objective = self._topk_cost_objective(adjusted, label)
        return {
            "objective": float(objective),
            "rank_ic": float(self._daily_rank_ic(controlled, label)),
            "memory_boost": float(memory_boost),
            "robust_rank_blend": float(robust_rank_blend),
            "prediction_shrinkage": float(prediction_shrinkage),
            "is_identity_control": self._is_identity_final_score_control(
                robust_rank_blend,
                prediction_shrinkage,
            ),
        }

    def _final_score_control_beats(self, candidate: Dict[str, float], current: Dict[str, float], baseline: Dict[str, float]) -> bool:
        if not self._final_control_passes_baseline_floor(candidate, baseline):
            return False
        if self._final_control_objective_margin(candidate, baseline) <= self._FINAL_CONTROL_EPS:
            return False

        objective_delta = candidate["objective"] - current["objective"]
        if objective_delta > self._FINAL_CONTROL_EPS:
            return True
        if abs(objective_delta) > self._FINAL_CONTROL_EPS:
            return False

        rank_ic_delta = candidate["rank_ic"] - current["rank_ic"]
        if rank_ic_delta > self._FINAL_CONTROL_EPS:
            return True
        if abs(rank_ic_delta) > self._FINAL_CONTROL_EPS:
            return False
        return self._final_control_tiebreak_key(candidate) > self._final_control_tiebreak_key(current)

    def _final_control_passes_baseline_floor(self, candidate: Dict[str, float], baseline: Dict[str, float]) -> bool:
        if not np.isfinite(candidate["objective"]):
            return False
        if not self._final_control_grid_has_identity():
            return True
        if candidate["is_identity_control"]:
            return True
        return candidate["rank_ic"] >= baseline["rank_ic"] - self._FINAL_CONTROL_EPS

    def _final_control_objective_margin(self, candidate: Dict[str, float], baseline: Dict[str, float]) -> float:
        if candidate["is_identity_control"]:
            return candidate["objective"] - baseline["objective"]
        if not self._final_score_control_grid_opt_in:
            return 0.0
        return candidate["objective"] - baseline["objective"]

    def _is_better_penalty_state(self, candidate: Dict[str, float], current: Optional[Dict[str, float]]) -> bool:
        if current is None:
            return True
        objective_delta = candidate["objective"] - current["objective"]
        if objective_delta > self._FINAL_CONTROL_EPS:
            return True
        if abs(objective_delta) > self._FINAL_CONTROL_EPS:
            return False

        identity_delta = int(candidate["is_identity_control"]) - int(current.get("is_identity_control", False))
        if identity_delta != 0:
            return identity_delta > 0

        rank_ic_delta = candidate["rank_ic"] - current.get("rank_ic", float("-inf"))
        if rank_ic_delta > self._FINAL_CONTROL_EPS:
            return True
        if abs(rank_ic_delta) > self._FINAL_CONTROL_EPS:
            return False
        return self._final_control_tiebreak_key(candidate) > self._final_control_tiebreak_key(current)

    @classmethod
    def _final_control_tiebreak_key(cls, state: Dict[str, float]):
        return (
            1 if state["is_identity_control"] else 0,
            -abs(float(state["robust_rank_blend"])),
            -abs(float(state["prediction_shrinkage"]) - 1.0),
        )

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        pred_raw = self._predict_frame(dataset, segment)
        pred_scores = self._prepare_prediction_scores(pred_raw)
        regimes = self._assign_row_regimes(pred_raw)
        blended = self._blend_by_regime(pred_scores, regimes)
        blended = self._apply_risk_controls(blended)
        blended = self._apply_final_score_controls(blended)
        return self._apply_cost_aware_transform(
            blended,
            memory_boost=self.memory_boost,
            turnover_penalty=self.turnover_penalty,
            volatility_penalty=self.risk_penalty,
        )

    def _build_model_specs(self, specs: Optional[Sequence[Dict]]) -> List[_ModelSpec]:
        default_specs = [
            {
                "name": "de_h1",
                "model_type": "double_ensemble",
                "horizon": 1,
                "model_kwargs": {"random_state": self.random_state},
            },
            {
                "name": "lgb_h5",
                "model_type": "lightgbm",
                "horizon": 5,
                "model_kwargs": {
                    "seed": self.random_state,
                    "feature_fraction_seed": self.random_state,
                    "bagging_seed": self.random_state,
                    "data_random_seed": self.random_state,
                },
            },
            {
                "name": "lin_h10",
                "model_type": "linear",
                "horizon": 10,
                "model_kwargs": {"estimator": "ridge", "alpha": 1.0},
            },
        ]
        specs = list(default_specs if specs is None else specs)
        parsed_specs = []
        for i, raw in enumerate(specs):
            horizon = int(raw.get("horizon", 1))
            if horizon <= 0:
                raise ValueError("horizon must be positive.")
            parsed_specs.append(
                _ModelSpec(
                    name=str(raw.get("name", f"model_{i}")),
                    model_type=str(raw.get("model_type", "double_ensemble")).lower(),
                    horizon=horizon,
                    model_kwargs=dict(raw.get("model_kwargs", {})),
                )
            )
        return parsed_specs

    def _build_specs_from_compat_config(
        self,
        horizon_model_specs: Optional[Sequence[Dict]],
        horizon_days: Optional[Sequence[int]],
        double_ensemble_kwargs: Optional[Dict],
        lightgbm_kwargs: Optional[Dict],
        linear_kwargs: Optional[Dict],
    ) -> Optional[Sequence[Dict]]:
        if horizon_model_specs is not None:
            return horizon_model_specs

        de_kwargs = dict(double_ensemble_kwargs or {})
        lgb_kwargs = dict(lightgbm_kwargs or {})
        lin_kwargs = dict(linear_kwargs or {})

        horizons = [1, 5, 10]
        if horizon_days is not None:
            horizons = []
            for h in horizon_days:
                h_int = int(h)
                if h_int <= 0:
                    raise ValueError("horizon_days must contain only positive integers.")
                if h_int not in horizons:
                    horizons.append(h_int)
            if not horizons:
                raise ValueError("horizon_days must not be empty.")

        model_type_order = ["double_ensemble", "lightgbm", "linear"]
        kwargs_by_type = {
            "double_ensemble": de_kwargs,
            "lightgbm": lgb_kwargs,
            "linear": lin_kwargs,
        }
        specs = []
        for i, horizon in enumerate(horizons):
            model_type = model_type_order[i % len(model_type_order)]
            specs.append(
                {
                    "name": f"{model_type[:3]}_h{horizon}",
                    "model_type": model_type,
                    "horizon": int(horizon),
                    "model_kwargs": dict(kwargs_by_type[model_type]),
                }
            )
        return specs

    def _build_model(self, spec: _ModelSpec) -> Model:
        if spec.model_type in {"double_ensemble", "de"}:
            kwargs = dict(spec.model_kwargs)
            kwargs.setdefault("random_state", self.random_state)
            return DEnsembleModel(**kwargs)
        if spec.model_type in {"lightgbm", "lgb", "lgbm"}:
            kwargs = dict(spec.model_kwargs)
            kwargs.setdefault("seed", self.random_state)
            kwargs.setdefault("feature_fraction_seed", self.random_state)
            kwargs.setdefault("bagging_seed", self.random_state)
            kwargs.setdefault("data_random_seed", self.random_state)
            return LGBModel(**kwargs)
        if spec.model_type in {"linear", "lin"}:
            return LinearModel(**spec.model_kwargs)
        raise ValueError(f"Unsupported model_type: {spec.model_type}")

    def _predict_frame(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.DataFrame:
        preds = {spec.name: self.models[spec.name].predict(dataset, segment) for spec in self.model_specs}
        return pd.DataFrame(preds)

    def _prepare_prediction_scores(self, pred_frame: pd.DataFrame) -> pd.DataFrame:
        pred_frame = pred_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.use_rank_score:
            pred_frame = self._cross_sectional_rank_score(pred_frame)
        return pred_frame

    def _fit_regime_thresholds(self, pred_frame: pd.DataFrame):
        daily_feat = self._daily_regime_features(pred_frame)
        self.regime_consensus_thresholds = self._quantile_thresholds(
            daily_feat["consensus"], self.regime_consensus_quantiles
        )
        self.regime_disagreement_thresholds = self._quantile_thresholds(
            daily_feat["disagreement"], self.regime_disagreement_quantiles
        )
        self.num_regimes = (len(self.regime_consensus_thresholds) + 1) * (len(self.regime_disagreement_thresholds) + 1)

    def _learn_regime_weights(
        self, pred_frame: pd.DataFrame, label: pd.Series, regimes: pd.Series
    ) -> Dict[int, Dict[str, float]]:
        result = {}
        for regime_id in range(self.num_regimes):
            mask = regimes == regime_id
            if int(mask.sum()) < self.min_regime_samples:
                result[regime_id] = dict(self.global_weights)
                continue
            weights = self._learn_weights(pred_frame.loc[mask], label.loc[mask])
            result[regime_id] = weights
        return result

    def _learn_weights(self, pred_frame: pd.DataFrame, label: pd.Series) -> Dict[str, float]:
        if pred_frame.empty:
            return self._equal_weights()

        best_score = None
        best_weights = None
        for weights in self._weight_grid(len(self.model_specs)):
            if not self._check_monotonic_constraint(weights):
                continue
            score = pd.Series(pred_frame.values @ weights, index=pred_frame.index)
            score = self._apply_risk_controls(score)
            objective = self._topk_cost_objective(score, label)
            if best_score is None or objective > best_score:
                best_score = objective
                best_weights = weights

        if best_weights is None:
            return self._equal_weights()
        best_weights = np.clip(best_weights, 1e-12, None)
        best_weights = best_weights / best_weights.sum()
        return {spec.name: float(weight) for spec, weight in zip(self.model_specs, best_weights)}

    def _learn_memory_boost(self, score: pd.Series, label: pd.Series) -> float:
        best_boost = 0.0
        best_objective = None
        for boost in self.memory_boost_grid:
            adjusted = self._apply_turnover_boost(score, float(boost)) if boost > 0 else score
            objective = self._topk_cost_objective(adjusted, label)
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_boost = float(boost)
        return best_boost

    def _blend_by_regime(self, pred_frame: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        return self._blend_by_regime_with_weights(pred_frame, regimes, self.regime_weights, self.global_weights)

    def _blend_by_regime_with_weights(
        self,
        pred_frame: pd.DataFrame,
        regimes: pd.Series,
        regime_weights: Dict[int, Dict[str, float]],
        global_weights: Dict[str, float],
    ) -> pd.Series:
        blended = pd.Series(np.zeros(len(pred_frame), dtype=float), index=pred_frame.index)
        for regime_id in range(self.num_regimes):
            mask = regimes == regime_id
            if not bool(mask.any()):
                continue
            weights = regime_weights.get(regime_id, global_weights)
            weight_array = np.array([weights[spec.name] for spec in self.model_specs], dtype=float)
            blended.loc[mask] = pred_frame.loc[mask, [spec.name for spec in self.model_specs]].values @ weight_array
        return blended

    def _assign_row_regimes(self, pred_frame: pd.DataFrame) -> pd.Series:
        if not isinstance(pred_frame.index, pd.MultiIndex):
            return pd.Series(np.zeros(len(pred_frame), dtype=int), index=pred_frame.index)
        daily_feat = self._daily_regime_features(pred_frame)
        n_disagreement_bins = len(self.regime_disagreement_thresholds) + 1
        consensus_bin = np.searchsorted(
            self.regime_consensus_thresholds, daily_feat["consensus"].values, side="right"
        ).astype(int)
        disagreement_bin = np.searchsorted(
            self.regime_disagreement_thresholds, daily_feat["disagreement"].values, side="right"
        ).astype(int)
        daily_regimes = pd.Series(
            consensus_bin * n_disagreement_bins + disagreement_bin, index=daily_feat.index, dtype=int
        )
        date_level = self._date_level(pred_frame.index)
        row_dates = pred_frame.index.get_level_values(date_level)
        return pd.Series(daily_regimes.reindex(row_dates).fillna(0).values.astype(int), index=pred_frame.index)

    def _daily_regime_features(self, pred_frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(pred_frame.index, pd.MultiIndex):
            zero = pd.Series([0.0], index=pd.Index([0]))
            return pd.DataFrame({"consensus": zero, "disagreement": zero})

        date_level = self._date_level(pred_frame.index)
        pred_z = self._cross_sectional_zscore(pred_frame.fillna(0.0))
        consensus = pred_z.mean(axis=1).groupby(level=date_level).std().fillna(0.0)
        disagreement = pred_z.std(axis=1).groupby(level=date_level).mean().fillna(0.0)
        out = pd.DataFrame({"consensus": consensus, "disagreement": disagreement})
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _topk_cost_objective(self, score: pd.Series, label: pd.Series) -> float:
        common_index = score.index.intersection(label.index)
        if len(common_index) == 0:
            return float("-inf")
        score = score.loc[common_index]
        label = label.loc[common_index]
        if not isinstance(score.index, pd.MultiIndex):
            return float("-inf")

        date_level = self._date_level(score.index)
        inst_level = self._instrument_level(score.index)
        daily_returns = []
        daily_turnovers = []
        prev_inst = None

        for _, daily_score in score.groupby(level=date_level, sort=True):
            daily_label = label.loc[daily_score.index]
            selected_index = self._topk_index(daily_score)
            daily_returns.append(float(daily_label.loc[selected_index].mean()))
            selected_inst = set(selected_index.get_level_values(inst_level))
            if prev_inst is not None:
                overlap = len(prev_inst.intersection(selected_inst))
                daily_turnovers.append(1.0 - overlap / max(len(selected_inst), 1))
            prev_inst = selected_inst

        if not daily_returns:
            return float("-inf")
        mean_return = float(np.mean(daily_returns))
        turnover = float(np.mean(daily_turnovers)) if daily_turnovers else 0.0
        risk = float(np.std(daily_returns))
        return mean_return - self.turnover_penalty * turnover - self.risk_penalty * risk

    def _apply_risk_controls(self, score: pd.Series) -> pd.Series:
        score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if not isinstance(score.index, pd.MultiIndex):
            clipped = score.clip(-self.zscore_clip, self.zscore_clip)
            return clipped

        date_level = self._date_level(score.index)
        if self.neutralize_daily_mean:
            score = score - score.groupby(level=date_level).transform("mean")
        zscore = self._cross_sectional_zscore(score)
        return zscore.clip(-self.zscore_clip, self.zscore_clip).fillna(0.0)

    def _apply_final_score_controls(
        self,
        score: pd.Series,
        robust_rank_blend: Optional[float] = None,
        prediction_shrinkage: Optional[float] = None,
    ) -> pd.Series:
        rank_blend = self.robust_rank_blend if robust_rank_blend is None else float(robust_rank_blend)
        shrinkage = self.prediction_shrinkage if prediction_shrinkage is None else float(prediction_shrinkage)
        if not 0.0 <= rank_blend <= 1.0:
            raise ValueError("robust_rank_blend must be in [0, 1].")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("prediction_shrinkage must be in [0, 1].")

        controlled = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if rank_blend > 0:
            rank_score = self._rank_preserving_score(controlled)
            controlled = (1.0 - rank_blend) * controlled + rank_blend * rank_score
        if shrinkage < 1.0:
            controlled = controlled * shrinkage
        return controlled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _rank_preserving_score(self, score: pd.Series) -> pd.Series:
        rank_score = self._cross_sectional_rank_score(score.astype(float))
        return self._cross_sectional_zscore(rank_score).reindex(score.index).fillna(0.0)

    def _iter_final_score_controls(self):
        yielded_identity = False
        for rank_blend in self.robust_rank_blend_grid:
            for shrinkage in self.prediction_shrinkage_grid:
                yielded_identity = yielded_identity or self._is_identity_final_score_control(rank_blend, shrinkage)
                yield float(rank_blend), float(shrinkage)
        if self._final_score_control_grid_opt_in and not yielded_identity:
            yield 0.0, 1.0

    def _final_control_grid_has_identity(self) -> bool:
        return bool(self._final_score_control_grid_opt_in)

    @staticmethod
    def _is_identity_final_score_control(robust_rank_blend: float, prediction_shrinkage: float) -> bool:
        return abs(float(robust_rank_blend)) <= 1e-12 and abs(float(prediction_shrinkage) - 1.0) <= 1e-12

    def _daily_rank_ic(self, score: pd.Series, label: pd.Series) -> float:
        common_index = score.index.intersection(label.index)
        if len(common_index) == 0 or not isinstance(score.index, pd.MultiIndex):
            return float("-inf")

        score = score.loc[common_index].replace([np.inf, -np.inf], np.nan)
        label = label.loc[common_index].replace([np.inf, -np.inf], np.nan)
        date_level = self._date_level(score.index)
        daily_ic = []
        for _, daily_score in score.groupby(level=date_level, sort=True):
            daily_label = label.loc[daily_score.index]
            valid_mask = daily_score.notna() & daily_label.notna()
            if int(valid_mask.sum()) < 2:
                continue
            ic = daily_score.loc[valid_mask].rank(pct=True).corr(daily_label.loc[valid_mask].rank(pct=True))
            if np.isfinite(ic):
                daily_ic.append(float(ic))
        if not daily_ic:
            return float("-inf")
        return float(np.mean(daily_ic))

    def _apply_turnover_boost(self, score: pd.Series, boost: float) -> pd.Series:
        if boost <= 0:
            return score
        return self._apply_cost_aware_transform(
            score,
            memory_boost=boost,
            turnover_penalty=0.0,
            volatility_penalty=0.0,
        )

    def _apply_cost_aware_transform(
        self,
        score: pd.Series,
        memory_boost: float,
        turnover_penalty: float,
        volatility_penalty: float,
    ) -> pd.Series:
        score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        memory_boost = max(float(memory_boost), 0.0)
        turnover_penalty = max(float(turnover_penalty), 0.0)
        volatility_penalty = max(float(volatility_penalty), 0.0)

        if not isinstance(score.index, pd.MultiIndex):
            volatility = self._build_daily_volatility_proxy(score, previous_score=None)
            return transform_scores_for_cost(
                score,
                previous_holdings=None,
                volatility=volatility,
                previous_scores=None,
                previous_holding_boost=memory_boost,
                turnover_penalty=turnover_penalty,
                volatility_penalty=volatility_penalty,
                smoothing_alpha=0.0,
                normalize_scores=False,
                use_holding_weight=False,
            )

        date_level = self._date_level(score.index)
        inst_level = self._instrument_level(score.index)
        adjusted_parts = []
        previous_holdings = None
        previous_score = None

        for _, daily_score_raw in score.groupby(level=date_level, sort=True):
            inst_index = pd.Index(daily_score_raw.index.get_level_values(inst_level))
            daily_score = pd.Series(daily_score_raw.values, index=inst_index, dtype=float)
            volatility = self._build_daily_volatility_proxy(daily_score, previous_score)
            adjusted_daily = transform_scores_for_cost(
                daily_score,
                previous_holdings=previous_holdings,
                volatility=volatility,
                previous_scores=previous_score,
                previous_holding_boost=memory_boost,
                turnover_penalty=turnover_penalty,
                volatility_penalty=volatility_penalty,
                smoothing_alpha=0.0,
                normalize_scores=False,
                use_holding_weight=False,
            )
            selected_inst = self._topk_index(adjusted_daily)
            previous_holdings = pd.Series(1.0, index=pd.Index(selected_inst))
            previous_score = daily_score
            adjusted_parts.append(pd.Series(adjusted_daily.values, index=daily_score_raw.index))

        if not adjusted_parts:
            return score
        return pd.concat(adjusted_parts).sort_index()

    @staticmethod
    def _build_daily_volatility_proxy(daily_score: pd.Series, previous_score: Optional[pd.Series]) -> pd.Series:
        if previous_score is None:
            volatility = (daily_score - float(daily_score.mean())).abs()
        else:
            prev_aligned = previous_score.reindex(daily_score.index)
            fallback = float(daily_score.mean()) if len(daily_score) else 0.0
            prev_aligned = prev_aligned.fillna(fallback)
            volatility = (daily_score - prev_aligned).abs()
        return volatility.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _check_monotonic_constraint(self, weights: np.ndarray) -> bool:
        if not self.enforce_horizon_monotonic:
            return True
        horizon_weight = {}
        for w, spec in zip(weights, self.model_specs):
            horizon_weight[spec.horizon] = horizon_weight.get(spec.horizon, 0.0) + float(w)
        ordered_horizons = sorted(horizon_weight)
        values = [horizon_weight[h] for h in ordered_horizons]
        if self.monotonic_direction == "decreasing":
            return all(values[i] >= values[i + 1] - 1e-12 for i in range(len(values) - 1))
        return all(values[i] <= values[i + 1] + 1e-12 for i in range(len(values) - 1))

    def _weight_grid(self, n_models: int):
        grid = np.arange(0.0, 1.0 + 1e-9, self.search_step)
        if n_models == 1:
            yield np.array([1.0], dtype=float)
            return
        if n_models == 2:
            for w0 in grid:
                yield np.array([w0, max(0.0, 1.0 - w0)], dtype=float)
            return
        if n_models == 3:
            for w0 in grid:
                for w1 in grid:
                    w2 = 1.0 - w0 - w1
                    if w2 < -1e-9:
                        continue
                    yield np.array([w0, w1, max(0.0, w2)], dtype=float)
            return

        # Keep combinatorics bounded for >3 models.
        rng = np.random.default_rng(self.random_state)
        for _ in range(512):
            raw = rng.random(n_models)
            raw_sum = float(raw.sum())
            if raw_sum <= 0:
                continue
            yield raw / raw_sum

    def _equal_weights(self) -> Dict[str, float]:
        w = 1.0 / len(self.model_specs)
        return {spec.name: w for spec in self.model_specs}

    def _topk_index(self, score: pd.Series) -> pd.Index:
        if len(score) <= self.topk:
            return score.index
        values = score.values
        topk_pos = np.argpartition(values, -self.topk)[-self.topk :]
        return score.index[topk_pos]

    @staticmethod
    def _quantile_thresholds(values: pd.Series, quantiles: Sequence[float]) -> List[float]:
        cleaned = values.replace([np.inf, -np.inf], np.nan).dropna()
        if cleaned.empty:
            return []
        valid_quantiles = sorted({float(q) for q in quantiles if 0 < float(q) < 1})
        if not valid_quantiles:
            return []
        thresholds = cleaned.quantile(valid_quantiles).values
        return sorted(float(v) for v in np.unique(thresholds))

    @staticmethod
    def _normalize_float_grid(values: Optional[Sequence[float]]) -> List[float]:
        if values is None:
            return []
        return [float(v) for v in values]

    @staticmethod
    def _normalize_bounded_float_grid(
        values: Sequence[float], name: str, lower: float, upper: float
    ) -> List[float]:
        grid = [float(v) for v in values]
        if not grid:
            raise ValueError(f"{name} must not be empty.")
        if any(not np.isfinite(v) or v < lower or v > upper for v in grid):
            raise ValueError(f"{name} values must be finite and in [{lower}, {upper}].")
        return grid

    @staticmethod
    def _uniform_quantiles(num_regimes: int) -> List[float]:
        if num_regimes <= 1:
            return []
        return [i / num_regimes for i in range(1, num_regimes)]

    @staticmethod
    def _align_prediction_and_label(pred_frame: pd.DataFrame, label_df: pd.DataFrame):
        common_index = pred_frame.index.intersection(label_df.index)
        if len(common_index) == 0:
            raise ValueError("Prediction index and label index do not overlap.")
        return pred_frame.loc[common_index], label_df.loc[common_index]

    @staticmethod
    def _cross_sectional_rank_score(values: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(values.index, pd.MultiIndex):
            return values.rank(pct=True).fillna(0.0)
        date_level = RegimeHorizonCostEnsembleModel._date_level(values.index)
        return values.groupby(level=date_level).rank(pct=True).fillna(0.0)

    @staticmethod
    def _cross_sectional_zscore(values: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(values.index, pd.MultiIndex):
            centered = values - values.mean()
            std = values.std()
            return (centered / (std + 1e-12)).fillna(0.0)
        date_level = RegimeHorizonCostEnsembleModel._date_level(values.index)
        grouped = values.groupby(level=date_level)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        return ((values - mean) / (std + 1e-12)).fillna(0.0)

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return np.squeeze(label)
        raise ValueError("RegimeHorizonCostEnsembleModel only supports single-label training.")

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("RegimeHorizonCostEnsembleModel requires a MultiIndex index.")
        return "datetime" if "datetime" in index.names else index.names[0]

    @staticmethod
    def _instrument_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("RegimeHorizonCostEnsembleModel requires a MultiIndex index.")
        return "instrument" if "instrument" in index.names else index.names[-1]


class _HorizonLabelDataset:
    """Dataset adapter that replaces train/valid labels with horizon-smoothed targets."""

    def __init__(self, dataset: DatasetH, horizon: int, objective_kwargs: Optional[Dict] = None):
        self.dataset = dataset
        self.horizon = max(int(horizon), 1)
        self.segments = dataset.segments
        self.objective_kwargs = dict(objective_kwargs or {})
        self._label_cache: Dict[str, pd.DataFrame] = {}
        for segment in ("train", "valid"):
            if segment in self.segments:
                raw_label = dataset.prepare(segment, col_set=["label"], data_key=DataHandlerLP.DK_L)
                label_df = self._to_label_frame(raw_label)
                self._label_cache[segment] = self._build_horizon_label(
                    label_df,
                    self.horizon,
                    objective_kwargs=self.objective_kwargs if self.objective_kwargs else None,
                )

    def prepare(self, segments, col_set="__all", data_key=DataHandlerLP.DK_I, **kwargs):
        df = self.dataset.prepare(segments, col_set=col_set, data_key=data_key, **kwargs)
        if data_key != DataHandlerLP.DK_L:
            return df
        if not self._contains_label(col_set):
            return df
        if isinstance(segments, (list, tuple)) and isinstance(df, (list, tuple)):
            replaced = [self._replace_label_for_segment(seg, seg_df) for seg, seg_df in zip(segments, df)]
            return type(df)(replaced) if isinstance(df, tuple) else replaced
        return self._replace_label_for_segment(segments, df)

    def _replace_label_for_segment(self, segment, df):
        if not isinstance(segment, str) or segment not in self._label_cache:
            return df
        if not isinstance(df, pd.DataFrame):
            return df
        label_df = self._label_cache[segment].reindex(df.index)
        return self._replace_label_columns(df, label_df)

    @staticmethod
    def _replace_label_columns(df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
        if label_df.empty:
            return df

        label_values = label_df.copy()
        for c in label_values.columns:
            label_values.loc[:, c] = pd.to_numeric(label_values[c], errors="coerce")

        if isinstance(df.columns, pd.MultiIndex):
            if "label" not in df.columns.get_level_values(0):
                return df
            updated = df.copy()
            label_cols = [col for col in updated.columns if col[0] == "label"]
            if not label_cols:
                return updated

            for i, label_col in enumerate(label_cols):
                source_col = label_values.columns[min(i, label_values.shape[1] - 1)]
                source = label_values[source_col].reindex(updated.index)
                target_dtype = updated[label_col].dtype
                updated.loc[:, label_col] = source.to_numpy(dtype=target_dtype, na_value=np.nan)
            return updated

        if "label" in df.columns:
            updated = df.copy()
            source = label_values.iloc[:, 0].reindex(updated.index)
            target_dtype = updated["label"].dtype
            updated.loc[:, "label"] = source.to_numpy(dtype=target_dtype, na_value=np.nan)
            return updated

        matched_cols = [c for c in label_values.columns if c in df.columns]
        if matched_cols:
            updated = df.copy()
            for c in matched_cols:
                source = label_values[c].reindex(updated.index)
                target_dtype = updated[c].dtype
                updated.loc[:, c] = source.to_numpy(dtype=target_dtype, na_value=np.nan)
            return updated
        return df

    @staticmethod
    def _to_label_frame(raw_label: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if isinstance(raw_label, pd.DataFrame):
            if "label" in raw_label.columns:
                label_obj = raw_label["label"]
                if isinstance(label_obj, pd.DataFrame):
                    return label_obj
                return label_obj.to_frame("label")
            return raw_label
        if isinstance(raw_label, pd.Series):
            name = "label" if raw_label.name is None else raw_label.name
            return raw_label.to_frame(name)
        raise ValueError("Unsupported label object type for horizon adapter.")

    @staticmethod
    def _contains_label(col_set) -> bool:
        if col_set == "label":
            return True
        if isinstance(col_set, (list, tuple, set)):
            return "label" in col_set
        return False

    @staticmethod
    def _build_horizon_label(label_df: pd.DataFrame, horizon: int, objective_kwargs: Optional[Dict] = None) -> pd.DataFrame:
        if label_df.empty:
            return label_df.copy()
        if objective_kwargs:
            return build_objective_label_frame(label_df, base_horizon=horizon, **objective_kwargs)
        if horizon <= 1:
            return label_df.copy()

        label_s = label_df.iloc[:, 0].astype(float)
        if isinstance(label_s.index, pd.MultiIndex):
            inst_level = "instrument" if "instrument" in label_s.index.names else label_s.index.names[-1]
            label_h = label_s.groupby(level=inst_level, group_keys=False).apply(
                lambda x: _forward_window_mean(x, horizon)
            )
        else:
            label_h = _forward_window_mean(label_s, horizon)

        out = label_df.copy()
        target_dtype = out.iloc[:, 0].dtype
        aligned = label_h.reindex(label_df.index)
        out.iloc[:, 0] = aligned.to_numpy(dtype=target_dtype, na_value=np.nan)
        return out


def _forward_window_mean(series: pd.Series, window: int) -> pd.Series:
    """Forward-looking rolling mean within one segment and one instrument."""

    values = series.values.astype(float)
    n = len(values)
    if n == 0 or window <= 1:
        return series

    values_for_sum = np.nan_to_num(values, nan=0.0)
    valid = np.isfinite(values).astype(float)
    csum = np.cumsum(np.insert(values_for_sum, 0, 0.0))
    cvalid = np.cumsum(np.insert(valid, 0, 0.0))
    out = np.empty(n, dtype=float)

    for i in range(n):
        j = min(i + window, n)
        denom = cvalid[j] - cvalid[i]
        if denom <= 0:
            out[i] = np.nan
        else:
            out[i] = (csum[j] - csum[i]) / denom

    out_s = pd.Series(out, index=series.index)
    return out_s.fillna(series)
