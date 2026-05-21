# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, Optional, Sequence, Text, Union

import numpy as np
import pandas as pd

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from .adaptive_ensemble import AdaptiveEnsembleModel
from .double_ensemble import DEnsembleModel
from .gbdt import LGBModel
from .linear import LinearModel


class VolReliabilityReRankDEnsembleModel(Model):
    """Validation-learned blend plus a soft top-k re-ranker.

    The model trains DoubleEnsemble, AdaptiveEnsemble, LightGBM, and Linear
    on train/valid only. Validation is then used to learn nonnegative blend
    weights and a calibration gate that boosts names with reliable local
    structure, which can change same-day ordering beyond a monotone transform.
    """

    def __init__(
        self,
        double_ensemble_kwargs: Optional[Dict] = None,
        adaptive_ensemble_kwargs: Optional[Dict] = None,
        lightgbm_kwargs: Optional[Dict] = None,
        linear_kwargs: Optional[Dict] = None,
        topk: int = 50,
        turnover_penalty: float = 0.0002,
        blend_search_step: float = 0.1,
        reliability_power_grid: Optional[Sequence[float]] = None,
        rel_coef_grid: Optional[Sequence[float]] = None,
        vol_coef_grid: Optional[Sequence[float]] = None,
        int_coef_grid: Optional[Sequence[float]] = None,
        vol_window: int = 20,
        stability_window: int = 10,
        blend_coef: float = 0.2,
        gate_strength_grid: Optional[Sequence[float]] = None,
        gate_tau_grid: Optional[Sequence[float]] = None,
        gate_quantile_grid: Optional[Sequence[float]] = None,
        use_rank_score: bool = True,
        random_state: int = 42,
        min_weight: float = 1e-6,
    ):
        if topk <= 0:
            raise ValueError("topk must be positive.")
        if not 0 < blend_search_step <= 1:
            raise ValueError("blend_search_step must be in (0, 1].")
        if vol_window <= 0 or stability_window <= 0:
            raise ValueError("vol_window and stability_window must be positive.")
        if not 0 <= blend_coef <= 1:
            raise ValueError("blend_coef must be in [0, 1].")

        self.logger = get_module_logger("VolReliabilityReRankDEnsembleModel")
        self.topk = int(topk)
        self.turnover_penalty = float(turnover_penalty)
        self.blend_search_step = float(blend_search_step)
        self.reliability_power_grid = list(reliability_power_grid or [0.5, 1.0, 1.5, 2.0])
        self.rel_coef_grid = list(rel_coef_grid or [0.0, 0.05, 0.1, 0.15])
        self.vol_coef_grid = list(vol_coef_grid or [0.0, 0.03, 0.06, 0.1])
        self.int_coef_grid = list(int_coef_grid or [0.0, 0.03, 0.06, 0.1])
        self.vol_window = int(vol_window)
        self.stability_window = int(stability_window)
        self.blend_coef = float(blend_coef)
        self.gate_strength_grid = list(gate_strength_grid or [0.0, 0.05, 0.1, 0.15, 0.2])
        self.gate_tau_grid = list(gate_tau_grid or [0.01, 0.02, 0.05, 0.1])
        self.gate_quantile_grid = list(gate_quantile_grid or [0.6, 0.7, 0.8, 0.9])
        self.use_rank_score = bool(use_rank_score)
        self.random_state = int(random_state)
        self.min_weight = float(min_weight)

        self.double_ensemble_kwargs = dict(double_ensemble_kwargs or {})
        self.adaptive_ensemble_kwargs = dict(adaptive_ensemble_kwargs or {})
        self.lightgbm_kwargs = dict(lightgbm_kwargs or {})
        self.linear_kwargs = dict(linear_kwargs or {})

        self.double_ensemble_kwargs.setdefault("random_state", random_state)
        self.lightgbm_kwargs.setdefault("seed", random_state)
        self.lightgbm_kwargs.setdefault("feature_fraction_seed", random_state)
        self.lightgbm_kwargs.setdefault("bagging_seed", random_state)
        self.lightgbm_kwargs.setdefault("data_random_seed", random_state)

        self.double_ensemble_model = DEnsembleModel(**self.double_ensemble_kwargs)
        self.adaptive_ensemble_model = AdaptiveEnsembleModel(**self.adaptive_ensemble_kwargs)
        self.lightgbm_model = LGBModel(**self.lightgbm_kwargs)
        self.linear_model = LinearModel(**self.linear_kwargs)

        self.model_order = ["double_ensemble", "adaptive_ensemble", "lightgbm", "linear"]
        self.model_weights: Dict[str, float] = {}
        self.gate_strength = 0.0
        self.gate_tau = 0.02
        self.gate_quantile = 0.8
        self.instrument_reliability = pd.Series(dtype=float)
        self.instrument_volatility = pd.Series(dtype=float)
        self.instrument_instability = pd.Series(dtype=float)
        self.reliability_power = 1.0
        self.rel_coef = 0.0
        self.vol_coef = 0.0
        self.int_coef = 0.0

    def fit(self, dataset: DatasetH):
        self.logger.info("Training DoubleEnsemble sub-model.")
        self.double_ensemble_model.fit(dataset)

        self.logger.info("Training AdaptiveEnsemble sub-model.")
        self.adaptive_ensemble_model.fit(dataset)

        self.logger.info("Training LightGBM sub-model.")
        self.lightgbm_model.fit(dataset)

        self.logger.info("Training Linear sub-model.")
        self.linear_model.fit(dataset)

        valid_pred = self._predict_frame(dataset, "valid")
        valid_label = dataset.prepare("valid", col_set=["label"], data_key=DataHandlerLP.DK_L)["label"]
        valid_pred, valid_label = self._align_prediction_and_label(valid_pred, valid_label)
        valid_label_s = pd.Series(self._squeeze_label(valid_label), index=valid_pred.index, name="label")

        self.instrument_reliability = self._fit_instrument_reliability(valid_pred, valid_label_s)
        self.instrument_volatility, self.instrument_instability = self._fit_vol_instability(valid_pred)
        self.model_weights = self._learn_nonnegative_weights(valid_pred, valid_label_s)
        (
            self.reliability_power,
            self.rel_coef,
            self.vol_coef,
            self.int_coef,
            self.gate_strength,
            self.gate_tau,
            self.gate_quantile,
        ) = self._learn_gate(valid_pred, valid_label_s)
        self.logger.info("Validation weights: %s", self.model_weights)
        self.logger.info(
            "Validation gate: power=%s rel=%s vol=%s int=%s strength=%s tau=%s quantile=%s",
            self.reliability_power,
            self.rel_coef,
            self.vol_coef,
            self.int_coef,
            self.gate_strength,
            self.gate_tau,
            self.gate_quantile,
        )

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.model_weights:
            raise ValueError("model is not fitted yet!")
        pred_frame = self._predict_frame(dataset, segment)
        return self._apply_blend_and_gate(pred_frame)

    def _predict_frame(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.DataFrame:
        preds = {
            "double_ensemble": self.double_ensemble_model.predict(dataset, segment),
            "adaptive_ensemble": self.adaptive_ensemble_model.predict(dataset, segment),
            "lightgbm": self.lightgbm_model.predict(dataset, segment),
            "linear": self.linear_model.predict(dataset, segment),
        }
        return pd.DataFrame(preds)

    def _learn_nonnegative_weights(self, pred_frame: pd.DataFrame, label: pd.Series) -> Dict[str, float]:
        grid = np.arange(0.0, 1.0 + 1e-9, self.blend_search_step)
        best_score = None
        best_weights = None
        for w0 in grid:
            for w1 in grid:
                for w2 in grid:
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -1e-9:
                        continue
                    weights = np.array([w0, w1, w2, max(0.0, w3)], dtype=float)
                    if weights.sum() <= 0:
                        continue
                    weights = np.clip(weights, self.min_weight, None)
                    weights = weights / weights.sum()
                    score = self._topk_objective(self._blend(pred_frame, weights), label)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_weights = weights

        if best_weights is None:
            best_weights = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
        return {name: float(weight) for name, weight in zip(self.model_order, best_weights)}

    def _learn_gate(self, pred_frame: pd.DataFrame, label: pd.Series):
        base = self._blend(pred_frame, np.asarray([self.model_weights[name] for name in self.model_order], dtype=float))
        reliability = self._build_reliability_score(pred_frame)
        volatility = self._build_volatility_score(pred_frame)
        instability = self._build_instability_score(pred_frame)

        best = None
        best_params = (1.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.8)
        for power in self.reliability_power_grid:
            rel_term = self._cross_sectional_zscore(np.sign(reliability) * np.power(np.abs(reliability), float(power)))
            for rel_coef in self.rel_coef_grid:
                for vol_coef in self.vol_coef_grid:
                    for int_coef in self.int_coef_grid:
                        gate_source = self._cross_sectional_zscore(
                            float(rel_coef) * rel_term + float(vol_coef) * volatility + float(int_coef) * instability
                        )
                        for strength in self.gate_strength_grid:
                            for tau in self.gate_tau_grid:
                                for q in self.gate_quantile_grid:
                                    gated = self._apply_gate(base, gate_source, float(strength), float(tau), float(q))
                                    score = self._topk_objective(gated, label)
                                    if best is None or score > best:
                                        best = score
                                        best_params = (
                                            float(power),
                                            float(rel_coef),
                                            float(vol_coef),
                                            float(int_coef),
                                            float(strength),
                                            float(tau),
                                            float(q),
                                        )
        return best_params

    def _apply_blend_and_gate(self, pred_frame: pd.DataFrame) -> pd.Series:
        weights = np.asarray([self.model_weights[name] for name in self.model_order], dtype=float)
        base = self._blend(pred_frame, weights)
        reliability = self._build_reliability_score(pred_frame)
        rel_term = self._cross_sectional_zscore(np.sign(reliability) * np.power(np.abs(reliability), self.reliability_power))
        volatility = self._build_volatility_score(pred_frame)
        instability = self._build_instability_score(pred_frame)
        gate_source = self._cross_sectional_zscore(
            self.rel_coef * rel_term + self.vol_coef * volatility + self.int_coef * instability
        )
        gated = self._apply_gate(base, gate_source, self.gate_strength, self.gate_tau, self.gate_quantile)
        return (1.0 - self.blend_coef) * base + self.blend_coef * gated

    def _blend(self, pred_frame: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        return pd.Series(pred_frame[self.model_order].values @ weights, index=pred_frame.index)

    def _apply_gate(self, base: pd.Series, gate_source: pd.Series, strength: float, tau: float, quantile: float):
        if strength <= 0:
            return base
        date_level = self._date_level(base.index)
        pieces = []
        safe_tau = max(float(tau), 1e-6)
        for _, daily_base in base.groupby(level=date_level, sort=True):
            daily_gate = gate_source.loc[daily_base.index].fillna(0.0)
            threshold = float(daily_gate.quantile(quantile)) if len(daily_gate) else 0.0
            soft = 1.0 / (1.0 + np.exp(-(daily_gate.values - threshold) / safe_tau))
            signed = np.where(daily_gate.values >= threshold, 1.0, -1.0)
            gate = pd.Series(strength * signed * soft, index=daily_base.index)
            pieces.append(daily_base + gate.reindex(daily_base.index).fillna(0.0))
        return pd.concat(pieces).sort_index() if pieces else base.copy()

    def _build_reliability_score(self, pred_frame: pd.DataFrame) -> pd.Series:
        score = self._cross_sectional_zscore(pred_frame.mean(axis=1).fillna(0.0))
        if self.instrument_reliability.empty or not isinstance(score.index, pd.MultiIndex):
            return score
        inst_level = self._instrument_level(score.index)
        inst_idx = score.index.get_level_values(inst_level)
        rel = self.instrument_reliability.reindex(inst_idx).fillna(0.0)
        return self._cross_sectional_zscore(pd.Series(rel.values, index=score.index))

    def _build_volatility_score(self, pred_frame: pd.DataFrame) -> pd.Series:
        dispersion = pred_frame.std(axis=1).fillna(0.0)
        if isinstance(pred_frame.index, pd.MultiIndex):
            date_level = self._date_level(pred_frame.index)
            dispersion = dispersion.groupby(level=date_level).transform(lambda x: x.rank(pct=True))
        if self.instrument_volatility.empty or not isinstance(pred_frame.index, pd.MultiIndex):
            return self._cross_sectional_zscore(dispersion.fillna(0.0))
        inst_level = self._instrument_level(pred_frame.index)
        inst_idx = pred_frame.index.get_level_values(inst_level)
        vol = self.instrument_volatility.reindex(inst_idx).fillna(0.0)
        return self._cross_sectional_zscore(pd.Series(vol.values, index=pred_frame.index))

    def _build_instability_score(self, pred_frame: pd.DataFrame) -> pd.Series:
        dispersion = pred_frame.std(axis=1).fillna(0.0)
        if not isinstance(pred_frame.index, pd.MultiIndex):
            return self._cross_sectional_zscore(dispersion.fillna(0.0))
        inst_level = self._instrument_level(pred_frame.index)
        inst_idx = pred_frame.index.get_level_values(inst_level)
        instability = self.instrument_instability.reindex(inst_idx).fillna(0.0)
        return self._cross_sectional_zscore(pd.Series(instability.values, index=pred_frame.index))

    def _fit_instrument_reliability(self, pred_frame: pd.DataFrame, label: pd.Series) -> pd.Series:
        base = self._cross_sectional_zscore(pred_frame.mean(axis=1).fillna(0.0))
        if not isinstance(base.index, pd.MultiIndex):
            return pd.Series(0.0, index=pd.Index([], name="instrument"))
        inst_level = self._instrument_level(base.index)
        date_level = self._date_level(base.index)
        pred_centered = base - base.groupby(level=date_level).transform("mean")
        label_centered = label - label.groupby(level=date_level).transform("mean")
        cov = (pred_centered * label_centered).groupby(level=inst_level).mean()
        pred_std = pred_centered.groupby(level=inst_level).std()
        label_std = label_centered.groupby(level=inst_level).std()
        reliability = cov / (pred_std * label_std + 1e-12)
        count = base.groupby(level=inst_level).size().astype(float)
        shrink = count / (count + 30.0)
        reliability = (reliability.fillna(0.0) * shrink).clip(-1.0, 1.0)
        return reliability.fillna(0.0)

    def _fit_vol_instability(self, pred_frame: pd.DataFrame):
        if not isinstance(pred_frame.index, pd.MultiIndex):
            empty = pd.Series(dtype=float)
            return empty, empty
        inst_level = self._instrument_level(pred_frame.index)
        date_level = self._date_level(pred_frame.index)
        mean_score = pred_frame.mean(axis=1).fillna(0.0)
        grouped = mean_score.groupby(level=inst_level, sort=False)
        vol_map = {}
        instab_map = {}
        for inst, series in grouped:
            series = series.sort_index(level=date_level)
            if len(series) < 2:
                vol_map[inst] = 0.0
                instab_map[inst] = 0.0
                continue
            vol = series.rolling(self.vol_window, min_periods=2).std().mean()
            instab = series.diff().abs().rolling(self.stability_window, min_periods=2).mean().mean()
            vol_map[inst] = float(0.0 if pd.isna(vol) else vol)
            instab_map[inst] = float(0.0 if pd.isna(instab) else instab)
        return pd.Series(vol_map).fillna(0.0), pd.Series(instab_map).fillna(0.0)

    def _topk_objective(self, score: pd.Series, label: pd.Series) -> float:
        df = pd.DataFrame({"score": score, "label": label}).dropna()
        if df.empty:
            return float("-inf")
        date_level = self._date_level(df.index)
        inst_level = self._instrument_level(df.index)
        daily_returns = []
        daily_turnovers = []
        prev_selected = None
        for _, daily_df in df.groupby(level=date_level, sort=True):
            topk_df = daily_df.nlargest(self.topk, columns="score")
            daily_returns.append(float(topk_df["label"].mean()))
            selected = set(topk_df.index.get_level_values(inst_level))
            if prev_selected is not None:
                overlap = len(prev_selected.intersection(selected))
                daily_turnovers.append(1.0 - overlap / max(len(selected), 1))
            prev_selected = selected
        turnover = float(np.mean(daily_turnovers)) if daily_turnovers else 0.0
        return float(np.mean(daily_returns) - self.turnover_penalty * turnover)

    @staticmethod
    def _align_prediction_and_label(pred_frame: pd.DataFrame, label_df: pd.DataFrame):
        common_index = pred_frame.index.intersection(label_df.index)
        if len(common_index) == 0:
            raise ValueError("Prediction index and label index do not overlap.")
        return pred_frame.loc[common_index], label_df.loc[common_index]

    @staticmethod
    def _cross_sectional_zscore(values: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(values.index, pd.MultiIndex):
            centered = values - values.mean()
            return (centered / (values.std() + 1e-12)).fillna(0.0)
        date_level = VolReliabilityReRankDEnsembleModel._date_level(values.index)
        grouped = values.groupby(level=date_level)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        return ((values - mean) / (std + 1e-12)).fillna(0.0)

    @staticmethod
    def _squeeze_label(label_df: pd.DataFrame) -> np.ndarray:
        label = label_df.values
        if label.ndim == 2 and label.shape[1] == 1:
            return np.squeeze(label)
        raise ValueError("VolReliabilityReRankDEnsembleModel only supports single-label training.")

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("VolReliabilityReRankDEnsembleModel requires a MultiIndex index.")
        return "datetime" if "datetime" in index.names else index.names[0]

    @staticmethod
    def _instrument_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("VolReliabilityReRankDEnsembleModel requires a MultiIndex index.")
        return "instrument" if "instrument" in index.names else index.names[-1]
