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
from .finite_dnn import FiniteDNNModelPytorch
from .regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel


class RegimeFiniteDNNResidualModel(Model):
    """CPU regime anchor plus finite-guard DNN residual auxiliary."""

    def __init__(
        self,
        anchor_kwargs: Optional[Dict] = None,
        dnn_kwargs: Optional[Dict] = None,
        residual_weight_grid: Optional[Sequence[float]] = None,
        topk: int = 3,
        n_drop: int = 1,
        turnover_penalty: float = 0.0007,
        min_improvement: float = 0.0,
        use_rank_score: bool = True,
        target_mode: str = "residual_rank",
        candidate_pool: Optional[int] = None,
        gate_threshold: float = 0.5,
        residual_clip: float = 0.2,
        restrict_residual_to_anchor_pool: bool = True,
        random_state: int = 42,
    ):
        if topk <= 0:
            raise ValueError("topk must be positive.")
        self.logger = get_module_logger("RegimeFiniteDNNResidualModel")
        self.anchor_kwargs = dict(anchor_kwargs or {})
        self.anchor_kwargs.setdefault("random_state", random_state)
        self.dnn_kwargs = dict(dnn_kwargs or {})
        self.dnn_kwargs.setdefault("seed", random_state)
        self.residual_weight_grid = [float(x) for x in (residual_weight_grid or [0.0, 0.05, 0.1, 0.15, 0.2])]
        if not self.residual_weight_grid:
            raise ValueError("residual_weight_grid must not be empty.")
        self.topk = int(topk)
        self.n_drop = int(n_drop)
        self.turnover_penalty = float(turnover_penalty)
        self.min_improvement = float(min_improvement)
        self.use_rank_score = bool(use_rank_score)
        self.target_mode = str(target_mode or "residual_rank").strip().lower()
        if self.target_mode not in {"residual_rank", "topk_gate", "topk_penalty"}:
            raise ValueError("target_mode must be 'residual_rank', 'topk_gate', or 'topk_penalty'.")
        self.candidate_pool = int(candidate_pool) if candidate_pool is not None else max(self.topk * 4, self.topk)
        if self.candidate_pool < self.topk:
            raise ValueError("candidate_pool must be >= topk.")
        self.gate_threshold = float(gate_threshold)
        self.residual_clip = float(residual_clip)
        if self.residual_clip < 0.0:
            raise ValueError("residual_clip must be non-negative.")
        self.restrict_residual_to_anchor_pool = bool(restrict_residual_to_anchor_pool)
        self.random_state = int(random_state)
        self.anchor_model = RegimeHorizonCostEnsembleModel(**self.anchor_kwargs)
        self.residual_model = FiniteDNNModelPytorch(**self.dnn_kwargs)
        self.residual_weight = 0.0
        self.validation_summary: Dict[str, float] = {}
        self.fitted = False

    def fit(self, dataset: DatasetH):
        self.logger.info("Training CPU anchor model.")
        self.anchor_model.fit(dataset)

        train_label = self._prepare_label_series(dataset, "train")
        valid_label = self._prepare_label_series(dataset, "valid")
        train_anchor = self.anchor_model.predict(dataset, "train")
        valid_anchor = self.anchor_model.predict(dataset, "valid")

        train_residual = self._build_aux_target(train_label, train_anchor)
        valid_residual = self._build_aux_target(valid_label, valid_anchor)
        residual_dataset = _ResidualLabelDataset(dataset, {"train": train_residual, "valid": valid_residual})

        self.logger.info("Training finite DNN residual model.")
        self.residual_model.fit(residual_dataset)

        residual_valid = self._transform_aux_prediction(self.residual_model.predict(dataset, "valid"))
        valid_frame = pd.DataFrame(
            {"anchor": valid_anchor, "residual": residual_valid, "label": valid_label}
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if valid_frame.empty:
            raise ValueError("Validation residual frame is empty after alignment.")

        base_reward = self._portfolio_objective(valid_frame["anchor"], valid_frame["label"])
        best_weight = 0.0
        best_reward = base_reward
        for weight in self.residual_weight_grid:
            score = self._combine(valid_frame["anchor"], valid_frame["residual"], float(weight))
            reward = self._portfolio_objective(score, valid_frame["label"])
            if reward > best_reward:
                best_reward = reward
                best_weight = float(weight)

        if best_reward < base_reward + self.min_improvement:
            best_weight = 0.0
            best_reward = base_reward
        self.residual_weight = best_weight
        self.validation_summary = {
            "base_objective": float(base_reward),
            "best_objective": float(best_reward),
            "selected_residual_weight": float(best_weight),
        }
        self.logger.info("Residual DNN selected weight: %s", self.residual_weight)
        self.logger.info("Residual DNN validation summary: %s", self.validation_summary)
        self.fitted = True

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        anchor = self.anchor_model.predict(dataset, segment)
        if self.residual_weight == 0.0:
            return anchor
        residual = self.residual_model.predict(dataset, segment)
        residual = self._transform_aux_prediction(residual)
        frame = pd.DataFrame({"anchor": anchor, "residual": residual}).replace([np.inf, -np.inf], np.nan)
        out = self._combine(frame["anchor"].fillna(0.0), frame["residual"].fillna(0.0), self.residual_weight)
        return out.reindex(anchor.index).fillna(anchor)

    def _combine(self, anchor: pd.Series, residual: pd.Series, weight: float) -> pd.Series:
        if self.use_rank_score:
            anchor_score = self._cross_sectional_rank_score(anchor)
            if self.target_mode == "topk_gate":
                residual = self._mask_to_anchor_pool(anchor_score, residual)
            elif self.target_mode == "topk_penalty":
                residual = self._mask_to_anchor_pool(anchor_score, residual.clip(lower=0.0, upper=1.0))
                return anchor_score - float(weight) * residual
            else:
                residual = self._stabilize_residual_rank_signal(anchor_score, residual)
            return anchor_score + float(weight) * residual
        if self.target_mode == "residual_rank":
            residual = self._stabilize_residual_rank_signal(anchor, residual)
        if self.target_mode == "topk_penalty":
            return anchor - float(weight) * residual.clip(lower=0.0, upper=1.0)
        return anchor + float(weight) * residual

    def _build_aux_target(self, label: pd.Series, anchor: pd.Series) -> pd.Series:
        if self.target_mode == "topk_gate":
            return self._build_topk_gate_target(label, anchor)
        if self.target_mode == "topk_penalty":
            return self._build_topk_penalty_target(label, anchor)
        return self._build_residual_target(label, anchor)

    def _transform_aux_prediction(self, pred: pd.Series) -> pd.Series:
        if self.target_mode == "topk_gate":
            return (pred - self.gate_threshold).clip(-0.5, 0.5)
        if self.target_mode == "topk_penalty":
            return pred.clip(lower=0.0, upper=1.0)
        if self.use_rank_score:
            return (self._cross_sectional_rank_score(pred) - 0.5).clip(
                lower=-self.residual_clip, upper=self.residual_clip
            )
        return pred

    def _stabilize_residual_rank_signal(self, anchor_signal: pd.Series, residual: pd.Series) -> pd.Series:
        residual_signal = residual
        if self.restrict_residual_to_anchor_pool:
            residual_signal = self._mask_to_anchor_pool(anchor_signal, residual_signal)
        if self.residual_clip > 0.0:
            residual_signal = residual_signal.clip(lower=-self.residual_clip, upper=self.residual_clip)
        else:
            residual_signal = residual_signal * 0.0
        return residual_signal

    def _build_residual_target(self, label: pd.Series, anchor: pd.Series) -> pd.Series:
        frame = pd.DataFrame({"label": label, "anchor": anchor}).replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            raise ValueError("Residual target frame is empty after alignment.")
        label_score = self._cross_sectional_rank_score(frame["label"]) - 0.5
        anchor_score = self._cross_sectional_rank_score(frame["anchor"]) - 0.5
        return (label_score - anchor_score).rename("residual")

    def _build_topk_gate_target(self, label: pd.Series, anchor: pd.Series) -> pd.Series:
        frame = pd.DataFrame({"label": label, "anchor": anchor}).replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            raise ValueError("Gate target frame is empty after alignment.")
        date_level = "datetime" if "datetime" in frame.index.names else frame.index.names[0]
        targets = []
        for _, group in frame.groupby(level=date_level, sort=True):
            pool_index = group["anchor"].nlargest(min(self.candidate_pool, len(group))).index
            y = pd.Series(0.0, index=group.index, dtype="float32")
            if len(pool_index) > 0:
                threshold = float(group.loc[pool_index, "label"].median())
                y.loc[pool_index] = (group.loc[pool_index, "label"] >= threshold).astype("float32")
            targets.append(y)
        return pd.concat(targets).sort_index().rename("gate")

    def _build_topk_penalty_target(self, label: pd.Series, anchor: pd.Series) -> pd.Series:
        frame = pd.DataFrame({"label": label, "anchor": anchor}).replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            raise ValueError("Penalty target frame is empty after alignment.")
        date_level = "datetime" if "datetime" in frame.index.names else frame.index.names[0]
        targets = []
        for _, group in frame.groupby(level=date_level, sort=True):
            pool_index = group["anchor"].nlargest(min(self.candidate_pool, len(group))).index
            y = pd.Series(0.0, index=group.index, dtype="float32")
            if len(pool_index) > 0:
                threshold = float(group.loc[pool_index, "label"].median())
                y.loc[pool_index] = (group.loc[pool_index, "label"] < threshold).astype("float32")
            targets.append(y)
        return pd.concat(targets).sort_index().rename("penalty")

    def _mask_to_anchor_pool(self, anchor_score: pd.Series, residual: pd.Series) -> pd.Series:
        if not isinstance(anchor_score.index, pd.MultiIndex):
            return residual
        date_level = "datetime" if "datetime" in anchor_score.index.names else anchor_score.index.names[0]
        pieces = []
        for _, group in anchor_score.groupby(level=date_level, sort=True):
            pool_index = group.nlargest(min(self.candidate_pool, len(group))).index
            part = pd.Series(0.0, index=group.index, dtype=float)
            part.loc[pool_index] = residual.reindex(pool_index).fillna(0.0)
            pieces.append(part)
        return pd.concat(pieces).sort_index()

    def _portfolio_objective(self, score: pd.Series, label: pd.Series) -> float:
        frame = pd.DataFrame({"score": score, "label": label}).replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            return float("-inf")
        date_level = "datetime" if "datetime" in frame.index.names else frame.index.names[0]
        inst_level = "instrument" if "instrument" in frame.index.names else frame.index.names[-1]
        daily_returns = []
        turnovers = []
        prev_inst = None
        for _, group in frame.groupby(level=date_level, sort=True):
            selected = group["score"].nlargest(min(self.topk, len(group))).index
            daily_returns.append(float(group.loc[selected, "label"].mean()))
            curr_inst = set(selected.get_level_values(inst_level))
            if prev_inst is not None:
                turnovers.append(1.0 - len(prev_inst & curr_inst) / max(len(curr_inst), 1))
            prev_inst = curr_inst
        if not daily_returns:
            return float("-inf")
        return float(np.mean(daily_returns) - self.turnover_penalty * (np.mean(turnovers) if turnovers else 0.0))

    @staticmethod
    def _prepare_label_series(dataset: DatasetH, segment: Union[Text, slice]) -> pd.Series:
        raw = dataset.prepare(segment, col_set=["label"], data_key=DataHandlerLP.DK_L)
        label = raw["label"]
        if isinstance(label, pd.DataFrame):
            label = label.iloc[:, 0]
        return pd.to_numeric(label, errors="coerce")

    @staticmethod
    def _cross_sectional_rank_score(values: pd.Series) -> pd.Series:
        if not isinstance(values.index, pd.MultiIndex):
            return values.rank(pct=True).fillna(0.0)
        date_level = "datetime" if "datetime" in values.index.names else values.index.names[0]
        return values.groupby(level=date_level).rank(pct=True).fillna(0.0)


class _ResidualLabelDataset:
    def __init__(self, dataset: DatasetH, residuals: Dict[str, pd.Series]):
        self.dataset = dataset
        self.segments = dataset.segments
        self.residuals = {key: val.sort_index() for key, val in residuals.items()}

    def prepare(self, segment, col_set=None, data_key=DataHandlerLP.DK_I, **kwargs):
        data = self.dataset.prepare(segment, col_set=col_set, data_key=data_key, **kwargs)
        if isinstance(segment, str) and segment in self.residuals and self._uses_label(col_set):
            return self._replace_label(data, self.residuals[segment])
        return data

    @staticmethod
    def _uses_label(col_set) -> bool:
        if col_set is None:
            return True
        if col_set == "label":
            return True
        if isinstance(col_set, (list, tuple, set)):
            return "label" in col_set
        return False

    @staticmethod
    def _replace_label(data: pd.DataFrame, residual: pd.Series) -> pd.DataFrame:
        out = data.copy()
        common = out.index.intersection(residual.index)
        if len(common) == 0:
            return out
        if isinstance(out.columns, pd.MultiIndex):
            label_cols = [col for col in out.columns if col[0] == "label"]
            if label_cols:
                out.loc[common, label_cols[0]] = residual.loc[common].values
        elif "label" in out.columns:
            out.loc[common, "label"] = residual.loc[common].values
        return out
