# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

import numpy as np
import pandas as pd

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from .transcendence_hybrid import TranscendenceHybridModel, _BaseModelSpec


class CostAwareTranscendenceHybrid(TranscendenceHybridModel):
    """TranscendenceHybridModel with cost-aware (turnover-penalized) IR objective.

    Extends TranscendenceHybridModel by replacing the portfolio objective with
    a costed-IR computation that subtracts turnover costs from daily returns:

        costed_return[t] = raw_return[t] - turnover_penalty * turnover[t]
        costed_IR = mean(costed_returns) / std(costed_returns) * sqrt(252)

    This encourages weight selection and residual tuning that favor low-turnover
    alpha strategies, improving net-of-cost performance.

    Additionally supports a ``turnover_penalty_grid`` for grid-searching the
    best penalty value on validation data.
    """

    def __init__(
        self,
        base_model_specs: Optional[Sequence[Dict]] = None,
        use_rank_score: bool = True,
        topk: int = 45,
        search_step: float = 0.1,
        max_random_weight_candidates: int = 512,
        ir_weight: float = 1.0,
        annret_weight: float = 4.0,
        ann_scaler: int = 252,
        residual_learner: Optional[Dict] = None,
        deep_branch: Optional[Dict] = None,
        random_state: int = 42,
        # Cost-aware parameters
        turnover_penalty: float = 0.0007,
        turnover_penalty_grid: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        super().__init__(
            base_model_specs=base_model_specs,
            use_rank_score=use_rank_score,
            topk=topk,
            search_step=search_step,
            max_random_weight_candidates=max_random_weight_candidates,
            ir_weight=ir_weight,
            annret_weight=annret_weight,
            ann_scaler=ann_scaler,
            residual_learner=residual_learner,
            deep_branch=deep_branch,
            random_state=random_state,
        )
        self.turnover_penalty = float(turnover_penalty)
        self.turnover_penalty_grid = (
            sorted({float(v) for v in turnover_penalty_grid})
            if turnover_penalty_grid
            else [self.turnover_penalty]
        )
        self.logger = get_module_logger("CostAwareTranscendenceHybrid")

    # ------------------------------------------------------------------
    # Fit override: grid-search turnover penalty on validation
    # ------------------------------------------------------------------

    def fit(self, dataset: DatasetH):
        """Fit base models, then grid-search the best turnover penalty."""
        # Train all base models and deep branch (no penalty yet).
        self.models = {}
        self.model_order = []

        for spec in self.specs:
            model = self._build_base_model(spec)
            self.logger.info("Training base model %s (%s).", spec.name, spec.model_type)
            model.fit(dataset)
            self.models[spec.name] = model
            self.model_order.append(spec.name)

        deep_model, deep_name, deep_msg = self._build_optional_deep_model()
        self.deep_branch_active = deep_model is not None
        self.deep_branch_message = deep_msg
        if deep_model is not None and deep_name is not None:
            self.logger.info("Training optional deep branch %s.", deep_name)
            deep_model.fit(dataset)
            self.models[deep_name] = deep_model
            self.model_order.append(deep_name)
        else:
            self.logger.info("Deep branch inactive: %s", deep_msg)

        valid_label = self._prepare_label_series(dataset, "valid", data_key=DataHandlerLP.DK_R)
        valid_pred_raw = self._predict_frame(dataset, "valid")
        valid_pred, valid_label = self._align_frame_and_label(valid_pred_raw, valid_label)
        valid_pred = self._prepare_prediction_scores(valid_pred)

        # Grid-search turnover penalty: for each candidate penalty, learn
        # weights and residual, then evaluate costed IR.
        if len(self.turnover_penalty_grid) > 1:
            self.logger.info(
                "Grid-searching turnover penalty over %d candidates: %s",
                len(self.turnover_penalty_grid),
                self.turnover_penalty_grid,
            )
            best_penalty = self.turnover_penalty_grid[0]
            best_objective = float("-inf")

            for penalty in self.turnover_penalty_grid:
                self.turnover_penalty = float(penalty)
                weights = self._learn_weights(valid_pred, valid_label)
                blend = self._blend_with_weights(valid_pred, weights)
                objective = self._costed_portfolio_objective(blend, valid_label)
                self.logger.info(
                    "  penalty=%.6f  costed_objective=%.6f", penalty, objective
                )
                if objective > best_objective:
                    best_objective = objective
                    best_penalty = float(penalty)

            self.turnover_penalty = best_penalty
            self.logger.info(
                "Selected turnover_penalty=%.6f (costed_objective=%.6f)",
                best_penalty,
                best_objective,
            )

        # Final weight learning with the selected penalty.
        self.model_weights = self._learn_weights(valid_pred, valid_label)
        self.logger.info("Learned base weights: %s", self.model_weights)

        valid_blend = self._blend_with_weights(valid_pred, self.model_weights)
        valid_ir, valid_annret, valid_hit = self._portfolio_metrics(valid_blend, valid_label)
        self.validation_summary = {
            "valid_ir": float(valid_ir),
            "valid_annret": float(valid_annret),
            "valid_hit_ratio": float(valid_hit),
            "turnover_penalty": float(self.turnover_penalty),
        }

        self._fit_optional_residual(dataset)
        self.fitted = True

    # ------------------------------------------------------------------
    # Costed objective (the core modification)
    # ------------------------------------------------------------------

    def _costed_portfolio_objective(self, score: pd.Series, label: pd.Series) -> float:
        """Costed IR: mean(costed_returns) / std(costed_returns) * sqrt(ann_scaler).

        costed_return[t] = daily_return[t] - turnover_penalty * daily_turnover[t]
        """
        ir, annret, _ = self._costed_portfolio_metrics(score, label)
        return self.ir_weight * ir + self.annret_weight * annret

    def _costed_portfolio_metrics(
        self, score: pd.Series, label: pd.Series
    ) -> Tuple[float, float, float]:
        """Compute IR, annualized return, and hit ratio after subtracting
        turnover costs from daily returns."""
        aligned = score.index.intersection(label.index)
        if len(aligned) == 0:
            return float("-inf"), float("-inf"), 0.0
        score = self._clean_series(score.loc[aligned])
        label = self._clean_series(label.loc[aligned])

        if not isinstance(score.index, pd.MultiIndex):
            corr = score.rank(pct=True).corr(label.rank(pct=True))
            val = 0.0 if corr is None or not np.isfinite(corr) else float(corr)
            return val, val, 0.0

        date_level = self._date_level(score.index)
        inst_level = (
            "instrument" if "instrument" in score.index.names else score.index.names[-1]
        )

        costed_returns: List[float] = []
        prev_selected: Optional[set] = None

        for _, daily_score in score.groupby(level=date_level, sort=True):
            daily_label = label.loc[daily_score.index]
            if daily_score.empty:
                continue
            idx = self._topk_index(daily_score)
            day_ret = float(daily_label.loc[idx].mean())
            if not np.isfinite(day_ret):
                continue

            # Compute turnover
            current_selected = set(idx.get_level_values(inst_level))
            if prev_selected is not None and len(current_selected) > 0:
                overlap = len(prev_selected.intersection(current_selected))
                turnover = 1.0 - overlap / max(len(current_selected), 1)
            else:
                turnover = 0.0
            prev_selected = current_selected

            costed_ret = day_ret - self.turnover_penalty * turnover
            costed_returns.append(costed_ret)

        if not costed_returns:
            return float("-inf"), float("-inf"), 0.0

        ret_arr = np.asarray(costed_returns, dtype=float)
        mean_ret = float(np.mean(ret_arr))
        std_ret = float(np.std(ret_arr))
        ir = float((mean_ret / (std_ret + 1e-12)) * np.sqrt(self.ann_scaler))
        annret = float(mean_ret * self.ann_scaler)
        hit_ratio = float(np.mean(ret_arr > 0))
        return ir, annret, hit_ratio

    # ------------------------------------------------------------------
    # Override objective to use costed version
    # ------------------------------------------------------------------

    def _portfolio_objective(self, score: pd.Series, label: pd.Series) -> float:
        """Override: use costed-IR objective instead of raw IR."""
        return self._costed_portfolio_objective(score, label)

    def _portfolio_metrics(
        self, score: pd.Series, label: pd.Series
    ) -> Tuple[float, float, float]:
        """Override: use costed portfolio metrics."""
        return self._costed_portfolio_metrics(score, label)
