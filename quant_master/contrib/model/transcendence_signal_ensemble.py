# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Text, Tuple, Union

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
from .regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel
from .transcendence_hybrid import TranscendenceHybridModel


@dataclass
class _BaseLearnerSpec:
    name: str
    model_type: str
    model_kwargs: Dict
    class_name: Optional[str]
    module_path: Optional[str]


class TranscendenceSignalEnsembleModel(Model):
    """
    Validation-selected rank ensemble for Transcendence.

    Key rule:
    - base learners are fit on train/valid
    - all ensemble / execution hyper-parameters are selected only on valid
    - test segment is prediction-only
    """

    def __init__(
        self,
        base_learner_specs: Optional[Sequence[Dict]] = None,
        use_rank_score: bool = True,
        search_step: float = 0.1,
        max_random_weight_candidates: int = 192,
        refine_top_weight_candidates: int = 20,
        topk_grid: Optional[Sequence[int]] = None,
        n_drop_grid: Optional[Sequence[int]] = None,
        memory_boost_grid: Optional[Sequence[float]] = None,
        turnover_penalty_grid: Optional[Sequence[float]] = None,
        volatility_penalty_grid: Optional[Sequence[float]] = None,
        open_cost: float = 0.0001,
        close_cost: float = 0.0006,
        ir_weight: float = 1.0,
        annret_weight: float = 3.0,
        hit_ratio_weight: float = 0.15,
        max_drawdown_penalty: float = 0.25,
        ann_scaler: int = 252,
        random_state: int = 42,
        manual_weight_candidates: Optional[Sequence[Union[Sequence[float], Dict[str, float]]]] = None,
        weight_constraints: Optional[Dict] = None,
    ):
        if not 0 < float(search_step) <= 1:
            raise ValueError("search_step must be in (0, 1].")
        if int(max_random_weight_candidates) <= 0:
            raise ValueError("max_random_weight_candidates must be positive.")
        if int(refine_top_weight_candidates) <= 0:
            raise ValueError("refine_top_weight_candidates must be positive.")
        if ann_scaler <= 0:
            raise ValueError("ann_scaler must be positive.")

        self.logger = get_module_logger("TranscendenceSignalEnsembleModel")
        self.use_rank_score = bool(use_rank_score)
        self.search_step = float(search_step)
        self.max_random_weight_candidates = int(max_random_weight_candidates)
        self.refine_top_weight_candidates = int(refine_top_weight_candidates)
        self.topk_grid = [int(x) for x in (topk_grid or [45])]
        self.n_drop_grid = [int(x) for x in (n_drop_grid or [3, 4])]
        self.memory_boost_grid = [float(x) for x in (memory_boost_grid or [0.0, 0.005])]
        self.turnover_penalty_grid = [float(x) for x in (turnover_penalty_grid or [0.0, 0.00005, 0.0001])]
        self.volatility_penalty_grid = [float(x) for x in (volatility_penalty_grid or [0.0])]
        self.open_cost = float(open_cost)
        self.close_cost = float(close_cost)
        self.ir_weight = float(ir_weight)
        self.annret_weight = float(annret_weight)
        self.hit_ratio_weight = float(hit_ratio_weight)
        self.max_drawdown_penalty = float(max_drawdown_penalty)
        self.ann_scaler = int(ann_scaler)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)
        self.manual_weight_candidates = list(manual_weight_candidates or [])
        self.weight_constraints = dict(weight_constraints or {})

        self.specs = self._parse_specs(base_learner_specs)
        self.models: Dict[str, Model] = {}
        self.model_order: List[str] = []
        self.model_weights: Dict[str, float] = {}
        self.best_topk: int = int(self.topk_grid[0])
        self.best_n_drop: int = int(self.n_drop_grid[0])
        self.best_memory_boost: float = float(self.memory_boost_grid[0])
        self.best_turnover_penalty: float = float(self.turnover_penalty_grid[0])
        self.best_volatility_penalty: float = float(self.volatility_penalty_grid[0])
        self.validation_summary: Dict[str, float] = {}
        self.weight_candidate_diagnostics: List[Dict[str, float]] = []
        self.fitted = False

    def fit(self, dataset: DatasetH):
        self.models = {}
        self.model_order = []
        for spec in self.specs:
            model = self._build_model(spec)
            self.logger.info("Training base learner %s (%s).", spec.name, spec.model_type)
            model.fit(dataset)
            self.models[spec.name] = model
            self.model_order.append(spec.name)

        valid_label = self._prepare_label_series(dataset, "valid")
        valid_raw = self._predict_frame(dataset, "valid")
        valid_pred, valid_label = self._align_frame_and_label(valid_raw, valid_label)
        valid_pred = self._prepare_prediction_scores(valid_pred)
        valid_folds = self._build_valid_folds(valid_pred.index)

        candidate_weights = list(self._weight_candidates(len(self.model_order)))
        if not candidate_weights:
            raise ValueError("No ensemble weight candidates generated.")

        quick_rows = []
        quick_topk = int(self.topk_grid[0])
        quick_n_drop = int(self.n_drop_grid[0])
        for weight in candidate_weights:
            blend = pd.Series(valid_pred.values @ weight, index=valid_pred.index)
            fold_objs = []
            for fold_dates in valid_folds:
                metrics = self._simulate_portfolio_metrics(
                    blend,
                    valid_label,
                    fold_dates=fold_dates,
                    topk=quick_topk,
                    n_drop=quick_n_drop,
                    memory_boost=0.0,
                    turnover_penalty=0.0,
                    volatility_penalty=0.0,
                )
                fold_objs.append(self._portfolio_objective(metrics))
            quick_score = float(np.mean(fold_objs))
            quick_rows.append((quick_score, weight))
            self.weight_candidate_diagnostics.append(
                {
                    "quick_objective": quick_score,
                    **{f"weight_{name}": float(w) for name, w in zip(self.model_order, weight)},
                }
            )

        quick_rows.sort(key=lambda x: x[0], reverse=True)
        refine_n = min(self.refine_top_weight_candidates, len(quick_rows))
        refined_weights = [w for _, w in quick_rows[:refine_n]]

        best = None
        for weight in refined_weights:
            blend = pd.Series(valid_pred.values @ weight, index=valid_pred.index)
            for topk in self.topk_grid:
                for n_drop in self.n_drop_grid:
                    for memory_boost in self.memory_boost_grid:
                        for turnover_penalty in self.turnover_penalty_grid:
                            for volatility_penalty in self.volatility_penalty_grid:
                                fold_metrics = []
                                for fold_dates in valid_folds:
                                    metrics = self._simulate_portfolio_metrics(
                                        blend,
                                        valid_label,
                                        fold_dates=fold_dates,
                                        topk=int(topk),
                                        n_drop=int(n_drop),
                                        memory_boost=float(memory_boost),
                                        turnover_penalty=float(turnover_penalty),
                                        volatility_penalty=float(volatility_penalty),
                                    )
                                    fold_metrics.append(metrics)
                                mean_metrics = self._mean_metrics(fold_metrics)
                                objective = self._portfolio_objective(mean_metrics)
                                candidate = {
                                    "objective": float(objective),
                                    "weights": weight,
                                    "topk": int(topk),
                                    "n_drop": int(n_drop),
                                    "memory_boost": float(memory_boost),
                                    "turnover_penalty": float(turnover_penalty),
                                    "volatility_penalty": float(volatility_penalty),
                                    "metrics": mean_metrics,
                                }
                                if best is None or candidate["objective"] > best["objective"]:
                                    best = candidate

        if best is None:
            raise ValueError("Validation selection failed to produce a candidate.")

        best_weights = np.clip(np.asarray(best["weights"], dtype=float), 1e-12, None)
        best_weights = best_weights / best_weights.sum()
        self.model_weights = {name: float(w) for name, w in zip(self.model_order, best_weights)}
        self.best_topk = int(best["topk"])
        self.best_n_drop = int(best["n_drop"])
        self.best_memory_boost = float(best["memory_boost"])
        self.best_turnover_penalty = float(best["turnover_penalty"])
        self.best_volatility_penalty = float(best["volatility_penalty"])
        self.validation_summary = {
            "valid_objective": float(best["objective"]),
            "valid_ir": float(best["metrics"]["ir"]),
            "valid_annret": float(best["metrics"]["annret"]),
            "valid_hit_ratio": float(best["metrics"]["hit_ratio"]),
            "valid_turnover": float(best["metrics"]["turnover"]),
            "valid_max_drawdown": float(best["metrics"]["max_drawdown"]),
            "selected_topk": float(self.best_topk),
            "selected_n_drop": float(self.best_n_drop),
            "selected_memory_boost": float(self.best_memory_boost),
            "selected_turnover_penalty": float(self.best_turnover_penalty),
            "selected_volatility_penalty": float(self.best_volatility_penalty),
        }
        self.logger.info("Validation-selected ensemble weights: %s", self.model_weights)
        self.logger.info("Validation-selected execution params: %s", self.validation_summary)
        self.fitted = True

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        raw_pred = self._predict_frame(dataset, segment)
        prepared = self._prepare_prediction_scores(raw_pred)
        blend = self._blend_with_weights(prepared, self.model_weights)
        return self._apply_cost_transform(
            blend,
            topk=self.best_topk,
            n_drop=self.best_n_drop,
            memory_boost=self.best_memory_boost,
            turnover_penalty=self.best_turnover_penalty,
            volatility_penalty=self.best_volatility_penalty,
        )

    def _build_model(self, spec: _BaseLearnerSpec) -> Model:
        if spec.module_path and spec.class_name:
            module = importlib.import_module(spec.module_path)
            cls = getattr(module, spec.class_name)
            return cls(**dict(spec.model_kwargs))

        model_type = spec.model_type.lower()
        kwargs = dict(spec.model_kwargs)
        if model_type in {"double_ensemble", "de"}:
            kwargs.setdefault("random_state", self.random_state)
            return DEnsembleModel(**kwargs)
        if model_type in {"lightgbm", "lgb", "lgbm"}:
            kwargs.setdefault("seed", self.random_state)
            kwargs.setdefault("feature_fraction_seed", self.random_state)
            kwargs.setdefault("bagging_seed", self.random_state)
            kwargs.setdefault("data_random_seed", self.random_state)
            return LGBModel(**kwargs)
        if model_type in {"linear", "lin"}:
            return LinearModel(**kwargs)
        if model_type in {"regime_horizon", "regime"}:
            kwargs.setdefault("random_state", self.random_state)
            return RegimeHorizonCostEnsembleModel(**kwargs)
        if model_type in {"transcendence_hybrid", "hybrid"}:
            kwargs.setdefault("random_state", self.random_state)
            return TranscendenceHybridModel(**kwargs)
        raise ValueError(f"Unsupported model_type: {spec.model_type}")

    def _predict_frame(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.DataFrame:
        preds = {}
        for name in self.model_order:
            preds[name] = self.models[name].predict(dataset, segment)
        frame = pd.DataFrame(preds)
        return frame.replace([np.inf, -np.inf], np.nan)

    def _prepare_prediction_scores(self, pred_frame: pd.DataFrame) -> pd.DataFrame:
        pred_frame = pred_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.use_rank_score:
            pred_frame = self._cross_sectional_rank_score(pred_frame)
        return pred_frame.fillna(0.0)

    def _prepare_label_series(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.Series:
        raw_label = dataset.prepare(segment, col_set=["label"], data_key=DataHandlerLP.DK_L)
        label_df = self._to_label_frame(raw_label)
        label = pd.to_numeric(label_df.iloc[:, 0], errors="coerce")
        return label.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _to_label_frame(raw_label: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if isinstance(raw_label, pd.Series):
            name = "label" if raw_label.name is None else raw_label.name
            return raw_label.to_frame(name)
        if isinstance(raw_label, pd.DataFrame):
            if "label" in raw_label.columns:
                label_obj = raw_label["label"]
                if isinstance(label_obj, pd.DataFrame):
                    return label_obj
                return label_obj.to_frame("label")
            return raw_label
        raise ValueError("Unsupported label object type.")

    @staticmethod
    def _align_frame_and_label(pred_frame: pd.DataFrame, label: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        common = pred_frame.index.intersection(label.index)
        if len(common) == 0:
            raise ValueError("Prediction and label indices do not overlap.")
        pred = pred_frame.loc[common]
        y = label.loc[common]
        mask = np.isfinite(y.values) & np.all(np.isfinite(pred.values), axis=1)
        pred = pred.loc[mask]
        y = y.loc[mask]
        if pred.empty:
            raise ValueError("All aligned rows are invalid after NaN filtering.")
        return pred, y

    def _weight_candidates(self, n_models: int) -> Iterable[np.ndarray]:
        for arr in self._manual_weight_candidates(n_models):
            if self._candidate_satisfies_weight_constraints(arr, n_models):
                yield arr

        if n_models == 1:
            arr = np.array([1.0], dtype=float)
            if self._candidate_satisfies_weight_constraints(arr, n_models):
                yield arr
            return

        yielded = set()

        def _try_emit(arr: np.ndarray):
            key = tuple(np.round(arr, 6).tolist())
            if key in yielded:
                return None
            yielded.add(key)
            if not self._candidate_satisfies_weight_constraints(arr, n_models):
                return None
            return arr

        equal_w = np.ones(n_models, dtype=float) / n_models
        out = _try_emit(equal_w)
        if out is not None:
            yield out

        for i in range(n_models):
            onehot = np.zeros(n_models, dtype=float)
            onehot[i] = 1.0
            out = _try_emit(onehot)
            if out is not None:
                yield out

        if n_models <= 3:
            for arr in self._grid_simplex_weights(n_models, self.search_step):
                out = _try_emit(arr)
                if out is not None:
                    yield out

        remain = max(self.max_random_weight_candidates - len(yielded), 0)
        for _ in range(remain):
            arr = self.rng.dirichlet(np.ones(n_models))
            out = _try_emit(arr)
            if out is not None:
                yield out

    def _candidate_satisfies_weight_constraints(self, arr: np.ndarray, n_models: int) -> bool:
        if not self.weight_constraints:
            return True

        max_aux_weight = self.weight_constraints.get("max_aux_weight")
        if max_aux_weight is None:
            return True

        aux_indices = self.weight_constraints.get("aux_indices")
        if aux_indices is None:
            anchor_index = int(self.weight_constraints.get("anchor_index", 0))
            aux_indices = [idx for idx in range(n_models) if idx != anchor_index]

        limit = float(max_aux_weight)
        for idx in aux_indices:
            if int(idx) < 0 or int(idx) >= n_models:
                raise ValueError(f"weight_constraints aux index {idx} is out of range for {n_models} models.")
            if float(arr[int(idx)]) > limit + 1e-12:
                return False
        return True

    def _manual_weight_candidates(self, n_models: int) -> Iterable[np.ndarray]:
        for raw in self.manual_weight_candidates:
            if isinstance(raw, dict):
                arr = np.array([float(raw.get(name, 0.0)) for name in self.model_order], dtype=float)
            else:
                arr = np.array(list(raw), dtype=float)
            if arr.shape != (n_models,):
                raise ValueError(
                    f"manual_weight_candidates entries must contain {n_models} weights; got shape {arr.shape}."
                )
            if not np.isfinite(arr).all() or (arr < 0).any() or arr.sum() <= 0:
                raise ValueError("manual_weight_candidates values must be finite, non-negative, and sum positive.")
            yield arr / arr.sum()

    @staticmethod
    def _grid_simplex_weights(n_models: int, step: float) -> Iterable[np.ndarray]:
        grid = np.arange(0.0, 1.0 + 1e-9, step)
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

    def _simulate_portfolio_metrics(
        self,
        score: pd.Series,
        label: pd.Series,
        fold_dates: Optional[pd.Index],
        topk: int,
        n_drop: int,
        memory_boost: float,
        turnover_penalty: float,
        volatility_penalty: float,
    ) -> Dict[str, float]:
        common = score.index.intersection(label.index)
        if len(common) == 0:
            return {
                "annret": float("-inf"),
                "ir": float("-inf"),
                "hit_ratio": 0.0,
                "turnover": 1.0,
                "max_drawdown": -1.0,
            }
        s = score.loc[common].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = label.loc[common].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if not isinstance(s.index, pd.MultiIndex):
            corr = s.rank(pct=True).corr(y.rank(pct=True))
            val = 0.0 if corr is None or not np.isfinite(corr) else float(corr)
            return {
                "annret": val,
                "ir": val,
                "hit_ratio": 0.0,
                "turnover": 0.0,
                "max_drawdown": 0.0,
            }

        date_level = self._date_level(s.index)
        inst_level = self._instrument_level(s.index)
        if fold_dates is not None:
            dates = s.index.get_level_values(date_level)
            mask = dates.isin(fold_dates)
            s = s.loc[mask]
            y = y.loc[mask]
        if s.empty:
            return {
                "annret": float("-inf"),
                "ir": float("-inf"),
                "hit_ratio": 0.0,
                "turnover": 1.0,
                "max_drawdown": -1.0,
            }

        daily_net_returns: List[float] = []
        turnover_list: List[float] = []
        prev_holdings = None
        prev_score = None

        for _, day_score_raw in s.groupby(level=date_level, sort=True):
            day_label_raw = y.loc[day_score_raw.index]
            inst_index = pd.Index(day_score_raw.index.get_level_values(inst_level))
            day_score = pd.Series(day_score_raw.values, index=inst_index, dtype=float)
            day_label = pd.Series(day_label_raw.values, index=inst_index, dtype=float)

            vol = self._build_daily_volatility_proxy(day_score, prev_score)
            adj = transform_scores_for_cost(
                day_score,
                previous_holdings=prev_holdings,
                volatility=vol,
                previous_scores=prev_score,
                previous_holding_boost=float(memory_boost),
                turnover_penalty=float(turnover_penalty),
                volatility_penalty=float(volatility_penalty),
                smoothing_alpha=0.0,
                normalize_scores=False,
                use_holding_weight=False,
            )
            selected = self._select_with_n_drop(adj, prev_holdings, topk=int(topk), n_drop=int(n_drop))
            if len(selected) == 0:
                continue
            gross_ret = float(day_label.loc[selected].mean())
            prev_inst = set(prev_holdings.index.tolist()) if isinstance(prev_holdings, pd.Series) else set()
            curr_inst = set(selected.tolist())
            overlap = len(prev_inst.intersection(curr_inst)) if prev_inst else 0
            replacement_ratio = 0.0 if not prev_inst else (1.0 - overlap / max(len(curr_inst), 1))
            two_sided_turnover = min(2.0 * replacement_ratio, 2.0)
            tx_cost = two_sided_turnover * (self.open_cost + self.close_cost) * 0.5
            net_ret = gross_ret - tx_cost

            daily_net_returns.append(float(net_ret))
            turnover_list.append(float(two_sided_turnover))
            prev_holdings = pd.Series(1.0, index=selected)
            prev_score = day_score

        if not daily_net_returns:
            return {
                "annret": float("-inf"),
                "ir": float("-inf"),
                "hit_ratio": 0.0,
                "turnover": 1.0,
                "max_drawdown": -1.0,
            }

        arr = np.asarray(daily_net_returns, dtype=float)
        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr))
        annret = mean_ret * self.ann_scaler
        ir = (mean_ret / (std_ret + 1e-12)) * np.sqrt(self.ann_scaler)
        hit_ratio = float(np.mean(arr > 0))
        turnover = float(np.mean(turnover_list)) if turnover_list else 0.0
        equity = pd.Series(1.0 + arr).cumprod()
        running_max = equity.cummax()
        drawdown = (equity / (running_max + 1e-12) - 1.0).min()
        max_drawdown = float(drawdown if np.isfinite(drawdown) else 0.0)
        return {
            "annret": float(annret),
            "ir": float(ir),
            "hit_ratio": float(hit_ratio),
            "turnover": float(turnover),
            "max_drawdown": float(max_drawdown),
        }

    def _portfolio_objective(self, metrics: Dict[str, float]) -> float:
        maxdd = float(metrics["max_drawdown"])
        dd_penalty = max(0.0, -maxdd) * self.max_drawdown_penalty
        return (
            self.ir_weight * float(metrics["ir"])
            + self.annret_weight * float(metrics["annret"])
            + self.hit_ratio_weight * float(metrics["hit_ratio"])
            - dd_penalty
        )

    @staticmethod
    def _mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
        keys = ["annret", "ir", "hit_ratio", "turnover", "max_drawdown"]
        out = {}
        for k in keys:
            vals = [float(x[k]) for x in rows]
            out[k] = float(np.mean(vals))
        return out

    def _apply_cost_transform(
        self,
        score: pd.Series,
        topk: int,
        n_drop: int,
        memory_boost: float,
        turnover_penalty: float,
        volatility_penalty: float,
    ) -> pd.Series:
        score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if not isinstance(score.index, pd.MultiIndex):
            return score

        date_level = self._date_level(score.index)
        inst_level = self._instrument_level(score.index)
        out_parts = []
        prev_holdings = None
        prev_score = None

        for _, day_score_raw in score.groupby(level=date_level, sort=True):
            inst_index = pd.Index(day_score_raw.index.get_level_values(inst_level))
            day_score = pd.Series(day_score_raw.values, index=inst_index, dtype=float)
            vol = self._build_daily_volatility_proxy(day_score, prev_score)
            adj = transform_scores_for_cost(
                day_score,
                previous_holdings=prev_holdings,
                volatility=vol,
                previous_scores=prev_score,
                previous_holding_boost=float(memory_boost),
                turnover_penalty=float(turnover_penalty),
                volatility_penalty=float(volatility_penalty),
                smoothing_alpha=0.0,
                normalize_scores=False,
                use_holding_weight=False,
            )
            selected = self._select_with_n_drop(adj, prev_holdings, topk=int(topk), n_drop=int(n_drop))
            # Keep full cross-section scores while boosting stability among selected names.
            out_parts.append(pd.Series(adj.values, index=day_score_raw.index))
            prev_holdings = pd.Series(1.0, index=selected)
            prev_score = day_score

        if not out_parts:
            return score
        out = pd.concat(out_parts).sort_index()
        if self.use_rank_score:
            out = self._cross_sectional_rank_score(out)
        return out.fillna(0.0)

    @staticmethod
    def _build_daily_volatility_proxy(day_score: pd.Series, previous_score: Optional[pd.Series]) -> pd.Series:
        if previous_score is None:
            vol = (day_score - float(day_score.mean())).abs()
        else:
            prev = previous_score.reindex(day_score.index)
            prev = prev.fillna(float(day_score.mean()) if len(day_score) else 0.0)
            vol = (day_score - prev).abs()
        return vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _select_with_n_drop(
        score: pd.Series, prev_holdings: Optional[pd.Series], topk: int, n_drop: int
    ) -> pd.Index:
        ranked = score.sort_values(ascending=False)
        if ranked.empty:
            return pd.Index([])
        topk = max(int(topk), 1)
        if prev_holdings is None or prev_holdings.empty:
            return ranked.index[: min(topk, len(ranked))]

        prev_names = [x for x in prev_holdings.index.tolist() if x in ranked.index]
        if not prev_names:
            return ranked.index[: min(topk, len(ranked))]

        n_drop = max(int(n_drop), 0)
        n_drop = min(n_drop, len(prev_names), topk)
        hold_scores = score.reindex(prev_names).fillna(float(score.min()) - 1.0)
        drop_names = set(hold_scores.nsmallest(n_drop).index.tolist())
        kept = [x for x in prev_names if x not in drop_names]

        selected = list(kept)
        selected_set = set(selected)
        for inst in ranked.index:
            if inst in selected_set:
                continue
            selected.append(inst)
            selected_set.add(inst)
            if len(selected) >= topk:
                break
        return pd.Index(selected[:topk])

    def _blend_with_weights(self, pred_frame: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        if pred_frame.empty:
            return pd.Series(dtype=float)
        w = np.array([weights.get(c, 0.0) for c in pred_frame.columns], dtype=float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(pred_frame.columns), dtype=float) / len(pred_frame.columns)
        return pd.Series(pred_frame.values @ w, index=pred_frame.index).fillna(0.0)

    @staticmethod
    def _build_valid_folds(index: pd.Index) -> List[Optional[pd.Index]]:
        if not isinstance(index, pd.MultiIndex):
            return [None]
        date_level = "datetime" if "datetime" in index.names else index.names[0]
        dates = pd.Index(index.get_level_values(date_level).unique()).sort_values()
        years = pd.Index(pd.to_datetime(dates).year).unique().tolist()
        if len(years) <= 1:
            return [None]
        folds = []
        for y in years:
            fold_dates = dates[pd.to_datetime(dates).year == y]
            folds.append(pd.Index(fold_dates))
        return folds

    @staticmethod
    def _cross_sectional_rank_score(values: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(values.index, pd.MultiIndex):
            return values.rank(pct=True).fillna(0.0)
        date_level = "datetime" if "datetime" in values.index.names else values.index.names[0]
        return values.groupby(level=date_level).rank(pct=True).fillna(0.0)

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("A MultiIndex index is required.")
        return "datetime" if "datetime" in index.names else index.names[0]

    @staticmethod
    def _instrument_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("A MultiIndex index is required.")
        return "instrument" if "instrument" in index.names else index.names[-1]

    def _parse_specs(self, specs: Optional[Sequence[Dict]]) -> List[_BaseLearnerSpec]:
        if specs is None:
            specs = [
                {
                    "name": "regime_de_7406",
                    "model_type": "regime_horizon",
                    "model_kwargs": {
                        "horizon_model_specs": [
                            {
                                "name": "de_h1",
                                "model_type": "double_ensemble",
                                "horizon": 1,
                                "model_kwargs": {
                                    "base_model": "gbm",
                                    "loss": "mse",
                                    "num_models": 3,
                                    "enable_sr": True,
                                    "enable_fs": True,
                                    "alpha1": 1,
                                    "alpha2": 1,
                                    "bins_sr": 10,
                                    "bins_fs": 5,
                                    "decay": 0.5,
                                    "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4],
                                    "sub_weights": [1, 1, 1],
                                    "epochs": 28,
                                    "colsample_bytree": 0.8879,
                                    "learning_rate": 0.2,
                                    "subsample": 0.8789,
                                    "lambda_l1": 205.6999,
                                    "lambda_l2": 580.9768,
                                    "max_depth": 8,
                                    "num_leaves": 210,
                                    "num_threads": 20,
                                    "verbosity": -1,
                                    "random_state": self.random_state,
                                },
                            }
                        ],
                        "topk": 45,
                        "search_step": 0.1,
                        "turnover_penalty": 0.00005,
                        "risk_penalty": 0.0,
                        "memory_boost_grid": [0.0, 0.005],
                        "use_rank_score": False,
                        "zscore_clip": 100.0,
                        "neutralize_daily_mean": False,
                        "enforce_horizon_monotonic": False,
                        "random_state": self.random_state,
                    },
                },
                {
                    "name": "de_baseline",
                    "model_type": "double_ensemble",
                    "model_kwargs": {
                        "base_model": "gbm",
                        "loss": "mse",
                        "num_models": 3,
                        "enable_sr": True,
                        "enable_fs": True,
                        "alpha1": 1,
                        "alpha2": 1,
                        "bins_sr": 10,
                        "bins_fs": 5,
                        "decay": 0.5,
                        "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4],
                        "sub_weights": [1, 1, 1],
                        "epochs": 28,
                        "colsample_bytree": 0.8879,
                        "learning_rate": 0.2,
                        "subsample": 0.8789,
                        "lambda_l1": 205.6999,
                        "lambda_l2": 580.9768,
                        "max_depth": 8,
                        "num_leaves": 210,
                        "num_threads": 20,
                        "verbosity": -1,
                        "random_state": self.random_state,
                    },
                },
                {
                    "name": "lgb_aux",
                    "model_type": "lightgbm",
                    "model_kwargs": {
                        "loss": "mse",
                        "early_stopping_rounds": 60,
                        "num_boost_round": 450,
                        "learning_rate": 0.04,
                        "colsample_bytree": 0.86,
                        "subsample": 0.86,
                        "lambda_l1": 6.0,
                        "lambda_l2": 100.0,
                        "max_depth": 8,
                        "min_data_in_leaf": 120,
                        "num_leaves": 127,
                        "num_threads": 12,
                    },
                },
                {
                    "name": "lin_aux",
                    "model_type": "linear",
                    "model_kwargs": {"estimator": "ridge", "alpha": 0.8, "fit_intercept": False},
                },
            ]

        parsed: List[_BaseLearnerSpec] = []
        used = set()
        for i, raw in enumerate(specs):
            name = str(raw.get("name", f"base_{i}"))
            if name in used:
                raise ValueError(f"Duplicate base learner name: {name}")
            used.add(name)
            parsed.append(
                _BaseLearnerSpec(
                    name=name,
                    model_type=str(raw.get("model_type", "double_ensemble")),
                    model_kwargs=dict(raw.get("model_kwargs", {})),
                    class_name=(str(raw["class"]) if "class" in raw else None),
                    module_path=(str(raw["module_path"]) if "module_path" in raw else None),
                )
            )
        if not parsed:
            raise ValueError("base_learner_specs must contain at least one spec.")
        return parsed
