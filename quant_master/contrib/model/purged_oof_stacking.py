"""Purged walk-forward out-of-fold stacking for heterogeneous QuantMaster models."""

from copy import deepcopy
import gc
from typing import Dict, List, Optional, Text, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.loader import StaticDataLoader
from ...log import get_module_logger
from ...model.base import Model
from ...utils import auto_filter_kwargs, init_instance_by_config


class PurgedWalkForwardStackingModel(Model):
    """Learn non-negative ridge blend weights exclusively from purged OOF predictions.

    Each fold has three distinct segments: ``train`` for fitting preprocessing and
    model parameters, ``valid`` for base-model early stopping, and ``oof`` for the
    meta learner. A fresh dataset/handler is built for every fold.
    """

    def __init__(
        self,
        base_models: Dict[str, dict],
        folds: List[dict],
        fold_dataset_config: dict,
        purge_periods: int = 5,
        embargo_periods: int = 0,
        ridge_alpha: float = 1.0,
        rank_normalize: bool = True,
        selection_end_time: str = "2023-12-31",
        refit_full: bool = True,
        final_test_purge_periods: int = 0,
        require_test_after_selection: bool = False,
        reuse_final_raw_data: bool = False,
        weight_half_life_periods: Optional[int] = None,
        max_model_weight: float = 1.0,
    ):
        if len(base_models) < 2:
            raise ValueError("OOF stacking requires at least two heterogeneous base models.")
        if not folds:
            raise ValueError("At least one walk-forward fold is required.")
        if purge_periods < 0 or embargo_periods < 0 or final_test_purge_periods < 0:
            raise ValueError("purge_periods and embargo_periods must be non-negative.")
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative.")
        if weight_half_life_periods is not None:
            normalized_half_life = int(weight_half_life_periods)
            if normalized_half_life <= 0 or normalized_half_life != weight_half_life_periods:
                raise ValueError("weight_half_life_periods must be a positive integer when configured.")
        if not 0 < max_model_weight <= 1:
            raise ValueError("max_model_weight must be in (0, 1].")
        if max_model_weight * len(base_models) < 1:
            raise ValueError("max_model_weight is too small for simplex-constrained model weights.")
        if not isinstance(fold_dataset_config, dict):
            raise TypeError("fold_dataset_config must be a declarative dataset config.")

        self.logger = get_module_logger(self.__class__.__name__)
        self.base_model_configs = deepcopy(base_models)
        self.model_order = list(base_models)
        self.folds = deepcopy(folds)
        self.fold_dataset_config = deepcopy(fold_dataset_config)
        self.purge_periods = int(purge_periods)
        self.embargo_periods = int(embargo_periods)
        self.ridge_alpha = float(ridge_alpha)
        self.rank_normalize = bool(rank_normalize)
        self.selection_end_time = pd.Timestamp(selection_end_time)
        self.refit_full = bool(refit_full)
        self.final_test_purge_periods = int(final_test_purge_periods)
        self.require_test_after_selection = bool(require_test_after_selection)
        self.reuse_final_raw_data = bool(reuse_final_raw_data)
        self.weight_half_life_periods = normalized_half_life if weight_half_life_periods is not None else None
        self.max_model_weight = float(max_model_weight)
        self._shared_raw_data = None

        self.model_weights: Dict[str, float] = {}
        self.base_models_: Dict[str, Model] = {}
        self.oof_predictions_: Optional[pd.DataFrame] = None
        self.oof_labels_: Optional[pd.Series] = None
        self.causal_oof_predictions_: Optional[pd.Series] = None
        self.fold_diagnostics_: List[dict] = []

    def fit(self, dataset: DatasetH, reweighter=None):
        self._validate_final_refit_segments(dataset)
        final_calendar = self._calendar_from_dataset(dataset)
        if self.reuse_final_raw_data:
            raw_data = getattr(dataset.handler, "_data", None)
            if not isinstance(raw_data, pd.DataFrame):
                raise ValueError("reuse_final_raw_data=True requires the final handler to retain a raw DataFrame.")
            self._shared_raw_data = raw_data
        oof_predictions = []
        oof_labels = []
        causal_oof_predictions = []
        diagnostics = []
        previous_oof_end = None

        for fold_number, fold in enumerate(self.folds):
            normalized_fold = self._normalize_fold(fold, fold_number)
            if previous_oof_end is not None and pd.Timestamp(normalized_fold["oof"][0]) <= pd.Timestamp(
                previous_oof_end
            ):
                raise ValueError("OOF folds must be strictly increasing and non-overlapping.")
            embargo_actual = None
            if previous_oof_end is not None:
                embargo_actual = self._periods_between(
                    final_calendar, previous_oof_end, normalized_fold["oof"][0]
                )
                if embargo_actual < self.embargo_periods:
                    raise ValueError(
                        f"Fold {fold_number} has only {embargo_actual} embargo periods; "
                        f"{self.embargo_periods} are required."
                    )
            fold_dataset = self._build_fold_dataset(normalized_fold)
            calendar = self._calendar_from_dataset(fold_dataset)
            train_valid_purge_actual = self._periods_between(
                calendar, normalized_fold["train"][1], normalized_fold["valid"][0]
            )
            if train_valid_purge_actual < self.purge_periods:
                raise ValueError(
                    f"Fold {fold_number} has only {train_valid_purge_actual} purge periods between train and valid; "
                    f"{self.purge_periods} are required."
                )
            valid_oof_purge_actual = self._periods_between(
                calendar, normalized_fold["valid"][1], normalized_fold["oof"][0]
            )
            if valid_oof_purge_actual < self.purge_periods:
                raise ValueError(
                    f"Fold {fold_number} has only {valid_oof_purge_actual} purge periods between valid and OOF; "
                    f"{self.purge_periods} are required."
                )
            prediction_frame = {}
            for model_name, model_config in self.base_model_configs.items():
                model = init_instance_by_config(deepcopy(model_config), accept_types=Model)
                auto_filter_kwargs(model.fit)(fold_dataset, reweighter=reweighter)
                prediction_frame[model_name] = self._as_series(model.predict(fold_dataset, "test"), model_name)

            prediction_frame = pd.DataFrame(prediction_frame).sort_index()
            label = self._fold_label(fold_dataset)
            common_index = prediction_frame.index.intersection(label.index)
            prediction_frame = prediction_frame.loc[common_index]
            label = label.loc[common_index]
            if prediction_frame.empty:
                raise ValueError(f"Fold {fold_number} produced no aligned OOF predictions.")

            if oof_predictions:
                prior_predictions = pd.concat(oof_predictions).sort_index()
                prior_labels = pd.concat(oof_labels).sort_index()
                causal_weights = self._learn_weights(prior_predictions, prior_labels)
            else:
                equal_weight = 1.0 / len(self.model_order)
                causal_weights = {name: equal_weight for name in self.model_order}
            causal_blend = self._prediction_scores(prediction_frame).loc[:, self.model_order].dot(
                pd.Series(causal_weights).reindex(self.model_order)
            )

            fold_diag = {
                "fold": fold_number,
                "train": normalized_fold["train"],
                "valid": normalized_fold["valid"],
                "oof": normalized_fold["oof"],
                "rows": len(prediction_frame),
                "dates": prediction_frame.index.get_level_values("datetime").nunique(),
                "train_valid_purge_periods": train_valid_purge_actual,
                "valid_oof_purge_periods": valid_oof_purge_actual,
                "embargo_periods": embargo_actual,
                "stack_weights": causal_weights,
                "causal_stack": self._rank_ic_summary(causal_blend, label),
                "models": {
                    name: self._rank_ic_summary(prediction_frame[name], label)
                    for name in self.model_order
                },
            }
            diagnostics.append(fold_diag)
            oof_predictions.append(prediction_frame)
            oof_labels.append(label)
            causal_oof_predictions.append(causal_blend.rename("score"))
            previous_oof_end = normalized_fold["oof"][1]

        combined_predictions = pd.concat(oof_predictions).sort_index()
        combined_labels = pd.concat(oof_labels).sort_index()
        if combined_predictions.index.has_duplicates or combined_labels.index.has_duplicates:
            raise ValueError("OOF folds overlap; each meta-learning row must be predicted exactly once.")

        self.oof_predictions_ = combined_predictions
        self.oof_labels_ = combined_labels
        self.causal_oof_predictions_ = pd.concat(causal_oof_predictions).sort_index()
        self.fold_diagnostics_ = diagnostics
        self.model_weights = self._learn_weights(combined_predictions, combined_labels)
        del fold_dataset
        self._shared_raw_data = None
        gc.collect()
        if self.refit_full:
            if self.reuse_final_raw_data and (
                not hasattr(dataset.handler, "_infer") or not hasattr(dataset.handler, "_learn")
            ):
                dataset.handler.fit_process_data()
            self.base_models_ = {}
            for model_name, model_config in self.base_model_configs.items():
                model = init_instance_by_config(deepcopy(model_config), accept_types=Model)
                auto_filter_kwargs(model.fit)(dataset, reweighter=reweighter)
                self.base_models_[model_name] = model
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.model_weights or not self.base_models_:
            raise ValueError("Model must be fitted with refit_full=True before prediction.")
        frame = pd.DataFrame(
            {
                name: self._as_series(model.predict(dataset, segment), name)
                for name, model in self.base_models_.items()
            }
        )
        scores = self._prediction_scores(frame)
        weights = pd.Series(self.model_weights).reindex(self.model_order)
        return scores.loc[:, self.model_order].dot(weights).rename("score")

    def get_oof_diagnostics(self):
        if self.oof_predictions_ is None:
            raise ValueError("Model is not fitted yet.")
        fitted_blend = self._prediction_scores(self.oof_predictions_).dot(
            pd.Series(self.model_weights).reindex(self.model_order)
        )
        causal_blend = self.causal_oof_predictions_
        yearly = {}
        years = self.oof_predictions_.index.get_level_values("datetime").year
        for year in sorted(set(years)):
            mask = years == year
            yearly[str(year)] = self._rank_ic_summary(causal_blend.loc[mask], self.oof_labels_.loc[mask])
        return {
            "selection_end_time": str(self.selection_end_time.date()),
            "weight_half_life_periods": getattr(self, "weight_half_life_periods", None),
            "max_model_weight": getattr(self, "max_model_weight", 1.0),
            "weights": dict(self.model_weights),
            "folds": deepcopy(self.fold_diagnostics_),
            "oof_rows": len(self.oof_predictions_),
            "oof_dates": self.oof_predictions_.index.get_level_values("datetime").nunique(),
            "overall": self._rank_ic_summary(causal_blend, self.oof_labels_),
            "fitted_oof": self._rank_ic_summary(fitted_blend, self.oof_labels_),
            "yearly": yearly,
        }

    def assert_oof_gate(
        self,
        min_rank_ic: float = 0.04,
        min_rank_icir: float = 0.25,
        require_positive_years: bool = True,
    ):
        diagnostics = self.get_oof_diagnostics()
        failures = []
        overall = diagnostics["overall"]
        if not np.isfinite(overall["rank_ic"]) or overall["rank_ic"] < float(min_rank_ic):
            failures.append(f"overall RankIC {overall['rank_ic']:.6f} < {float(min_rank_ic):.6f}")
        if not np.isfinite(overall["rank_icir"]) or overall["rank_icir"] < float(min_rank_icir):
            failures.append(f"overall RankICIR {overall['rank_icir']:.6f} < {float(min_rank_icir):.6f}")
        if require_positive_years:
            nonpositive = [year for year, values in diagnostics["yearly"].items() if values["rank_ic"] <= 0]
            if nonpositive:
                failures.append(f"non-positive yearly RankIC: {', '.join(nonpositive)}")
        if failures:
            raise ValueError("OOF prediction gate failed: " + "; ".join(failures))
        return diagnostics

    def _build_fold_dataset(self, fold):
        config = deepcopy(self.fold_dataset_config)
        kwargs = config.setdefault("kwargs", {})
        kwargs["segments"] = {"train": fold["train"], "valid": fold["valid"], "test": fold["oof"]}
        handler_kwargs = kwargs.get("handler", {}).setdefault("kwargs", {})
        handler_kwargs["fit_start_time"] = fold["train"][0]
        handler_kwargs["fit_end_time"] = fold["train"][1]
        handler_kwargs["end_time"] = fold["oof"][1]
        if self.reuse_final_raw_data:
            if self._shared_raw_data is None:
                raise ValueError("Shared final raw data is unavailable while constructing a fold dataset.")
            handler_kwargs["instruments"] = None
            handler_kwargs["data_loader"] = StaticDataLoader(config=self._shared_raw_data)
        return init_instance_by_config(config, accept_types=DatasetH)

    def _validate_final_refit_segments(self, dataset):
        for name in ("train", "valid"):
            if name in dataset.segments and pd.Timestamp(dataset.segments[name][1]) > self.selection_end_time:
                raise ValueError(f"Final {name} segment exceeds selection_end_time {self.selection_end_time.date()}.")
        if "valid" in dataset.segments and "test" in dataset.segments:
            valid_end = pd.Timestamp(dataset.segments["valid"][1])
            test_start = pd.Timestamp(dataset.segments["test"][0])
            if test_start <= valid_end:
                raise ValueError("Final valid and test segments must be strictly ordered and non-overlapping.")
            calendar = self._calendar_from_dataset(dataset)
            actual_purge = self._periods_between(calendar, valid_end, test_start)
            if actual_purge < self.final_test_purge_periods:
                raise ValueError(
                    f"Final split has only {actual_purge} purge periods between valid and test; "
                    f"{self.final_test_purge_periods} are required."
                )
            if self.require_test_after_selection and test_start <= self.selection_end_time:
                raise ValueError(
                    f"Final test must start after selection_end_time {self.selection_end_time.date()}."
                )

    def _normalize_fold(self, fold, fold_number):
        missing = {"train", "valid", "oof"}.difference(fold)
        if missing:
            raise ValueError(f"Fold {fold_number} is missing segments: {sorted(missing)}")
        result = {name: tuple(fold[name]) for name in ("train", "valid", "oof")}
        train_start, train_end = map(pd.Timestamp, result["train"])
        valid_start, valid_end = map(pd.Timestamp, result["valid"])
        oof_start, oof_end = map(pd.Timestamp, result["oof"])
        if not (train_start <= train_end < valid_start <= valid_end < oof_start <= oof_end):
            raise ValueError(f"Fold {fold_number} must be ordered train -> valid -> purge -> OOF.")
        if oof_end > self.selection_end_time:
            raise ValueError(f"Fold {fold_number} OOF ends after selection_end_time {self.selection_end_time.date()}.")
        return result

    @staticmethod
    def _calendar_from_dataset(dataset):
        data = getattr(dataset.handler, "_data", None)
        if data is None or not isinstance(data.index, pd.MultiIndex) or "datetime" not in data.index.names:
            raise ValueError("Fold dataset must expose its loaded datetime index for purge validation.")
        return pd.DatetimeIndex(data.index.get_level_values("datetime").unique()).sort_values()

    @staticmethod
    def _periods_between(calendar, left, right):
        left, right = pd.Timestamp(left), pd.Timestamp(right)
        return int(((calendar > left) & (calendar < right)).sum())

    @staticmethod
    def _fold_label(dataset):
        label = dataset.prepare("test", col_set=["label"], data_key=DataHandlerLP.DK_R)["label"]
        if isinstance(label, pd.DataFrame):
            if label.shape[1] != 1:
                raise ValueError("OOF stacking supports one label column.")
            label = label.iloc[:, 0]
        return label.rename("label")

    @staticmethod
    def _as_series(prediction, name):
        if isinstance(prediction, pd.DataFrame):
            if prediction.shape[1] != 1:
                raise ValueError(f"Base model {name} returned multiple prediction columns.")
            prediction = prediction.iloc[:, 0]
        if not isinstance(prediction, pd.Series):
            raise TypeError(f"Base model {name} must return a pandas Series or one-column DataFrame.")
        return prediction.rename(name)

    def _learn_weights(self, predictions, label):
        scores = self._prediction_scores(predictions).loc[:, self.model_order]
        frame = scores.copy()
        frame["__label"] = label.reindex(scores.index)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) <= len(self.model_order):
            raise ValueError("Not enough finite OOF rows to fit stacking weights.")

        date_level = "datetime" if "datetime" in frame.index.names else frame.index.names[0]
        grouped = frame.groupby(level=date_level, sort=False)
        x_frame = frame.loc[:, self.model_order] - grouped[self.model_order].transform("mean")
        ranked_label = grouped["__label"].rank(pct=True)
        y_series = ranked_label - ranked_label.groupby(level=date_level, sort=False).transform("mean")
        day_sizes = y_series.groupby(level=date_level, sort=False).transform("size").astype(float)
        row_weights = np.ones(len(frame), dtype=float)
        weight_half_life_periods = getattr(self, "weight_half_life_periods", None)
        if weight_half_life_periods is not None:
            row_dates = pd.DatetimeIndex(frame.index.get_level_values(date_level))
            unique_dates = row_dates.unique().sort_values()
            age = len(unique_dates) - 1 - unique_dates.get_indexer(row_dates)
            row_weights = np.exp2(-age / weight_half_life_periods)
            # Keep one trading day as one observation and preserve the scale of ridge_alpha.
            daily_weights = pd.Series(row_weights, index=frame.index).groupby(level=date_level).first()
            row_weights /= float(daily_weights.mean())
        row_scale = np.sqrt(row_weights / day_sizes.to_numpy())
        x = x_frame.to_numpy(dtype=float) * row_scale[:, None]
        y = y_series.to_numpy(dtype=float) * row_scale
        day_count = max(1, frame.index.get_level_values(date_level).nunique())
        gram = x.T @ x / day_count
        cross = x.T @ y / day_count
        regularized_gram = gram + self.ridge_alpha * np.eye(x.shape[1])

        def objective(weights):
            return 0.5 * weights @ regularized_gram @ weights - cross @ weights

        def gradient(weights):
            return regularized_gram @ weights - cross

        initial = np.full(len(self.model_order), 1.0 / len(self.model_order), dtype=float)
        max_model_weight = getattr(self, "max_model_weight", 1.0)
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, max_model_weight)] * len(self.model_order),
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"ftol": 1e-12, "maxiter": 500},
        )
        weights = result.x if result.success and np.isfinite(result.x).all() else initial
        weights = np.clip(weights, 0.0, max_model_weight)
        weights /= weights.sum()
        return {name: float(weight) for name, weight in zip(self.model_order, weights)}

    def _prediction_scores(self, predictions):
        if not self.rank_normalize:
            return predictions
        if not isinstance(predictions.index, pd.MultiIndex) or "datetime" not in predictions.index.names:
            raise ValueError("rank_normalize=True requires a MultiIndex with a datetime level.")
        return predictions.groupby(level="datetime").rank(pct=True)

    @staticmethod
    def _daily_rank_ic(prediction, label):
        frame = pd.concat([prediction.rename("prediction"), label.rename("label")], axis=1).dropna()
        return frame.groupby(level="datetime").apply(
            lambda group: group["prediction"].rank().corr(group["label"].rank())
        )

    @classmethod
    def _rank_ic_summary(cls, prediction, label):
        daily = cls._daily_rank_ic(prediction, label).replace([np.inf, -np.inf], np.nan).dropna()
        mean = float(daily.mean()) if len(daily) else float("nan")
        std = float(daily.std()) if len(daily) > 1 else float("nan")
        rank_icir = mean / std if np.isfinite(std) and std > 0 else float("nan")
        return {"rank_ic": mean, "rank_icir": rank_icir, "days": int(len(daily))}
