# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Text, Tuple, Union

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover - guarded by runtime checks
    lgb = None

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from .double_ensemble import DEnsembleModel
from .gbdt import LGBModel
from .linear import LinearModel


@dataclass
class _BaseModelSpec:
    name: str
    model_type: str
    model_kwargs: Dict


class TranscendenceHybridModel(Model):
    """Hybrid model with rank ensemble, validation portfolio objective and optional residual/deep branches."""

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
    ):
        if topk <= 0:
            raise ValueError("topk must be positive.")
        if not 0 < search_step <= 1:
            raise ValueError("search_step must be in (0, 1].")
        if max_random_weight_candidates <= 0:
            raise ValueError("max_random_weight_candidates must be positive.")
        if ann_scaler <= 0:
            raise ValueError("ann_scaler must be positive.")

        self.logger = get_module_logger("TranscendenceHybridModel")
        self.use_rank_score = bool(use_rank_score)
        self.topk = int(topk)
        self.search_step = float(search_step)
        self.max_random_weight_candidates = int(max_random_weight_candidates)
        self.ir_weight = float(ir_weight)
        self.annret_weight = float(annret_weight)
        self.ann_scaler = int(ann_scaler)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)

        self.specs = self._parse_specs(base_model_specs)
        self.residual_cfg = dict(residual_learner or {})
        self.deep_cfg = dict(deep_branch or {})

        self.models: Dict[str, Model] = {}
        self.model_order: List[str] = []
        self.model_weights: Dict[str, float] = {}
        self.residual_model = None
        self.residual_weight = 0.0
        self.residual_use_rank = True
        self.deep_branch_active = False
        self.deep_branch_message = "disabled"
        self.validation_summary: Dict[str, float] = {}
        self.fitted = False

    def fit(self, dataset: DatasetH):
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

        valid_label = self._prepare_label_series(dataset, "valid")
        valid_pred_raw = self._predict_frame(dataset, "valid")
        valid_pred, valid_label = self._align_frame_and_label(valid_pred_raw, valid_label)
        valid_pred = self._prepare_prediction_scores(valid_pred)

        self.model_weights = self._learn_weights(valid_pred, valid_label)
        self.logger.info("Learned base weights: %s", self.model_weights)

        valid_blend = self._blend_with_weights(valid_pred, self.model_weights)
        valid_ir, valid_annret, valid_hit = self._portfolio_metrics(valid_blend, valid_label)
        self.validation_summary = {
            "valid_ir": float(valid_ir),
            "valid_annret": float(valid_annret),
            "valid_hit_ratio": float(valid_hit),
        }

        self._fit_optional_residual(dataset)
        self.fitted = True

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        raw_pred = self._predict_frame(dataset, segment)
        prepared = self._prepare_prediction_scores(raw_pred)
        base_score = self._blend_with_weights(prepared, self.model_weights)
        final_score = base_score

        if self.residual_model is not None and self.residual_weight != 0.0:
            features = self._prepare_feature_frame(dataset, segment)
            residual_raw = pd.Series(self.residual_model.predict(features.values), index=features.index)
            residual_raw = self._clean_series(residual_raw)
            if self.residual_use_rank:
                residual_raw = self._cross_sectional_rank_score(residual_raw)
            aligned = base_score.index.intersection(residual_raw.index)
            final_score = base_score.loc[aligned] + self.residual_weight * residual_raw.loc[aligned]
            final_score = final_score.reindex(base_score.index).fillna(0.0)
        return final_score

    def _fit_optional_residual(self, dataset: DatasetH):
        enabled = bool(self.residual_cfg.get("enabled", True))
        if not enabled:
            self.logger.info("Residual learner disabled.")
            return
        if lgb is None:
            self.logger.warning("Residual learner disabled because lightgbm is unavailable.")
            return

        train_feature = self._prepare_feature_frame(dataset, "train")
        valid_feature = self._prepare_feature_frame(dataset, "valid")
        train_label = self._prepare_label_series(dataset, "train")
        valid_label = self._prepare_label_series(dataset, "valid")

        train_base = self._blend_segment(dataset, "train")
        valid_base = self._blend_segment(dataset, "valid")

        common_train = train_feature.index.intersection(train_label.index).intersection(train_base.index)
        common_valid = valid_feature.index.intersection(valid_label.index).intersection(valid_base.index)
        if len(common_train) == 0 or len(common_valid) == 0:
            self.logger.warning("Residual learner skipped due to empty aligned train/valid data.")
            return

        x_train = train_feature.loc[common_train]
        y_train = train_label.loc[common_train] - train_base.loc[common_train]
        x_valid = valid_feature.loc[common_valid]
        y_valid = valid_label.loc[common_valid] - valid_base.loc[common_valid]

        mask_train = np.isfinite(y_train.values) & np.all(np.isfinite(x_train.values), axis=1)
        mask_valid = np.isfinite(y_valid.values) & np.all(np.isfinite(x_valid.values), axis=1)
        x_train = x_train.iloc[mask_train]
        y_train = y_train.iloc[mask_train]
        x_valid = x_valid.iloc[mask_valid]
        y_valid = y_valid.iloc[mask_valid]
        if x_train.empty or x_valid.empty:
            self.logger.warning("Residual learner skipped because all aligned rows became invalid after NaN filtering.")
            return

        params = {
            "objective": "mse",
            "verbosity": -1,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "max_depth": 8,
            "min_data_in_leaf": 150,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l1": 5.0,
            "lambda_l2": 80.0,
            "num_threads": 8,
            "seed": self.random_state,
            "feature_fraction_seed": self.random_state,
            "bagging_seed": self.random_state,
            "data_random_seed": self.random_state,
        }
        params.update(dict(self.residual_cfg.get("lgb_params", {})))

        num_boost_round = int(self.residual_cfg.get("num_boost_round", 300))
        early_stopping_rounds = int(self.residual_cfg.get("early_stopping_rounds", 40))
        self.residual_use_rank = bool(self.residual_cfg.get("use_rank_score", self.use_rank_score))
        weight_grid = list(self.residual_cfg.get("weight_grid", [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]))

        self.logger.info("Training residual LightGBM learner.")
        dtrain = lgb.Dataset(x_train.values, label=y_train.values, free_raw_data=False)
        dvalid = lgb.Dataset(x_valid.values, label=y_valid.values, free_raw_data=False)
        self.residual_model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(50)],
        )

        residual_valid = pd.Series(self.residual_model.predict(x_valid.values), index=x_valid.index)
        residual_valid = self._clean_series(residual_valid)
        if self.residual_use_rank:
            residual_valid = self._cross_sectional_rank_score(residual_valid)

        base_valid = valid_base.loc[common_valid].reindex(x_valid.index).fillna(0.0)
        label_valid = valid_label.loc[common_valid].reindex(x_valid.index)
        best_weight = 0.0
        best_objective = None
        for weight in weight_grid:
            blend = base_valid + float(weight) * residual_valid
            objective = self._portfolio_objective(blend, label_valid)
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_weight = float(weight)
        self.residual_weight = best_weight
        self.logger.info("Residual learner weight selected: %s", self.residual_weight)

    def _blend_segment(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.Series:
        raw_pred = self._predict_frame(dataset, segment)
        prepared = self._prepare_prediction_scores(raw_pred)
        return self._blend_with_weights(prepared, self.model_weights)

    def _predict_frame(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.DataFrame:
        preds = {}
        for name in self.model_order:
            preds[name] = self.models[name].predict(dataset, segment)
        frame = pd.DataFrame(preds)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        return frame

    def _build_base_model(self, spec: _BaseModelSpec) -> Model:
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
        raise ValueError(f"Unsupported model_type: {spec.model_type}")

    def _build_optional_deep_model(self) -> Tuple[Optional[Model], Optional[str], str]:
        enabled = bool(self.deep_cfg.get("enabled", False))
        if not enabled:
            return None, None, "deep_branch.enabled=false"

        try:
            importlib.import_module("torch")
        except ModuleNotFoundError:
            return None, None, "torch is not installed"

        class_name = str(self.deep_cfg.get("class", "GRU"))
        module_path = self.deep_cfg.get("module_path")
        if module_path is None:
            default_map = {
                "GRU": "quant_master.contrib.model.pytorch_gru",
                "ALSTM": "quant_master.contrib.model.pytorch_alstm",
                "HIST": "quant_master.contrib.model.pytorch_hist",
                "TransformerModel": "quant_master.contrib.model.pytorch_transformer_ts",
                "TFTModel": "quant_master.contrib.model.pytorch_tft",
            }
            module_path = default_map.get(class_name, "quant_master.contrib.model.pytorch_gru")

        try:
            mod = importlib.import_module(str(module_path))
            cls = getattr(mod, class_name)
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            return None, None, f"failed to import deep branch {module_path}.{class_name}: {exc}"

        kwargs = dict(self.deep_cfg.get("kwargs", {}))
        kwargs.setdefault("seed", self.random_state)
        deep_name = str(self.deep_cfg.get("name", f"deep_{class_name.lower()}"))
        return cls(**kwargs), deep_name, "enabled"

    def _learn_weights(self, pred_frame: pd.DataFrame, label: pd.Series) -> Dict[str, float]:
        if pred_frame.empty:
            return {}
        names = list(pred_frame.columns)
        n_models = len(names)
        best_weights = None
        best_obj = None

        for weight in self._weight_candidates(n_models):
            blend = pd.Series(pred_frame.values @ weight, index=pred_frame.index)
            obj = self._portfolio_objective(blend, label)
            if best_obj is None or obj > best_obj:
                best_obj = obj
                best_weights = weight

        if best_weights is None:
            best_weights = np.ones(n_models, dtype=float) / n_models
        best_weights = np.clip(best_weights, 1e-12, None)
        best_weights = best_weights / best_weights.sum()
        return {name: float(w) for name, w in zip(names, best_weights)}

    def _weight_candidates(self, n_models: int) -> Iterable[np.ndarray]:
        if n_models == 1:
            yield np.array([1.0], dtype=float)
            return

        yielded = set()

        def _emit(arr: np.ndarray):
            rounded = tuple(np.round(arr, 6).tolist())
            if rounded in yielded:
                return
            yielded.add(rounded)
            yield arr

        equal_w = np.ones(n_models, dtype=float) / n_models
        for arr in _emit(equal_w):
            yield arr

        for i in range(n_models):
            onehot = np.zeros(n_models, dtype=float)
            onehot[i] = 1.0
            for arr in _emit(onehot):
                yield arr

        if n_models <= 3:
            for arr in self._grid_simplex_weights(n_models, self.search_step):
                for out in _emit(arr):
                    yield out

        sample_count = max(self.max_random_weight_candidates - len(yielded), 0)
        for _ in range(sample_count):
            w = self.rng.dirichlet(np.ones(n_models))
            for arr in _emit(w):
                yield arr

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

    def _portfolio_objective(self, score: pd.Series, label: pd.Series) -> float:
        ir, annret, _ = self._portfolio_metrics(score, label)
        return self.ir_weight * ir + self.annret_weight * annret

    def _portfolio_metrics(self, score: pd.Series, label: pd.Series) -> Tuple[float, float, float]:
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
        daily_ret = []
        for _, daily_score in score.groupby(level=date_level, sort=True):
            daily_label = label.loc[daily_score.index]
            if daily_score.empty:
                continue
            idx = self._topk_index(daily_score)
            day_ret = daily_label.loc[idx].mean()
            if np.isfinite(day_ret):
                daily_ret.append(float(day_ret))

        if not daily_ret:
            return float("-inf"), float("-inf"), 0.0
        ret_arr = np.asarray(daily_ret, dtype=float)
        mean_ret = float(np.mean(ret_arr))
        std_ret = float(np.std(ret_arr))
        ir = float((mean_ret / (std_ret + 1e-12)) * np.sqrt(self.ann_scaler))
        annret = float(mean_ret * self.ann_scaler)
        hit_ratio = float(np.mean(ret_arr > 0))
        return ir, annret, hit_ratio

    def _prepare_prediction_scores(self, pred_frame: pd.DataFrame) -> pd.DataFrame:
        pred_frame = pred_frame.replace([np.inf, -np.inf], np.nan)
        pred_frame = pred_frame.fillna(0.0)
        if self.use_rank_score:
            pred_frame = self._cross_sectional_rank_score(pred_frame)
        return pred_frame.fillna(0.0)

    def _blend_with_weights(self, pred_frame: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        if pred_frame.empty:
            return pd.Series(dtype=float)
        weight_arr = np.array([weights.get(c, 0.0) for c in pred_frame.columns], dtype=float)
        if not np.isfinite(weight_arr).all() or weight_arr.sum() <= 0:
            weight_arr = np.ones(len(pred_frame.columns), dtype=float) / len(pred_frame.columns)
        return pd.Series(pred_frame.values @ weight_arr, index=pred_frame.index).fillna(0.0)

    def _prepare_feature_frame(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.DataFrame:
        feature = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        if not isinstance(feature, pd.DataFrame):
            raise ValueError("feature data must be a pandas DataFrame.")
        feature = feature.replace([np.inf, -np.inf], np.nan)
        return feature.fillna(0.0)

    def _prepare_label_series(self, dataset: DatasetH, segment: Union[Text, slice]) -> pd.Series:
        raw_label = dataset.prepare(segment, col_set=["label"], data_key=DataHandlerLP.DK_L)
        label_df = self._to_label_frame(raw_label)
        label_s = label_df.iloc[:, 0]
        label_s = pd.to_numeric(label_s, errors="coerce")
        return self._clean_series(label_s)

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
            raise ValueError("All aligned validation rows are invalid after NaN filtering.")
        return pred, y

    def _topk_index(self, score: pd.Series) -> pd.Index:
        if len(score) <= self.topk:
            return score.index
        values = score.values
        pos = np.argpartition(values, -self.topk)[-self.topk :]
        return score.index[pos]

    @staticmethod
    def _cross_sectional_rank_score(values: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(values.index, pd.MultiIndex):
            return values.rank(pct=True).fillna(0.0)
        date_level = "datetime" if "datetime" in values.index.names else values.index.names[0]
        return values.groupby(level=date_level).rank(pct=True).fillna(0.0)

    @staticmethod
    def _clean_series(values: pd.Series) -> pd.Series:
        values = values.replace([np.inf, -np.inf], np.nan)
        if isinstance(values.index, pd.MultiIndex):
            date_level = "datetime" if "datetime" in values.index.names else values.index.names[0]
            values = values.groupby(level=date_level).transform(lambda s: s.fillna(float(s.median()) if not s.dropna().empty else 0.0))
        return values.fillna(0.0)

    @staticmethod
    def _date_level(index: pd.Index):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("A MultiIndex with datetime level is required.")
        return "datetime" if "datetime" in index.names else index.names[0]

    def _parse_specs(self, specs: Optional[Sequence[Dict]]) -> List[_BaseModelSpec]:
        if specs is None:
            specs = [
                {
                    "name": "de_main",
                    "model_type": "double_ensemble",
                    "model_kwargs": {
                        "base_model": "gbm",
                        "loss": "mse",
                        "num_models": 3,
                        "enable_sr": True,
                        "enable_fs": True,
                        "alpha1": 1.0,
                        "alpha2": 1.0,
                        "bins_sr": 10,
                        "bins_fs": 5,
                        "decay": 0.5,
                        "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4],
                        "sub_weights": [1, 1, 1],
                        "epochs": 24,
                        "learning_rate": 0.15,
                        "colsample_bytree": 0.88,
                        "subsample": 0.88,
                        "lambda_l1": 60.0,
                        "lambda_l2": 240.0,
                        "max_depth": 8,
                        "num_leaves": 180,
                        "num_threads": 8,
                        "verbosity": -1,
                    },
                },
                {
                    "name": "lgb_aux",
                    "model_type": "lightgbm",
                    "model_kwargs": {
                        "loss": "mse",
                        "early_stopping_rounds": 60,
                        "num_boost_round": 500,
                        "learning_rate": 0.04,
                        "colsample_bytree": 0.85,
                        "subsample": 0.85,
                        "lambda_l1": 8.0,
                        "lambda_l2": 100.0,
                        "max_depth": 8,
                        "min_data_in_leaf": 100,
                        "num_leaves": 127,
                        "num_threads": 8,
                    },
                },
                {
                    "name": "lin_aux",
                    "model_type": "linear",
                    "model_kwargs": {"estimator": "ridge", "alpha": 0.8, "fit_intercept": False},
                },
            ]

        parsed: List[_BaseModelSpec] = []
        used_names = set()
        for i, raw in enumerate(specs):
            name = str(raw.get("name", f"base_{i}"))
            if name in used_names:
                raise ValueError(f"Duplicate base model name: {name}")
            used_names.add(name)
            parsed.append(
                _BaseModelSpec(
                    name=name,
                    model_type=str(raw.get("model_type", "double_ensemble")),
                    model_kwargs=dict(raw.get("model_kwargs", {})),
                )
            )
        if not parsed:
            raise ValueError("base_model_specs must contain at least one model.")
        return parsed
