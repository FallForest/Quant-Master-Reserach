from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_master.contrib.model.purged_oof_stacking import PurgedWalkForwardStackingModel
from quant_master.data.dataset.loader import StaticDataLoader
from quant_master.model.base import Model


class FeatureModel(Model):
    def __init__(self, feature):
        self.feature = feature
        self.fitted_segments = []

    def fit(self, dataset, reweighter=None):
        self.fitted_segments = ["train", "valid"]
        dataset.prepare("train", col_set=["feature", "label"])
        dataset.prepare("valid", col_set=["feature", "label"])
        return self

    def predict(self, dataset, segment="test"):
        return dataset.prepare(segment, col_set="feature")[self.feature]


class FrameDataset:
    def __init__(self, frame, segments):
        self.frame = frame
        self.segments = segments
        self.handler = SimpleNamespace(_data=frame)

    def prepare(self, segment, col_set=None, data_key=None):
        start, end = self.segments[segment]
        dates = self.frame.index.get_level_values("datetime")
        selected = self.frame.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
        if col_set == "feature":
            return selected["feature"]
        if col_set == ["label"]:
            return selected[[*(col for col in selected.columns if col[0] == "label")]]
        return selected


class InMemoryOOFStack(PurgedWalkForwardStackingModel):
    def __init__(self, fold_datasets, **kwargs):
        self._fold_datasets = fold_datasets
        super().__init__(fold_dataset_config={"class": "unused"}, **kwargs)

    def _build_fold_dataset(self, fold):
        return self._fold_datasets[fold["oof"][0]]


def _frame():
    dates = pd.bdate_range("2020-01-01", "2020-05-29")
    instruments = ["A", "B", "C", "D", "E"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    row = np.arange(len(index), dtype=float)
    instrument_code = np.tile(np.arange(len(instruments), dtype=float), len(dates))
    f0 = np.sin(row / 7.0) + instrument_code
    f1 = np.cos(row / 11.0) - instrument_code * 0.2
    label = 0.8 * f0 + 0.2 * f1
    columns = pd.MultiIndex.from_tuples(
        [("feature", "F0"), ("feature", "F1"), ("label", "LABEL0")]
    )
    return pd.DataFrame(np.column_stack([f0, f1, label]), index=index, columns=columns)


def _folds():
    return [
        {
            "train": ("2020-01-01", "2020-01-09"),
            "valid": ("2020-01-13", "2020-01-17"),
            "oof": ("2020-01-21", "2020-01-24"),
        },
        {
            "train": ("2020-01-01", "2020-01-30"),
            "valid": ("2020-02-03", "2020-02-07"),
            "oof": ("2020-02-11", "2020-02-14"),
        },
    ]


def _model(purge_periods=1, embargo_periods=1, folds=None):
    frame = _frame()
    folds = folds or _folds()
    fold_datasets = {
        fold["oof"][0]: FrameDataset(
            frame,
            {"train": fold["train"], "valid": fold["valid"], "test": fold["oof"]},
        )
        for fold in folds
    }
    model = InMemoryOOFStack(
        fold_datasets=fold_datasets,
        base_models={
            "feature_0": {"class": FeatureModel, "kwargs": {"feature": "F0"}},
            "feature_1": {"class": FeatureModel, "kwargs": {"feature": "F1"}},
        },
        folds=folds,
        purge_periods=purge_periods,
        embargo_periods=embargo_periods,
        selection_end_time="2023-12-31",
        ridge_alpha=0.1,
    )
    outer = FrameDataset(
        frame,
        {
            "train": ("2020-01-01", "2020-03-31"),
            "valid": ("2020-04-01", "2020-04-30"),
            "test": ("2020-05-01", "2020-05-29"),
        },
    )
    return model, outer


def test_stacking_learns_only_from_non_overlapping_oof_predictions():
    model, outer = _model()

    model.fit(outer)
    prediction = model.predict(outer)
    diagnostics = model.get_oof_diagnostics()

    assert not model.oof_predictions_.index.has_duplicates
    assert set(model.model_weights) == {"feature_0", "feature_1"}
    assert sum(model.model_weights.values()) == pytest.approx(1.0)
    assert model.model_weights["feature_0"] > model.model_weights["feature_1"]
    assert diagnostics["oof_rows"] == 40
    assert len(diagnostics["folds"]) == 2
    assert set(diagnostics["overall"]) == {"rank_ic", "rank_icir", "days"}
    assert set(diagnostics["fitted_oof"]) == {"rank_ic", "rank_icir", "days"}
    assert diagnostics["folds"][0]["stack_weights"] == {"feature_0": 0.5, "feature_1": 0.5}
    assert diagnostics["folds"][0]["train_valid_purge_periods"] == 1
    assert diagnostics["folds"][0]["valid_oof_purge_periods"] == 1
    assert diagnostics["yearly"]["2020"]["days"] == 8
    assert prediction.index.get_level_values("datetime").min() == pd.Timestamp("2020-05-01")


def test_recent_oof_half_life_adapts_weights_without_using_future_rows():
    model, _ = _model()
    model.ridge_alpha = 0.0
    dates = pd.bdate_range("2020-01-01", periods=10)
    instruments = ["A", "B", "C", "D", "E"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    ascending = np.tile(np.arange(5, dtype=float), len(dates))
    descending = -ascending
    predictions = pd.DataFrame({"feature_0": ascending, "feature_1": descending}, index=index)
    label = pd.Series(ascending, index=index)
    recent = index.get_level_values("datetime") >= dates[-2]
    label.loc[recent] = descending[recent]

    model.weight_half_life_periods = None
    full_history_weights = model._learn_weights(predictions, label)
    model.weight_half_life_periods = 1
    recent_weights = model._learn_weights(predictions, label)

    assert full_history_weights["feature_0"] > full_history_weights["feature_1"]
    assert recent_weights["feature_1"] > recent_weights["feature_0"]


def test_stacking_enforces_model_weight_cap():
    model, outer = _model()
    model.ridge_alpha = 0.0
    model.max_model_weight = 0.55

    model.fit(outer)

    assert max(model.model_weights.values()) <= 0.55 + 1e-10


def test_stacking_keeps_pre_robust_weight_pickle_state_compatible():
    model, outer = _model()
    del model.weight_half_life_periods
    del model.max_model_weight

    model.fit(outer)
    diagnostics = model.get_oof_diagnostics()

    assert diagnostics["weight_half_life_periods"] is None
    assert diagnostics["max_model_weight"] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weight_half_life_periods": 0}, "weight_half_life_periods"),
        ({"weight_half_life_periods": 0.5}, "weight_half_life_periods"),
        ({"max_model_weight": 0.4}, "too small"),
    ],
)
def test_stacking_rejects_invalid_robust_weight_settings(kwargs, message):
    frame = _frame()
    folds = _folds()
    fold_datasets = {
        fold["oof"][0]: FrameDataset(frame, {"train": fold["train"], "valid": fold["valid"], "test": fold["oof"]})
        for fold in folds
    }

    with pytest.raises(ValueError, match=message):
        InMemoryOOFStack(
            fold_datasets=fold_datasets,
            base_models={
                "feature_0": {"class": FeatureModel, "kwargs": {"feature": "F0"}},
                "feature_1": {"class": FeatureModel, "kwargs": {"feature": "F1"}},
            },
            folds=folds,
            **kwargs,
        )


def test_stacking_rejects_insufficient_purge_periods():
    model, outer = _model(purge_periods=2)

    with pytest.raises(ValueError, match="only 1 purge periods"):
        model.fit(outer)


def test_stacking_rejects_oof_selection_in_2024():
    folds = _folds()
    folds[1]["oof"] = ("2024-01-02", "2024-01-05")
    model, _ = _model(folds=folds)

    with pytest.raises(ValueError, match="ends after selection_end_time"):
        model._normalize_fold(folds[1], 1)


def test_stacking_rejects_folds_that_move_backwards_in_time():
    model, outer = _model(folds=list(reversed(_folds())))

    with pytest.raises(ValueError, match="strictly increasing"):
        model.fit(outer)


def test_stacking_rejects_embargo_before_building_next_fold():
    folds = _folds()
    folds[1] = {
        "train": ("2020-01-01", "2020-01-09"),
        "valid": ("2020-01-13", "2020-01-17"),
        "oof": ("2020-01-27", "2020-01-30"),
    }
    model, outer = _model(embargo_periods=1, folds=folds)

    with pytest.raises(ValueError, match="only 0 embargo periods"):
        model.fit(outer)


def test_oof_gate_enforces_overall_and_yearly_quality():
    model, outer = _model()
    model.fit(outer)

    diagnostics = model.assert_oof_gate(min_rank_ic=0.49, min_rank_icir=0.25)

    assert diagnostics["overall"]["rank_ic"] >= 0.49
    with pytest.raises(ValueError, match="overall RankIC"):
        model.assert_oof_gate(min_rank_ic=1.01, min_rank_icir=0.25)


def test_fold_dataset_config_locks_processor_fit_to_fold_train(monkeypatch):
    model, _ = _model()
    model.fold_dataset_config = {
        "class": "DatasetH",
        "kwargs": {"handler": {"class": "Alpha158", "kwargs": {"start_time": "2010-01-01"}}},
    }
    captured = {}

    def capture(config, accept_types=None):
        captured.update(config)
        return "dataset"

    monkeypatch.setattr("quant_master.contrib.model.purged_oof_stacking.init_instance_by_config", capture)

    assert PurgedWalkForwardStackingModel._build_fold_dataset(model, _folds()[0]) == "dataset"
    handler_kwargs = captured["kwargs"]["handler"]["kwargs"]
    assert handler_kwargs["fit_start_time"] == "2020-01-01"
    assert handler_kwargs["fit_end_time"] == "2020-01-09"
    assert captured["kwargs"]["segments"]["test"] == ("2020-01-21", "2020-01-24")


def test_fold_dataset_can_reuse_unprocessed_final_handler_data(monkeypatch):
    model, _ = _model()
    model.reuse_final_raw_data = True
    model._shared_raw_data = _frame()
    model.fold_dataset_config = {
        "class": "DatasetH",
        "kwargs": {"handler": {"class": "Alpha360TDX", "kwargs": {"instruments": "market"}}},
    }
    captured = {}

    def capture(config, accept_types=None):
        captured.update(config)
        return "dataset"

    monkeypatch.setattr("quant_master.contrib.model.purged_oof_stacking.init_instance_by_config", capture)

    assert PurgedWalkForwardStackingModel._build_fold_dataset(model, _folds()[0]) == "dataset"
    handler_kwargs = captured["kwargs"]["handler"]["kwargs"]
    assert handler_kwargs["instruments"] is None
    assert isinstance(handler_kwargs["data_loader"], StaticDataLoader)
    assert handler_kwargs["data_loader"]._config is model._shared_raw_data


def test_final_split_can_require_purge_and_post_selection_test():
    model, outer = _model()
    model.final_test_purge_periods = 1
    model.require_test_after_selection = True
    model.selection_end_time = pd.Timestamp("2020-04-30")
    outer.segments["test"] = ("2020-05-04", "2020-05-29")

    model._validate_final_refit_segments(outer)

    outer.segments["test"] = ("2020-04-30", "2020-05-29")
    with pytest.raises(ValueError, match="strictly ordered"):
        model._validate_final_refit_segments(outer)


def test_final_split_rejects_insufficient_purge():
    model, outer = _model()
    model.final_test_purge_periods = 2

    with pytest.raises(ValueError, match="only 0 purge periods"):
        model._validate_final_refit_segments(outer)
