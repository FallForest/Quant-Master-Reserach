from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from quant_master.data.dataset import DatasetH
from quant_master.utils.exceptions import LoadObjectError
from quant_master.workflow.record_temp import SigAnaRecord, SignalRecord


def _frame(column: str, reverse: bool = False) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2023-12-29", "2024-01-03"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    values = [1.0, 2.0, 3.0] * 2
    if reverse:
        values = list(reversed(values))
    return pd.DataFrame({column: values}, index=index)


def _label_frame(reverse: bool = False) -> pd.DataFrame:
    frame = _frame("label", reverse=reverse)
    second_day = pd.Timestamp("2024-01-03")
    frame.loc[(second_day, ["B", "C"]), "label"] = frame.loc[(second_day, ["C", "B"]), "label"].to_numpy()
    return frame


class _Handler:
    def __init__(self) -> None:
        self.prepared_segments: list[str] = []

    def fetch(self, segment: str, **kwargs: Any) -> pd.DataFrame:
        self.prepared_segments.append(segment)
        return _label_frame(reverse=segment == "valid")


class _Dataset(DatasetH):
    def __init__(self) -> None:
        self.handler = _Handler()
        self.segments = {"valid": "valid", "test": "test"}


class _Model:
    def __init__(self) -> None:
        self.predicted_segments: list[str] = []

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        self.predicted_segments.append(segment)
        return _frame("score", reverse=segment == "valid")["score"]


class _Recorder:
    experiment_id = "experiment"

    def __init__(self) -> None:
        self.objects: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.tags: dict[str, Any] = {}

    def save_objects(self, artifact_path: str | None = None, **kwargs: Any) -> None:
        prefix = f"{artifact_path}/" if artifact_path else ""
        self.objects.update({f"{prefix}{name}": value for name, value in kwargs.items()})

    def load_object(self, name: str) -> Any:
        try:
            return self.objects[name]
        except KeyError as exc:
            raise LoadObjectError(name) from exc

    def log_metrics(self, **kwargs: Any) -> None:
        self.metrics.update(kwargs)

    def set_tags(self, **kwargs: Any) -> None:
        self.tags.update(kwargs)


def test_signal_record_writes_segmented_artifacts_and_legacy_test_aliases() -> None:
    recorder = _Recorder()
    model = _Model()
    dataset = _Dataset()

    record = SignalRecord(model=model, dataset=dataset, recorder=recorder, segments=["valid", "test"])
    record.generate()

    assert model.predicted_segments == ["valid", "test"]
    assert dataset.handler.prepared_segments == ["valid", "test"]
    assert {"pred_valid.pkl", "label_valid.pkl", "pred_test.pkl", "label_test.pkl"} <= recorder.objects.keys()
    assert recorder.objects["pred.pkl"].equals(recorder.objects["pred_test.pkl"])
    assert recorder.objects["label.pkl"].equals(recorder.objects["label_test.pkl"])
    assert recorder.tags["signal.segments"] == "valid,test"


def test_signal_analysis_logs_namespaced_metrics_and_keeps_test_aliases() -> None:
    recorder = _Recorder()
    SignalRecord(
        model=_Model(), dataset=_Dataset(), recorder=recorder, segments=["valid", "test"]
    ).generate()

    analysis = SigAnaRecord(recorder=recorder, segments=["valid", "test"])
    objects = analysis._generate()

    assert "signal.valid.IC" in recorder.metrics
    assert "signal.test.IC" in recorder.metrics
    assert "IC" in recorder.metrics
    assert "IC_effective_days" not in recorder.metrics
    assert recorder.metrics["signal.valid.IC_effective_days"] == 2
    assert recorder.metrics["signal.valid.Rank_IC_effective_days"] == 2
    assert recorder.metrics["signal.valid.IC_positive_day_ratio"] == 1.0
    assert recorder.metrics["signal.valid.Rank_IC_positive_day_ratio"] == 1.0
    assert recorder.metrics["signal.valid.ICIR"] == recorder.metrics["signal.valid.ICIR_daily"]
    assert recorder.metrics["signal.valid.ICIR_annualized"] == pytest.approx(
        recorder.metrics["signal.valid.ICIR_daily"] * 252**0.5
    )
    assert "signal.valid.year.2023.IC" in recorder.metrics
    assert "signal.valid.year.2023.Rank_IC" in recorder.metrics
    assert "signal.valid.year.2024.IC" in recorder.metrics
    assert "signal.valid.year.2024.Rank_IC" in recorder.metrics
    assert {"ic_valid.pkl", "ric_valid.pkl", "ic_test.pkl", "ric_test.pkl", "ic.pkl", "ric.pkl"} <= objects.keys()
    assert recorder.tags["signal.analysis_segments"] == "valid,test"
    assert recorder.tags["signal.analysis_schema"] == "segmented-v2"
    assert recorder.tags["signal.icir_ann_scaler"] == 252


def test_signal_record_defaults_to_legacy_test_behavior() -> None:
    recorder = _Recorder()
    model = _Model()

    record = SignalRecord(model=model, dataset=_Dataset(), recorder=recorder)
    record.generate()

    assert {"pred.pkl", "label.pkl", "pred_test.pkl", "label_test.pkl"} <= recorder.objects.keys()
    assert "pred_valid.pkl" not in recorder.objects
    assert model.predicted_segments == ["test"]
    assert record.list() == ["pred.pkl", "label.pkl"]
