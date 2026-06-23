import json

import pytest

from quant_master.model import trainer


class FakeRecorder:
    pass


class FakeRecorderWrapper:
    def __init__(self):
        self.params = {}
        self.tags = {}
        self.metrics = {}
        self.objects = {}
        self.artifacts = {}
        self.recorder = FakeRecorder()

    def get_recorder(self):
        return self.recorder

    def log_params(self, **kwargs):
        self.params.update(kwargs)

    def set_tags(self, **kwargs):
        self.tags.update(kwargs)

    def log_metrics(self, step=None, **kwargs):
        self.metrics.update(kwargs)

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        self.objects.update(kwargs)

    def log_artifact(self, local_path, artifact_path=None):
        with open(local_path) as f:
            self.artifacts["experiment_summary.json"] = json.load(f)


class FailingMetadataRecorderWrapper(FakeRecorderWrapper):
    def log_metrics(self, step=None, **kwargs):
        raise RuntimeError("metric logging failed")

    def log_artifact(self, local_path, artifact_path=None):
        raise RuntimeError("artifact logging failed")


class DummyModel:
    def __init__(self):
        self.fit_calls = []

    def fit(self, dataset, reweighter=None):
        self.fit_calls.append((dataset, reweighter))


class FailingModel(DummyModel):
    def fit(self, dataset, reweighter=None):
        super().fit(dataset, reweighter=reweighter)
        raise RuntimeError("fit failed")


class DummyDataset:
    def __init__(self):
        self.config_calls = []

    def config(self, **kwargs):
        self.config_calls.append(kwargs)


class DummyRecord:
    generated = []

    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def generate(self):
        self.generated.append((self.config, self.kwargs))


def _task_config():
    return {
        "model": {
            "class": "ToyModel",
            "module_path": "tests.fake_model",
            "kwargs": {"depth": 2},
        },
        "dataset": {
            "class": "ToyDataset",
            "module_path": "tests.fake_dataset",
            "kwargs": {
                "handler": {
                    "class": "ToyHandler",
                    "module_path": "tests.fake_handler",
                },
                "segments": {
                    "train": ("2020-01-01", "2020-02-01"),
                    "valid": ("2020-02-01", "2020-03-01"),
                    "test": ("2020-03-01", "2020-04-01"),
                },
            },
        },
        "record": [
            {"class": "SignalRecord", "module_path": "quant_master.workflow.record_temp"},
            {"class": "SigAnaRecord", "module_path": "quant_master.workflow.record_temp"},
        ],
    }


def _patch_trainer(monkeypatch, model, dataset, fake_r):
    def fake_init_instance_by_config(config, *args, **kwargs):
        if config.get("class") == "ToyModel":
            return model
        if config.get("class") == "ToyDataset":
            return dataset
        return DummyRecord(config, **kwargs)

    DummyRecord.generated = []
    monkeypatch.setattr(trainer, "R", fake_r)
    monkeypatch.setattr(trainer, "init_instance_by_config", fake_init_instance_by_config)


def test_task_train_logs_metadata_summary_without_market_data(monkeypatch):
    task_config = _task_config()
    fake_r = FakeRecorderWrapper()
    model = DummyModel()
    dataset = DummyDataset()
    _patch_trainer(monkeypatch, model, dataset, fake_r)

    trainer._log_task_info(task_config)
    trainer._exe_task(task_config)

    assert fake_r.params["model.class"] == "ToyModel"
    assert fake_r.tags["dataset.module_path"] == "tests.fake_dataset"
    assert fake_r.tags["handler.class"] == "ToyHandler"
    assert fake_r.tags["segments.train"] == '["2020-01-01", "2020-02-01"]'
    assert len(fake_r.tags["task_hash"]) == 64
    assert fake_r.objects["task"] is task_config

    assert model.fit_calls == [(dataset, None)]
    assert dataset.config_calls == [{"dump_all": False, "recursive": True}]
    assert fake_r.objects["params.pkl"] is model
    assert fake_r.objects["dataset"] is dataset
    assert len(DummyRecord.generated) == 2

    summary = fake_r.artifacts["experiment_summary.json"]
    assert summary["model.class"] == "ToyModel"
    assert summary["dataset.class"] == "ToyDataset"
    assert summary["handler.module_path"] == "tests.fake_handler"
    assert summary["record.count"] == 2
    assert summary["record.classes"] == ["SignalRecord", "SigAnaRecord"]
    assert summary["fit_status"] == "success"
    assert summary["fit_duration_seconds"] >= 0
    assert fake_r.metrics["fit_duration_seconds"] == summary["fit_duration_seconds"]
    assert fake_r.tags["fit_status"] == "success"
    assert fake_r.tags["train_stage"] == "records"


def test_task_train_fit_failure_logs_and_reraises(monkeypatch):
    task_config = _task_config()
    fake_r = FakeRecorderWrapper()
    model = FailingModel()
    dataset = DummyDataset()
    _patch_trainer(monkeypatch, model, dataset, fake_r)

    with pytest.raises(RuntimeError, match="fit failed"):
        trainer._exe_task(task_config)

    summary = fake_r.artifacts["experiment_summary.json"]
    assert summary["fit_status"] == "failed"
    assert summary["fit_duration_seconds"] >= 0
    assert fake_r.tags["fit_status"] == "failed"
    assert fake_r.tags["train_stage"] == "fit"
    assert "params.pkl" not in fake_r.objects


def test_successful_fit_survives_metadata_logging_failure(monkeypatch):
    task_config = _task_config()
    fake_r = FailingMetadataRecorderWrapper()
    model = DummyModel()
    dataset = DummyDataset()
    _patch_trainer(monkeypatch, model, dataset, fake_r)

    trainer._exe_task(task_config)

    assert model.fit_calls == [(dataset, None)]
    assert fake_r.objects["params.pkl"] is model
    assert fake_r.objects["dataset"] is dataset
    assert dataset.config_calls == [{"dump_all": False, "recursive": True}]
    assert len(DummyRecord.generated) == 2
    assert fake_r.metrics == {}
    assert fake_r.artifacts == {}
    assert fake_r.tags["train_stage"] == "records"
