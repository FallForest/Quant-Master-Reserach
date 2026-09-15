import pytest

from quant_master.data.dataset import DatasetH
from quant_master.data.dataset.handler import DataHandlerLP
from quant_master.data.dataset.processor import ZScoreNorm


def _handler(fit_end_time):
    handler = object.__new__(DataHandlerLP)
    handler.shared_processors = []
    handler.infer_processors = [
        ZScoreNorm(fit_start_time="2020-01-01", fit_end_time=fit_end_time, fields_group="feature")
    ]
    handler.learn_processors = []
    return handler


def test_dataset_rejects_processor_fit_end_after_train_end():
    with pytest.raises(ValueError, match="must not be later than the train segment end"):
        DatasetH(
            handler=_handler("2021-01-01"),
            segments={"train": ("2020-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
        )


def test_dataset_rejects_handler_config_before_instantiation():
    handler_config = {
        "class": "ThisClassMustNotBeResolved",
        "kwargs": {"fit_end_time": "2021-01-01"},
    }

    with pytest.raises(ValueError, match="handler.fit_end_time"):
        DatasetH(handler=handler_config, segments={"train": ("2020-01-01", "2020-12-31")})


def test_dataset_accepts_processor_fit_end_equal_to_train_end():
    dataset = DatasetH(
        handler=_handler("2020-12-31"),
        segments={"train": ("2020-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    )

    assert dataset.segments["train"][1] == "2020-12-31"
