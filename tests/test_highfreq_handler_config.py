"""Configuration-level tests for minute-bar feature expansion."""

import pytest

from quant_master.contrib.data.highfreq_handler import HighFreqGeneralHandler


def test_highfreq_general_handler_default_lags_are_backward_compatible():
    handler = HighFreqGeneralHandler.__new__(HighFreqGeneralHandler)
    handler.day_length = 240
    handler.columns = ["$close"]
    handler.lags = [0, 240]
    handler.feature_families = ()
    handler.feature_windows = {}

    fields, names = handler.get_feature_config()

    assert names == ["$close", "$close_1", "$volume", "$volume_1"]
    assert len(fields) == len(names)
    assert HighFreqGeneralHandler._normalize_feature_families(None) == ()
    assert HighFreqGeneralHandler._normalize_feature_windows((), None) == {}


def test_highfreq_general_handler_default_label_config_is_training_ready():
    handler = HighFreqGeneralHandler(
        instruments="csi300",
        start_time="2020-01-01",
        end_time="2020-01-02",
        init_data=False,
    )

    assert handler.get_label_config() == (["Ref($close, -2) / Ref($close, -1) - 1"], ["LABEL0"])
    assert handler.data_loader.is_group is True
    assert handler.data_loader.fields["label"] == (["Ref($close, -2) / Ref($close, -1) - 1"], ["LABEL0"])


def test_highfreq_general_handler_allows_custom_label_override():
    custom_label = (["Ref($vwap, -2) / Ref($vwap, -1) - 1"], ["LABEL0"])
    handler = HighFreqGeneralHandler(
        instruments="csi300",
        start_time="2020-01-01",
        end_time="2020-01-02",
        init_data=False,
        label=custom_label,
    )

    assert handler.data_loader.fields["label"] == custom_label


def test_highfreq_general_handler_accepts_extra_minute_lags():
    handler = HighFreqGeneralHandler.__new__(HighFreqGeneralHandler)
    handler.day_length = 48
    handler.columns = ["$close", "$vwap"]
    handler.lags = [0, 12, 48]
    handler.feature_families = ()
    handler.feature_windows = {}

    _, names = handler.get_feature_config()

    assert names == [
        "$close",
        "$vwap",
        "$close_lag12",
        "$vwap_lag12",
        "$close_1",
        "$vwap_1",
        "$volume",
        "$volume_lag12",
        "$volume_1",
    ]


def test_highfreq_general_handler_normalizes_missing_current_bar_without_reordering_lags():
    assert HighFreqGeneralHandler._normalize_lags(48, [12, 0, 48]) == [12, 0, 48]
    assert HighFreqGeneralHandler._normalize_lags(48, [12, 48]) == [0, 12, 48]


@pytest.mark.parametrize("day_length", [0, -1, True, 48.0, "48"])
def test_highfreq_general_handler_rejects_invalid_day_length(day_length):
    with pytest.raises(ValueError, match="day_length"):
        HighFreqGeneralHandler._normalize_day_length(day_length)


@pytest.mark.parametrize(
    "lags",
    [
        [0, 12, 12],
        [0, -1],
        [0, True],
        [0, "12"],
    ],
)
def test_highfreq_general_handler_rejects_ambiguous_lags(lags):
    with pytest.raises(ValueError):
        HighFreqGeneralHandler._normalize_lags(48, lags)


def test_highfreq_general_handler_adds_configured_minute_feature_families():
    handler = HighFreqGeneralHandler.__new__(HighFreqGeneralHandler)
    handler.day_length = 48
    handler.columns = ["$close"]
    handler.lags = [0]
    handler.feature_families = HighFreqGeneralHandler._normalize_feature_families(
        ["intraday_shape", "momentum", "volume_distribution", "realized_volatility"]
    )
    handler.feature_windows = HighFreqGeneralHandler._normalize_feature_windows(
        handler.feature_families,
        {
            "momentum": [6],
            "realized_volatility": [12],
            "volume_distribution": [6],
            "intraday_shape": [12],
        },
    )

    fields, names = handler.get_feature_config()

    assert names[-4:] == ["MOM6", "RVOL12", "VDIST6", "ISHAPE12"]
    assert "Ref($close, 6)" in fields[-4]
    assert "Std($close/Ref($close, 1)-1, 12)" in fields[-3]
    assert "Mean($volume, 6)" in fields[-2]
    assert "Max($high, 12)" in fields[-1]


@pytest.mark.parametrize(
    "feature_families,feature_windows",
    [
        (["unknown"], None),
        (["momentum", "momentum"], None),
        (["momentum"], {"unknown": [6]}),
        (["momentum"], {"realized_volatility": [6]}),
        (["momentum"], {"momentum": [0]}),
        (["momentum"], {"momentum": [True]}),
        (["momentum"], {"momentum": None}),
        (["momentum"], {"momentum": [6, 6]}),
    ],
)
def test_highfreq_general_handler_rejects_invalid_feature_family_config(feature_families, feature_windows):
    with pytest.raises(ValueError):
        normalized_families = HighFreqGeneralHandler._normalize_feature_families(feature_families)
        HighFreqGeneralHandler._normalize_feature_windows(normalized_families, feature_windows)
