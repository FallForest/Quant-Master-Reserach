import pandas as pd
import pytest

from quant_master.contrib.data.transcendence_handler import TranscendenceAlpha
from quant_master.data.dataset.processor import DropnaLabel, Fillna


def test_transcendence_handler_skips_market_relative_features_without_benchmark():
    handler = TranscendenceAlpha(
        instruments="csi300",
        start_time="2024-01-01",
        end_time="2024-12-31",
        fit_start_time="2024-01-01",
        fit_end_time="2024-06-30",
        benchmark=None,
        include_alpha158_base=False,
        init_data=False,
    )

    _, feature_names = handler.get_feature_config()

    assert "TX_EXCESS_RET_5" not in feature_names
    assert "TX_REL_MOM_5" not in feature_names
    assert "TX_BETA_5" not in feature_names
    assert "TX_IDIO_VOL_5" not in feature_names


def test_transcendence_handler_keeps_market_relative_features_with_benchmark():
    handler = TranscendenceAlpha(
        instruments="csi300",
        start_time="2024-01-01",
        end_time="2024-12-31",
        fit_start_time="2024-01-01",
        fit_end_time="2024-06-30",
        benchmark="SH000905",
        include_alpha158_base=False,
        init_data=False,
    )

    _, feature_names = handler.get_feature_config()

    assert "TX_EXCESS_RET_5" in feature_names
    assert "TX_REL_MOM_5" in feature_names
    assert "TX_BETA_5" in feature_names
    assert "TX_IDIO_VOL_5" in feature_names


def test_transcendence_handler_default_fillna_only_fills_features():
    handler = TranscendenceAlpha(
        instruments="csi300",
        start_time="2024-01-01",
        end_time="2024-12-31",
        fit_start_time="2024-01-01",
        fit_end_time="2024-06-30",
        include_alpha158_base=False,
        init_data=False,
    )

    fillna_processors = [processor for processor in handler.infer_processors if isinstance(processor, Fillna)]

    assert len(fillna_processors) == 1
    assert fillna_processors[0].fields_group == "feature"

    columns = pd.MultiIndex.from_tuples([("feature", "F0"), ("label", "LABEL0")])
    frame = pd.DataFrame([[float("nan"), float("nan")], [1.0, 0.1]], columns=columns)
    filled = fillna_processors[0](frame)

    assert pd.isna(filled.loc[0, ("label", "LABEL0")])
    assert list(DropnaLabel()(filled).index) == [1]


def test_transcendence_handler_configurable_index_and_industry_exposures():
    handler = TranscendenceAlpha(
        instruments="csi300",
        start_time="2024-01-01",
        end_time="2024-12-31",
        fit_start_time="2024-01-01",
        fit_end_time="2024-06-30",
        include_alpha158_base=False,
        index_exposures=["SH000300"],
        industry_exposures={"bank": "SH000905"},
        exposure_windows=[5],
        init_data=False,
    )
    _, names = handler.get_feature_config()
    assert "TX_IDX_EXCESS_RET_SH000300_5" in names
    assert "TX_IDX_BETA_SH000300_5" in names
    assert "TX_IND_EXCESS_RET_BANK_5" in names
    assert "TX_IND_BETA_BANK_5" in names


def test_transcendence_handler_rejects_invalid_exposure_window():
    with pytest.raises(ValueError, match="exposure_windows"):
        TranscendenceAlpha(include_alpha158_base=False, exposure_windows=[0], init_data=False)


@pytest.mark.parametrize("windows", [[True], [1.0], ["20"], [20, 20]])
def test_transcendence_handler_rejects_non_integer_or_duplicate_exposure_windows(windows):
    with pytest.raises(ValueError, match="exposure_windows"):
        TranscendenceAlpha(include_alpha158_base=False, exposure_windows=windows, init_data=False)


def test_transcendence_handler_rejects_empty_industry_exposure_label():
    with pytest.raises(ValueError, match="industry_exposures labels"):
        TranscendenceAlpha(
            include_alpha158_base=False,
            industry_exposures={"": "SH000905"},
            init_data=False,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index_exposures": [300]},
        {"industry_exposures": {1: "SH000905"}},
        {"industry_exposures": {"bank": 905}},
    ],
)
def test_transcendence_handler_rejects_non_string_exposure_symbols(kwargs):
    with pytest.raises(ValueError):
        TranscendenceAlpha(include_alpha158_base=False, init_data=False, **kwargs)
