from quant_master.contrib.data.transcendence_handler import TranscendenceAlpha


def test_transcendence_handler_skips_market_relative_features_without_benchmark():
    handler = TranscendenceAlpha(
        instruments="csi300",
        start_time="2024-01-01",
        end_time="2024-12-31",
        fit_start_time="2024-01-01",
        fit_end_time="2024-06-30",
        benchmark=None,
        include_alpha158_base=False,
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
    )

    _, feature_names = handler.get_feature_config()

    assert "TX_EXCESS_RET_5" in feature_names
    assert "TX_REL_MOM_5" in feature_names
    assert "TX_BETA_5" in feature_names
    assert "TX_IDIO_VOL_5" in feature_names
