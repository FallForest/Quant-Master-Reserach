import types

import pandas as pd

from quant_master.contrib.data.alpha_tdx_minute_handler import Alpha158TDXMinute, Alpha158TDXMinuteLoader


def test_alpha158_tdx_minute_handler_combines_daily_base_and_minute_increment():
    handler = Alpha158TDXMinute(
        instruments="csi300",
        start_time="2020-01-01",
        end_time="2020-01-31",
        fit_start_time="2020-01-01",
        fit_end_time="2020-01-15",
        init_data=False,
        minute_lags=[0, 6, 240],
        minute_feature_families=["momentum"],
        minute_feature_windows={"momentum": [6, 12]},
    )

    fields = handler.data_loader.fields
    _, day_names = fields["feature_day"]
    _, minute_names = fields["feature_minute"]

    assert len(day_names) == 158
    assert all(name.startswith("DAY_") for name in day_names)
    assert all(name.startswith("MIN_") for name in minute_names)
    assert "MIN_$close_lag6" in minute_names
    assert "MIN_MOM6" in minute_names
    assert "MIN_MOM12" in minute_names
    assert fields["label"] == (["Ref($close, -2) / Ref($close, -1) - 1"], ["LABEL0"])


def test_alpha158_tdx_minute_loader_collapses_feature_groups_for_training():
    loader = Alpha158TDXMinuteLoader(
        config={
            "feature_day": (["$close"], ["DAY_$close"]),
            "feature_minute": (["$close"], ["MIN_$close"]),
            "label": (["Ref($close, -2) / Ref($close, -1) - 1"], ["LABEL0"]),
        },
        freq={"feature_day": "day", "feature_minute": "1min", "label": "day"},
        inst_processors={"feature_day": [], "feature_minute": [], "label": []},
    )
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2020-01-02"), "SH600000")],
        names=["datetime", "instrument"],
    )

    def fake_load_group_df(self, instruments, exprs, names, start_time=None, end_time=None, gp_name=None):
        return pd.DataFrame([[1.0] * len(names)], index=index, columns=names)

    loader.load_group_df = types.MethodType(fake_load_group_df, loader)

    frame = loader.load(["SH600000"], "2020-01-02", "2020-01-02")

    assert ("feature", "DAY_$close") in frame.columns
    assert ("feature", "MIN_$close") in frame.columns
    assert ("label", "LABEL0") in frame.columns


def test_alpha158_tdx_minute_loader_collapses_minute_rows_to_daily_last_bar():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-07-20 09:31:00"), "SH600000"),
            (pd.Timestamp("2026-07-20 15:00:00"), "SH600000"),
            (pd.Timestamp("2026-07-21 09:31:00"), "SH600000"),
            (pd.Timestamp("2026-07-21 15:00:00"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({"MIN_$close": [1.0, 2.0, 3.0, 4.0]}, index=index)

    collapsed = Alpha158TDXMinuteLoader._collapse_minute_to_daily(frame)

    expected_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-07-20"), "SH600000"),
            (pd.Timestamp("2026-07-21"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    pd.testing.assert_index_equal(collapsed.index, expected_index)
    assert collapsed["MIN_$close"].tolist() == [2.0, 4.0]
