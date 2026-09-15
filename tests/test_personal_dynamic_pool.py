import pandas as pd
import pytest

from quant_master.data.personal_dynamic_pool import (
    PersonalDynamicPoolConfig,
    _select_chunk,
    _replace_spans_for_range,
    _resolve_update_dates,
    classify_board,
)


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("SH600000", "main"),
        ("SZ000001", "main"),
        ("SZ300750", "chinext"),
        ("SH688981", "star"),
        ("BJ920001", "unknown"),
        ("SH000300", "unknown"),
    ],
)
def test_classify_board_is_explicit(symbol, expected):
    assert classify_board(symbol) == expected


def test_resolve_update_dates_supports_single_day_and_calendar_range():
    calendar = pd.bdate_range("2020-01-02", "2020-01-10")

    single = _resolve_update_dates(calendar, update_date="2020-01-06")
    assert list(single) == [pd.Timestamp("2020-01-06")]

    dates = _resolve_update_dates(calendar, start_date="2020-01-03", end_date="2020-01-09")
    assert list(dates) == list(pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"]))


def test_replace_spans_for_range_merges_rewritten_window_with_neighbors():
    calendar = pd.bdate_range("2020-01-02", "2020-01-10")
    existing = {
        "SH600000": [
            (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")),
            (pd.Timestamp("2020-01-08"), pd.Timestamp("2020-01-10")),
        ],
        "SZ000001": [(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-10"))],
    }
    replacement = {
        "SH600000": [(pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07"))],
        "SZ300750": [(pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07"))],
    }

    combined = _replace_spans_for_range(
        existing,
        replacement,
        calendar,
        pd.Timestamp("2020-01-06"),
        pd.Timestamp("2020-01-07"),
    )

    assert combined["SH600000"] == [(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-10"))]
    assert combined["SZ000001"] == [
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")),
        (pd.Timestamp("2020-01-08"), pd.Timestamp("2020-01-10")),
    ]
    assert combined["SZ300750"] == [(pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07"))]


def test_select_chunk_requires_prior_activity_and_keeps_current_day_trade_check():
    dates = pd.DatetimeIndex(["2024-01-03"])
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-03"), "SH600000"),
            (pd.Timestamp("2024-01-03"), "SZ000001"),
            (pd.Timestamp("2024-01-03"), "BJ920001"),
        ],
        names=["datetime", "instrument"],
    )
    data = pd.DataFrame(
        [
            [10.0, 1000.0, 60_000_000.0, 55_000_000.0, 120.0, 0.010],
            [10.0, 1000.0, 60_000_000.0, 55_000_000.0, 120.0, 0.007],
            [10.0, 1000.0, 60_000_000.0, 55_000_000.0, 120.0, 0.020],
        ],
        index=index,
    )
    config = PersonalDynamicPoolConfig(min_pool_size=1, max_pool_size=10)

    states, diagnostics = _select_chunk(data, dates, config)

    assert states[dates[0]] == {"SH600000"}
    assert diagnostics[0]["liquidity_eligible_rows"] == 2
    assert diagnostics[0]["eligible_rows"] == 1
