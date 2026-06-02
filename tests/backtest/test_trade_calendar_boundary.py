import pandas as pd

from quant_master.backtest.utils import TradeCalendarManager
from quant_master.utils.time import epsilon_change


def test_get_step_time_uses_next_calendar_when_available():
    calendar = TradeCalendarManager.__new__(TradeCalendarManager)
    calendar.freq = "day"
    calendar._calendar = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]).to_numpy()
    calendar.start_index = 0
    calendar.end_index = 1
    calendar.trade_len = 2
    calendar.trade_step = 0

    start_time, end_time = calendar.get_step_time(trade_step=1)

    assert start_time == pd.Timestamp("2024-01-03")
    assert end_time == epsilon_change(pd.Timestamp("2024-01-04"))


def test_get_step_time_last_step_does_not_overflow_calendar_tail():
    calendar = TradeCalendarManager.__new__(TradeCalendarManager)
    calendar.freq = "day"
    calendar._calendar = pd.to_datetime(["2024-01-02", "2024-01-03"]).to_numpy()
    calendar.start_index = 0
    calendar.end_index = 1
    calendar.trade_len = 2
    calendar.trade_step = 1

    start_time, end_time = calendar.get_step_time()

    assert start_time == pd.Timestamp("2024-01-03")
    assert end_time == epsilon_change(pd.Timestamp("2024-01-04"))
