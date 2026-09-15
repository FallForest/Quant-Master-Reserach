# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Alpha158TDX plus minute-bar feature increments for training."""

import pandas as pd

from quant_master.contrib.data.amount_flow_handler import Alpha158TDX
from quant_master.contrib.data.highfreq_handler import HighFreqGeneralHandler
from quant_master.contrib.data.handler import _DEFAULT_LEARN_PROCESSORS, check_transform_proc
from quant_master.data.dataset.loader import QuantMasterDataLoader
from quant_master.data.dataset.handler import DataHandlerLP


def _prefix_names(prefix, names):
    return [f"{prefix}{name}" for name in names]


class Alpha158TDXMinuteLoader(QuantMasterDataLoader):
    """Collapse multiple feature groups into the standard ``feature`` group."""

    @staticmethod
    def _collapse_minute_to_daily(df):
        if df.empty or not isinstance(df.index, pd.MultiIndex):
            return df

        names = list(df.index.names)
        if "datetime" in names:
            datetime_level = names.index("datetime")
        else:
            datetime_level = 0
        if "instrument" in names:
            instrument_level = names.index("instrument")
        else:
            instrument_level = 1 if datetime_level == 0 and df.index.nlevels > 1 else 0

        work = df.copy()
        datetime_values = pd.to_datetime(work.index.get_level_values(datetime_level)).normalize()
        instrument_values = work.index.get_level_values(instrument_level)
        work = work.assign(__daily_datetime=datetime_values, __instrument=instrument_values)
        work = work.sort_index().groupby(["__daily_datetime", "__instrument"], sort=True).tail(1)
        work.index = pd.MultiIndex.from_arrays(
            [work.pop("__daily_datetime"), work.pop("__instrument")],
            names=["datetime", "instrument"],
        )
        return work.sort_index()

    def load_group_df(self, instruments, exprs, names, start_time=None, end_time=None, gp_name=None):
        original_filter_pipe = self.filter_pipe
        if gp_name == "feature_minute":
            self.filter_pipe = None
        try:
            df = super().load_group_df(instruments, exprs, names, start_time, end_time, gp_name)
        finally:
            self.filter_pipe = original_filter_pipe
        if gp_name == "feature_minute":
            df = self._collapse_minute_to_daily(df)
        return df

    def load(self, instruments=None, start_time=None, end_time=None):
        df = super().load(instruments, start_time, end_time)
        if self.is_group:
            df.columns = df.columns.map(lambda x: ("feature", x[1]) if x[0].startswith("feature") else x)
        return df


class Alpha158TDXMinute(DataHandlerLP):
    """Daily Alpha158TDX plus optional minute-bar features in one training handler."""

    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        minute_day_length=240,
        minute_freq="1min",
        minute_columns=("$open", "$high", "$low", "$close", "$vwap"),
        minute_lags=None,
        minute_feature_families=HighFreqGeneralHandler.FEATURE_FAMILIES,
        minute_feature_windows=None,
        label=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        if inst_processors is None:
            inst_processors = {"feature_day": [], "feature_minute": [], "label": []}

        day_fields, day_names = Alpha158TDX(init_data=False).get_feature_config()
        minute_handler = HighFreqGeneralHandler(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            infer_processors=[],
            learn_processors=[],
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            drop_raw=True,
            day_length=minute_day_length,
            freq=minute_freq,
            columns=list(minute_columns),
            lags=minute_lags,
            feature_families=minute_feature_families,
            feature_windows=minute_feature_windows,
            label=label,
            init_data=False,
        )
        minute_fields, minute_names = minute_handler.get_feature_config()
        label = minute_handler.get_label_config() if label is None else label

        data_loader = Alpha158TDXMinuteLoader(
            config={
                "feature_day": (day_fields, _prefix_names("DAY_", day_names)),
                "feature_minute": (minute_fields, _prefix_names("MIN_", minute_names)),
                "label": label,
            },
            filter_pipe=filter_pipe,
            freq={
                "feature_day": "day",
                "feature_minute": minute_freq,
                "label": "day",
            },
            inst_processors=inst_processors,
        )

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2) / Ref($close, -1) - 1"], ["LABEL0"]

    @staticmethod
    def get_feature_config():
        day_fields, day_names = Alpha158TDX(init_data=False).get_feature_config()
        minute_fields, minute_names = HighFreqGeneralHandler(
            feature_families=HighFreqGeneralHandler.FEATURE_FAMILIES,
            init_data=False,
        ).get_feature_config()
        return (
            list(day_fields) + list(minute_fields),
            _prefix_names("DAY_", day_names) + _prefix_names("MIN_", minute_names),
        )


__all__ = ("Alpha158TDXMinute",)
