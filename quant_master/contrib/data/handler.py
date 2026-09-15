# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from quant_master.contrib.data.loader import Alpha158DL, Alpha360DL
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        data_loader=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        label = kwargs.pop("label", self.get_label_config())
        if data_loader is None:
            data_loader = {
                "class": "QuantMasterDataLoader",
                "kwargs": {
                    "config": {
                        "feature": self.get_feature_config(),
                        "label": label,
                    },
                    "filter_pipe": filter_pipe,
                    "freq": freq,
                    "inst_processors": inst_processors,
                },
            }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    def get_feature_config(self):
        return Alpha360DL.get_feature_config()


class Alpha360vwap(Alpha360):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        data_loader=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        label = kwargs.pop("label", self.get_label_config())
        if data_loader is None:
            data_loader = {
                "class": "QuantMasterDataLoader",
                "kwargs": {
                    "config": {
                        "feature": self.get_feature_config(),
                        "label": label,
                    },
                    "filter_pipe": filter_pipe,
                    "freq": freq,
                    "inst_processors": inst_processors,
                },
            }
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

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class Alpha158USLead(Alpha158):
    """Alpha158 extended with US market lead-lag features.

    Uses ChangeInstrument operator to load US index data (e.g., S&P 500)
    as cross-sectional features. US market closes at 4AM Beijing time,
    so previous day's US return is a legitimate non-forward-looking feature
    for predicting A-share returns.
    """

    def __init__(self, us_index="GSPC", **kwargs):
        self.us_index = us_index
        super().__init__(**kwargs)

    def get_feature_config(self):
        fields, names = Alpha158DL.get_feature_config(
            {
                "kbar": {},
                "price": {
                    "windows": [0],
                    "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
                },
                "rolling": {},
            }
        )
        idx = self.us_index

        # Previous day US return (core lead-lag signal)
        fields.append(f"ChangeInstrument('{idx}', $close/Ref($close, 1)-1)")
        names.append("US_RET1")

        # 5-day US momentum
        fields.append(f"ChangeInstrument('{idx}', $close/Ref($close, 5)-1)")
        names.append("US_MOM5")

        # 20-day US momentum
        fields.append(f"ChangeInstrument('{idx}', $close/Ref($close, 20)-1)")
        names.append("US_MOM20")

        # US 20-day realized volatility
        fields.append(f"ChangeInstrument('{idx}', Std($close/Ref($close,1)-1, 20))")
        names.append("US_RVOL20")

        # US 5-day mean return (smoothed signal)
        fields.append(f"ChangeInstrument('{idx}', Mean($close/Ref($close,1)-1, 5))")
        names.append("US_MARET5")

        # US RSI-like indicator (10-day)
        fields.append(
            f"ChangeInstrument('{idx}', "
            f"Sum(Greater($close-Ref($close,1),0),10)"
            f"/(Sum(Abs($close-Ref($close,1)),10)+1e-12))"
        )
        names.append("US_RSI10")

        # US 10-day momentum acceleration
        fields.append(
            f"ChangeInstrument('{idx}', "
            f"($close/Ref($close,5)-1)-($close/Ref($close,10)-1))"
        )
        names.append("US_ACCEL_5_10")

        # US trend strength (slope normalized by price)
        fields.append(f"ChangeInstrument('{idx}', Slope($close, 10)/$close)")
        names.append("US_BETA10")

        return fields, names
