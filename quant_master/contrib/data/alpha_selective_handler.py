# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Alpha158Selective: Alpha158 + 4 carefully selected alpha factors.

Only adds factors with highest expected IC improvement and lowest risk:
1. Overnight return (OVERNIGHT_RET) — unique info channel
2. VWAP deviation (VWAP_DEV) — intraday order flow
3. EMA crossover (EMA_CROSS_5_20) — smooth trend
4. Return skewness (RET_SKEW_20) — tail risk
"""

from quant_master.contrib.data.loader import Alpha158DL
from quant_master.contrib.data.handler import Alpha158, check_transform_proc
from ...data.dataset.handler import DataHandlerLP


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]


class Alpha158Selective(DataHandlerLP):
    """Alpha158 + 4 selectively chosen alpha factors (~162 total features)."""

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
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QuantMasterDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
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

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    @staticmethod
    def get_feature_config():
        # Base Alpha158 features
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

        # 1. Overnight return: gap between previous close and today's open
        fields.append("$open/Ref($close, 1)-1")
        names.append("OVERNIGHT_RET")

        # 2. VWAP deviation: close vs volume-weighted average price
        fields.append("($close-$vwap)/($vwap+1e-12)")
        names.append("VWAP_DEV")

        # 3. EMA crossover (smooth trend signal)
        fields.append("EMA($close, 5)/EMA($close, 20)-1")
        names.append("EMA_CROSS_5_20")

        # 4. Return skewness (tail risk asymmetry)
        fields.append("Skew($close/Ref($close, 1)-1, 20)")
        names.append("RET_SKEW_20")

        return fields, names
