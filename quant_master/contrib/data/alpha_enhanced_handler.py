# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Alpha158Enhanced: Alpha158 + 10 orthogonal alpha factors + optional US lead-lag.

New factor families (all use daily OHLCV only):
1. Overnight return (OVERNIGHT_RET)
2. VWAP deviation (VWAP_DEV)
3. Drawdown from high (DD_20, DD_60)
4. Recovery from low (REC_10, REC_20)
5. EMA crossover (EMA_CROSS_5_20, EMA_CROSS_10_40)
6. Return skewness (RET_SKEW_20, RET_SKEW_60)
7. Amihud illiquidity (AMIHUD_20)
8. Volume shock z-score (VOL_Z_10, VOL_Z_20)
9. Range compression (RANGE_COMP_5_20)
10. OBV trend (OBV_TREND_20)
"""

from quant_master.contrib.data.loader import Alpha158DL
from quant_master.contrib.data.handler import Alpha158, check_transform_proc
from ...data.dataset.handler import DataHandlerLP


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]


class Alpha158Enhanced(DataHandlerLP):
    """Alpha158 + 10 orthogonal alpha factor families (~180 total features).

    All new factors use only daily OHLCV data and standard expression operators.
    No external data sources or intraday data required.
    """

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
        us_index=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QuantMasterDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(us_index=us_index),
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
    def get_feature_config(us_index=None):
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

        # === NEW ALPHA FACTORS ===

        # 1. Overnight return: gap between previous close and today's open
        fields.append("$open/Ref($close, 1)-1")
        names.append("OVERNIGHT_RET")

        # 2. VWAP deviation: close vs volume-weighted average price
        fields.append("($close-$vwap)/($vwap+1e-12)")
        names.append("VWAP_DEV")

        # 3. Drawdown from high (bounded in [-1, 0])
        fields.append("$close/Max($high, 20)-1")
        names.append("DD_20")
        fields.append("$close/Max($high, 60)-1")
        names.append("DD_60")

        # 4. Recovery from low (bounded in [0, +inf))
        fields.append("$close/Min($low, 10)-1")
        names.append("REC_10")
        fields.append("$close/Min($low, 20)-1")
        names.append("REC_20")

        # 5. EMA crossover (smooth trend signal, less noisy than SMA)
        fields.append("EMA($close, 5)/EMA($close, 20)-1")
        names.append("EMA_CROSS_5_20")
        fields.append("EMA($close, 10)/EMA($close, 40)-1")
        names.append("EMA_CROSS_10_40")

        # 6. Return skewness (tail risk asymmetry)
        fields.append("Skew($close/Ref($close, 1)-1, 20)")
        names.append("RET_SKEW_20")
        fields.append("Skew($close/Ref($close, 1)-1, 60)")
        names.append("RET_SKEW_60")

        # 7. Amihud illiquidity ratio (price impact per unit volume)
        fields.append("Mean(Abs($close/Ref($close, 1)-1)/($close*$volume+1e-12), 20)")
        names.append("AMIHUD_20")

        # 8. Volume shock z-score
        fields.append("($volume-Mean($volume, 10))/(Std($volume, 10)+1e-12)")
        names.append("VOL_Z_10")
        fields.append("($volume-Mean($volume, 20))/(Std($volume, 20)+1e-12)")
        names.append("VOL_Z_20")

        # 9. Price range compression (short/long volatility ratio)
        fields.append(
            "Mean(($high-$low)/($close+1e-12), 5)"
            "/(Mean(($high-$low)/($close+1e-12), 20)+1e-12)"
        )
        names.append("RANGE_COMP_5_20")

        # 10. OBV trend (accumulation/distribution)
        fields.append(
            "Sum(Sign($close-Ref($close, 1))*$volume, 20)"
            "/(Mean($volume, 20)+1e-12)"
        )
        names.append("OBV_TREND_20")

        # Optional: US market lead-lag features
        if us_index:
            idx = us_index
            fields.append(f"ChangeInstrument('{idx}', $close/Ref($close, 1)-1)")
            names.append("US_RET1")
            fields.append(f"ChangeInstrument('{idx}', $close/Ref($close, 5)-1)")
            names.append("US_MOM5")
            fields.append(f"ChangeInstrument('{idx}', $close/Ref($close, 20)-1)")
            names.append("US_MOM20")
            fields.append(f"ChangeInstrument('{idx}', Std($close/Ref($close,1)-1, 20))")
            names.append("US_RVOL20")
            fields.append(f"ChangeInstrument('{idx}', Mean($close/Ref($close,1)-1, 5))")
            names.append("US_MARET5")

        return fields, names
