from typing import List, Tuple

from quant_master.contrib.data.handler import (
    _DEFAULT_INFER_PROCESSORS,
    _DEFAULT_LEARN_PROCESSORS,
    check_transform_proc,
)
from quant_master.contrib.data.loader import Alpha158DL
from quant_master.data.dataset.handler import DataHandlerLP


class TranscendenceAlpha(DataHandlerLP):
    """Expanded factor handler for Transcendence experiments."""

    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        include_alpha158_base=True,
        benchmark=None,
        **kwargs,
    ):
        self.include_alpha158_base = include_alpha158_base
        self.benchmark = benchmark

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

    def get_feature_config(self) -> Tuple[List[str], List[str]]:
        fields: List[str] = []
        names: List[str] = []

        if self.include_alpha158_base:
            base_fields, base_names = Alpha158DL.get_feature_config(
                {
                    "kbar": {},
                    "price": {
                        "windows": [0, 1, 2, 3, 4, 5, 10, 20],
                        "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
                    },
                    "volume": {"windows": [0, 1, 2, 3, 4, 5, 10, 20]},
                    "rolling": {
                        "windows": [5, 10, 20, 30, 60],
                        "include": [
                            "ROC",
                            "MA",
                            "STD",
                            "BETA",
                            "RSQR",
                            "RESI",
                            "MAX",
                            "LOW",
                            "QTLU",
                            "QTLD",
                            "RANK",
                            "RSV",
                            "IMAX",
                            "IMIN",
                            "IMXD",
                            "CORR",
                            "CORD",
                            "CNTP",
                            "CNTN",
                            "CNTD",
                            "SUMP",
                            "SUMN",
                            "SUMD",
                            "VMA",
                            "VSTD",
                            "WVMA",
                            "VSUMP",
                            "VSUMN",
                            "VSUMD",
                        ],
                    },
                }
            )
            fields.extend(base_fields)
            names.extend(base_names)

        used = set(names)

        def add(name: str, expr: str):
            if name in used:
                return
            names.append(name)
            fields.append(expr)
            used.add(name)

        ret = "$close/Ref($close, 1)-1"
        vret = "Log($volume+1)-Log(Ref($volume, 1)+1)"
        benchmark = str(self.benchmark).strip() if self.benchmark else ""
        has_benchmark = bool(benchmark)
        mret = f"ChangeInstrument('{benchmark}', $close/Ref($close, 1)-1)" if has_benchmark else None

        # 1) multi-period momentum / reversal
        for d in [3, 5, 10, 20, 40, 60, 120]:
            add(f"TX_MOM_{d}", f"$close/Ref($close, {d})-1")
            add(f"TX_REV_{d}", f"Ref($close, {d})/$close-1")
            add(f"TX_VWAP_MOM_{d}", f"$vwap/Ref($vwap, {d})-1")
            add(f"TX_VOL_MOM_{d}", f"Log($volume+1)-Log(Ref($volume, {d})+1)")
            add(f"TX_MOM_RV_{d}", f"($close/Ref($close, {d})-1)/(Std({ret}, {d})+1e-12)")
            add(f"TX_MOM_ACCEL_{d}", f"($close/Ref($close, {d})-1)-(Ref($close, {d})/Ref($close, {2 * d})-1)")

        for short_w, long_w in [(5, 20), (10, 40), (20, 60), (20, 120), (40, 120)]:
            add(
                f"TX_MOM_SPREAD_{short_w}_{long_w}",
                f"($close/Ref($close, {short_w})-1)-($close/Ref($close, {long_w})-1)",
            )

        # 2) volume-price divergence
        for d in [5, 10, 20, 40, 60]:
            add(f"TX_PV_DIV_{d}", f"Mean({ret}, {d})-Mean({vret}, {d})")
            add(f"TX_PV_CORR_{d}", f"Corr({ret}, {vret}, {d})")
            add(f"TX_PV_COV_{d}", f"Cov({ret}, {vret}, {d})")
            add(f"TX_PV_LAGCORR_{d}", f"Corr({ret}, Ref({vret}, 1), {d})")
            add(
                f"TX_PV_IMBAL_{d}",
                f"Mean(Abs({ret}), {d})/(Mean(Abs({vret}), {d})+1e-12)",
            )

        # 3) volatility compression / expansion
        for d in [5, 10, 20, 40, 60]:
            add(f"TX_RVOL_{d}", f"Std({ret}, {d})")
            add(f"TX_RANGE_VOL_{d}", f"Std(($high-$low)/($close+1e-12), {d})")
            add(f"TX_AMP_{d}", f"Mean(($high-$low)/(Ref($close, 1)+1e-12), {d})")

        for short_w, long_w in [(5, 20), (10, 40), (20, 60)]:
            add(
                f"TX_VOL_REGIME_{short_w}_{long_w}",
                f"Std({ret}, {short_w})/(Std({ret}, {long_w})+1e-12)",
            )
            add(
                f"TX_AMP_REGIME_{short_w}_{long_w}",
                f"Mean(($high-$low)/($close+1e-12), {short_w})/(Mean(($high-$low)/($close+1e-12), {long_w})+1e-12)",
            )

        # 4) price-volume correlation and liquidity proxies
        for d in [5, 10, 20, 40, 60]:
            add(f"TX_RET_VOL_CORR_{d}", f"Corr({ret}, Log($volume+1), {d})")
            add(f"TX_ABSRET_VRET_CORR_{d}", f"Corr(Abs({ret}), {vret}, {d})")
            add(
                f"TX_AMIHUD_{d}",
                f"Mean(Abs({ret})/($close*$volume+1e-12), {d})",
            )
            add(f"TX_DVOL_Z_{d}", f"($close*$volume)/(Mean($close*$volume, {d})+1e-12)-1")
            add(f"TX_TURN_Z_{d}", f"$volume/(Mean($volume, {d})+1e-12)-1")

        # 5) skewness / tail risk
        for d in [10, 20, 40, 60]:
            add(f"TX_SKEW_{d}", f"Skew({ret}, {d})")
            add(f"TX_KURT_{d}", f"Kurt({ret}, {d})")
            add(f"TX_DOWNSIDE_{d}", f"Std(Less({ret}, 0), {d})/(Std({ret}, {d})+1e-12)")
            add(f"TX_TAIL_Q05_{d}", f"Quantile({ret}, {d}, 0.05)")
            add(f"TX_TAIL_Q95_{d}", f"Quantile({ret}, {d}, 0.95)")
            add(f"TX_LEFT_TAIL_GAP_{d}", f"Med({ret}, {d})-Quantile({ret}, {d}, 0.1)")

        # 6) market-relative strength (if benchmark data exists)
        if has_benchmark:
            for d in [5, 10, 20, 40, 60, 120]:
                add(f"TX_EXCESS_RET_{d}", f"Mean({ret}, {d})-ChangeInstrument('{benchmark}', Mean({ret}, {d}))")
                add(
                    f"TX_REL_MOM_{d}",
                    f"($close/Ref($close, {d}))/(ChangeInstrument('{benchmark}', $close/Ref($close, {d}))+1e-12)-1",
                )
                add(
                    f"TX_BETA_{d}",
                    f"Cov({ret}, {mret}, {d})/(ChangeInstrument('{benchmark}', Var({mret}, {d}))+1e-12)",
                )
                add(f"TX_IDIO_VOL_{d}", f"Std({ret}, {d})-Abs(Cov({ret}, {mret}, {d}))")

        return fields, names

    @staticmethod
    def get_label_config():
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
