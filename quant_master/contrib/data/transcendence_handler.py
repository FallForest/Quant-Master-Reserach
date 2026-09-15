import re
from typing import Dict, Iterable, List, Mapping, Tuple

from quant_master.contrib.data.handler import (
    _DEFAULT_LEARN_PROCESSORS,
    check_transform_proc,
)
from quant_master.contrib.data.loader import Alpha158DL
from quant_master.data.dataset.handler import DataHandlerLP


_TRANSCENDENCE_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]


class TranscendenceAlpha(DataHandlerLP):
    """Expanded factor handler for Transcendence experiments."""

    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_TRANSCENDENCE_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        include_alpha158_base=True,
        benchmark=None,
        index_exposures: Iterable[str] = (),
        industry_exposures: Mapping[str, str] = None,
        exposure_windows: Iterable[int] = (20, 60),
        **kwargs,
    ):
        self.include_alpha158_base = include_alpha158_base
        self.benchmark = benchmark
        self.index_exposures = self._validate_exposure_instruments(index_exposures, "index_exposures")
        self.industry_exposures = self._validate_industry_exposures(industry_exposures or {})
        self.exposure_windows = self._validate_exposure_windows(exposure_windows)
        self._validate_exposure_labels()

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

        # 7) configurable index and industry exposures.  Instruments are
        # resolved through ChangeInstrument, keeping the feature path causal
        # and compatible with the local provider.  ``industry_exposures`` is a
        # mapping from a stable label (e.g. ``bank``) to an index instrument.
        exposure_specs = [("IDX", str(inst), str(inst)) for inst in self.index_exposures]
        exposure_specs.extend(("IND", label, inst) for label, inst in self.industry_exposures.items())
        for kind, label, instrument in exposure_specs:
            safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper() or "EXPOSURE"
            market_ret = f"ChangeInstrument({instrument!r},$close/(Ref($close,1)+1e-12)-1)"
            for d in self.exposure_windows:
                add(
                    f"TX_{kind}_EXCESS_RET_{safe_label}_{d}",
                    f"Mean({ret},{d})-ChangeInstrument({instrument!r},Mean({ret},{d}))",
                )
                add(
                    f"TX_{kind}_BETA_{safe_label}_{d}",
                    f"Cov({ret},{market_ret},{d})/(Var({market_ret},{d})+1e-12)",
                )

        return fields, names

    @staticmethod
    def _validate_exposure_instruments(values: Iterable[str], name: str) -> Tuple[str, ...]:
        if values is None:
            return ()
        if isinstance(values, str):
            values = (values,)
        try:
            raw_values = tuple(values)
        except TypeError as exc:
            raise ValueError(f"{name} must contain non-empty instrument symbols") from exc
        if any(not isinstance(value, str) for value in raw_values):
            raise ValueError(f"{name} must contain non-empty instrument symbols")
        result = tuple(value.strip() for value in raw_values)
        if any(not v or re.fullmatch(r"[A-Za-z0-9._^-]+", v) is None for v in result):
            raise ValueError(f"{name} must contain non-empty instrument symbols")
        return result

    @classmethod
    def _validate_industry_exposures(cls, values: Mapping[str, str]) -> Dict[str, str]:
        if not isinstance(values, Mapping):
            raise ValueError("industry_exposures must be a mapping of label to instrument")
        result = {}
        for key, value in values.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("industry_exposures must map string labels to instrument symbols")
            label, instrument = key.strip(), value.strip()
            if not label:
                raise ValueError("industry_exposures labels must be non-empty strings")
            if not instrument or re.fullmatch(r"[A-Za-z0-9._^-]+", instrument) is None:
                raise ValueError("industry_exposures values must be valid instrument symbols")
            result[label] = instrument
        return result

    @staticmethod
    def _validate_exposure_windows(values: Iterable[int]) -> Tuple[int, ...]:
        if isinstance(values, int) and not isinstance(values, bool):
            values = (values,)
        if isinstance(values, (str, bytes)):
            raise ValueError("exposure_windows must contain positive integers")
        try:
            result = tuple(values)
        except TypeError as exc:
            raise ValueError("exposure_windows must contain positive integers") from exc
        if not result or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in result):
            raise ValueError("exposure_windows must contain positive integers")
        if len(set(result)) != len(result):
            raise ValueError("exposure_windows must not contain duplicate values")
        return result

    def _validate_exposure_labels(self) -> None:
        labels = list(self.index_exposures) + list(self.industry_exposures)
        normalized = [re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper() or "EXPOSURE" for label in labels]
        if len(set(normalized)) != len(normalized):
            raise ValueError("index_exposures and industry_exposures must have distinct normalized labels")

    @staticmethod
    def get_label_config():
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
