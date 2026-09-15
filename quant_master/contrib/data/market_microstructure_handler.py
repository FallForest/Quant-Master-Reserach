# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Low-cost daily market-microstructure proxies for the local TDX dataset."""

import re
from typing import Iterable, List, Optional, Tuple

from quant_master.contrib.data.amount_flow_handler import Alpha158TDX
from quant_master.contrib.data.feature_registry import FeatureRegistry, FeatureSpec


class Alpha158MarketMicrostructureProxy(Alpha158TDX):
    """Alpha158TDX plus causal proxies derived from daily OHLCV and amount.

    These features estimate effects that normally require intraday trades or an
    order book.  The ``MSP_`` prefix makes that limitation explicit.  In
    particular, the limit-distance columns use fixed +/-10% reference levels;
    they are not exchange limit flags and do not infer ST, 20%, or 30% regimes.

    Parameters
    ----------
    benchmark : str, optional
        Instrument used for the optional market-beta and idiosyncratic-volatility
        proxies, for example ``SH000300``.
    feature_families : iterable of str, optional
        Include only these registered proxy families.  ``None`` includes all.
    exclude_feature_families : iterable of str
        Registered proxy families to remove after inclusion filtering.
    feature_names : iterable of str, optional
        Include only these registered feature columns after family filtering.
    """

    EXTRA_FEATURE_PREFIX = "MSP_"
    _BENCHMARK_PATTERN = re.compile(r"^[A-Za-z0-9._^-]+$")

    def __init__(
        self,
        benchmark: Optional[str] = None,
        feature_families: Optional[Iterable[str]] = None,
        exclude_feature_families: Iterable[str] = (),
        feature_names: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> None:
        self.benchmark = benchmark
        self.feature_families = (
            feature_families
            if feature_families is None or isinstance(feature_families, str)
            else tuple(feature_families)
        )
        self.exclude_feature_families = (
            exclude_feature_families
            if isinstance(exclude_feature_families, str)
            else tuple(exclude_feature_families)
        )
        self.feature_names = (
            feature_names if feature_names is None or isinstance(feature_names, str) else tuple(feature_names)
        )
        super().__init__(**kwargs)

    @classmethod
    def get_market_microstructure_feature_registry(cls, benchmark: Optional[str] = None) -> FeatureRegistry:
        """Return proxy expressions together with family and availability metadata."""

        ret = "($close/(Ref($close,1)+1e-12)-1)"
        lagged_ret = f"Ref({ret},1)"
        log_hl = "Log(($high+1e-12)/($low+1e-12))"
        log_co = "Log(($close+1e-12)/($open+1e-12))"

        roll_cov = f"Cov({ret},{lagged_ret},20)"
        roll_spread = f"If({roll_cov}<0,2*Power(0-{roll_cov},0.5),0)"

        two_day_beta = f"Power({log_hl},2)+Power(Ref({log_hl},1),2)"
        two_day_high = "Greater($high,Ref($high,1))"
        two_day_low = "Less($low,Ref($low,1))"
        two_day_gamma = f"Power(Log(({two_day_high}+1e-12)/({two_day_low}+1e-12)),2)"
        cs_denominator = "0.1715728752538097"
        cs_alpha = (
            f"(Power(2*({two_day_beta}),0.5)-Power(({two_day_beta}),0.5))/{cs_denominator}"
            f"-Power(({two_day_gamma})/{cs_denominator},0.5)"
        )
        cs_exp_alpha = f"Power(2.718281828459045,({cs_alpha}))"
        cs_daily_spread = f"If(({cs_alpha})>0,2*({cs_exp_alpha}-1)/(1+{cs_exp_alpha}),0)"

        parkinson_var = f"Mean(Power({log_hl},2),20)/2.772588722239781"
        gk_daily_var = f"0.5*Power({log_hl},2)-0.3862943611198906*Power({log_co},2)"
        gk_var = f"Mean(({gk_daily_var}),20)"
        rs_daily_var = (
            "Log(($high+1e-12)/($close+1e-12))*Log(($high+1e-12)/($open+1e-12))"
            "+Log(($low+1e-12)/($close+1e-12))*Log(($low+1e-12)/($open+1e-12))"
        )
        rs_var = f"Mean(({rs_daily_var}),20)"

        realized_var = f"Sum(Power({ret},2),20)"
        bipower_var = f"1.5707963267948966*Sum(Abs({ret})*Abs({lagged_ret}),20)"
        jump_var = f"({realized_var})-({bipower_var})"

        specs = [
            FeatureSpec(
                name="MSP_ROLL_SPREAD20",
                family="roll_spread",
                descriptor=roll_spread,
                window=20,
                data_requirements=("close",),
                available_lag=0,
                source="tdx_daily_proxy",
            ),
            FeatureSpec(
                name="MSP_CORWIN_SCHULTZ20",
                family="corwin_schultz_spread",
                descriptor=f"Mean(({cs_daily_spread}),20)",
                window=20,
                data_requirements=("high", "low"),
                available_lag=0,
                source="tdx_daily_proxy",
            ),
            FeatureSpec(
                name="MSP_PARKINSON20",
                family="parkinson_volatility",
                descriptor=f"Power(({parkinson_var}),0.5)",
                window=20,
                data_requirements=("high", "low"),
                available_lag=0,
                source="tdx_daily_proxy",
            ),
            FeatureSpec(
                name="MSP_GARMAN_KLASS20",
                family="garman_klass_volatility",
                descriptor=f"If(({gk_var})>0,Power(({gk_var}),0.5),0)",
                window=20,
                data_requirements=("open", "high", "low", "close"),
                available_lag=0,
                source="tdx_daily_proxy",
            ),
            FeatureSpec(
                name="MSP_ROGERS_SATCHELL20",
                family="rogers_satchell_volatility",
                descriptor=f"If(({rs_var})>0,Power(({rs_var}),0.5),0)",
                window=20,
                data_requirements=("open", "high", "low", "close"),
                available_lag=0,
                source="tdx_daily_proxy",
            ),
            FeatureSpec(
                name="MSP_ZERO_RET20",
                family="price_staleness",
                descriptor=f"Mean(Abs({ret})<1e-6,20)",
                window=20,
                data_requirements=("close",),
                available_lag=0,
                source="tdx_daily_proxy",
            ),
            FeatureSpec(
                name="MSP_UPLIMIT10_DIST",
                family="price_limit_distance",
                descriptor="(1.1*Ref($close,1)-$high)/(Ref($close,1)+1e-12)",
                window=1,
                data_requirements=("high", "close"),
                available_lag=0,
                source="tdx_daily_proxy",
            ),
            FeatureSpec(
                name="MSP_DOWNLIMIT10_DIST",
                family="price_limit_distance",
                descriptor="($low-0.9*Ref($close,1))/(Ref($close,1)+1e-12)",
                window=1,
                data_requirements=("low", "close"),
                available_lag=0,
                source="tdx_daily_proxy",
            ),
            FeatureSpec(
                name="MSP_RET_AUTOCORR20",
                family="return_dependence",
                descriptor=f"Corr({ret},{lagged_ret},20)",
                window=20,
                data_requirements=("close",),
                available_lag=0,
                source="tdx_daily_proxy",
            ),
            FeatureSpec(
                name="MSP_JUMP_SHARE20",
                family="jump_risk",
                descriptor=f"If(({jump_var})>0,({jump_var})/(({realized_var})+1e-12),0)",
                window=20,
                data_requirements=("close",),
                available_lag=0,
                source="tdx_daily_proxy",
            ),
        ]

        if benchmark is not None:
            if not isinstance(benchmark, str) or not benchmark.strip():
                raise ValueError("benchmark must be a non-empty instrument string or None")
            benchmark = benchmark.strip()
            if cls._BENCHMARK_PATTERN.fullmatch(benchmark) is None:
                raise ValueError("benchmark contains unsupported characters")
            market_ret = f"ChangeInstrument({benchmark!r},$close/(Ref($close,1)+1e-12)-1)"
            market_var = f"Var({market_ret},20)"
            stock_market_cov = f"Cov({ret},{market_ret},20)"
            idiosyncratic_var = f"Var({ret},20)-Power({stock_market_cov},2)/(({market_var})+1e-12)"
            specs.extend(
                [
                    FeatureSpec(
                        name="MSP_MARKET_BETA20",
                        family="benchmark_risk",
                        descriptor=f"({stock_market_cov})/(({market_var})+1e-12)",
                        window=20,
                        data_requirements=("close",),
                        available_lag=0,
                        source=f"tdx_daily_proxy:{benchmark}",
                    ),
                    FeatureSpec(
                        name="MSP_IDIO_VOL20",
                        family="benchmark_risk",
                        descriptor=f"If(({idiosyncratic_var})>0,Power(({idiosyncratic_var}),0.5),0)",
                        window=20,
                        data_requirements=("close",),
                        available_lag=0,
                        source=f"tdx_daily_proxy:{benchmark}",
                    ),
                ]
            )

        return FeatureRegistry(specs)

    @classmethod
    def get_market_microstructure_feature_config(
        cls,
        benchmark: Optional[str] = None,
        feature_families: Optional[Iterable[str]] = None,
        exclude_feature_families: Iterable[str] = (),
        feature_names: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Return loader-ready proxy expressions, optionally filtered by family."""

        registry = cls.get_market_microstructure_feature_registry(benchmark=benchmark)
        selected = registry.select(families=feature_families, exclude_families=exclude_feature_families)
        if feature_names is not None:
            requested = (feature_names,) if isinstance(feature_names, str) else tuple(feature_names)
            if any(not isinstance(name, str) or not name.strip() for name in requested):
                raise ValueError("feature_names must contain non-empty strings")
            requested_names = {name.strip() for name in requested}
            unknown = requested_names.difference(spec.name for spec in registry)
            if unknown:
                raise ValueError(f"unknown feature name(s): {', '.join(sorted(unknown))}")
            selected = tuple(spec for spec in selected if spec.name in requested_names)
        return FeatureRegistry(selected).expand_columns()

    def get_feature_config(self) -> Tuple[List[str], List[str]]:
        fields, names = super().get_feature_config()
        extra_fields, extra_names = self.get_market_microstructure_feature_config(
            benchmark=self.benchmark,
            feature_families=self.feature_families,
            exclude_feature_families=self.exclude_feature_families,
            feature_names=self.feature_names,
        )
        return list(fields) + extra_fields, list(names) + extra_names


__all__ = ("Alpha158MarketMicrostructureProxy",)
