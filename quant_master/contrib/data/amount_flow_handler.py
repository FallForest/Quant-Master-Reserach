# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Alpha158 features backed by the fields available in the local TDX dataset."""

from typing import List, Tuple

from quant_master.contrib.data.handler import Alpha158, Alpha360


class Alpha360TDX(Alpha360):
    """Alpha360 with a causal VWAP proxy backed by native TDX fields."""

    VWAP_PROXY = "($amount/($volume*100+1e-12))"

    @staticmethod
    def get_feature_config() -> Tuple[List[str], List[str]]:
        from quant_master.contrib.data.loader import Alpha360DL

        fields, names = Alpha360DL.get_feature_config()
        return [field.replace("$vwap", Alpha360TDX.VWAP_PROXY) for field in fields], list(names)


class Alpha158TDX(Alpha158):
    """Alpha158 with a causal VWAP proxy backed by native TDX fields."""

    VWAP_PROXY = "($amount/($volume*100+1e-12))"

    def get_feature_config(self) -> Tuple[List[str], List[str]]:
        fields, names = super().get_feature_config()
        return [field.replace("$vwap", self.VWAP_PROXY) for field in fields], list(names)


class Alpha158AmountFlow(Alpha158TDX):
    """Alpha158 plus amount, liquidity, and close-to-open structure features.

    The TDX provider stores volume in lots of 100 shares.  It does not store a
    native ``$vwap`` series, so amount divided by share volume is used as the
    daily VWAP proxy.  All extra expressions use current or past observations;
    they are intended for close-of-day inference and next-session execution.
    """

    EXTRA_FEATURE_PREFIX = "AF_"
    VWAP_PROXY = "($amount/($volume*100+1e-12))"
    RECOMMENDED_FEATURES = (
        "AF_RANGE_VOL20",
        "AF_AMOUNT_REL60",
        "AF_OVERNIGHT_INTRADAY_SPREAD",
        "AF_RET_AMOUNT_CORR60",
        "AF_SIGNED_AMOUNT20",
        "AF_RET_AMOUNT_CORR20",
        "AF_SIGNED_AMOUNT5",
        "AF_RET_AMOUNT_CORR10",
        "AF_AMOUNT_MOM20",
        "AF_AMOUNT_MOM5",
    )

    @classmethod
    def get_amount_flow_feature_config(cls) -> Tuple[List[str], List[str]]:
        ret = "($close/Ref($close, 1)-1)"
        abs_ret = f"Abs({ret})"
        log_amount = "Log($amount+1)"
        vwap = cls.VWAP_PROXY

        fields = [
            f"($close-{vwap})/({vwap}+1e-12)",
            "$amount/(Mean($amount, 5)+1e-12)-1",
            "$amount/(Mean($amount, 20)+1e-12)-1",
            "$amount/(Mean($amount, 60)+1e-12)-1",
            f"({log_amount}-Mean({log_amount}, 20))/(Std({log_amount}, 20)+1e-12)",
            f"({log_amount}-Mean({log_amount}, 60))/(Std({log_amount}, 60)+1e-12)",
            f"{log_amount}-Log(Ref($amount, 5)+1)",
            f"{log_amount}-Log(Ref($amount, 20)+1)",
            f"Mean({abs_ret}/($amount+1e-12), 5)",
            f"Mean({abs_ret}/($amount+1e-12), 20)",
            f"Mean({abs_ret}/($amount+1e-12), 60)",
            f"Corr({ret}, {log_amount}, 10)",
            f"Corr({ret}, {log_amount}, 20)",
            f"Corr({ret}, {log_amount}, 60)",
            f"Corr({abs_ret}, {log_amount}, 20)",
            f"Mean(Sign({ret})*$amount, 5)/(Mean($amount, 5)+1e-12)",
            f"Mean(Sign({ret})*$amount, 20)/(Mean($amount, 20)+1e-12)",
            "$open/Ref($close, 1)-1",
            "($open/Ref($close, 1)-1)-($close/$open-1)",
            f"Std({ret}, 5)/(Std({ret}, 20)+1e-12)",
            "Std(Log(($high+1e-12)/($low+1e-12)), 20)",
        ]
        names = [
            "AF_CLOSE_VWAP_DEV",
            "AF_AMOUNT_REL5",
            "AF_AMOUNT_REL20",
            "AF_AMOUNT_REL60",
            "AF_AMOUNT_Z20",
            "AF_AMOUNT_Z60",
            "AF_AMOUNT_MOM5",
            "AF_AMOUNT_MOM20",
            "AF_AMIHUD5",
            "AF_AMIHUD20",
            "AF_AMIHUD60",
            "AF_RET_AMOUNT_CORR10",
            "AF_RET_AMOUNT_CORR20",
            "AF_RET_AMOUNT_CORR60",
            "AF_ABSRET_AMOUNT_CORR20",
            "AF_SIGNED_AMOUNT5",
            "AF_SIGNED_AMOUNT20",
            "AF_OVERNIGHT_RET",
            "AF_OVERNIGHT_INTRADAY_SPREAD",
            "AF_VOL_REGIME5_20",
            "AF_RANGE_VOL20",
        ]
        return fields, names

    def get_feature_config(self) -> Tuple[List[str], List[str]]:
        fields, names = super().get_feature_config()
        fields = list(fields)
        names = list(names)

        candidate_fields, candidate_names = self.get_amount_flow_feature_config()
        candidates = dict(zip(candidate_names, candidate_fields))
        extra_names = list(self.RECOMMENDED_FEATURES)
        extra_fields = [candidates[name] for name in extra_names]
        return fields + extra_fields, names + extra_names


class Alpha158AmountFlowCore(Alpha158AmountFlow):
    """Two-feature ablation selected by the Alpha168 model importance scan."""

    RECOMMENDED_FEATURES = (
        "AF_RANGE_VOL20",
        "AF_OVERNIGHT_INTRADAY_SPREAD",
    )


class Alpha158AmountFlowCoreBaseline(Alpha158AmountFlowCore):
    """Baseline-preserving ablation that only appends the two core signals.

    This class intentionally leaves Alpha158's native VWAP0 expression
    unchanged so the experiment isolates the incremental signals from the TDX
    VWAP proxy correction.
    """

    def get_feature_config(self) -> Tuple[List[str], List[str]]:
        fields, names = Alpha158.get_feature_config(self)
        candidate_fields, candidate_names = self.get_amount_flow_feature_config()
        candidates = dict(zip(candidate_names, candidate_fields))
        extra_names = list(self.RECOMMENDED_FEATURES)
        return list(fields) + [candidates[name] for name in extra_names], list(names) + extra_names
