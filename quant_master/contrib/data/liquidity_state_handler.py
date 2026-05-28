from typing import List, Tuple

from quant_master.contrib.data.handler import Alpha158


class Alpha158LiquidityState(Alpha158):
    """Alpha158 plus past/current liquidity and state proxies."""

    EXTRA_FEATURE_PREFIX = "LS_"

    @staticmethod
    def get_liquidity_state_feature_config() -> Tuple[List[str], List[str]]:
        ret = "$close/Ref($close, 1)-1"
        volume_ret = "Log($volume+1)-Log(Ref($volume, 1)+1)"
        dollar_volume = "$close*$volume"
        spread_proxy = "Abs($vwap-$close)/($close+1e-12)"
        range_proxy = "($high-$low)/($close+1e-12)"

        fields = [
            spread_proxy,
            f"Mean({spread_proxy}, 5)",
            f"Mean({spread_proxy}, 20)",
            f"({dollar_volume})/(Mean({dollar_volume}, 5)+1e-12)-1",
            f"({dollar_volume})/(Mean({dollar_volume}, 20)+1e-12)-1",
            "$volume/(Mean($volume, 5)+1e-12)-1",
            "$volume/(Mean($volume, 20)+1e-12)-1",
            f"Mean(Abs({ret})/({dollar_volume}+1e-12), 5)",
            f"Mean(Abs({ret})/({dollar_volume}+1e-12), 20)",
            f"Mean({range_proxy}, 5)",
            f"Mean({range_proxy}, 20)",
            f"Std({range_proxy}, 20)",
            f"Std({ret}, 20)",
            f"Std({ret}, 60)",
            f"Corr({ret}, Log($volume+1), 20)",
            f"Corr(Abs({ret}), Log($volume+1), 20)",
            f"Corr({ret}, {volume_ret}, 20)",
        ]
        names = [
            "LS_SPREAD_1",
            "LS_SPREAD_MA5",
            "LS_SPREAD_MA20",
            "LS_DVOL_REL5",
            "LS_DVOL_REL20",
            "LS_VOLUME_REL5",
            "LS_VOLUME_REL20",
            "LS_AMIHUD_5",
            "LS_AMIHUD_20",
            "LS_RANGE_MA5",
            "LS_RANGE_MA20",
            "LS_RANGE_VOL20",
            "LS_RET_VOL20",
            "LS_RET_VOL60",
            "LS_RET_VOLUME_CORR20",
            "LS_ABSRET_VOLUME_CORR20",
            "LS_RET_VOLCHG_CORR20",
        ]
        return fields, names

    def get_feature_config(self) -> Tuple[List[str], List[str]]:
        fields, names = super().get_feature_config()
        extra_fields, extra_names = self.get_liquidity_state_feature_config()
        return list(fields) + extra_fields, list(names) + extra_names
