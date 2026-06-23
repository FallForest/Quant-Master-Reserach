"""共享工具函数：符号归一化、费用计算、行情查询。"""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if not s:
        return ""
    if s.startswith(("SH", "SZ", "BJ")):
        return s
    if s.startswith("6"):
        return "SH" + s
    if s.startswith(("4", "8", "9")):
        return "BJ" + s
    return "SZ" + s


DEFAULT_FEE_SETTINGS = {
    "stock_commission_rate": 0.0001,
    "etf_commission_rate": 0.00005,
    "stamp_duty_rate": 0.0005,
    "sh_transfer_fee_rate": 0.00001,
}


def fee_settings_from_raw(raw: dict) -> dict:
    return {
        "stock_commission_rate": float(raw.get("stock_commission_rate", DEFAULT_FEE_SETTINGS["stock_commission_rate"])),
        "etf_commission_rate": float(raw.get("etf_commission_rate", DEFAULT_FEE_SETTINGS["etf_commission_rate"])),
        "stamp_duty_rate": float(raw.get("stamp_duty_rate", DEFAULT_FEE_SETTINGS["stamp_duty_rate"])),
        "sh_transfer_fee_rate": float(raw.get("sh_transfer_fee_rate", DEFAULT_FEE_SETTINGS["sh_transfer_fee_rate"])),
    }


def is_etf(instrument: str) -> bool:
    normalized = normalize_symbol(instrument)
    code6 = normalized[2:] if len(normalized) > 2 else normalized
    return code6.startswith(("15", "16", "18", "50", "51", "56", "58"))


def is_shanghai_market(instrument: str) -> bool:
    normalized = normalize_symbol(instrument)
    return normalized.startswith("SH")


def calc_trade_fee(instrument: str, amount: float, side: str, fee_settings: dict) -> dict:
    normalized_side = str(side or "").strip().lower()
    trade_amount = abs(float(amount or 0))
    if trade_amount <= 0:
        return {"commission": 0.0, "stampDuty": 0.0, "transferFee": 0.0, "total": 0.0}
    is_etf_inst = is_etf(instrument)
    commission_rate = fee_settings["etf_commission_rate"] if is_etf_inst else fee_settings["stock_commission_rate"]
    commission = trade_amount * commission_rate
    stamp_duty = trade_amount * fee_settings["stamp_duty_rate"] if (not is_etf_inst and normalized_side == "sell") else 0.0
    transfer_fee = trade_amount * fee_settings["sh_transfer_fee_rate"] if is_shanghai_market(instrument) else 0.0
    total = commission + stamp_duty + transfer_fee
    return {
        "commission": round(commission, 2),
        "stampDuty": round(stamp_duty, 2),
        "transferFee": round(transfer_fee, 2),
        "total": round(total, 2),
    }


def lookup_name(names: dict, instrument: str) -> str:
    if not names:
        return instrument
    normalized = normalize_symbol(instrument)
    code6 = normalized[2:] if len(normalized) > 2 else normalized
    return (
        names.get(instrument)
        or names.get(instrument.upper())
        or names.get(normalized)
        or names.get(normalized.lower())
        or names.get(code6)
        or instrument
    )


def lookup_quote_price(quotes: dict, instrument: str, fallback: float) -> float:
    if not quotes:
        return fallback
    normalized = normalize_symbol(instrument)
    code6 = normalized[2:] if len(normalized) > 2 else normalized
    quote = (
        quotes.get(instrument)
        or quotes.get(instrument.upper())
        or quotes.get(normalized)
        or quotes.get(normalized.lower())
        or quotes.get(code6)
    )
    if not isinstance(quote, dict):
        return fallback
    price = quote.get("price", fallback)
    try:
        price = float(price)
    except (TypeError, ValueError):
        return fallback
    return price if price > 0 else fallback


def lookup_local_close(data: Any, instrument: str, fallback: float) -> float:
    """从本地日线数据中查找最新收盘价。"""
    if data is None:
        return fallback
    normalized = normalize_symbol(instrument)
    candidates = [instrument, normalized]
    code6 = normalized[2:] if len(normalized) > 2 else normalized
    if code6 not in candidates:
        candidates.append(code6)
    for candidate in candidates:
        try:
            bars = data.get_kline(candidate, freq="day")
        except Exception:
            continue
        if not bars:
            continue
        close_price = bars[-1].get("close")
        try:
            close_price = float(close_price)
        except (TypeError, ValueError):
            continue
        if close_price > 0:
            return close_price
    return fallback


def _last_non_nan(arr, default=None):
    for j in range(len(arr) - 1, -1, -1):
        v = float(arr[j])
        if not np.isnan(v):
            return v
    return default


def _prev_non_nan_nonzero(arr, before_idx):
    for j in range(before_idx - 1, -1, -1):
        v = float(arr[j])
        if not np.isnan(v) and v != 0:
            return v
    return None


def normalize_day_str(value) -> str:
    return str(value or "")[:10]


def truthy_param(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def fmt_rate_wan(rate: float) -> str:
    """Format rate to Chinese 万 label, e.g. 0.0001 -> '万1'"""
    v = rate * 10000
    return f"万{v:.1f}" if v != int(v) else f"万{int(v)}"


def fmt_rate_shiwan(rate: float) -> str:
    """Format rate to Chinese 十万 label, e.g. 0.00001 -> '十万1'"""
    v = rate * 100000
    return f"十万{v:.1f}" if v != int(v) else f"十万{int(v)}"


def build_fee_labels(fee_settings: dict) -> dict:
    """Generate dynamic Chinese fee labels from raw fee settings."""
    stock_rate = float(fee_settings.get("stock_commission_rate", DEFAULT_FEE_SETTINGS["stock_commission_rate"]))
    etf_rate = float(fee_settings.get("etf_commission_rate", DEFAULT_FEE_SETTINGS["etf_commission_rate"]))
    stamp_rate = float(fee_settings.get("stamp_duty_rate", DEFAULT_FEE_SETTINGS["stamp_duty_rate"]))
    sh_rate = float(fee_settings.get("sh_transfer_fee_rate", DEFAULT_FEE_SETTINGS["sh_transfer_fee_rate"]))

    stock_label = f"股票{fmt_rate_wan(stock_rate)}"
    etf_label = f"ETF{fmt_rate_wan(etf_rate)}"
    stamp_label = f"印花税买入不收，卖出{fmt_rate_wan(stamp_rate)}"
    sh_label = f"沪市过户费买卖{fmt_rate_shiwan(sh_rate)}，深市包含在佣金里"

    summary_parts = [f"{stock_label}，{etf_label}", stamp_label, sh_label]

    return {
        "stockCommissionLabel": stock_label,
        "etfCommissionLabel": etf_label,
        "stampDutyLabel": stamp_label,
        "shTransferFeeLabel": sh_label,
        "feeRuleSummary": "；".join(summary_parts),
    }
