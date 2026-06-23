"""持仓路由：读取 live_data 持仓文件，结合实时行情返回持仓详情。"""
from __future__ import annotations

import logging
from datetime import date as _date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..datadir import DataDir
from ..dependencies import get_data, get_tdx_quote
from ..helpers import DEFAULT_FEE_SETTINGS, calc_trade_fee, fee_settings_from_raw
from ..position_service import enrich_positions, load_orders, load_positions_file, save_positions_file
from ..schemas import PositionAddRequest, PositionSellRequest, PositionUpdateRequest, SetAccountRequest, SetCashRequest
from ..tdx_quote import TDXQuote

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/positions", tags=["positions"])


@router.get("")
def current(data: DataDir = Depends(get_data), tdx: TDXQuote = Depends(get_tdx_quote)):
    try:
        raw = load_positions_file()
        return enrich_positions(raw, data=data, tdx=tdx)
    except Exception as exc:
        logger.exception("positions error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
def history(limit: int = 30):
    try:
        return {"orders": load_orders(limit=limit)}
    except Exception as exc:
        logger.exception("positions history error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("")
def add_or_update(body: PositionAddRequest, data: DataDir = Depends(get_data), tdx: TDXQuote = Depends(get_tdx_quote)):
    instrument = body.instrument.strip().upper()
    if not instrument:
        raise HTTPException(status_code=400, detail="instrument 不能为空")

    shares = int(body.shares)
    price = float(body.price)

    raw = load_positions_file()
    fee_settings = fee_settings_from_raw(raw)
    trade_amount = shares * price
    fee_breakdown = calc_trade_fee(instrument, trade_amount, "buy", fee_settings)
    positions = raw.get("positions", {})

    if instrument in positions:
        old = positions[instrument]
        old_shares = old.get("shares", 0)
        old_price = old.get("price", 0)
        new_shares = old_shares + shares
        new_price = (old_shares * old_price + shares * price + fee_breakdown["total"]) / new_shares if new_shares > 0 else price
        positions[instrument] = {"shares": new_shares, "price": round(new_price, 4)}
    else:
        positions[instrument] = {"shares": shares, "price": round((trade_amount + fee_breakdown["total"]) / shares, 4)}

    raw["positions"] = positions
    raw["date"] = str(_date.today())
    raw["cash"] = round(raw.get("cash", 0) - trade_amount - fee_breakdown["total"], 2)
    save_positions_file(raw)

    result = enrich_positions(raw, data=data, tdx=tdx)
    result["lastFeeBreakdown"] = fee_breakdown
    return result


@router.post("/sell")
def sell_position(body: PositionSellRequest, data: DataDir = Depends(get_data), tdx: TDXQuote = Depends(get_tdx_quote)):
    instrument = body.instrument.strip().upper()
    if not instrument:
        raise HTTPException(status_code=400, detail="instrument 不能为空")

    raw = load_positions_file()
    fee_settings = fee_settings_from_raw(raw)
    positions = raw.get("positions", {})

    if instrument not in positions:
        raise HTTPException(status_code=404, detail=f"持仓 {instrument} 不存在")

    pos = positions[instrument]
    current_shares = pos.get("shares", 0)

    sell_shares = int(body.shares) if body.shares is not None else current_shares
    if sell_shares <= 0:
        raise HTTPException(status_code=400, detail="卖出数量必须大于 0")
    if sell_shares > current_shares:
        raise HTTPException(status_code=400, detail=f"持仓不足：当前 {current_shares}，需要卖出 {sell_shares}")

    sell_price = float(body.price) if body.price is not None else pos.get("price", 0)
    if sell_price <= 0:
        raise HTTPException(status_code=400, detail="卖出价格必须大于 0")

    trade_amount = sell_shares * sell_price
    fee_breakdown = calc_trade_fee(instrument, trade_amount, "sell", fee_settings)

    remaining = current_shares - sell_shares
    if remaining <= 0:
        del positions[instrument]
    else:
        positions[instrument] = {"shares": remaining, "price": pos.get("price", 0)}

    raw["positions"] = positions
    raw["date"] = str(_date.today())
    raw["cash"] = round(raw.get("cash", 0) + trade_amount - fee_breakdown["total"], 2)
    save_positions_file(raw)

    result = enrich_positions(raw, data=data, tdx=tdx)
    result["lastFeeBreakdown"] = fee_breakdown
    return result


@router.post("/cash")
def set_cash(body: SetCashRequest, data: DataDir = Depends(get_data), tdx: TDXQuote = Depends(get_tdx_quote)):
    if body.cash < 0:
        raise HTTPException(status_code=400, detail="cash 必须 >= 0")
    raw = load_positions_file()
    raw["cash"] = float(body.cash)
    raw["date"] = str(_date.today())
    save_positions_file(raw)
    return enrich_positions(raw, data=data, tdx=tdx)


@router.post("/account")
def set_account(body: SetAccountRequest, data: DataDir = Depends(get_data), tdx: TDXQuote = Depends(get_tdx_quote)):
    rate_fields = {
        "stockCommissionRate": "stock_commission_rate",
        "etfCommissionRate": "etf_commission_rate",
        "stampDutyRate": "stamp_duty_rate",
        "shTransferFeeRate": "sh_transfer_fee_rate",
    }

    raw = load_positions_file()
    for api_key, raw_key in rate_fields.items():
        value = getattr(body, api_key, None)
        if value is None:
            value = raw.get(raw_key, DEFAULT_FEE_SETTINGS[raw_key])
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{api_key} 必须为数字")
        if value < 0:
            raise HTTPException(status_code=400, detail=f"{api_key} 必须 >= 0")
        raw[raw_key] = value

    raw["capital_amount"] = body.capitalAmount
    raw["trading_fee_rate"] = raw["stock_commission_rate"]
    raw["date"] = str(_date.today())
    save_positions_file(raw)
    return enrich_positions(raw, data=data, tdx=tdx)


@router.post("/{instrument}")
def update(instrument: str, body: PositionUpdateRequest, data: DataDir = Depends(get_data), tdx: TDXQuote = Depends(get_tdx_quote)):
    instrument = instrument.strip().upper()
    if not instrument:
        raise HTTPException(status_code=400, detail="instrument 不能为空")

    raw = load_positions_file()
    fee_settings = fee_settings_from_raw(raw)
    positions = raw.get("positions", {})

    if instrument not in positions:
        raise HTTPException(status_code=404, detail=f"持仓 {instrument} 不存在")

    pos = positions[instrument]
    old_shares = pos.get("shares", 0)
    old_price = pos.get("price", 0)

    if body.shares is not None:
        new_shares = int(body.shares)
        if new_shares <= 0:
            raise HTTPException(status_code=400, detail="shares 必须大于 0")
        pos["shares"] = new_shares

    if body.price is not None:
        new_price = float(body.price)
        if new_price <= 0:
            raise HTTPException(status_code=400, detail="price 必须大于 0")
        pos["price"] = round(new_price, 4)

    old_cost = old_shares * old_price
    new_cost = pos["shares"] * pos["price"]
    old_fee = calc_trade_fee(instrument, old_cost, "buy", fee_settings)["total"] if old_cost > 0 else 0.0
    new_fee = calc_trade_fee(instrument, new_cost, "buy", fee_settings)["total"] if new_cost > 0 else 0.0
    raw["cash"] = round(raw.get("cash", 0) + old_cost + old_fee - new_cost - new_fee, 2)

    raw["date"] = str(_date.today())
    save_positions_file(raw)

    result = enrich_positions(raw, data=data, tdx=tdx)
    result["lastFeeBreakdown"] = calc_trade_fee(instrument, new_cost, "buy", fee_settings) if new_cost > 0 else {
        "commission": 0.0, "stampDuty": 0.0, "transferFee": 0.0, "total": 0.0,
    }
    return result


@router.delete("/{instrument}")
def remove(instrument: str, data: DataDir = Depends(get_data), tdx: TDXQuote = Depends(get_tdx_quote)):
    instrument = instrument.strip().upper()
    if not instrument:
        raise HTTPException(status_code=400, detail="instrument 不能为空")

    raw = load_positions_file()
    positions = raw.get("positions", {})

    if instrument not in positions:
        raise HTTPException(status_code=404, detail=f"持仓 {instrument} 不存在")

    pos = positions[instrument]
    fee_settings = fee_settings_from_raw(raw)
    sell_amount = pos.get("shares", 0) * pos.get("price", 0)
    fee_breakdown = calc_trade_fee(instrument, sell_amount, "sell", fee_settings)
    raw["cash"] = round(raw.get("cash", 0) + sell_amount - fee_breakdown["total"], 2)
    del positions[instrument]

    raw["date"] = str(_date.today())
    save_positions_file(raw)

    result = enrich_positions(raw, data=data, tdx=tdx)
    result["lastFeeBreakdown"] = fee_breakdown
    return result
