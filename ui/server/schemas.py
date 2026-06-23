"""Pydantic 请求模型。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class WatchlistAddRequest(BaseModel):
    symbol: str


class PositionAddRequest(BaseModel):
    instrument: str
    shares: float
    price: float


class PositionUpdateRequest(BaseModel):
    shares: Optional[float] = None
    price: Optional[float] = None


class PositionSellRequest(BaseModel):
    instrument: str
    shares: Optional[float] = None  # None = sell all
    price: Optional[float] = None  # None = use cost price


class SetCashRequest(BaseModel):
    cash: float


class SetAccountRequest(BaseModel):
    capitalAmount: float
    stockCommissionRate: Optional[float] = None
    etfCommissionRate: Optional[float] = None
    stampDutyRate: Optional[float] = None
    shTransferFeeRate: Optional[float] = None


class ExecutionPreviewRequest(BaseModel):
    trades: list[Any] = []
    risk: Optional[dict[str, Any]] = None


class ExecutionSubmitRequest(BaseModel):
    orders: list[Any] = []
    brokerKind: Optional[str] = None
    dryRun: Optional[bool] = None
    risk: Optional[dict[str, Any]] = None
    confirm: bool = False


class ModelRunRequest(BaseModel):
    date: str
    top_k: int = 30
