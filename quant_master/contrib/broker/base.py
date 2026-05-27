# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd


class BrokerOrderDir(Enum):
    BUY = 1
    SELL = 0


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class BrokerOrder:
    """Represents a live order submitted to a broker."""

    stock_id: str
    price: float
    amount: int  # must be multiple of 100 for A-shares
    direction: BrokerOrderDir
    order_id: Optional[str] = None
    deal_amount: int = 0
    status: OrderStatus = OrderStatus.PENDING
    order_time: Optional[pd.Timestamp] = None


@dataclass
class Position:
    stock_id: str
    volume: int
    available_volume: int
    cost_price: float
    current_price: float = 0.0
    market_value: float = 0.0


@dataclass
class AccountInfo:
    total_assets: float
    available_cash: float
    market_value: float
    frozen_amount: float = 0.0


class BaseBroker(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def login(self, account: str, password: str, **kwargs) -> bool:
        ...

    @abstractmethod
    def logout(self) -> None:
        ...

    @abstractmethod
    def buy(self, stock_id: str, price: float, amount: int) -> BrokerOrder:
        ...

    @abstractmethod
    def sell(self, stock_id: str, price: float, amount: int) -> BrokerOrder:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def query_orders(self) -> List[BrokerOrder]:
        ...

    @abstractmethod
    def query_deals(self) -> List[dict]:
        ...

    @abstractmethod
    def query_positions(self) -> List[Position]:
        ...

    @abstractmethod
    def query_account(self) -> AccountInfo:
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...
