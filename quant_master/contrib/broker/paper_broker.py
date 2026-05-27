# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""In-memory broker for validating live-order flows without real trading."""

from __future__ import annotations

import itertools
from typing import List, Optional

from .base import (
    AccountInfo,
    BaseBroker,
    BrokerOrder,
    BrokerOrderDir,
    OrderStatus,
    Position,
)


class PaperBroker(BaseBroker):
    """A deterministic broker implementation for dry-run and CI tests."""

    def __init__(
        self,
        available_cash: float = 100000.0,
        positions: Optional[List[Position]] = None,
    ):
        self.account = AccountInfo(
            total_assets=float(available_cash),
            available_cash=float(available_cash),
            market_value=0.0,
            frozen_amount=0.0,
        )
        self.positions = list(positions or [])
        self.orders: List[BrokerOrder] = []
        self._ids = itertools.count(1)
        self._connected = True

    def login(self, account: str, password: str, **kwargs) -> bool:
        return True

    def logout(self) -> None:
        self._connected = False

    def buy(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        return self._submit(stock_id, price, amount, BrokerOrderDir.BUY, dry_run=kwargs.get("dry_run", False))

    def sell(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        return self._submit(stock_id, price, amount, BrokerOrderDir.SELL, dry_run=kwargs.get("dry_run", False))

    def cancel_order(self, order_id: str) -> bool:
        for order in self.orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def query_orders(self) -> List[BrokerOrder]:
        return list(self.orders)

    def query_deals(self) -> List[dict]:
        return []

    def query_positions(self) -> List[Position]:
        return list(self.positions)

    def query_account(self) -> AccountInfo:
        return self.account

    def is_connected(self) -> bool:
        return self._connected

    def _submit(self, stock_id: str, price: float, amount: int, direction: BrokerOrderDir, *, dry_run: bool) -> BrokerOrder:
        order = BrokerOrder(
            stock_id=stock_id,
            price=price,
            amount=amount,
            direction=direction,
            order_id=f"PAPER-{next(self._ids):06d}",
            status=OrderStatus.PENDING,
        )
        if not dry_run:
            self.orders.append(order)
        return order
