# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""XiadanBroker — Live trading via xiadan.exe window automation.

Uses Win32 API to directly control xiadan's trading interface.
No easytrader, no HTTP protocol, no CAPTCHA service.

Requires:
    - xiadan.exe running and logged in
    - Trading window visible (网上股票交易系统5.0)
    - Buy/Sell form tab active

Usage::

    from quant_master.contrib.broker import XiadanBroker
    broker = XiadanBroker()
    broker.connect()
    account = broker.query_account()
    order = broker.buy("600519", price=1273.38, amount=100)
"""

import time
from typing import List, Optional

from quant_master.log import get_module_logger

from .base import (
    AccountInfo,
    BaseBroker,
    BrokerOrder,
    BrokerOrderDir,
    OrderStatus,
    Position,
)

logger = get_module_logger("XiadanBroker")


class XiadanBroker(BaseBroker):
    """Live trading broker controlling xiadan.exe via Win32 API.

    This broker works by directly reading/writing the xiadan trading
    window controls using the Windows API (ctypes). It does NOT use
    easytrader, HTTP protocol, or any external service.
    """

    def __init__(self):
        self._tdx = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to running xiadan window."""
        try:
            from scripts.tdx_direct import TDXDirect
            self._tdx = TDXDirect()
            self._tdx.connect()
            self._connected = True
            logger.info("Connected to xiadan window")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to xiadan: {e}")
            self._connected = False
            return False

    def login(self, account: str, password: str, **kwargs) -> bool:
        """Not needed — xiadan must be pre-logged in."""
        logger.info("XiadanBroker assumes xiadan is already logged in")
        return self._connected

    def logout(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._tdx is not None

    def query_account(self) -> AccountInfo:
        """Query account info from xiadan window."""
        tdx = self._tdx
        return AccountInfo(
            available_cash=tdx.get_balance(),
            market_value=tdx.get_market_value(),
            total_assets=tdx.get_total_assets(),
            frozen_amount=0.0,
        )

    def buy(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        """Place a buy order via xiadan."""
        tdx = self._tdx
        logger.info(f"Buy {stock_id} price={price} amount={amount}")

        tdx.switch_to_buy_tab()
        tdx.set_buy(stock_id, price, amount)

        if kwargs.get("dry_run", False):
            logger.info("Dry run — not clicking Buy button")
            return BrokerOrder(
                stock_id=stock_id, price=price, amount=amount,
                direction=BrokerOrderDir.BUY, status=OrderStatus.PENDING,
            )

        result = tdx.click_buy()
        if result.get("confirmed"):
            logger.info(f"Buy order confirmed: {result.get('info', {})}")
        else:
            logger.warning("Buy order confirmation not found")

        return BrokerOrder(
            stock_id=stock_id, price=price, amount=amount,
            direction=BrokerOrderDir.BUY, status=OrderStatus.PENDING,
        )

    def sell(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        """Place a sell order via xiadan."""
        tdx = self._tdx
        logger.info(f"Sell {stock_id} price={price} amount={amount}")

        # xiadan buy/sell panes share the same input control IDs after switching
        # into the visible trading dialog, so we reuse the same setters here.
        tdx.switch_to_buy_tab()
        tdx.set_stock_code(stock_id)
        tdx.set_buy_price(price)
        tdx.set_buy_qty(amount)

        if kwargs.get("dry_run", False):
            return BrokerOrder(
                stock_id=stock_id, price=price, amount=amount,
                direction=BrokerOrderDir.SELL, status=OrderStatus.PENDING,
            )

        result = tdx.click_sell()
        if result.get("confirmed"):
            logger.info(f"Sell order confirmed: {result.get('info', {})}")
        else:
            logger.warning("Sell order confirmation not found")

        return BrokerOrder(
            stock_id=stock_id, price=price, amount=amount,
            direction=BrokerOrderDir.SELL, status=OrderStatus.PENDING,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel all pending orders (xiadan doesn't support single-order cancel by ID)."""
        logger.info(f"Cancelling orders (xiadan: cancel all)")
        self._tdx.cancel_all()
        return True

    def query_orders(self) -> List[BrokerOrder]:
        """Not directly supported via xiadan window automation."""
        logger.warning("query_orders not supported via xiadan window automation")
        return []

    def query_deals(self) -> List[dict]:
        """Not directly supported via xiadan window automation."""
        logger.warning("query_deals not supported via xiadan window automation")
        return []

    def query_positions(self) -> List[Position]:
        """Not directly supported via xiadan window automation."""
        logger.warning("query_positions not supported via xiadan window automation")
        return []
