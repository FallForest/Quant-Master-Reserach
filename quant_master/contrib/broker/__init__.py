# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

from .base import BaseBroker, BrokerOrder, BrokerOrderDir, OrderStatus, Position, AccountInfo
from .easytrader_broker import EasytraderBroker
from .execution import ExecutionResult, LiveOrderExecutor, LiveOrderRequest
from .factory import create_broker
from .paper_broker import PaperBroker
from .tcdll_broker import TcDllBroker
from .tdx.broker import TDXBroker
from .xiadan_broker import XiadanBroker


def order_to_broker_order(order, price: float) -> BrokerOrder:
    """Convert a backtest Order to a live BrokerOrder.

    Parameters
    ----------
    order : quant_master.backtest.decision.Order
        The backtest order
    price : float
        The limit price for the live order

    Returns
    -------
    BrokerOrder
    """
    from quant_master.backtest.decision import OrderDir

    return BrokerOrder(
        stock_id=order.stock_id,
        price=price,
        amount=int(order.amount),
        direction=BrokerOrderDir.BUY if order.direction == OrderDir.BUY else BrokerOrderDir.SELL,
    )
