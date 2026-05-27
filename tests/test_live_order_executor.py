from typing import List

from quant_master.contrib.broker import (
    AccountInfo,
    BrokerOrder,
    BrokerOrderDir,
    LiveOrderExecutor,
    LiveOrderRequest,
    OrderStatus,
    Position,
)
from quant_master.contrib.broker.base import BaseBroker
from quant_master.contrib.broker.paper_broker import PaperBroker


class MockBroker(BaseBroker):
    def __init__(self):
        self.orders: List[BrokerOrder] = []
        self.positions = [
            Position(stock_id="600519", volume=300, available_volume=200, cost_price=1500.0),
        ]
        self.account = AccountInfo(
            total_assets=200000.0,
            available_cash=50000.0,
            market_value=150000.0,
            frozen_amount=0.0,
        )

    def login(self, account: str, password: str, **kwargs) -> bool:
        return True

    def logout(self) -> None:
        return None

    def buy(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        order = BrokerOrder(
            stock_id=stock_id,
            price=price,
            amount=amount,
            direction=BrokerOrderDir.BUY,
            order_id="B001",
            status=OrderStatus.PENDING,
        )
        self.orders.append(order)
        return order

    def sell(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        order = BrokerOrder(
            stock_id=stock_id,
            price=price,
            amount=amount,
            direction=BrokerOrderDir.SELL,
            order_id="S001",
            status=OrderStatus.PENDING,
        )
        self.orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def query_orders(self) -> List[BrokerOrder]:
        return list(self.orders)

    def query_deals(self) -> List[dict]:
        return []

    def query_positions(self) -> List[Position]:
        return list(self.positions)

    def query_account(self) -> AccountInfo:
        return self.account

    def is_connected(self) -> bool:
        return True


class FailingBroker(MockBroker):
    def __init__(self):
        super().__init__()
        self.submit_attempts = 0

    def buy(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        self.submit_attempts += 1
        raise RuntimeError("broker offline")


def test_live_order_executor_accepts_valid_buy():
    broker = MockBroker()
    executor = LiveOrderExecutor(broker, max_order_value=30000.0)

    result = executor.submit(
        LiveOrderRequest(
            stock_id="000001",
            price=12.3,
            amount=100,
            direction=BrokerOrderDir.BUY,
        ),
        dry_run=False,
    )

    assert result.accepted is True
    assert result.rejection_reason == ""
    assert result.broker_order is not None
    assert result.post_check_status == "order_id_received"


def test_live_order_executor_rejects_cash_overuse():
    broker = MockBroker()
    executor = LiveOrderExecutor(broker, max_position_ratio=0.5)

    result = executor.submit(
        LiveOrderRequest(
            stock_id="000001",
            price=400.0,
            amount=100,
            direction=BrokerOrderDir.BUY,
        )
    )

    assert result.accepted is False
    assert result.rejection_reason == "insufficient_available_cash"


def test_live_order_executor_rejects_invalid_lot_size():
    broker = MockBroker()
    executor = LiveOrderExecutor(broker)

    result = executor.submit(
        LiveOrderRequest(
            stock_id="000001",
            price=12.3,
            amount=150,
            direction=BrokerOrderDir.BUY,
        )
    )

    assert result.accepted is False
    assert result.rejection_reason == "amount_must_be_multiple_of_100"


def test_live_order_executor_checks_sell_position():
    broker = MockBroker()
    executor = LiveOrderExecutor(broker)

    result = executor.submit(
        LiveOrderRequest(
            stock_id="600519",
            price=1680.0,
            amount=300,
            direction=BrokerOrderDir.SELL,
        )
    )

    assert result.accepted is False
    assert result.rejection_reason == "insufficient_available_position"


def test_live_order_executor_can_skip_account_state_checks():
    broker = MockBroker()
    broker.account.available_cash = 0.0
    executor = LiveOrderExecutor(broker, validate_account_state=False)

    result = executor.submit(
        LiveOrderRequest(
            stock_id="000001",
            price=12.3,
            amount=100,
            direction=BrokerOrderDir.BUY,
        )
    )

    assert result.accepted is True
    assert result.broker_order is not None


def test_live_order_executor_accepts_paper_broker_order():
    broker = PaperBroker(available_cash=100000.0)
    executor = LiveOrderExecutor(broker)

    result = executor.submit(
        LiveOrderRequest(
            stock_id="000001",
            price=12.3,
            amount=100,
            direction=BrokerOrderDir.BUY,
        )
    )

    assert result.accepted is True
    assert result.broker_order is not None
    assert result.broker_order.order_id.startswith("PAPER-")
    assert result.post_check_status == "order_id_received"


def test_live_order_executor_returns_structured_submit_failure():
    broker = FailingBroker()
    executor = LiveOrderExecutor(broker)

    result = executor.submit(
        LiveOrderRequest(
            stock_id="000001",
            price=12.3,
            amount=100,
            direction=BrokerOrderDir.BUY,
        )
    )

    assert result.accepted is False
    assert result.broker_order is None
    assert result.rejection_reason == "RuntimeError: broker offline"
    assert result.post_check_status == "submit_failed"
    assert broker.submit_attempts == 1


def test_live_order_executor_stops_batch_after_submit_failure():
    broker = FailingBroker()
    executor = LiveOrderExecutor(broker)

    results = executor.submit_many(
        [
            LiveOrderRequest(
                stock_id="000001",
                price=12.3,
                amount=100,
                direction=BrokerOrderDir.BUY,
            ),
            LiveOrderRequest(
                stock_id="000002",
                price=11.0,
                amount=100,
                direction=BrokerOrderDir.BUY,
            ),
        ]
    )

    assert [item.post_check_status for item in results] == ["submit_failed", "skipped"]
    assert results[1].rejection_reason == "skipped_after_submit_failure"
    assert broker.submit_attempts == 1
