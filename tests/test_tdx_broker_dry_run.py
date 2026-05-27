from quant_master.contrib.broker import BrokerOrderDir
from quant_master.contrib.broker.tdx.broker import TDXBroker


def test_tdx_broker_buy_dry_run_does_not_call_client():
    broker = TDXBroker(host="127.0.0.1")

    def fail(*args, **kwargs):
        raise AssertionError("dry_run must not call TQLEX")

    broker.client.call_tqlex = fail
    order = broker.buy("000001", 12.3, 100, dry_run=True)

    assert order.order_id == "DRY-RUN"
    assert order.direction == BrokerOrderDir.BUY


def test_tdx_broker_sell_dry_run_does_not_call_client():
    broker = TDXBroker(host="127.0.0.1")

    def fail(*args, **kwargs):
        raise AssertionError("dry_run must not call TQLEX")

    broker.client.call_tqlex = fail
    order = broker.sell("600519", 1680.0, 100, dry_run=True)

    assert order.order_id == "DRY-RUN"
    assert order.direction == BrokerOrderDir.SELL
