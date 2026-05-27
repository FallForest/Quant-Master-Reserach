import json
import sys

import scripts.run_live_orders as cli
from quant_master.contrib.broker import BrokerOrder, BrokerOrderDir, OrderStatus


class CliBroker:
    def __init__(self, *, fail_submit=False):
        self.fail_submit = fail_submit
        self.dry_run_values = []
        self.query_account_calls = 0
        self.submit_attempts = 0

    def connect(self):
        return None

    def buy(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        self.submit_attempts += 1
        self.dry_run_values.append(kwargs.get("dry_run"))
        if self.fail_submit:
            raise RuntimeError("submit exploded")
        return BrokerOrder(
            stock_id=stock_id,
            price=price,
            amount=amount,
            direction=BrokerOrderDir.BUY,
            order_id="CLI-001",
            status=OrderStatus.PENDING,
        )

    def sell(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        raise AssertionError("sell should not be called in this test")

    def query_account(self):
        self.query_account_calls += 1
        raise AssertionError("query_account should be skipped")

    def query_positions(self):
        return []

    def query_orders(self):
        return []


def test_run_live_orders_dry_run_and_skip_account_check(monkeypatch, capsys):
    broker = CliBroker()
    monkeypatch.setattr(cli, "create_broker", lambda *args, **kwargs: broker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_live_orders.py",
            "--broker",
            "paper",
            "--skip-account-check",
            "--dry-run",
            "--orders",
            "000001,buy,12.3,100",
        ],
    )

    exit_code = cli.main()
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload[0]["accepted"] is True
    assert payload[0]["order_id"] == "CLI-001"
    assert broker.dry_run_values == [True]
    assert broker.query_account_calls == 0


def test_run_live_orders_outputs_json_and_skips_after_submit_exception(monkeypatch, capsys):
    broker = CliBroker(fail_submit=True)
    monkeypatch.setattr(cli, "create_broker", lambda *args, **kwargs: broker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_live_orders.py",
            "--broker",
            "paper",
            "--skip-account-check",
            "--orders",
            "000001,buy,12.3,100",
            "000002,buy,11.0,100",
        ],
    )

    exit_code = cli.main()
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload[0]["accepted"] is False
    assert payload[0]["rejection_reason"] == "RuntimeError: submit exploded"
    assert payload[0]["post_check_status"] == "submit_failed"
    assert payload[0]["order_id"] is None
    assert payload[1]["accepted"] is False
    assert payload[1]["rejection_reason"] == "skipped_after_submit_failure"
    assert payload[1]["post_check_status"] == "skipped"
    assert broker.submit_attempts == 1


def test_run_live_orders_passes_tcdll_runtime_directories(monkeypatch):
    broker = CliBroker()
    captured = {}

    def fake_create_broker(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return broker

    monkeypatch.setattr(cli, "create_broker", fake_create_broker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_live_orders.py",
            "--broker",
            "tcdll",
            "--caller-path",
            r"C:\tools\tdx_caller.exe",
            "--xiadan-work-dir",
            r"C:\silkriver\xiadan",
            "--dll-dir",
            r"C:\silkriver\dlls",
            "--skip-account-check",
            "--dry-run",
            "--orders",
            "000001,buy,12.3,100",
        ],
    )

    assert cli.main() == 0
    assert captured["args"] == ("tcdll",)
    assert captured["kwargs"] == {
        "caller_path": r"C:\tools\tdx_caller.exe",
        "xiadan_work_dir": r"C:\silkriver\xiadan",
        "dll_dir": r"C:\silkriver\dlls",
        "enable_live": False,
    }
