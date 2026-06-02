import requests


def test_execution_submit_requires_confirm(server_url):
    payload = {
        "orders": [{"stockId": "SH600001", "side": "buy", "price": 10.0, "amount": 800}],
        "brokerKind": "paper",
        "dryRun": True,
        "confirm": False,
    }
    r = requests.post(f"{server_url}/api/execution/submit", json=payload, timeout=5)
    assert r.status_code == 400
    assert r.json()["error"] == "confirm_required"


def test_execution_submit_blocks_live_when_disabled(server_url):
    payload = {
        "orders": [{"stockId": "SH600001", "side": "buy", "price": 10.0, "amount": 800}],
        "brokerKind": "paper",
        "dryRun": False,
        "confirm": True,
    }
    r = requests.post(f"{server_url}/api/execution/submit", json=payload, timeout=5)
    assert r.status_code == 403
    assert r.json()["error"] == "live_trading_disabled"


def test_execution_submit_paper_dry_run(server_url):
    cash_seed = requests.post(
        f"{server_url}/api/positions/cash",
        json={"cash": 50000},
        timeout=5,
    )
    assert cash_seed.status_code == 200

    seed = requests.post(
        f"{server_url}/api/positions",
        json={"instrument": "SH600003", "shares": 1000, "price": 8.0},
        timeout=5,
    )
    assert seed.status_code == 200

    payload = {
        "orders": [
            {"stockId": "SH600003", "side": "sell", "price": 8.0, "amount": 900},
            {"stockId": "SH600002", "side": "buy", "price": 5.0, "amount": 800},
        ],
        "brokerKind": "paper",
        "dryRun": True,
        "confirm": True,
    }
    r = requests.post(f"{server_url}/api/execution/submit", json=payload, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["brokerKind"] == "paper"
    assert data["dryRun"] is True
    assert data["summary"]["accepted"] == 2
    assert data["summary"]["rejected"] == 0
    assert len(data["results"]) == 2
    assert data["results"][0]["side"] == "sell"
    assert data["results"][1]["side"] == "buy"


def test_execution_submit_invalid_orders_return_rejections(server_url):
    payload = {
        "orders": [{"stockId": "SH600001", "side": "buy", "price": 10.0, "amount": 850}],
        "brokerKind": "paper",
        "dryRun": True,
        "confirm": True,
    }
    r = requests.post(f"{server_url}/api/execution/submit", json=payload, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["summary"]["accepted"] == 0
    assert data["summary"]["rejected"] == 1
    assert data["results"][0]["rejectionReason"] == "amount_must_be_multiple_of_100"
