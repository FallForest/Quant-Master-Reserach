def test_execution_preview(client):
    payload = {
        "trades": [
            {"instrument": "SH600001", "name": "Test A", "side": "buy", "deltaShares": 800, "currentPrice": 10.0},
            {"instrument": "SZ000001", "name": "Test B", "side": "sell", "deltaShares": -500, "currentPrice": 8.0},
            {"instrument": "SH600003", "name": "Hold", "side": "hold", "deltaShares": 0, "currentPrice": 9.0},
        ]
    }
    r = client.post("/api/execution/preview", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["summary"]["totalOrders"] == 2
    assert data["summary"]["validOrders"] == 2
    assert data["summary"]["invalidOrders"] == 0
    assert data["orders"][0]["stockId"] == "SH600001"
    assert data["orders"][0]["valid"] is True
    assert data["orders"][1]["stockId"] == "SZ000001"
    assert data["orders"][1]["valid"] is True


def test_execution_preview_rejects_order_value_limit(client):
    payload = {
        "trades": [{"instrument": "SH600001", "side": "buy", "deltaShares": 800, "currentPrice": 10.0}],
        "risk": {"maxOrderValue": 5000},
    }
    r = client.post("/api/execution/preview", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["orders"][0]["valid"] is False
    assert data["orders"][0]["validationError"] == "order_value_exceeds_limit"
