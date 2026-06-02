import requests


def test_buffered_rebalance_preview(server_url):
    r = requests.get(
        f"{server_url}/api/strategy-buffered-rebalance?top_k=2&hold_topk=3&risk_degree=0.9&weight_mode=equal",
        timeout=5,
    )
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data
    assert "trades" in data
    assert "targetPositions" in data
    assert data["config"]["topK"] == 2
    assert data["config"]["holdTopk"] == 3
