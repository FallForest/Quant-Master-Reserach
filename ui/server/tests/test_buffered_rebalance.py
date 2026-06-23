from server import app


class _FakeModelService:
    def list_models(self):
        return [{"alias": "test_model"}]

    def get_predictions(self, alias, date=None, top_k=5):
        return {
            "date": "2025-02-20",
            "topK": top_k,
            "stocks": [
                {"rank": 1, "instrument": "SH600001", "name": "Test Stock A", "score": 0.9123},
                {"rank": 2, "instrument": "SH600002", "name": "Test Stock B", "score": 0.8751},
                {"rank": 3, "instrument": "SH600003", "name": "Test Stock C", "score": 0.8012},
            ][:top_k],
            "totalStocks": 3,
            "scoreStats": {"mean": 0.8629, "std": 0.056, "min": 0.8012, "max": 0.9123},
            "source": "test",
        }


def test_buffered_rebalance_preview(client, monkeypatch):
    fake_model = _FakeModelService()
    monkeypatch.setattr(app, "model_service", fake_model)

    r = client.get(
        "/api/strategy-buffered-rebalance?top_k=2&hold_topk=3&risk_degree=0.9&weight_mode=equal",
    )
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data
    assert "trades" in data
    assert "targetPositions" in data
    assert data["config"]["topK"] == 2
    assert data["config"]["holdTopk"] == 3
