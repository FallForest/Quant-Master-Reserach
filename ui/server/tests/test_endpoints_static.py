"""Tests for endpoints that don't need real data (static / demo responses)."""
import pytest
import requests


def test_experiments(server_url):
    r = requests.get(f"{server_url}/api/experiments", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "experiments" in data
    assert isinstance(data["experiments"], list)
    for exp in data["experiments"]:
        assert "id" in exp
        assert "name" in exp
        assert "status" in exp


def test_portfolio(server_url):
    r = requests.get(f"{server_url}/api/portfolio", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "holdings" in data
    assert isinstance(data["holdings"], list)
    assert "timeline" in data
    assert isinstance(data["timeline"], list)
    assert "summary" in data
    assert isinstance(data["summary"], dict)
    assert "allocation" in data
    assert isinstance(data["allocation"], list)


def test_model_performance(server_url):
    r = requests.get(f"{server_url}/api/model-performance", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "groupReturns" in data
    assert isinstance(data["groupReturns"], dict)
    assert "summary" in data
    assert "icir" in data["summary"]


def test_model_performance_filter(server_url):
    r = requests.get(f"{server_url}/api/model-performance?model=lightgbm", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "groupReturns" in data


def test_strategies(server_url):
    r = requests.get(f"{server_url}/api/strategies", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "strategies" in data
    assert len(data["strategies"]) == 6
    assert data["strategies"][0]["id"] == "topk_dropout"


def test_optimizer(server_url):
    r = requests.get(f"{server_url}/api/optimizer", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "methods" in data
    assert len(data["methods"]) == 5
    assert "comparison" in data
    assert isinstance(data["comparison"], list)
    assert "sectors" in data
    assert isinstance(data["sectors"], list)


def test_model_catalog(server_url):
    r = requests.get(f"{server_url}/api/model-catalog", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert len(data["models"]) == 40
    assert "categories" in data
    assert len(data["categories"]) == 10


def test_attribution(server_url):
    r = requests.get(f"{server_url}/api/attribution", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "monthly" in data
    assert isinstance(data["monthly"], list)
    assert "bySector" in data
    assert isinstance(data["bySector"], list)
    assert "summary" in data
    assert isinstance(data["summary"], dict)


def test_factor_analysis_default(server_url):
    r = requests.get(f"{server_url}/api/factor/analysis", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "metrics" in data
    assert isinstance(data["metrics"], dict)
    assert "icSeries" in data
    assert isinstance(data["icSeries"], list)
    assert "groupReturns" in data
    assert isinstance(data["groupReturns"], dict)


def test_factor_analysis_alpha360(server_url):
    r = requests.get(f"{server_url}/api/factor/analysis?factor=Alpha360", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "metrics" in data


def test_models_registry(server_url):
    r = requests.get(f"{server_url}/api/models", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert len(data["models"]) > 0
    assert "handlers" in data
    assert len(data["handlers"]) > 0


def test_pipeline_run(server_url):
    r = requests.post(f"{server_url}/api/pipeline/run", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "runId" in data
    assert data["runId"].startswith("run_")


def test_backtest_run(server_url):
    r = requests.post(f"{server_url}/api/backtest/run", json={}, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "runId" in data
    assert data["runId"].startswith("bt_")


def test_backtest_results(server_url):
    r = requests.post(f"{server_url}/api/backtest/run", json={}, timeout=5)
    assert r.status_code == 200
    run_id = r.json()["runId"]
    r2 = requests.get(f"{server_url}/api/backtest/results/{run_id}", timeout=5)
    assert r2.status_code == 200
    data = r2.json()
    assert "results" in data
    assert "metrics" in data["results"]
    assert "daily" in data["results"]


def test_pipeline_status_not_found(server_url):
    r = requests.get(f"{server_url}/api/pipeline/status/nonexistent", timeout=5)
    assert r.status_code == 404
