"""Tests for async endpoints (stock-select, backtest status)."""
import pytest
import requests


def test_stock_select_run(server_url):
    config = {
        "model_id": "lgb",
        "handler_id": "alpha158",
        "universe": "50",
        "test_date": "2025-02-20",
        "train_start": "2018-01-01",
        "train_end": "2023-12-31",
        "valid_start": "2024-01-01",
        "valid_end": "2025-06-30",
        "top_n": 10,
    }
    r = requests.post(f"{server_url}/api/stock-select/run", json=config, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "runId" in data


def test_stock_select_status(server_url):
    # A run was started by test_stock_select_run; ss_1 is the first run id.
    # Check its status (it may be running or already finished).
    r = requests.get(f"{server_url}/api/stock-select/status/ss_1", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "progress" in data
    assert "done" in data


def test_stock_select_status_not_found(server_url):
    r = requests.get(f"{server_url}/api/stock-select/status/bogus", timeout=5)
    assert r.status_code == 404


def test_stock_select_results_not_found(server_url):
    r = requests.get(f"{server_url}/api/stock-select/results/bogus", timeout=5)
    assert r.status_code == 404


def test_backtest_status_not_found(server_url):
    r = requests.get(f"{server_url}/api/backtest/status/bogus", timeout=5)
    assert r.status_code == 404
