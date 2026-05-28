"""Tests for endpoints that need data (using FakeDataDir via the server fixture)."""
import pytest
import requests


def test_browser_stocks(server_url):
    r = requests.get(f"{server_url}/api/browser/stocks", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "stocks" in data
    assert len(data["stocks"]) == 3
    for item in data["stocks"]:
        assert "symbol" in item
        assert "name" in item


def test_browser_quotes(server_url):
    r = requests.get(f"{server_url}/api/browser/quotes", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "quotes" in data
    assert len(data["quotes"]) == 3


def test_browser_kline(server_url):
    r = requests.get(f"{server_url}/api/browser/kline/sh600001", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "kline" in data
    assert isinstance(data["kline"], list)
    assert len(data["kline"]) > 0
    assert "quote" in data
    assert isinstance(data["quote"], dict)


def test_overview(server_url):
    r = requests.get(f"{server_url}/api/overview", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["stockCount"] == 3
    assert data["calendarDays"] == 30
    assert "fieldStats" in data
    assert isinstance(data["fieldStats"], list)


def test_pipeline_global_status(server_url):
    r = requests.get(f"{server_url}/api/pipeline/status", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "lastUpdate" in data
