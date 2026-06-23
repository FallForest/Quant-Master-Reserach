"""Tests for endpoints that need data (using FakeDataDir via the client fixture)."""
import pytest

from server import app


class _EmptyTDXQuote:
    def fetch_quotes(self, symbols):
        return {}

    def get_quote(self, symbol):
        return None

    def get_today_kline(self, symbol):
        return []

    def fetch_today_day_bar_from_eastmoney(self, symbol):
        return None


class _FutureDayTDXQuote:
    def fetch_quotes(self, symbols):
        return {}

    def get_quote(self, symbol):
        return None

    def get_today_kline(self, symbol):
        return [
            {"date": "2025-02-21 09:30", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000},
            {"date": "2025-02-21 10:30", "open": 10.1, "high": 10.4, "low": 10.0, "close": 10.35, "volume": 2000},
        ]


class _SameDayTDXQuote:
    def fetch_quotes(self, symbols):
        return {}

    def get_quote(self, symbol):
        return None

    def get_today_kline(self, symbol):
        return [
            {"date": "2025-02-20 09:30", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000},
            {"date": "2025-02-20 10:30", "open": 10.1, "high": 10.4, "low": 10.0, "close": 10.35, "volume": 2000},
        ]


def test_browser_stocks(client):
    r = client.get("/api/browser/stocks")
    assert r.status_code == 200
    data = r.json()
    assert "stocks" in data
    assert len(data["stocks"]) == 3
    for item in data["stocks"]:
        assert "symbol" in item
        assert "name" in item


def test_browser_quotes(client):
    r = client.get("/api/browser/quotes?symbols=sh600001,sh600002,sh600003")
    assert r.status_code == 200
    data = r.json()
    assert "quotes" in data
    assert len(data["quotes"]) == 3


def test_browser_kline(client):
    r = client.get("/api/browser/kline/sh600001")
    assert r.status_code == 200
    data = r.json()
    assert "kline" in data
    assert isinstance(data["kline"], list)
    assert len(data["kline"]) > 0
    assert "quote" in data
    assert isinstance(data["quote"], dict)
    assert data["realtime"]["included"] is False


def test_browser_kline_include_realtime_appends_today_bar(monkeypatch, client):
    monkeypatch.setattr(app, "tdx_quote", _FutureDayTDXQuote())

    r = client.get("/api/browser/kline/sh600001?freq=1d&includeRealtime=1")
    assert r.status_code == 200
    data = r.json()
    assert data["realtime"]["included"] is True
    assert data["realtime"]["date"] == "2025-02-21"
    assert data["realtime"]["partial"] is True
    last = data["kline"][-1]
    assert last["date"] == "2025-02-21"
    assert last["realtime"] is True
    assert last["partial"] is True
    assert data["quote"]["lastClose"] == data["kline"][-2]["close"]


def test_browser_kline_include_realtime_replaces_same_day(monkeypatch, client):
    monkeypatch.setattr(app, "tdx_quote", _SameDayTDXQuote())

    r = client.get("/api/browser/kline/sh600001?freq=1d&includeRealtime=1")
    assert r.status_code == 200
    data = r.json()
    dates = [item["date"] for item in data["kline"]]
    assert dates.count("2025-02-20") == 1
    last = data["kline"][-1]
    assert last["date"] == "2025-02-20"
    assert last["close"] == 10.35
    assert last["realtime"] is True
    assert data["quote"]["lastClose"] == data["kline"][-2]["close"]


def test_browser_kline_include_realtime_gracefully_falls_back(monkeypatch, client):
    monkeypatch.setattr(app, "tdx_quote", _EmptyTDXQuote())

    historical = client.get("/api/browser/kline/sh600001?freq=1d").json()
    r = client.get("/api/browser/kline/sh600001?freq=1d&includeRealtime=1")
    assert r.status_code == 200
    data = r.json()
    assert data["realtime"]["included"] is False
    assert data["kline"] == historical["kline"]
    assert data["quote"] == historical["quote"]


def test_overview(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    data = r.json()
    assert data["stockCount"] == 3
    assert data["calendarDays"] == 30
    assert data["effectiveLastDate"] == "2025-02-20"
    assert data["calendarLastDate"] == "2025-02-20"
    assert data["marketEffectiveLastDate"] == "2025-02-20"
    assert data["equityCount"] == 3
    assert data["equityCoveredAtLastDate"] == 3
    assert data["equityCoverageAtLastDate"] == 1.0
    assert "fieldStats" in data
    assert isinstance(data["fieldStats"], list)


def test_pipeline_global_status(client):
    r = client.get("/api/pipeline/status")
    assert r.status_code == 200
    data = r.json()
    assert data["lastUpdate"] == "2025-02-20"
    assert data["effectiveLastDate"] == "2025-02-20"
    assert data["calendarLastDate"] == "2025-02-20"
    assert data["marketEffectiveLastDate"] == "2025-02-20"
    assert data["equityCount"] == 3
    assert data["equityCoveredAtLastDate"] == 3
    assert data["equityCoverageAtLastDate"] == 1.0
    assert "dataDir" in data
    assert "syncStats" in data


def test_watchlist_crud(client):
    r = client.get("/api/watchlist")
    assert r.status_code == 200
    data = r.json()
    initial_symbols = list(data["symbols"])
    initial_count = data["count"]

    r = client.post("/api/watchlist", json={"symbol": "600001"})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "SH600001" in data["symbols"]
    assert data["count"] >= initial_count

    r = client.post("/api/watchlist", json={"symbol": "sh600001"})
    assert r.status_code == 200
    data = r.json()
    assert data["symbols"].count("SH600001") == 1

    r = client.post("/api/watchlist", json={"symbol": "000001"})
    assert r.status_code == 200
    data = r.json()
    assert "SH600001" in data["symbols"]
    assert "SZ000001" in data["symbols"]

    r = client.delete("/api/watchlist/600001")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "SH600001" not in data["symbols"]


def test_watchlist_invalid_symbol(client):
    r = client.post("/api/watchlist", json={"symbol": ""})
    assert r.status_code == 400
    data = r.json()
    assert data["error"] == "symbol 不能为空"
