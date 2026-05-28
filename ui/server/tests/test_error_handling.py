"""Tests for error handling and CORS behaviour."""
import pytest
import requests


def test_unknown_get_returns_404(server_url):
    r = requests.get(f"{server_url}/api/nonexistent", timeout=5)
    assert r.status_code == 404


def test_unknown_post_returns_404(server_url):
    r = requests.post(f"{server_url}/api/nonexistent", timeout=5)
    assert r.status_code == 404


def test_options_returns_200_with_cors(server_url):
    r = requests.options(f"{server_url}/api/overview", timeout=5)
    assert r.status_code == 200
    assert r.headers.get("Access-Control-Allow-Origin") == "*"


def test_cors_on_get(server_url):
    r = requests.get(f"{server_url}/api/strategies", timeout=5)
    assert r.headers.get("Access-Control-Allow-Origin") == "*"


def test_pipeline_status_missing_run(server_url):
    r = requests.get(f"{server_url}/api/pipeline/status/bogus_run", timeout=5)
    assert r.status_code == 404
    data = r.json()
    assert "error" in data
