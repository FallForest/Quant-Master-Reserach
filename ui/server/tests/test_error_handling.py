"""Tests for error handling and CORS behaviour."""


def test_unknown_get_returns_404(client):
    r = client.get("/api/nonexistent")
    assert r.status_code == 404


def test_unknown_post_returns_404(client):
    r = client.post("/api/nonexistent")
    assert r.status_code == 404


def test_options_returns_200_with_cors(client):
    r = client.options(
        "/api/overview",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") is not None


def test_cors_on_get(client):
    r = client.get("/api/overview", headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") is not None
