"""Pytest fixtures for the QuantMaster UI server tests."""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

# Add ui/ to sys.path so `import server` works
_ui_dir = str(Path(__file__).resolve().parent.parent.parent)
if _ui_dir not in sys.path:
    sys.path.insert(0, _ui_dir)

from server import app  # noqa: E402
from server.config import LIVE_DATA_DIR
from server.routers import browser, execution, model, pipeline, position, strategy  # noqa: E402
from server.tests.mock_datadir import FakeDataDir, FakeTDXQuote  # noqa: E402


def _create_test_app(tmp_path):
    """Create a minimal FastAPI app for testing (no lifespan, no quant_master init)."""
    fastapi_app = FastAPI(title="QuantMaster Test API")

    @fastapi_app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    fastapi_app.include_router(browser.router, prefix="/api")
    fastapi_app.include_router(model.router, prefix="/api")
    fastapi_app.include_router(pipeline.router, prefix="/api")
    fastapi_app.include_router(position.router, prefix="/api")
    fastapi_app.include_router(strategy.router, prefix="/api")
    fastapi_app.include_router(execution.router, prefix="/api")


    # Initialize shared state (both on module and FastAPI app.state for Depends)
    fake_data = FakeDataDir(tmp_path)
    fake_tdx = FakeTDXQuote()
    app.data = fake_data
    app.tdx_quote = fake_tdx
    app.pipeline_runs = {}
    if hasattr(app, "backtest_runs"):
        app.backtest_runs = {}
    app.model_service = None
    fastapi_app.state.data = fake_data
    fastapi_app.state.tdx_quote = fake_tdx
    fastapi_app.state.model_service = None

    return fastapi_app


@pytest.fixture()
def client(tmp_path):
    """Provide a TestClient with mock data."""
    fastapi_app = _create_test_app(tmp_path)
    with TestClient(fastapi_app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_server_app_state(tmp_path, monkeypatch):
    app.data = FakeDataDir(tmp_path)
    app.tdx_quote = FakeTDXQuote()
    app.pipeline_runs = {}
    if hasattr(app, "backtest_runs"):
        app.backtest_runs = {}
    # 隔离 live_data 路径，防止测试写入真实持仓/执行历史
    monkeypatch.setattr("server.config.LIVE_DATA_DIR", tmp_path)
    monkeypatch.setattr("server.position_service.LIVE_DATA_DIR", tmp_path)
    monkeypatch.setattr("server.routers.browser.LIVE_DATA_DIR", tmp_path)
    monkeypatch.setattr("server.execution_service.LIVE_DATA_DIR", tmp_path)
