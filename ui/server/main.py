"""FastAPI 主应用。"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import quant_master
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from datetime import datetime
from . import app
from .config import CORS_ORIGINS, DIST_DIR
from .datadir import DEFAULT_DATA_DIR, create_data_dir, get_effective_data_dir
from .model_service import ModelService
from .routers import browser, execution, model, pipeline, position, strategy
from .stock_cache import build_stock_summary
from .sync import auto_sync_daily, schedule_daily_sync
from .tdx_quote import TDXQuote

_log = logging.getLogger(__name__)

_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".map": "application/json",
}


def _write_runtime_pid():
    pid_file = Path(__file__).resolve().parent.parent / ".server-pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """应用生命周期：启动时初始化服务，关闭时清理。"""
    data_dir = fastapi_app.state._data_dir
    port = fastapi_app.state._port

    # 初始化全局状态
    app.data = create_data_dir(data_dir)
    app.tdx_quote = TDXQuote()
    threading.Thread(target=app.tdx_quote._get_api, daemon=True).start()

    provider_uri = str(Path(data_dir).expanduser().resolve())
    quant_master.init(provider_uri=provider_uri, region="cn")
    _log.info("QuantMaster initialized with provider_uri=%s", provider_uri)

    app.model_service = ModelService()

    # 同步到 FastAPI app.state，供 dependencies.py 注入使用
    fastapi_app.state.data = app.data
    fastapi_app.state.tdx_quote = app.tdx_quote
    fastapi_app.state.model_service = app.model_service

    # 启动时同步构建股票摘要缓存，保证第一个 /browser/stocks 请求即可命中缓存
    _log.info("Building stock summary cache...")
    build_stock_summary(app.data)
    _log.info("Stock summary cache ready")

    # 后台同步
    if datetime.now().hour >= 15:
        threading.Thread(target=auto_sync_daily, args=(None, app.data), daemon=True).start()
    threading.Thread(target=schedule_daily_sync, args=(None, app.data), daemon=True).start()

    _write_runtime_pid()
    _log.info("API server running on http://127.0.0.1:%s", port)
    _log.info("Data dir: %s", get_effective_data_dir(app.data, data_dir))
    _log.info("Model registry: %d model(s) loaded", len(app.model_service.list_models()))

    yield


def create_app(data_dir: str = str(DEFAULT_DATA_DIR), port: int = 5174) -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    fastapi_app = FastAPI(
        title="QuantMaster API",
        description="量化投资平台 API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # 保存启动参数到 app.state
    fastapi_app.state._data_dir = data_dir
    fastapi_app.state._port = port

    # CORS
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 自定义异常处理：将 HTTPException 的 detail 转为 {"error": "..."} 格式（兼容前端）
    @fastapi_app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    # 注册路由
    fastapi_app.include_router(browser.router, prefix="/api")
    fastapi_app.include_router(model.router, prefix="/api")
    fastapi_app.include_router(pipeline.router, prefix="/api")
    fastapi_app.include_router(position.router, prefix="/api")
    fastapi_app.include_router(strategy.router, prefix="/api")
    fastapi_app.include_router(execution.router, prefix="/api")


    # 静态文件 + SPA fallback
    if DIST_DIR.exists():
        fastapi_app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

        @fastapi_app.get("/{path:path}")
        async def serve_static(path: str):
            """SPA fallback：非 API 路径返回对应文件，不存在则返回 index.html。"""
            if path.startswith("api/"):
                return JSONResponse(status_code=404, content={"error": "Not Found"})
            file_path = DIST_DIR / path
            if file_path.is_file():
                ext = file_path.suffix.lower()
                media_type = _MIME_TYPES.get(ext, "application/octet-stream")
                return FileResponse(str(file_path), media_type=media_type)
            # SPA fallback
            index_path = DIST_DIR / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path), media_type="text/html; charset=utf-8")
            return JSONResponse(status_code=404, content={"error": "Not Found"})

    return fastapi_app
