"""FastAPI 依赖注入。"""
from __future__ import annotations

from fastapi import Request

from .datadir import DataDir
from .model_service import ModelService
from .tdx_quote import TDXQuote


def get_data(request: Request) -> DataDir:
    return request.app.state.data


def get_tdx_quote(request: Request) -> TDXQuote:
    return request.app.state.tdx_quote


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service
