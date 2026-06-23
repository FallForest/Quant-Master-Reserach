"""执行路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..execution_service import build_order_preview, get_execution_config, load_history, submit_orders
from ..schemas import ExecutionPreviewRequest, ExecutionSubmitRequest

router = APIRouter(prefix="/execution", tags=["execution"])


@router.get("/config")
def config():
    return get_execution_config()


@router.post("/preview")
def preview(body: ExecutionPreviewRequest):
    return build_order_preview(body.trades, risk=body.risk or {})


@router.post("/submit")
def submit(body: ExecutionSubmitRequest):
    try:
        return submit_orders(
            body.orders,
            broker_kind=body.brokerKind,
            dry_run=body.dryRun,
            risk=body.risk or {},
            confirm=bool(body.confirm),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
def history(limit: int = 30):
    return load_history(limit=limit)
