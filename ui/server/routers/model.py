"""模型选股路由：模型列表、预测排名、单股时序、回测报告、实时选股。"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_model_service
from ..model_service import ModelService
from ..schemas import ModelRunRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


def _service(svc: ModelService = Depends(get_model_service)) -> ModelService:
    if svc is None:
        raise HTTPException(status_code=503, detail="ModelService not initialized")
    return svc


@router.get("")
def list_models(svc: ModelService = Depends(_service)):
    return {"models": svc.list_models()}


@router.post("/{alias}/run")
def run_prediction(alias: str, body: ModelRunRequest, svc: ModelService = Depends(_service)):
    try:
        result = svc.run_prediction(alias, date=body.date, top_k=body.top_k)
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("run_prediction error for %s", alias)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{alias}/info")
def model_info(alias: str, svc: ModelService = Depends(_service)):
    try:
        return svc.get_model_info(alias)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("model_info error for %s", alias)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{alias}/predictions")
def model_predictions(
    alias: str,
    date: Optional[str] = Query(None),
    top_k: int = Query(30),
    svc: ModelService = Depends(_service),
):
    try:
        return svc.get_predictions(alias, date=date, top_k=top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("model_predictions error for %s", alias)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{alias}/dates")
def model_dates(alias: str, svc: ModelService = Depends(_service)):
    try:
        dates = svc.get_available_dates(alias)
        return {"dates": dates}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("model_dates error for %s", alias)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{alias}/stock/{instrument}")
def model_stock_prediction(alias: str, instrument: str, svc: ModelService = Depends(_service)):
    try:
        return svc.get_stock_prediction(alias, instrument)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("model_stock_prediction error for %s/%s", alias, instrument)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{alias}/report")
def model_report(alias: str, svc: ModelService = Depends(_service)):
    try:
        return svc.get_backtest_report(alias)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("model_report error for %s", alias)
        raise HTTPException(status_code=500, detail=str(exc))
