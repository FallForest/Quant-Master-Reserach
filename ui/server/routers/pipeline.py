"""数据管线路由：同步状态与触发同步。"""
from __future__ import annotations

from fastapi import APIRouter

from .. import app
from ..datadir import get_effective_data_dir
from ..sync import get_cache_status, get_data_health_snapshot, get_sync_status, start_auto_sync_daily

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _build_status_response(effective_dir: str) -> dict:
    sync_st = get_sync_status()
    stats = sync_st.get("lastStats") or {}
    if sync_st["running"]:
        # A full health scan walks every feature file. During sync, serve the
        # previous snapshot so frequent progress polling remains responsive.
        health = {
            "effectiveLastDate": stats.get("effectiveLastDate") or sync_st.get("lastSync"),
            "calendarLastDate": stats.get("calendarLastDate"),
            "marketEffectiveLastDate": stats.get("marketEffectiveLastDate") or sync_st.get("lastSync"),
            "equityCoverageAtLastDate": stats.get("equityCoverageAtLastDate", 0.0),
            "equityCoveredAtLastDate": stats.get("equityCoveredAtLastDate", 0),
            "equityCount": stats.get("equityCount", 0),
            "calendarCoverage": stats.get("calendarCoverage", 0.0),
            "calendarCoveredEquities": stats.get("calendarCoveredEquities", 0),
            "calendarHealthy": stats.get("calendarHealthy", True),
            "calendarInvalidLineCount": stats.get("calendarInvalidLineCount", 0),
            "sampleInvalidCalendarLines": stats.get("sampleInvalidCalendarLines", []),
            "calendarDuplicateCount": stats.get("calendarDuplicateCount", 0),
            "calendarOrdered": stats.get("calendarOrdered", True),
        }
    else:
        health = get_data_health_snapshot(effective_dir)
    cache_status = get_cache_status()
    resp = {
        "lastUpdate": health["effectiveLastDate"] or "--",
        "effectiveLastDate": health["effectiveLastDate"] or "--",
        "calendarLastDate": health["calendarLastDate"] or "--",
        "marketEffectiveLastDate": health["marketEffectiveLastDate"] or "--",
        "equityCoverageAtLastDate": health["equityCoverageAtLastDate"],
        "equityCoveredAtLastDate": health["equityCoveredAtLastDate"],
        "equityCount": health["equityCount"],
        "calendarCoverage": health["calendarCoverage"],
        "calendarCoveredEquities": health["calendarCoveredEquities"],
        "calendarHealthy": health.get("calendarHealthy", True),
        "calendarInvalidLineCount": health.get("calendarInvalidLineCount", 0),
        "sampleInvalidCalendarLines": health.get("sampleInvalidCalendarLines", []),
        "calendarDuplicateCount": health.get("calendarDuplicateCount", 0),
        "calendarOrdered": health.get("calendarOrdered", True),
        "dataDir": effective_dir,
        "syncStats": stats,
        "syncing": bool(sync_st["running"]),
        "cacheRefresh": cache_status,
    }
    if sync_st["lastError"]:
        resp["syncError"] = sync_st["lastError"]
    if cache_status.get("lastError"):
        resp["cacheError"] = cache_status["lastError"]
    # 进度信息（同步中进行时有效）
    if sync_st.get("progressPhase"):
        resp["syncProgress"] = {
            "phase": sync_st["progressPhase"],
            "total": sync_st["progressTotal"],
            "done": sync_st["progressDone"],
            "label": sync_st["progressLabel"],
        }
    return resp


@router.get("/status")
def global_status():
    effective_dir = get_effective_data_dir(app.data)
    return _build_status_response(effective_dir)


@router.post("/trigger")
def sync_trigger():
    if not start_auto_sync_daily(None, app.data, force=True):
        return {"ok": False, "error": "同步正在进行中"}
    sync_st = get_sync_status()
    return {
        "ok": True,
        "msg": "同步已启动",
        "syncProgress": {
            "phase": sync_st["progressPhase"],
            "total": sync_st["progressTotal"],
            "done": sync_st["progressDone"],
            "label": sync_st["progressLabel"],
        },
    }
