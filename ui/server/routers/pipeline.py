"""数据管线路由：同步状态与触发同步。"""
from __future__ import annotations

import threading

from fastapi import APIRouter

from .. import app
from ..datadir import get_effective_data_dir
from ..sync import auto_sync_daily, get_data_health_snapshot, get_sync_status

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _build_status_response(effective_dir: str) -> dict:
    health = get_data_health_snapshot(effective_dir)
    sync_st = get_sync_status()
    stats = sync_st.get("lastStats") or {}
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
    }
    if sync_st["running"]:
        resp["syncing"] = True
    if sync_st["lastError"]:
        resp["syncError"] = sync_st["lastError"]
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
    st = get_sync_status()
    if st["running"]:
        return {"ok": False, "error": "同步正在进行中"}
    t = threading.Thread(target=auto_sync_daily, args=(None, app.data), kwargs={"force": True}, daemon=True)
    t.start()
    return {"ok": True, "msg": "同步已启动"}
