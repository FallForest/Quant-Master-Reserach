"""数据管线：同步状态与触发同步。"""
import threading

from .. import app
from ..datadir import get_effective_data_dir
from ..sync import get_sync_status, get_data_health_snapshot, auto_sync_daily


def _build_status_response(effective_dir):
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
        "dataDir": effective_dir,
        "syncStats": stats,
    }
    if sync_st["running"]:
        resp["syncing"] = True
    if sync_st["lastError"]:
        resp["syncError"] = sync_st["lastError"]
    return resp


def global_status(rh):
    effective_dir = get_effective_data_dir(app.data)
    return rh._json_response(_build_status_response(effective_dir))


def sync_trigger(rh):
    """手动触发数据同步。"""
    st = get_sync_status()
    if st["running"]:
        rh._json_response({"ok": False, "error": "同步正在进行中"})
        return
    t = threading.Thread(
        target=auto_sync_daily,
        args=(None, app.data),
        daemon=True,
    )
    t.start()
    rh._json_response({"ok": True, "msg": "同步已启动"})
