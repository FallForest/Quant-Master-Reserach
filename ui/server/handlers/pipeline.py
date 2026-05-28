"""数据管线：同步状态、触发同步、管线运行。"""
import threading
import time

from .. import app
from ..sync import get_sync_status, _get_last_update_date, auto_sync_daily


def global_status(rh):
    last_date = _get_last_update_date(app.data.data_dir if app.data else "")
    resp = {"lastUpdate": last_date or "--"}
    sync_st = get_sync_status()
    if sync_st["running"]:
        resp["syncing"] = True
    if sync_st["lastError"]:
        resp["syncError"] = sync_st["lastError"]
    return rh._json_response(resp)


def run(rh):
    run_id = f"run_{app.next_pipeline_counter()}"
    app.pipeline_runs[run_id] = {
        "startTime": time.time(),
        "done": False,
        "success": False,
        "logs": [],
        "step": "同步中",
        "progress": 0,
    }
    st = get_sync_status()
    if not st["running"]:
        t = threading.Thread(
            target=auto_sync_daily,
            args=(app.data.data_dir, app.data),
            daemon=True,
        )
        t.start()
        app.pipeline_runs[run_id]["logs"].append(
            {"level": "info", "msg": f"[{time.strftime('%H:%M:%S')}] 数据同步已启动"}
        )
    else:
        app.pipeline_runs[run_id]["logs"].append(
            {"level": "info", "msg": f"[{time.strftime('%H:%M:%S')}] 同步已在进行中"}
        )
    rh._json_response({"runId": run_id})


def status(rh, run_id):
    run = app.pipeline_runs.get(run_id)
    if not run:
        return rh._json_response({"error": "run not found"}, status=404)
    sync_st = get_sync_status()
    was_started = run.get("step") != "排队中"
    done = run.get("done", False)
    if not done and was_started and not sync_st["running"]:
        done = True
        run["done"] = True
        run["success"] = not sync_st.get("lastError")
        run["logs"].append({"level": "success", "msg": f"[{time.strftime('%H:%M:%S')}] 数据同步完成"})
    success = run.get("success", done and not sync_st.get("lastError"))
    resp = {
        "progress": 100 if done else (50 if sync_st["running"] else 0),
        "step": "完成" if done else run.get("step", "同步中"),
        "logs": run.get("logs", []),
        "done": done,
        "success": success,
    }
    if sync_st.get("lastError"):
        resp["error"] = sync_st["lastError"]
    rh._json_response(resp)


def sync_trigger(rh):
    """手动触发数据同步。"""
    st = get_sync_status()
    if st["running"]:
        rh._json_response({"ok": False, "error": "同步正在进行中"})
        return
    t = threading.Thread(
        target=auto_sync_daily,
        args=(app.data.data_dir, app.data),
        daemon=True,
    )
    t.start()
    rh._json_response({"ok": True, "msg": "同步已启动"})
