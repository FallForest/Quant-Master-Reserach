"""Regression tests for data-sync status and trigger behavior."""

import numpy as np
import threading
import time

from server import sync
from server.datadir import DataDir
from server.routers import pipeline


def _reset_sync_status():
    sync._set_sync_status(
        running=False,
        lastSync=None,
        lastError=None,
        lastStats=None,
        progressPhase=None,
        progressTotal=0,
        progressDone=0,
        progressLabel=None,
    )


def test_background_start_publishes_progress_before_thread_runs(monkeypatch):
    class DeferredThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return None

    _reset_sync_status()
    monkeypatch.setattr(sync.threading, "Thread", DeferredThread)
    try:
        assert sync.start_auto_sync_daily("unused", force=True) is True
        status = sync.get_sync_status()
        assert status["running"] is True
        assert status["progressPhase"] == "starting"
        assert status["progressLabel"]
        assert sync.start_auto_sync_daily("unused", force=True) is False
    finally:
        _reset_sync_status()


def test_running_status_uses_snapshot_without_rescanning(monkeypatch):
    status = {
        "running": True,
        "lastSync": "2026-07-21",
        "lastError": None,
        "lastStats": {"effectiveLastDate": "2026-07-21", "equityCount": 5200},
        "progressPhase": "fetching",
        "progressTotal": 5200,
        "progressDone": 125,
        "progressLabel": "fetching",
    }
    monkeypatch.setattr(pipeline, "get_sync_status", lambda: status)
    monkeypatch.setattr(
        pipeline,
        "get_data_health_snapshot",
        lambda _data_dir: (_ for _ in ()).throw(AssertionError("unexpected health scan")),
    )

    response = pipeline._build_status_response("unused")

    assert response["syncing"] is True
    assert response["lastUpdate"] == "2026-07-21"
    assert response["equityCount"] == 5200
    assert response["syncProgress"] == {
        "phase": "fetching",
        "total": 5200,
        "done": 125,
        "label": "fetching",
    }


def test_status_exposes_independent_cache_refresh(monkeypatch):
    status = {
        "running": False,
        "lastSync": "2026-07-21",
        "lastError": None,
        "lastStats": {},
        "progressPhase": None,
        "progressTotal": 0,
        "progressDone": 0,
        "progressLabel": None,
    }
    cache = {"running": True, "startedAt": 123.0, "finishedAt": None, "lastError": None}
    monkeypatch.setattr(pipeline, "get_sync_status", lambda: status)
    monkeypatch.setattr(pipeline, "get_cache_status", lambda: cache)
    monkeypatch.setattr(pipeline, "get_data_health_snapshot", lambda _data_dir: {
        "effectiveLastDate": "2026-07-21",
        "calendarLastDate": "2026-07-21",
        "marketEffectiveLastDate": "2026-07-21",
        "equityCoverageAtLastDate": 1.0,
        "equityCoveredAtLastDate": 1,
        "equityCount": 1,
        "calendarCoverage": 1.0,
        "calendarCoveredEquities": 1,
        "calendarHealthy": True,
        "calendarInvalidLineCount": 0,
        "sampleInvalidCalendarLines": [],
        "calendarDuplicateCount": 0,
        "calendarOrdered": True,
    })

    response = pipeline._build_status_response("unused")

    assert response["syncing"] is False
    assert response["cacheRefresh"] == cache


def test_cache_refresh_coalesces_requests_during_a_running_build(monkeypatch):
    calls = []
    finished = threading.Event()

    def fake_build(data):
        calls.append(data)
        if len(calls) == 1:
            sync.start_cache_refresh(data)
        else:
            finished.set()

    monkeypatch.setattr(sync, "build_stock_summary", fake_build)
    with sync._cache_lock:
        sync._cache_status.update(running=False, startedAt=None, finishedAt=None, lastError=None)
        sync._cache_refresh_pending = False

    assert sync.start_cache_refresh("data") is True
    assert finished.wait(1.0)
    deadline = time.time() + 1.0
    while sync.get_cache_status()["running"] and time.time() < deadline:
        time.sleep(0.01)
    assert calls == ["data", "data"]
    assert sync.get_cache_status()["running"] is False


def test_trigger_returns_initial_progress(client, monkeypatch):
    initial = {
        "running": True,
        "progressPhase": "starting",
        "progressTotal": 0,
        "progressDone": 0,
        "progressLabel": "正在检查本地数据...",
    }
    monkeypatch.setattr(pipeline, "start_auto_sync_daily", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(pipeline, "get_sync_status", lambda: initial)

    response = client.post("/api/pipeline/trigger")

    assert response.status_code == 200
    assert response.json()["syncProgress"] == {
        "phase": "starting",
        "total": 0,
        "done": 0,
        "label": "正在检查本地数据...",
    }


def test_tail_preparation_uses_file_sizes_and_reports_progress(tmp_path, monkeypatch):
    data_root = tmp_path / "tail-case"
    symbol_dir = data_root / "features" / "sh600000"
    symbol_dir.mkdir(parents=True)
    close_path = symbol_dir / "close.day.bin"
    open_path = symbol_dir / "open.day.bin"
    np.array([0, 1, 2, 3, 4], dtype="<f4").tofile(close_path)
    np.array([0, 1, 2], dtype="<f4").tofile(open_path)
    real_fromfile = np.fromfile
    read_counts = []

    def tracked_fromfile(*args, **kwargs):
        read_counts.append(kwargs.get("count"))
        return real_fromfile(*args, **kwargs)

    monkeypatch.setattr(sync.np, "fromfile", tracked_fromfile)
    progress = []

    updated = sync._extend_tail_files(
        data_root,
        ["2026-07-20", "2026-07-21", "2026-07-22"],
        ["open", "close"],
        extra_buffer=1,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert updated == 1
    assert read_counts and all(count == 1 for count in read_counts)
    assert open_path.stat().st_size == close_path.stat().st_size
    assert progress == [(0, 1), (1, 1)]


def test_missing_instrument_manifest_is_rebuilt_from_provider(tmp_path, monkeypatch):
    class FakeQuote:
        def fetch_instruments(self):
            return {
                "SH600000": {"code": "600000", "name": "Alpha"},
                "SZ000001": {"code": "000001", "name": "Beta"},
            }

        def _invalidate(self):
            return None

    monkeypatch.setattr(sync, "TDXQuote", FakeQuote)

    count = sync._discover_instruments_from_tdx(tmp_path)

    assert count == 2
    assert DataDir(str(tmp_path)).get_instruments() == [
        ("SH600000", "1900-01-01", "1900-01-01"),
        ("SZ000001", "1900-01-01", "1900-01-01"),
    ]
    assert "600000\tAlpha" in (tmp_path / "instruments" / "names.txt").read_text(encoding="utf-8")


def test_missing_instrument_manifest_is_an_empty_snapshot(tmp_path):
    # 用独立子目录：autouse fixture 会把 FakeDataDir 的伪数据写进 tmp_path 本身，
    # 直接拿 tmp_path 当 provider 读到的是那份清单，而不是"清单缺失"的情形。
    empty_provider = tmp_path / "empty_provider"
    empty_provider.mkdir()
    assert DataDir(str(empty_provider)).get_instruments() == []
