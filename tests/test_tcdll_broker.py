import io
import subprocess
import time

import pytest

from quant_master.contrib.broker import BrokerOrderDir, TcDllBroker


def test_tcdll_broker_dry_run_does_not_require_connection():
    broker = TcDllBroker(caller_path="missing.exe")

    order = broker.buy("000001", 12.3, 100, dry_run=True)

    assert order.order_id == "DRY-RUN"
    assert order.direction == BrokerOrderDir.BUY


def test_tcdll_broker_requires_live_flag_for_real_order():
    broker = TcDllBroker(caller_path="missing.exe")

    try:
        broker.buy("000001", 12.3, 100)
    except RuntimeError as exc:
        assert "enable_live=True" in str(exc)
    else:
        raise AssertionError("TcDllBroker should reject live orders without enable_live=True")


def test_tcdll_broker_rejects_live_helper_command_without_live_flag():
    broker = TcDllBroker(caller_path="missing.exe")

    with pytest.raises(RuntimeError, match="enable_live=True"):
        broker._command("GRIDJY 1 Zqdm=000001|")


def test_tcdll_broker_rejects_unknown_safe_probe():
    broker = TcDllBroker(caller_path="missing.exe")

    with pytest.raises(ValueError, match="Unsupported safe Tc.dll probe"):
        broker.safe_probe("gridjy")


def test_tcdll_broker_status_probe_can_return_safe_rejection():
    broker = TcDllBroker(caller_path="missing.exe")
    broker._proc = _FakeProc(stdout=io.StringIO("ERR unsafe STATUS disabled; use STATUS_RAW for crash diagnostics\n"), stderr=io.StringIO(""))
    broker._connected = True
    broker._start_readers()

    assert broker.safe_probe("status").startswith("ERR unsafe STATUS")


def test_tcdll_broker_gridjy_dryrun_uses_non_live_command(tmp_path):
    broker = TcDllBroker(caller_path="missing.exe")
    broker._proc = _FakeProc(stdout=io.StringIO("OK GRIDJY_DRYRUN entry=DRYRUN params_len=4 args=7 no_dll_call=1\n"), stderr=io.StringIO(""))
    broker._connected = True
    broker._start_readers()

    line = broker.gridjy_dryrun("DRYRUN", "A=B|")

    assert line.startswith("OK GRIDJY_DRYRUN")
    assert broker._proc.stdin.getvalue() == "GRIDJY_DRYRUN DRYRUN A=B|\n"


def test_tcdll_broker_probe_returns_structured_payload():
    broker = TcDllBroker(caller_path="missing.exe")
    broker._proc = _FakeProc(stdout=io.StringIO("OK GRIDJY_DRYRUN entry=DRYRUN params_len=4 args=7 no_dll_call=1\n"), stderr=io.StringIO(""))
    broker._connected = True
    broker._start_readers()

    payload = broker.probe("gridjy-dryrun", gridjy_entry="DRYRUN", gridjy_params="A=B|")

    assert payload["probe"] == "gridjy_dryrun"
    assert payload["ok"] is True
    assert payload["entry"] == "DRYRUN"
    assert payload["params_len"] == 4
    assert payload["args"] == 7
    assert payload["no_dll_call"] == 1
    assert payload["raw"].startswith("OK GRIDJY_DRYRUN")
    assert broker._proc.stdin.getvalue() == "GRIDJY_DRYRUN DRYRUN A=B|\n"


def test_tcdll_broker_parse_probe_line_keeps_error_as_structured_payload():
    payload = TcDllBroker.parse_probe_line("ERR unsafe STATUS disabled; use STATUS_RAW for crash diagnostics")

    assert payload["ok"] is False
    assert payload["status"] == "ERR"
    assert payload["raw"].startswith("ERR unsafe STATUS")
    assert "unsafe STATUS" in payload["message"]


def test_tcdll_broker_uses_explicit_xiadan_work_dir(monkeypatch, tmp_path):
    caller = tmp_path / "tdx_caller.exe"
    caller.write_text("stub")
    work_dir = tmp_path / "xiadan"
    work_dir.mkdir()
    dll_dir = tmp_path / "dlls"
    dll_dir.mkdir()
    popen_calls = []

    class FakePopen(_FakeProc):
        def __init__(self, args, **kwargs):
            popen_calls.append((args, kwargs))
            super().__init__(stdout=io.StringIO("READY\n"), stderr=io.StringIO(""))

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    broker = TcDllBroker(caller_path=str(caller), xiadan_work_dir=str(work_dir), dll_dir=str(dll_dir))

    assert broker.connect() is True

    args, kwargs = popen_calls[0]
    assert kwargs["cwd"] == str(work_dir)
    assert "--work-dir" in args
    assert str(work_dir) in args
    assert "--dll-dir" in args
    assert str(dll_dir) in args


def test_tcdll_broker_drains_stderr_without_blocking():
    broker = TcDllBroker(caller_path="missing.exe")
    stdout = io.StringIO("READY\nOK PING\n")
    stderr = io.StringIO(("diagnostic line\n" * 5000))
    proc = _FakeProc(stdout=stdout, stderr=stderr)
    broker._proc = proc

    broker._start_readers()

    assert broker._readline(timeout=1.0) == "READY"
    assert broker._readline(timeout=1.0) == "OK PING"

    deadline = time.time() + 1.0
    while "diagnostic line" not in broker._recent_stderr() and time.time() < deadline:
        time.sleep(0.01)
    assert "diagnostic line" in broker._recent_stderr()


def test_tcdll_broker_timeout_includes_stderr_tail():
    broker = TcDllBroker(caller_path="missing.exe", command_timeout=0.01)
    proc = _FakeProc(stdout=_BlockingStream(), stderr=io.StringIO("CALL_BEGIN function=TC_CreateAll\n"))
    proc.returncode = None
    broker._proc = proc

    broker._start_readers()

    deadline = time.time() + 1.0
    while "TC_CreateAll" not in broker._recent_stderr() and time.time() < deadline:
        time.sleep(0.01)
    with pytest.raises((RuntimeError, TimeoutError)) as exc_info:
        broker._readline(timeout=0.01)
    assert "TC_CreateAll" in str(exc_info.value)


class _FakeProc:
    def __init__(self, *, stdout, stderr):
        self.stdin = io.StringIO()
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class _BlockingStream:
    def readline(self):
        time.sleep(1.0)
        return ""
