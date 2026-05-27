import json
import sys

import scripts.test_tcdll_connect as cli


class CliBroker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.connected = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def is_connected(self):
        return self.connected

    def diagnostics(self):
        return {
            "caller_path": self.kwargs.get("caller_path"),
            "xiadan_work_dir": self.kwargs.get("xiadan_work_dir"),
            "dll_dir": self.kwargs.get("dll_dir"),
            "connected": self.connected,
        }

    def probe(self, probe, *, gridjy_entry="DRYRUN", gridjy_params=""):
        return {
            "ok": True,
            "status": "OK",
            "probe": probe.replace("-", "_"),
            "entry": gridjy_entry,
            "params_len": len(gridjy_params),
            "raw": "OK GRIDJY_DRYRUN entry=DRYRUN params_len=4 args=7 no_dll_call=1",
        }


def test_tcdll_connect_outputs_structured_probe(monkeypatch, capsys):
    monkeypatch.setattr(cli, "TcDllBroker", CliBroker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_tcdll_connect.py",
            "--caller-path",
            r"C:\tools\tdx_caller.exe",
            "--xiadan-work-dir",
            r"C:\silkriver\xiadan",
            "--dll-dir",
            r"C:\silkriver\dlls",
            "--probe",
            "gridjy-dryrun",
            "--gridjy-entry",
            "DRYRUN",
            "--gridjy-params",
            "A=B|",
        ],
    )

    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["connected"] is True
    assert payload["probe"] == "gridjy_dryrun"
    assert payload["result"]["ok"] is True
    assert payload["result"]["entry"] == "DRYRUN"
    assert payload["result"]["params_len"] == 4
    assert payload["diagnostics"]["xiadan_work_dir"] == r"C:\silkriver\xiadan"
    assert payload["diagnostics"]["dll_dir"] == r"C:\silkriver\dlls"


def test_tcdll_connect_ready_probe_has_structured_result(monkeypatch, capsys):
    monkeypatch.setattr(cli, "TcDllBroker", CliBroker)
    monkeypatch.setattr(sys, "argv", ["test_tcdll_connect.py", "--probe", "ready"])

    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["probe"] == "ready"
    assert payload["result"] == {"ok": True, "status": "READY", "raw": "READY", "probe": "ready"}
