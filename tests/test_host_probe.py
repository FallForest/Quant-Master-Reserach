import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOST_PROBE_PATH = ROOT / "scripts" / "host_probe.py"


spec = importlib.util.spec_from_file_location("host_probe", HOST_PROBE_PATH)
host_probe = importlib.util.module_from_spec(spec)
sys.modules["host_probe"] = host_probe
spec.loader.exec_module(host_probe)


def test_parse_process_csv_filters_xiadan_related_rows():
    text = (
        '"ProcessId","Name","ExecutablePath","CommandLine"\n'
        '"101","xiadan.exe","C:\\silkriver\\xiadan.exe","C:\\silkriver\\xiadan.exe"\n'
        '"202","notepad.exe","C:\\Windows\\notepad.exe","notepad.exe"\n'
    )
    items = host_probe.parse_process_csv(text)
    assert len(items) == 1
    assert items[0]["pid"] == 101
    assert items[0]["name"] == "xiadan.exe"


def test_parse_netstat_marks_related_listening_pid():
    text = """
  Proto  Local Address          Foreign Address        State           PID
  TCP    127.0.0.1:7708         0.0.0.0:0              LISTENING       101
  TCP    127.0.0.1:7709         127.0.0.1:51234        ESTABLISHED     101
  UDP    0.0.0.0:9900           *:*                                    303
"""
    items = host_probe.parse_netstat(text, related_pids=[101])
    assert [item["port"] for item in items] == [7708, 9900]
    assert items[0]["related_process"] is True
    assert items[1]["protocol"] == "UDP"


def test_inspect_config_root_reports_existence_without_reading_contents(tmp_path):
    (tmp_path / "xiadan.exe").write_text("dummy", encoding="utf-8")
    (tmp_path / "T0002").mkdir()
    result = host_probe.inspect_config_root(tmp_path)
    paths = {item["relative_path"]: item for item in result["key_paths"]}
    assert result["root_exists"] is True
    assert paths["xiadan.exe"]["exists"] is True
    assert paths["xiadan.exe"]["is_dir"] is False
    assert paths["T0002"]["exists"] is True
    assert paths["T0002"]["is_dir"] is True


def test_enable_live_is_rejected():
    try:
        host_probe.parse_args(["--enable-live"])
    except SystemExit as exc:
        assert "--enable-live is forbidden" in str(exc)
    else:
        raise AssertionError("--enable-live must be rejected")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        return


def test_optional_http_probe_uses_get_to_localhost_only():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        result = host_probe.probe_http_port(port, ["/ALIVE"], timeout=1.0)
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert result[0]["url"] == f"http://127.0.0.1:{port}/ALIVE"
    assert result[0]["ok"] is True
    assert result[0]["status"] == 200
    assert "body" not in result[0]


def test_main_outputs_structured_json(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        host_probe,
        "collect_processes",
        lambda: {"ok": True, "error": "", "items": [{"pid": 101, "name": "xiadan.exe"}]},
    )
    monkeypatch.setattr(
        host_probe,
        "collect_listening_ports",
        lambda related_pids=(): {"ok": True, "error": "", "items": []},
    )
    rc = host_probe.main(["--config-root", str(tmp_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["safety"]["mode"] == "scan-only"
    assert payload["safety"]["live_enabled"] is False
    assert payload["safety"]["order_attempted"] is False
    assert payload["processes"]["items"][0]["pid"] == 101
