import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "xiadan_window_probe.py"

spec = importlib.util.spec_from_file_location("xiadan_window_probe", PROBE_PATH)
xiadan_window_probe = importlib.util.module_from_spec(spec)
sys.modules["xiadan_window_probe"] = xiadan_window_probe
spec.loader.exec_module(xiadan_window_probe)


def test_interesting_class_filters_static_button_edit_list_grid():
    assert xiadan_window_probe.class_is_interesting("Static")
    assert xiadan_window_probe.class_is_interesting("Button")
    assert xiadan_window_probe.class_is_interesting("Edit")
    assert xiadan_window_probe.class_is_interesting("SysListView32")
    assert xiadan_window_probe.class_is_interesting("CustomGridCtrl")
    assert not xiadan_window_probe.class_is_interesting("AfxWnd42")


def test_control_hints_include_known_ids_text_and_grid_candidate():
    hints = xiadan_window_probe.control_hints("SysListView32", 1016, "可用资金")
    assert "available_cash_candidate" in hints
    assert "cash" in hints
    assert "position_or_order_list_grid_candidate" in hints


def test_window_matching_uses_title_or_process_path():
    assert xiadan_window_probe.matches_xiadan_window(
        "网上股票交易系统5.0", "", ["网上股票交易系统", "用户登录"], ["xiadan.exe"]
    )
    assert xiadan_window_probe.matches_xiadan_window(
        "用户登录", "", ["网上股票交易系统", "用户登录"], ["xiadan.exe"]
    )
    assert xiadan_window_probe.matches_xiadan_window(
        "main", r"C:\silkriver\xiadan.exe", ["网上股票交易系统", "用户登录"], ["xiadan.exe"]
    )
    assert not xiadan_window_probe.matches_xiadan_window(
        "notepad", r"C:\Windows\notepad.exe", ["网上股票交易系统", "用户登录"], ["xiadan.exe"]
    )


def test_window_matching_normalizes_width_case_and_spacing():
    assert xiadan_window_probe.matches_xiadan_window(
        " 网上股票交易系统 ５.０ ", "", ["网上股票交易系统5.0"], ["xiadan.exe"]
    )
    assert xiadan_window_probe.matches_xiadan_window(
        "用 户 登 录", "", ["用户登录"], ["xiadan.exe"]
    )
    assert xiadan_window_probe.matches_xiadan_window(
        "main", r"C:\Broker\XIADAN.EXE", ["用户登录"], ["xiadan.exe"]
    )


def test_summarize_page_role_marks_order_and_query_candidates():
    controls = [
        {"hints": ["stock_code_edit_candidate"], "text": ""},
        {"hints": ["quantity_edit_candidate"], "text": ""},
        {"hints": ["position_or_order_list_grid_candidate"], "text": "证券代码 证券名称 可用股份"},
    ]
    roles = xiadan_window_probe.summarize_page_role(controls)
    assert "buy_or_order_entry_page_candidate" in roles
    assert "positions_or_query_page_candidate" in roles


def test_forbidden_mutating_flags_are_rejected():
    for flag in ["--enable-live", "--click", "--type", "--submit", "--navigate", "--write"]:
        try:
            xiadan_window_probe.parse_args([flag])
        except SystemExit as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError(f"{flag} must be rejected")


def test_main_writes_json_output_without_live_actions(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        xiadan_window_probe,
        "build_report",
        lambda args: {
            "schema_version": 1,
            "ok": True,
            "safety": xiadan_window_probe.safety_block(),
            "windows": [],
        },
    )
    output = tmp_path / "probe.json"
    rc = xiadan_window_probe.main(["--output", str(output)])
    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 0
    assert stdout_payload["safety"]["click_attempted"] is False
    assert file_payload["safety"]["write_control_attempted"] is False


def test_build_report_uses_best_text_for_window_matching(monkeypatch):
    class FakeReader:
        def enum_top_windows(self):
            return [100]

        def is_visible(self, hwnd):
            return True

        def get_window_text(self, hwnd):
            return ""

        def get_best_text(self, hwnd):
            return "用户登录"

        def get_process_id(self, hwnd):
            return 123

        def process_image_path(self, pid):
            return r"C:\Broker\helper.exe"

    monkeypatch.setattr(xiadan_window_probe, "is_windows", lambda: True)
    monkeypatch.setattr(xiadan_window_probe, "Win32Reader", FakeReader)
    monkeypatch.setattr(
        xiadan_window_probe,
        "collect_window",
        lambda reader, hwnd, include_hidden=False: {"hwnd": f"0x{hwnd:x}", "controls": [], "page_roles": []},
    )
    args = xiadan_window_probe.parse_args([])
    report = xiadan_window_probe.build_report(args)
    assert report["ok"] is True
    assert report["windows"] == [{"hwnd": "0x64", "controls": [], "page_roles": []}]
