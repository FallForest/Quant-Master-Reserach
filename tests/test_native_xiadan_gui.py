import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "native_xiadan_gui.py"

spec = importlib.util.spec_from_file_location("native_xiadan_gui", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
native_xiadan_gui = importlib.util.module_from_spec(spec)
sys.modules["native_xiadan_gui"] = native_xiadan_gui
spec.loader.exec_module(native_xiadan_gui)


Rect = native_xiadan_gui.Rect
ControlSnapshot = native_xiadan_gui.ControlSnapshot
WindowSnapshot = native_xiadan_gui.WindowSnapshot


def make_control(ctrl_id: int, text: str, *, hwnd: str, enabled: bool = True, class_name: str = "Edit"):
    return ControlSnapshot(
        hwnd=hwnd,
        parent="0x100",
        ctrl_id=ctrl_id,
        class_name=class_name,
        text=text,
        visible=True,
        enabled=enabled,
        rect=Rect(0, 0, 10, 10),
    )


def make_login_window(
    *,
    title: str = native_xiadan_gui.LOGIN_TITLE_KEYWORDS[0],
    account_text: str = "",
    password_text: str = "",
    cert_text: str = "",
):
    controls = (
        make_control(1011, account_text, hwnd="0x201"),
        make_control(1012, password_text, hwnd="0x202"),
        make_control(3401, cert_text, hwnd="0x203"),
        make_control(2317, "login mode", hwnd="0x205", class_name="Static"),
        make_control(2318, "normal", hwnd="0x206", class_name="Static"),
        make_control(1237, "select", hwnd="0x207", class_name="ComboBox"),
        make_control(1006, "login", hwnd="0x204", class_name="Button"),
    )
    return WindowSnapshot(
        hwnd=0x100,
        title=title,
        class_name="AfxWnd",
        visible=True,
        enabled=True,
        rect=Rect(0, 0, 400, 300),
        pid=321,
        process_path=r"C:\Broker\xiadan.exe",
        controls=controls,
    )


def make_trading_window(*, title: str = "xiadan main"):
    controls = (
        make_control(1032, "000001", hwnd="0x301"),
        make_control(1033, "10.10", hwnd="0x302"),
        make_control(1034, "100", hwnd="0x303"),
    )
    return WindowSnapshot(
        hwnd=0x300,
        title=title,
        class_name="AfxWnd",
        visible=True,
        enabled=True,
        rect=Rect(0, 0, 800, 600),
        pid=321,
        process_path=r"C:\Broker\xiadan.exe",
        controls=controls,
    )


def test_validate_args_rejects_mutating_modes_without_confirm_fill_only():
    parser = native_xiadan_gui.build_parser()
    args = parser.parse_args(["--mode", native_xiadan_gui.MODE_FILL_LOGIN])
    with pytest.raises(SystemExit, match="confirm-fill-only"):
        native_xiadan_gui.validate_args(args)


def test_validate_args_rejects_attempt_login_without_live_confirm():
    parser = native_xiadan_gui.build_parser()
    args = parser.parse_args(["--mode", native_xiadan_gui.MODE_ATTEMPT_LOGIN, "--confirm-fill-only"])
    with pytest.raises(SystemExit, match="confirm-live-login"):
        native_xiadan_gui.validate_args(args)


def test_keyboard_input_uses_compatible_ulong_ptr_type():
    assert native_xiadan_gui.ULONG_PTR is not None
    key_input = native_xiadan_gui._keyboard_input(vk=0x41)
    assert key_input.ki.dwExtraInfo is not None


def test_summarize_window_includes_login_mode_controls():
    summary = native_xiadan_gui.summarize_window(make_login_window())
    login_fields = summary["login_fields"]
    assert login_fields["login_mode_label"]["ctrl_id"] == 2317
    assert login_fields["login_mode_current_text"]["ctrl_id"] == 2318
    assert login_fields["login_mode_selector"]["ctrl_id"] == 1237


def test_run_probe_attempt_login_reports_state_transition_and_popup(monkeypatch, tmp_path):
    before_window = make_login_window()
    after_window = make_trading_window()
    windows_by_call = [(before_window,), (after_window,)]
    set_calls = []
    click_calls = []
    keyboard_calls = []

    def supplier():
        return windows_by_call.pop(0)

    monkeypatch.setattr(native_xiadan_gui, "is_windows", lambda: True)
    monkeypatch.setattr(
        native_xiadan_gui,
        "set_text",
        lambda hwnd, text: set_calls.append((hwnd, text)) or (False if hwnd == 0x202 else True),
    )
    monkeypatch.setattr(
        native_xiadan_gui,
        "keyboard_fill_control",
        lambda hwnd, *, parent_hwnd, text: keyboard_calls.append((hwnd, parent_hwnd, text)) or True,
    )
    monkeypatch.setattr(native_xiadan_gui, "click_control", lambda hwnd: click_calls.append(hwnd) or True)
    monkeypatch.setattr(native_xiadan_gui.time, "sleep", lambda _: None)
    monkeypatch.setenv("XIADAN_LOGIN_ACCOUNT", "acct-secret-7788")
    monkeypatch.setenv("XIADAN_LOGIN_PASSWORD", "pwd-secret-8899")
    monkeypatch.setenv("XIADAN_LOGIN_CERT_PASSWORD", "cert-123")

    output_path = tmp_path / "attempt_login.json"
    result = native_xiadan_gui.run_probe(
        mode=native_xiadan_gui.MODE_ATTEMPT_LOGIN,
        confirm_fill_only=True,
        confirm_live_login=True,
        output=output_path,
        window_supplier=supplier,
        popup_supplier=lambda: [
            {
                "hwnd": "0x999",
                "title": "prompt",
                "class_name": "#32770",
                "texts": ["otp 123456", "password"],
            }
        ],
    )

    assert result.ok is True
    assert result.app_state == native_xiadan_gui.STATE_TRADING_MAIN_WINDOW
    assert click_calls == [0x204]
    assert [call[0] for call in set_calls] == [0x201, 0x202, 0x203]
    assert keyboard_calls == [(0x202, 0x100, "pwd-secret-8899")]
    login_attempt = result.details["login_attempt"]
    assert login_attempt["before_state"] == native_xiadan_gui.STATE_LOGIN_WINDOW
    assert login_attempt["after_state"] == native_xiadan_gui.STATE_TRADING_MAIN_WINDOW
    assert login_attempt["still_in_login_window"] is False
    assert login_attempt["entered_trading_main_window"] is True
    assert login_attempt["click_attempted"] is True
    assert login_attempt["click_result"] == "clicked"
    assert login_attempt["login_button"]["hwnd"] == "0x204"
    assert login_attempt["login_button"]["ctrl_id"] == 1006
    assert login_attempt["used_password_keyboard_fallback"] is True
    assert login_attempt["mutation"]["field_outcomes"]["login_account"]["method"] == "wm_settext"
    assert login_attempt["mutation"]["field_outcomes"]["login_password"]["method"] == "keyboard_fallback"
    assert login_attempt["mutation"]["field_outcomes"]["login_cert_password"]["method"] == "wm_settext"
    assert login_attempt["popup_summaries"] == [
        {
            "hwnd": "0x999",
            "title": "prompt",
            "class_name": "#32770",
            "texts": ["[redacted:10 chars]", "[redacted]"],
        }
    ]

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["details"]["login_attempt"]["after_state"] == native_xiadan_gui.STATE_TRADING_MAIN_WINDOW
    assert payload["details"]["login_attempt"]["click_attempted"] is True
    assert payload["details"]["login_attempt"]["click_result"] == "clicked"
    assert payload["details"]["login_attempt"]["login_button"]["hwnd"] == "0x204"
    assert payload["details"]["login_attempt"]["mutation"]["field_outcomes"]["login_password"]["method"] == "keyboard_fallback"
    assert payload["safety"]["clicked_login_button"] is True
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "acct-secret-7788" not in serialized
    assert "pwd-secret-8899" not in serialized
    assert "cert-123" not in serialized


def test_run_probe_attempt_login_does_not_click_when_credentials_missing(monkeypatch, tmp_path):
    before_window = make_login_window()
    click_calls = []
    keyboard_calls = []

    monkeypatch.setattr(native_xiadan_gui, "is_windows", lambda: True)
    monkeypatch.setattr(native_xiadan_gui, "set_text", lambda hwnd, text: True)
    monkeypatch.setattr(
        native_xiadan_gui,
        "keyboard_fill_control",
        lambda hwnd, *, parent_hwnd, text: keyboard_calls.append((hwnd, parent_hwnd, text)) or True,
    )
    monkeypatch.setattr(native_xiadan_gui, "click_control", lambda hwnd: click_calls.append(hwnd) or True)
    monkeypatch.setattr(native_xiadan_gui.time, "sleep", lambda _: None)
    monkeypatch.delenv("XIADAN_LOGIN_ACCOUNT", raising=False)
    monkeypatch.setenv("XIADAN_LOGIN_PASSWORD", "pwd-secret-8899")
    monkeypatch.delenv("XIADAN_LOGIN_CERT_PASSWORD", raising=False)

    result = native_xiadan_gui.run_probe(
        mode=native_xiadan_gui.MODE_ATTEMPT_LOGIN,
        confirm_fill_only=True,
        confirm_live_login=True,
        output=tmp_path / "missing_creds.json",
        window_supplier=lambda: (before_window,),
        popup_supplier=lambda: [],
    )

    assert result.ok is False
    assert result.error == "login_credentials_incomplete:login_account"
    assert result.safety["clicked_login_button"] is False
    assert result.details["login_attempt"]["click_attempted"] is False
    assert result.details["login_attempt"]["click_result"] == "blocked_missing_credentials"
    assert result.details["login_attempt"]["login_button"]["ctrl_id"] == 1006
    assert click_calls == []
    assert keyboard_calls == []


def test_fill_login_mode_never_uses_keyboard_fallback(monkeypatch, tmp_path):
    before_window = make_login_window()
    keyboard_calls = []

    monkeypatch.setattr(native_xiadan_gui, "is_windows", lambda: True)
    monkeypatch.setattr(
        native_xiadan_gui,
        "set_text",
        lambda hwnd, text: False if hwnd == 0x202 else True,
    )
    monkeypatch.setattr(
        native_xiadan_gui,
        "keyboard_fill_control",
        lambda hwnd, *, parent_hwnd, text: keyboard_calls.append((hwnd, parent_hwnd, text)) or True,
    )
    monkeypatch.setattr(native_xiadan_gui.time, "sleep", lambda _: None)
    monkeypatch.setenv("XIADAN_LOGIN_ACCOUNT", "FILL-ACCOUNT")
    monkeypatch.setenv("XIADAN_LOGIN_PASSWORD", "FILL-PASSWORD")
    monkeypatch.setenv("XIADAN_LOGIN_CERT_PASSWORD", "FILL-CERT")

    result = native_xiadan_gui.run_probe(
        mode=native_xiadan_gui.MODE_FILL_LOGIN,
        confirm_fill_only=True,
        output=tmp_path / "fill_login.json",
        window_supplier=lambda: (before_window,),
        popup_supplier=lambda: [],
    )

    assert result.ok is True
    assert result.details["mutation"]["field_outcomes"]["login_password"]["method"] == "failed"
    assert result.details["mutation"]["field_outcomes"]["login_password"]["reason"] == "wm_settext_failed"
    assert keyboard_calls == []


def test_main_stdout_and_artifact_do_not_leak_login_secrets(monkeypatch, tmp_path, capsys):
    before_window = make_login_window()
    after_window = make_login_window(
        account_text="visible-account",
        password_text="visible-password",
        cert_text="visible-cert",
    )
    windows_by_call = [(before_window,), (after_window,)]

    def supplier():
        return windows_by_call.pop(0)

    monkeypatch.setattr(native_xiadan_gui, "is_windows", lambda: True)
    monkeypatch.setattr(native_xiadan_gui, "discover_xiadan_windows", supplier)
    monkeypatch.setattr(native_xiadan_gui, "set_text", lambda hwnd, text: True)
    monkeypatch.setattr(native_xiadan_gui, "keyboard_fill_control", lambda hwnd, *, parent_hwnd, text: True)
    monkeypatch.setattr(native_xiadan_gui, "click_control", lambda hwnd: True)
    monkeypatch.setattr(native_xiadan_gui.time, "sleep", lambda _: None)
    monkeypatch.setenv("XIADAN_LOGIN_ACCOUNT", "TOPSECRET-ACCOUNT")
    monkeypatch.setenv("XIADAN_LOGIN_PASSWORD", "TOPSECRET-PASSWORD")
    monkeypatch.setenv("XIADAN_LOGIN_CERT_PASSWORD", "TOPSECRET-CERT")

    output_path = tmp_path / "stdout_payload.json"
    rc = native_xiadan_gui.main(
        [
            "--mode",
            native_xiadan_gui.MODE_ATTEMPT_LOGIN,
            "--confirm-fill-only",
            "--confirm-live-login",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    serialized_stdout = json.dumps(stdout_payload, ensure_ascii=False)
    serialized_file = json.dumps(file_payload, ensure_ascii=False)

    for secret in ("TOPSECRET-ACCOUNT", "TOPSECRET-PASSWORD", "TOPSECRET-CERT"):
        assert secret not in serialized_stdout
        assert secret not in serialized_file

    login_fields = stdout_payload["details"]["login_attempt"]["after_window_summary"]["login_fields"]
    assert login_fields["login_account"]["text"]["preview"] == "[hidden]"
    assert login_fields["login_password"]["text"]["preview"] == "[hidden]"
    assert login_fields["login_cert_password"]["text"]["preview"] == "[hidden]"
