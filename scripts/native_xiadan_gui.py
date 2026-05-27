#!/usr/bin/env python
"""Safe native xiadan GUI state probe.

The default mode is read-only and distinguishes at least these states:
`not_started`, `login_window`, and `trading_main_window`.

Explicit fill modes may populate visible controls for inspection, but this
script never clicks the login button, never clicks the buy button, and never
confirms an order dialog.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Sequence


BM_CLICK = 0x00F5
WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
WM_SETTEXT = 0x000C
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
SW_RESTORE = 9
VK_BACK = 0x08
VK_CONTROL = 0x11
VK_KEY_A = 0x41
VK_DELETE = 0x2E
VK_SHIFT = 0x10
VK_MENU = 0x12

DEFAULT_TITLE_KEYWORDS = ("网上股票交易系统", "xiadan", "下单", "交易", "用户登录")
DEFAULT_PROCESS_KEYWORDS = ("xiadan.exe",)

STATE_NOT_STARTED = "not_started"
STATE_LOGIN_WINDOW = "login_window"
STATE_TRADING_MAIN_WINDOW = "trading_main_window"
STATE_UNKNOWN_XIADAN_WINDOW = "unknown_xiadan_window"

MODE_STATUS = "status"
MODE_FILL_BUY = "fill_buy"
MODE_FILL_LOGIN = "fill_login"
MODE_ATTEMPT_LOGIN = "attempt_login"

WINDOW_KIND_LOGIN = "login"
WINDOW_KIND_TRADING = "trading"
WINDOW_KIND_UNKNOWN = "unknown"

LOGIN_TITLE_KEYWORDS = ("用户登录", "登录")
LOGIN_TEXT_KEYWORDS = ("资金账号", "交易密码", "证书密码", "登录模式", "营业部")
TRADING_TEXT_KEYWORDS = ("买入", "证券代码", "股票代码", "可用资金", "可买", "委托")

CID_STOCK_CODE = 1032
CID_STOCK_NAME = 1036
CID_BUY_PRICE = 1033
CID_BUY_QTY = 1034
CID_BUY_BTN = 1006
CID_RESET_BTN = 1007
CID_AVAILABLE_CASH = 1016
CID_MARKET_VAL = 1014
CID_TOTAL_ASSETS = 1015
CID_AVAILABLE_QTY = 1018

LOGIN_ACCOUNT_IDS = (1011, 1001)
LOGIN_PASSWORD_IDS = (1012,)
LOGIN_CERT_PASSWORD_IDS = (3401,)
LOGIN_BUTTON_IDS = (1006,)
LOGIN_MODE_LABEL_IDS = (2317,)
LOGIN_MODE_VALUE_IDS = (2318,)
LOGIN_MODE_SELECTOR_IDS = (1237,)

WINDOWS = sys.platform.startswith("win")
user32 = ctypes.windll.user32 if WINDOWS else None
kernel32 = ctypes.windll.kernel32 if WINDOWS else None
ULONG_PTR = getattr(ctypes.wintypes, "ULONG_PTR", ctypes.c_size_t)


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class ControlSnapshot:
    hwnd: str
    parent: str
    ctrl_id: int
    class_name: str
    text: str
    visible: bool
    enabled: bool
    rect: Rect

    def to_public_dict(self, *, window_kind: str) -> dict[str, object]:
        role = identify_control_role(window_kind=window_kind, control_id=self.ctrl_id, text=self.text)
        sensitive = is_sensitive_role(role)
        return {
            "hwnd": self.hwnd,
            "parent": self.parent,
            "ctrl_id": self.ctrl_id,
            "class_name": self.class_name,
            "visible": self.visible,
            "enabled": self.enabled,
            "rect": self.rect.to_dict(),
            "role": role,
            "text": redact_text(self.text, sensitive=sensitive),
        }


@dataclass(frozen=True)
class WindowSnapshot:
    hwnd: int
    title: str
    class_name: str
    visible: bool
    enabled: bool
    rect: Rect
    pid: int
    process_path: str
    controls: tuple[ControlSnapshot, ...]

    @property
    def kind(self) -> str:
        return classify_window(self.title, self.controls)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "hwnd": _hwnd(self.hwnd),
            "title": self.title,
            "class_name": self.class_name,
            "visible": self.visible,
            "enabled": self.enabled,
            "rect": self.rect.to_dict(),
            "pid": self.pid,
            "process_path": self.process_path,
            "kind": self.kind,
            "controls": [control.to_public_dict(window_kind=self.kind) for control in self.controls],
        }


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    app_state: str
    route_viable: bool
    error: str
    safety: dict[str, object]
    windows: tuple[WindowSnapshot, ...]
    selected_window: WindowSnapshot | None
    details: dict[str, object]
    artifact_path: str

    def to_payload(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "app_state": self.app_state,
            "route_viable": self.route_viable,
            "error": self.error,
            "safety": self.safety,
            "selected_window": _hwnd(self.selected_window.hwnd) if self.selected_window else "",
            "details": self.details,
            "windows": [window.to_public_dict() for window in self.windows],
        }


def _hwnd(hwnd: int | None) -> str:
    return "" if not hwnd else f"0x{int(hwnd):x}"


def is_windows() -> bool:
    return WINDOWS


def rect_of(hwnd: int) -> Rect:
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return Rect(rect.left, rect.top, rect.right, rect.bottom)


def window_thread_pid(hwnd: int) -> tuple[int, int]:
    pid = ctypes.wintypes.DWORD()
    thread_id = int(user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)))
    return thread_id, int(pid.value)


def is_visible(hwnd: int) -> bool:
    return bool(user32.IsWindowVisible(hwnd))


def is_enabled(hwnd: int) -> bool:
    return bool(user32.IsWindowEnabled(hwnd))


def get_class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def get_window_text(hwnd: int, limit: int = 2048) -> str:
    length = int(user32.GetWindowTextLengthW(hwnd))
    size = max(2, min(limit, length + 1 if length > 0 else 512))
    buf = ctypes.create_unicode_buffer(size)
    user32.GetWindowTextW(hwnd, buf, size)
    return buf.value


def get_message_text(hwnd: int, limit: int = 2048) -> str:
    length = int(user32.SendMessageW(hwnd, WM_GETTEXTLENGTH, 0, 0))
    if length <= 0:
        return ""
    size = min(limit, length + 1)
    buf = ctypes.create_unicode_buffer(size)
    user32.SendMessageW(hwnd, WM_GETTEXT, size, buf)
    return buf.value


def get_best_text(hwnd: int) -> str:
    primary = get_window_text(hwnd)
    secondary = get_message_text(hwnd)
    return secondary if len(secondary) > len(primary) else primary


def get_process_path(pid: int) -> str:
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = ctypes.wintypes.DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        ok = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return buf.value if ok else ""
    finally:
        kernel32.CloseHandle(handle)


def enum_top_windows() -> list[int]:
    hwnds: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def callback(hwnd: int, _lparam: int) -> bool:
        hwnds.append(int(hwnd))
        return True

    user32.EnumWindows(callback, 0)
    return hwnds


def enum_child_windows(parent_hwnd: int) -> list[int]:
    hwnds: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def callback(hwnd: int, _lparam: int) -> bool:
        hwnds.append(int(hwnd))
        return True

    user32.EnumChildWindows(parent_hwnd, callback, 0)
    return hwnds


def matches_xiadan_window(
    title: str,
    process_path: str,
    title_keywords: Sequence[str] = DEFAULT_TITLE_KEYWORDS,
    process_keywords: Sequence[str] = DEFAULT_PROCESS_KEYWORDS,
) -> bool:
    title_lower = title.lower()
    process_lower = process_path.lower()
    return any(keyword.lower() in title_lower for keyword in title_keywords) or any(
        keyword.lower() in process_lower for keyword in process_keywords
    )


def identify_control_role(*, window_kind: str, control_id: int, text: str) -> str:
    text_lower = text.lower()
    if window_kind == WINDOW_KIND_TRADING:
        if control_id == CID_STOCK_CODE:
            return "buy_stock_code"
        if control_id == CID_BUY_PRICE:
            return "buy_price"
        if control_id == CID_BUY_QTY:
            return "buy_qty"
        if control_id == CID_BUY_BTN:
            return "buy_button"
        if control_id == CID_RESET_BTN:
            return "reset_button"
        if control_id == CID_STOCK_NAME:
            return "stock_name"
        if control_id == CID_AVAILABLE_CASH:
            return "available_cash"
        if control_id == CID_MARKET_VAL:
            return "market_value"
        if control_id == CID_TOTAL_ASSETS:
            return "total_assets"
        if control_id == CID_AVAILABLE_QTY:
            return "available_qty"
    if window_kind == WINDOW_KIND_LOGIN:
        if control_id in LOGIN_ACCOUNT_IDS:
            return "login_account"
        if control_id in LOGIN_PASSWORD_IDS:
            return "login_password"
        if control_id in LOGIN_CERT_PASSWORD_IDS:
            return "login_cert_password"
        if control_id in LOGIN_MODE_LABEL_IDS:
            return "login_mode_label"
        if control_id in LOGIN_MODE_VALUE_IDS:
            return "login_mode_current_text"
        if control_id in LOGIN_MODE_SELECTOR_IDS:
            return "login_mode_selector"
        if control_id in LOGIN_BUTTON_IDS:
            return "login_button"
        if "登录模式" in text or "login mode" in text_lower:
            return "login_mode"
    if "密码" in text:
        return "password_label"
    if "账号" in text:
        return "account_label"
    return "generic"


def is_sensitive_role(role: str) -> bool:
    return role in {"login_account", "login_password", "login_cert_password"}


def redact_text(text: str, *, sensitive: bool) -> object:
    if not sensitive:
        return text
    if not text:
        return {"masked": True, "empty": True, "length": 0, "preview": "[hidden]"}
    return {"masked": True, "empty": False, "length": len(text), "preview": "[hidden]"}


def sanitize_public_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    if any(token in stripped for token in ("密码", "账号", "证书")) or "password" in lowered or "account" in lowered:
        return "[redacted]"
    digit_count = sum(ch.isdigit() for ch in stripped)
    if digit_count >= 6:
        return f"[redacted:{len(stripped)} chars]"
    return stripped


def collect_window(hwnd: int) -> WindowSnapshot:
    _thread_id, pid = window_thread_pid(hwnd)
    controls: list[ControlSnapshot] = []
    for child in enum_child_windows(hwnd):
        if not is_visible(child):
            continue
        controls.append(
            snapshot_control(child, parent_hwnd=hwnd),
        )
    return WindowSnapshot(
        hwnd=hwnd,
        title=get_window_text(hwnd),
        class_name=get_class_name(hwnd),
        visible=is_visible(hwnd),
        enabled=is_enabled(hwnd),
        rect=rect_of(hwnd),
        pid=pid,
        process_path=get_process_path(pid),
        controls=tuple(controls),
    )


def snapshot_control(hwnd: int, *, parent_hwnd: int) -> ControlSnapshot:
    return ControlSnapshot(
        hwnd=_hwnd(hwnd),
        parent=_hwnd(parent_hwnd),
        ctrl_id=int(user32.GetDlgCtrlID(hwnd)),
        class_name=get_class_name(hwnd),
        text=get_best_text(hwnd),
        visible=is_visible(hwnd),
        enabled=is_enabled(hwnd),
        rect=rect_of(hwnd),
    )


def classify_window(title: str, controls: Sequence[ControlSnapshot]) -> str:
    title_lower = title.lower()
    control_ids = {control.ctrl_id for control in controls}
    texts = "\n".join(control.text for control in controls)
    if any(keyword in title for keyword in LOGIN_TITLE_KEYWORDS):
        return WINDOW_KIND_LOGIN
    login_hits = sum(1 for keyword in LOGIN_TEXT_KEYWORDS if keyword in texts)
    if control_ids.intersection(LOGIN_ACCOUNT_IDS + LOGIN_PASSWORD_IDS + LOGIN_CERT_PASSWORD_IDS) and login_hits >= 1:
        return WINDOW_KIND_LOGIN
    trading_id_hits = {CID_STOCK_CODE, CID_BUY_PRICE, CID_BUY_QTY}.intersection(control_ids)
    if len(trading_id_hits) >= 2:
        return WINDOW_KIND_TRADING
    if any(keyword in texts for keyword in TRADING_TEXT_KEYWORDS):
        return WINDOW_KIND_TRADING
    if "xiadan" in title_lower or "交易" in title or "下单" in title:
        return WINDOW_KIND_UNKNOWN
    return WINDOW_KIND_UNKNOWN


def summarize_window(window: WindowSnapshot) -> dict[str, object]:
    if window.kind == WINDOW_KIND_LOGIN:
        summary: dict[str, object] = {
            "window_kind": window.kind,
            "title": window.title,
            "login_fields": {},
        }
        for control in window.controls:
            role = identify_control_role(window_kind=window.kind, control_id=control.ctrl_id, text=control.text)
            if role in {
                "login_account",
                "login_password",
                "login_cert_password",
                "login_mode",
                "login_mode_label",
                "login_mode_current_text",
                "login_mode_selector",
                "login_button",
            }:
                summary["login_fields"][role] = {
                    "ctrl_id": control.ctrl_id,
                    "class_name": control.class_name,
                    "enabled": control.enabled,
                    "visible": control.visible,
                    "text": redact_text(control.text, sensitive=is_sensitive_role(role)),
                }
        return summary

    if window.kind == WINDOW_KIND_TRADING:
        trading_fields: dict[str, object] = {}
        for control in window.controls:
            role = identify_control_role(window_kind=window.kind, control_id=control.ctrl_id, text=control.text)
            if role in {
                "buy_stock_code",
                "buy_price",
                "buy_qty",
                "buy_button",
                "stock_name",
                "available_cash",
                "market_value",
                "total_assets",
                "available_qty",
            }:
                trading_fields[role] = {
                    "ctrl_id": control.ctrl_id,
                    "class_name": control.class_name,
                    "enabled": control.enabled,
                    "visible": control.visible,
                    "text": control.text,
                }
        return {
            "window_kind": window.kind,
            "title": window.title,
            "buy_fields": trading_fields,
        }

    return {"window_kind": window.kind, "title": window.title}


def discover_xiadan_windows(
    *,
    title_keywords: Sequence[str] = DEFAULT_TITLE_KEYWORDS,
    process_keywords: Sequence[str] = DEFAULT_PROCESS_KEYWORDS,
) -> list[WindowSnapshot]:
    if not is_windows():
        return []
    windows: list[WindowSnapshot] = []
    for hwnd in enum_top_windows():
        if not is_visible(hwnd):
            continue
        title = get_window_text(hwnd)
        _thread_id, pid = window_thread_pid(hwnd)
        process_path = get_process_path(pid)
        if not matches_xiadan_window(title, process_path, title_keywords, process_keywords):
            continue
        windows.append(collect_window(hwnd))
    return windows


def derive_app_state(windows: Sequence[WindowSnapshot]) -> tuple[str, WindowSnapshot | None]:
    for window in windows:
        if window.kind == WINDOW_KIND_TRADING:
            return STATE_TRADING_MAIN_WINDOW, window
    for window in windows:
        if window.kind == WINDOW_KIND_LOGIN:
            return STATE_LOGIN_WINDOW, window
    if windows:
        return STATE_UNKNOWN_XIADAN_WINDOW, windows[0]
    return STATE_NOT_STARTED, None


def find_control_by_role(window: WindowSnapshot, role: str) -> ControlSnapshot | None:
    for control in window.controls:
        if identify_control_role(window_kind=window.kind, control_id=control.ctrl_id, text=control.text) == role:
            return control
    return None


def set_text(hwnd: int, text: str) -> bool:
    return bool(user32.SendMessageW(hwnd, WM_SETTEXT, 0, ctypes.c_wchar_p(str(text))))


def click_control(hwnd: int) -> bool:
    return bool(user32.SendMessageW(hwnd, BM_CLICK, 0, 0))


def _keyboard_input(*, vk: int = 0, scan: int = 0, flags: int = 0) -> INPUT:
    return INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(
            wVk=vk,
            wScan=scan,
            dwFlags=flags,
            time=0,
            dwExtraInfo=ctypes.POINTER(ctypes.c_ulong)(),
        ),
    )


def _send_inputs(inputs: Sequence[INPUT]) -> bool:
    if not inputs:
        return True
    array_type = INPUT * len(inputs)
    payload = array_type(*inputs)
    sent = int(user32.SendInput(len(payload), payload, ctypes.sizeof(INPUT)))
    return sent == len(payload)


def _utf16_code_units(text: str) -> list[int]:
    data = text.encode("utf-16-le")
    return [data[index] | (data[index + 1] << 8) for index in range(0, len(data), 2)]


def _keybd_event(vk: int, *, keyup: bool = False) -> None:
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP if keyup else 0, 0)


def _tap_virtual_key(vk: int) -> None:
    _keybd_event(vk, keyup=False)
    time.sleep(0.03)
    _keybd_event(vk, keyup=True)
    time.sleep(0.03)


def _type_text_via_keybd_event(text: str) -> bool:
    for ch in text:
        vk_state = int(user32.VkKeyScanW(ord(ch)))
        if vk_state == -1:
            return False
        vk = vk_state & 0xFF
        shift_state = (vk_state >> 8) & 0xFF
        modifiers: list[int] = []
        if shift_state & 0x01:
            modifiers.append(VK_SHIFT)
        if shift_state & 0x02:
            modifiers.append(VK_CONTROL)
        if shift_state & 0x04:
            modifiers.append(VK_MENU)
        for modifier in modifiers:
            _keybd_event(modifier, keyup=False)
            time.sleep(0.02)
        _tap_virtual_key(vk)
        for modifier in reversed(modifiers):
            _keybd_event(modifier, keyup=True)
            time.sleep(0.02)
    return True


def _attach_thread_input(target_hwnd: int) -> tuple[int, int, bool]:
    current_tid = int(kernel32.GetCurrentThreadId())
    target_tid = int(user32.GetWindowThreadProcessId(target_hwnd, None))
    attached = False
    if current_tid != target_tid:
        attached = bool(user32.AttachThreadInput(current_tid, target_tid, True))
    return current_tid, target_tid, attached


def _detach_thread_input(current_tid: int, target_tid: int, attached: bool) -> None:
    if attached:
        user32.AttachThreadInput(current_tid, target_tid, False)


def keyboard_fill_control(hwnd: int, *, parent_hwnd: int, text: str) -> bool:
    if not is_windows() or not text:
        return False
    if not bool(user32.IsWindow(hwnd)):
        return False

    current_tid, target_tid, attached = _attach_thread_input(parent_hwnd or hwnd)
    try:
        if parent_hwnd and bool(user32.IsWindow(parent_hwnd)):
            user32.ShowWindow(parent_hwnd, SW_RESTORE)
            user32.SetForegroundWindow(parent_hwnd)
            user32.BringWindowToTop(parent_hwnd)
            user32.SetActiveWindow(parent_hwnd)
            time.sleep(0.15)

        user32.SetFocus(hwnd)
        time.sleep(0.1)

        _keybd_event(VK_CONTROL, keyup=False)
        time.sleep(0.02)
        _tap_virtual_key(VK_KEY_A)
        _keybd_event(VK_CONTROL, keyup=True)
        time.sleep(0.03)
        _tap_virtual_key(VK_DELETE)

        time.sleep(0.08)
        result = _type_text_via_keybd_event(text)
        time.sleep(0.12)
        return result
    finally:
        _detach_thread_input(current_tid, target_tid, attached)


def fill_buy_form(window: WindowSnapshot, *, code: str, price: str, qty: str) -> dict[str, object]:
    changed: list[dict[str, object]] = []
    for role, value in (("buy_stock_code", code), ("buy_price", price), ("buy_qty", qty)):
        control = find_control_by_role(window, role)
        if not control or not control.enabled:
            changed.append({"role": role, "changed": False, "reason": "control_not_available"})
            continue
        changed.append(
            {
                "role": role,
                "changed": set_text(int(control.hwnd, 16), value),
                "ctrl_id": control.ctrl_id,
            }
        )
        time.sleep(0.05)
    return {"mutated": any(item.get("changed") for item in changed), "changes": changed}


def _env_secret(env_name: str) -> str:
    value = os.environ.get(env_name, "")
    return value.strip()


def fill_login_form(
    window: WindowSnapshot,
    *,
    account_env: str,
    password_env: str,
    cert_password_env: str,
    allow_password_keyboard_fallback: bool = False,
    keyboard_fallback: Callable[..., bool] | None = None,
) -> dict[str, object]:
    changes: list[dict[str, object]] = []
    plan = (
        ("login_account", account_env),
        ("login_password", password_env),
        ("login_cert_password", cert_password_env),
    )
    for role, env_name in plan:
        control = find_control_by_role(window, role)
        if not control:
            changes.append({"role": role, "changed": False, "reason": "control_not_available"})
            continue
        secret = _env_secret(env_name)
        if not secret:
            changes.append(
                {
                    "role": role,
                    "changed": False,
                    "method": "failed",
                    "reason": "env_missing",
                    "env": env_name,
                }
            )
            continue
        change = {
            "role": role,
            "changed": False,
            "method": "failed",
            "ctrl_id": control.ctrl_id,
            "env": env_name,
            "value_length": len(secret),
            "attempted_methods": ["wm_settext"],
        }
        if set_text(int(control.hwnd, 16), secret):
            change["changed"] = True
            change["method"] = "wm_settext"
        elif (
            allow_password_keyboard_fallback
            and role in {"login_password", "login_cert_password"}
            and keyboard_fallback is not None
        ):
            change["attempted_methods"].append("keyboard_fallback")
            if keyboard_fallback(int(control.hwnd, 16), parent_hwnd=window.hwnd, text=secret):
                change["changed"] = True
                change["method"] = "keyboard_fallback"
            else:
                change["reason"] = "keyboard_fallback_failed"
        else:
            change["reason"] = "wm_settext_failed"
        changes.append(change)
        time.sleep(0.05)
    field_outcomes = {
        str(item.get("role")): {
            "changed": bool(item.get("changed")),
            "method": str(item.get("method", "failed")),
            "reason": str(item.get("reason", "")),
            "ctrl_id": int(item.get("ctrl_id", 0) or 0),
            "attempted_methods": list(item.get("attempted_methods", [])),
            "value_length": int(item.get("value_length", 0) or 0),
        }
        for item in changes
        if isinstance(item, dict) and item.get("role")
    }
    return {
        "mutated": any(item.get("changed") for item in changes),
        "changes": changes,
        "field_outcomes": field_outcomes,
    }


def missing_required_login_roles(mutation: dict[str, object]) -> list[str]:
    changes = mutation.get("changes", [])
    if not isinstance(changes, list):
        return ["login_account", "login_password"]
    by_role = {
        str(item.get("role")): item
        for item in changes
        if isinstance(item, dict) and item.get("role") in {"login_account", "login_password"}
    }
    missing: list[str] = []
    for role in ("login_account", "login_password"):
        change = by_role.get(role)
        if not change or not change.get("changed"):
            missing.append(role)
    return missing


def summarize_popup_window(window: WindowSnapshot) -> dict[str, object]:
    texts: list[str] = []
    for control in window.controls:
        safe_text = sanitize_public_text(control.text)
        if safe_text and safe_text not in texts:
            texts.append(safe_text)
        if len(texts) >= 5:
            break
    return {
        "hwnd": _hwnd(window.hwnd),
        "title": sanitize_public_text(window.title),
        "class_name": window.class_name,
        "texts": texts,
    }


def sanitize_popup_summary(summary: dict[str, object]) -> dict[str, object]:
    texts = summary.get("texts", [])
    safe_texts = []
    if isinstance(texts, list):
        safe_texts = [sanitize_public_text(str(text)) for text in texts if sanitize_public_text(str(text))]
    return {
        "hwnd": str(summary.get("hwnd", "")),
        "title": sanitize_public_text(str(summary.get("title", ""))),
        "class_name": str(summary.get("class_name", "")),
        "texts": safe_texts[:5],
    }


def discover_popup_summaries(*, target_pid: int | None, exclude_hwnds: Sequence[int]) -> list[dict[str, object]]:
    if not is_windows():
        return []
    exclude = {int(hwnd) for hwnd in exclude_hwnds}
    popups: list[dict[str, object]] = []
    for hwnd in enum_top_windows():
        if int(hwnd) in exclude or not is_visible(hwnd):
            continue
        _thread_id, pid = window_thread_pid(hwnd)
        if target_pid is not None and pid != target_pid:
            continue
        snapshot = collect_window(hwnd)
        if snapshot.kind in {WINDOW_KIND_LOGIN, WINDOW_KIND_TRADING}:
            continue
        if not snapshot.title and not snapshot.controls:
            continue
        popups.append(summarize_popup_window(snapshot))
    return popups


def attempt_login(
    window: WindowSnapshot,
    *,
    account_env: str,
    password_env: str,
    cert_password_env: str,
    wait_seconds: float,
    allow_password_keyboard_fallback: bool,
    window_supplier: Callable[[], Sequence[WindowSnapshot]],
    popup_supplier: Callable[[], Sequence[dict[str, object]]] | None,
) -> dict[str, object]:
    mutation = fill_login_form(
        window,
        account_env=account_env,
        password_env=password_env,
        cert_password_env=cert_password_env,
        allow_password_keyboard_fallback=allow_password_keyboard_fallback,
        keyboard_fallback=keyboard_fill_control if allow_password_keyboard_fallback else None,
    )
    login_button = find_control_by_role(window, "login_button")
    login_button_details = {
        "hwnd": str(getattr(login_button, "hwnd", "")) if login_button else "",
        "ctrl_id": int(getattr(login_button, "ctrl_id", 0) or 0) if login_button else 0,
        "enabled": bool(getattr(login_button, "enabled", False)) if login_button else False,
        "visible": bool(getattr(login_button, "visible", False)) if login_button else False,
    }
    clicked_login = False
    click_attempted = False
    click_result = "not_attempted"
    error = ""
    missing_roles = missing_required_login_roles(mutation)
    if missing_roles:
        error = f"login_credentials_incomplete:{','.join(missing_roles)}"
        click_result = "blocked_missing_credentials"
    elif not login_button or not login_button.enabled:
        error = "login_button_not_available"
        click_result = "login_button_not_available"
    else:
        click_attempted = True
        clicked_login = click_control(int(login_button.hwnd, 16))
        click_result = "clicked" if clicked_login else "click_failed"
        time.sleep(wait_seconds)

    after_windows = tuple(window_supplier())
    after_state, after_selected = derive_app_state(after_windows)
    popup_summaries = [
        sanitize_popup_summary(summary)
        for summary in list(
        popup_supplier()
        if popup_supplier is not None
        else discover_popup_summaries(
            target_pid=window.pid,
            exclude_hwnds=[candidate.hwnd for candidate in after_windows],
        )
        )
    ]
    return {
        "mutation": mutation,
        "clicked_login": clicked_login,
        "click_attempted": click_attempted,
        "click_result": click_result,
        "login_button": login_button_details,
        "wait_seconds": wait_seconds,
        "before_state": STATE_LOGIN_WINDOW,
        "after_state": after_state,
        "still_in_login_window": after_state == STATE_LOGIN_WINDOW,
        "entered_trading_main_window": after_state == STATE_TRADING_MAIN_WINDOW,
        "used_password_keyboard_fallback": any(
            isinstance(item, dict) and item.get("method") == "keyboard_fallback"
            for item in mutation.get("changes", [])
        ),
        "popup_summaries": popup_summaries,
        "after_windows": after_windows,
        "after_selected_window": after_selected,
        "before_window_summary": summarize_window(window),
        "after_window_summary": summarize_window(after_selected) if after_selected else {},
        "error": error,
    }


def serialize_login_attempt_details(login_attempt_details: dict[str, object]) -> dict[str, object]:
    if not login_attempt_details:
        return {}
    return {
        "mutation": login_attempt_details.get("mutation", {"mutated": False, "changes": []}),
        "clicked_login": bool(login_attempt_details.get("clicked_login")),
        "click_attempted": bool(login_attempt_details.get("click_attempted")),
        "click_result": str(login_attempt_details.get("click_result", "")),
        "login_button": dict(login_attempt_details.get("login_button", {})),
        "wait_seconds": float(login_attempt_details.get("wait_seconds", 0.0)),
        "before_state": str(login_attempt_details.get("before_state", "")),
        "after_state": str(login_attempt_details.get("after_state", "")),
        "still_in_login_window": bool(login_attempt_details.get("still_in_login_window")),
        "entered_trading_main_window": bool(login_attempt_details.get("entered_trading_main_window")),
        "used_password_keyboard_fallback": bool(login_attempt_details.get("used_password_keyboard_fallback")),
        "popup_summaries": list(login_attempt_details.get("popup_summaries", [])),
        "before_window_summary": dict(login_attempt_details.get("before_window_summary", {})),
        "after_window_summary": dict(login_attempt_details.get("after_window_summary", {})),
        "after_selected_window": _hwnd(getattr(login_attempt_details.get("after_selected_window"), "hwnd", None)),
        "after_window_count": len(login_attempt_details.get("after_windows", ()) or ()),
        "error": str(login_attempt_details.get("error", "")),
    }


def safety_block(
    *,
    mode: str,
    confirm_fill_only: bool,
    confirm_live_login: bool,
    login_env_names: Sequence[str],
) -> dict[str, object]:
    return {
        "mode": mode,
        "confirm_fill_only": confirm_fill_only,
        "confirm_live_login": confirm_live_login,
        "default_read_only": mode == MODE_STATUS,
        "submitted_order": False,
        "clicked_buy_button": False,
        "clicked_login_button": False,
        "confirmed_order_dialog": False,
        "navigation_attempted": False,
        "credential_output_written": False,
        "login_env_names": list(login_env_names),
        "notes": (
            "Status mode is read-only. Fill modes only write visible controls after explicit confirmation. "
            "Login attempts require an extra live-login confirmation. This script never clicks buy and never submits an order."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe native xiadan GUI state probe")
    parser.add_argument(
        "--mode",
        choices=[MODE_STATUS, MODE_FILL_BUY, MODE_FILL_LOGIN, MODE_ATTEMPT_LOGIN],
        default=MODE_STATUS,
    )
    parser.add_argument("--confirm-fill-only", action="store_true", help="Required for any mutating fill mode")
    parser.add_argument("--confirm-live-login", action="store_true", help="Required for login-button clicks")
    parser.add_argument("--login-wait-seconds", type=float, default=3.0, help="Fixed wait after login button click")
    parser.add_argument("--code", default="000676")
    parser.add_argument("--price", default="5.50")
    parser.add_argument("--qty", default="100")
    parser.add_argument("--login-account-env", default="XIADAN_LOGIN_ACCOUNT")
    parser.add_argument("--login-password-env", default="XIADAN_LOGIN_PASSWORD")
    parser.add_argument("--login-cert-password-env", default="XIADAN_LOGIN_CERT_PASSWORD")
    parser.add_argument("--output", default=str(Path("artifacts") / "native_xiadan_gui_latest.json"))
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.mode != MODE_STATUS and not args.confirm_fill_only:
        raise SystemExit("mutating fill modes require --confirm-fill-only")
    if args.mode == MODE_ATTEMPT_LOGIN and not args.confirm_live_login:
        raise SystemExit("attempt_login requires --confirm-live-login")


def write_payload(output_path: str | Path, payload: dict[str, object]) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(serialized + "\n", encoding="utf-8")
    return str(path)


def run_probe(
    *,
    mode: str = MODE_STATUS,
    confirm_fill_only: bool = False,
    confirm_live_login: bool = False,
    code: str = "000676",
    price: str = "5.50",
    qty: str = "100",
    output: str | Path = Path("artifacts") / "native_xiadan_gui_latest.json",
    login_account_env: str = "XIADAN_LOGIN_ACCOUNT",
    login_password_env: str = "XIADAN_LOGIN_PASSWORD",
    login_cert_password_env: str = "XIADAN_LOGIN_CERT_PASSWORD",
    login_wait_seconds: float = 3.0,
    window_supplier: Callable[[], Sequence[WindowSnapshot]] | None = None,
    popup_supplier: Callable[[], Sequence[dict[str, object]]] | None = None,
) -> ProbeResult:
    login_env_names = (login_account_env, login_password_env, login_cert_password_env)
    safety = safety_block(
        mode=mode,
        confirm_fill_only=confirm_fill_only,
        confirm_live_login=confirm_live_login,
        login_env_names=login_env_names,
    )

    if not is_windows():
        result = ProbeResult(
            ok=False,
            app_state=STATE_NOT_STARTED,
            route_viable=False,
            error="native_xiadan_gui.py requires Windows",
            safety=safety,
            windows=tuple(),
            selected_window=None,
            details={"host": {"platform": platform.platform(), "python": sys.version.split()[0]}},
            artifact_path="",
        )
        payload = result.to_payload()
        artifact_path = write_payload(output, payload)
        return ProbeResult(**{**result.__dict__, "artifact_path": artifact_path})

    supplier = window_supplier or discover_xiadan_windows
    windows = tuple(supplier())
    app_state, selected_window = derive_app_state(windows)
    route_viable = app_state in {STATE_LOGIN_WINDOW, STATE_TRADING_MAIN_WINDOW}
    mutation: dict[str, object] = {"mutated": False, "changes": []}
    error = ""
    login_attempt_details: dict[str, object] = {}

    if app_state == STATE_NOT_STARTED:
        error = "xiadan_not_started"
    elif app_state == STATE_UNKNOWN_XIADAN_WINDOW:
        error = "xiadan_window_state_unknown"
    elif mode != MODE_STATUS and not confirm_fill_only:
        error = "mutating_fill_requires_confirm_fill_only"
    elif mode == MODE_FILL_BUY:
        if app_state != STATE_TRADING_MAIN_WINDOW or not selected_window:
            error = "buy_fill_requires_trading_main_window"
        else:
            mutation = fill_buy_form(selected_window, code=code, price=price, qty=qty)
    elif mode == MODE_FILL_LOGIN:
        if app_state != STATE_LOGIN_WINDOW or not selected_window:
            error = "login_fill_requires_login_window"
        else:
            mutation = fill_login_form(
                selected_window,
                account_env=login_account_env,
                password_env=login_password_env,
                cert_password_env=login_cert_password_env,
            )
    elif mode == MODE_ATTEMPT_LOGIN:
        if not confirm_live_login:
            error = "attempt_login_requires_confirm_live_login"
        elif app_state != STATE_LOGIN_WINDOW or not selected_window:
            error = "attempt_login_requires_login_window"
        else:
            login_attempt_details = attempt_login(
                selected_window,
                account_env=login_account_env,
                password_env=login_password_env,
                cert_password_env=login_cert_password_env,
                wait_seconds=login_wait_seconds,
                allow_password_keyboard_fallback=confirm_fill_only and confirm_live_login,
                window_supplier=supplier,
                popup_supplier=popup_supplier,
            )
            mutation = login_attempt_details["mutation"]
            safety["clicked_login_button"] = bool(login_attempt_details["clicked_login"])
            error = str(login_attempt_details.get("error", ""))
            windows = tuple(login_attempt_details["after_windows"])
            selected_window = login_attempt_details["after_selected_window"]
            app_state = str(login_attempt_details["after_state"])
            route_viable = app_state in {STATE_LOGIN_WINDOW, STATE_TRADING_MAIN_WINDOW}

    safety["write_control_attempted"] = bool(mutation.get("changes"))
    safety["write_control_succeeded"] = bool(mutation.get("mutated"))

    details: dict[str, object] = {
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "mutation": mutation,
        "window_summary": summarize_window(selected_window) if selected_window else {},
        "requested_buy_fill": {
            "code": code,
            "price": price,
            "qty": qty,
        },
        "login_attempt": serialize_login_attempt_details(login_attempt_details),
    }

    result = ProbeResult(
        ok=app_state in {STATE_LOGIN_WINDOW, STATE_TRADING_MAIN_WINDOW} and not error,
        app_state=app_state,
        route_viable=route_viable,
        error=error,
        safety=safety,
        windows=windows,
        selected_window=selected_window,
        details=details,
        artifact_path="",
    )
    payload = result.to_payload()
    artifact_path = write_payload(output, payload)
    return ProbeResult(**{**result.__dict__, "artifact_path": artifact_path})


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    result = run_probe(
        mode=args.mode,
        confirm_fill_only=args.confirm_fill_only,
        confirm_live_login=args.confirm_live_login,
        code=args.code,
        price=args.price,
        qty=args.qty,
        output=args.output,
        login_account_env=args.login_account_env,
        login_password_env=args.login_password_env,
        login_cert_password_env=args.login_cert_password_env,
        login_wait_seconds=args.login_wait_seconds,
    )
    payload = json.dumps(result.to_payload(), ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
