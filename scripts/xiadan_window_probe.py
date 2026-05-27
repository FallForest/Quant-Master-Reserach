#!/usr/bin/env python
"""Read-only xiadan window/control probe.

This script enumerates visible xiadan windows and child controls, then emits
structured JSON with visible text, class names, control IDs, and screen
coordinates. It intentionally does not click, focus, type, submit orders, or
switch pages.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import platform
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

DEFAULT_TITLE_KEYWORDS = (
    "用户登录",
    "网上股票交易系统",
    "股票交易系统",
    "xiadan",
    "下单",
)
DEFAULT_PROCESS_KEYWORDS = ("xiadan.exe",)
INTERESTING_CLASS_TOKENS = ("static", "button", "edit", "list", "grid")
LIST_GRID_CLASS_TOKENS = ("list", "grid", "datagrid", "table", "report")

KNOWN_CONTROL_IDS = {
    1006: "order_submit_button_candidate",
    1007: "reset_button_candidate",
    1014: "market_value_candidate",
    1015: "total_assets_candidate",
    1016: "available_cash_candidate",
    1024: "latest_price_candidate",
    1032: "stock_code_edit_candidate",
    1033: "price_edit_candidate",
    1034: "quantity_edit_candidate",
    1036: "stock_name_candidate",
    1463: "account_name_candidate",
}

TEXT_HINTS = {
    "stock_name": ("名称", "证券名称", "股票名称"),
    "available_quantity": ("可买", "可卖", "可用数量", "可买数量"),
    "cash": ("资金", "余额", "可用金额", "可取", "资产", "市值"),
    "positions": ("持仓", "股份余额", "可用股份", "证券代码", "证券名称"),
    "orders": ("委托", "撤单", "成交", "申报", "合同编号"),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_windows() -> bool:
    return sys.platform.startswith("win")


def rect_to_dict(rect: ctypes.wintypes.RECT) -> dict:
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    return {
        "left": int(rect.left),
        "top": int(rect.top),
        "right": int(rect.right),
        "bottom": int(rect.bottom),
        "width": width,
        "height": height,
    }


def class_is_interesting(class_name: str) -> bool:
    lowered = class_name.lower()
    return any(token in lowered for token in INTERESTING_CLASS_TOKENS)


def class_is_list_or_grid(class_name: str) -> bool:
    lowered = class_name.lower()
    return any(token in lowered for token in LIST_GRID_CLASS_TOKENS)


def normalize_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(ch for ch in normalized if not ch.isspace())


def text_hints(text: str) -> list[str]:
    normalized_text = normalize_match_text(text)
    found = []
    for hint, tokens in TEXT_HINTS.items():
        if any(normalize_match_text(token) in normalized_text for token in tokens):
            found.append(hint)
    return found


def control_hints(class_name: str, control_id: int, text: str) -> list[str]:
    hints = []
    known = KNOWN_CONTROL_IDS.get(control_id)
    if known:
        hints.append(known)
    if class_is_list_or_grid(class_name):
        hints.append("position_or_order_list_grid_candidate")
    hints.extend(text_hints(text))
    return sorted(set(hints))


class Win32Reader:
    """Small read-only wrapper around user32/kernel32 calls."""

    def __init__(self) -> None:
        if not is_windows():
            raise RuntimeError("xiadan_window_probe.py requires Windows")
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32

    def get_window_text(self, hwnd: int, limit: int = 4096) -> str:
        length = int(self.user32.GetWindowTextLengthW(hwnd))
        size = max(2, min(limit, length + 1 if length > 0 else 512))
        buf = ctypes.create_unicode_buffer(size)
        self.user32.GetWindowTextW(hwnd, buf, size)
        return buf.value

    def get_message_text(self, hwnd: int, limit: int = 4096) -> str:
        length = int(self.user32.SendMessageW(hwnd, WM_GETTEXTLENGTH, 0, 0))
        if length <= 0:
            return ""
        size = min(limit, length + 1)
        buf = ctypes.create_unicode_buffer(size)
        self.user32.SendMessageW(hwnd, WM_GETTEXT, size, buf)
        return buf.value

    def get_best_text(self, hwnd: int) -> str:
        primary = self.get_window_text(hwnd)
        secondary = self.get_message_text(hwnd)
        if len(secondary) > len(primary):
            return secondary
        return primary

    def get_class_name(self, hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(256)
        self.user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    def get_rect(self, hwnd: int) -> dict:
        rect = ctypes.wintypes.RECT()
        self.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return rect_to_dict(rect)

    def get_control_id(self, hwnd: int) -> int:
        return int(self.user32.GetDlgCtrlID(hwnd))

    def is_visible(self, hwnd: int) -> bool:
        return bool(self.user32.IsWindowVisible(hwnd))

    def is_enabled(self, hwnd: int) -> bool:
        return bool(self.user32.IsWindowEnabled(hwnd))

    def get_process_id(self, hwnd: int) -> int:
        pid = ctypes.wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

    def process_image_path(self, pid: int) -> str:
        handle = self.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ""
        try:
            size = ctypes.wintypes.DWORD(32768)
            buf = ctypes.create_unicode_buffer(size.value)
            ok = self.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
            return buf.value if ok else ""
        finally:
            self.kernel32.CloseHandle(handle)

    def enum_top_windows(self) -> list[int]:
        hwnds: list[int] = []

        @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def callback(hwnd: int, _lparam: int) -> bool:
            hwnds.append(int(hwnd))
            return True

        self.user32.EnumWindows(callback, 0)
        return hwnds

    def enum_child_windows(self, hwnd: int) -> list[int]:
        hwnds: list[int] = []

        @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def callback(child: int, _lparam: int) -> bool:
            hwnds.append(int(child))
            return True

        self.user32.EnumChildWindows(hwnd, callback, 0)
        return hwnds


def matches_xiadan_window(
    title: str,
    process_path: str,
    title_keywords: Sequence[str],
    process_keywords: Sequence[str],
) -> bool:
    normalized_title = normalize_match_text(title)
    normalized_process = normalize_match_text(process_path)
    return any(normalize_match_text(keyword) in normalized_title for keyword in title_keywords if keyword) or any(
        normalize_match_text(keyword) in normalized_process for keyword in process_keywords if keyword
    )


def summarize_page_role(controls: Sequence[dict]) -> list[str]:
    hints = {hint for control in controls for hint in control.get("hints", [])}
    texts = normalize_match_text("\n".join(control.get("text", "") for control in controls))
    roles = []
    if {"stock_code_edit_candidate", "price_edit_candidate", "quantity_edit_candidate"} & hints:
        roles.append("buy_or_order_entry_page_candidate")
    if "position_or_order_list_grid_candidate" in hints or any(
        normalize_match_text(token) in texts for token in TEXT_HINTS["positions"]
    ):
        roles.append("positions_or_query_page_candidate")
    if any(normalize_match_text(token) in texts for token in TEXT_HINTS["orders"]):
        roles.append("orders_or_query_page_candidate")
    return sorted(set(roles))


def collect_window(reader: Win32Reader, hwnd: int, include_hidden: bool = False) -> dict:
    child_controls = []
    for child in reader.enum_child_windows(hwnd):
        visible = reader.is_visible(child)
        class_name = reader.get_class_name(child)
        text = reader.get_best_text(child)
        control_id = reader.get_control_id(child)
        if not include_hidden and not visible:
            continue
        if not class_is_interesting(class_name) and not text:
            continue
        child_controls.append(
            {
                "hwnd": f"0x{child:x}",
                "control_id": control_id,
                "known_control": KNOWN_CONTROL_IDS.get(control_id, ""),
                "class_name": class_name,
                "visible": visible,
                "enabled": reader.is_enabled(child),
                "rect": reader.get_rect(child),
                "text": text,
                "hints": control_hints(class_name, control_id, text),
            }
        )

    pid = reader.get_process_id(hwnd)
    return {
        "hwnd": f"0x{hwnd:x}",
        "title": reader.get_best_text(hwnd),
        "class_name": reader.get_class_name(hwnd),
        "visible": reader.is_visible(hwnd),
        "enabled": reader.is_enabled(hwnd),
        "rect": reader.get_rect(hwnd),
        "pid": pid,
        "process_path": reader.process_image_path(pid),
        "page_roles": summarize_page_role(child_controls),
        "controls": child_controls,
        "control_count": len(child_controls),
    }


def build_report(args: argparse.Namespace) -> dict:
    if not is_windows():
        return {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
            "safety": safety_block(),
            "ok": False,
            "error": "xiadan_window_probe.py requires Windows",
            "windows": [],
        }

    reader = Win32Reader()
    windows = []
    for hwnd in reader.enum_top_windows():
        if not reader.is_visible(hwnd):
            continue
        title = reader.get_best_text(hwnd)
        pid = reader.get_process_id(hwnd)
        process_path = reader.process_image_path(pid)
        if not matches_xiadan_window(title, process_path, args.title_keyword, args.process_keyword):
            continue
        windows.append(collect_window(reader, hwnd, include_hidden=args.include_hidden))

    return {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "safety": safety_block(),
        "ok": bool(windows),
        "error": "" if windows else "no visible xiadan window matched title/process keywords",
        "match": {
            "title_keywords": list(args.title_keyword),
            "process_keywords": list(args.process_keyword),
        },
        "windows": windows,
        "summary": summarize_report(windows),
    }


def summarize_report(windows: Sequence[dict]) -> dict:
    all_controls = [control for window in windows for control in window["controls"]]
    hints: dict[str, int] = {}
    for control in all_controls:
        for hint in control.get("hints", []):
            hints[hint] = hints.get(hint, 0) + 1
    return {
        "window_count": len(windows),
        "visible_control_count": len(all_controls),
        "hint_counts": dict(sorted(hints.items())),
        "page_roles": sorted({role for window in windows for role in window.get("page_roles", [])}),
    }


def safety_block() -> dict:
    return {
        "mode": "read-only-window-enumeration",
        "live_enabled": False,
        "click_attempted": False,
        "keyboard_attempted": False,
        "write_control_attempted": False,
        "order_attempted": False,
        "navigation_attempted": False,
        "notes": "Only reads window/control metadata and text. It never clicks, focuses, types, or submits.",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    forbidden = {"--enable-live", "--click", "--type", "--submit", "--navigate", "--write"}
    if argv and any(arg in forbidden for arg in argv):
        raise SystemExit("live/control-mutating flags are forbidden for xiadan_window_probe.py")

    parser = argparse.ArgumentParser(description="Read-only xiadan window/control probe")
    parser.add_argument("--title-keyword", action="append", default=list(DEFAULT_TITLE_KEYWORDS))
    parser.add_argument("--process-keyword", action="append", default=list(DEFAULT_PROCESS_KEYWORDS))
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden child controls in JSON")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args, unknown = parser.parse_known_args(argv)
    if any(arg in forbidden for arg in unknown):
        raise SystemExit("live/control-mutating flags are forbidden for xiadan_window_probe.py")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
