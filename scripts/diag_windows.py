#!/usr/bin/env python
"""Diagnose xiadan window hierarchy — find all Edit controls and their state."""

import ctypes
import ctypes.wintypes

user32 = ctypes.windll.user32
WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
WINDOW_TITLE = "网上股票交易系统5.0"


def find_main_window():
    results = []
    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def enum_cb(hwnd, lParam):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if WINDOW_TITLE in buf.value and user32.IsWindowVisible(hwnd):
            count = [0]
            @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
            def cnt_cb(ch, lp):
                count[0] += 1
                return True
            user32.EnumChildWindows(hwnd, cnt_cb, 0)
            if count[0] > 0:
                results.append(hwnd)
        return True
    user32.EnumWindows(enum_cb, 0)
    return results[0] if results else None


def enumerate_all_edits(hwnd_main):
    """Find all Edit controls under the main window."""
    edits = []

    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def cb(hwnd, lParam):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        if buf.value in ("Edit", "TEdit"):
            cid = user32.GetDlgCtrlID(hwnd)
            visible = user32.IsWindowVisible(hwnd)
            enabled = user32.IsWindowEnabled(hwnd)

            # Try multiple text retrieval methods
            text_getdlg = ctypes.create_unicode_buffer(256)
            user32.GetDlgItemTextW(user32.GetParent(hwnd), cid, text_getdlg, 256)

            text_getwin = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, text_getwin, 256)

            wm_len = user32.SendMessageW(hwnd, WM_GETTEXTLENGTH, 0, 0)
            wm_text = ""
            if 0 < wm_len < 256:
                b = ctypes.create_unicode_buffer(wm_len + 1)
                user32.SendMessageW(hwnd, WM_GETTEXT, wm_len + 1, b)
                wm_text = b.value

            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))

            edits.append({
                "hwnd": hwnd,
                "cid": cid,
                "class": buf.value,
                "visible": visible,
                "enabled": enabled,
                "getdlg": text_getdlg.value,
                "getwin": text_getwin.value,
                "wm_len": wm_len,
                "wm_text": wm_text,
                "rect": (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top),
            })
        return True

    user32.EnumChildWindows(hwnd_main, cb, 0)
    return edits


def enumerate_dialogs(hwnd_main):
    """Find all #32770 dialog windows."""
    dialogs = []

    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def cb(hwnd, lParam):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        if buf.value == "#32770":
            visible = user32.IsWindowVisible(hwnd)
            child_count = [0]
            @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
            def cnt(ch, lp):
                child_count[0] += 1
                return True
            user32.EnumChildWindows(hwnd, cnt, 0)

            # Check for buy button
            btn = user32.GetDlgItem(hwnd, 1006)
            btn_visible = btn and user32.IsWindowVisible(btn)
            btn_text = ""
            if btn:
                b = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(btn, b, 256)
                btn_text = b.value

            dialogs.append({
                "hwnd": hwnd,
                "visible": visible,
                "children": child_count[0],
                "buy_btn_visible": btn_visible,
                "buy_btn_text": btn_text,
            })
        return True

    user32.EnumChildWindows(hwnd_main, cb, 0)
    return dialogs


def main():
    hwnd = find_main_window()
    if not hwnd:
        print("xiadan window not found")
        return

    print(f"Main window: {hwnd:#x}")
    print()

    # Find dialogs
    dialogs = enumerate_dialogs(hwnd)
    print(f"Found {len(dialogs)} #32770 dialogs:")
    for d in dialogs:
        print(f"  hwnd={d['hwnd']:#x} visible={d['visible']} children={d['children']} "
              f"buy_btn={d['buy_btn_visible']} \"{d['buy_btn_text']}\"")
    print()

    # Find all Edit controls
    edits = enumerate_all_edits(hwnd)
    print(f"Found {len(edits)} Edit controls:")
    for e in edits:
        print(f"  CID={e['cid']} hwnd={e['hwnd']:#x} visible={e['visible']} enabled={e['enabled']} "
              f"rect={e['rect']}")
        print(f"    GetDlgItemText: \"{e['getdlg']}\"")
        print(f"    GetWindowText:   \"{e['getwin']}\"")
        print(f"    WM_GETTEXT(len={e['wm_len']}): \"{e['wm_text']}\"")
        print()

    # Focus on CID 1032 specifically
    print("=== CID 1032 (Stock Code) details ===")
    for d in dialogs:
        hwnd_edit = user32.GetDlgItem(d['hwnd'], 1032)
        if hwnd_edit:
            print(f"  In dialog {d['hwnd']:#x}: hwnd={hwnd_edit:#x} visible={user32.IsWindowVisible(hwnd_edit)}")
            parent = user32.GetParent(hwnd_edit)
            print(f"  Parent: {parent:#x}")
            # Check style
            GWL_STYLE = -16
            style = user32.GetWindowLongW(hwnd_edit, GWL_STYLE)
            print(f"  Style: {style:#x} (ES_READONLY={bool(style & 0x800)})")


if __name__ == "__main__":
    main()
