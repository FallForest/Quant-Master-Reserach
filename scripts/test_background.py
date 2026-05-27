#!/usr/bin/env python
"""Test background text entry via EM_SETSEL + EM_REPLACESEL — no focus needed."""

import ctypes
import ctypes.wintypes
import time

user32 = ctypes.windll.user32

WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
EM_SETSEL = 0x00B1
EM_REPLACESEL = 0x00C2
WM_CHAR = 0x0102
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_RETURN = 0x0D
WINDOW_TITLE = "网上股票交易系统5.0"

CID_STOCK_CODE = 1032
CID_BUY_PRICE = 1033
CID_BUY_QTY = 1034


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


def find_buy_dialog(parent_hwnd):
    result = []
    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def enum_cb(child, lParam):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(child, buf, 256)
        if buf.value == "#32770":
            btn = user32.GetDlgItem(child, 1006)
            if btn and user32.IsWindowVisible(btn):
                result.append(child)
        return True
    user32.EnumChildWindows(parent_hwnd, enum_cb, 0)
    return result[0] if result else None


def get_edit_text(hwnd_edit):
    length = user32.SendMessageW(hwnd_edit, WM_GETTEXTLENGTH, 0, 0)
    if length > 0:
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.SendMessageW(hwnd_edit, WM_GETTEXT, length + 1, buf)
        return buf.value
    return ""


def set_edit_text_em(hwnd_edit, text):
    """Set text using EM_SETSEL + EM_REPLACESEL — works in background, triggers MFC validation."""
    # Select all
    user32.SendMessageW(hwnd_edit, EM_SETSEL, 0, -1)
    time.sleep(0.05)
    # Replace with new text (must be null-terminated)
    text_buf = ctypes.create_unicode_buffer(text)
    user32.SendMessageW(hwnd_edit, EM_REPLACESEL, 0, text_buf)
    time.sleep(0.1)


def set_edit_text_wmchar(hwnd_edit, text):
    """Set text via WM_CHAR messages — background compatible."""
    # Select all: Ctrl+A
    VK_CONTROL = 0x11
    VK_A = 0x41
    VK_DELETE = 0x2E
    user32.SendMessageW(hwnd_edit, WM_KEYDOWN, VK_CONTROL, 0)
    user32.SendMessageW(hwnd_edit, WM_CHAR, ord('a'), 0)
    user32.SendMessageW(hwnd_edit, WM_KEYUP, VK_CONTROL, 0)
    time.sleep(0.05)
    # Delete
    user32.SendMessageW(hwnd_edit, WM_KEYDOWN, VK_DELETE, 0)
    user32.SendMessageW(hwnd_edit, WM_KEYUP, VK_DELETE, 0)
    time.sleep(0.05)
    # Type characters
    for ch in text:
        user32.SendMessageW(hwnd_edit, WM_CHAR, ord(ch), 0)
        time.sleep(0.03)


def main():
    hwnd = find_main_window()
    if not hwnd:
        print("xiadan not found")
        return

    hwnd_dlg = find_buy_dialog(hwnd)
    if not hwnd_dlg:
        print("Buy dialog not found")
        return

    hwnd_code = user32.GetDlgItem(hwnd_dlg, CID_STOCK_CODE)
    hwnd_price = user32.GetDlgItem(hwnd_dlg, CID_BUY_PRICE)
    hwnd_qty = user32.GetDlgItem(hwnd_dlg, CID_BUY_QTY)

    print(f"Main: {hwnd:#x}, Dialog: {hwnd_dlg:#x}")
    print(f"Stock code edit: {hwnd_code:#x}")
    print(f"Price edit: {hwnd_price:#x}")
    print(f"Qty edit: {hwnd_qty:#x}")
    print()

    # === Test 1: EM_SETSEL + EM_REPLACESEL ===
    print("=== Test 1: EM_SETSEL + EM_REPLACESEL ===")
    set_edit_text_em(hwnd_code, "600519")
    time.sleep(0.3)
    text = get_edit_text(hwnd_code)
    print(f"  Stock code after EM: \"{text}\"")

    if text == "600519":
        print("  SUCCESS! EM_REPLACESEL worked!")
        # Trigger lookup
        user32.SendMessageW(hwnd_code, WM_KEYDOWN, VK_RETURN, 0)
        user32.SendMessageW(hwnd_code, WM_KEYUP, VK_RETURN, 0)
        time.sleep(1.5)
        text2 = get_edit_text(hwnd_code)
        print(f"  After Enter: \"{text2}\"")

        # Check stock name (CID 1036)
        def get_stock_name():
            result = []
            @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
            def cb(child, lParam):
                if user32.GetDlgCtrlID(child) == 1036:
                    buf = ctypes.create_unicode_buffer(256)
                    user32.GetClassNameW(child, buf, 256)
                    if buf.value == "Static":
                        rect = ctypes.wintypes.RECT()
                        user32.GetWindowRect(child, ctypes.byref(rect))
                        if 170 <= rect.top <= 210:
                            length = user32.SendMessageW(child, WM_GETTEXTLENGTH, 0, 0)
                            if length > 0:
                                b = ctypes.create_unicode_buffer(length + 1)
                                user32.SendMessageW(child, WM_GETTEXT, length + 1, b)
                                result.append(b.value)
                return True
            user32.EnumChildWindows(hwnd, cb, 0)
            return result[0] if result else ""

        name = get_stock_name()
        print(f"  Stock name: \"{name}\"")
    else:
        print("  FAILED — trying WM_CHAR...")
        set_edit_text_wmchar(hwnd_code, "600519")
        time.sleep(0.3)
        text = get_edit_text(hwnd_code)
        print(f"  Stock code after WM_CHAR: \"{text}\"")

        if text == "600519":
            print("  SUCCESS! WM_CHAR worked!")
            user32.SendMessageW(hwnd_code, WM_KEYDOWN, VK_RETURN, 0)
            user32.SendMessageW(hwnd_code, WM_KEYUP, VK_RETURN, 0)
            time.sleep(1.5)
        else:
            print("  Both methods FAILED")

    # === Test 2: Price field ===
    print("\n=== Test 2: Price (EM_SETSEL+EM_REPLACESEL) ===")
    set_edit_text_em(hwnd_price, "100.500")
    time.sleep(0.3)
    text = get_edit_text(hwnd_price)
    print(f"  Price: \"{text}\"")

    # === Test 3: Qty field ===
    print("\n=== Test 3: Qty (EM_SETSEL+EM_REPLACESEL) ===")
    set_edit_text_em(hwnd_qty, "100")
    time.sleep(0.3)
    text = get_edit_text(hwnd_qty)
    print(f"  Qty: \"{text}\"")


if __name__ == "__main__":
    main()
