#!/usr/bin/env python
"""Buy 100 shares of 智度股份 (000676) — v6 with AttachThreadInput."""

import ctypes
import ctypes.wintypes
import time
import sys

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Constants
WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
WM_CHAR = 0x0102
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_COMMAND = 0x0111
BM_CLICK = 0x00F5
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_RETURN = 0x0D
VK_F1 = 0x70

WINDOW_TITLE = "网上股票交易系统5.0"
CID_STOCK_CODE = 1032
CID_BUY_PRICE = 1033
CID_BUY_QTY = 1034
CID_BUY_BTN = 1006
CID_RESET_BTN = 1007
CID_STOCK_NAME = 1036


class KI(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_uint), ("time", ctypes.c_uint),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _U(ctypes.Union):
    _fields_ = [("ki", KI)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("u", _U)]


def make_key_input(vk, flags=0):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.u.ki.wVk = vk
    inp.u.ki.dwFlags = flags
    return inp


def send_key(vk):
    user32.SendInput(1, ctypes.byref(make_key_input(vk)), ctypes.sizeof(INPUT))
    time.sleep(0.05)
    user32.SendInput(1, ctypes.byref(make_key_input(vk, KEYEVENTF_KEYUP)), ctypes.sizeof(INPUT))
    time.sleep(0.05)


def send_text(text):
    for ch in text:
        vk = user32.VkKeyScanW(ord(ch)) & 0xFF
        send_key(vk)


def send_ctrl_a():
    VK_CONTROL = 0x11
    VK_A = 0x41
    user32.SendInput(1, ctypes.byref(make_key_input(VK_CONTROL)), ctypes.sizeof(INPUT))
    send_key(VK_A)
    user32.SendInput(1, ctypes.byref(make_key_input(VK_CONTROL, KEYEVENTF_KEYUP)), ctypes.sizeof(INPUT))
    time.sleep(0.05)


def attach_thread_input(target_hwnd):
    """Attach our thread's input to the target window's thread."""
    our_tid = kernel32.GetCurrentThreadId()
    target_tid = user32.GetWindowThreadProcessId(target_hwnd, None)
    if our_tid != target_tid:
        result = user32.AttachThreadInput(our_tid, target_tid, True)
        return result
    return True


def detach_thread_input(target_hwnd):
    our_tid = kernel32.GetCurrentThreadId()
    target_tid = user32.GetWindowThreadProcessId(target_hwnd, None)
    if our_tid != target_tid:
        user32.AttachThreadInput(our_tid, target_tid, False)


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


def find_visible_buy_dialog(parent_hwnd):
    result = []
    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def enum_cb(child, lParam):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(child, buf, 256)
        if buf.value == "#32770":
            btn = user32.GetDlgItem(child, CID_BUY_BTN)
            if btn and user32.IsWindowVisible(btn):
                result.append(child)
        return True
    user32.EnumChildWindows(parent_hwnd, enum_cb, 0)
    return result[0] if result else None


def get_text(hwnd_dlg, cid):
    buf = ctypes.create_unicode_buffer(256)
    user32.GetDlgItemTextW(hwnd_dlg, cid, buf, 256)
    return buf.value


def find_static_by_position(parent_hwnd, cid, y_min, y_max):
    result = []
    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def enum_cb(child, lParam):
        ctrl_id = user32.GetDlgCtrlID(child)
        if ctrl_id == cid:
            buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(child, buf, 256)
            if buf.value == "Static":
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(child, ctypes.byref(rect))
                if y_min <= rect.top <= y_max:
                    length = user32.SendMessageW(child, WM_GETTEXTLENGTH, 0, 0)
                    if length > 0:
                        b = ctypes.create_unicode_buffer(length + 1)
                        user32.SendMessageW(child, WM_GETTEXT, length + 1, b)
                        result.append(b.value)
        return True
    user32.EnumChildWindows(parent_hwnd, enum_cb, 0)
    return result[0] if result else ""


def focus_and_type(hwnd_main, hwnd_edit, text):
    """Focus edit via AttachThreadInput + SetFocus, then type via SendInput."""
    # Attach input to xiadan's thread
    attach_thread_input(hwnd_main)

    # Bring window to foreground
    user32.SetForegroundWindow(hwnd_main)
    time.sleep(0.1)

    # Now SetFocus should work
    user32.SetFocus(hwnd_edit)
    time.sleep(0.2)

    # Detach
    detach_thread_input(hwnd_main)

    # Select all + delete
    send_ctrl_a()
    send_key(0x2E)  # VK_DELETE
    time.sleep(0.05)

    # Type
    send_text(text)
    time.sleep(0.2)


def type_via_wmchar(hwnd_edit, text):
    """Type text via WM_CHAR directly to the edit control."""
    VK_CONTROL = 0x11
    VK_A = 0x41
    VK_DELETE = 0x2E
    # Clear: Ctrl+A, Delete
    user32.SendMessageW(hwnd_edit, WM_KEYDOWN, VK_CONTROL, 0)
    user32.SendMessageW(hwnd_edit, WM_KEYDOWN, VK_A, 0)
    user32.SendMessageW(hwnd_edit, WM_KEYUP, VK_A, 0)
    user32.SendMessageW(hwnd_edit, WM_KEYUP, VK_CONTROL, 0)
    time.sleep(0.05)
    user32.SendMessageW(hwnd_edit, WM_KEYDOWN, VK_DELETE, 0)
    user32.SendMessageW(hwnd_edit, WM_KEYUP, VK_DELETE, 0)
    time.sleep(0.05)
    for ch in text:
        user32.SendMessageW(hwnd_edit, WM_CHAR, ord(ch), 0)
        time.sleep(0.03)


def close_popups():
    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def cb(hwnd2, lParam):
        if user32.IsWindowVisible(hwnd2):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd2, buf, 256)
            if buf.value == "#32770":
                for cid in [2, 1]:
                    btn = user32.GetDlgItem(hwnd2, cid)
                    if btn:
                        t = ctypes.create_unicode_buffer(256)
                        user32.GetWindowTextW(btn, t, 256)
                        if "否" in t.value or "确定" in t.value:
                            user32.PostMessageW(btn, BM_CLICK, 0, 0)
        return True
    user32.EnumWindows(cb, 0)
    time.sleep(0.5)


def click_physical(hwnd_main, hwnd_btn):
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd_btn, ctypes.byref(rect))
    cx = (rect.left + rect.right) // 2
    cy = (rect.top + rect.bottom) // 2
    user32.SetForegroundWindow(hwnd_main)
    time.sleep(0.1)
    user32.SetCursorPos(cx, cy)
    time.sleep(0.1)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)


def handle_confirm():
    found = []
    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def cb(hwnd2, lParam):
        if user32.IsWindowVisible(hwnd2):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd2, buf, 256)
            if buf.value == "#32770":
                texts = []
                @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
                def gt(ch, lp):
                    length = user32.SendMessageW(ch, 0x000E, 0, 0)
                    if 0 < length < 1000:
                        b = ctypes.create_unicode_buffer(length + 1)
                        user32.SendMessageW(ch, 0x000D, length + 1, b)
                        texts.append(b.value)
                    return True
                user32.EnumChildWindows(hwnd2, gt, 0)
                if any("委托确认" in t or "提示" in t for t in texts):
                    found.append((hwnd2, texts))
        return True
    user32.EnumWindows(cb, 0)

    for popup_hwnd, texts in found:
        for t in texts:
            if len(t) > 5:
                for line in t.split("\n"):
                    line = line.strip()
                    if line and ("买入" in line or "数量" in line or "价格" in line or "代码" in line):
                        print(f"  {line}")
        # Click 是 (IDYES = 6)
        yes_btn = user32.GetDlgItem(popup_hwnd, 6)
        if yes_btn:
            user32.PostMessageW(yes_btn, BM_CLICK, 0, 0)
            print("  Clicked 是")
            return True
    return False


def main():
    hwnd = find_main_window()
    if not hwnd:
        print("ERROR: xiadan window not found")
        sys.exit(1)

    print(f"Main window: {hwnd:#x}")
    close_popups()

    user32.ShowWindow(hwnd, 9)
    time.sleep(0.3)

    # Attach thread input for focus operations
    attach_thread_input(hwnd)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)

    # F1 for buy tab
    print("F1 -> buy tab")
    send_key(VK_F1)
    time.sleep(0.5)
    detach_thread_input(hwnd)

    hwnd_dlg = find_visible_buy_dialog(hwnd)
    if not hwnd_dlg:
        print("ERROR: Buy dialog not visible")
        sys.exit(1)

    hwnd_code = user32.GetDlgItem(hwnd_dlg, CID_STOCK_CODE)
    hwnd_price = user32.GetDlgItem(hwnd_dlg, CID_BUY_PRICE)
    hwnd_qty = user32.GetDlgItem(hwnd_dlg, CID_BUY_QTY)
    hwnd_btn = user32.GetDlgItem(hwnd_dlg, CID_BUY_BTN)
    hwnd_reset = user32.GetDlgItem(hwnd_dlg, CID_RESET_BTN)

    # Reset form
    print("Reset...")
    click_physical(hwnd, hwnd_reset)
    time.sleep(0.5)

    # === SET STOCK CODE ===
    print("Setting code...")
    # Method: AttachThreadInput + SetFocus + SendInput
    attach_thread_input(hwnd)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.1)
    user32.SetFocus(hwnd_code)
    time.sleep(0.2)
    detach_thread_input(hwnd)

    # Verify focus
    focused = user32.GetFocus()
    print(f"  Focused hwnd: {focused:#x}, target: {hwnd_code:#x}, match: {focused == hwnd_code}")

    # Type code
    send_ctrl_a()
    send_key(0x2E)  # Delete
    time.sleep(0.05)
    send_text("000676")
    time.sleep(0.1)

    code = get_text(hwnd_dlg, CID_STOCK_CODE)
    print(f"  After type: code=\"{code}\"")

    # If SendInput didn't work, fallback to WM_CHAR
    if code != "000676":
        print("  SendInput failed, trying WM_CHAR fallback...")
        type_via_wmchar(hwnd_code, "000676")
        code = get_text(hwnd_dlg, CID_STOCK_CODE)
        print(f"  After WM_CHAR: code=\"{code}\"")

    # Press Enter to trigger lookup
    attach_thread_input(hwnd)
    user32.SetFocus(hwnd_code)
    time.sleep(0.1)
    detach_thread_input(hwnd)
    send_key(VK_RETURN)
    time.sleep(1.5)

    code = get_text(hwnd_dlg, CID_STOCK_CODE)
    name = find_static_by_position(hwnd, CID_STOCK_NAME, 170, 210)
    print(f"  Result: code={code}, name={name}")

    # === SET PRICE ===
    print("Setting price...")
    attach_thread_input(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.SetFocus(hwnd_price)
    time.sleep(0.2)
    detach_thread_input(hwnd)

    send_ctrl_a()
    send_key(0x2E)
    time.sleep(0.05)
    send_text("6.110")
    time.sleep(0.1)
    print(f"  Price: \"{get_text(hwnd_dlg, CID_BUY_PRICE)}\"")

    # === SET QTY ===
    print("Setting qty...")
    attach_thread_input(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.SetFocus(hwnd_qty)
    time.sleep(0.2)
    detach_thread_input(hwnd)

    send_ctrl_a()
    send_key(0x2E)
    time.sleep(0.05)
    send_text("100")
    time.sleep(0.1)

    qty = get_text(hwnd_dlg, CID_BUY_QTY)
    print(f"  Qty: \"{qty}\"")

    # Move focus away
    send_key(0x09)  # Tab
    time.sleep(0.2)

    # === CLICK BUY ===
    print("Clicking buy...")
    click_physical(hwnd, hwnd_btn)
    time.sleep(1.5)

    if handle_confirm():
        print("Order confirmed!")
    else:
        print("No confirmation dialog")

    time.sleep(0.5)
    print(f"Final: code=\"{get_text(hwnd_dlg, CID_STOCK_CODE)}\"")


if __name__ == "__main__":
    main()
