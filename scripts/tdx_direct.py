#!/usr/bin/env python
"""TDX Direct — Programmatic trading via Win32 API (no easytrader).

Directly controls xiadan.exe trading window using Windows API.
No GUI framework, no CAPTCHA service, no external dependencies.

Architecture:
    xiadan.exe (32-bit) ──Win32 API──> TDXDirect (64-bit Python works fine)

Usage:
    from scripts.tdx_direct import TDXDirect
    tdx = TDXDirect()
    tdx.connect()

    # Query account
    print(tdx.get_balance())       # 5720.79
    print(tdx.get_account_name())  # 钟晖卓

    # Place buy order
    tdx.switch_to_buy_tab()
    tdx.set_buy("000676", 6.11, 100)
    tdx.click_buy()  # SUBMITS THE ORDER
"""

import ctypes
import ctypes.wintypes
import time

user32 = ctypes.windll.user32

# Win32 constants
WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
BM_CLICK = 0x00F5
VK_F1 = 0x70
VK_CONTROL = 0x11
VK_A = 0x41
VK_DELETE = 0x2E
VK_RETURN = 0x0D
VK_TAB = 0x09
KEYEVENTF_KEYUP = 0x0002

# Control IDs (CIDs)
CID_STOCK_CODE = 1032
CID_STOCK_NAME = 1036
CID_BUY_PRICE = 1033
CID_BUY_QTY = 1034
CID_BUY_BTN = 1006
CID_RESET_BTN = 1007
CID_AVAILABLE = 1016
CID_MARKET_VAL = 1014
CID_TOTAL_ASSETS = 1015
CID_ACCOUNT_NAME = 1463
CID_LATEST_PRICE = 1024

WINDOW_TITLE = "网上股票交易系统5.0"


# --- Low-level helpers ---

def _keybd_press(vk):
    """Press and release a key via keybd_event."""
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.02)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    time.sleep(0.02)


def _keybd_type(text):
    """Type text character by character via keybd_event."""
    for ch in str(text):
        vk = user32.VkKeyScanW(ord(ch)) & 0xFF
        _keybd_press(vk)


def _keybd_ctrl_a():
    """Ctrl+A to select all."""
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    _keybd_press(VK_A)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)


def _keybd_delete():
    """Press Delete key."""
    _keybd_press(VK_DELETE)


def _click_screen(hwnd_main, hwnd_ctrl):
    """Click on a control by moving cursor to its center."""
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd_ctrl, ctypes.byref(rect))
    cx = (rect.left + rect.right) // 2
    cy = (rect.top + rect.bottom) // 2
    user32.SetForegroundWindow(hwnd_main)
    time.sleep(0.1)
    user32.SetCursorPos(cx, cy)
    time.sleep(0.1)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.02)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
    time.sleep(0.3)


def _get_text(hwnd_dlg, cid):
    """Get text from a dialog control."""
    buf = ctypes.create_unicode_buffer(256)
    user32.GetDlgItemTextW(hwnd_dlg, cid, buf, 256)
    return buf.value


def _find_main_window():
    """Find the visible xiadan trading window."""
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


def _find_visible_buy_dialog(parent_hwnd):
    """Find the #32770 dialog where the buy button is currently visible."""
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


def _find_static_by_position(parent_hwnd, cid, y_min, y_max):
    """Find a Static control by CID and approximate Y position."""
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


def _close_popups():
    """Close any open confirmation/error popup dialogs."""
    closed = []

    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def cb(hwnd, lParam):
        if user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, buf, 256)
            if buf.value == "#32770":
                for cid in [2, 1]:  # IDCANCEL, IDOK
                    btn = user32.GetDlgItem(hwnd, cid)
                    if btn:
                        t = ctypes.create_unicode_buffer(256)
                        user32.GetWindowTextW(btn, t, 256)
                        if "否" in t.value or "确定" in t.value:
                            user32.PostMessageW(btn, BM_CLICK, 0, 0)
                            closed.append(t.value)
        return True

    user32.EnumWindows(cb, 0)
    time.sleep(0.3)
    return closed


class TDXDirect:
    """Direct programmatic trading via Win32 API on xiadan window."""

    def __init__(self):
        self.hwnd = None      # main visible window
        self.hwnd_dlg = None  # dialog containing buy/sell form

    def connect(self):
        """Connect to running xiadan trading window."""
        _close_popups()
        self.hwnd = _find_main_window()
        if not self.hwnd:
            raise RuntimeError("xiadan window not found. Is xiadan.exe running and logged in?")

        user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE
        time.sleep(0.3)
        return True

    def switch_to_buy_tab(self):
        """Press F1 to switch to the buy tab."""
        _close_popups()
        user32.SetForegroundWindow(self.hwnd)
        time.sleep(0.2)
        _keybd_press(VK_F1)
        time.sleep(0.5)

        self.hwnd_dlg = _find_visible_buy_dialog(self.hwnd)
        if not self.hwnd_dlg:
            raise RuntimeError("Buy tab not visible after F1")
        return True

    def _ensure_buy_tab(self):
        """Ensure we're on the buy tab."""
        if not self.hwnd_dlg:
            self.switch_to_buy_tab()
        elif not user32.IsWindowVisible(user32.GetDlgItem(self.hwnd_dlg, CID_BUY_BTN)):
            self.switch_to_buy_tab()

    # --- Account queries ---

    def get_account_name(self):
        return _find_static_by_position(self.hwnd, CID_ACCOUNT_NAME, 0, 100)

    def get_balance(self):
        for text in self._get_statics(CID_AVAILABLE):
            try:
                return float(text.replace(",", ""))
            except (ValueError, TypeError):
                continue
        return 0.0

    def get_market_value(self):
        for text in self._get_statics(CID_MARKET_VAL):
            try:
                return float(text.replace(",", ""))
            except (ValueError, TypeError):
                continue
        return 0.0

    def get_total_assets(self):
        for text in self._get_statics(CID_TOTAL_ASSETS):
            try:
                return float(text.replace(",", ""))
            except (ValueError, TypeError):
                continue
        return 0.0

    # --- Stock lookup ---

    def set_stock_code(self, code):
        """Set stock code and trigger lookup."""
        self._ensure_buy_tab()
        hwnd_code = user32.GetDlgItem(self.hwnd_dlg, CID_STOCK_CODE)

        _click_screen(self.hwnd, hwnd_code)
        _keybd_ctrl_a()
        _keybd_delete()
        _keybd_type(code)
        time.sleep(0.1)
        _keybd_press(VK_RETURN)
        time.sleep(1.5)

    def get_stock_name(self):
        return _find_static_by_position(self.hwnd, CID_STOCK_NAME, 170, 210)

    def get_latest_price(self):
        text = _find_static_by_position(self.hwnd, CID_LATEST_PRICE, 220, 240)
        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    # --- Buy/Sell form operations ---

    def set_buy_price(self, price):
        """Set buy price by clicking + typing."""
        self._ensure_buy_tab()
        hwnd_price = user32.GetDlgItem(self.hwnd_dlg, CID_BUY_PRICE)
        _click_screen(self.hwnd, hwnd_price)
        _keybd_ctrl_a()
        _keybd_delete()
        _keybd_type(f"{float(price):.3f}")
        time.sleep(0.1)

    def set_buy_qty(self, qty):
        """Set buy quantity by clicking + typing."""
        self._ensure_buy_tab()
        hwnd_qty = user32.GetDlgItem(self.hwnd_dlg, CID_BUY_QTY)
        _click_screen(self.hwnd, hwnd_qty)
        _keybd_ctrl_a()
        _keybd_delete()
        _keybd_type(str(int(qty)))
        time.sleep(0.1)
        # Tab away to trigger validation
        _keybd_press(VK_TAB)
        time.sleep(0.2)

    def set_buy(self, code, price, qty):
        """Set up a complete buy order."""
        self.set_stock_code(code)
        self.set_buy_price(price)
        self.set_buy_qty(qty)

    def click_buy(self):
        """Click Buy button and confirm. SUBMITS THE ORDER!"""
        self._ensure_buy_tab()
        hwnd_btn = user32.GetDlgItem(self.hwnd_dlg, CID_BUY_BTN)
        text = _get_text(self.hwnd_dlg, CID_BUY_BTN)
        if "买" not in text:
            raise RuntimeError(f"Button '{text}' is not a Buy button")

        _click_screen(self.hwnd, hwnd_btn)
        time.sleep(1.5)
        return self._handle_confirm_dialog()

    def click_sell(self):
        """Click Sell button and confirm. SUBMITS THE ORDER!"""
        self._ensure_buy_tab()
        hwnd_btn = user32.GetDlgItem(self.hwnd_dlg, CID_BUY_BTN)
        text = _get_text(self.hwnd_dlg, CID_BUY_BTN)
        if "卖" not in text:
            raise RuntimeError(f"Button '{text}' is not a Sell button")

        _click_screen(self.hwnd, hwnd_btn)
        time.sleep(1.5)
        return self._handle_confirm_dialog()

    def reset_form(self):
        """Clear the buy/sell form via physical mouse click."""
        self._ensure_buy_tab()
        hwnd_reset = user32.GetDlgItem(self.hwnd_dlg, CID_RESET_BTN)
        _click_screen(self.hwnd, hwnd_reset)
        time.sleep(0.5)

    # --- Cancel orders ---

    def cancel_all(self):
        hwnd_btn = user32.GetDlgItem(self.hwnd_dlg, 30001)
        if hwnd_btn:
            _click_screen(self.hwnd, hwnd_btn)
            time.sleep(0.5)
            self._handle_confirm_dialog()

    # --- Form state ---

    def get_form_state(self):
        return {
            "stock_code": _get_text(self.hwnd_dlg, CID_STOCK_CODE),
            "stock_name": self.get_stock_name(),
            "buy_price": _get_text(self.hwnd_dlg, CID_BUY_PRICE),
            "buy_qty": _get_text(self.hwnd_dlg, CID_BUY_QTY),
            "available": self.get_balance(),
            "latest_price": self.get_latest_price(),
        }

    # --- Internal helpers ---

    def _get_statics(self, cid):
        results = []

        @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def cb(child, lParam):
            if user32.GetDlgCtrlID(child) == cid:
                buf = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(child, buf, 256)
                if buf.value == "Static":
                    length = user32.SendMessageW(child, WM_GETTEXTLENGTH, 0, 0)
                    if 0 < length < 100:
                        b = ctypes.create_unicode_buffer(length + 1)
                        user32.SendMessageW(child, WM_GETTEXT, length + 1, b)
                        results.append(b.value)
            return True

        user32.EnumChildWindows(self.hwnd, cb, 0)
        return results

    def _handle_confirm_dialog(self):
        """Find confirmation dialog, print info, click 是(Yes)."""
        found = []

        @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def cb(hwnd, lParam):
            if user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(hwnd, buf, 256)
                if buf.value == "#32770":
                    texts = []

                    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
                    def gt(ch, lp):
                        length = user32.SendMessageW(ch, WM_GETTEXTLENGTH, 0, 0)
                        if 0 < length < 1000:
                            b = ctypes.create_unicode_buffer(length + 1)
                            user32.SendMessageW(ch, WM_GETTEXT, length + 1, b)
                            texts.append(b.value)
                        return True

                    user32.EnumChildWindows(hwnd, gt, 0)
                    if any("委托确认" in t for t in texts):
                        found.append((hwnd, texts))
            return True

        user32.EnumWindows(cb, 0)

        for popup_hwnd, texts in found:
            info = {}
            for t in texts:
                for line in t.split("\n"):
                    line = line.strip()
                    if "代码" in line:
                        info["code"] = line
                    elif "价格" in line:
                        info["price"] = line
                    elif "数量" in line:
                        info["qty"] = line

            # Click 是 (IDYES = 6)
            yes_btn = user32.GetDlgItem(popup_hwnd, 6)
            if yes_btn:
                user32.PostMessageW(yes_btn, BM_CLICK, 0, 0)
                return {"confirmed": True, "info": info}

        return {"confirmed": False}


def main():
    tdx = TDXDirect()
    tdx.connect()
    print(f"Connected to xiadan")
    print(f"Account: {tdx.get_account_name()}")
    print(f"Available: {tdx.get_balance()}")

    tdx.switch_to_buy_tab()
    print("\nForm state:")
    for k, v in tdx.get_form_state().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
