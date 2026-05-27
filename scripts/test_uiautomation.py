#!/usr/bin/env python
"""Test UI Automation API for background stock code entry."""

import ctypes
import ctypes.wintypes
import time
from comtypes.client import CreateObject
from comtypes import GUID, COMMETHOD, HRESULT, IUnknown

# COM interfaces for UI Automation
# We'll use the simpler approach via the UIAutomationCore COM object

import comtypes.gen.UIAutomationClient as UIA

user32 = ctypes.windll.user32


def find_main_window():
    results = []
    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def enum_cb(hwnd, lParam):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if "网上股票交易系统5.0" in buf.value and user32.IsWindowVisible(hwnd):
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


def main():
    hwnd = find_main_window()
    if not hwnd:
        print("xiadan not found")
        return

    print(f"Main window: {hwnd:#x}")

    # Create UI Automation
    uia = CreateObject("{ff48dba4-60ef-4201-aa87-54103eef594e}")  # CUIAutomation
    uia = uia.QueryInterface(UIA.IUIAutomation)

    # Get element from window handle
    root = uia.ElementFromHandle(hwnd)
    print(f"Root: {root.CurrentName}")

    # Find the stock code Edit control
    # Create property conditions
    cond_edit = uia.CreatePropertyCondition(UIA.UIA_ControlTypePropertyId, UIA.UIA_EditControlTypeId)
    print("Created Edit control condition")

    # Find all edit controls
    edits = root.FindAll(UIA.TreeScope_Descendants, cond_edit)
    print(f"Found {edits.Length} edit controls")

    for i in range(edits.Length):
        elem = edits.GetElement(i)
        name = elem.CurrentName
        automation_id = elem.CurrentAutomationId
        class_name = elem.CurrentClassName
        is_enabled = elem.CurrentIsEnabled

        print(f"\n  [{i}] Name=\"{name}\" AutomationId=\"{automation_id}\" Class=\"{class_name}\" Enabled={is_enabled}")

        # Try to get ValuePattern
        try:
            vp = elem.GetCurrentPattern(UIA.UIA_ValuePatternId)
            if vp:
                vp = vp.QueryInterface(UIA.IUIAutomationValuePattern)
                print(f"    ValuePattern: value=\"{vp.CurrentValue}\" readonly={vp.CurrentIsReadOnly}")
        except Exception as e:
            print(f"    No ValuePattern: {e}")

        # Get bounding rect
        rect = elem.CurrentBoundingRectangle
        print(f"    Rect: left={rect.left}, top={rect.top}, w={rect.right - rect.left}, h={rect.bottom - rect.top}")

    # Now find the stock code field specifically
    # It should be in the buy dialog area (screen coords around x=1029, y=465 from our diag)
    print("\n=== Trying to find stock code edit by position ===")
    for i in range(edits.Length):
        elem = edits.GetElement(i)
        rect = elem.CurrentBoundingRectangle
        # Stock code is at approximately (1029, 465, 84, 18)
        if 1000 < rect.left < 1060 and 440 < rect.top < 490:
            print(f"Found stock code edit at position!")
            try:
                vp = elem.GetCurrentPattern(UIA.UIA_ValuePatternId)
                if vp:
                    vp = vp.QueryInterface(UIA.IUIAutomationValuePattern)
                    if not vp.CurrentIsReadOnly:
                        print(f"Setting value to '600519'...")
                        vp.SetValue("600519")
                        time.sleep(1.0)
                        print(f"New value: \"{vp.CurrentValue}\"")

                        # Check stock name
                        time.sleep(1.5)
                        print(f"Value after lookup: \"{vp.CurrentValue}\"")
                    else:
                        print("Value is read-only!")
            except Exception as e:
                print(f"Error: {e}")
            break

    # Also check stock name static
    print("\n=== Checking stock name ===")
    cond_static = uia.CreatePropertyCondition(UIA.UIA_ControlTypePropertyId, UIA.UIA_TextControlTypeId)
    statics = root.FindAll(UIA.TreeScope_Descendants, cond_static)
    for i in range(statics.Length):
        elem = statics.GetElement(i)
        rect = elem.CurrentBoundingRectangle
        if 170 < rect.top < 210 and rect.left > 1000:
            name = elem.CurrentName
            if name and len(name) > 1:
                print(f"  Text at y={rect.top}: \"{name}\"")


if __name__ == "__main__":
    main()
