"""
List top-level windows and read geometry for screen capture (Windows).
Uses ctypes — no extra pip dependency.
"""

from __future__ import annotations

import sys
from typing import Any

if sys.platform != "win32":
    def clip_bbox_for_mss(sct: Any, left: int, top: int, width: int, height: int) -> dict[str, int] | None:
        return None

    def list_capture_windows(exclude_hwnd: int | None = None) -> list[dict[str, Any]]:
        return []

    def get_window_rect_pixels(hwnd: int) -> tuple[int, int, int, int] | None:
        return None

    def grab_window_bgr_hwnd(hwnd: int) -> Any:
        return None

else:
    import ctypes
    import numpy as np
    import cv2
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    DIB_RGB_COLORS = 0
    SRCCOPY = 0x00CC0020
    PW_RENDERFULLCONTENT = 0x00000002
    BI_RGB = 0

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    def get_window_rect_pixels(hwnd: int) -> tuple[int, int, int, int] | None:
        """Return (left, top, width, height) in screen pixels, or None if invalid."""
        rect = RECT()
        if not user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w < 8 or h < 8:
            return None
        return (int(rect.left), int(rect.top), int(w), int(h))

    def clip_bbox_for_mss(sct: Any, left: int, top: int, width: int, height: int) -> dict[str, int] | None:
        """Intersect window rect with the virtual screen (mss monitors[0])."""
        vs = sct.monitors[0]
        vl, vt, vw, vh = vs["left"], vs["top"], vs["width"], vs["height"]
        vr, vb = vl + vw, vt + vh
        x1 = max(left, vl)
        y1 = max(top, vt)
        x2 = min(left + width, vr)
        y2 = min(top + height, vb)
        iw, ih = x2 - x1, y2 - y1
        if iw < 4 or ih < 4:
            return None
        return {"left": int(x1), "top": int(y1), "width": int(iw), "height": int(ih)}

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = (
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        )

    class BITMAPINFO(ctypes.Structure):
        _fields_ = (("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3))

    def grab_window_bgr_hwnd(hwnd: int) -> Any:
        """
        Capture a single window into a BGR image (OpenCV).
        Uses PrintWindow + GetDIBits — avoids GDI BitBlt from screen (wrong layer under DWM/GPU).
        """
        hwnd = int(hwnd)
        hwin = wintypes.HWND(hwnd)
        if not user32.IsWindow(hwin):
            return None

        rect = RECT()
        if not user32.GetWindowRect(hwin, ctypes.byref(rect)):
            return None
        w = int(rect.right - rect.left)
        h = int(rect.bottom - rect.top)
        if w < 2 or h < 2 or w > 16384 or h > 16384:
            return None

        if hasattr(user32, "PrintWindow"):
            user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
            user32.PrintWindow.restype = wintypes.BOOL

        hdc_screen = user32.GetDC(0)
        if not hdc_screen:
            return None
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        if not hbmp or not hdc_mem:
            if hbmp:
                gdi32.DeleteObject(hbmp)
            if hdc_mem:
                gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(0, hdc_screen)
            return None

        gdi32.SelectObject(hdc_mem, hbmp)

        ok = False
        if hasattr(user32, "PrintWindow"):
            ok = bool(user32.PrintWindow(hwin, hdc_mem, PW_RENDERFULLCONTENT))
            if not ok:
                ok = bool(user32.PrintWindow(hwin, hdc_mem, 0))
        if not ok:
            hdc_win = user32.GetWindowDC(hwin)
            if hdc_win:
                gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_win, 0, 0, SRCCOPY)
                user32.ReleaseDC(hwin, hdc_win)

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        buf = ctypes.create_string_buffer(w * h * 4)
        lines = gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buf, ctypes.byref(bmi), DIB_RGB_COLORS)

        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)

        if lines != h:
            return None

        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4)).copy()
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    def list_capture_windows(exclude_hwnd: int | None = None) -> list[dict[str, Any]]:
        """
        Visible top-level windows with a title, suitable for capture.
        Each dict: hwnd (int), title (str).
        """
        found: list[dict[str, Any]] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def callback(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            if user32.IsIconic(hwnd):
                return True
            if exclude_hwnd is not None and int(hwnd) == int(exclude_hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value.strip()
            if not title:
                return True
            if get_window_rect_pixels(int(hwnd)) is None:
                return True
            found.append({"hwnd": int(hwnd), "title": title})
            return True

        user32.EnumWindows(callback, 0)
        # Stable sort by title for UX
        found.sort(key=lambda x: x["title"].lower())
        return found
