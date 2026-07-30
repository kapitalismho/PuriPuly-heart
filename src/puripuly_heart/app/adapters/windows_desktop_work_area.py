from __future__ import annotations

import sys

from puripuly_heart.app.ports.desktop_overlay import DesktopWorkArea


class WindowsDesktopWorkAreaAdapter:
    def primary_work_area(self) -> DesktopWorkArea | None:
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            rect = wintypes.RECT()
            if not ctypes.windll.user32.SystemParametersInfoW(
                0x0030,
                0,
                ctypes.byref(rect),
                0,
            ):
                return None
            return (
                rect.left,
                rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
            )
        except Exception:
            return None
