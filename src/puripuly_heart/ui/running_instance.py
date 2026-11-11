from __future__ import annotations

import ctypes
from ctypes import wintypes

from puripuly_heart.config.settings_vnext.defaults import (
    detect_system_locale,
    resolve_first_run_ui_locale,
)
from puripuly_heart.ui.i18n import t_for_locale

_SW_RESTORE = 9
_MB_ICONINFORMATION = 0x40
_MB_SETFOREGROUND = 0x10000


def _user32() -> ctypes.WinDLL:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.MessageBoxW.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
    return user32


def notify_running_instance(user32: ctypes.WinDLL | None = None) -> str:
    user32 = _user32() if user32 is None else user32
    locale = resolve_first_run_ui_locale(detect_system_locale())
    title = t_for_locale(locale, "app.title")
    window = user32.FindWindowW(None, title)
    if window and user32.IsWindowVisible(window):
        if user32.IsIconic(window):
            user32.ShowWindow(window, _SW_RESTORE)
        if user32.SetForegroundWindow(window):
            return "focused"
    user32.MessageBoxW(
        None,
        t_for_locale(locale, "app.already_running"),
        title,
        _MB_ICONINFORMATION | _MB_SETFOREGROUND,
    )
    return "notified"
