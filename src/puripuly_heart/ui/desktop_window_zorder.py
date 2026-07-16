from __future__ import annotations

import asyncio
import ctypes
import os
from collections.abc import Awaitable, Callable
from ctypes import wintypes
from dataclasses import dataclass
from typing import Protocol

_GWL_EXSTYLE = -20
_GW_OWNER = 4
_HWND_TOPMOST = -1
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOACTIVATE = 0x0010
_SWP_ASYNCWINDOWPOS = 0x4000
_WS_EX_TOPMOST = 0x00000008
_WS_EX_TRANSPARENT = 0x00000020


@dataclass(frozen=True, slots=True)
class WindowZOrderResult:
    applied: bool
    reason: str
    win32_error: int | None = None
    click_through_confirmed: bool = False
    topmost_style_present: bool = False


@dataclass(frozen=True, slots=True)
class WindowEnumerationResult:
    windows: tuple[int, ...]
    win32_error: int | None = None


class WindowZOrderPort(Protocol):
    def bind_process(self, pid: int) -> None: ...

    async def reassert_topmost_after_click_through(self) -> WindowZOrderResult: ...

    def close(self) -> None: ...


class Win32WindowApi(Protocol):
    def top_level_windows_for_process(self, pid: int) -> WindowEnumerationResult: ...

    def is_window(self, hwnd: int) -> bool: ...

    def process_id(self, hwnd: int) -> int | None: ...

    def extended_style(self, hwnd: int) -> int: ...

    def set_topmost_no_activate(self, hwnd: int) -> tuple[bool, int | None]: ...


class NoopWindowZOrderPort:
    def bind_process(self, pid: int) -> None:
        return None

    async def reassert_topmost_after_click_through(self) -> WindowZOrderResult:
        return WindowZOrderResult(applied=False, reason="unsupported_platform")

    def close(self) -> None:
        return None


class WindowsWindowZOrderPort:
    def __init__(
        self,
        *,
        api: Win32WindowApi | None = None,
        timeout_s: float = 0.5,
        poll_interval_s: float = 0.01,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._api = api or _CtypesWin32WindowApi()
        self._timeout_s = max(0.0, float(timeout_s))
        self._poll_interval_s = max(0.001, float(poll_interval_s))
        self._sleep = sleep
        self._pid: int | None = None
        self._binding_generation = 0
        self._closed = False

    def bind_process(self, pid: int) -> None:
        if self._closed:
            return
        self._pid = int(pid) if int(pid) > 0 else None
        self._binding_generation += 1

    async def reassert_topmost_after_click_through(self) -> WindowZOrderResult:
        pid = self._pid
        generation = self._binding_generation
        if self._closed:
            return WindowZOrderResult(applied=False, reason="closed")
        if pid is None:
            return WindowZOrderResult(applied=False, reason="process_unbound")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout_s
        hwnd: int | None = None
        fallback_hwnd: int | None = None
        click_through_confirmed = False
        ambiguous = False

        while True:
            if not self._binding_is_current(pid, generation):
                return WindowZOrderResult(applied=False, reason="binding_changed")
            enumeration = self._api.top_level_windows_for_process(pid)
            if enumeration.win32_error is not None:
                return WindowZOrderResult(
                    applied=False,
                    reason="enum_windows_failed",
                    win32_error=enumeration.win32_error,
                )
            candidates = tuple(
                candidate
                for candidate in enumeration.windows
                if self._window_belongs_to_process(candidate, pid)
            )
            transparent_candidates = tuple(
                candidate
                for candidate in candidates
                if self._api.extended_style(candidate) & _WS_EX_TRANSPARENT
            )
            ambiguous = len(transparent_candidates) > 1 or (
                not transparent_candidates and len(candidates) > 1
            )
            if len(transparent_candidates) == 1:
                hwnd = transparent_candidates[0]
                click_through_confirmed = True
                break
            fallback_hwnd = candidates[0] if len(candidates) == 1 else None
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await self._sleep(min(self._poll_interval_s, remaining))

        if hwnd is None and ambiguous:
            return WindowZOrderResult(applied=False, reason="ambiguous_window")
        hwnd = hwnd or fallback_hwnd
        if hwnd is None:
            return WindowZOrderResult(applied=False, reason="window_not_found")
        if not self._binding_is_current(pid, generation):
            return WindowZOrderResult(applied=False, reason="binding_changed")
        if not self._window_belongs_to_process(hwnd, pid):
            return WindowZOrderResult(applied=False, reason="window_changed")

        applied, win32_error = self._api.set_topmost_no_activate(hwnd)
        if not applied:
            return WindowZOrderResult(
                applied=False,
                reason="set_window_pos_failed",
                win32_error=win32_error,
                click_through_confirmed=click_through_confirmed,
            )

        deadline = loop.time() + self._timeout_s
        while True:
            if not self._binding_is_current(pid, generation):
                return WindowZOrderResult(applied=False, reason="binding_changed")
            if not self._window_belongs_to_process(hwnd, pid):
                return WindowZOrderResult(applied=False, reason="window_changed")
            if self._api.extended_style(hwnd) & _WS_EX_TOPMOST:
                return WindowZOrderResult(
                    applied=True,
                    reason="applied" if click_through_confirmed else "applied_unconfirmed",
                    click_through_confirmed=click_through_confirmed,
                    topmost_style_present=True,
                )
            remaining = deadline - loop.time()
            if remaining <= 0:
                return WindowZOrderResult(
                    applied=False,
                    reason="topmost_style_missing",
                    click_through_confirmed=click_through_confirmed,
                )
            await self._sleep(min(self._poll_interval_s, remaining))

    def close(self) -> None:
        self._closed = True
        self._pid = None
        self._binding_generation += 1

    def _binding_is_current(self, pid: int, generation: int) -> bool:
        return not self._closed and self._pid == pid and self._binding_generation == generation

    def _window_belongs_to_process(self, hwnd: int, pid: int) -> bool:
        return self._api.is_window(hwnd) and self._api.process_id(hwnd) == pid


class _CtypesWin32WindowApi:
    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._enum_proc_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )
        self._user32.EnumWindows.argtypes = [self._enum_proc_type, wintypes.LPARAM]
        self._user32.EnumWindows.restype = wintypes.BOOL
        self._user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindowVisible.restype = wintypes.BOOL
        self._user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
        self._user32.GetWindow.restype = wintypes.HWND
        self._user32.IsWindow.argtypes = [wintypes.HWND]
        self._user32.IsWindow.restype = wintypes.BOOL
        self._user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        self._user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        self._user32.SetWindowPos.restype = wintypes.BOOL

    def top_level_windows_for_process(self, pid: int) -> WindowEnumerationResult:
        windows: list[int] = []

        def collect(hwnd: int, _lparam: int) -> bool:
            if self.process_id(hwnd) != pid:
                return True
            if not self._user32.IsWindowVisible(hwnd):
                return True
            if self._user32.GetWindow(hwnd, _GW_OWNER):
                return True
            windows.append(int(hwnd))
            return True

        callback = self._enum_proc_type(collect)
        ctypes.set_last_error(0)
        if not self._user32.EnumWindows(callback, 0):
            return WindowEnumerationResult(
                windows=(),
                win32_error=ctypes.get_last_error(),
            )
        return WindowEnumerationResult(windows=tuple(windows))

    def is_window(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindow(hwnd))

    def process_id(self, hwnd: int) -> int | None:
        process_id = wintypes.DWORD()
        if not self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id)):
            return None
        return int(process_id.value)

    def extended_style(self, hwnd: int) -> int:
        return int(self._user32.GetWindowLongPtrW(hwnd, _GWL_EXSTYLE))

    def set_topmost_no_activate(self, hwnd: int) -> tuple[bool, int | None]:
        ctypes.set_last_error(0)
        applied = bool(
            self._user32.SetWindowPos(
                hwnd,
                wintypes.HWND(_HWND_TOPMOST),
                0,
                0,
                0,
                0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_ASYNCWINDOWPOS,
            )
        )
        return applied, None if applied else ctypes.get_last_error()


def create_window_z_order_port() -> WindowZOrderPort:
    if os.name != "nt":
        return NoopWindowZOrderPort()
    return WindowsWindowZOrderPort()
