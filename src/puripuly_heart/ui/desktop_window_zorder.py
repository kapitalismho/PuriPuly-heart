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
_WINDOW_TITLE_MAX_CHARS = 512
_WS_EX_TOPMOST = 0x00000008
_WS_EX_TRANSPARENT = 0x00000020
WINDOWS_WINDOW_VISIBILITY_TIMEOUT_S = 2.5


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


@dataclass(frozen=True, slots=True)
class WindowVisibilityConfirmation:
    confirmed: bool
    reason: str
    hwnd: int | None = None
    title_confirmed: bool = False
    visible_confirmed: bool = False
    bounds_confirmed: bool = False
    win32_error: int | None = None
    observed_bounds: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class WindowBoundsConfirmation:
    confirmed: bool
    reason: str
    hwnd: int | None = None
    title_confirmed: bool = False
    bounds_confirmed: bool = False
    win32_error: int | None = None
    observed_bounds: tuple[int, int, int, int] | None = None


class WindowZOrderPort(Protocol):
    def bind_process(self, pid: int) -> None: ...

    async def confirm_window_bounds(
        self,
        expected_title: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> WindowBoundsConfirmation: ...

    async def reassert_topmost_after_click_through(self) -> WindowZOrderResult: ...

    async def confirm_window_visible(
        self,
        expected_title: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> WindowVisibilityConfirmation: ...

    def close(self) -> None: ...


class Win32WindowApi(Protocol):
    def top_level_windows_for_process(self, pid: int) -> WindowEnumerationResult: ...

    def all_top_level_windows_for_process(self, pid: int) -> WindowEnumerationResult: ...

    def is_window(self, hwnd: int) -> bool: ...

    def is_window_visible(self, hwnd: int) -> bool: ...

    def window_title(self, hwnd: int) -> str: ...

    def window_bounds(self, hwnd: int) -> tuple[int, int, int, int] | None: ...

    def process_id(self, hwnd: int) -> int | None: ...

    def extended_style(self, hwnd: int) -> int: ...

    def set_topmost_no_activate(self, hwnd: int) -> tuple[bool, int | None]: ...


class NoopWindowZOrderPort:
    def bind_process(self, pid: int) -> None:
        return None

    async def reassert_topmost_after_click_through(self) -> WindowZOrderResult:
        return WindowZOrderResult(applied=False, reason="unsupported_platform")

    async def confirm_window_bounds(
        self,
        expected_title: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> WindowBoundsConfirmation:
        return WindowBoundsConfirmation(
            confirmed=True,
            reason="framework_authority",
            bounds_confirmed=True,
            observed_bounds=(x, y, width, height),
        )

    async def confirm_window_visible(
        self,
        expected_title: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> WindowVisibilityConfirmation:
        _ = (expected_title, x, y, width, height)
        return WindowVisibilityConfirmation(
            confirmed=True,
            reason="framework_authority",
            visible_confirmed=True,
            bounds_confirmed=True,
            observed_bounds=(x, y, width, height),
        )

    def close(self) -> None:
        return None


class WindowsWindowZOrderPort:
    def __init__(
        self,
        *,
        api: Win32WindowApi | None = None,
        timeout_s: float = 0.5,
        poll_interval_s: float = 0.01,
        bounds_retain_s: float = 0.05,
        visibility_timeout_s: float = WINDOWS_WINDOW_VISIBILITY_TIMEOUT_S,
        visibility_retain_s: float = 0.6,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._api = api or _CtypesWin32WindowApi()
        self._timeout_s = max(0.0, float(timeout_s))
        self._poll_interval_s = max(0.001, float(poll_interval_s))
        self._bounds_retain_s = max(0.0, float(bounds_retain_s))
        self._visibility_timeout_s = max(0.0, float(visibility_timeout_s))
        self._visibility_retain_s = max(0.0, float(visibility_retain_s))
        self._sleep = sleep
        self._pid: int | None = None
        self._binding_generation = 0
        self._closed = False

    async def confirm_window_bounds(
        self,
        expected_title: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> WindowBoundsConfirmation:
        pid = self._pid
        generation = self._binding_generation
        if self._closed:
            return WindowBoundsConfirmation(confirmed=False, reason="closed")
        if pid is None:
            return WindowBoundsConfirmation(confirmed=False, reason="process_unbound")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout_s
        hwnd: int | None = None
        title_confirmed = False
        while True:
            if not self._binding_is_current(pid, generation):
                return WindowBoundsConfirmation(confirmed=False, reason="binding_changed")
            enumeration = self._api.all_top_level_windows_for_process(pid)
            if enumeration.win32_error is not None:
                return WindowBoundsConfirmation(
                    confirmed=False,
                    reason="enum_windows_failed",
                    win32_error=enumeration.win32_error,
                )
            candidates = tuple(
                candidate
                for candidate in enumeration.windows
                if self._window_belongs_to_process(candidate, pid)
            )
            titled = tuple(
                candidate
                for candidate in candidates
                if self._api.window_title(candidate) == expected_title
            )
            if len(titled) == 1:
                hwnd = titled[0]
                title_confirmed = True
                break
            if len(candidates) == 1:
                hwnd = candidates[0]
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                return WindowBoundsConfirmation(
                    confirmed=False,
                    reason="ambiguous_window" if candidates else "window_not_found",
                )
            await self._sleep(min(self._poll_interval_s, remaining))

        logical_bounds = (int(x), int(y), int(width), int(height))
        target_bounds: tuple[int, int, int, int] | None = None
        observed_bounds: tuple[int, int, int, int] | None = None
        confirmed_since: float | None = None
        while True:
            if not self._binding_is_current(pid, generation):
                return WindowBoundsConfirmation(
                    confirmed=False,
                    reason="binding_changed",
                    hwnd=hwnd,
                    title_confirmed=title_confirmed,
                )
            if not self._window_belongs_to_process(hwnd, pid):
                return WindowBoundsConfirmation(
                    confirmed=False,
                    reason="window_changed",
                    hwnd=hwnd,
                    title_confirmed=title_confirmed,
                )
            now = loop.time()
            observed_bounds = self._api.window_bounds(hwnd)
            if observed_bounds is not None:
                scaled_target = _native_target_bounds(logical_bounds, observed_bounds)
                if scaled_target is not None and scaled_target != target_bounds:
                    target_bounds = scaled_target
                    confirmed_since = None
            if (
                observed_bounds is not None
                and target_bounds is not None
                and _window_bounds_close(observed_bounds, target_bounds)
            ):
                if confirmed_since is None:
                    confirmed_since = now
                if now - confirmed_since >= self._bounds_retain_s:
                    return WindowBoundsConfirmation(
                        confirmed=True,
                        reason="confirmed",
                        hwnd=hwnd,
                        title_confirmed=title_confirmed,
                        bounds_confirmed=True,
                        observed_bounds=observed_bounds,
                    )
            else:
                confirmed_since = None
            remaining = deadline - now
            if remaining <= 0:
                return WindowBoundsConfirmation(
                    confirmed=False,
                    reason="bounds_not_retained",
                    hwnd=hwnd,
                    title_confirmed=title_confirmed,
                    bounds_confirmed=confirmed_since is not None,
                    observed_bounds=observed_bounds,
                )
            await self._sleep(min(self._poll_interval_s, remaining))

    async def confirm_window_visible(
        self,
        expected_title: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> WindowVisibilityConfirmation:
        pid = self._pid
        generation = self._binding_generation
        if self._closed:
            return WindowVisibilityConfirmation(confirmed=False, reason="closed")
        if pid is None:
            return WindowVisibilityConfirmation(confirmed=False, reason="process_unbound")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._visibility_timeout_s
        hwnd: int | None = None
        title_confirmed = False
        while True:
            if not self._binding_is_current(pid, generation):
                return WindowVisibilityConfirmation(confirmed=False, reason="binding_changed")
            enumeration = self._api.all_top_level_windows_for_process(pid)
            if enumeration.win32_error is not None:
                return WindowVisibilityConfirmation(
                    confirmed=False,
                    reason="enum_windows_failed",
                    win32_error=enumeration.win32_error,
                )
            candidates = tuple(
                candidate
                for candidate in enumeration.windows
                if self._window_belongs_to_process(candidate, pid)
            )
            titled = tuple(
                candidate
                for candidate in candidates
                if self._api.window_title(candidate) == expected_title
            )
            if len(titled) == 1:
                hwnd = titled[0]
                title_confirmed = True
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                if len(candidates) == 1:
                    hwnd = candidates[0]
                break
            await self._sleep(min(self._poll_interval_s, remaining))

        if hwnd is None:
            return WindowVisibilityConfirmation(confirmed=False, reason="window_not_found")
        if not self._window_belongs_to_process(hwnd, pid):
            return WindowVisibilityConfirmation(confirmed=False, reason="window_changed")

        logical_bounds = (x, y, width, height)
        target_bounds: tuple[int, int, int, int] | None = None
        observed_bounds: tuple[int, int, int, int] | None = None
        confirmed_since: float | None = None
        visible_confirmed = False
        bounds_confirmed = False
        deadline = loop.time() + self._visibility_timeout_s
        while True:
            if not self._binding_is_current(pid, generation):
                return WindowVisibilityConfirmation(
                    confirmed=False,
                    reason="binding_changed",
                    hwnd=hwnd,
                )
            if not self._window_belongs_to_process(hwnd, pid):
                return WindowVisibilityConfirmation(
                    confirmed=False,
                    reason="window_changed",
                    hwnd=hwnd,
                )
            now = loop.time()
            visible_confirmed = self._api.is_window_visible(hwnd)
            observed_bounds = self._api.window_bounds(hwnd)
            if observed_bounds is not None:
                scaled_target = _native_target_bounds(logical_bounds, observed_bounds)
                if scaled_target is not None and scaled_target != target_bounds:
                    target_bounds = scaled_target
                    confirmed_since = None
            bounds_confirmed = (
                observed_bounds is not None
                and target_bounds is not None
                and _window_bounds_close(observed_bounds, target_bounds)
            )
            if visible_confirmed and bounds_confirmed:
                if confirmed_since is None:
                    confirmed_since = now
                elif now - confirmed_since >= self._visibility_retain_s:
                    return WindowVisibilityConfirmation(
                        confirmed=True,
                        reason="confirmed",
                        hwnd=hwnd,
                        title_confirmed=title_confirmed,
                        visible_confirmed=True,
                        bounds_confirmed=True,
                        observed_bounds=observed_bounds,
                    )
            else:
                confirmed_since = None
            remaining = deadline - loop.time()
            if remaining <= 0:
                return WindowVisibilityConfirmation(
                    confirmed=False,
                    reason="visible_bounds_not_retained",
                    hwnd=hwnd,
                    title_confirmed=title_confirmed,
                    visible_confirmed=visible_confirmed,
                    bounds_confirmed=bounds_confirmed,
                    observed_bounds=observed_bounds,
                )
            await self._sleep(min(self._poll_interval_s, remaining))

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


def _native_target_bounds(
    logical_bounds: tuple[int, int, int, int],
    actual_bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x, y, width, height = logical_bounds
    _actual_x, _actual_y, actual_width, actual_height = actual_bounds
    if width <= 0 or height <= 0 or actual_width <= 0 or actual_height <= 0:
        return None
    width_scale = actual_width / width
    height_scale = actual_height / height
    scale_tolerance = max(0.02, 2.0 / min(width, height))
    if abs(width_scale - height_scale) > scale_tolerance:
        return None
    scale = (width_scale + height_scale) / 2.0
    return (
        int(x * scale),
        int(y * scale),
        actual_width,
        actual_height,
    )


def _window_bounds_close(
    actual: tuple[int, int, int, int],
    expected: tuple[int, int, int, int],
) -> bool:
    return all(abs(left - right) <= 2 for left, right in zip(actual, expected, strict=True))


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
        self._user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self._user32.GetWindowTextW.restype = ctypes.c_int
        self._user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self._user32.GetWindowRect.restype = wintypes.BOOL

    def all_top_level_windows_for_process(self, pid: int) -> WindowEnumerationResult:
        windows: list[int] = []

        def collect(hwnd: int, _lparam: int) -> bool:
            if self.process_id(hwnd) != pid:
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

    def is_window_visible(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindowVisible(hwnd))

    def window_title(self, hwnd: int) -> str:
        buffer = ctypes.create_unicode_buffer(_WINDOW_TITLE_MAX_CHARS)
        self._user32.GetWindowTextW(hwnd, buffer, _WINDOW_TITLE_MAX_CHARS)
        return buffer.value

    def window_bounds(self, hwnd: int) -> tuple[int, int, int, int] | None:
        rect = wintypes.RECT()
        if not self._user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return (
            int(rect.left),
            int(rect.top),
            int(rect.right - rect.left),
            int(rect.bottom - rect.top),
        )

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
