from __future__ import annotations

import contextlib
import ctypes
import logging
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes

from puripuly_heart.core.runtime_logging import emit_basic_log

logger = logging.getLogger(__name__)

_WM_CLIPBOARDUPDATE = 0x031D
_WM_CLIPBOARD_WATCHER_STOP = 0x8000 + 51
_CF_UNICODETEXT = 13
_HWND_MESSAGE = wintypes.HWND(-3)


class ClipboardWatcherError(RuntimeError):
    """Raised when the Windows clipboard watcher cannot start."""


if sys.platform == "win32":
    _LRESULT = wintypes.LPARAM
    _HCURSOR = wintypes.HANDLE

    _WNDPROC = ctypes.WINFUNCTYPE(
        _LRESULT,
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )

    class _WNDCLASS(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", _WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", _HCURSOR),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]


class WindowsClipboardWatcher:
    """Event-driven Windows clipboard watcher for short Unicode text updates."""

    def __init__(self, on_text: Callable[[str], None]) -> None:
        self._on_text = on_text
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._stop_requested = threading.Event()
        self._hwnd: int | None = None
        self._class_name = f"PuriPulyClipboardWatcher-{id(self):x}"
        self._wndproc = None
        self._registered_class_hinstance: int | None = None
        self._start_error: BaseException | None = None

    def start(self) -> None:
        if sys.platform != "win32":
            raise ClipboardWatcherError("clipboard watcher is only available on Windows")
        if self._thread is not None and self._thread.is_alive():
            return

        self._ready.clear()
        self._stopped.clear()
        self._stop_requested.clear()
        self._start_error = None
        self._thread = threading.Thread(
            target=self._run_message_loop,
            name="PuriPulyClipboardWatcher",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            self._stop_requested.set()
            self._post_stop_message(self._thread)
            if self._thread.is_alive():
                self._thread.join(timeout=2.0)
            self._clear_thread_refs_if_stopped()
            raise ClipboardWatcherError("clipboard watcher did not start")
        if self._start_error is not None:
            error = self._start_error
            self._clear_thread_refs_if_stopped()
            raise ClipboardWatcherError("clipboard watcher failed to start") from error
        if self._hwnd is None or self._thread is None or not self._thread.is_alive():
            self._clear_thread_refs_if_stopped()
            raise ClipboardWatcherError("clipboard watcher stopped before it was ready")

    def stop(self) -> None:
        self._stop_requested.set()
        if sys.platform != "win32":
            return
        thread = self._thread
        if thread is None:
            return
        self._post_stop_message(thread)
        if thread.is_alive():
            thread.join(timeout=2.0)
        if thread.is_alive():
            emit_basic_log(
                logger,
                "[Clipboard] The clipboard watcher did not stop before shutdown.",
                level=logging.WARNING,
            )
            return
        self._clear_thread_refs_if_stopped()

    def _clear_thread_refs_if_stopped(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        self._thread = None
        self._hwnd = None
        self._wndproc = None
        self._registered_class_hinstance = None

    def _post_stop_message(self, thread: threading.Thread | None) -> None:
        hwnd = self._hwnd
        if not hwnd or thread is None or not thread.is_alive():
            return
        try:
            self._configure_win32_api()
            ctypes.windll.user32.PostMessageW(hwnd, _WM_CLIPBOARD_WATCHER_STOP, 0, 0)
        except Exception:
            emit_basic_log(
                logger,
                "[Clipboard] The clipboard watcher stop request failed.",
                level=logging.ERROR,
            )

    def _configure_win32_api(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.restype = wintypes.BOOL

        user32.RegisterClassW.argtypes = [ctypes.POINTER(_WNDCLASS)]
        user32.RegisterClassW.restype = wintypes.ATOM
        user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        user32.UnregisterClassW.restype = wintypes.BOOL
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.DefWindowProcW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.DefWindowProcW.restype = _LRESULT
        user32.AddClipboardFormatListener.argtypes = [wintypes.HWND]
        user32.AddClipboardFormatListener.restype = wintypes.BOOL
        user32.RemoveClipboardFormatListener.argtypes = [wintypes.HWND]
        user32.RemoveClipboardFormatListener.restype = wintypes.BOOL
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.DestroyWindow.restype = wintypes.BOOL
        user32.PostMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.PostMessageW.restype = wintypes.BOOL
        user32.PostQuitMessage.argtypes = [ctypes.c_int]
        user32.PostQuitMessage.restype = None
        user32.GetMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        ]
        user32.GetMessageW.restype = wintypes.BOOL
        user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.TranslateMessage.restype = wintypes.BOOL
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DispatchMessageW.restype = _LRESULT
        user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
        user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = wintypes.BOOL
        user32.GetClipboardData.argtypes = [wintypes.UINT]
        user32.GetClipboardData.restype = wintypes.HANDLE

    def _run_message_loop(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = None

        def wndproc(hwnd, msg, wparam, lparam):
            if msg == _WM_CLIPBOARDUPDATE:
                try:
                    text = self._read_clipboard_text(hwnd)
                except Exception:
                    logger.debug("Failed to read clipboard text", exc_info=True)
                    return 0
                if text is not None:
                    try:
                        self._on_text(text.strip())
                    except Exception:
                        emit_basic_log(
                            logger,
                            "[Clipboard] Accepted clipboard text could not be submitted.",
                            level=logging.ERROR,
                        )
                return 0
            if msg == _WM_CLIPBOARD_WATCHER_STOP:
                self._cleanup_window(hwnd)
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        try:
            self._configure_win32_api()
            self._wndproc = _WNDPROC(wndproc)
            wndclass = _WNDCLASS()
            wndclass.lpfnWndProc = self._wndproc
            wndclass.hInstance = kernel32.GetModuleHandleW(None)
            wndclass.lpszClassName = self._class_name
            if not user32.RegisterClassW(ctypes.byref(wndclass)):
                raise ClipboardWatcherError("failed to register clipboard watcher class")
            self._registered_class_hinstance = wndclass.hInstance

            hwnd = user32.CreateWindowExW(
                0,
                self._class_name,
                self._class_name,
                0,
                0,
                0,
                0,
                0,
                _HWND_MESSAGE,
                None,
                wndclass.hInstance,
                None,
            )
            if not hwnd:
                raise ClipboardWatcherError("failed to create clipboard watcher window")

            self._hwnd = hwnd
            if self._stop_requested.is_set():
                self._cleanup_window(hwnd)
                return
            if not user32.AddClipboardFormatListener(hwnd):
                raise ClipboardWatcherError("failed to register clipboard listener")
            if self._stop_requested.is_set():
                self._cleanup_window(hwnd)
                return

            self._ready.set()
            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except BaseException as exc:
            if not self._ready.is_set():
                self._start_error = exc
                self._ready.set()
            else:
                emit_basic_log(
                    logger,
                    "[Clipboard] The clipboard watcher stopped after an internal failure.",
                    level=logging.ERROR,
                )
        finally:
            if self._hwnd:
                self._cleanup_window(self._hwnd)
            else:
                self._unregister_window_class()
            self._stopped.set()

    def _cleanup_window(self, hwnd: int) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        with contextlib.suppress(Exception):
            user32.RemoveClipboardFormatListener(hwnd)
        with contextlib.suppress(Exception):
            user32.DestroyWindow(hwnd)
        hinstance = self._registered_class_hinstance
        if hinstance is None:
            with contextlib.suppress(Exception):
                hinstance = kernel32.GetModuleHandleW(None)
        self._unregister_window_class(hinstance)
        self._hwnd = None

    def _unregister_window_class(self, hinstance: int | None = None) -> None:
        if hinstance is None:
            hinstance = self._registered_class_hinstance
        if hinstance is None:
            return
        with contextlib.suppress(Exception):
            ctypes.windll.user32.UnregisterClassW(self._class_name, hinstance)
        if self._registered_class_hinstance == hinstance:
            self._registered_class_hinstance = None

    def _read_clipboard_text(self, hwnd: int) -> str | None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        if not user32.IsClipboardFormatAvailable(_CF_UNICODETEXT):
            return None
        if not user32.OpenClipboard(hwnd):
            return None
        try:
            handle = user32.GetClipboardData(_CF_UNICODETEXT)
            if not handle:
                return None
            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                return None
            try:
                return ctypes.wstring_at(ptr)
            finally:
                kernel32.GlobalUnlock(handle)
        finally:
            user32.CloseClipboard()


def create_clipboard_watcher(on_text: Callable[[str], None]) -> WindowsClipboardWatcher:
    return WindowsClipboardWatcher(on_text)
