from __future__ import annotations

import pytest

from puripuly_heart.core.clipboard import watcher as watcher_module
from puripuly_heart.core.clipboard.watcher import (
    ClipboardWatcherError,
    create_clipboard_watcher,
)


def test_clipboard_watcher_start_raises_on_non_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher_module.sys, "platform", "linux")
    watcher = create_clipboard_watcher(lambda _text: None)

    with pytest.raises(ClipboardWatcherError, match="only available on Windows"):
        watcher.start()

    watcher.stop()


def test_cleanup_window_unregisters_window_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    hinstance = object()

    class FakeUser32:
        def RemoveClipboardFormatListener(self, hwnd: int) -> int:
            calls.append(("remove", hwnd))
            return 1

        def DestroyWindow(self, hwnd: int) -> int:
            calls.append(("destroy", hwnd))
            return 1

        def UnregisterClassW(self, class_name: str, instance: object) -> int:
            calls.append(("unregister", class_name, instance))
            return 1

    class FakeKernel32:
        def GetModuleHandleW(self, name: str | None) -> object:
            calls.append(("module", name))
            return hinstance

    class FakeWindll:
        user32 = FakeUser32()
        kernel32 = FakeKernel32()

    monkeypatch.setattr(watcher_module.ctypes, "windll", FakeWindll(), raising=False)
    watcher = create_clipboard_watcher(lambda _text: None)
    watcher._hwnd = 123

    watcher._cleanup_window(123)

    assert calls == [
        ("remove", 123),
        ("destroy", 123),
        ("module", None),
        ("unregister", watcher._class_name, hinstance),
    ]
    assert watcher._hwnd is None


def test_start_timeout_requests_stop_and_joins_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher_module.sys, "platform", "win32")
    created_threads: list[FakeThread] = []

    class FakeThread:
        def __init__(self, *, target, name: str, daemon: bool) -> None:
            self.target = target
            self.name = name
            self.daemon = daemon
            self.alive = True
            self.join_timeouts: list[float | None] = []
            self.started = False
            created_threads.append(self)

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)

    monkeypatch.setattr(watcher_module.threading, "Thread", FakeThread)
    watcher = create_clipboard_watcher(lambda _text: None)

    def wait_for_ready_timeout(timeout: float) -> bool:
        assert timeout == 2.0
        return False

    monkeypatch.setattr(watcher._ready, "wait", wait_for_ready_timeout)

    with pytest.raises(ClipboardWatcherError, match="did not start"):
        watcher.start()

    assert watcher._stop_requested.is_set()
    assert created_threads[0].started is True
    assert created_threads[0].join_timeouts == [2.0]


@pytest.mark.skipif(
    watcher_module.sys.platform != "win32",
    reason="Win32 window class path is Windows-only",
)
def test_message_loop_unregisters_class_when_window_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hinstance = 9876

    class FakeFunction:
        def __init__(self, return_value: object = 1) -> None:
            self.return_value = return_value
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args: object) -> object:
            self.calls.append(args)
            return self.return_value

    class FakeUser32:
        def __init__(self) -> None:
            self.RegisterClassW = FakeFunction(1)
            self.UnregisterClassW = FakeFunction(1)
            self.CreateWindowExW = FakeFunction(0)
            self.DefWindowProcW = FakeFunction(0)
            self.AddClipboardFormatListener = FakeFunction(1)
            self.RemoveClipboardFormatListener = FakeFunction(1)
            self.DestroyWindow = FakeFunction(1)
            self.PostMessageW = FakeFunction(1)
            self.PostQuitMessage = FakeFunction(None)
            self.GetMessageW = FakeFunction(0)
            self.TranslateMessage = FakeFunction(1)
            self.DispatchMessageW = FakeFunction(0)
            self.IsClipboardFormatAvailable = FakeFunction(0)
            self.OpenClipboard = FakeFunction(0)
            self.CloseClipboard = FakeFunction(1)
            self.GetClipboardData = FakeFunction(0)

    class FakeKernel32:
        def __init__(self) -> None:
            self.GetModuleHandleW = FakeFunction(hinstance)
            self.GlobalLock = FakeFunction(0)
            self.GlobalUnlock = FakeFunction(1)

    class FakeWindll:
        def __init__(self) -> None:
            self.user32 = FakeUser32()
            self.kernel32 = FakeKernel32()

    fake_windll = FakeWindll()
    monkeypatch.setattr(watcher_module.ctypes, "windll", fake_windll, raising=False)
    watcher = create_clipboard_watcher(lambda _text: None)

    watcher._run_message_loop()

    assert isinstance(watcher._start_error, ClipboardWatcherError)
    assert watcher._ready.is_set()
    assert fake_windll.user32.UnregisterClassW.calls == [(watcher._class_name, hinstance)]
