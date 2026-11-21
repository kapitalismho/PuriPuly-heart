from __future__ import annotations

import _overlapped
import _winapi
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from puripuly_heart.config.process_capture_resolution import ResolvedProcessCaptureIdentity
from puripuly_heart.core.audio.windows_process_table import windows_process_create_time
from puripuly_heart.core.diagnostics import exception_diagnostics
from puripuly_heart.core.messages import ErrorDiagnostics

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_INVALID_PARAMETER = 87


@dataclass(slots=True)
class WindowsProcessIdentityWatch:
    identity: ResolvedProcessCaptureIdentity
    on_terminal: Callable[[str, ErrorDiagnostics | None], None]
    identity_verified: bool = field(default=False, init=False)
    terminal_reason: str | None = field(default=None, init=False)
    failure_diagnostics: ErrorDiagnostics | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop: int | None = field(default=None, init=False)
    _closing: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        try:
            process = _winapi.OpenProcess(
                _winapi.SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION, False, self.identity.pid
            )
        except OSError as exc:
            if exc.winerror == _ERROR_INVALID_PARAMETER:
                self.terminal_reason = "target_exited"
            else:
                self._unavailable(exc)
            return
        try:
            if _winapi.WaitForSingleObject(process, 0) == _winapi.WAIT_OBJECT_0:
                self.terminal_reason = "target_exited"
                return
            created = windows_process_create_time(process)
            if self.identity.instance_id != f"{self.identity.pid}:{created}":
                self.terminal_reason = "target_identity_mismatch"
                return
            stop = _overlapped.CreateEvent(None, True, False, None)
            self._stop = stop
            self._thread = threading.Thread(target=self._wait, args=(stop, process), daemon=True)
            self._thread.start()
            self.identity_verified = True
        except OSError as exc:
            self._unavailable(exc)
        finally:
            if not self.identity_verified:
                _winapi.CloseHandle(process)
                if self._stop is not None:
                    _winapi.CloseHandle(self._stop)
                    self._stop = None
                self._thread = None

    def _unavailable(self, exc: OSError) -> None:
        self.terminal_reason = "identity_unavailable"
        self.failure_diagnostics = exception_diagnostics(
            exc,
            component="capture.peer",
            operation="process_watch",
            code="capture.peer.identity_unavailable",
        )

    @property
    def watch_thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _wait(self, stop: int, process: int) -> None:
        try:
            result = _winapi.WaitForMultipleObjects((stop, process), False, _winapi.INFINITE)
            if result == _winapi.WAIT_OBJECT_0:
                return
            self.terminal_reason = "target_exited"
        except OSError as exc:
            self._unavailable(exc)
        finally:
            _winapi.CloseHandle(process)
            with self._lock:
                _winapi.CloseHandle(stop)
                self._stop = None
                notify = not self._closing
        if notify:
            self.on_terminal(self.terminal_reason, self.failure_diagnostics)

    def close(self) -> None:
        with self._lock:
            self._closing = True
            if self._stop is not None:
                _winapi.SetEvent(self._stop)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
            if thread.is_alive():
                raise TimeoutError("process identity watch cleanup unconfirmed")
