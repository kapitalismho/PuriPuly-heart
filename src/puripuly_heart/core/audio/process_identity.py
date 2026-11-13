from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from puripuly_heart.config.process_capture_resolution import (
    CurrentUserProcessSnapshotPort,
    ProcessSnapshot,
    ResolvedProcessCaptureIdentity,
)
from puripuly_heart.core.audio.windows_process_table import (
    WindowsProcessRecord,
    snapshot_windows_processes,
)
from puripuly_heart.core.diagnostics import exception_diagnostics
from puripuly_heart.core.messages import ErrorDiagnostics

_WATCH_POLL_INTERVAL_S = 0.05
_WATCH_JOIN_TIMEOUT_S = 0.5


def default_current_user_process_snapshots() -> CurrentUserProcessSnapshotPort:
    if sys.platform == "win32":
        return WindowsCurrentUserProcessSnapshots()
    return PsutilCurrentUserProcessSnapshots()


@dataclass(slots=True)
class WindowsCurrentUserProcessSnapshots:
    scan: Callable[[], Iterable[WindowsProcessRecord]] = snapshot_windows_processes

    def snapshots(self) -> Iterable[ProcessSnapshot]:
        snapshots: list[ProcessSnapshot] = []
        for record in self.scan():
            executable_path = record.executable_path
            if not isinstance(executable_path, str) or record.create_time is None:
                instance_id = None
            else:
                instance_id = f"{record.pid}:{record.create_time}"
            snapshots.append(
                ProcessSnapshot(
                    pid=record.pid,
                    parent_pid=record.parent_pid or None,
                    is_current_user=record.is_current_user,
                    executable_path=executable_path,
                    instance_id=instance_id,
                )
            )
        return tuple(snapshots)


@dataclass(slots=True)
class PsutilCurrentUserProcessSnapshots:
    def snapshots(self) -> Iterable[ProcessSnapshot]:
        psutil = _import_psutil()
        current_username = psutil.Process().username()
        parent_pid_by_pid = _parent_pid_map(psutil)
        snapshots: list[ProcessSnapshot] = []
        for process in psutil.process_iter(["pid", "exe", "username", "create_time"]):
            try:
                info = process.info
                executable_path = info.get("exe")
                create_time = info.get("create_time")
                if not isinstance(executable_path, str) or create_time is None:
                    instance_id = None
                else:
                    instance_id = f"{info['pid']}:{create_time}"
                snapshots.append(
                    ProcessSnapshot(
                        pid=int(info["pid"]),
                        parent_pid=(
                            int(parent_pid_by_pid[info["pid"]])
                            if parent_pid_by_pid.get(info["pid"])
                            else None
                        ),
                        is_current_user=info.get("username") == current_username,
                        executable_path=executable_path,
                        instance_id=instance_id,
                    )
                )
            except (
                psutil.AccessDenied,
                psutil.NoSuchProcess,
                psutil.ZombieProcess,
                KeyError,
                TypeError,
            ):
                continue
        return tuple(snapshots)


@dataclass(slots=True)
class _PsutilProcessIdentityWatch:
    process: object
    identity: ResolvedProcessCaptureIdentity
    on_terminal: Callable[[str, ErrorDiagnostics | None], None]
    psutil: object
    _closed: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    identity_verified: bool = field(init=False)
    terminal_reason: str | None = None
    failure_diagnostics: ErrorDiagnostics | None = None

    def __post_init__(self) -> None:
        self.identity_verified = self._matches_identity()
        if self.identity_verified:
            self._thread = threading.Thread(target=self._wait_for_exit, daemon=True)
            self._thread.start()

    @property
    def watch_thread_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def close(self) -> None:
        self._closed.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_WATCH_JOIN_TIMEOUT_S)
            if thread.is_alive():
                raise TimeoutError("process identity watch cleanup unconfirmed")

    def _set_terminal(self, reason: str, exception: BaseException | None = None) -> bool:
        self.terminal_reason = reason
        if exception is not None:
            self.failure_diagnostics = exception_diagnostics(
                exception,
                component="capture.peer",
                operation="process_watch",
                code=f"capture.peer.{reason}",
            )
        return False

    def _matches_identity(self) -> bool:
        try:
            if f"{self.process.pid}:{self.process.create_time()}" != self.identity.instance_id:
                return self._set_terminal("target_identity_mismatch")
            return True
        except self.psutil.AccessDenied as exc:
            return self._set_terminal("identity_unavailable", exc)
        except (self.psutil.NoSuchProcess, self.psutil.ZombieProcess):
            return self._set_terminal("target_exited")

    def _still_matches_running_identity(self) -> bool:
        try:
            if not self._matches_identity():
                return False
            if not self.process.is_running():
                return self._set_terminal("target_exited")
            return True
        except self.psutil.AccessDenied as exc:
            return self._set_terminal("identity_unavailable", exc)
        except (self.psutil.NoSuchProcess, self.psutil.ZombieProcess):
            return self._set_terminal("target_exited")

    def _wait_for_exit(self) -> None:
        while not self._closed.wait(timeout=_WATCH_POLL_INTERVAL_S):
            if not self._still_matches_running_identity():
                break
        if not self._closed.is_set():
            self.on_terminal(self.terminal_reason or "identity_unavailable", self.failure_diagnostics)


@dataclass(frozen=True, slots=True)
class PsutilProcessIdentityWatcher:
    def watch(
        self,
        identity: ResolvedProcessCaptureIdentity,
        on_terminal: Callable[[str, ErrorDiagnostics | None], None],
    ) -> _PsutilProcessIdentityWatch:
        psutil = _import_psutil()
        try:
            process = psutil.Process(identity.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
            watch = _PsutilProcessIdentityWatch(
                process=_UnavailableProcess(),
                identity=identity,
                on_terminal=on_terminal,
                psutil=psutil,
            )
            watch._set_terminal(
                "identity_unavailable" if isinstance(exc, psutil.AccessDenied) else "target_exited",
                exc if isinstance(exc, psutil.AccessDenied) else None,
            )
            return watch
        return _PsutilProcessIdentityWatch(
            process=process,
            identity=identity,
            on_terminal=on_terminal,
            psutil=psutil,
        )


class _UnavailableProcess:
    pid = -1

    def create_time(self) -> float:
        return -1.0

    def is_running(self) -> bool:
        return False


def _import_psutil():  # noqa: ANN201
    import psutil

    return psutil


def _parent_pid_map(psutil):  # noqa: ANN001, ANN201
    provider = getattr(getattr(psutil, "_psplatform", None), "ppid_map", None)
    if callable(provider):
        return provider()
    return {process.pid: process.ppid() for process in psutil.process_iter(["pid"])}
