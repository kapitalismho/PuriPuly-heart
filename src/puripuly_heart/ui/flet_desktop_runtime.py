from __future__ import annotations

import asyncio
import contextlib
import ctypes
import logging
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from ctypes import wintypes
from pathlib import Path
from typing import Protocol

from puripuly_heart.core.diagnostics import (
    current_diagnostic_operation,
    emit_failure,
    safe_exception_fields,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.windows_process_ownership import WindowsKillOnCloseProcessJob

logger = logging.getLogger(__name__)


REQUIRED_FLET_DESKTOP_HOOKS = (
    "__locate_and_unpack_flet_view",
    "open_flet_view_async",
)


class UnsupportedFletDesktopRuntimeError(RuntimeError):
    pass


class FletDesktopCleanupError(RuntimeError):
    def __init__(self, message: str, diagnostics: ErrorDiagnostics) -> None:
        super().__init__(message)
        self.failure_diagnostics = diagnostics
        self.cleanup_complete = False


class FletDesktopViewProcess(Protocol):
    pid: int | None
    returncode: int | None

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


type FletDesktopViewProcessTraceSink = Callable[[str, dict[str, object]], None]


class FletDesktopProcessJob(Protocol):
    failure_reason: str | None

    def assign(self, pid: int) -> bool: ...

    def close(self) -> None: ...


class FletDesktopViewCloseRequester(Protocol):
    failure_reason: str | None

    def request_close(self, pid: int) -> bool: ...


class _NoopFletDesktopProcessJob:
    def __init__(self, failure_reason: str) -> None:
        self.failure_reason = failure_reason

    def assign(self, pid: int) -> bool:
        return False

    def close(self) -> None:
        return None


class _NoopFletDesktopViewCloseRequester:
    def __init__(self, failure_reason: str) -> None:
        self.failure_reason = failure_reason

    def request_close(self, pid: int) -> bool:
        _ = pid
        return False


class _WindowsFletDesktopViewCloseRequester:
    _WM_CLOSE = 0x0010

    def __init__(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._enum_proc_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )
        user32.EnumWindows.argtypes = [self._enum_proc_type, wintypes.LPARAM]
        user32.EnumWindows.restype = wintypes.BOOL
        user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.PostMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.PostMessageW.restype = wintypes.BOOL
        self._user32 = user32
        self.failure_reason: str | None = None

    def request_close(self, pid: int) -> bool:
        windows: list[int] = []

        def collect(hwnd: int, _lparam: int) -> bool:
            process_id = wintypes.DWORD()
            if self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id)):
                if int(process_id.value) == pid:
                    windows.append(int(hwnd))
            return True

        callback = self._enum_proc_type(collect)
        ctypes.set_last_error(0)
        if not self._user32.EnumWindows(callback, 0):
            self.failure_reason = f"enum_windows_failed:{ctypes.get_last_error()}"
            return False
        if not windows:
            self.failure_reason = "window_not_found"
            return False
        for hwnd in windows:
            ctypes.set_last_error(0)
            if not self._user32.PostMessageW(hwnd, self._WM_CLOSE, 0, 0):
                self.failure_reason = f"post_close_failed:{ctypes.get_last_error()}"
                return False
        self.failure_reason = None
        return True


def create_flet_desktop_process_job() -> FletDesktopProcessJob:
    if os.name != "nt":
        return _NoopFletDesktopProcessJob("unsupported_platform")
    try:
        return WindowsKillOnCloseProcessJob()
    except OSError as exc:
        return _NoopFletDesktopProcessJob(f"job_creation_failed:{exc.errno}")


def create_flet_desktop_view_close_requester() -> FletDesktopViewCloseRequester:
    if os.name != "nt":
        return _NoopFletDesktopViewCloseRequester("unsupported_platform")
    try:
        return _WindowsFletDesktopViewCloseRequester()
    except OSError as exc:
        return _NoopFletDesktopViewCloseRequester(f"close_requester_creation_failed:{exc.errno}")


class FletDesktopViewProcessOwner:
    def __init__(
        self,
        *,
        graceful_timeout_s: float = 1.0,
        terminate_timeout_s: float = 0.5,
        trace_sink: FletDesktopViewProcessTraceSink | None = None,
        process_job: FletDesktopProcessJob | None = None,
        close_requester: FletDesktopViewCloseRequester | None = None,
    ) -> None:
        self._graceful_timeout_s = max(0.0, float(graceful_timeout_s))
        self._terminate_timeout_s = max(0.0, float(terminate_timeout_s))
        self._trace_sink = trace_sink
        self._process_job = process_job or create_flet_desktop_process_job()
        self._close_requester = close_requester or create_flet_desktop_view_close_requester()
        self._lock = asyncio.Lock()
        self._process: FletDesktopViewProcess | None = None
        self._pid_file: str | None = None
        self._endpoint_identity: str | None = None
        self._generation = 0
        self._accepting = True
        self._close_task: asyncio.Task[None] | None = None
        self._in_flight_launches = 0
        self._in_flight_drained = asyncio.Event()
        self._in_flight_drained.set()
        self._rejected_cleanup_debt: list[tuple[FletDesktopViewProcess, str | None]] = []
        self._primary_cleanup_exception: BaseException | None = None
        self._primary_cleanup_diagnostics: ErrorDiagnostics | None = None
        self._terminal_cleanup_diagnostics: ErrorDiagnostics | None = None
        self._observer_failure_count = 0
        self._observer_failure_diagnostics: ErrorDiagnostics | None = None
        self._close_operation: dict[str, str] = {}
        self._reap_lock = asyncio.Lock()
        self._trace_started_at = time.monotonic()

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def endpoint_identity(self) -> str | None:
        return self._endpoint_identity

    @property
    def process_info(self) -> tuple[int, str | None] | None:
        process = self._process
        if process is None or process.pid is None:
            return None
        return int(process.pid), self._pid_file

    @property
    def cleanup_snapshot(self) -> dict[str, object]:
        return {
            "pending_startups": self._in_flight_launches,
            "pending_processes": int(self._process is not None) + len(self._rejected_cleanup_debt),
            "cleanup_complete": (
                self._close_task is not None
                and self._close_task.done()
                and not self._close_task.cancelled()
                and self._close_task.exception() is None
                and self._process is None
                and not self._rejected_cleanup_debt
                and self._in_flight_launches == 0
            ),
            "diagnostic_id": (
                self._terminal_cleanup_diagnostics.diagnostic_id
                if self._terminal_cleanup_diagnostics is not None
                else None
            ),
            "observer_failure_count": self._observer_failure_count,
            "observer_failure_diagnostic_id": (
                self._observer_failure_diagnostics.diagnostic_id
                if self._observer_failure_diagnostics is not None
                else None
            ),
        }

    def retains_process_cleanup(self, process: FletDesktopViewProcess) -> bool:
        return self._process is process or any(
            owned is process for owned, _ in self._rejected_cleanup_debt
        )

    async def begin_launch(self) -> bool:
        async with self._lock:
            if not self._accepting:
                return False
            self._in_flight_launches += 1
            if self._in_flight_launches == 1:
                self._in_flight_drained.clear()
            return True

    async def end_launch(self) -> None:
        async with self._lock:
            if self._in_flight_launches > 0:
                self._in_flight_launches -= 1
            if self._in_flight_launches == 0:
                self._in_flight_drained.set()

    async def attach(
        self,
        process: FletDesktopViewProcess,
        pid_file: str | None,
        *,
        endpoint_identity: str | None = None,
    ) -> bool:
        async with self._lock:
            if self._accepting and self._process is None and self._close_task is None:
                self._generation += 1
                self._process = process
                self._pid_file = pid_file
                self._endpoint_identity = endpoint_identity
                job_assigned = False
                job_failure_reason: str | None = None
                if process.pid is not None:
                    try:
                        job_assigned = self._process_job.assign(int(process.pid))
                    except Exception as exc:
                        job_failure_reason = "job_assignment_failed"
                        self._failure("job_assignment_failed", "startup", exc)
                    else:
                        if not job_assigned:
                            job_failure_reason = getattr(
                                self._process_job,
                                "failure_reason",
                                "job_assignment_rejected",
                            )
                else:
                    job_failure_reason = "process_pid_missing"
                self._emit(
                    "process_started",
                    generation=self._generation,
                    pid=process.pid,
                    kill_on_job_close=job_assigned,
                    job_failure_reason=job_failure_reason,
                )
                if not job_assigned:
                    self._emit(
                        "job_assignment_failed",
                        generation=self._generation,
                        pid=process.pid,
                        reason=job_failure_reason,
                    )
                return True
        await self.reap_rejected_process(process, pid_file)
        return False

    async def close(self) -> None:
        async with self._lock:
            self._accepting = False
            if self._close_task is None or (
                self._close_task.done()
                and (
                    self._process is not None
                    or self._pid_file is not None
                    or self._in_flight_launches
                    or self._rejected_cleanup_debt
                    or self._close_task.cancelled()
                    or self._close_task.exception() is not None
                )
            ):
                self._close_operation = current_diagnostic_operation()
                self._close_task = asyncio.create_task(self._close_once())
            task = self._close_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            raise

    async def _close_once(self) -> None:
        process = self._process
        pid_file = self._pid_file
        generation = self._generation
        self._emit("stop_requested", generation=generation, pid=self._pid(process))
        exit_confirmed = process is None or process.returncode is not None
        pid_file_removed = pid_file is None
        try:
            if process is not None:
                if not exit_confirmed:
                    self._request_graceful_close(generation, process)
                    exit_confirmed = await self._wait_for_process_exit(
                        process, self._graceful_timeout_s, generation
                    )
                if not exit_confirmed:
                    self._emit(
                        "graceful_close_timeout", generation=generation, pid=self._pid(process)
                    )
                    self._emit("terminate_requested", generation=generation, pid=self._pid(process))
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    except Exception as exc:
                        self._failure("terminate_failed", "process_terminate", exc)
                        self._emit(
                            "terminate_failed",
                            generation=generation,
                            pid=self._pid(process),
                            **safe_exception_fields(exc),
                        )
                    exit_confirmed = await self._wait_for_process_exit(
                        process, self._terminate_timeout_s, generation
                    )
                if not exit_confirmed:
                    self._emit("kill_requested", generation=generation, pid=self._pid(process))
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    except Exception as exc:
                        self._failure("kill_failed", "process_terminate", exc)
                        self._emit(
                            "kill_failed",
                            generation=generation,
                            pid=self._pid(process),
                            **safe_exception_fields(exc),
                        )
                    try:
                        self._process_job.close()
                    except Exception as exc:
                        self._failure("job_close_failed", "process_terminate", exc)
                    exit_confirmed = await self._wait_for_process_exit(
                        process,
                        max(0.1, self._terminate_timeout_s),
                        generation,
                    )
                if exit_confirmed:
                    self._emit(
                        "process_exited",
                        generation=generation,
                        pid=self._pid(process),
                        returncode=process.returncode,
                    )
        finally:
            if self._in_flight_launches:
                try:
                    await asyncio.wait_for(
                        self._in_flight_drained.wait(),
                        timeout=max(0.1, self._graceful_timeout_s + self._terminate_timeout_s),
                    )
                except TimeoutError as exc:
                    self._failure("launch_cleanup_pending", "startup", exc)
            for rejected_process, rejected_pid_file in tuple(self._rejected_cleanup_debt):
                try:
                    await self.reap_rejected_process(rejected_process, rejected_pid_file)
                except FletDesktopCleanupError:
                    pass
            if exit_confirmed:
                pid_file_removed = self._remove_pid_file(pid_file, generation)
            try:
                self._process_job.close()
            except Exception as exc:
                self._failure("job_close_failed", "process_terminate", exc)
                exit_confirmed = False
            async with self._lock:
                if exit_confirmed and pid_file_removed and self._process is process:
                    self._process = None
                    self._pid_file = None
                    self._endpoint_identity = None
            incomplete = (
                not exit_confirmed
                or not pid_file_removed
                or bool(self._in_flight_launches or self._rejected_cleanup_debt)
            )
            if incomplete:
                self._emit(
                    "cleanup_incomplete",
                    generation=generation,
                    pid=self._pid(process),
                    process_exited=exit_confirmed,
                    pid_file_retained=not pid_file_removed,
                    launch_cleanup_failed=bool(
                        self._in_flight_launches or self._rejected_cleanup_debt
                    ),
                )
        if incomplete:
            message = (
                "Flet desktop view process did not exit"
                if not exit_confirmed
                else (
                    "Flet desktop view PID file cleanup failed"
                    if not pid_file_removed
                    else "Flet desktop view launch cleanup failed"
                )
            )
            self._terminal_cleanup_diagnostics = self._failure(
                "cleanup_failed",
                "cleanup",
                self._primary_cleanup_exception,
                severity="error",
                disposition="failed",
                fields={
                    "cleanup_complete": False,
                    "pending_startups": self._in_flight_launches,
                    "count": len(self._rejected_cleanup_debt),
                },
            )
            raise FletDesktopCleanupError(
                message, self._terminal_cleanup_diagnostics
            ) from self._primary_cleanup_exception
        if self._terminal_cleanup_diagnostics is not None:
            self._failure(
                "cleanup_recovered",
                "cleanup",
                severity="info",
                disposition="recovered",
                cause=self._terminal_cleanup_diagnostics,
                fields={"cleanup_complete": True},
            )

    def _request_graceful_close(
        self,
        generation: int,
        process: FletDesktopViewProcess | None,
    ) -> None:
        self._emit(
            "graceful_close_requested",
            generation=generation,
            pid=self._pid(process),
        )
        pid = self._pid(process)
        if pid is None:
            self._emit(
                "graceful_close_failed",
                generation=generation,
                pid=None,
                reason="process_pid_missing",
            )
            return
        try:
            requested = self._close_requester.request_close(pid)
        except Exception as exc:
            self._failure("graceful_close_failed", "process_terminate", exc)
            self._emit(
                "graceful_close_failed",
                generation=generation,
                pid=pid,
                **safe_exception_fields(exc),
            )
            return
        if not requested:
            self._emit(
                "graceful_close_failed",
                generation=generation,
                pid=pid,
                reason=getattr(self._close_requester, "failure_reason", None),
            )

    async def reap_rejected_process(
        self,
        process: FletDesktopViewProcess,
        pid_file: str | None,
    ) -> None:
        async with self._reap_lock:
            await self._reap_rejected_process_once(process, pid_file)

    async def _reap_rejected_process_once(
        self,
        process: FletDesktopViewProcess,
        pid_file: str | None,
    ) -> None:
        if not any(owned is process for owned, _ in self._rejected_cleanup_debt):
            self._rejected_cleanup_debt.append((process, pid_file))
        generation = self._generation
        self._emit("process_rejected", generation=generation, pid=self._pid(process))
        exited = process.returncode is not None
        for action in (process.terminate, process.kill):
            if exited:
                break
            try:
                action()
            except ProcessLookupError:
                pass
            except Exception as exc:
                self._failure("rejected_cleanup_failed", "process_terminate", exc)
            exited = await self._wait_for_process_exit(
                process,
                max(0.1, self._terminate_timeout_s),
                generation,
            )
        if exited and self._remove_pid_file(pid_file, generation):
            self._rejected_cleanup_debt = [
                item for item in self._rejected_cleanup_debt if item[0] is not process
            ]
            return
        self._accepting = False
        self._terminal_cleanup_diagnostics = self._failure(
            "rejected_cleanup_failed",
            "cleanup",
            self._primary_cleanup_exception,
            severity="error",
            disposition="failed",
            fields={"cleanup_complete": False, "count": len(self._rejected_cleanup_debt)},
        )
        raise FletDesktopCleanupError(
            "Rejected Flet desktop view cleanup incomplete",
            self._terminal_cleanup_diagnostics,
        ) from self._primary_cleanup_exception

    async def _wait_for_process_exit(
        self,
        process: FletDesktopViewProcess,
        timeout_s: float,
        generation: int,
    ) -> bool:
        if process.returncode is not None:
            return True
        task = asyncio.create_task(process.wait())
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        except TimeoutError:
            return False
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            self._emit(
                "process_wait_failed",
                generation=generation,
                pid=self._pid(process),
                exception_type="CancelledError",
            )
        except Exception as exc:
            self._failure("process_wait_failed", "process_wait", exc)
            self._emit(
                "process_wait_failed",
                generation=generation,
                pid=self._pid(process),
                **safe_exception_fields(exc),
            )
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return process.returncode is not None

    def _remove_pid_file(self, pid_file: str | None, generation: int) -> bool:
        if not pid_file:
            return True
        try:
            Path(pid_file).unlink(missing_ok=True)
        except OSError as exc:
            self._failure("pid_file_remove_failed", "temp_cleanup", exc)
            self._emit(
                "pid_file_remove_failed",
                generation=generation,
                **safe_exception_fields(exc),
            )
            return False
        self._emit("pid_file_removed", generation=generation, pid_file_present=False)
        return True

    @staticmethod
    def _pid(process: FletDesktopViewProcess | None) -> int | None:
        if process is None or process.pid is None:
            return None
        return int(process.pid)

    def _emit(self, event: str, **fields: object) -> None:
        sink = self._trace_sink
        if sink is None:
            return
        fields.setdefault("parent_pid", os.getpid())
        fields["monotonic_ms"] = round((time.monotonic() - self._trace_started_at) * 1000, 3)
        try:
            sink(event, fields)
        except Exception as exc:
            self._observer_failure_count += 1
            if self._observer_failure_diagnostics is None:
                self._observer_failure_diagnostics = ErrorDiagnostics(
                    component="overlay_process",
                    operation="observe",
                    code="output_observer_failed",
                    category="lifecycle",
                    visibility="diagnostic_only",
                    content_policy="metadata_only",
                    status_code=None,
                    retry_after_ms=None,
                    fields={"count": self._observer_failure_count},
                )
                emit_failure(
                    logger,
                    self._observer_failure_diagnostics,
                    severity="warning",
                    disposition="degraded",
                    stage="routing_observer",
                    exception=exc,
                )

    def _failure(
        self,
        event: str,
        stage: str,
        exception: BaseException | None = None,
        *,
        severity: str = "warning",
        disposition: str = "degraded",
        fields: dict[str, object] | None = None,
        cause: ErrorDiagnostics | None = None,
    ) -> ErrorDiagnostics:
        diagnostics = ErrorDiagnostics(
            component="overlay_process",
            operation="process_lifecycle",
            code=f"desktop_view_{event}",
            category="lifecycle",
            visibility="basic",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields=fields or {},
            cause_id=(
                (cause or self._primary_cleanup_diagnostics).diagnostic_id
                if cause or self._primary_cleanup_diagnostics
                else None
            ),
            generation=self._generation,
            operation_id=self._close_operation.get("operation_id"),
            request_id=self._close_operation.get("request_id"),
        )
        if self._primary_cleanup_diagnostics is None and disposition != "recovered":
            self._primary_cleanup_diagnostics = diagnostics
            self._primary_cleanup_exception = exception
        emit_failure(
            logger,
            diagnostics,
            severity=severity,
            disposition=disposition,
            stage=stage,
            exception=exception,
        )
        return diagnostics


def _installed_flet_desktop_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("flet-desktop")
    except PackageNotFoundError:
        return "unknown"


def require_flet_desktop_hooks(module: object) -> None:
    missing = [
        name for name in REQUIRED_FLET_DESKTOP_HOOKS if not callable(getattr(module, name, None))
    ]
    if not missing:
        return
    raise UnsupportedFletDesktopRuntimeError(
        f"flet_desktop {_installed_flet_desktop_version()} does not expose "
        f"{', '.join(missing)}, which the desktop overlay needs to launch its hidden view "
        "and own the view process. Pin flet-desktop to a supported version or update "
        "puripuly_heart.ui.flet_desktop_runtime."
    )


async def open_hidden_view(
    page_url: str,
    assets_dir: str | None,
    hidden: bool,
) -> tuple[asyncio.subprocess.Process, str | None]:
    import flet_desktop

    require_flet_desktop_hooks(flet_desktop)
    locate_view = getattr(flet_desktop, "__locate_and_unpack_flet_view")
    args, flet_env, pid_file = locate_view(page_url, assets_dir, hidden)
    kwargs: dict[str, object] = {"env": flet_env}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        )

    return (
        await asyncio.create_subprocess_exec(args[0], *args[1:], **kwargs),
        pid_file,
    )


@contextlib.contextmanager
def patch_hidden_view_launcher(
    *,
    on_process_started: Callable[[int, str | None], None] | None = None,
    process_owner: FletDesktopViewProcessOwner | None = None,
) -> Iterator[None]:
    import flet_desktop

    require_flet_desktop_hooks(flet_desktop)

    async def launch(
        page_url: str,
        assets_dir: str | None,
        hidden: bool,
    ) -> tuple[asyncio.subprocess.Process, str | None]:
        if process_owner is None:
            process, pid_file = await open_hidden_view(page_url, assets_dir, hidden)
            if on_process_started is not None and process.pid is not None:
                on_process_started(int(process.pid), pid_file)
            return process, pid_file
        if not await process_owner.begin_launch():
            raise RuntimeError("Flet desktop view owner is closed")
        process: asyncio.subprocess.Process | None = None
        pid_file: str | None = None
        attached = False
        spawn_task = asyncio.create_task(open_hidden_view(page_url, assets_dir, hidden))
        try:
            try:
                process, pid_file = await asyncio.shield(spawn_task)
            except asyncio.CancelledError:
                try:
                    process, pid_file = await asyncio.shield(spawn_task)
                except asyncio.CancelledError:
                    if not spawn_task.done():
                        spawn_task.cancel()
                        with contextlib.suppress(BaseException):
                            await spawn_task
                    raise
                raise
            attached = await process_owner.attach(
                process,
                pid_file,
                endpoint_identity=page_url,
            )
            if attached and on_process_started is not None and process.pid is not None:
                on_process_started(int(process.pid), pid_file)
            result = (process, pid_file)
            process = None
            pid_file = None
            return result
        except BaseException:
            if (
                process is not None
                and not attached
                and not process_owner.retains_process_cleanup(process)
            ):
                try:
                    await process_owner.reap_rejected_process(process, pid_file)
                except FletDesktopCleanupError:
                    pass
            raise
        finally:
            await process_owner.end_launch()

    original = flet_desktop.open_flet_view_async
    flet_desktop.open_flet_view_async = launch
    try:
        yield
    finally:
        flet_desktop.open_flet_view_async = original
