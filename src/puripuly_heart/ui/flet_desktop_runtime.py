from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from ctypes import wintypes
from pathlib import Path
from typing import Protocol

REQUIRED_FLET_DESKTOP_HOOKS = (
    "__locate_and_unpack_flet_view",
    "open_flet_view_async",
)


class UnsupportedFletDesktopRuntimeError(RuntimeError):
    pass


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


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsKillOnCloseProcessJob:
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        information = _JobObjectExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(job)
            raise OSError(error, "SetInformationJobObject failed")
        self._kernel32 = kernel32
        self._job = job
        self.failure_reason: str | None = None

    def assign(self, pid: int) -> bool:
        job = self._job
        if not job:
            self.failure_reason = "job_closed"
            return False
        process = self._kernel32.OpenProcess(
            self._PROCESS_TERMINATE | self._PROCESS_SET_QUOTA,
            False,
            pid,
        )
        if not process:
            self.failure_reason = f"open_process_failed:{ctypes.get_last_error()}"
            return False
        try:
            assigned = bool(self._kernel32.AssignProcessToJobObject(job, process))
            if assigned:
                self.failure_reason = None
            else:
                self.failure_reason = f"assign_process_failed:{ctypes.get_last_error()}"
            return assigned
        finally:
            self._kernel32.CloseHandle(process)

    def close(self) -> None:
        job = self._job
        self._job = None
        if job:
            self._kernel32.CloseHandle(job)


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
        return _WindowsKillOnCloseProcessJob()
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
        self._trace_started_at = time.monotonic()

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def process_info(self) -> tuple[int, str | None] | None:
        process = self._process
        if process is None or process.pid is None:
            return None
        return int(process.pid), self._pid_file

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
                        job_failure_reason = f"job_assignment_exception:{type(exc).__name__}"
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
                    endpoint_identity=endpoint_identity,
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
        await self._reap_rejected_process(process, pid_file)
        return False

    async def close(self) -> None:
        async with self._lock:
            self._accepting = False
            if self._close_task is None or (
                self._close_task.done()
                and (self._process is not None or self._pid_file is not None)
            ):
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
                        process,
                        self._graceful_timeout_s,
                        generation,
                    )
                if not exit_confirmed:
                    self._emit(
                        "graceful_close_timeout",
                        generation=generation,
                        pid=self._pid(process),
                    )
                    self._emit(
                        "terminate_requested",
                        generation=generation,
                        pid=self._pid(process),
                    )
                    terminate_failed = False
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    except Exception as exc:
                        terminate_failed = True
                        self._emit(
                            "terminate_failed",
                            generation=generation,
                            pid=self._pid(process),
                            exception_type=type(exc).__name__,
                        )
                    if not terminate_failed:
                        exit_confirmed = await self._wait_for_process_exit(
                            process,
                            self._terminate_timeout_s,
                            generation,
                        )
                if not exit_confirmed:
                    self._emit(
                        "kill_requested",
                        generation=generation,
                        pid=self._pid(process),
                    )
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    except Exception as exc:
                        self._emit(
                            "kill_failed",
                            generation=generation,
                            pid=self._pid(process),
                            exception_type=type(exc).__name__,
                        )
                    self._process_job.close()
                    exit_confirmed = await self._wait_for_process_exit(process, None, generation)
                if not exit_confirmed:
                    raise RuntimeError("Flet desktop view process did not exit")
                self._emit(
                    "process_exited",
                    generation=generation,
                    pid=self._pid(process),
                    returncode=process.returncode,
                )
        finally:
            if exit_confirmed:
                pid_file_removed = self._remove_pid_file(pid_file, generation)
            self._process_job.close()
            async with self._lock:
                if exit_confirmed and pid_file_removed and self._process is process:
                    self._process = None
                    self._pid_file = None
                    self._endpoint_identity = None
                    self._trace_sink = None
            if not exit_confirmed or not pid_file_removed:
                self._emit(
                    "cleanup_incomplete",
                    generation=generation,
                    pid=self._pid(process),
                    process_exited=exit_confirmed,
                    pid_file_retained=not pid_file_removed,
                )
        if not pid_file_removed:
            raise RuntimeError("Flet desktop view PID file cleanup failed")

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
            self._emit(
                "graceful_close_failed",
                generation=generation,
                pid=pid,
                exception_type=type(exc).__name__,
            )
            return
        if not requested:
            self._emit(
                "graceful_close_failed",
                generation=generation,
                pid=pid,
                reason=getattr(self._close_requester, "failure_reason", None),
            )

    async def _reap_rejected_process(
        self,
        process: FletDesktopViewProcess,
        pid_file: str | None,
    ) -> None:
        generation = self._generation
        self._emit(
            "process_rejected",
            generation=generation,
            pid=self._pid(process),
        )
        exited = await self._wait_for_process_exit(process, 0.0, generation)
        if not exited:
            terminate_failed = False
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            except Exception:
                terminate_failed = True
            if not terminate_failed:
                exited = await self._wait_for_process_exit(
                    process,
                    self._terminate_timeout_s,
                    generation,
                )
            if not exited:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                exited = await self._wait_for_process_exit(process, None, generation)
        if not exited:
            self._emit(
                "cleanup_incomplete",
                generation=generation,
                pid=self._pid(process),
                process_exited=False,
                pid_file_retained=pid_file is not None,
            )
            raise RuntimeError("Rejected Flet desktop view process did not exit")
        if not self._remove_pid_file(pid_file, generation):
            raise RuntimeError("Rejected Flet desktop view PID file cleanup failed")

    async def _wait_for_process_exit(
        self,
        process: FletDesktopViewProcess,
        timeout_s: float | None,
        generation: int,
    ) -> bool:
        if process.returncode is not None:
            return True
        task = asyncio.create_task(process.wait())
        try:
            if timeout_s is None:
                await task
            else:
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
            self._emit(
                "process_wait_failed",
                generation=generation,
                pid=self._pid(process),
                exception_type=type(exc).__name__,
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
            self._emit(
                "pid_file_remove_failed",
                generation=generation,
                pid_file=Path(pid_file).name,
                exception_type=type(exc).__name__,
            )
            return False
        self._emit("pid_file_removed", generation=generation, pid_file=Path(pid_file).name)
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
        sink(event, fields)


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
        process, pid_file = await open_hidden_view(page_url, assets_dir, hidden)
        accepted = process_owner is None or await process_owner.attach(
            process,
            pid_file,
            endpoint_identity=page_url,
        )
        if accepted and on_process_started is not None and process.pid is not None:
            on_process_started(int(process.pid), pid_file)
        return process, pid_file

    original = flet_desktop.open_flet_view_async
    flet_desktop.open_flet_view_async = launch
    try:
        yield
    finally:
        flet_desktop.open_flet_view_async = original
