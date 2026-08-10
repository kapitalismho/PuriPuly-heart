from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS = 1
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
CREATE_SUSPENDED = 0x00000004
MAX_JOB_MEMORY_BYTES = 24 * 1024**3
MAX_JOB_MEMORY_HEADROOM_BYTES = 1024**3


class WindowsJobError(RuntimeError):
    pass


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
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


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _last_error(action: str) -> WindowsJobError:
    code = ctypes.get_last_error()
    return WindowsJobError(f"{action} failed with Windows error {code}: {ctypes.FormatError(code)}")


class WindowsMemoryJob:
    def __init__(self, limit_bytes: int = MAX_JOB_MEMORY_BYTES) -> None:
        if sys.platform != "win32":
            raise WindowsJobError("R1 hard memory supervision requires Windows")
        self.limit_bytes = int(limit_bytes)
        if self.limit_bytes <= 0:
            raise WindowsJobError("R1 hard memory limit must be positive")
        self.headroom_bytes = min(
            MAX_JOB_MEMORY_HEADROOM_BYTES,
            max(8 * 1024**2, self.limit_bytes // 8),
        )
        if self.headroom_bytes >= self.limit_bytes:
            raise WindowsJobError("R1 hard memory ceiling is too small for supervision")
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.ntdll = ctypes.WinDLL("ntdll")
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        self.kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel32.TerminateJobObject.restype = wintypes.BOOL
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL
        self.ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        self.ntdll.NtResumeProcess.restype = wintypes.LONG
        self.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self.handle = self.kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise _last_error("CreateJobObjectW")
        self.preassignment_commit_bytes: int | None = None
        self.effective_job_memory_limit_bytes: int | None = None
        self._launched = False
        try:
            self._set_limits(JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, 0)
        except BaseException:
            self.close()
            raise

    def _set_limits(self, flags: int, job_memory_limit: int) -> None:
        information = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        information.BasicLimitInformation.LimitFlags = flags
        information.JobMemoryLimit = job_memory_limit
        if not self.kernel32.SetInformationJobObject(
            self.handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise _last_error("SetInformationJobObject")

    def _process_commit_bytes(self, process_handle: wintypes.HANDLE) -> int:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        if not self.psapi.GetProcessMemoryInfo(
            process_handle,
            ctypes.byref(counters),
            ctypes.sizeof(counters),
        ):
            raise _last_error("GetProcessMemoryInfo")
        return int(counters.PagefileUsage)

    def launch(
        self,
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
    ) -> subprocess.Popen:
        if self._launched:
            raise WindowsJobError("a Windows memory job may launch only one root process")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            creationflags=CREATE_SUSPENDED,
        )
        process_handle = wintypes.HANDLE(int(process._handle))
        try:
            preassignment_commit = self._process_commit_bytes(process_handle)
            effective_limit = self.limit_bytes - self.headroom_bytes - preassignment_commit
            if effective_limit <= 0:
                raise WindowsJobError("suspended worker already exceeds the hard memory ceiling")
            self._set_limits(
                JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                effective_limit,
            )
            if not self.kernel32.AssignProcessToJobObject(self.handle, process_handle):
                raise _last_error("AssignProcessToJobObject")
        except BaseException:
            process.kill()
            process.wait()
            raise
        self.preassignment_commit_bytes = preassignment_commit
        self.effective_job_memory_limit_bytes = effective_limit
        self._launched = True
        status = int(self.ntdll.NtResumeProcess(process_handle))
        if status != 0:
            process.kill()
            process.wait()
            raise WindowsJobError(
                f"NtResumeProcess failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}"
            )
        return process

    def peak_memory_bytes(self) -> int:
        information = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        returned = wintypes.DWORD()
        if not self.kernel32.QueryInformationJobObject(
            self.handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
            ctypes.byref(returned),
        ):
            raise _last_error("QueryInformationJobObject")
        return int(information.PeakJobMemoryUsed)

    def active_processes(self) -> int:
        information = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        returned = wintypes.DWORD()
        if not self.kernel32.QueryInformationJobObject(
            self.handle,
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
            ctypes.byref(returned),
        ):
            raise _last_error("QueryInformationJobObject")
        return int(information.ActiveProcesses)

    def terminate(self, exit_code: int = 1) -> None:
        if not self.kernel32.TerminateJobObject(self.handle, int(exit_code)):
            raise _last_error("TerminateJobObject")

    def close(self) -> None:
        if getattr(self, "handle", None):
            if not self.kernel32.CloseHandle(self.handle):
                raise _last_error("CloseHandle")
            self.handle = None

    def __enter__(self) -> WindowsMemoryJob:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        self.close()
        return False
