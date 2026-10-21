from __future__ import annotations

import ctypes
import string
from dataclasses import dataclass
from functools import cache

_TH32CS_SNAPPROCESS = 0x00000002
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TOKEN_QUERY = 0x0008
_TOKEN_USER_CLASS = 1
_SYSTEM_PROCESS_ID_INFORMATION_CLASS = 0x58
_MAX_PATH_CHARS = 32767
_TOKEN_USER_BUFFER_BYTES = 256
_WINDOWS_TO_UNIX_EPOCH_100NS = 116444736000000000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


@dataclass(frozen=True, slots=True)
class WindowsProcessEntry:
    pid: int
    parent_pid: int
    image_name: str


@dataclass(frozen=True, slots=True)
class WindowsProcessRecord:
    pid: int
    parent_pid: int
    image_name: str
    executable_path: str | None
    is_current_user: bool
    create_time: float | None


def list_windows_processes() -> tuple[WindowsProcessEntry, ...]:
    api = _api()
    snapshot = api.kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    entry = api.PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    entries: list[WindowsProcessEntry] = []
    try:
        found = api.kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            entries.append(
                WindowsProcessEntry(
                    pid=int(entry.th32ProcessID),
                    parent_pid=int(entry.th32ParentProcessID),
                    image_name=entry.szExeFile,
                )
            )
            found = api.kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        api.kernel32.CloseHandle(snapshot)
    return tuple(entries)


def snapshot_windows_processes() -> tuple[WindowsProcessRecord, ...]:
    api = _api()
    current_user_sid = _token_user_sid(api, api.kernel32.GetCurrentProcess())
    device_prefixes = _dos_device_prefixes(api)
    path_buffer = ctypes.create_unicode_buffer(_MAX_PATH_CHARS + 1)
    records: list[WindowsProcessRecord] = []
    for entry in list_windows_processes():
        executable_path = _executable_path(api, entry.pid, path_buffer, device_prefixes)
        is_current_user = False
        create_time = None
        handle = api.kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, entry.pid)
        if handle:
            try:
                sid = _token_user_sid(api, handle)
                is_current_user = sid is not None and sid == current_user_sid
                create_time = _create_time(api, handle)
            finally:
                api.kernel32.CloseHandle(handle)
        records.append(
            WindowsProcessRecord(
                pid=entry.pid,
                parent_pid=entry.parent_pid,
                image_name=entry.image_name,
                executable_path=executable_path,
                is_current_user=is_current_user,
                create_time=create_time,
            )
        )
    return tuple(records)


def _executable_path(
    api: _WindowsApi,
    pid: int,
    buffer: ctypes.Array[ctypes.c_wchar],
    device_prefixes: tuple[tuple[str, str], ...],
) -> str | None:
    info = api.SYSTEM_PROCESS_ID_INFORMATION()
    info.ProcessId = pid
    info.ImageName.Length = 0
    info.ImageName.MaximumLength = _MAX_PATH_CHARS * ctypes.sizeof(ctypes.c_wchar)
    info.ImageName.Buffer = ctypes.cast(buffer, ctypes.c_void_p)
    status = api.ntdll.NtQuerySystemInformation(
        _SYSTEM_PROCESS_ID_INFORMATION_CLASS,
        ctypes.byref(info),
        ctypes.sizeof(info),
        None,
    )
    if status != 0:
        return None
    native_path = ctypes.wstring_at(
        info.ImageName.Buffer, info.ImageName.Length // ctypes.sizeof(ctypes.c_wchar)
    )
    folded = native_path.casefold()
    for device, drive in device_prefixes:
        if folded.startswith(device):
            return drive + native_path[len(device) - 1 :]
    return native_path


def _dos_device_prefixes(api: _WindowsApi) -> tuple[tuple[str, str], ...]:
    buffer = ctypes.create_unicode_buffer(1024)
    prefixes: list[tuple[str, str]] = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:"
        if api.kernel32.QueryDosDeviceW(drive, buffer, len(buffer)):
            prefixes.append((buffer.value.casefold() + "\\", drive))
    return tuple(prefixes)


def _token_user_sid(api: _WindowsApi, process_handle: int) -> bytes | None:
    from ctypes import wintypes

    token = wintypes.HANDLE()
    if not api.advapi32.OpenProcessToken(process_handle, _TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        buffer = ctypes.create_string_buffer(_TOKEN_USER_BUFFER_BYTES)
        returned = wintypes.DWORD()
        if not api.advapi32.GetTokenInformation(
            token,
            _TOKEN_USER_CLASS,
            buffer,
            len(buffer),
            ctypes.byref(returned),
        ):
            return None
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        if not sid:
            return None
        return ctypes.string_at(sid, api.advapi32.GetLengthSid(sid))
    finally:
        api.kernel32.CloseHandle(token)


def _create_time(api: _WindowsApi, process_handle: int) -> float | None:
    from ctypes import wintypes

    created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
    if not api.kernel32.GetProcessTimes(
        process_handle,
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        return None
    ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
    return float(ticks - _WINDOWS_TO_UNIX_EPOCH_100NS) / 10_000_000.0


@dataclass(frozen=True, slots=True)
class _WindowsApi:
    kernel32: object
    advapi32: object
    ntdll: object
    PROCESSENTRY32W: type
    SYSTEM_PROCESS_ID_INFORMATION: type


@cache
def _api() -> _WindowsApi:
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    class UNICODE_STRING(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", ctypes.c_void_p),
        ]

    class SYSTEM_PROCESS_ID_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("ProcessId", ctypes.c_void_p),
            ("ImageName", UNICODE_STRING),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")

    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.QueryDosDeviceW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.QueryDosDeviceW.restype = wintypes.DWORD
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.GetLengthSid.argtypes = [ctypes.c_void_p]
    advapi32.GetLengthSid.restype = wintypes.DWORD
    ntdll.NtQuerySystemInformation.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG),
    ]
    ntdll.NtQuerySystemInformation.restype = wintypes.LONG

    return _WindowsApi(
        kernel32=kernel32,
        advapi32=advapi32,
        ntdll=ntdll,
        PROCESSENTRY32W=PROCESSENTRY32W,
        SYSTEM_PROCESS_ID_INFORMATION=SYSTEM_PROCESS_ID_INFORMATION,
    )


__all__ = [
    "WindowsProcessEntry",
    "WindowsProcessRecord",
    "list_windows_processes",
    "snapshot_windows_processes",
]
