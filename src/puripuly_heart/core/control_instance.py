from __future__ import annotations

import ctypes
import hashlib
import json
import msvcrt
import os
import stat
import threading
import uuid
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO


class InstanceAlreadyRunning(RuntimeError):
    pass


class InstanceSecurityError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class InstanceRecord:
    port: int
    token: str
    instance_id: str


_OWNER_SECURITY_INFORMATION = 1
_DACL_SECURITY_INFORMATION = 4
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
_SE_DACL_PROTECTED = 0x1000
_FILE_ALL_ACCESS = 0x1F01FF
_LOCKFILE_FAIL_IMMEDIATELY = 1
_LOCKFILE_EXCLUSIVE_LOCK = 2
_ERROR_LOCK_VIOLATION = 33
_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 1
_FILE_SHARE_WRITE = 2
_FILE_SHARE_DELETE = 4
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_NORMAL = 0x80
_local_leases: set[str] = set()
_local_leases_lock = threading.Lock()


class _Acl(ctypes.Structure):
    _fields_ = [
        ("revision", ctypes.c_ubyte),
        ("reserved", ctypes.c_ubyte),
        ("size", ctypes.c_ushort),
        ("ace_count", ctypes.c_ushort),
        ("reserved2", ctypes.c_ushort),
    ]


class _AceHeader(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ubyte), ("flags", ctypes.c_ubyte), ("size", ctypes.c_ushort)]


class _TokenUser(ctypes.Structure):
    _fields_ = [("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD)]


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("internal", ctypes.c_void_p),
        ("internal_high", ctypes.c_void_p),
        ("offset", wintypes.DWORD),
        ("offset_high", wintypes.DWORD),
        ("event", ctypes.c_void_p),
    ]


class _WindowsSecurity:
    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Application instance security requires Windows")
        self.advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetCurrentProcess.restype = ctypes.c_void_p
        self.kernel.LocalFree.argtypes = [ctypes.c_void_p]
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self.kernel.CreateFileW.restype = ctypes.c_void_p
        self.kernel.LockFileEx.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_Overlapped),
        ]
        self.kernel.UnlockFileEx.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_Overlapped),
        ]
        self.advapi.OpenProcessToken.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.advapi.GetTokenInformation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi.ConvertSidToStringSidW.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.advapi.GetNamedSecurityInfoW.argtypes = [
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.advapi.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
        self.advapi.GetSecurityDescriptorControl.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi.GetSecurityDescriptorDacl.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.BOOL),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.BOOL),
        ]
        self.advapi.GetAce.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.advapi.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        token = ctypes.c_void_p()
        if not self.advapi.OpenProcessToken(
            self.kernel.GetCurrentProcess(), 8, ctypes.byref(token)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            length = wintypes.DWORD()
            self.advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(length))
            buffer = ctypes.create_string_buffer(length.value)
            if not self.advapi.GetTokenInformation(token, 1, buffer, length, ctypes.byref(length)):
                raise ctypes.WinError(ctypes.get_last_error())
            sid_string = ctypes.c_void_p()
            if not self.advapi.ConvertSidToStringSidW(
                _TokenUser.from_buffer(buffer).sid, ctypes.byref(sid_string)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                self.sid = ctypes.wstring_at(sid_string)
            finally:
                self.kernel.LocalFree(sid_string)
        finally:
            self.kernel.CloseHandle(token)
        self.descriptor = ctypes.c_void_p()
        if not self.advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f"O:{self.sid}G:{self.sid}D:P(A;;FA;;;{self.sid})",
            1,
            ctypes.byref(self.descriptor),
            None,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        owner = ctypes.c_void_p()
        if not self.advapi.GetSecurityDescriptorOwner(
            self.descriptor, ctypes.byref(owner), ctypes.byref(wintypes.BOOL())
        ):
            self.kernel.LocalFree(self.descriptor)
            raise ctypes.WinError(ctypes.get_last_error())
        self.sid_pointer = owner

    def close(self) -> None:
        self.kernel.LocalFree(self.descriptor)

    def open_record(self, path: Path) -> TextIO:
        handle = self.kernel.CreateFileW(
            str(path),
            _GENERIC_READ,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None,
            _OPEN_EXISTING,
            _FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if handle in (None, ctypes.c_void_p(-1).value):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_NOINHERIT)
        except BaseException:
            self.kernel.CloseHandle(handle)
            raise
        try:
            return os.fdopen(descriptor, "r", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise

    def protect(self, path: Path) -> None:
        if not self.advapi.SetFileSecurityW(
            str(path),
            _OWNER_SECURITY_INFORMATION
            | _DACL_SECURITY_INFORMATION
            | _PROTECTED_DACL_SECURITY_INFORMATION,
            self.descriptor,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        self.verify_path(path)

    def verify_path(self, path: Path) -> None:
        if path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise InstanceSecurityError(f"Instance path is a reparse point: {path}")
        owner = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        code = self.advapi.GetNamedSecurityInfoW(
            str(path),
            1,
            _OWNER_SECURITY_INFORMATION | _DACL_SECURITY_INFORMATION,
            ctypes.byref(owner),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
        if code:
            raise ctypes.WinError(code)
        try:
            if not owner or not self.advapi.EqualSid(owner, self.sid_pointer):
                raise InstanceSecurityError("Instance security owner differs from current user")
            control = ctypes.c_ushort()
            revision = wintypes.DWORD()
            if not self.advapi.GetSecurityDescriptorControl(
                descriptor, ctypes.byref(control), ctypes.byref(revision)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            present = wintypes.BOOL()
            defaulted = wintypes.BOOL()
            acl = ctypes.c_void_p()
            if not self.advapi.GetSecurityDescriptorDacl(
                descriptor, ctypes.byref(present), ctypes.byref(acl), ctypes.byref(defaulted)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if (
                not (control.value & _SE_DACL_PROTECTED)
                or not present.value
                or not acl.value
                or _Acl.from_address(acl.value).ace_count != 1
            ):
                raise InstanceSecurityError("Instance security DACL is not exclusive")
            ace = ctypes.c_void_p()
            if not self.advapi.GetAce(acl, 0, ctypes.byref(ace)):
                raise ctypes.WinError(ctypes.get_last_error())
            header = _AceHeader.from_address(ace.value)
            mask = ctypes.c_uint32.from_address(ace.value + 4).value
            if (
                header.type != 0
                or header.flags != 0
                or header.size < 12
                or mask != _FILE_ALL_ACCESS
                or not self.advapi.EqualSid(ace.value + 8, self.sid_pointer)
            ):
                raise InstanceSecurityError("Instance security grants access beyond current user")
        finally:
            self.kernel.LocalFree(descriptor)

    def lock(self, stream: BinaryIO) -> bool:
        handle = msvcrt.get_osfhandle(stream.fileno())
        position = _Overlapped()
        if self.kernel.LockFileEx(
            handle,
            _LOCKFILE_EXCLUSIVE_LOCK | _LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            ctypes.byref(position),
        ):
            return True
        error = ctypes.get_last_error()
        if error == _ERROR_LOCK_VIOLATION:
            return False
        raise ctypes.WinError(error)

    def unlock(self, stream: BinaryIO) -> None:
        position = _Overlapped()
        if not self.kernel.UnlockFileEx(
            msvcrt.get_osfhandle(stream.fileno()), 0, 1, 0, ctypes.byref(position)
        ):
            raise ctypes.WinError(ctypes.get_last_error())


def _exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _control_root(security: _WindowsSecurity) -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if not local or not Path(local).is_dir():
        raise InstanceSecurityError("LOCALAPPDATA is unavailable")
    root = Path(local) / "puripuly-heart" / "control"
    for directory in (root.parent, root):
        if not _exists(directory):
            try:
                directory.mkdir()
            except FileExistsError:
                pass
            else:
                security.protect(directory)
        if directory == root:
            security.verify_path(directory)
    return root


def _identity(settings_path: Path, security: _WindowsSecurity) -> tuple[Path, str]:
    canonical = os.path.normcase(str(settings_path.resolve(strict=False)))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return _control_root(security) / f"{digest}.json", canonical


def _lock_stream(path: Path, security: _WindowsSecurity) -> BinaryIO:
    if not _exists(path):
        try:
            with path.open("xb"):
                security.protect(path)
        except FileExistsError:
            pass
    security.verify_path(path)
    return path.open("r+b")


def _record(path: Path, security: _WindowsSecurity, *, temporary: bool = False) -> InstanceRecord:
    security.verify_path(path)
    if not path.is_file() or path.stat().st_size > 4096:
        raise InstanceSecurityError("Instance record is not a bounded regular file")
    try:
        with security.open_record(path) as reader:
            data = json.load(reader)
    except (UnicodeError, ValueError) as exc:
        raise InstanceSecurityError("Instance record is invalid") from exc
    if not isinstance(data, dict) or set(data) != {"port", "token", "instance_id", "settings_path"}:
        raise InstanceSecurityError("Instance record has invalid fields")
    settings_path = data["settings_path"]
    if (
        not isinstance(settings_path, str)
        or not settings_path
        or os.path.normcase(str(Path(settings_path).resolve(strict=False))) != settings_path
    ):
        raise InstanceSecurityError("Instance record has invalid settings identity")
    digest = hashlib.sha256(settings_path.encode("utf-8")).hexdigest()
    expected = f"{digest}.json"
    if path.name != expected and not (
        temporary and path.name.startswith(expected + ".") and path.name.endswith(".tmp")
    ):
        raise InstanceSecurityError("Instance record has mismatched settings identity")
    port, token, instance_id = data["port"], data["token"], data["instance_id"]
    if (
        type(port) is not int
        or not 1 <= port <= 65535
        or not isinstance(token, str)
        or not 32 <= len(token) <= 256
        or not isinstance(instance_id, str)
    ):
        raise InstanceSecurityError("Instance record has invalid values")
    try:
        uuid.UUID(instance_id)
    except ValueError as exc:
        raise InstanceSecurityError("Instance record has invalid session identity") from exc
    return InstanceRecord(port, token, instance_id)


class InstanceLease:
    def __init__(
        self, path: Path, settings_path: str, stream: BinaryIO, security: _WindowsSecurity
    ) -> None:
        self._path = path
        self._settings_path = settings_path
        self._stream = stream
        self._security = security
        self._published: InstanceRecord | None = None

    def publish(self, *, port: int, token: str, instance_id: str) -> None:
        if self._stream is None:
            raise RuntimeError("Instance lease is closed")
        record = InstanceRecord(port, token, instance_id)
        if (
            type(port) is not int
            or not 1 <= port <= 65535
            or not isinstance(token, str)
            or not 32 <= len(token) <= 256
            or not isinstance(instance_id, str)
        ):
            raise ValueError("Invalid application endpoint")
        try:
            uuid.UUID(instance_id)
        except ValueError as exc:
            raise ValueError("Invalid application instance identity") from exc
        payload = json.dumps(
            {
                "port": port,
                "token": token,
                "instance_id": instance_id,
                "settings_path": self._settings_path,
            },
            separators=(",", ":"),
        )
        temporary = self._path.with_name(f"{self._path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as output:
                self._security.protect(temporary)
                output.write(payload)
                output.flush()
            _record(temporary, self._security, temporary=True)
            os.replace(temporary, self._path)
            self._security.verify_path(self._path)
            self._published = record
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        stream = self._stream
        if stream is None:
            return
        self._stream = None
        try:
            if (
                self._published is not None
                and _exists(self._path)
                and _record(self._path, self._security) == self._published
            ):
                self._path.unlink()
        finally:
            with _local_leases_lock:
                _local_leases.discard(str(self._path))
            try:
                self._security.unlock(stream)
            finally:
                stream.close()
                self._security.close()

    def __enter__(self) -> InstanceLease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def acquire(settings_path: Path) -> InstanceLease:
    security = _WindowsSecurity()
    try:
        path, canonical = _identity(settings_path, security)
        with _local_leases_lock:
            if str(path) in _local_leases:
                raise InstanceAlreadyRunning(f"Application instance already owns {settings_path}")
            stream = _lock_stream(path.with_suffix(".lock"), security)
            try:
                locked = security.lock(stream)
            except BaseException:
                stream.close()
                raise
            if not locked:
                stream.close()
                raise InstanceAlreadyRunning(f"Application instance already owns {settings_path}")
            _local_leases.add(str(path))
        try:
            if _exists(path):
                security.verify_path(path)
                path.unlink()
        except BaseException:
            with _local_leases_lock:
                _local_leases.discard(str(path))
            security.unlock(stream)
            stream.close()
            raise
        return InstanceLease(path, canonical, stream, security)
    except BaseException:
        security.close()
        raise


def discover(settings_path: Path) -> InstanceRecord | None:
    security = _WindowsSecurity()
    try:
        path, _canonical = _identity(settings_path, security)
        if not _exists(path):
            return None
        record = _record(path, security)
        lock_path = path.with_suffix(".lock")
        if not _exists(lock_path):
            return None
        security.verify_path(lock_path)
        with _local_leases_lock:
            if str(path) in _local_leases:
                return record
        with lock_path.open("r+b") as stream:
            if not security.lock(stream):
                return record
            security.unlock(stream)
            return None
    finally:
        security.close()


def discover_instances() -> list[dict[str, str | int]]:
    security = _WindowsSecurity()
    try:
        local = os.environ.get("LOCALAPPDATA")
        if not local or not Path(local).is_dir():
            raise InstanceSecurityError("LOCALAPPDATA is unavailable")
        root = Path(local) / "puripuly-heart" / "control"
        if not _exists(root):
            return []
        security.verify_path(root)
        records: list[dict[str, str | int]] = []
        for path in root.glob("*.json"):
            try:
                record = _record(path, security)
                with security.open_record(path) as reader:
                    settings_path = json.load(reader)["settings_path"]
                if discover(Path(settings_path)) != record:
                    continue
            except OSError, ValueError, InstanceSecurityError, KeyError:
                continue
            records.append(
                {
                    "settings_path": settings_path,
                    "port": record.port,
                    "instance_id": record.instance_id,
                }
            )
        return records
    finally:
        security.close()
