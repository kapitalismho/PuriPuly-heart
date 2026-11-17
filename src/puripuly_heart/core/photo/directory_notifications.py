from __future__ import annotations

import asyncio
import ctypes
import os
import stat
import struct
import threading
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from puripuly_heart.core.owned_thread import run_owned_thread_call


@dataclass(frozen=True, slots=True)
class DirectoryChanges:
    paths: frozenset[Path] = frozenset()
    created: frozenset[Path] = frozenset()
    rescan: bool = False


class DirectoryNotificationSource(Protocol):
    async def start(self, roots: tuple[Path, ...]) -> None: ...

    async def wait_changes(self) -> DirectoryChanges: ...

    async def close(self) -> None: ...


class WindowsDirectoryNotifications:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths: set[Path] = set()
        self._created: set[Path] = set()
        self._rescan = False
        self._failure: BaseException | None = None
        self._watches: list[_DirectoryWatch] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._available = asyncio.Event()
        self._active = False
        self._notified = False

    async def start(self, roots: tuple[Path, ...]) -> None:
        if self._active:
            return
        if os.name != "nt":
            raise RuntimeError("Photo directory notifications require Windows")
        self._loop = asyncio.get_running_loop()
        self._available.clear()
        with self._lock:
            self._paths.clear()
            self._created.clear()
            self._rescan = False
            self._failure = None
            self._notified = False
        self._active = True
        try:
            for root in dict.fromkeys(roots):
                watch = _DirectoryWatch(root, self._publish)
                self._watches.append(watch)
                await run_owned_thread_call(watch.start)
        except BaseException:
            await self.close()
            raise

    def _publish(
        self, changes: DirectoryChanges | None = None, failure: BaseException | None = None
    ) -> None:
        with self._lock:
            if failure is not None:
                self._failure = failure
            if changes is not None:
                if changes.rescan or len(self._paths | changes.paths) > 4096:
                    self._rescan = True
                    self._paths.clear()
                    self._created.clear()
                elif not self._rescan:
                    self._paths.update(changes.paths)
                    self._created.update(changes.created)
            notify = not self._notified
            self._notified = True
        loop = self._loop
        if loop is not None and notify:
            loop.call_soon_threadsafe(self._available.set)

    async def wait_changes(self) -> DirectoryChanges:
        while True:
            self._available.clear()
            with self._lock:
                self._notified = False
                if self._failure is not None:
                    raise self._failure
                if self._paths or self._rescan:
                    changes = DirectoryChanges(
                        frozenset(self._paths), frozenset(self._created), self._rescan
                    )
                    self._paths.clear()
                    self._created.clear()
                    self._rescan = False
                    return changes
            if not self._active:
                raise RuntimeError("Directory notifications are closed")
            await self._available.wait()

    async def close(self) -> None:
        self._active = False
        watches, self._watches = self._watches, []
        for watch in watches:
            watch.stop()
        try:
            await run_owned_thread_call(lambda: self._join(watches))
        finally:
            self._loop = None
            self._available.set()

    @staticmethod
    def _join(watches: list[_DirectoryWatch]) -> None:
        for watch in watches:
            watch.join()


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


class _DirectoryWatch:
    def __init__(
        self,
        root: Path,
        publish: Callable[[DirectoryChanges | None, BaseException | None], None],
    ) -> None:
        self.root = root
        self._publish = publish
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._startup_failure: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._stop_handle = None
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._bind()

    def _bind(self) -> None:
        kernel = self._kernel
        kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        ]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.CreateEventW.restype = wintypes.HANDLE
        kernel.ReadDirectoryChangesW.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, wintypes.BOOL,
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(_Overlapped), ctypes.c_void_p,
        ]
        kernel.ReadDirectoryChangesW.restype = wintypes.BOOL
        kernel.WaitForMultipleObjects.argtypes = [
            wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE), wintypes.BOOL, wintypes.DWORD,
        ]
        kernel.WaitForMultipleObjects.restype = wintypes.DWORD
        kernel.GetOverlappedResult.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_Overlapped),
            ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
        ]
        kernel.GetOverlappedResult.restype = wintypes.BOOL
        kernel.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Overlapped)]
        kernel.CancelIoEx.restype = wintypes.BOOL
        for name in ("SetEvent", "ResetEvent", "CloseHandle"):
            function = getattr(kernel, name)
            function.argtypes = [wintypes.HANDLE]
            function.restype = wintypes.BOOL

    def start(self) -> None:
        self._stop_handle = self._kernel.CreateEventW(None, True, False, None)
        if not self._stop_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._thread = threading.Thread(target=self._run, name="photo-directory", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._kernel.CloseHandle(self._stop_handle)
            self._stop_handle = None
            raise
        self._ready.wait()
        if self._startup_failure is not None:
            raise self._startup_failure

    def stop(self) -> None:
        self._stopping.set()
        if self._stop_handle is not None:
            self._kernel.SetEvent(self._stop_handle)

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
        if self._stop_handle is not None:
            self._kernel.CloseHandle(self._stop_handle)
            self._stop_handle = None

    def _open(self) -> tuple[Path, tuple[int, int], int]:
        anchor = self.root.parent
        while True:
            try:
                status = anchor.stat()
                if stat.S_ISDIR(status.st_mode):
                    break
            except (FileNotFoundError, NotADirectoryError):
                pass
            parent = anchor.parent
            if parent == anchor:
                raise FileNotFoundError(f"No existing ancestor for {self.root}")
            anchor = parent
        handle = self._kernel.CreateFileW(str(anchor), 1, 7, None, 3, 0x42000000, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return anchor, (status.st_dev, status.st_ino), handle

    def _changes(self, anchor: Path, data: bytes) -> DirectoryChanges:
        paths: set[Path] = set()
        created: set[Path] = set()
        offset = 0
        while offset < len(data):
            if offset + 12 > len(data):
                raise RuntimeError("Truncated directory notification")
            next_offset, action, length = struct.unpack_from("<III", data, offset)
            end = offset + 12 + length
            if end > len(data) or length % 2:
                raise RuntimeError("Invalid directory notification name")
            path = anchor / data[offset + 12:end].decode("utf-16-le")
            if path == self.root or self.root in path.parents or path in self.root.parents:
                paths.add(path)
                if action in (1, 5):
                    created.add(path)
            if not next_offset:
                break
            if next_offset < 12 + length or offset + next_offset >= len(data):
                raise RuntimeError("Invalid directory notification offset")
            offset += next_offset
        return DirectoryChanges(frozenset(paths), frozenset(created))

    def _run(self) -> None:
        kernel = self._kernel
        handle = None
        event = None
        pending = False
        overlapped = _Overlapped()
        transferred = wintypes.DWORD()
        try:
            event = kernel.CreateEventW(None, True, False, None)
            if not event:
                raise ctypes.WinError(ctypes.get_last_error())
            overlapped.hEvent = event
            anchor, identity, handle = self._open()
            buffer = ctypes.create_string_buffer(65536)
            waits = (wintypes.HANDLE * 2)(self._stop_handle, event)
            recoveries = 0
            rescan_pending = False
            while not self._stopping.is_set():
                kernel.ResetEvent(event)
                if not kernel.ReadDirectoryChangesW(
                    handle, buffer, len(buffer), True, 0x0000015B, None,
                    ctypes.byref(overlapped), None,
                ):
                    error = ctypes.get_last_error()
                    if error != 997:
                        raise ctypes.WinError(error)
                pending = True
                if rescan_pending:
                    self._publish(DirectoryChanges(rescan=True), None)
                    rescan_pending = False
                self._ready.set()
                result = kernel.WaitForMultipleObjects(2, waits, False, 0xFFFFFFFF)
                if result == 0:
                    break
                if result != 1:
                    raise ctypes.WinError(ctypes.get_last_error())
                success = kernel.GetOverlappedResult(
                    handle, ctypes.byref(overlapped), ctypes.byref(transferred), False
                )
                pending = False
                error = 0 if success else ctypes.get_last_error()
                if not success and error != 1022:
                    try:
                        status = anchor.stat()
                        replaced = identity != (status.st_dev, status.st_ino)
                    except (FileNotFoundError, NotADirectoryError):
                        replaced = True
                    if not replaced or error not in (2, 3, 5):
                        raise ctypes.WinError(error)
                if not success or transferred.value == 0:
                    recoveries += 1
                    if recoveries > 2:
                        raise RuntimeError("Directory notifications repeatedly failed to resynchronize")
                    kernel.CloseHandle(handle)
                    handle = None
                    anchor, identity, handle = self._open()
                    rescan_pending = True
                    continue
                recoveries = 0
                changes = self._changes(anchor, buffer.raw[:transferred.value])
                if changes.paths:
                    self._publish(changes, None)
        except BaseException as error:
            if not self._stopping.is_set():
                self._startup_failure = error
                self._publish(None, error)
        finally:
            if handle is not None:
                if pending:
                    kernel.CancelIoEx(handle, ctypes.byref(overlapped))
                    kernel.GetOverlappedResult(
                        handle, ctypes.byref(overlapped), ctypes.byref(transferred), True
                    )
                kernel.CloseHandle(handle)
            if event is not None:
                kernel.CloseHandle(event)
            self._ready.set()
