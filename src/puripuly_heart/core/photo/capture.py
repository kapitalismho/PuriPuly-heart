from __future__ import annotations

import asyncio
import codecs
import ctypes
import fnmatch
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.core.lifecycle import LifecycleScope
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.photo.directory_notifications import (
    DirectoryChanges,
    DirectoryNotificationSource,
    WindowsDirectoryNotifications,
)
from puripuly_heart.core.vrchat_scene_tailer import default_vrchat_log_directory

_CAMERA_SIGNAL = "[VRC Camera] Took screenshot to:"


def default_photo_directory() -> Path:
    if os.name == "nt":
        buffer = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 0x27, None, 0, buffer) == 0:
            return Path(buffer.value) / "VRChat"
    return Path.home() / "Pictures" / "VRChat"


@dataclass(frozen=True, slots=True)
class PhotoCapture:
    path: Path
    file_size: int
    file_id: tuple[int, int]
    shutter_unix_ms: int
    observed_monotonic: float


class PhotoCaptureWatcher:
    def __init__(
        self,
        photo_directory: Path | None = None,
        log_directory: Path | None = None,
        *,
        notification_source: DirectoryNotificationSource | None = None,
    ) -> None:
        self.photo_directory = photo_directory
        self.log_directory = log_directory or default_vrchat_log_directory()
        self._source = notification_source or WindowsDirectoryNotifications()
        self._lock = asyncio.Lock()
        self._available = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._scope = LifecycleScope("PhotoCaptureWatcher")
        self._started = False
        self._failure: BaseException | None = None
        self._capture: PhotoCapture | None = None
        self._seen: dict[Path, tuple[int, int]] = {}
        self._baseline_seen: dict[Path, tuple[int, int]] = {}
        self._logs: dict[Path, os.stat_result] = {}
        self._baseline_logs: dict[Path, tuple[tuple[int, int], int]] = {}
        self._log_path: Path | None = None
        self._log_id: tuple[int, int] | None = None
        self._log_offset = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending_line = ""
        self._pending_paths: dict[Path, float] = {}
        self._start_unix_ns = 0

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            self.photo_directory = (self.photo_directory or default_photo_directory()).resolve()
            self.log_directory = self.log_directory.resolve()
            self._failure = None
            self._capture = None
            self._available.clear()
            self._seen.clear()
            self._baseline_seen.clear()
            self._logs.clear()
            self._baseline_logs.clear()
            self._log_path = None
            self._log_id = None
            self._pending_paths.clear()
            self._start_unix_ns = time.time_ns()
            try:
                await self._source.start((self.photo_directory, self.log_directory))
                await run_owned_thread_call(self._baseline)
                self._started = True
                if self._scope.is_closed:
                    self._scope = LifecycleScope("PhotoCaptureWatcher")
                self._task = self._scope.create_task(self._run(), name="capture")
            except BaseException:
                await self._source.close()
                raise

    async def wait_capture(self) -> PhotoCapture:
        if not self._started:
            raise RuntimeError("Photo capture watcher is not started")
        while True:
            self._available.clear()
            if self._failure is not None:
                raise self._failure
            if self._capture is not None:
                capture, self._capture = self._capture, None
                return capture
            if not self._started:
                raise RuntimeError("Photo capture watcher is closed")
            await self._available.wait()

    async def close(self) -> None:
        async with self._lock:
            self._started = False
            self._task = None
            try:
                await self._scope.close()
            finally:
                try:
                    await self._source.close()
                finally:
                    self._capture = None
                    self._available.set()

    def _files(self, root: Path) -> dict[Path, os.stat_result]:
        found: dict[Path, os.stat_result] = {}
        try:
            for directory, _, names in os.walk(root):
                for name in names:
                    if Path(name).suffix.lower() != ".png":
                        continue
                    path = Path(directory) / name
                    try:
                        found[path] = path.stat()
                    except OSError:
                        continue
        except OSError:
            pass
        return found

    def _scan_logs(self) -> None:
        self._logs.clear()
        try:
            for path in self.log_directory.glob("output_log_*.txt"):
                try:
                    self._logs[path] = path.stat()
                except OSError:
                    continue
        except OSError:
            pass

    def _attach_log(self, path: Path, status: os.stat_result, *, baseline: bool) -> None:
        self._log_path = path
        self._log_id = (status.st_dev, status.st_ino)
        initial = self._baseline_logs.get(path)
        if baseline:
            self._log_offset = status.st_size
        elif initial is not None and initial[0] == self._log_id:
            self._log_offset = initial[1]
        else:
            self._log_offset = 0
        self._pending_line = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _baseline(self) -> None:
        self._scan_logs()
        self._baseline_logs = {
            path: ((status.st_dev, status.st_ino), status.st_size)
            for path, status in self._logs.items()
            if self._birth_ns(status) < self._start_unix_ns
        }
        if self._logs:
            path = max(self._logs, key=lambda item: (self._logs[item].st_mtime_ns, str(item)))
            status = self._logs[path]
            self._attach_log(path, status, baseline=self._birth_ns(status) < self._start_unix_ns)
        assert self.photo_directory is not None
        self._seen = {
            path: (status.st_dev, status.st_ino)
            for path, status in self._files(self.photo_directory).items()
        }
        self._baseline_seen = self._seen.copy()

    @staticmethod
    def _birth_ns(status: os.stat_result) -> int:
        return getattr(status, "st_birthtime_ns", status.st_ctime_ns)

    def _read_log(self, now: float) -> None:
        if not self._logs:
            return
        path = max(self._logs, key=lambda item: (self._logs[item].st_mtime_ns, str(item)))
        try:
            status = path.stat()
            identity = (status.st_dev, status.st_ino)
            if path != self._log_path or identity != self._log_id:
                self._attach_log(path, status, baseline=False)
            elif status.st_size < self._log_offset:
                self._baseline_logs.pop(path, None)
                self._attach_log(path, status, baseline=False)
            with path.open("rb") as stream:
                stream.seek(self._log_offset)
                while chunk := stream.read(65536):
                    self._log_offset += len(chunk)
                    text = self._pending_line + self._decoder.decode(chunk)
                    lines = text.split("\n")
                    self._pending_line = lines.pop()[-8192:]
                    for line in lines:
                        if _CAMERA_SIGNAL not in line:
                            continue
                        raw = line.split(_CAMERA_SIGNAL, 1)[1].strip().strip('"\r')
                        candidate = Path(raw)
                        if candidate.is_absolute() and candidate.suffix.lower() == ".png":
                            self._pending_paths.setdefault(candidate.resolve(), now)
        except OSError:
            return

    def _process(self, changes: DirectoryChanges) -> PhotoCapture | None:
        assert self.photo_directory is not None
        now = time.monotonic()
        wall = time.time()
        files: dict[Path, os.stat_result] = {}
        log_changed = changes.rescan
        if changes.rescan:
            files.update(self._files(self.photo_directory))
            self._scan_logs()
        for path in changes.paths:
            if path == self.photo_directory or path in self.photo_directory.parents:
                if path in changes.created:
                    files.update(self._files(self.photo_directory))
            elif self.photo_directory in path.parents:
                if path.suffix.lower() == ".png":
                    try:
                        files[path] = path.stat()
                    except OSError:
                        pass
                elif path in changes.created and path.is_dir():
                    files.update(self._files(path))
            if path == self.log_directory or path in self.log_directory.parents:
                self._scan_logs()
                log_changed = True
            elif path.parent == self.log_directory and fnmatch.fnmatch(
                path.name, "output_log_*.txt"
            ):
                try:
                    self._logs[path] = path.stat()
                except OSError:
                    self._logs.pop(path, None)
                log_changed = True
        if log_changed:
            self._read_log(now)
        for path, observed in tuple(self._pending_paths.items()):
            if now - observed > 10:
                self._pending_paths.pop(path, None)
                continue
            try:
                files[path] = path.stat()
                self._pending_paths.pop(path, None)
            except OSError:
                pass
        new: list[tuple[Path, os.stat_result]] = []
        for path, status in files.items():
            if not stat.S_ISREG(status.st_mode):
                continue
            identity = (status.st_dev, status.st_ino)
            baseline_id = self._baseline_seen.get(path)
            created = any(path == item or item in path.parents for item in changes.created)
            if baseline_id is not None and (
                created or (changes.rescan and self._birth_ns(status) >= self._start_unix_ns)
            ):
                self._baseline_seen.pop(path, None)
                if self._seen.get(path) == baseline_id:
                    self._seen.pop(path, None)
            if self._seen.get(path) != identity:
                new.append((path, status))
                self._baseline_seen.pop(path, None)
            self._seen[path] = identity
        if not new:
            return None
        path, status = max(
            new,
            key=lambda item: (self._birth_ns(item[1]), item[1].st_mtime_ns, str(item[0])),
        )
        shutter = min(wall, getattr(status, "st_birthtime", min(status.st_ctime, status.st_mtime)))
        observed = now - max(0, wall - shutter)
        return PhotoCapture(
            path, status.st_size, (status.st_dev, status.st_ino), int(shutter * 1000), observed
        )

    async def _run(self) -> None:
        try:
            if self._log_path is not None and self._log_path not in self._baseline_logs:
                capture = await run_owned_thread_call(
                    lambda: self._process(DirectoryChanges(paths=frozenset({self._log_path})))
                )
                if capture is not None:
                    self._capture = capture
                    self._available.set()
            while True:
                if self._pending_paths:
                    timeout = asyncio.timeout(0.05)
                    try:
                        async with timeout:
                            changes = await self._source.wait_changes()
                    except TimeoutError:
                        if not timeout.expired():
                            raise
                        changes = DirectoryChanges()
                else:
                    changes = await self._source.wait_changes()
                capture = await run_owned_thread_call(lambda: self._process(changes))
                if capture is not None:
                    self._capture = capture
                    self._available.set()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._failure = error
            self._available.set()
