from __future__ import annotations

import codecs
import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.core.owned_thread import run_owned_thread_call
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
        self, photo_directory: Path | None = None, log_directory: Path | None = None
    ) -> None:
        self.photo_directory = photo_directory
        self.log_directory = log_directory or default_vrchat_log_directory()
        self._seen: dict[Path, tuple[int, int]] = {}
        self._log_path: Path | None = None
        self._log_id: tuple[int, int] | None = None
        self._log_offset = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending_line = ""
        self._pending_paths: dict[Path, float] = {}
        self._started = False

    async def start(self) -> None:
        await run_owned_thread_call(self._baseline)

    async def poll(self) -> PhotoCapture | None:
        return await run_owned_thread_call(self._poll)

    def _files(self) -> dict[Path, os.stat_result]:
        root = self.photo_directory
        if root is None:
            root = self.photo_directory = default_photo_directory()
        found: dict[Path, os.stat_result] = {}
        try:
            for path in root.rglob("*.png"):
                try:
                    if path.is_file():
                        found[path.resolve()] = path.stat()
                except OSError:
                    continue
        except OSError:
            pass
        return found

    def _latest_log(self) -> Path | None:
        try:
            return max(
                self.log_directory.glob("output_log_*.txt"),
                key=lambda path: path.stat().st_mtime_ns,
                default=None,
            )
        except OSError:
            return None

    def _attach_log(self, path: Path, *, baseline: bool) -> None:
        status = path.stat()
        self._log_path = path
        self._log_id = (status.st_dev, status.st_ino)
        self._log_offset = status.st_size if baseline else 0
        self._pending_line = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _baseline(self) -> None:
        self._seen = {path: (s.st_dev, s.st_ino) for path, s in self._files().items()}
        path = self._latest_log()
        if path is not None:
            try:
                self._attach_log(path, baseline=True)
            except OSError:
                pass
        self._started = True

    def _log_paths(self, now: float) -> None:
        path = self._latest_log()
        if path is None:
            return
        try:
            status = path.stat()
            identity = (status.st_dev, status.st_ino)
            if path != self._log_path or identity != self._log_id:
                self._attach_log(path, baseline=False)
            elif status.st_size < self._log_offset:
                self._attach_log(path, baseline=True)
            with path.open("rb") as stream:
                stream.seek(self._log_offset)
                chunk = stream.read(65536)
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

    def _poll(self) -> PhotoCapture | None:
        if not self._started:
            self._baseline()
            return None
        now = time.monotonic()
        wall = int(time.time() * 1000)
        self._log_paths(now)
        files = self._files()
        for path, observed in tuple(self._pending_paths.items()):
            if now - observed > 10:
                self._pending_paths.pop(path, None)
                continue
            try:
                files[path] = path.stat()
                self._pending_paths.pop(path, None)
            except OSError:
                pass
        new = [
            (path, status)
            for path, status in files.items()
            if self._seen.get(path) != (status.st_dev, status.st_ino)
        ]
        self._seen.update((path, (s.st_dev, s.st_ino)) for path, s in files.items())
        if not new:
            return None
        path, status = max(
            new,
            key=lambda item: (
                getattr(item[1], "st_birthtime", item[1].st_ctime),
                item[1].st_mtime_ns,
                str(item[0]),
            ),
        )
        shutter = min(
            wall / 1000, getattr(status, "st_birthtime", min(status.st_ctime, status.st_mtime))
        )
        observed = now - max(0, wall / 1000 - shutter)
        return PhotoCapture(
            path, status.st_size, (status.st_dev, status.st_ino), int(shutter * 1000), observed
        )
