from __future__ import annotations

import codecs
import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

_READ_CHUNK_BYTES = 65536
_ANCHOR_BYTES = 64
_NAME_SKEW_FUTURE_S = 120.0
_NAME_BEFORE_START_TOLERANCE_S = 5.0
_CTIME_TOLERANCE_S = 120.0
_MTIME_AFTER_START_TOLERANCE_S = 1.0
_LIVE_FRESHNESS_S = 30.0
_CREATION_SKEW_S = 0.001
_LOG_NAME = re.compile(r"^output_log_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})\.txt$")


class VrchatSceneLogTruncated(Exception):
    pass


def default_vrchat_log_directory() -> Path:
    profile = os.environ.get("USERPROFILE", "")
    if profile:
        return Path(profile) / "AppData" / "LocalLow" / "VRChat" / "VRChat"
    return Path.home() / "AppData" / "LocalLow" / "VRChat" / "VRChat"


def log_start_time(file_name: str) -> float | None:
    match = _LOG_NAME.match(file_name)
    if match is None:
        return None
    try:
        fields = tuple(map(int, match.groups()))
        moment = time.mktime((*fields, 0, 0, -1))
    except OverflowError, ValueError:
        return None
    if moment <= 0:
        return None
    if time.localtime(moment)[:6] != fields:
        return None
    return moment


@dataclass(frozen=True, slots=True)
class VrchatSceneLogCandidate:
    path: Path
    name_time: float
    created: float
    modified: float


def _earliest_log_name_time(start_second: int) -> float:
    return start_second - _NAME_BEFORE_START_TOLERANCE_S


def select_vrchat_log(
    candidates: tuple[VrchatSceneLogCandidate, ...],
    *,
    create_time: float,
    now: float,
) -> Path | None:
    start_second = math.floor(create_time)
    earliest_name_time = _earliest_log_name_time(start_second)
    owned = [
        candidate
        for candidate in candidates
        if candidate.name_time <= now + _NAME_SKEW_FUTURE_S
        and candidate.name_time >= earliest_name_time
        and (
            candidate.name_time >= start_second
            or abs(candidate.created - create_time) <= _CTIME_TOLERANCE_S
        )
        and not (
            candidate.name_time == start_second
            and candidate.created < create_time - _CREATION_SKEW_S
        )
    ]
    live = [
        candidate
        for candidate in owned
        if candidate.modified >= now - _LIVE_FRESHNESS_S
        and candidate.modified >= create_time - _MTIME_AFTER_START_TOLERANCE_S
    ]
    if live:
        return _newest(live).path
    origin = [
        candidate
        for candidate in owned
        if abs(candidate.created - create_time) <= _CTIME_TOLERANCE_S
        and candidate.modified >= create_time - _MTIME_AFTER_START_TOLERANCE_S
    ]
    if origin:
        return _newest(origin).path
    return None


def _newest(
    candidates: list[VrchatSceneLogCandidate],
) -> VrchatSceneLogCandidate:
    return max(candidates, key=lambda item: (item.name_time, item.modified, item.path.name))


@dataclass(slots=True)
class VrchatSceneLogTailer[T]:
    log_directory: Path
    project_line: Callable[[str], T | None]
    read_chunk_bytes: int = _READ_CHUNK_BYTES
    _path: Path | None = field(default=None, init=False, repr=False)
    _offset: int = field(default=0, init=False, repr=False)
    _pending_text: str = field(default="", init=False, repr=False)
    _decoder: codecs.IncrementalDecoder | None = field(default=None, init=False, repr=False)
    _identity: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _anchor: bytes = field(default=b"", init=False, repr=False)
    _observed_mtime: float | None = field(default=None, init=False, repr=False)

    @property
    def active_path(self) -> Path | None:
        return self._path

    @property
    def offset(self) -> int:
        return self._offset

    def scan_candidates(self, *, create_time: float) -> tuple[VrchatSceneLogCandidate, ...]:
        earliest_name_time = _earliest_log_name_time(math.floor(create_time))
        try:
            paths = tuple(
                path
                for path in self.log_directory.iterdir()
                if path.name.startswith("output_log_") and path.suffix == ".txt"
            )
        except FileNotFoundError:
            return ()
        found: list[VrchatSceneLogCandidate] = []
        for path in paths:
            name_time = log_start_time(path.name)
            if name_time is None or name_time < earliest_name_time:
                continue
            try:
                status = path.stat()
            except FileNotFoundError:
                continue
            found.append(
                VrchatSceneLogCandidate(
                    path=path,
                    name_time=name_time,
                    created=status.st_ctime,
                    modified=status.st_mtime,
                )
            )
        return tuple(found)

    def begin_replay(self, path: Path) -> list[T]:
        self._path = path
        self._offset = 0
        self._pending_text = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._identity = None
        self._anchor = b""
        self._observed_mtime = None
        return self._consume(path, 0)

    def read_new_lines(self) -> list[T]:
        path = self._path
        if path is None:
            return []
        status = path.stat()
        if self._identity is not None and (status.st_dev, status.st_ino) != self._identity:
            raise VrchatSceneLogTruncated(str(path))
        if status.st_size < self._offset:
            raise VrchatSceneLogTruncated(str(path))
        if not self._anchor_matches(path):
            raise VrchatSceneLogTruncated(str(path))
        if status.st_size == self._offset:
            if self._observed_mtime is not None and status.st_mtime != self._observed_mtime:
                raise VrchatSceneLogTruncated(str(path))
            return []
        return self._consume(path, self._offset)

    def _record_observation(self, path: Path) -> None:
        status = path.stat()
        self._identity = (status.st_dev, status.st_ino)
        self._observed_mtime = status.st_mtime

    def _anchor_matches(self, path: Path) -> bool:
        anchor = self._anchor
        if not anchor:
            return True
        with open(path, "rb") as stream:
            stream.seek(self._offset - len(anchor))
            return stream.read(len(anchor)) == anchor

    def _consume(self, path: Path, start: int) -> list[T]:
        decoder = self._decoder
        if decoder is None:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            self._decoder = decoder
        partial = [self._pending_text] if self._pending_text else []
        results: list[T] = []
        offset = start
        with open(path, "rb") as stream:
            stream.seek(start)
            while True:
                chunk = stream.read(self.read_chunk_bytes)
                if not chunk:
                    break
                offset += len(chunk)
                if len(chunk) >= _ANCHOR_BYTES:
                    self._anchor = chunk[-_ANCHOR_BYTES:]
                else:
                    self._anchor = (self._anchor + chunk)[-_ANCHOR_BYTES:]
                parts = decoder.decode(chunk, False).split("\n")
                tail = parts.pop()
                if parts and partial:
                    partial.append(parts[0])
                    parts[0] = "".join(partial)
                    partial.clear()
                for line in parts:
                    result = self.project_line(line.removesuffix("\r"))
                    if result is not None:
                        results.append(result)
                if tail:
                    partial.append(tail)
        self._pending_text = "".join(partial)
        self._offset = offset
        self._record_observation(path)
        return results


__all__ = [
    "VrchatSceneLogCandidate",
    "VrchatSceneLogTailer",
    "VrchatSceneLogTruncated",
    "default_vrchat_log_directory",
    "log_start_time",
    "select_vrchat_log",
]
