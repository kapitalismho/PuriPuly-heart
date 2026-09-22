from __future__ import annotations

import codecs
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

_READ_CHUNK_BYTES = 65536
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
        moment = time.mktime(
            (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
                int(match.group(5)),
                int(match.group(6)),
                0,
                0,
                -1,
            )
        )
    except OverflowError, ValueError:
        return None
    if moment <= 0:
        return None
    local = time.localtime(moment)
    if (
        local.tm_year,
        local.tm_mon,
        local.tm_mday,
        local.tm_hour,
        local.tm_min,
        local.tm_sec,
    ) != tuple(int(match.group(index)) for index in range(1, 7)):
        return None
    return moment


@dataclass(frozen=True, slots=True)
class VrchatSceneLogCandidate:
    path: Path
    name_time: float
    created: float
    modified: float


def select_vrchat_log(
    candidates: tuple[VrchatSceneLogCandidate, ...],
    *,
    create_time: float,
    now: float,
) -> Path | None:
    start_second = math.floor(create_time)
    owned = [
        candidate
        for candidate in candidates
        if candidate.name_time <= now + _NAME_SKEW_FUTURE_S
        and candidate.name_time >= start_second - _NAME_BEFORE_START_TOLERANCE_S
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
class VrchatSceneLogTailer:
    log_directory: Path
    read_chunk_bytes: int = _READ_CHUNK_BYTES
    _path: Path | None = field(default=None, init=False, repr=False)
    _offset: int = field(default=0, init=False, repr=False)
    _pending_text: str = field(default="", init=False, repr=False)
    _decoder: codecs.IncrementalDecoder | None = field(default=None, init=False, repr=False)
    _identity: tuple[int, int, float] | None = field(default=None, init=False, repr=False)
    _observed_mtime: float | None = field(default=None, init=False, repr=False)

    @property
    def active_path(self) -> Path | None:
        return self._path

    @property
    def offset(self) -> int:
        return self._offset

    def scan_candidates(self) -> tuple[VrchatSceneLogCandidate, ...]:
        try:
            paths = tuple(self.log_directory.glob("output_log_*.txt"))
        except OSError:
            return ()
        found: list[VrchatSceneLogCandidate] = []
        for path in paths:
            name_time = log_start_time(path.name)
            if name_time is None:
                continue
            try:
                status = path.stat()
            except OSError:
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

    def candidate_mtime(self, path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def begin_replay(self, path: Path) -> list[str]:
        self._path = path
        self._offset = 0
        self._pending_text = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._identity = None
        self._observed_mtime = None
        lines = self._consume(path, 0)
        self._record_observation(path)
        return lines

    def read_new_lines(self) -> list[str]:
        path = self._path
        if path is None:
            return []
        try:
            status = path.stat()
        except OSError:
            raise
        if (
            self._identity is not None
            and (
                status.st_dev,
                status.st_ino,
                status.st_ctime,
            )
            != self._identity
        ):
            raise VrchatSceneLogTruncated(str(path))
        if status.st_size < self._offset:
            raise VrchatSceneLogTruncated(str(path))
        if status.st_size == self._offset:
            if self._observed_mtime is not None and status.st_mtime != self._observed_mtime:
                raise VrchatSceneLogTruncated(str(path))
            return []
        lines = self._consume(path, self._offset)
        self._record_observation(path)
        return lines

    def _record_observation(self, path: Path) -> None:
        try:
            status = path.stat()
        except OSError:
            return
        self._identity = (status.st_dev, status.st_ino, status.st_ctime)
        self._observed_mtime = status.st_mtime

    def _consume(self, path: Path, start: int) -> list[str]:
        decoder = self._decoder
        if decoder is None:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            self._decoder = decoder
        pieces: list[str] = []
        offset = start
        with open(path, "rb") as stream:
            stream.seek(start)
            while True:
                chunk = stream.read(self.read_chunk_bytes)
                if not chunk:
                    break
                offset += len(chunk)
                pieces.append(decoder.decode(chunk, False))
        text = self._pending_text + "".join(pieces)
        self._offset = offset
        if text.endswith("\n"):
            self._pending_text = ""
            return [item[:-1] if item.endswith("\r") else item for item in text.split("\n")[:-1]]
        head, _, tail = text.rpartition("\n")
        if not head and not tail:
            return []
        if not head:
            self._pending_text = tail
            return []
        self._pending_text = tail
        return [item[:-1] if item.endswith("\r") else item for item in head.split("\n")]


__all__ = [
    "VrchatSceneLogCandidate",
    "VrchatSceneLogTailer",
    "VrchatSceneLogTruncated",
    "default_vrchat_log_directory",
    "log_start_time",
    "select_vrchat_log",
]
