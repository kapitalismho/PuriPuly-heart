from __future__ import annotations

import asyncio
import logging
import ntpath
import time
from dataclasses import dataclass, field
from pathlib import Path

from puripuly_heart.config.process_capture_resolution import (
    CurrentUserProcessSnapshotPort,
    ProcessSnapshot,
    ResolvedProcessCaptureIdentity,
)
from puripuly_heart.config.settings_vnext.schema import ProcessCaptureTargetIntent
from puripuly_heart.core.audio.process_identity import (
    PsutilCurrentUserProcessSnapshots,
    PsutilProcessIdentityWatcher,
)
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.runtime_logging import emit_basic_log
from puripuly_heart.core.vrchat_scene import SceneSnapshotProvider, VrchatSceneSnapshot
from puripuly_heart.core.vrchat_scene_events import UNAVAILABLE_SNAPSHOT
from puripuly_heart.core.vrchat_scene_parser import parse_vrchat_scene_line
from puripuly_heart.core.vrchat_scene_tailer import (
    VrchatSceneLogCandidate,
    VrchatSceneLogTailer,
    VrchatSceneLogTruncated,
    default_vrchat_log_directory,
    select_vrchat_log,
)
from puripuly_heart.core.vrchat_scene_tracker import VrchatScenePresenceTracker

logger = logging.getLogger(__name__)

_SCOPE_NAME = "VrchatSceneService"
_MONITOR_TASK = "monitor"
_DEFAULT_POLL_INTERVAL_S = 3.0
_DEFAULT_QUIESCENCE_S = 2.0


def _by_path(found: tuple[VrchatSceneLogCandidate, ...], path: Path) -> VrchatSceneLogCandidate:
    for candidate in found:
        if candidate.path == path:
            return candidate
    raise LookupError("selected log vanished before replay")


def _create_time_of(instance_id: str | None) -> float | None:
    if not isinstance(instance_id, str):
        return None
    _head, separator, tail = instance_id.rpartition(":")
    if not separator or not tail:
        return None
    try:
        create_time = float(tail)
    except ValueError:
        return None
    if create_time <= 0:
        return None
    return create_time


def _is_newer(selected: VrchatSceneLogCandidate, current: VrchatSceneLogCandidate) -> bool:
    return (selected.name_time, selected.created, selected.modified) > (
        current.name_time,
        current.created,
        current.modified,
    )


@dataclass(slots=True)
class VrchatSceneService(SceneSnapshotProvider):
    snapshots: CurrentUserProcessSnapshotPort = field(
        default_factory=PsutilCurrentUserProcessSnapshots
    )
    watcher_factory: PsutilProcessIdentityWatcher = field(
        default_factory=PsutilProcessIdentityWatcher
    )
    log_directory: Path = field(default_factory=default_vrchat_log_directory)
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S
    quiescence_s: float = _DEFAULT_QUIESCENCE_S
    _scope: LifecycleScope = field(
        default_factory=lambda: LifecycleScope(_SCOPE_NAME), init=False, repr=False
    )
    _tracker: VrchatScenePresenceTracker = field(
        default_factory=VrchatScenePresenceTracker, init=False, repr=False
    )
    _tailer: VrchatSceneLogTailer = field(init=False, repr=False)
    _snapshot: VrchatSceneSnapshot = field(default=UNAVAILABLE_SNAPSHOT, init=False, repr=False)
    _generation: int = field(default=0, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _in_lifecycle: bool = field(default=False, init=False, repr=False)
    _instance_id: str | None = field(default=None, init=False, repr=False)
    _log_candidate: VrchatSceneLogCandidate | None = field(default=None, init=False, repr=False)
    _last_activity: float = field(default=0.0, init=False, repr=False)
    _watch: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.poll_interval_s <= 0:
            raise ValueError("scene poll interval must be positive")
        if self.quiescence_s < 0:
            raise ValueError("scene quiescence must be non-negative")
        self._tailer = VrchatSceneLogTailer(self.log_directory)

    @property
    def owner_name(self) -> str:
        return _SCOPE_NAME

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def accepting_ingress(self) -> bool:
        return self._started

    def snapshot(self) -> VrchatSceneSnapshot:
        return self._snapshot

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "status": self._snapshot.status,
            "in_lifecycle": self._in_lifecycle,
            "generation": self._generation,
        }

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._scope.is_closed:
            self._scope = LifecycleScope(_SCOPE_NAME)
        self._generation += 1
        start_lifecycle_task(self._scope, self._run(), name=f"{_MONITOR_TASK}-{self._generation}")

    def stop_ingress(self) -> None:
        self._started = False
        self._generation += 1
        self._close_watch()
        self._watch = None
        self._in_lifecycle = False
        self._instance_id = None
        self._log_candidate = None
        self._tailer = VrchatSceneLogTailer(self.log_directory)
        self._tracker.reset()
        self._publish_snapshot(UNAVAILABLE_SNAPSHOT)

    async def close(self) -> None:
        self.stop_ingress()
        scope = self._scope
        self._scope = LifecycleScope(_SCOPE_NAME)
        await scope.close()

    async def _run(self) -> None:
        while self._started:
            generation = self._generation
            try:
                await self._poll_once(generation)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._mark_file_failure(self._generation, type(error).__name__)
            try:
                await asyncio.sleep(self.poll_interval_s)
            except asyncio.CancelledError:
                raise

    async def _poll_once(self, generation: int) -> None:
        if self._in_lifecycle and self._watch_healthy():
            if generation != self._generation or not self._started:
                return
            await self._poll_log(generation)
            self._settle_if_quiet(generation)
            return
        identity = await run_owned_thread_call(self._find_vrchat_identity)
        if generation != self._generation or not self._started:
            return
        if identity is None:
            self._end_lifecycle()
            return
        if not self._in_lifecycle or identity.instance_id != self._instance_id:
            self._begin_lifecycle(identity)
            await self._replay_current(generation)
            return
        await self._poll_log(generation)
        self._settle_if_quiet(generation)

    def _watch_healthy(self) -> bool:
        watch = self._watch
        if watch is None or not self._in_lifecycle or self._instance_id is None:
            return False
        try:
            if not getattr(watch, "identity_verified", False):
                return False
            if not getattr(watch, "watch_thread_alive", False):
                return False
        except Exception:
            return False
        return True

    def _find_vrchat_identity(self) -> ResolvedProcessCaptureIdentity | None:
        try:
            items = tuple(self.snapshots.snapshots())
        except Exception:
            return None
        best: ProcessSnapshot | None = None
        best_create = -1.0
        for item in items:
            if not item.is_current_user or item.pid <= 0:
                continue
            executable_path = item.executable_path
            if not isinstance(executable_path, str) or not executable_path:
                continue
            if ntpath.basename(executable_path).casefold() != "vrchat.exe":
                continue
            create_time = _create_time_of(item.instance_id)
            if create_time is None:
                continue
            if best is None or create_time > best_create:
                best = item
                best_create = create_time
        if best is None:
            return None
        if not isinstance(best.executable_path, str) or not best.instance_id:
            return None
        try:
            target = ProcessCaptureTargetIntent.vrchat(best.executable_path)
        except Exception:
            return None
        try:
            return ResolvedProcessCaptureIdentity(
                pid=best.pid, target=target, instance_id=best.instance_id
            )
        except Exception:
            return None

    def _begin_lifecycle(self, identity: ResolvedProcessCaptureIdentity) -> None:
        self._close_watch()
        self._watch = None
        self._in_lifecycle = True
        self._instance_id = identity.instance_id
        self._log_candidate = None
        self._tailer = VrchatSceneLogTailer(self.log_directory)
        self._tracker.reset()
        self._publish_snapshot(self._tracker.snapshot())
        self._touch()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        generation = self._generation
        instance_id = identity.instance_id
        try:
            self._watch = self.watcher_factory.watch(
                identity,
                lambda: loop.call_soon_threadsafe(self._handle_terminal, generation, instance_id),
            )
        except Exception:
            self._watch = None

    def _end_lifecycle(self) -> None:
        if not self._in_lifecycle:
            if self._snapshot.status != "unavailable":
                self._publish_snapshot(UNAVAILABLE_SNAPSHOT)
            return
        self._generation += 1
        self._close_watch()
        self._watch = None
        self._in_lifecycle = False
        self._instance_id = None
        self._log_candidate = None
        self._tailer = VrchatSceneLogTailer(self.log_directory)
        self._tracker.reset()
        self._publish_snapshot(UNAVAILABLE_SNAPSHOT)

    def _handle_terminal(self, generation: int, instance_id: str) -> None:
        if generation != self._generation or not self._started:
            return
        if not self._in_lifecycle or instance_id != self._instance_id:
            return
        self._end_lifecycle()

    async def _replay_current(self, generation: int) -> None:
        create_time = _create_time_of(self._instance_id)
        if create_time is None:
            return
        tailer = self._tailer
        found = await run_owned_thread_call(tailer.scan_candidates)
        if not self._fresh(generation) or tailer is not self._tailer:
            return
        selected = select_vrchat_log(found, create_time=create_time, now=time.time())
        if selected is None:
            return
        try:
            lines = await run_owned_thread_call(lambda: tailer.begin_replay(selected))
        except Exception as error:
            self._mark_file_failure(generation, type(error).__name__)
            return
        if not self._fresh(generation) or tailer is not self._tailer:
            return
        self._log_candidate = _by_path(found, selected)
        self._feed_lines(lines)
        self._touch()
        self._settle_if_quiet(generation)

    async def _poll_log(self, generation: int) -> None:
        create_time = _create_time_of(self._instance_id)
        if create_time is None:
            return
        tailer = self._tailer
        current = self._log_candidate
        found = await run_owned_thread_call(tailer.scan_candidates)
        if not self._fresh(generation) or tailer is not self._tailer:
            return
        selected = select_vrchat_log(found, create_time=create_time, now=time.time())
        if selected is None:
            if current is None:
                return
        elif current is None:
            await self._rebuild(generation, tailer, _by_path(found, selected))
            return
        elif selected != current.path and _is_newer(_by_path(found, selected), current):
            await self._rebuild(generation, tailer, _by_path(found, selected))
            return
        try:
            lines = await run_owned_thread_call(tailer.read_new_lines)
        except VrchatSceneLogTruncated:
            await self._rebuild_same_file(generation, tailer)
            return
        except Exception as error:
            self._mark_file_failure(generation, type(error).__name__)
            return
        if not self._fresh(generation) or tailer is not self._tailer:
            return
        if self._feed_lines(lines) > 0:
            self._touch()
        self._settle_if_quiet(generation)

    async def _rebuild(
        self,
        generation: int,
        tailer: VrchatSceneLogTailer,
        selected: VrchatSceneLogCandidate,
    ) -> None:
        self._tracker.on_transition()
        self._publish_snapshot(self._tracker.snapshot())
        self._touch()
        try:
            lines = await run_owned_thread_call(lambda: tailer.begin_replay(selected.path))
        except Exception as error:
            self._mark_file_failure(generation, type(error).__name__)
            return
        if not self._fresh(generation) or tailer is not self._tailer:
            return
        self._log_candidate = selected
        self._feed_lines(lines)
        self._touch()
        self._settle_if_quiet(generation)

    async def _rebuild_same_file(self, generation: int, tailer: VrchatSceneLogTailer) -> None:
        current = self._log_candidate
        self._tracker.on_transition()
        self._publish_snapshot(self._tracker.snapshot())
        if current is None:
            return
        try:
            rebuilt = await run_owned_thread_call(lambda: tailer.begin_replay(current.path))
        except Exception as error:
            self._mark_file_failure(generation, type(error).__name__)
            return
        if not self._fresh(generation) or tailer is not self._tailer:
            return
        self._feed_lines(rebuilt)
        self._touch()
        self._settle_if_quiet(generation)

    def _settle_if_quiet(self, generation: int) -> None:
        if generation != self._generation or not self._started or not self._in_lifecycle:
            return
        if self._tracker.status != "syncing":
            return
        try:
            now = asyncio.get_running_loop().time()
        except RuntimeError:
            return
        if now - self._last_activity < self.quiescence_s:
            return
        self._tracker.settle_if_quiet()
        self._publish_snapshot(self._tracker.snapshot())

    def _publish_snapshot(self, snapshot: VrchatSceneSnapshot) -> None:
        previous = self._snapshot.status
        self._snapshot = snapshot
        if snapshot.status == previous:
            return
        return

    def _mark_file_failure(self, generation: int, reason: str = "unknown") -> None:
        if generation != self._generation or not self._started or not self._in_lifecycle:
            return
        previous = self._snapshot.status
        self._tracker.on_file_failure()
        self._snapshot = self._tracker.snapshot()
        if self._snapshot.status == previous:
            return
        emit_basic_log(
            logger,
            "[VRChat Scene] Scene context became unavailable · Cause %s",
            reason if reason.isidentifier() else "unclassified",
            level=logging.WARNING,
        )

    def _feed_lines(self, lines: list[str]) -> int:
        parsed = 0
        for line in lines:
            event = parse_vrchat_scene_line(line)
            if event is not None:
                parsed += 1
                self._tracker.on_event(event)
        self._publish_snapshot(self._tracker.snapshot())
        return parsed

    def _touch(self) -> None:
        try:
            self._last_activity = asyncio.get_running_loop().time()
        except RuntimeError:
            return

    def _fresh(self, generation: int) -> bool:
        return generation == self._generation and self._started and self._in_lifecycle

    def _close_watch(self) -> None:
        watch = self._watch
        if watch is None:
            return
        close = getattr(watch, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            return


__all__ = ["VrchatSceneService"]
