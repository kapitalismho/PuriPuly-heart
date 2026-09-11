from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from puripuly_heart import __version__

from . import openvr_vendor
from .diagnostics import OverlayDiagnosticsRecorder, default_overlay_diagnostics_dir
from .manifest import (
    OVERLAY_CONTRACT_VERSION,
    OVERLAY_EXECUTION_CONTRACT,
    OverlayLaunchManifest,
    normalize_overlay_logging_mode,
)

logger = logging.getLogger(__name__)

OVERLAY_EXECUTABLE_NAME = "PuriPulyHeartOverlay.exe"
OPENVR_RUNTIME_DLL_NAME = "openvr_api.dll"
QUIET_TAIL_PROFILE_ENV = "PURIPULY_OVERLAY_QUIET_TAIL_PROFILE"
HANDOFF_EXPERIMENT_ENV = "PURIPULY_OVERLAY_HANDOFF_EXPERIMENT"
HANDOFF_EXPERIMENT_OFF = "off"
HANDOFF_EXPERIMENT_CACHED_FRAME_REHANDOFF = "cached_frame_rehandoff"
_HANDOFF_EXPERIMENT_VALUES = frozenset(
    {HANDOFF_EXPERIMENT_OFF, HANDOFF_EXPERIMENT_CACHED_FRAME_REHANDOFF}
)


def normalize_handoff_experiment(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("overlay handoff experiment must be a string")
    normalized = value.strip().lower()
    if normalized not in _HANDOFF_EXPERIMENT_VALUES:
        raise ValueError("overlay handoff experiment must be off or cached_frame_rehandoff")
    return normalized


_EXIT_CODE_TO_FAILURE_REASON = {
    10: "contract_mismatch",
    12: "bridge_auth_failed",
    20: "openvr_init_failed",
    21: "renderer_init_failed",
}
_WINDOW_BOUNDS_EVENT_PERSIST_RULES = {
    "user": True,
    "reset": True,
    "programmatic": False,
    "launch_repair": False,
}
_WINDOW_BOUNDS_EVENT_KEYS = {"event", "source", "persist", "x", "y", "width", "height"}
_WINDOW_BOUNDS_EVENT_OPTIONAL_KEYS = {"bounds_epoch", "generation"}
_MIN_DESKTOP_WINDOW_WIDTH = 480
_MIN_DESKTOP_WINDOW_HEIGHT = 160
_INTERACTION_MODE_EVENT_MODES = {"edit", "pass_through"}
_INTERACTION_MODE_EVENT_KEYS = {"event", "mode"}
_RESET_TO_BOTTOM_CENTER_EVENT_KEYS = {"event"}
_REVERSE_DIAGNOSTIC_LIMIT = 128
_REVERSE_LINE_BYTE_LIMIT = 4 * 1024
_REVERSE_CONTROL_SLOT_LIMIT = 8
_REVERSE_LIFECYCLE_CONTROL_TYPES = frozenset(
    {"overlay_ready", "startup_error", "runtime_error", "shutdown_complete", "owner_status"}
)
_REVERSE_TERMINAL_CONTROL_TYPES = frozenset({"startup_error", "runtime_error"})
_SHUTDOWN_EVIDENCE_LIMIT = 16
_SHUTDOWN_STDERR_EVIDENCE_LIMIT = 8


class _BoundedProcessEventQueue:
    def __init__(self) -> None:
        self._controls: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._lifecycle_controls: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._diagnostics: deque[dict[str, object]] = deque(maxlen=_REVERSE_DIAGNOSTIC_LIMIT)
        self._available = asyncio.Event()
        self.dropped_diagnostics = 0
        self.rejected_controls = 0

    async def put(self, event: dict[str, object]) -> bool:
        try:
            self.put_nowait(event)
        except asyncio.QueueFull:
            self.rejected_controls += 1
            return False
        return True

    def put_nowait(self, event: dict[str, object]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "overlay_trace":
            if len(self._diagnostics) >= _REVERSE_DIAGNOSTIC_LIMIT:
                self.dropped_diagnostics += 1
            self._diagnostics.append(event)
        elif event_type in _REVERSE_LIFECYCLE_CONTROL_TYPES:
            key = (
                "terminal_failure" if event_type in _REVERSE_TERMINAL_CONTROL_TYPES else event_type
            )
            if key == "terminal_failure" and key in self._lifecycle_controls:
                self._available.set()
                return
            self._lifecycle_controls.pop(key, None)
            self._lifecycle_controls[key] = event
        else:
            payload = event.get("payload")
            payload_event = str(payload.get("event", "")) if isinstance(payload, dict) else ""
            key = f"{event_type}:{payload_event}"
            if key not in self._controls and len(self._controls) >= _REVERSE_CONTROL_SLOT_LIMIT:
                raise asyncio.QueueFull
            self._controls.pop(key, None)
            self._controls[key] = event
        self._available.set()

    async def get(self) -> dict[str, object]:
        while True:
            try:
                return self.get_nowait()
            except asyncio.QueueEmpty:
                self._available.clear()
                if not self.empty():
                    self._available.set()
                    continue
                await self._available.wait()

    def get_nowait(self) -> dict[str, object]:
        if self._lifecycle_controls:
            _, event = self._lifecycle_controls.popitem(last=False)
        elif self._controls:
            _, event = self._controls.popitem(last=False)
        elif self._diagnostics:
            event = self._diagnostics.popleft()
        else:
            raise asyncio.QueueEmpty
        if self.empty():
            self._available.clear()
        return event

    def empty(self) -> bool:
        return not self._lifecycle_controls and not self._controls and not self._diagnostics


class OverlayPreparationError(Exception):
    def __init__(self, failure_reason: str, message: str | None = None) -> None:
        super().__init__(message or failure_reason)
        self.failure_reason = failure_reason


class OverlayManagedProcess(Protocol):
    async def next_event(self) -> dict[str, object]: ...
    async def wait_for_exit(self) -> int | None: ...
    async def finish_readers(self) -> None: ...
    async def terminate(self) -> None: ...
    def set_logging_mode(self, mode: str) -> None: ...


class OverlayProcessRunner(Protocol):
    def configure_runtime(
        self,
        *,
        quiet_tail_profile: str,
        handoff_experiment: str,
    ) -> None: ...

    def prepare(self, manifest: OverlayLaunchManifest) -> Path: ...
    async def spawn(
        self,
        executable_path: Path,
        manifest_path: Path,
    ) -> OverlayManagedProcess: ...


@dataclass(slots=True)
class _AsyncioOverlayProcess:
    process: asyncio.subprocess.Process
    overlay_instance_id: str | None = None
    task_factory: Any | None = None
    terminate_grace_s: float = 1.0
    kill_exit_timeout_s: float = 2.0
    reader_cleanup_timeout_s: float = 1.0
    _events: _BoundedProcessEventQueue = field(default_factory=_BoundedProcessEventQueue)
    _reader_tasks: list[asyncio.Task[None]] = field(default_factory=list)
    _diagnostics: OverlayDiagnosticsRecorder | None = None
    _lifecycle_sink: Callable[[str, dict[str, object]], None] | None = None
    _logging_mode: str = field(init=False, default="basic")

    def __post_init__(self) -> None:
        self._start_reader(self.process.stdout, "stdout")
        self._start_reader(self.process.stderr, "stderr")

    def attach_diagnostics(
        self,
        diagnostics: OverlayDiagnosticsRecorder,
        *,
        overlay_instance_id: str,
    ) -> None:
        self._diagnostics = diagnostics
        self.overlay_instance_id = overlay_instance_id

    def set_logging_mode(self, mode: str) -> None:
        self._logging_mode = normalize_overlay_logging_mode(mode)

    def attach_lifecycle_sink(
        self,
        sink: Callable[[str, dict[str, object]], None] | None,
    ) -> None:
        self._lifecycle_sink = sink

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    def drain_events(self) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except asyncio.QueueEmpty:
                return events

    async def next_event(self) -> dict[str, object]:
        return await self._events.get()

    async def wait_for_exit(self) -> int | None:
        return await self.process.wait()

    async def finish_readers(self) -> None:
        await self._finish_readers()

    async def terminate(self) -> None:
        if self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.terminate()
        await self._wait_for_returncode_during_terminate_grace()
        if self.process.returncode is None:
            kill = getattr(self.process, "kill", None)
            if callable(kill):
                sink = self._lifecycle_sink
                if sink is not None:
                    sink("kill_requested", {"pid": self.pid})
                with contextlib.suppress(ProcessLookupError):
                    kill()
        wait_task = asyncio.create_task(
            self.wait_for_exit(),
            name="OverlayProcess:exit-confirmation",
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(wait_task),
                timeout=max(0.0, self.kill_exit_timeout_s),
            )
        except TimeoutError as exc:
            sink = self._lifecycle_sink
            if sink is not None:
                sink("termination_unconfirmed", {"pid": self.pid})
            raise RuntimeError("overlay child termination was not confirmed") from exc
        await self.finish_readers()

    async def _wait_for_returncode_during_terminate_grace(self) -> None:
        grace_s = max(0.0, self.terminate_grace_s)
        if grace_s <= 0.0:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace_s
        while self.process.returncode is None:
            remaining_s = deadline - loop.time()
            if remaining_s <= 0.0:
                return
            await asyncio.sleep(min(remaining_s, 0.05))

    def _start_reader(self, stream: asyncio.StreamReader | None, stream_name: str) -> None:
        if stream is None:
            return
        self._reader_tasks.append(
            self._create_task(
                self._read_stream(stream, stream_name),
                task_name=f"process-read-{stream_name}",
            )
        )

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        task_name: str,
    ) -> asyncio.Task[Any]:
        if self.task_factory is not None:
            return self.task_factory(coroutine, task_name=task_name)
        return asyncio.create_task(coroutine, name=f"OverlayProcess:{task_name}")

    async def _read_stream(self, stream: asyncio.StreamReader, stream_name: str) -> None:
        try:
            while True:
                raw_line = await stream.readline()
                if not raw_line:
                    return
                if len(raw_line) > _REVERSE_LINE_BYTE_LIMIT:
                    if self._diagnostics is not None:
                        self._diagnostics.note_input_rejected("oversized_child_line")
                        self._diagnostics.record_child_line(stream_name, "oversized_line_discarded")
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                event = self._parse_event_line(line)
                if event is not None:
                    dropped_before = self._events.dropped_diagnostics
                    accepted = await self._events.put(event)
                    if (
                        self._diagnostics is not None
                        and self._events.dropped_diagnostics > dropped_before
                    ):
                        self._diagnostics.note_input_rejected(
                            "reverse_diagnostic_overflow",
                            count=self._events.dropped_diagnostics - dropped_before,
                        )
                    if not accepted:
                        payload = event.get("payload")
                        payload_event = (
                            str(payload.get("event", "")) if isinstance(payload, dict) else ""
                        )
                        sink = self._lifecycle_sink
                        if sink is not None:
                            sink(
                                "reverse_control_rejected",
                                {
                                    "type": str(event.get("type", "")),
                                    "payload_event": payload_event,
                                    "reason": "control_capacity",
                                },
                            )
                        self._events.put_nowait(
                            {
                                "type": "runtime_error",
                                "failure_reason": "reverse_control_capacity",
                                "rejected_control_type": str(event.get("type", "")),
                                "rejected_payload_event": payload_event,
                            }
                        )
                    continue
                if line and self._diagnostics is not None:
                    if self._diagnostics.ingest_native_child_line(line):
                        pass
                    elif self._should_capture_failure_line(line, stream_name):
                        self._diagnostics.record_child_line(stream_name, line)
                self._log_passthrough_line(line, stream_name)
        except asyncio.CancelledError:
            raise

    def _parse_event_line(self, line: str) -> dict[str, object] | None:
        if not line:
            return None

        candidates = [line]
        if line.startswith("EVENT "):
            candidates.insert(0, line[len("EVENT ") :].strip())

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("type"), str):
                return payload
        return None

    def _log_passthrough_line(self, line: str, stream_name: str) -> None:
        if not line:
            return
        if stream_name == "stderr" or "[ERROR]" in line:
            logger.error(line)
            return
        if "[WARN]" in line:
            logger.warning(line)
            return
        if self._logging_mode == "detailed":
            logger.info(line)

    def _should_capture_failure_line(self, line: str, stream_name: str) -> bool:
        return stream_name == "stderr" or "[WARN]" in line or "[ERROR]" in line

    async def _finish_readers(self) -> None:
        tasks = tuple(self._reader_tasks)
        if not tasks:
            return
        timeout_s = max(0.0, self.reader_cleanup_timeout_s)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        eof_wait_s = timeout_s * 0.9
        done, pending = await asyncio.wait(tasks, timeout=eof_wait_s)
        cancelled_count = len(pending)
        if pending:
            sink = self._lifecycle_sink
            if sink is not None:
                sink(
                    "process_readers_cancelled",
                    {"count": cancelled_count, "reason": "reader_finish_timeout"},
                )
            for task in pending:
                task.cancel()
            remaining_s = max(0.0, deadline - loop.time())
            cancelled_done, pending = await asyncio.wait(pending, timeout=remaining_s)
            done.update(cancelled_done)
        failures: list[BaseException] = []
        for task in done:
            if task.cancelled():
                continue
            try:
                failure = task.exception()
            except asyncio.CancelledError:
                continue
            if failure is not None:
                failures.append(failure)
        unresolved = [task for task in tasks if task not in done]
        failed = [task for task in done if not task.cancelled() and task.exception() is not None]
        self._reader_tasks = unresolved + failed
        if unresolved or failures:
            sink = self._lifecycle_sink
            if sink is not None:
                sink(
                    "process_reader_cleanup_failed",
                    {
                        "unresolved_count": len(unresolved),
                        "failure_count": len(failures),
                    },
                )
            raise RuntimeError("overlay child reader cleanup failed")
        sink = self._lifecycle_sink
        if sink is not None:
            sink(
                "process_readers_finished",
                {"cancelled_count": cancelled_count},
            )


@dataclass(slots=True)
class DefaultOverlayProcessRunner:
    executable_path: Path | None = None
    task_factory: Any | None = None
    quiet_tail_profile: str = "p05"
    handoff_experiment: str = HANDOFF_EXPERIMENT_OFF

    def configure_runtime(
        self,
        *,
        quiet_tail_profile: str,
        handoff_experiment: str,
    ) -> None:
        self.quiet_tail_profile = quiet_tail_profile
        self.handoff_experiment = normalize_handoff_experiment(handoff_experiment)

    def prepare(self, manifest: OverlayLaunchManifest) -> Path:
        _ = manifest
        if self.executable_path is not None:
            path = self.executable_path
        else:
            path = self._resolve_default_executable()
        if not path.exists():
            raise FileNotFoundError(path)
        stale_source = self._newer_local_dev_overlay_source(path)
        if stale_source is not None:
            raise OverlayPreparationError(
                "stale_overlay_build",
                f"staged overlay executable is older than overlay source: {stale_source}",
            )
        if path.name == OVERLAY_EXECUTABLE_NAME:
            bundled_runtime_path = self.ensure_bundled_openvr_runtime_dll(path)
            logger.info("[overlay] OpenVR runtime DLL ready at %s", bundled_runtime_path)
        return path

    async def spawn(
        self,
        executable_path: Path,
        manifest_path: Path,
    ) -> OverlayManagedProcess:
        command: tuple[str, ...]
        if executable_path.suffix.lower() == ".py":
            command = (str(sys.executable), str(executable_path), "--config", str(manifest_path))
        else:
            command = (str(executable_path), "--config", str(manifest_path))
        child_env = os.environ.copy()
        child_env[QUIET_TAIL_PROFILE_ENV] = self.quiet_tail_profile
        child_env[HANDOFF_EXPERIMENT_ENV] = normalize_handoff_experiment(self.handoff_experiment)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
        )
        return _AsyncioOverlayProcess(process=process, task_factory=self.task_factory)

    @classmethod
    def default_executable_candidates(
        cls,
        *,
        sys_executable: Path | None = None,
        repo_root: Path | None = None,
    ) -> tuple[Path, Path]:
        executable = (sys_executable or Path(sys.executable)).resolve()
        root = repo_root or Path(__file__).resolve().parents[4]
        return executable.with_name(OVERLAY_EXECUTABLE_NAME), root / "build" / "overlay" / (
            OVERLAY_EXECUTABLE_NAME
        )

    @classmethod
    def resolve_default_executable(
        cls,
        *,
        sys_executable: Path | None = None,
        repo_root: Path | None = None,
    ) -> Path:
        packaged_sibling, staged = cls.default_executable_candidates(
            sys_executable=sys_executable,
            repo_root=repo_root,
        )
        if packaged_sibling.exists() and staged.exists():
            if staged.stat().st_mtime > packaged_sibling.stat().st_mtime:
                return staged
            return packaged_sibling
        if packaged_sibling.exists():
            return packaged_sibling
        if staged.exists():
            return staged
        return packaged_sibling

    def _resolve_default_executable(self) -> Path:
        return self.resolve_default_executable()

    @classmethod
    def _newer_local_dev_overlay_source(cls, executable_path: Path) -> Path | None:
        repo_root = cls._local_dev_repo_root_for_staged_executable(executable_path)
        if repo_root is None:
            return None

        executable_mtime = executable_path.stat().st_mtime
        for source_path in cls._local_dev_overlay_source_paths(repo_root):
            if source_path.stat().st_mtime > executable_mtime:
                return source_path
        return None

    @classmethod
    def _local_dev_repo_root_for_staged_executable(cls, executable_path: Path) -> Path | None:
        if executable_path.name != OVERLAY_EXECUTABLE_NAME:
            return None
        if executable_path.parent.name != "overlay":
            return None
        build_dir = executable_path.parent.parent
        if build_dir.name != "build":
            return None

        repo_root = build_dir.parent
        source_root = repo_root / "native" / "overlay" / "src"
        if not source_root.exists():
            return None
        return repo_root

    @classmethod
    def _local_dev_overlay_source_paths(cls, repo_root: Path) -> tuple[Path, ...]:
        overlay_root = repo_root / "native" / "overlay"
        source_paths: list[Path] = []
        for relative_path in ("Cargo.toml", "Cargo.lock", "build.rs"):
            candidate = overlay_root / relative_path
            if candidate.exists():
                source_paths.append(candidate)

        source_root = overlay_root / "src"
        if source_root.exists():
            source_paths.extend(
                sorted(path for path in source_root.rglob("*.rs") if path.is_file())
            )
        return tuple(source_paths)

    @classmethod
    def bundled_openvr_runtime_dll_path(cls, executable_path: Path) -> Path:
        return executable_path.with_name(OPENVR_RUNTIME_DLL_NAME)

    @classmethod
    def ensure_bundled_openvr_runtime_dll(
        cls,
        executable_path: Path,
    ) -> Path:
        bundled_path = cls.bundled_openvr_runtime_dll_path(executable_path)
        if cls._local_dev_repo_root_for_staged_executable(executable_path) is not None:
            try:
                vendored_bundle = openvr_vendor.validate_vendored_openvr_bundle()
            except (FileNotFoundError, ValueError) as error:
                raise OverlayPreparationError("vendored_openvr_dll_missing", str(error)) from error
            return cls._refresh_staged_openvr_runtime_dll(bundled_path, vendored_bundle)
        return cls._validate_packaged_openvr_runtime_dll(bundled_path)

    @classmethod
    def _refresh_staged_openvr_runtime_dll(
        cls,
        bundled_path: Path,
        vendored_bundle: openvr_vendor.VendoredOpenVrBundle,
    ) -> Path:
        if cls._staged_openvr_runtime_dll_needs_refresh(bundled_path, vendored_bundle):
            bundled_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vendored_bundle.dll_path, bundled_path)
        return openvr_vendor.validate_openvr_runtime_dll(
            bundled_path,
            expected_sha256=vendored_bundle.dll_sha256,
        )

    @classmethod
    def _staged_openvr_runtime_dll_needs_refresh(
        cls,
        bundled_path: Path,
        vendored_bundle: openvr_vendor.VendoredOpenVrBundle,
    ) -> bool:
        if not bundled_path.is_file():
            return True

        try:
            openvr_vendor.validate_openvr_runtime_dll(
                bundled_path,
                expected_sha256=vendored_bundle.dll_sha256,
            )
        except ValueError:
            return True
        return False

    @classmethod
    def _validate_packaged_openvr_runtime_dll(
        cls,
        bundled_path: Path,
    ) -> Path:
        if not bundled_path.is_file():
            raise OverlayPreparationError(
                "packaged_openvr_dll_missing",
                f"Packaged OpenVR runtime DLL not found: {bundled_path}",
            )

        try:
            return openvr_vendor.validate_openvr_runtime_dll(bundled_path)
        except FileNotFoundError as error:
            raise OverlayPreparationError("packaged_openvr_dll_missing", str(error)) from error
        except ValueError as error:
            raise OverlayPreparationError("openvr_dll_hash_mismatch", str(error)) from error


@dataclass(slots=True)
class DesktopFletOverlayRunner:
    frozen: bool | None = None
    python_executable: Path | None = None
    app_executable: Path | None = None
    module_name: str = "puripuly_heart.ui.desktop_overlay"
    task_factory: Any | None = None

    def configure_runtime(
        self,
        *,
        quiet_tail_profile: str,
        handoff_experiment: str,
    ) -> None:
        _ = quiet_tail_profile
        if normalize_handoff_experiment(handoff_experiment) != HANDOFF_EXPERIMENT_OFF:
            raise ValueError("desktop overlay does not support handoff experiments")

    def prepare(self, manifest: OverlayLaunchManifest) -> Path:
        _ = manifest
        return self._launcher_executable()

    def build_command(
        self,
        manifest_path: Path,
        *,
        executable_path: Path | None = None,
    ) -> tuple[str, ...]:
        launcher = executable_path or self._launcher_executable()
        if self._is_frozen():
            return (str(launcher), "run-desktop-overlay", "--config", str(manifest_path))
        return (str(launcher), "-m", self.module_name, "--config", str(manifest_path))

    async def spawn(
        self,
        executable_path: Path,
        manifest_path: Path,
    ) -> OverlayManagedProcess:
        kwargs: dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
        process = await asyncio.create_subprocess_exec(
            *self.build_command(manifest_path, executable_path=executable_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        return _AsyncioOverlayProcess(process=process, task_factory=self.task_factory)

    def _is_frozen(self) -> bool:
        if self.frozen is not None:
            return self.frozen
        return bool(getattr(sys, "frozen", False))

    def _launcher_executable(self) -> Path:
        if self._is_frozen():
            return self.app_executable or Path(sys.executable)
        return self.python_executable or Path(sys.executable)


@dataclass(slots=True)
class OverlayProcessManager:
    process_runner: OverlayProcessRunner = field(default_factory=DefaultOverlayProcessRunner)
    startup_timeout_ms: int = 3000
    bridge_url: str = "ws://127.0.0.1:0"
    bridge_messages: asyncio.Queue[dict[str, object]] | None = None
    session_token: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    locale: str = "en"
    log_dir: str = "logs"
    log_level: str = "INFO"
    logging_mode: str = "basic"
    quiet_tail_profile: str = "p05"
    handoff_experiment: str = HANDOFF_EXPERIMENT_OFF
    renderer_events: asyncio.Queue[dict[str, object]] | None = None
    overlay_instance_id: str = field(default_factory=lambda: f"overlay-{uuid4()}")
    diagnostics_dir: Path = field(default_factory=default_overlay_diagnostics_dir)
    diagnostics: OverlayDiagnosticsRecorder | None = None
    task_factory: Any | None = None
    retry_ownership_changed: Callable[[bool], Awaitable[None]] | None = None
    graceful_shutdown_request: Callable[[], Awaitable[None]] | None = None
    graceful_shutdown_timeout_s: float = 3.0
    selected_target: str | None = None
    fallback_reason: str | None = None
    geometry_authority: str | None = None

    state: str = field(init=False, default="off")
    failure_reason: str | None = field(init=False, default=None)
    restart_scheduled: bool = field(init=False, default=False)
    _manifest_path: Path | None = field(init=False, default=None)
    _process: OverlayManagedProcess | None = field(init=False, default=None)
    _monitor_task: asyncio.Task[None] | None = field(init=False, default=None)
    _current_phase: str = field(init=False, default="off")
    _last_transition: str | None = field(init=False, default=None)
    _last_exit_code: int | None = field(init=False, default=None)
    _executable_path: Path | None = field(init=False, default=None)
    _executable_mtime: float | None = field(init=False, default=None)
    _failure_dumped: bool = field(init=False, default=False)
    _shutdown_requested: bool = field(init=False, default=False)
    _shutdown_request_sent: bool = field(init=False, default=False)
    _shutdown_acknowledged: bool = field(init=False, default=False)
    native_retry_owner_confirmed: bool = field(init=False, default=False)
    restart_refill_ready: bool = field(init=False, default=False)
    _qualified_health_started_at: float | None = field(init=False, default=None, repr=False)
    _last_qualified_health_challenge_id: int | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _accepted_ready_generation: int | None = field(init=False, default=None, repr=False)
    _trace_generation: int = field(init=False, default=0, repr=False)
    _last_trace_phase: str | None = field(init=False, default=None, repr=False)
    _active_process_event_task: asyncio.Task[dict[str, object]] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _active_process_exit_task: asyncio.Task[int | None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _late_spawn_reaper: asyncio.Task[None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _diagnostic_dump_task: asyncio.Task[dict[str, Any]] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _late_spawn_cleanup_failure: str | None = field(init=False, default=None, repr=False)
    _shutdown_graceful_request: str = field(init=False, default="not_attempted", repr=False)
    _shutdown_terminate_requested: bool = field(init=False, default=False, repr=False)
    _shutdown_kill_requested: bool = field(init=False, default=False, repr=False)
    _shutdown_exit_confirmed: bool = field(init=False, default=False, repr=False)
    _shutdown_forced: bool = field(init=False, default=False, repr=False)
    _shutdown_reader_cleanup: str = field(init=False, default="not_observed", repr=False)
    _shutdown_cleanup_succeeded: bool = field(init=False, default=False, repr=False)
    _shutdown_graceful_completed: bool = field(init=False, default=False, repr=False)
    _shutdown_terminal_cause: str | None = field(init=False, default=None, repr=False)
    _shutdown_evidence: deque[dict[str, object]] = field(
        init=False,
        default_factory=lambda: deque(maxlen=_SHUTDOWN_EVIDENCE_LIMIT),
        repr=False,
    )

    def __post_init__(self) -> None:
        self.logging_mode = normalize_overlay_logging_mode(self.logging_mode)
        self.handoff_experiment = normalize_handoff_experiment(self.handoff_experiment)
        if self.diagnostics is None:
            self.diagnostics = OverlayDiagnosticsRecorder(
                overlay_instance_id=self.overlay_instance_id,
                diagnostics_dir=self.diagnostics_dir,
                logging_mode=self.logging_mode,
            )
        else:
            self.diagnostics.set_logging_mode(self.logging_mode)

    def set_logging_mode(self, mode: str) -> None:
        self.logging_mode = normalize_overlay_logging_mode(mode)
        if self.diagnostics is not None:
            self.diagnostics.set_logging_mode(self.logging_mode)
        process = self._process
        if process is not None:
            set_logging_mode = getattr(process, "set_logging_mode", None)
            if callable(set_logging_mode):
                set_logging_mode(self.logging_mode)

    def _set_shutdown_failure(self, cause: str) -> None:
        if self._shutdown_terminal_cause is None:
            self._shutdown_terminal_cause = cause
        if self.failure_reason is None:
            self.failure_reason = cause

    def shutdown_receipt(self) -> dict[str, object]:
        stderr_evidence: list[dict[str, object]] = []
        if self.diagnostics is not None:
            for record in tuple(self.diagnostics.child_stderr_lines)[
                -_SHUTDOWN_STDERR_EVIDENCE_LIMIT:
            ]:
                line = record.get("line")
                if not isinstance(line, str):
                    continue
                encoded = line.encode("utf-8", errors="replace")
                stderr_evidence.append(
                    {
                        "byte_length": len(encoded),
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                    }
                )
        return {
            "graceful_request": self._shutdown_graceful_request,
            "acknowledged": self._shutdown_acknowledged,
            "terminate_requested": self._shutdown_terminate_requested,
            "kill_requested": self._shutdown_kill_requested,
            "forced": self._shutdown_forced,
            "exit_confirmed": self._shutdown_exit_confirmed,
            "exit_code": self._last_exit_code,
            "reader_cleanup": self._shutdown_reader_cleanup,
            "graceful_completed": self._shutdown_graceful_completed,
            "cleanup_succeeded": self._shutdown_cleanup_succeeded,
            "terminal_cause": self._shutdown_terminal_cause,
            "stdout_events": [dict(event) for event in self._shutdown_evidence],
            "stderr_diagnostics": stderr_evidence,
        }

    async def start(self) -> None:
        if self.state in {"starting", "connected"}:
            return
        if self._late_spawn_reaper is not None:
            if not self._late_spawn_reaper.done():
                self.state = "failed"
                self.failure_reason = "termination_unconfirmed"
                return
            await asyncio.gather(self._late_spawn_reaper, return_exceptions=True)
            self._late_spawn_reaper = None
            if self._late_spawn_cleanup_failure is not None:
                self.state = "failed"
                self.failure_reason = self._late_spawn_cleanup_failure
                return

        self.state = "starting"
        self._current_phase = "startup"
        self._last_transition = "spawn"
        self._last_exit_code = None
        self._failure_dumped = False
        self._shutdown_requested = False
        self._shutdown_request_sent = False
        self._shutdown_acknowledged = False
        self._shutdown_graceful_request = "not_attempted"
        self._shutdown_terminate_requested = False
        self._shutdown_kill_requested = False
        self._shutdown_exit_confirmed = False
        self._shutdown_forced = False
        self._shutdown_reader_cleanup = "not_observed"
        self._shutdown_graceful_completed = False
        self._shutdown_cleanup_succeeded = False
        self._shutdown_terminal_cause = None
        self._shutdown_evidence.clear()
        self.restart_scheduled = False
        self.failure_reason = None
        self._accepted_ready_generation = None
        self._qualified_health_started_at = None
        self._last_qualified_health_challenge_id = None
        self._trace_generation += 1
        self._last_trace_phase = None
        await self._set_native_retry_owner_confirmed(False, force_notify=True)

        manifest = self._build_manifest()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout_ms / 1000.0
        spawn_task = asyncio.create_task(
            self._prepare_and_spawn(manifest),
            name="OverlayProcessManager:prepare-and-spawn",
        )
        try:
            remaining = max(0.0, deadline - loop.time())
            executable_path, manifest_path, process = await asyncio.wait_for(
                asyncio.shield(spawn_task),
                timeout=remaining,
            )
            self._executable_path = executable_path
            self._manifest_path = manifest_path
            self._process = process
            self._record_process(
                "process_started",
                pid=getattr(process, "pid", None),
                manifest_path=manifest_path,
            )
            await self._wait_for_startup(deadline)
        except TimeoutError:
            if not spawn_task.done():
                self._late_spawn_reaper = asyncio.create_task(
                    self._reap_late_spawn(spawn_task),
                    name="OverlayProcessManager:late-spawn-reaper",
                )
                await self._fail("startup_timeout", terminate_process=False)
            else:
                await self._fail("startup_timeout")
        except asyncio.CancelledError:
            if not spawn_task.done():
                self._late_spawn_reaper = asyncio.create_task(
                    self._reap_late_spawn(spawn_task),
                    name="OverlayProcessManager:cancelled-spawn-reaper",
                )
            raise
        except OverlayPreparationError as error:
            await self._fail(error.failure_reason)
        except FileNotFoundError:
            await self._fail("missing_executable")
        except ValueError:
            await self._fail("manifest_invalid")
        except OSError:
            await self._fail("spawn_failed")

    async def _prepare_and_spawn(
        self,
        manifest: OverlayLaunchManifest,
    ) -> tuple[Path, Path, OverlayManagedProcess]:
        executable_path = await asyncio.to_thread(self.process_runner.prepare, manifest)
        self._executable_mtime = (
            executable_path.stat().st_mtime if executable_path.exists() else None
        )
        self._record_process(
            "spawn_requested",
            executable_path=executable_path,
            executable_mtime=self._executable_mtime,
            logging_mode=self.logging_mode,
        )
        self.process_runner.configure_runtime(
            quiet_tail_profile=self.quiet_tail_profile,
            handoff_experiment=self.handoff_experiment,
        )
        self._record_process(
            "runtime_configuration",
            quiet_tail_profile=self.quiet_tail_profile,
            handoff_experiment=self.handoff_experiment,
            experiment_only=self.handoff_experiment != HANDOFF_EXPERIMENT_OFF,
        )
        manifest_path = await asyncio.to_thread(self._write_manifest, manifest)
        self._record_process("manifest_written", manifest_path=manifest_path)
        process = await self.process_runner.spawn(executable_path, manifest_path)
        self._attach_process_diagnostics(process)
        return executable_path, manifest_path, process

    async def _reap_late_spawn(
        self,
        spawn_task: asyncio.Task[tuple[Path, Path, OverlayManagedProcess]],
    ) -> None:
        try:
            executable_path, manifest_path, process = await spawn_task
        except Exception:
            return
        self._executable_path = executable_path
        self._manifest_path = manifest_path
        self._process = process
        try:
            graceful_shutdown_complete = False
            if self.graceful_shutdown_request is not None:
                graceful_shutdown_complete = await self._request_graceful_shutdown_before_terminate(
                    process
                )
            if not graceful_shutdown_complete:
                await process.terminate()
            await self._finish_process_readers(process)
            await self._drain_process_events(process)
        except Exception:
            self._late_spawn_cleanup_failure = (
                "shutdown_cleanup_failed"
                if self._shutdown_reader_cleanup == "failed"
                else "termination_unconfirmed"
            )
            self.state = "failed"
            return
        finally:
            self._detach_process_lifecycle_sink(process)
        if self._process is process:
            self._process = None
        self._cleanup_manifest()

    async def stop(self) -> None:
        self.state = "stopping"
        self._current_phase = "stopping"
        self._shutdown_requested = True
        self._shutdown_cleanup_succeeded = False
        self._record_process("stop_requested")

        monitor_task = self._monitor_task
        self._monitor_task = None
        if monitor_task is not None:
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)

        process = self._process
        if process is not None:
            graceful_shutdown_complete = False
            if self.graceful_shutdown_request is not None:
                graceful_shutdown_complete = await self._request_graceful_shutdown_before_terminate(
                    process,
                    request_already_sent=self._shutdown_request_sent,
                )
            else:
                self._shutdown_graceful_request = "unavailable"
            if (
                not graceful_shutdown_complete
                and getattr(process, "returncode", None) is None
                and self._last_exit_code is None
            ):
                self._shutdown_terminate_requested = True
                self._shutdown_forced = True
                self._record_process("terminate_requested", pid=getattr(process, "pid", None))
                try:
                    await process.terminate()
                except Exception:
                    returncode = getattr(process, "returncode", None)
                    if isinstance(returncode, int):
                        self._last_exit_code = returncode
                        self._shutdown_exit_confirmed = True
                        self._shutdown_reader_cleanup = "failed"
                        cause = "shutdown_cleanup_failed"
                    else:
                        cause = "termination_unconfirmed"
                    self._set_shutdown_failure(cause)
                    self.restart_scheduled = False
                    self._record_process(
                        cause,
                        pid=getattr(process, "pid", None),
                        accepted=False,
                    )
                    raise
            returncode = getattr(process, "returncode", None)
            if isinstance(returncode, int):
                self._last_exit_code = returncode
                self._shutdown_exit_confirmed = True
            try:
                await self._finish_process_readers(process)
            except Exception:
                self.state = "failed"
                self._current_phase = "failed"
                self.restart_scheduled = False
                raise
            await self._drain_process_events(process)
            self._record_process(
                "process_exited",
                pid=getattr(process, "pid", None),
                returncode=returncode,
            )
            if not self._shutdown_exit_confirmed:
                self._set_shutdown_failure("termination_unconfirmed")
            elif self._shutdown_forced:
                self._set_shutdown_failure("shutdown_forced")
            elif self._last_exit_code != 0:
                self._set_shutdown_failure("runtime_exit_nonzero")
            elif (
                self.graceful_shutdown_request is not None and not self._shutdown_graceful_completed
            ):
                self._set_shutdown_failure("shutdown_not_acknowledged")
            self._detach_process_lifecycle_sink(process)
            if self._process is process:
                self._process = None
        await self._set_native_retry_owner_confirmed(False)

        try:
            self._cleanup_manifest()
        except Exception:
            self._set_shutdown_failure("shutdown_cleanup_failed")
            self.state = "failed"
            self._current_phase = "failed"
            self.restart_scheduled = False
            raise
        if self._shutdown_terminal_cause is None:
            self.state = "off"
            self._current_phase = "off"
            self._shutdown_cleanup_succeeded = True
        else:
            self.state = "failed"
            self._current_phase = "failed"
            self.restart_scheduled = False
        self._shutdown_requested = False
        self._shutdown_request_sent = False

    def mark_shutdown_requested(self, *, request_sent: bool = True) -> None:
        self._shutdown_requested = True
        self._shutdown_request_sent = request_sent
        self._record_process(
            "shutdown_requested",
            phase=self._current_phase,
            request_sent=request_sent,
        )

    def _build_manifest(self) -> OverlayLaunchManifest:
        return OverlayLaunchManifest(
            contract_version=OVERLAY_CONTRACT_VERSION,
            app_version=__version__,
            overlay_instance_id=self.overlay_instance_id,
            bridge_url=self.bridge_url,
            session_token=self.session_token,
            parent_pid=os.getpid(),
            startup_deadline_ms=self.startup_timeout_ms,
            log_dir=self.log_dir,
            log_level=self.log_level,
            locale=self.locale,
            logging_mode=self.logging_mode,
        )

    def _write_manifest(self, manifest: OverlayLaunchManifest) -> Path:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".json",
            prefix="puripuly-overlay-",
            delete=False,
        ) as handle:
            json.dump(manifest.to_dict(), handle)
        return Path(handle.name)

    async def _wait_for_startup(self, deadline: float | None = None) -> None:
        if self._process is None:
            await self._fail("unknown")
            return

        event_task = self._create_task(
            self._process.next_event(),
            task_name="startup-next-event",
        )
        bridge_task = self._create_bridge_event_task()
        exit_task = self._create_task(
            self._process.wait_for_exit(),
            task_name="startup-process-wait",
        )
        self._active_process_event_task = event_task
        self._active_process_exit_task = exit_task
        remaining = (
            self.startup_timeout_ms / 1000.0
            if deadline is None
            else max(0.0, deadline - asyncio.get_running_loop().time())
        )
        timeout_task = self._create_task(
            asyncio.sleep(remaining),
            task_name="startup-timeout",
        )

        try:
            while True:
                pending_tasks: set[asyncio.Task[object]] = {exit_task, timeout_task}
                pending_tasks.add(event_task)
                if bridge_task is not None:
                    pending_tasks.add(bridge_task)
                done, _pending = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if event_task in done:
                    outcome = await self._handle_lifecycle_event(
                        event_task.result(), allow_ready=True, trusted_process_event=True
                    )
                    if outcome == "ready":
                        self._current_phase = "connected"
                        self._last_transition = "overlay_ready"
                        handoff_exit_task = exit_task
                        exit_task = None
                        self._monitor_task = self._create_task(
                            self._monitor_connected_process(exit_task=handoff_exit_task),
                            task_name="connected-process-monitor",
                        )
                        await asyncio.sleep(0)
                        if handoff_exit_task.done() and self._monitor_task is not None:
                            await asyncio.shield(self._monitor_task)
                        return
                    if outcome == "failed":
                        return
                    event_task = self._create_task(
                        self._process.next_event(),
                        task_name="startup-next-event",
                    )
                    self._active_process_event_task = event_task

                if bridge_task is not None and bridge_task in done:
                    outcome = await self._handle_lifecycle_event(
                        bridge_task.result(),
                        allow_ready=True,
                        trusted_process_event=False,
                    )
                    if outcome == "ready":
                        self._current_phase = "connected"
                        self._last_transition = "bridge_ready"
                        handoff_exit_task = exit_task
                        exit_task = None
                        self._monitor_task = self._create_task(
                            self._monitor_connected_process(exit_task=handoff_exit_task),
                            task_name="connected-process-monitor",
                        )
                        await asyncio.sleep(0)
                        if handoff_exit_task.done() and self._monitor_task is not None:
                            await asyncio.shield(self._monitor_task)
                        return
                    if outcome == "failed":
                        return
                    bridge_task = self._create_bridge_event_task()

                if exit_task in done:
                    exit_code = exit_task.result()
                    self._last_exit_code = exit_code
                    self._record_process("process_exited", phase="startup", exit_code=exit_code)
                    await self._fail(
                        self._map_exit_code_to_failure_reason(exit_code),
                        terminate_process=False,
                    )
                    return

                if timeout_task in done:
                    await self._fail("startup_timeout")
                    return
        finally:
            for task in (event_task, bridge_task, exit_task, timeout_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *[
                    task
                    for task in (event_task, bridge_task, exit_task, timeout_task)
                    if task is not None
                ],
                return_exceptions=True,
            )
            if self._active_process_event_task is event_task:
                self._active_process_event_task = None
            if self._active_process_exit_task is exit_task:
                self._active_process_exit_task = None

    async def _monitor_connected_process(
        self,
        exit_task: asyncio.Task[int | None] | None = None,
    ) -> None:
        process = self._process
        if process is None:
            return
        event_task = self._create_task(
            process.next_event(),
            task_name="connected-next-event",
        )
        bridge_task = self._create_bridge_event_task()
        if exit_task is None:
            exit_task = self._create_task(
                process.wait_for_exit(),
                task_name="connected-process-wait",
            )
        self._active_process_event_task = event_task
        self._active_process_exit_task = exit_task
        try:
            while True:
                pending_tasks: set[asyncio.Task[object]] = {exit_task}
                pending_tasks.add(event_task)
                if bridge_task is not None:
                    pending_tasks.add(bridge_task)
                done, _pending = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if event_task in done:
                    if (
                        await self._handle_lifecycle_event(
                            event_task.result(),
                            allow_ready=False,
                            trusted_process_event=True,
                        )
                        == "failed"
                    ):
                        return
                    event_task = self._create_task(
                        process.next_event(),
                        task_name="connected-next-event",
                    )
                    self._active_process_event_task = event_task

                if bridge_task is not None and bridge_task in done:
                    if (
                        await self._handle_lifecycle_event(
                            bridge_task.result(),
                            allow_ready=False,
                            trusted_process_event=False,
                        )
                        == "failed"
                    ):
                        return
                    bridge_task = self._create_bridge_event_task()

                if exit_task in done:
                    exit_code = exit_task.result()
                    self._last_exit_code = exit_code
                    self._record_process("process_exited", phase="connected", exit_code=exit_code)
                    await self._reconcile_terminal_process_events(process, event_task)
                    if self.state == "connected" and exit_code is not None:
                        if self._shutdown_requested and exit_code == 0:
                            try:
                                await self._finish_process_readers(process)
                            except Exception:
                                self.restart_scheduled = False
                                self.state = "failed"
                                self._current_phase = "failed"
                                return
                            await self._drain_process_events(process)
                            self._detach_process_lifecycle_sink(process)
                            self._process = None
                            self._cleanup_manifest()
                            self.state = "stopping"
                            self._current_phase = "expected_shutdown"
                        else:
                            await self._fail("runtime_crashed", terminate_process=False)
                    return
        finally:
            if self._active_process_event_task is event_task:
                await self._reconcile_terminal_process_events(process, event_task)
                self._active_process_event_task = None
            for task in (bridge_task, exit_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *[task for task in (bridge_task, exit_task) if task is not None],
                return_exceptions=True,
            )
            if self._active_process_exit_task is exit_task:
                self._active_process_exit_task = None

    def _create_bridge_event_task(self) -> asyncio.Task[dict[str, object]] | None:
        if self.bridge_messages is None:
            return None
        return self._create_task(
            self.bridge_messages.get(),
            task_name="bridge-message",
        )

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        task_name: str,
    ) -> asyncio.Task[Any]:
        if self.task_factory is not None:
            return self.task_factory(coroutine, task_name=task_name)
        return asyncio.create_task(coroutine, name=f"OverlayProcessManager:{task_name}")

    @staticmethod
    def _create_cleanup_task(
        coroutine: Coroutine[Any, Any, Any],
        *,
        task_name: str,
    ) -> asyncio.Task[Any]:
        return asyncio.create_task(coroutine, name=f"OverlayProcessManager:{task_name}")

    async def _handle_lifecycle_event(
        self,
        event: object,
        *,
        allow_ready: bool,
        trusted_process_event: bool = True,
    ) -> str:
        if not isinstance(event, dict):
            self._record_process(
                "renderer_message_ignored",
                reason="malformed_message",
                accepted=False,
            )
            logger.warning(
                "[OverlayProcess] Ignoring malformed renderer message with type: %s",
                type(event).__name__,
            )
            return "ignored"

        event_type = str(event.get("type", ""))
        self._record_process(
            "lifecycle_event",
            phase=self._current_phase,
            event_type=event_type,
            failure_reason=event.get("failure_reason"),
            startup_phase=event.get("startup_phase"),
        )
        if event_type == "overlay_trace":
            component = event.get("component")
            trace_event = event.get("event")
            generation = event.get("generation")
            monotonic_ms = event.get("monotonic_ms")
            if (
                isinstance(component, str)
                and isinstance(trace_event, str)
                and (generation is None or self._is_non_negative_int(generation))
                and (monotonic_ms is None or self._is_finite_non_bool_number(monotonic_ms))
            ):
                phase = event.get("phase")
                if isinstance(phase, str):
                    self._last_trace_phase = phase
                self.record_lifecycle_trace(
                    component,
                    trace_event,
                    generation=generation,
                    monotonic_ms=monotonic_ms,
                    phase=phase,
                    accepted=event.get("accepted"),
                    event_generation=event.get("event_generation"),
                    geometry_authority=event.get("geometry_authority"),
                    canonical_bounds=event.get("canonical_bounds"),
                    observed_bounds=event.get("observed_bounds"),
                    pid=event.get("pid"),
                    parent_pid=event.get("parent_pid"),
                    endpoint_identity=event.get("endpoint_identity"),
                    returncode=event.get("returncode"),
                    kill_on_job_close=event.get("kill_on_job_close"),
                    job_failure_reason=event.get("job_failure_reason", event.get("reason")),
                    failure_phase=event.get("failure_phase", event.get("startup_phase")),
                    timeout_phase=event.get("timeout_phase"),
                )
            else:
                self._record_process(
                    "renderer_message_ignored",
                    reason="invalid_overlay_trace",
                    accepted=False,
                )
            return "ignored"
        if event_type == "shutdown_complete":
            if not trusted_process_event:
                self._record_process(
                    "renderer_message_ignored",
                    reason="untrusted_shutdown_complete",
                    accepted=False,
                )
                return "ignored"
            event_instance_id = event.get("overlay_instance_id")
            if event_instance_id != self.overlay_instance_id:
                self._record_process(
                    "renderer_message_ignored",
                    reason="stale_overlay_instance",
                    event_overlay_instance_id=event_instance_id,
                    accepted=False,
                )
                return "ignored"
            if not self._shutdown_acknowledged:
                self._shutdown_acknowledged = True
                self._record_process("graceful_shutdown_acknowledged")
            return "ignored"
        if allow_ready and trusted_process_event and event_type == "overlay_ready":
            event_instance_id = event.get("overlay_instance_id")
            if event_instance_id != self.overlay_instance_id:
                self._record_process(
                    "renderer_message_ignored",
                    reason="stale_overlay_instance",
                    event_overlay_instance_id=event_instance_id,
                    generation=event.get("generation"),
                    accepted=False,
                )
                return "ignored"
            if event.get("runtime_generation") != 1:
                await self._fail("unsupported_binary")
                return "failed"
            capabilities = event.get("capabilities")
            if (
                not isinstance(capabilities, dict)
                or capabilities.get("execution_contract") != OVERLAY_EXECUTION_CONTRACT
                or (
                    self.selected_target != "desktop"
                    and not self._supports_native_retry_ownership(event)
                )
            ):
                await self._fail("unsupported_binary")
                return "failed"
            ready_generation = event.get("generation")
            if ready_generation is not None and not self._is_positive_int(ready_generation):
                self._record_process(
                    "renderer_message_ignored",
                    reason="invalid_ready_generation",
                    accepted=False,
                )
                return "ignored"
            if ready_generation is not None and self._accepted_ready_generation is not None:
                self._record_process(
                    "renderer_message_ignored",
                    reason="duplicate_ready_generation",
                    generation=ready_generation,
                    accepted=False,
                )
                return "ignored"
            if isinstance(ready_generation, int):
                self._accepted_ready_generation = ready_generation
            await self._set_native_retry_owner_confirmed(self.selected_target != "desktop")
            self.state = "connected"
            self.failure_reason = None
            logger.info(
                "[OverlayProcess] Ready: overlay_instance_id=%s phase=%s manifest_path=%s",
                self.overlay_instance_id,
                self._current_phase,
                self._manifest_path,
            )
            return "ready"
        if event_type == "owner_status":
            if (
                event.get("overlay_instance_id") != self.overlay_instance_id
                or event.get("runtime_generation") != 1
            ):
                return "ignored"
            challenge_id = event.get("health_challenge_id")
            validated = event.get("health_challenge_validated") is True
            if (
                not validated
                or not isinstance(challenge_id, int)
                or isinstance(challenge_id, bool)
                or (
                    self._last_qualified_health_challenge_id is not None
                    and challenge_id <= self._last_qualified_health_challenge_id
                )
            ):
                return "ignored"
            self._last_qualified_health_challenge_id = challenge_id
            current_covered = (
                event.get("current_covered_handoff") is True and event.get("lease_valid") is True
            )
            observed_requested_hide = (
                event.get("confirmed_hide") is True
                and event.get("desired_visible") is False
                and event.get("observed_runtime_visible") is False
            )
            due_elapsed = event.get("due_elapsed_ms")
            healthy = (
                self.handoff_experiment == HANDOFF_EXPERIMENT_OFF
                and event.get("classification")
                in {"healthy_idle", "intentional_hidden", "no_drawable_content"}
                and isinstance(due_elapsed, (int, float))
                and not isinstance(due_elapsed, bool)
                and due_elapsed == 0
                and (current_covered or observed_requested_hide)
            )
            now = asyncio.get_running_loop().time()
            if healthy:
                if self._qualified_health_started_at is None:
                    self._qualified_health_started_at = now
                elif now - self._qualified_health_started_at >= 60.0:
                    self.restart_refill_ready = True
            else:
                self._qualified_health_started_at = None
            return "ignored"
        if event_type in {"startup_error", "runtime_error"}:
            startup_phase = event.get("startup_phase")
            if isinstance(startup_phase, str):
                self._last_trace_phase = startup_phase
            await self._fail(self._extract_failure_reason(event))
            return "failed"
        if event_type == "overlay_event":
            self._handle_renderer_event(event)
        return "ignored"

    def _supports_native_retry_ownership(self, event: dict[str, object]) -> bool:
        capabilities = event.get("capabilities")
        if not isinstance(capabilities, dict):
            return False
        capability = capabilities.get("native_presentation_retry")
        if not isinstance(capability, dict) or set(capability) != {"version", "ownership"}:
            return False
        version = capability.get("version")
        ownership = capability.get("ownership")
        return type(version) is int and version == 1 and ownership == "exclusive"

    async def _set_native_retry_owner_confirmed(
        self,
        confirmed: bool,
        *,
        force_notify: bool = False,
    ) -> None:
        confirmed = bool(confirmed)
        if confirmed == self.native_retry_owner_confirmed and not force_notify:
            return
        self.native_retry_owner_confirmed = confirmed
        if self.retry_ownership_changed is not None:
            await self.retry_ownership_changed(confirmed)

    def _handle_renderer_event(self, event: dict[str, object]) -> None:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            self._record_process(
                "renderer_message_ignored",
                event_type="overlay_event",
                reason="invalid_payload",
                accepted=False,
            )
            logger.warning("[OverlayProcess] Ignoring overlay_event without object payload")
            return

        renderer_event_type = payload.get("event")
        if renderer_event_type == "window_bounds_changed" and "generation" in payload:
            event_generation = payload.get("generation")
            if (
                self._accepted_ready_generation is None
                or event_generation != self._accepted_ready_generation
            ):
                self._record_process(
                    "renderer_event_dropped",
                    renderer_event=renderer_event_type,
                    reason="stale_generation",
                    generation=event_generation,
                    accepted=False,
                )
                return
        if not self._is_valid_renderer_event_payload(payload):
            self._record_process(
                "renderer_message_ignored",
                event_type="overlay_event",
                renderer_event=renderer_event_type,
                reason="invalid_payload",
                accepted=False,
            )
            logger.warning(
                "[OverlayProcess] Ignoring invalid renderer event: %r",
                renderer_event_type,
            )
            return

        if self.renderer_events is None:
            self._record_process(
                "renderer_event_diagnostic_only",
                renderer_event=renderer_event_type,
            )
            logger.info(
                "[OverlayProcess] Renderer event ignored without controller queue: %s",
                renderer_event_type,
            )
            return

        if renderer_event_type == "window_bounds_changed" and "bounds_epoch" in payload:
            payload = dict(payload)
            payload.pop("bounds_epoch", None)
            event = {**event, "payload": payload}

        try:
            self.renderer_events.put_nowait(event)
        except asyncio.QueueFull:
            self._record_process(
                "renderer_event_dropped",
                renderer_event=renderer_event_type,
                accepted=False,
            )
            logger.warning(
                "[OverlayProcess] Dropping renderer event because controller queue is full: %s",
                renderer_event_type,
            )

    def _is_valid_renderer_event_payload(self, payload: dict[object, object]) -> bool:
        renderer_event_type = payload.get("event")
        if renderer_event_type == "window_bounds_changed":
            return self._is_valid_window_bounds_changed_payload(payload)
        if renderer_event_type == "interaction_mode_changed":
            return self._is_valid_interaction_mode_changed_payload(payload)
        if renderer_event_type == "reset_to_bottom_center_requested":
            return self._is_valid_reset_to_bottom_center_requested_payload(payload)
        return False

    def _is_valid_window_bounds_changed_payload(self, payload: dict[object, object]) -> bool:
        keys = set(payload)
        if not _WINDOW_BOUNDS_EVENT_KEYS.issubset(keys):
            return False
        if keys - _WINDOW_BOUNDS_EVENT_KEYS - _WINDOW_BOUNDS_EVENT_OPTIONAL_KEYS:
            return False
        source = payload.get("source")
        persist = payload.get("persist")
        if not isinstance(source, str) or source not in _WINDOW_BOUNDS_EVENT_PERSIST_RULES:
            return False
        if not isinstance(persist, bool):
            return False
        if persist is not _WINDOW_BOUNDS_EVENT_PERSIST_RULES[source]:
            return False
        if "bounds_epoch" in payload and not self._is_non_negative_int(payload.get("bounds_epoch")):
            return False
        if "generation" in payload and not self._is_positive_int(payload.get("generation")):
            return False
        return (
            self._is_finite_non_bool_number(payload.get("x"))
            and self._is_finite_non_bool_number(payload.get("y"))
            and self._is_number_at_least(payload.get("width"), _MIN_DESKTOP_WINDOW_WIDTH)
            and self._is_number_at_least(payload.get("height"), _MIN_DESKTOP_WINDOW_HEIGHT)
        )

    def _is_valid_interaction_mode_changed_payload(self, payload: dict[object, object]) -> bool:
        mode = payload.get("mode")
        return (
            set(payload) == _INTERACTION_MODE_EVENT_KEYS
            and isinstance(mode, str)
            and mode in _INTERACTION_MODE_EVENT_MODES
        )

    def _is_valid_reset_to_bottom_center_requested_payload(
        self,
        payload: dict[object, object],
    ) -> bool:
        return (
            set(payload) == _RESET_TO_BOTTOM_CENTER_EVENT_KEYS
            and payload.get("event") == "reset_to_bottom_center_requested"
        )

    @staticmethod
    def _is_finite_non_bool_number(value: object) -> bool:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )

    @staticmethod
    def _is_non_negative_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    @staticmethod
    def _is_positive_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @classmethod
    def _is_number_at_least(cls, value: object, minimum: int) -> bool:
        return cls._is_finite_non_bool_number(value) and value >= minimum

    def _extract_failure_reason(self, event: dict[str, object]) -> str:
        failure_reason = event.get("failure_reason")
        if isinstance(failure_reason, str) and failure_reason:
            return failure_reason
        return "unknown"

    def _map_exit_code_to_failure_reason(self, exit_code: int | None) -> str:
        if exit_code is None:
            return "unknown"
        return _EXIT_CODE_TO_FAILURE_REASON.get(exit_code, "unknown")

    async def _request_graceful_shutdown_before_terminate(
        self,
        process: OverlayManagedProcess,
        *,
        request_already_sent: bool = False,
    ) -> bool:
        request = self.graceful_shutdown_request
        if request is None:
            self._shutdown_graceful_request = "unavailable"
            self._record_process("graceful_shutdown_unavailable")
            return False

        timeout_s = max(0.0, float(self.graceful_shutdown_timeout_s))
        if timeout_s <= 0.0:
            self._shutdown_graceful_request = (
                "already_sent" if request_already_sent else "not_attempted"
            )
            self._record_process("graceful_shutdown_timeout", acknowledged=False)
            return False

        active_event_task = self._active_process_event_task
        if active_event_task is not None:
            await self._reconcile_terminal_process_events(process, active_event_task)
        if self._active_process_event_task is active_event_task:
            self._active_process_event_task = None

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        request_failed = False
        if request_already_sent:
            self._shutdown_graceful_request = "already_sent"
        else:
            try:
                await asyncio.wait_for(request(), timeout=timeout_s)
                self._shutdown_request_sent = True
                self._shutdown_graceful_request = "sent"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                request_failed = True
                self._shutdown_graceful_request = "failed"
                self._record_process(
                    "graceful_shutdown_request_failed",
                    exception_type=type(exc).__name__,
                )

        self._record_process(
            "graceful_close_requested",
            request_already_sent=request_already_sent,
            request_failed=request_failed,
        )
        ack_task: asyncio.Task[dict[str, object]] | None = None
        if not self._shutdown_acknowledged:
            ack_task = self._create_cleanup_task(
                process.next_event(),
                task_name="graceful-shutdown-ack",
            )
        exit_task = self._active_process_exit_task
        owns_exit_task = exit_task is None
        if exit_task is None:
            exit_task = self._create_cleanup_task(
                process.wait_for_exit(),
                task_name="graceful-shutdown-process-wait",
            )
            self._active_process_exit_task = exit_task
        acknowledged = self._shutdown_acknowledged
        process_exited = self._process_exit_confirmed(process, exit_task)
        try:
            while True:
                if acknowledged and process_exited:
                    self._last_exit_code = self._process_exit_code(process, exit_task)
                    self._record_process(
                        "graceful_shutdown_process_exit",
                        exit_code=self._last_exit_code,
                    )
                    self._shutdown_graceful_completed = True
                    return True
                remaining_s = deadline - loop.time()
                if remaining_s <= 0.0:
                    break
                wait_tasks: set[asyncio.Task[object]] = set()
                if not process_exited:
                    wait_tasks.add(exit_task)
                if ack_task is not None:
                    wait_tasks.add(ack_task)
                if not wait_tasks:
                    break
                done, _pending = await asyncio.wait(
                    wait_tasks,
                    timeout=remaining_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    break
                if ack_task is not None and ack_task in done:
                    try:
                        event = ack_task.result()
                    except Exception:
                        break
                    await self._record_shutdown_lifecycle_event(event)
                    acknowledged = self._shutdown_acknowledged
                    ack_task = None
                    if not acknowledged:
                        ack_task = self._create_cleanup_task(
                            process.next_event(),
                            task_name="graceful-shutdown-ack",
                        )
                if exit_task in done:
                    if not exit_task.cancelled():
                        exit_task.result()
                    process_exited = self._process_exit_confirmed(process, exit_task)
                    if process_exited:
                        self._last_exit_code = self._process_exit_code(process, exit_task)
                        if ack_task is not None:
                            await self._reconcile_terminal_process_events(process, ack_task)
                            ack_task = None
                        else:
                            await self._drain_process_events(process)
                        acknowledged = self._shutdown_acknowledged
                        if acknowledged:
                            self._record_process(
                                "graceful_shutdown_process_exit",
                                exit_code=self._last_exit_code,
                            )
                            self._shutdown_graceful_completed = True
                            return True
                        if ack_task is None:
                            ack_task = self._create_cleanup_task(
                                process.next_event(),
                                task_name="graceful-shutdown-ack",
                            )
                        continue
                    break
        finally:
            process_exited = process_exited or self._process_exit_confirmed(process, exit_task)
            cleanup_tasks: list[asyncio.Task[object]] = []
            if ack_task is not None:
                if not ack_task.done():
                    ack_task.cancel()
                cleanup_tasks.append(ack_task)
            if owns_exit_task:
                if not exit_task.done():
                    exit_task.cancel()
                cleanup_tasks.append(exit_task)
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            if owns_exit_task and self._active_process_exit_task is exit_task:
                self._active_process_exit_task = None

        self._record_process(
            "graceful_shutdown_timeout",
            acknowledged=self._shutdown_acknowledged,
            process_exited=process_exited,
        )
        return False

    @staticmethod
    def _process_exit_confirmed(
        process: OverlayManagedProcess,
        exit_task: asyncio.Task[int | None],
    ) -> bool:
        _ = process
        if not exit_task.done() or exit_task.cancelled():
            return False
        try:
            return exit_task.exception() is None and exit_task.result() is not None
        except (asyncio.CancelledError, Exception):
            return False

    @staticmethod
    def _process_exit_code(
        process: OverlayManagedProcess,
        exit_task: asyncio.Task[int | None],
    ) -> int | None:
        returncode = getattr(process, "returncode", None)
        if isinstance(returncode, int):
            return returncode
        if not exit_task.done() or exit_task.cancelled():
            return None
        try:
            result = exit_task.result()
        except (asyncio.CancelledError, Exception):
            return None
        return result if isinstance(result, int) else None

    async def _fail(
        self,
        failure_reason: str,
        *,
        cleanup_manifest: bool = True,
        terminate_process: bool = True,
    ) -> None:
        self.state = "failing"
        self._set_shutdown_failure(failure_reason)
        failure_reason = self.failure_reason or failure_reason
        connected_session = self._last_transition in {"overlay_ready", "bridge_ready"}
        self.restart_scheduled = connected_session and not self._shutdown_requested
        self._current_phase = "failed"
        stdout_count = (
            len(self.diagnostics.child_stdout_lines) if self.diagnostics is not None else 0
        )
        stderr_count = (
            len(self.diagnostics.child_stderr_lines) if self.diagnostics is not None else 0
        )
        self._record_process(
            "failure",
            failure_reason=failure_reason,
            failure_phase=self._last_trace_phase or self._current_phase,
            phase=(
                "connected"
                if self._last_transition in {"overlay_ready", "bridge_ready"}
                else "startup"
            ),
            exit_code=self._last_exit_code,
            stdout_count=stdout_count,
            stderr_count=stderr_count,
        )
        logger.error(
            "[OverlayProcess] Failure: overlay_instance_id=%s phase=%s failure_reason=%s exit_code=%s last_transition=%s stdout_lines=%s stderr_lines=%s",
            self.overlay_instance_id,
            (
                "connected"
                if self._last_transition in {"overlay_ready", "bridge_ready"}
                else "startup"
            ),
            failure_reason,
            self._last_exit_code,
            self._last_transition,
            stdout_count,
            stderr_count,
        )

        if self.diagnostics is not None and not self._failure_dumped:
            self._diagnostic_dump_task = asyncio.create_task(
                self.diagnostics.dump_evidence(
                    outcome="failure",
                    failure_reason=failure_reason,
                    phase=(
                        "connected"
                        if self._last_transition in {"overlay_ready", "bridge_ready"}
                        else "startup"
                    ),
                    exit_code=self._last_exit_code,
                    manager_state="failed",
                    last_transition=self._last_transition,
                    manifest_path=self._manifest_path,
                    executable_path=self._executable_path,
                    executable_mtime=self._executable_mtime,
                    stdout_count=stdout_count,
                    stderr_count=stderr_count,
                ),
                name="OverlayProcessManager:diagnostic-dump",
            )
            self._failure_dumped = True

        process = self._process
        if terminate_process and process is not None:
            graceful_shutdown_complete = await self._request_graceful_shutdown_before_terminate(
                process
            )
            if (
                not graceful_shutdown_complete
                and getattr(process, "returncode", None) is None
                and self._last_exit_code is None
            ):
                self._shutdown_terminate_requested = True
                self._shutdown_forced = True
                self._record_process("terminate_requested", pid=getattr(process, "pid", None))
                try:
                    await process.terminate()
                except Exception:
                    self.restart_scheduled = False
                    self._record_process(
                        "termination_unconfirmed",
                        pid=getattr(process, "pid", None),
                        accepted=False,
                    )
                    self.state = "failed"
                    return
            try:
                await self._finish_process_readers(process)
            except Exception:
                self.restart_scheduled = False
                self.state = "failed"
                return
            returncode = getattr(process, "returncode", None)
            if isinstance(returncode, int):
                self._last_exit_code = returncode
                self._shutdown_exit_confirmed = True
            await self._drain_process_events(process)
            self._record_process(
                "process_exited",
                pid=getattr(process, "pid", None),
                returncode=getattr(process, "returncode", None),
            )
            self._detach_process_lifecycle_sink(process)
            if self._process is process:
                self._process = None
        elif not terminate_process:
            if process is not None:
                try:
                    await self._finish_process_readers(process)
                except Exception:
                    self.restart_scheduled = False
                    self.state = "failed"
                    return
                await self._drain_process_events(process)
                self._detach_process_lifecycle_sink(process)
            self._process = None
        await self._set_native_retry_owner_confirmed(False)

        if cleanup_manifest:
            self._cleanup_manifest()
        self.state = "failed"

    def _cleanup_manifest(self) -> None:
        manifest_path = self._manifest_path
        if manifest_path is None:
            return
        try:
            manifest_path.unlink()
        except FileNotFoundError:
            pass
        if self._manifest_path is manifest_path:
            self._manifest_path = None

    def _attach_process_diagnostics(self, process: OverlayManagedProcess) -> None:
        attach = getattr(process, "attach_diagnostics", None)
        if callable(attach) and self.diagnostics is not None:
            attach(self.diagnostics, overlay_instance_id=self.overlay_instance_id)
        attach_lifecycle_sink = getattr(process, "attach_lifecycle_sink", None)
        if callable(attach_lifecycle_sink):
            attach_lifecycle_sink(self._record_managed_process_lifecycle)
        set_logging_mode = getattr(process, "set_logging_mode", None)
        if callable(set_logging_mode):
            set_logging_mode(self.logging_mode)

    def _record_managed_process_lifecycle(
        self,
        lifecycle_event: str,
        fields: dict[str, object],
    ) -> None:
        if lifecycle_event == "kill_requested":
            self._shutdown_kill_requested = True
            self._shutdown_forced = True
        elif lifecycle_event == "process_readers_cancelled":
            self._shutdown_reader_cleanup = "cancelled"
        elif lifecycle_event == "process_readers_finished":
            self._shutdown_reader_cleanup = "complete"
        elif lifecycle_event == "process_reader_cleanup_failed":
            self._shutdown_reader_cleanup = "failed"
            self._set_shutdown_failure("shutdown_cleanup_failed")
        elif lifecycle_event == "termination_unconfirmed":
            self._set_shutdown_failure("termination_unconfirmed")
        record_fields = dict(fields)
        if "event" in record_fields:
            record_fields["managed_event"] = record_fields.pop("event")
        self._record_process(lifecycle_event, **record_fields)

    @staticmethod
    def _detach_process_lifecycle_sink(process: OverlayManagedProcess) -> None:
        attach_lifecycle_sink = getattr(process, "attach_lifecycle_sink", None)
        if callable(attach_lifecycle_sink):
            attach_lifecycle_sink(None)

    async def _reconcile_terminal_process_events(
        self,
        process: OverlayManagedProcess,
        event_task: asyncio.Task[dict[str, object]],
    ) -> None:
        if not event_task.done():
            event_task.cancel()
        results = await asyncio.gather(event_task, return_exceptions=True)
        event = results[0]
        if not isinstance(event, BaseException):
            await self._record_shutdown_lifecycle_event(event)
        await self._drain_process_events(process)

    async def _finish_process_readers(self, process: OverlayManagedProcess) -> None:
        try:
            await process.finish_readers()
        except Exception:
            self._shutdown_reader_cleanup = "failed"
            self._set_shutdown_failure("shutdown_cleanup_failed")
            raise
        if self._shutdown_reader_cleanup != "failed":
            self._shutdown_reader_cleanup = "complete"

    async def _drain_process_events(self, process: OverlayManagedProcess) -> None:
        drain_events = getattr(process, "drain_events", None)
        if not callable(drain_events):
            return
        for event in drain_events():
            await self._record_shutdown_lifecycle_event(event)

    async def _record_shutdown_lifecycle_event(self, event: object) -> None:
        if isinstance(event, dict):
            safe_event = {
                key: event[key]
                for key in (
                    "type",
                    "failure_reason",
                    "startup_phase",
                    "classification",
                    "stage",
                    "cleanup_failure_reason",
                    "exit_code",
                    "runtime_generation",
                    "overlay_instance_id",
                )
                if key in event
                and (isinstance(event[key], (str, int, float, bool)) or event[key] is None)
            }
            self._shutdown_evidence.append(safe_event)
            event_type = str(event.get("type", ""))
            if event_type in {"overlay_trace", "shutdown_complete"}:
                await self._handle_lifecycle_event(event, allow_ready=False)
                return
            if event_type in {"startup_error", "runtime_error"}:
                startup_phase = event.get("startup_phase")
                if isinstance(startup_phase, str):
                    self._last_trace_phase = startup_phase
                cause = self._extract_failure_reason(event)
                self._set_shutdown_failure(cause)
                self._record_process(
                    "lifecycle_event",
                    event_type=event_type,
                    failure_reason=cause,
                    accepted=True,
                    reason="shutdown_terminal",
                )
                return
        else:
            event_type = ""
        self._record_process(
            "lifecycle_event",
            event_type=event_type,
            accepted=False,
            reason="shutdown_drain",
        )

    def record_lifecycle_trace(
        self,
        component: str,
        event: str,
        **fields: object,
    ) -> None:
        self._record_process(
            "overlay_trace",
            trace_component=component,
            trace_event=event,
            **fields,
        )

    def _record_process(self, event: str, **fields: object) -> None:
        if self.diagnostics is not None:
            if fields.get("generation") is None:
                fields["generation"] = self._trace_generation
            if fields.get("selected_target") is None:
                fields["selected_target"] = self.selected_target
            if fields.get("fallback_reason") is None:
                fields["fallback_reason"] = self.fallback_reason
            if fields.get("geometry_authority") is None:
                fields["geometry_authority"] = self.geometry_authority
            if "parent_pid" not in fields:
                fields["parent_pid"] = os.getpid()
            if fields.get("phase") is None:
                fields["phase"] = self._current_phase
            if fields.get("accepted") is None:
                fields["accepted"] = True
            payload = self.diagnostics.record_process(event, **fields)
            if self.logging_mode == "detailed":
                logger.info(
                    "[OverlayProcess][Lifecycle] %s",
                    json.dumps(payload, ensure_ascii=True, sort_keys=True),
                )
