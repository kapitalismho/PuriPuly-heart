from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import OrderedDict, deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .diagnostics import OverlayDiagnosticsRecorder
from .manifest import normalize_overlay_logging_mode

logger = logging.getLogger("puripuly_heart.core.overlay.process")

_REVERSE_DIAGNOSTIC_LIMIT = 128
_REVERSE_LINE_BYTE_LIMIT = 4 * 1024
_REVERSE_CONTROL_SLOT_LIMIT = 8
_REVERSE_LIFECYCLE_CONTROL_TYPES = frozenset(
    {
        "overlay_ready",
        "startup_error",
        "runtime_error",
        "shutdown_complete",
        "owner_status",
        "logging_mode_status",
    }
)
_REVERSE_TERMINAL_CONTROL_TYPES = frozenset({"startup_error", "runtime_error"})
_REVERSE_DIAGNOSTIC_CONTROL_TYPES = frozenset(
    {"overlay_trace", "desktop_renderer_diagnostic"}
)
_DECLARED_LOG_LEVELS = {
    "[overlay][ERROR]": logging.ERROR,
    "[overlay][WARN]": logging.WARNING,
    "[overlay][INFO]": logging.INFO,
    "[overlay][DIAG]": logging.INFO,
}
_KNOWN_PROCESS_EVENT_TYPES = (
    _REVERSE_LIFECYCLE_CONTROL_TYPES
    | _REVERSE_DIAGNOSTIC_CONTROL_TYPES
    | frozenset({"overlay_event", "shutdown_ack"})
)


@dataclass(frozen=True, slots=True)
class OverlayProcessEvent:
    payload: object
    trust_origin: Literal["process_pipe", "bridge_reverse"]

    @property
    def trusted_process_event(self) -> bool:
        return self.trust_origin == "process_pipe"


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
        if event_type in _REVERSE_DIAGNOSTIC_CONTROL_TYPES:
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


class OverlayManagedProcess(Protocol):
    async def next_event(self) -> OverlayProcessEvent: ...
    async def wait_for_exit(self) -> int | None: ...
    async def finish_readers(self) -> None: ...
    async def terminate(self) -> None: ...
    def set_logging_mode(self, mode: str) -> None: ...


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

    def drain_events(self) -> list[OverlayProcessEvent]:
        events: list[OverlayProcessEvent] = []
        while True:
            try:
                events.append(
                    OverlayProcessEvent(
                        payload=self._events.get_nowait(),
                        trust_origin="process_pipe",
                    )
                )
            except asyncio.QueueEmpty:
                return events

    async def next_event(self) -> OverlayProcessEvent:
        return OverlayProcessEvent(
            payload=await self._events.get(),
            trust_origin="process_pipe",
        )

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
                    elif self._declared_level(line) is not None:
                        if self._should_capture_failure_line(line, stream_name):
                            self._diagnostics.record_child_line(stream_name, line)
                    else:
                        self._diagnostics.note_input_rejected("unstamped_child_line")
                self._log_passthrough_line(line, stream_name)
        except asyncio.CancelledError:
            raise

    def _parse_event_line(self, line: str) -> dict[str, object] | None:
        if not line:
            return None
        candidate = line[len("EVENT ") :].strip() if line.startswith("EVENT ") else line
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("type"), str)
            and payload["type"] in _KNOWN_PROCESS_EVENT_TYPES
        ):
            return payload
        return None

    @staticmethod
    def _declared_level(line: str) -> int | None:
        for prefix, level in _DECLARED_LOG_LEVELS.items():
            if line.startswith(prefix):
                return level
        return None

    def _log_passthrough_line(self, line: str, stream_name: str) -> None:
        _ = stream_name
        if not line:
            return
        level = self._declared_level(line)
        if level is None:
            return
        if level >= logging.ERROR:
            logger.error(line)
        elif level >= logging.WARNING:
            logger.warning(line)
        elif self._logging_mode == "detailed":
            logger.info(line)

    def _should_capture_failure_line(self, line: str, stream_name: str) -> bool:
        _ = stream_name
        level = self._declared_level(line)
        return level is not None and level >= logging.WARNING

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
