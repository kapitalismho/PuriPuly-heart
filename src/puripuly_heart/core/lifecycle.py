from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias, TypeVar

from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DIAGNOSTIC_VISIBILITY_DETAILED,
    SEVERITY_ERROR,
    ErrorDiagnostics,
)
from puripuly_heart.core.observability import DiagnosticEvent, DiagnosticsSink

_TaskResultT = TypeVar("_TaskResultT")
_DIAGNOSTIC_FIELD_VALUE_MAX_LENGTH = 128

CloseCallback = Callable[[], Awaitable[None] | None]
LifecycleShutdownPhase: TypeAlias = Literal[
    "freeze_ingress",
    "stop_external_producers",
    "run_owner_specific_drain_cancel_policies",
    "close_providers_and_output_adapters",
    "emit_final_shutdown_diagnostics",
    "flush_and_close_logging_diagnostics",
]
ShutdownCallback = Callable[[], Awaitable[None] | None]

SHUTDOWN_PHASE_FREEZE_INGRESS: Final[LifecycleShutdownPhase] = "freeze_ingress"
SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS: Final[LifecycleShutdownPhase] = "stop_external_producers"
SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL: Final[LifecycleShutdownPhase] = (
    "run_owner_specific_drain_cancel_policies"
)
SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS: Final[LifecycleShutdownPhase] = (
    "close_providers_and_output_adapters"
)
SHUTDOWN_PHASE_FINAL_DIAGNOSTICS: Final[LifecycleShutdownPhase] = "emit_final_shutdown_diagnostics"
SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS: Final[LifecycleShutdownPhase] = (
    "flush_and_close_logging_diagnostics"
)
LIFECYCLE_SHUTDOWN_PHASE_ORDER: Final[tuple[LifecycleShutdownPhase, ...]] = (
    SHUTDOWN_PHASE_FREEZE_INGRESS,
    SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
    SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
    SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS,
    SHUTDOWN_PHASE_FINAL_DIAGNOSTICS,
    SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS,
)
_LIFECYCLE_SHUTDOWN_PHASES: Final[frozenset[LifecycleShutdownPhase]] = frozenset(
    LIFECYCLE_SHUTDOWN_PHASE_ORDER
)


class LifecycleScopeClosedError(RuntimeError):
    """Raised when new work is registered after a lifecycle scope is closed."""


class LifecycleTaskNameInUseError(RuntimeError):
    """Raised when active work is registered under an existing task name."""


class LifecycleDiagnosticsUnavailableError(RuntimeError):
    """Raised when lifecycle failures exist but no diagnostics sink is available."""

    def __init__(self, scope_name: str, diagnostics: tuple[DiagnosticEvent, ...]) -> None:
        self.scope_name = scope_name
        self.diagnostics = diagnostics
        super().__init__(
            "Lifecycle diagnostics unavailable "
            f"for scope {scope_name!r}: {len(diagnostics)} event(s) pending"
        )


@dataclass(frozen=True, slots=True)
class _RegisteredCloseCallback:
    name: str
    callback: CloseCallback


@dataclass(frozen=True, slots=True)
class LifecycleShutdownCallback:
    """Named owner callback registered for one lifecycle shutdown phase."""

    phase: LifecycleShutdownPhase
    owner_name: str
    callback_name: str
    callback: ShutdownCallback


class LifecycleShutdownCoordinator:
    """Runs lifecycle owner callbacks through the phased shutdown DAG."""

    def __init__(self, *, diagnostics_sink: DiagnosticsSink | None = None) -> None:
        self._diagnostics_sink = diagnostics_sink
        self._callbacks: list[LifecycleShutdownCallback] = []
        self._pending_diagnostics: list[DiagnosticEvent] = []

    def register_callback(
        self,
        *,
        phase: LifecycleShutdownPhase,
        owner_name: str,
        callback_name: str,
        callback: ShutdownCallback,
    ) -> None:
        if phase not in _LIFECYCLE_SHUTDOWN_PHASES:
            raise ValueError(f"Unknown lifecycle shutdown phase: {phase!r}")
        self._callbacks.append(
            LifecycleShutdownCallback(
                phase=phase,
                owner_name=owner_name,
                callback_name=callback_name,
                callback=callback,
            )
        )

    async def run(self) -> None:
        for phase in LIFECYCLE_SHUTDOWN_PHASE_ORDER:
            for callback in tuple(self._callbacks):
                if callback.phase != phase:
                    continue
                await self._run_callback(callback)

        if self._pending_diagnostics:
            diagnostics = tuple(self._pending_diagnostics)
            self._pending_diagnostics.clear()
            raise LifecycleDiagnosticsUnavailableError("shutdown", diagnostics)

    async def _run_callback(self, callback: LifecycleShutdownCallback) -> None:
        try:
            result = callback.callback()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            await self._record_callback_exception(callback, exc)

    async def _record_callback_exception(
        self,
        callback: LifecycleShutdownCallback,
        exception: Exception,
    ) -> None:
        event = _shutdown_diagnostic_event(callback=callback, exception=exception)
        if self._diagnostics_sink is None:
            self._pending_diagnostics.append(event)
            return
        await self._diagnostics_sink.emit_diagnostic(event)


class LifecycleScope:
    """Owns named background tasks and close callbacks for one runtime scope."""

    def __init__(
        self,
        name: str,
        *,
        diagnostics_sink: DiagnosticsSink | None = None,
    ) -> None:
        self._name = name
        self._diagnostics_sink = diagnostics_sink
        self._closed = False
        self._closing = False
        self._close_completed = False
        self._close_lock = asyncio.Lock()
        self._tasks_by_name: dict[str, asyncio.Task[Any]] = {}
        self._task_names: dict[asyncio.Task[Any], str] = {}
        self._diagnosed_tasks: set[asyncio.Task[Any]] = set()
        self._close_callbacks: list[_RegisteredCloseCallback] = []
        self._pending_diagnostics: list[DiagnosticEvent] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def active_task_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks_by_name))

    def create_task(
        self,
        coroutine: Coroutine[Any, Any, _TaskResultT],
        *,
        name: str,
        eager_start: bool = False,
    ) -> asyncio.Task[_TaskResultT]:
        if self._closed:
            coroutine.close()
            raise LifecycleScopeClosedError(f"LifecycleScope {self._name!r} is closed to new tasks")
        if name in self._tasks_by_name:
            coroutine.close()
            raise LifecycleTaskNameInUseError(
                f"LifecycleScope {self._name!r} already has task {name!r}"
            )

        task_name = f"{self._name}:{name}"
        if eager_start:
            task = asyncio.Task(
                coroutine,
                loop=asyncio.get_running_loop(),
                name=task_name,
                eager_start=True,
            )
        else:
            task = asyncio.create_task(coroutine, name=task_name)
        self._tasks_by_name[name] = task
        self._task_names[task] = name
        task.add_done_callback(self._on_task_done)
        return task

    def register_close_callback(self, name: str, callback: CloseCallback) -> None:
        if self._closed:
            raise LifecycleScopeClosedError(
                f"LifecycleScope {self._name!r} is closed to new callbacks"
            )
        self._close_callbacks.append(_RegisteredCloseCallback(name=name, callback=callback))

    async def close(self) -> None:
        if self._close_completed:
            return

        async with self._close_lock:
            if self._close_completed:
                return

            self._closed = True
            self._closing = True
            try:
                await self._cancel_and_gather_tasks()
                await self._run_close_callbacks()
                await self._emit_pending_diagnostics()
                self._close_completed = True
            finally:
                self._closing = False

    async def _cancel_and_gather_tasks(self) -> None:
        task_entries = tuple(self._task_names.items())
        already_done = {task for task, _task_name in task_entries if task.done()}
        for task, _task_name in task_entries:
            if not task.done():
                task.cancel()

        if not task_entries:
            return

        results = await asyncio.gather(
            *(task for task, _task_name in task_entries),
            return_exceptions=True,
        )
        for (task, task_name), result in zip(task_entries, results, strict=True):
            self._tasks_by_name.pop(task_name, None)
            self._task_names.pop(task, None)
            if task in self._diagnosed_tasks:
                continue
            if isinstance(result, asyncio.CancelledError):
                continue
            if isinstance(result, BaseException):
                phase = "task_done" if task in already_done else "task_close"
                self._queue_task_exception_diagnostic(task, task_name, result, phase)

    async def _run_close_callbacks(self) -> None:
        while self._close_callbacks:
            callback = self._close_callbacks[0]
            try:
                result = callback.callback()
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                self._pending_diagnostics.append(
                    self._diagnostic_event(
                        phase="close_callback",
                        callback_name=callback.name,
                        exception=exc,
                    )
                )
            self._close_callbacks.pop(0)

    async def _emit_pending_diagnostics(self) -> None:
        if not self._pending_diagnostics:
            return

        if self._diagnostics_sink is None:
            diagnostics = tuple(self._pending_diagnostics)
            self._pending_diagnostics.clear()
            raise LifecycleDiagnosticsUnavailableError(self._name, diagnostics)

        while self._pending_diagnostics:
            event = self._pending_diagnostics[0]
            await self._diagnostics_sink.emit_diagnostic(event)
            self._pending_diagnostics.pop(0)

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        task_name = self._task_names.pop(task, None)
        if task_name is None:
            return
        self._tasks_by_name.pop(task_name, None)

        if task.cancelled():
            return
        try:
            exception = task.exception()
        except asyncio.CancelledError:
            return
        if exception is None:
            return
        self._queue_task_exception_diagnostic(task, task_name, exception, "task_done")

    def _queue_task_exception_diagnostic(
        self,
        task: asyncio.Task[Any],
        task_name: str,
        exception: BaseException,
        phase: str,
    ) -> None:
        self._diagnosed_tasks.add(task)
        self._pending_diagnostics.append(
            self._diagnostic_event(
                phase=phase,
                task_name=task_name,
                exception=exception,
            )
        )

    def _diagnostic_event(
        self,
        *,
        phase: str,
        exception: BaseException,
        task_name: str | None = None,
        callback_name: str | None = None,
    ) -> DiagnosticEvent:
        fields = {
            "scope_name": _safe_field_value(self._name),
            "phase": _safe_field_value(phase),
            "exception_class": _safe_field_value(type(exception).__name__),
        }
        if task_name is not None:
            fields["task_name"] = _safe_field_value(task_name)
        if callback_name is not None:
            fields["callback_name"] = _safe_field_value(callback_name)

        diagnostics = ErrorDiagnostics(
            component="lifecycle.scope",
            operation=phase,
            code="lifecycle_exception",
            category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
            visibility=DIAGNOSTIC_VISIBILITY_DETAILED,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields=fields,
        )
        return DiagnosticEvent(
            category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
            severity=SEVERITY_ERROR,
            visibility=DIAGNOSTIC_VISIBILITY_DETAILED,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            correlation_id=None,
            diagnostics=diagnostics,
            fields=fields,
        )


def start_lifecycle_task(
    scope: LifecycleScope,
    coroutine: Coroutine[Any, Any, _TaskResultT],
    *,
    name: str,
    eager_start: bool = False,
) -> asyncio.Task[_TaskResultT]:
    return scope.create_task(coroutine, name=name, eager_start=eager_start)


def _shutdown_diagnostic_event(
    *,
    callback: LifecycleShutdownCallback,
    exception: Exception,
) -> DiagnosticEvent:
    fields = {
        "phase": _safe_field_value(callback.phase),
        "owner_name": _safe_field_value(callback.owner_name),
        "callback_name": _safe_field_value(callback.callback_name),
        "exception_class": _safe_field_value(type(exception).__name__),
    }
    diagnostics = ErrorDiagnostics(
        component="lifecycle.shutdown",
        operation=callback.phase,
        code="lifecycle_shutdown_exception",
        category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=DIAGNOSTIC_VISIBILITY_DETAILED,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields=fields,
    )
    return DiagnosticEvent(
        category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
        severity=SEVERITY_ERROR,
        visibility=DIAGNOSTIC_VISIBILITY_DETAILED,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        correlation_id=None,
        diagnostics=diagnostics,
        fields=fields,
    )


def _safe_field_value(value: str) -> str:
    if len(value) <= _DIAGNOSTIC_FIELD_VALUE_MAX_LENGTH:
        return value
    return f"{value[: _DIAGNOSTIC_FIELD_VALUE_MAX_LENGTH - 1]}…"


__all__ = [
    "CloseCallback",
    "LIFECYCLE_SHUTDOWN_PHASE_ORDER",
    "LifecycleDiagnosticsUnavailableError",
    "LifecycleScope",
    "LifecycleScopeClosedError",
    "LifecycleShutdownCallback",
    "LifecycleShutdownCoordinator",
    "LifecycleShutdownPhase",
    "LifecycleTaskNameInUseError",
    "SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS",
    "SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS",
    "SHUTDOWN_PHASE_FINAL_DIAGNOSTICS",
    "SHUTDOWN_PHASE_FREEZE_INGRESS",
    "SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL",
    "SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS",
    "ShutdownCallback",
    "start_lifecycle_task",
]
