from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Literal

from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle

OverlaySessionStartStatus = Literal["already_active", "teardown_failed", "started", "superseded"]
OverlaySessionShutdownStatus = Literal["already_off", "failed", "stopped", "superseded"]
OverlaySessionTeardown = Callable[[], Awaitable[bool]]
OverlaySessionRuntimeFactory = Callable[[], OverlayRuntimeHandle]
OverlaySessionTargetFactory = Callable[[], str]
OverlaySessionStartingHandler = Callable[[OverlayRuntimeHandle, str], None]
OverlaySessionStartOperation = Callable[
    [OverlayRuntimeHandle],
    Coroutine[object, object, None],
]
OverlaySessionStateHandler = Callable[[], None]
OverlaySessionCompletionHandler = Callable[[], Awaitable[None]]
OverlaySessionResourceProbe = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class OverlaySessionStartExecution:
    state: str
    previous_runtime: OverlayRuntimeHandle | None
    teardown: OverlaySessionTeardown
    create_runtime: OverlaySessionRuntimeFactory
    resolve_target: OverlaySessionTargetFactory
    on_starting: OverlaySessionStartingHandler
    run_start: OverlaySessionStartOperation
    replace_starting: bool = False
    retire_previous: Callable[[], Awaitable[object | None]] | None = None
    is_current: Callable[[], bool] = field(default=lambda: True, repr=False)


OverlaySessionStartExecutionFactory = Callable[[], OverlaySessionStartExecution]


@dataclass(frozen=True, slots=True)
class OverlaySessionShutdownExecution:
    state: str
    has_resources: bool
    teardown: OverlaySessionTeardown
    has_resources_after_teardown: OverlaySessionResourceProbe
    on_stopping: OverlaySessionStateHandler
    on_failed: OverlaySessionCompletionHandler
    on_stopped: OverlaySessionCompletionHandler
    is_current: Callable[[], bool] = field(default=lambda: True, repr=False)


OverlaySessionShutdownExecutionFactory = Callable[[], OverlaySessionShutdownExecution]


@dataclass(frozen=True, slots=True)
class OverlaySessionTransitionDiagnostic:
    operation: Literal["start", "shutdown"]
    outcome: Literal[
        "already_active",
        "already_off",
        "cancelled",
        "failed",
        "started",
        "stopped",
        "teardown_failed",
        "superseded",
    ]
    failure_type: str | None = None
    stage: str | None = None


OverlaySessionTransitionDiagnosticSink = Callable[
    [OverlaySessionTransitionDiagnostic],
    None,
]


@dataclass(slots=True)
class OverlaySessionTransitionOwner:
    diagnostic_sink: OverlaySessionTransitionDiagnosticSink | None = field(
        default=None,
        repr=False,
    )
    _lock: asyncio.Lock | None = field(init=False, default=None, repr=False)

    @property
    def owner_name(self) -> str:
        return "OverlaySessionTransitionOwner"

    async def begin_start(
        self,
        execution_factory: OverlaySessionStartExecutionFactory,
    ) -> OverlaySessionStartStatus:
        async with self._serialization_lock():
            execution = execution_factory()
            if not execution.is_current():
                self._emit(
                    OverlaySessionTransitionDiagnostic(operation="start", outcome="superseded")
                )
                return "superseded"
            if execution.state == "connected" or (
                execution.state == "starting" and not execution.replace_starting
            ):
                self._emit(
                    OverlaySessionTransitionDiagnostic(
                        operation="start",
                        outcome="already_active",
                    )
                )
                return "already_active"
            stage = "teardown"
            try:
                preserved_presenter = None
                if execution.retire_previous is not None:
                    stage = "retire_presentation"
                    preserved_presenter = await execution.retire_previous()
                else:
                    teardown_succeeded = await execution.teardown()
                    if not teardown_succeeded:
                        self._emit(
                            OverlaySessionTransitionDiagnostic(
                                operation="start",
                                outcome="teardown_failed",
                                stage=stage,
                            )
                        )
                        return "teardown_failed"
                    if not execution.is_current():
                        self._emit(
                            OverlaySessionTransitionDiagnostic(
                                operation="start", outcome="superseded"
                            )
                        )
                        return "superseded"
                    previous_runtime = execution.previous_runtime
                    if previous_runtime is not None and previous_runtime.is_closed:
                        stage = "detach_presenter"
                        preserved_presenter = previous_runtime.detach_preserved_presenter()
                stage = "create_runtime"
                runtime = execution.create_runtime()
                if preserved_presenter is not None:
                    stage = "adopt_presenter"
                    runtime.adopt_presenter(preserved_presenter)
                if not execution.is_current():
                    self._emit(
                        OverlaySessionTransitionDiagnostic(operation="start", outcome="superseded")
                    )
                    return "superseded"
                stage = "resolve_target"
                target = execution.resolve_target()
                stage = "mark_starting"
                execution.on_starting(runtime, target)
                stage = "create_start_task"
                runtime.create_start_task(execution.run_start(runtime))
            except asyncio.CancelledError:
                self._emit(
                    OverlaySessionTransitionDiagnostic(
                        operation="start",
                        outcome="cancelled",
                    )
                )
                raise
            except Exception as exc:
                self._emit(
                    OverlaySessionTransitionDiagnostic(
                        operation="start",
                        outcome="failed",
                        failure_type=type(exc).__name__,
                        stage=stage,
                    )
                )
                raise
            self._emit(
                OverlaySessionTransitionDiagnostic(
                    operation="start",
                    outcome="started",
                )
            )
            return "started"

    async def shutdown(
        self,
        execution_factory: OverlaySessionShutdownExecutionFactory,
    ) -> OverlaySessionShutdownStatus:
        async with self._serialization_lock():
            execution = execution_factory()
            if not execution.is_current():
                self._emit(
                    OverlaySessionTransitionDiagnostic(operation="shutdown", outcome="superseded")
                )
                return "superseded"
            if not execution.has_resources and execution.state == "off":
                self._emit(
                    OverlaySessionTransitionDiagnostic(
                        operation="shutdown",
                        outcome="already_off",
                    )
                )
                return "already_off"
            try:
                execution.on_stopping()
                teardown_succeeded = await execution.teardown()
                if not teardown_succeeded and execution.has_resources_after_teardown():
                    await execution.on_failed()
                    self._emit(
                        OverlaySessionTransitionDiagnostic(
                            operation="shutdown",
                            outcome="failed",
                        )
                    )
                    return "failed"
                await execution.on_stopped()
            except asyncio.CancelledError:
                self._emit(
                    OverlaySessionTransitionDiagnostic(
                        operation="shutdown",
                        outcome="cancelled",
                    )
                )
                raise
            except Exception as exc:
                self._emit(
                    OverlaySessionTransitionDiagnostic(
                        operation="shutdown",
                        outcome="failed",
                        failure_type=type(exc).__name__,
                    )
                )
                raise
            self._emit(
                OverlaySessionTransitionDiagnostic(
                    operation="shutdown",
                    outcome="stopped",
                )
            )
            return "stopped"

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": ("_lock",),
            "operation_policy": (
                "serialize cross-generation overlay start and shutdown transitions"
            ),
            "cancellation_policy": "propagate cancellation without admitting another transition",
            "shutdown_policy": (
                "delegate generation teardown to OverlayRuntimeHandle before publishing completion"
            ),
        }

    def _emit(self, diagnostic: OverlaySessionTransitionDiagnostic) -> None:
        if self.diagnostic_sink is None:
            return
        with contextlib.suppress(Exception):
            self.diagnostic_sink(diagnostic)

    def _serialization_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


__all__ = [
    "OverlaySessionCompletionHandler",
    "OverlaySessionResourceProbe",
    "OverlaySessionRuntimeFactory",
    "OverlaySessionShutdownExecution",
    "OverlaySessionShutdownExecutionFactory",
    "OverlaySessionShutdownStatus",
    "OverlaySessionStartExecution",
    "OverlaySessionStartExecutionFactory",
    "OverlaySessionStartOperation",
    "OverlaySessionStartStatus",
    "OverlaySessionStartingHandler",
    "OverlaySessionStateHandler",
    "OverlaySessionTargetFactory",
    "OverlaySessionTeardown",
    "OverlaySessionTransitionDiagnostic",
    "OverlaySessionTransitionDiagnosticSink",
    "OverlaySessionTransitionOwner",
]
