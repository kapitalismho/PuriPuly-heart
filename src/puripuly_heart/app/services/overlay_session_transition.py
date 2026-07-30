from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Literal

from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle

OverlaySessionStartStatus = Literal["already_active", "teardown_failed", "started"]
OverlaySessionShutdownStatus = Literal["already_off", "failed", "stopped"]
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
    ]
    failure_type: str | None = None


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
            if execution.state in {"starting", "connected"}:
                self._emit(
                    OverlaySessionTransitionDiagnostic(
                        operation="start",
                        outcome="already_active",
                    )
                )
                return "already_active"
            try:
                teardown_succeeded = await execution.teardown()
                if not teardown_succeeded:
                    self._emit(
                        OverlaySessionTransitionDiagnostic(
                            operation="start",
                            outcome="teardown_failed",
                        )
                    )
                    return "teardown_failed"
                preserved_presenter = None
                previous_runtime = execution.previous_runtime
                if previous_runtime is not None and previous_runtime.is_closed:
                    preserved_presenter = previous_runtime.detach_preserved_presenter()
                runtime = execution.create_runtime()
                if preserved_presenter is not None:
                    runtime.adopt_presenter(preserved_presenter)
                target = execution.resolve_target()
                execution.on_starting(runtime, target)
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
