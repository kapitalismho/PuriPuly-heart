from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

from puripuly_heart.core.lifecycle import (
    LIFECYCLE_SHUTDOWN_PHASE_ORDER,
    LifecycleShutdownPhase,
)

ApplicationLifecycleState: TypeAlias = Literal[
    "running",
    "shutting_down",
    "completed",
    "completed_with_failures",
]
ApplicationShutdownCallbackFn: TypeAlias = Callable[
    ["ApplicationShutdownContext"],
    Awaitable[None] | None,
]
ApplicationShutdownDiagnosticsSink: TypeAlias = Callable[
    ["ApplicationShutdownDiagnostic"],
    Awaitable[None] | None,
]

DEFAULT_APPLICATION_SHUTDOWN_CALLBACK_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_APPLICATION_SHUTDOWN_DIAGNOSTIC_TIMEOUT_SECONDS: Final = 1.0


class ApplicationIntentRejectedError(RuntimeError):
    def __init__(self, intent_name: str, state: ApplicationLifecycleState) -> None:
        self.intent_name = intent_name
        self.state = state
        super().__init__(f"Application intent {intent_name!r} rejected while lifecycle is {state}")


class ApplicationShutdownRegistrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ApplicationShutdownFailure:
    phase: LifecycleShutdownPhase
    owner_name: str
    callback_name: str
    exception_class: str
    timed_out: bool


@dataclass(frozen=True, slots=True)
class ApplicationShutdownDiagnostic:
    phase: LifecycleShutdownPhase
    owner_name: str
    callback_name: str
    exception_class: str
    timed_out: bool


@dataclass(frozen=True, slots=True)
class ApplicationLifecycleSnapshot:
    state: ApplicationLifecycleState
    accepting_intents: bool
    phase: LifecycleShutdownPhase | None
    terminal: bool
    failure_count: int
    failures: tuple[ApplicationShutdownFailure, ...]
    phase_history: tuple[LifecycleShutdownPhase, ...]


@dataclass(frozen=True, slots=True)
class ApplicationShutdownContext:
    phase: LifecycleShutdownPhase
    failures: tuple[ApplicationShutdownFailure, ...]
    cleanup_exceptions: tuple[BaseException, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ApplicationShutdownCallback:
    phase: LifecycleShutdownPhase
    owner_name: str
    callback_name: str
    callback: ApplicationShutdownCallbackFn
    timeout_seconds: float = DEFAULT_APPLICATION_SHUTDOWN_CALLBACK_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.phase not in LIFECYCLE_SHUTDOWN_PHASE_ORDER:
            raise ValueError(f"Unknown application shutdown phase: {self.phase!r}")
        if not self.owner_name.strip():
            raise ValueError("Application shutdown owner_name must not be empty")
        if not self.callback_name.strip():
            raise ValueError("Application shutdown callback_name must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("Application shutdown timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class _RecordedFailure:
    summary: ApplicationShutdownFailure
    exception: BaseException = field(repr=False)


class ApplicationShutdownCoordinator:
    owner_name: Final = "ApplicationShutdownCoordinator"

    def __init__(
        self,
        callbacks: Sequence[ApplicationShutdownCallback] = (),
        *,
        diagnostics_sink: ApplicationShutdownDiagnosticsSink | None = None,
        diagnostics_timeout_seconds: float = DEFAULT_APPLICATION_SHUTDOWN_DIAGNOSTIC_TIMEOUT_SECONDS,
    ) -> None:
        if diagnostics_timeout_seconds <= 0:
            raise ValueError("Application shutdown diagnostics_timeout_seconds must be positive")
        self._callbacks = list(callbacks)
        self._diagnostics_sink = diagnostics_sink
        self._diagnostics_timeout_seconds = diagnostics_timeout_seconds
        self._state: ApplicationLifecycleState = "running"
        self._phase: LifecycleShutdownPhase | None = None
        self._phase_history: list[LifecycleShutdownPhase] = []
        self._failures: list[_RecordedFailure] = []
        self._shutdown_task: asyncio.Task[ApplicationLifecycleSnapshot] | None = None
        self._terminal_exception: BaseException | None = None
        self._timed_out_callback_tasks: set[asyncio.Task[None]] = set()

    @property
    def accepting_intents(self) -> bool:
        return self._state == "running"

    @property
    def is_terminal(self) -> bool:
        return self._state in {"completed", "completed_with_failures"}

    @property
    def snapshot(self) -> ApplicationLifecycleSnapshot:
        return ApplicationLifecycleSnapshot(
            state=self._state,
            accepting_intents=self.accepting_intents,
            phase=self._phase,
            terminal=self.is_terminal,
            failure_count=len(self._failures),
            failures=tuple(failure.summary for failure in self._failures),
            phase_history=tuple(self._phase_history),
        )

    def admit_intent(self, intent_name: str) -> None:
        if not self.accepting_intents:
            raise ApplicationIntentRejectedError(intent_name, self._state)

    def register_callback(self, callback: ApplicationShutdownCallback) -> None:
        if self._state != "running" or self._shutdown_task is not None:
            raise ApplicationShutdownRegistrationError(
                "Application shutdown callbacks cannot be registered after shutdown begins"
            )
        self._callbacks.append(callback)

    async def shutdown(self) -> ApplicationLifecycleSnapshot:
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(
                self._run_shutdown(),
                name="application-shutdown-coordinator",
            )
            self._shutdown_task = task
        try:
            snapshot = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        if self._terminal_exception is not None:
            raise self._terminal_exception
        return snapshot

    async def _run_shutdown(self) -> ApplicationLifecycleSnapshot:
        self._state = "shutting_down"
        for phase in LIFECYCLE_SHUTDOWN_PHASE_ORDER:
            self._phase = phase
            self._phase_history.append(phase)
            callbacks = tuple(callback for callback in self._callbacks if callback.phase == phase)
            for callback in callbacks:
                await self._run_callback(callback)

        self._phase = None
        self._state = "completed_with_failures" if self._failures else "completed"
        if self._failures:
            self._terminal_exception = _terminal_exception(self._failures)
        return self.snapshot

    async def _run_callback(self, callback: ApplicationShutdownCallback) -> None:
        context = ApplicationShutdownContext(
            phase=callback.phase,
            failures=tuple(failure.summary for failure in self._failures),
            cleanup_exceptions=tuple(failure.exception for failure in self._failures),
        )
        try:
            result = callback.callback(context)
            if not inspect.isawaitable(result):
                return
            task = asyncio.create_task(
                self._await_callback(result),
                name=(
                    "application-shutdown:"
                    f"{callback.phase}:{callback.owner_name}:{callback.callback_name}"
                ),
            )
            done, _pending = await asyncio.wait(
                {task},
                timeout=callback.timeout_seconds,
                return_when=asyncio.ALL_COMPLETED,
            )
            if task not in done:
                task.cancel()
                self._timed_out_callback_tasks.add(task)
                task.add_done_callback(self._on_timed_out_callback_done)
                raise TimeoutError(
                    "Application shutdown callback exceeded its coordinator deadline"
                )
            await task
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failure = _RecordedFailure(
                summary=ApplicationShutdownFailure(
                    phase=callback.phase,
                    owner_name=callback.owner_name,
                    callback_name=callback.callback_name,
                    exception_class=type(exc).__name__,
                    timed_out=isinstance(exc, TimeoutError),
                ),
                exception=exc,
            )
            self._failures.append(failure)
            await self._emit_diagnostic(failure.summary)

    async def _await_callback(self, result: Awaitable[None]) -> None:
        await result

    def _on_timed_out_callback_done(self, task: asyncio.Task[None]) -> None:
        self._timed_out_callback_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            return

    async def _emit_diagnostic(self, failure: ApplicationShutdownFailure) -> None:
        if self._diagnostics_sink is None:
            return
        diagnostic = ApplicationShutdownDiagnostic(
            phase=failure.phase,
            owner_name=failure.owner_name,
            callback_name=failure.callback_name,
            exception_class=failure.exception_class,
            timed_out=failure.timed_out,
        )
        try:
            result = self._diagnostics_sink(diagnostic)
            if inspect.isawaitable(result):
                task = asyncio.create_task(
                    self._await_callback(result),
                    name="application-shutdown:emit-diagnostic",
                )
                done, _pending = await asyncio.wait(
                    {task},
                    timeout=self._diagnostics_timeout_seconds,
                    return_when=asyncio.ALL_COMPLETED,
                )
                if task not in done:
                    task.cancel()
                    self._timed_out_callback_tasks.add(task)
                    task.add_done_callback(self._on_timed_out_callback_done)
                    raise TimeoutError(
                        "Application shutdown diagnostic delivery exceeded its coordinator deadline"
                    )
                await task
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._failures.append(
                _RecordedFailure(
                    summary=ApplicationShutdownFailure(
                        phase=failure.phase,
                        owner_name=self.owner_name,
                        callback_name="emit_diagnostic",
                        exception_class=type(exc).__name__,
                        timed_out=isinstance(exc, TimeoutError),
                    ),
                    exception=exc,
                )
            )


def application_shutdown_callback(
    *,
    phase: LifecycleShutdownPhase,
    owner_name: str,
    callback_name: str,
    callback: Callable[[], Awaitable[None] | None],
    timeout_seconds: float = DEFAULT_APPLICATION_SHUTDOWN_CALLBACK_TIMEOUT_SECONDS,
) -> ApplicationShutdownCallback:
    def invoke(_context: ApplicationShutdownContext) -> Awaitable[None] | None:
        return callback()

    return ApplicationShutdownCallback(
        phase=phase,
        owner_name=owner_name,
        callback_name=callback_name,
        callback=invoke,
        timeout_seconds=timeout_seconds,
    )


def _terminal_exception(failures: Sequence[_RecordedFailure]) -> BaseException:
    exceptions = [failure.exception for failure in failures]
    if len(exceptions) == 1:
        return exceptions[0]
    message = "Application shutdown completed with cleanup failures"
    if all(isinstance(exception, Exception) for exception in exceptions):
        return ExceptionGroup(
            message, [exception for exception in exceptions if isinstance(exception, Exception)]
        )
    return BaseExceptionGroup(message, exceptions)


__all__ = [
    "ApplicationIntentRejectedError",
    "ApplicationLifecycleSnapshot",
    "ApplicationLifecycleState",
    "ApplicationShutdownCallback",
    "ApplicationShutdownContext",
    "ApplicationShutdownCoordinator",
    "ApplicationShutdownDiagnostic",
    "ApplicationShutdownFailure",
    "ApplicationShutdownRegistrationError",
    "DEFAULT_APPLICATION_SHUTDOWN_CALLBACK_TIMEOUT_SECONDS",
    "DEFAULT_APPLICATION_SHUTDOWN_DIAGNOSTIC_TIMEOUT_SECONDS",
    "application_shutdown_callback",
]
