from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.llm.index_translate import is_index_service_failure
from puripuly_heart.core.messages import ErrorDiagnostics, TransactionResult

_INDEX_SELECTION = ("index_translate_35b", "public_api")
_COMMITTED = {
    "settings_commit_success_runtime_applied",
    "settings_commit_success_runtime_degraded",
    "settings_commit_success_runtime_interrupted",
}


def translation_selection(settings: AppSettingsVNext | None) -> tuple[str, str] | None:
    if settings is None:
        return None
    translation = settings.intent.translation
    return translation.model, translation.connection


@dataclass(frozen=True, slots=True)
class IndexManagedAllowanceObservation:
    state: str
    diagnostics: ErrorDiagnostics | None = None
    confirm_current: Callable[[], Awaitable[bool]] | None = None


@dataclass(slots=True)
class IndexSelectionOwner:
    settings_provider: Callable[[], AppSettingsVNext | None]
    probe: Callable[[], Awaitable[None]]
    observe_allowance: Callable[[], Awaitable[IndexManagedAllowanceObservation]]
    commit_fallback: Callable[
        [IndexManagedAllowanceObservation, Callable[[], bool], Callable[[], bool]],
        Awaitable[TransactionResult | None],
    ]
    message_sink: Callable[[str], None]
    diagnostic_sink: Callable[[ErrorDiagnostics], None]
    deadline_seconds: float = 15.0
    total_deadline_seconds: float = 30.0
    selected_runtime_applied: Callable[[], bool] | None = None
    timeout_sink: Callable[[], None] | None = None
    generation: int = field(init=False, default=0)
    availability: str = field(init=False, default="unselected")
    allowance_observation: IndexManagedAllowanceObservation | None = field(init=False, default=None)
    failure_diagnostics: ErrorDiagnostics | None = field(init=False, default=None)
    transition_result: TransactionResult | None = field(init=False, default=None)
    _selection: tuple[str, str] | None = field(init=False, default=None)
    _task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _work_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _authority_generation: int | None = field(init=False, default=None)
    _timed_out: bool = field(init=False, default=False)
    _tasks: set[asyncio.Task[None]] = field(init=False, default_factory=set, repr=False)
    _started: bool = field(init=False, default=False)
    _stopped: bool = field(init=False, default=False)
    _committing: bool = field(init=False, default=False)
    _request_failure: BaseException | None = field(init=False, default=None, repr=False)

    def start(self) -> None:
        if self._stopped:
            return
        self._started = True
        self.selection_changed(self.settings_provider())

    def selection_changed(self, settings: AppSettingsVNext | None) -> None:
        if not self._started or self._stopped:
            return
        selection = translation_selection(settings)
        if selection == self._selection:
            return
        self.generation += 1
        self._selection = selection
        previous = self._task
        own_commit = self._committing and selection in {
            ("gemma4_26b_31b", "managed"),
            ("gemma4_26b_31b", "openrouter"),
        }
        if previous is not None and not previous.done() and not own_commit:
            previous.cancel()
        self._task = previous if own_commit else None
        if not own_commit:
            self._authority_generation = None
            self._timed_out = False
            if self._work_task is not None:
                self._work_task.cancel()
            self.transition_result = None
            self._request_failure = None
        self.availability = "unselected"
        if selection == _INDEX_SELECTION:
            self._authority_generation = self.generation
            self.availability = "pending"
            self._task = self._schedule()

    def _schedule(self, exception: BaseException | None = None) -> asyncio.Task[None]:
        task = asyncio.create_task(
            self._run_owned(self.generation, exception), name=f"index-selection-{self.generation}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run_owned(self, generation: int, exception: BaseException | None) -> None:
        work = asyncio.create_task(
            self._check(generation) if exception is None else self._fallback(generation, exception),
            name=f"index-selection-work-{generation}",
        )
        self._work_task = work
        self._tasks.add(work)
        work.add_done_callback(self._work_settled)
        try:
            async with asyncio.timeout(self.total_deadline_seconds):
                await asyncio.shield(work)
        except asyncio.CancelledError:
            work.cancel()
            raise
        except TimeoutError:
            if self._authorized(generation):
                self._authority_generation = None
                self._timed_out = True
                self.availability = "transition_failed"
                if self.timeout_sink is not None:
                    self.timeout_sink()
                diagnostics = ErrorDiagnostics(
                    component="index_selection",
                    operation="selection_transition",
                    code="index_selection_deadline",
                    category="timeout",
                    visibility="basic",
                    content_policy="metadata_only",
                    status_code=None,
                    retry_after_ms=None,
                    fields={"reason": "timeout"},
                )
                self.failure_diagnostics = diagnostics
                self.diagnostic_sink(diagnostics)
            work.cancel()

    def _work_settled(self, work: asyncio.Task[None]) -> None:
        self._tasks.discard(work)
        if not work.cancelled():
            exception = work.exception()
            if exception is not None:
                self.diagnostic_sink(
                    ErrorDiagnostics(
                        component="index_selection",
                        operation="owned_work_cleanup",
                        code="index_selection_cleanup_failed",
                        category="lifecycle",
                        visibility="basic",
                        content_policy="metadata_only",
                        status_code=None,
                        retry_after_ms=None,
                        fields={"reason": "owned_work_failed"},
                    )
                )

    def _authorized(self, generation: int) -> bool:
        return not self._stopped and generation == self._authority_generation

    def _current(self, generation: int) -> bool:
        return (
            self._authorized(generation)
            and translation_selection(self.settings_provider()) == _INDEX_SELECTION
            and self._selection == _INDEX_SELECTION
        )

    @property
    def ready_for_enable(self) -> bool:
        if self._stopped or self._timed_out:
            return False
        if translation_selection(self.settings_provider()) != _INDEX_SELECTION:
            return (self._task is None or self._task.done()) and (
                self.transition_result is None
                or self.transition_result.status == "settings_commit_success_runtime_applied"
                or (self.selected_runtime_applied is not None and self.selected_runtime_applied())
            )
        return (
            self._selection == _INDEX_SELECTION
            and self._task is not None
            and self._task.done()
            and self.availability == "available"
            and (self.selected_runtime_applied is None or self.selected_runtime_applied())
        )

    async def prepare_enable(self) -> bool:
        self.start()
        if (
            self._timed_out
            and (self._work_task is None or self._work_task.done())
            and translation_selection(self.settings_provider()) != _INDEX_SELECTION
            and self.selected_runtime_applied is not None
            and self.selected_runtime_applied()
        ):
            self._timed_out = False
        while not self._stopped:
            self.selection_changed(self.settings_provider())
            task = self._task
            if task is None:
                return self.ready_for_enable
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                continue
            if task is self._task:
                return self.ready_for_enable
        return False

    async def _check(self, generation: int) -> None:
        try:
            async with asyncio.timeout(self.deadline_seconds):
                await self.probe()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._current(generation):
                await self._fallback(generation, exc)
        else:
            if self._current(generation):
                if self._request_failure is not None:
                    await self._fallback(generation, self._request_failure)
                else:
                    self.availability = "available"

    def request_failed(self, exception: BaseException) -> None:
        self.selection_changed(self.settings_provider())
        if (
            self._stopped
            or self._selection != _INDEX_SELECTION
            or not is_index_service_failure(exception)
            or self.availability not in {"available", "pending"}
        ):
            return
        if self._task is not None and not self._task.done():
            self._request_failure = exception
            return
        self.availability = "pending"
        self._task = self._schedule(exception)

    async def _fallback(self, generation: int, exception: BaseException) -> None:
        if not self._current(generation):
            return
        self.availability = "unavailable"
        self.failure_diagnostics = ErrorDiagnostics(
            component="index_selection",
            operation="availability",
            code="index_service_unavailable",
            category="service_unavailable",
            visibility="basic",
            content_policy="metadata_only",
            status_code=getattr(exception, "status_code", None),
            retry_after_ms=None,
            fields={"reason": getattr(exception, "reason", "unknown")},
        )
        self.diagnostic_sink(self.failure_diagnostics)
        try:
            async with asyncio.timeout(self.deadline_seconds):
                observation = await self.observe_allowance()
        except asyncio.CancelledError:
            raise
        except Exception:
            observation = IndexManagedAllowanceObservation(
                "unknown",
                ErrorDiagnostics(
                    component="index_selection",
                    operation="managed_allowance",
                    code="index_managed_allowance_unknown",
                    category="unknown",
                    visibility="basic",
                    content_policy="metadata_only",
                    status_code=None,
                    retry_after_ms=None,
                    fields={"reason": "observation_unavailable"},
                ),
            )
        if not self._current(generation):
            return
        self.allowance_observation = observation
        if observation.diagnostics is not None:
            self.diagnostic_sink(observation.diagnostics)
        self._committing = True
        try:
            result = await self.commit_fallback(
                observation,
                lambda: self._current(generation),
                lambda: self._authorized(generation),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.availability = "transition_failed"
            return
        finally:
            self._committing = False
        selected = translation_selection(self.settings_provider())
        connection = selected[1] if selected is not None else None
        if not self._authorized(generation):
            return
        self.transition_result = result
        if (
            not self._stopped
            and result is not None
            and result.status in _COMMITTED
            and connection in {"managed", "openrouter"}
            and selected == ("gemma4_26b_31b", connection)
            and self._authorized(generation)
        ):
            self.message_sink(f"messages.index_api_fallback_{connection}")
        elif self._current(generation):
            self.availability = "transition_failed"

    def stop_ingress(self) -> None:
        self._stopped = True
        self.generation += 1
        self._authority_generation = None
        for task in self._tasks:
            task.cancel()

    async def close(self) -> None:
        self.stop_ingress()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
