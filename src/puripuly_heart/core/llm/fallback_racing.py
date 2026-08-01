from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import UUID

from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.runtime_logging import SessionRuntimeLoggingService
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY
from puripuly_heart.domain.models import Translation


@dataclass(frozen=True, slots=True)
class LLMProviderAttempt:
    provider: LLMProvider
    start_after_ms: int = 0
    start_on_primary_error: bool = False
    log_summary: str | None = None

    def __post_init__(self) -> None:
        if self.start_after_ms < 0:
            raise ValueError("start_after_ms must be >= 0")


class LLMProviderRaceError(RuntimeError):
    def __init__(self, errors: tuple[Exception, ...]) -> None:
        self.errors = errors
        if len(errors) == 2:
            details = (
                f"primary failed: {type(errors[0]).__name__}: {errors[0]}; "
                f"fallback failed: {type(errors[1]).__name__}: {errors[1]}"
            )
        else:
            details = "; ".join(
                f"attempt {index} failed: {type(error).__name__}: {error}"
                for index, error in enumerate(errors)
            )
        super().__init__(details or "all LLM attempts failed")


@dataclass(slots=True)
class _BranchOutcome:
    result: Translation | None = None
    error: Exception | None = None
    elapsed_ms: int | None = None

    @property
    def resolved(self) -> bool:
        return self.result is not None or self.error is not None


@dataclass(slots=True)
class FallbackRacingLLMProvider(LLMProvider):
    primary: LLMProvider | None = None
    fallback: LLMProvider | None = None
    fallback_timeout_ms: int = FIXED_TRANSLATION_POLICY.first_hedge_delay_ms
    loser_grace_ms: int = FIXED_TRANSLATION_POLICY.loser_grace_ms
    attempts: tuple[LLMProviderAttempt, ...] = ()
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], Awaitable[None]] | None = None
    runtime_logging: SessionRuntimeLoggingService | None = None
    _inflight_tasks: set[asyncio.Task[object]] = field(
        init=False,
        default_factory=set,
        repr=False,
    )
    _state_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _close_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _providers_closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.fallback_timeout_ms = max(0, int(self.fallback_timeout_ms))
        self.loser_grace_ms = max(0, int(self.loser_grace_ms))
        attempts = tuple(self.attempts)
        if not attempts:
            if self.primary is None:
                raise ValueError("at least one LLM provider attempt is required")
            attempts = (LLMProviderAttempt(provider=self.primary),)
            if self.fallback is not None:
                attempts += (
                    LLMProviderAttempt(
                        provider=self.fallback,
                        start_after_ms=self.fallback_timeout_ms,
                        start_on_primary_error=True,
                    ),
                )
        if not attempts:
            raise ValueError("at least one LLM provider attempt is required")
        if self.primary is not None and attempts[0].provider is not self.primary:
            raise ValueError("the first LLM attempt must be the primary provider")
        self.primary = attempts[0].provider
        if self.fallback is None and len(attempts) > 1:
            self.fallback = attempts[1].provider
        self.attempts = attempts

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        params = {
            "utterance_id": utterance_id,
            "text": text,
            "system_prompt": system_prompt,
            "source_language": source_language,
            "target_language": target_language,
            "context": context,
        }
        started_at = self.clock()
        outcomes = [_BranchOutcome() for _ in self.attempts]
        provider_tasks: dict[int, asyncio.Task[object]] = {}
        schedule_tasks: dict[asyncio.Task[object], int] = {}
        schedule_errors: list[Exception] = []
        primary_error = asyncio.Event()
        winner_event = asyncio.Event()
        winner_index: int | None = None
        winner_result: Translation | None = None

        async def start_attempt(index: int) -> None:
            if winner_event.is_set() or index in provider_tasks:
                return
            task = await self._create_tracked_task(
                self.attempts[index].provider.translate(**params)
            )
            provider_tasks[index] = task
            self._emit_attempt_started(index)

        await start_attempt(0)
        for index, attempt in enumerate(self.attempts[1:], start=1):
            schedule_task = await self._create_tracked_task(
                self._wait_for_attempt_start(
                    attempt,
                    primary_error=primary_error,
                    winner_event=winner_event,
                )
            )
            schedule_tasks[schedule_task] = index

        try:
            while winner_index is None:
                active_tasks = set(provider_tasks.values()) | set(schedule_tasks)
                if not active_tasks:
                    break
                done, _ = await asyncio.wait(
                    active_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                completed_provider_indices = sorted(
                    index for index, task in provider_tasks.items() if task in done
                )
                for index in completed_provider_indices:
                    task = provider_tasks.pop(index)
                    await self._capture_outcome(
                        task=task,
                        outcome=outcomes[index],
                        started_at=started_at,
                    )
                    if index == 0 and outcomes[index].error is not None:
                        primary_error.set()
                    if outcomes[index].result is not None and winner_index is None:
                        winner_index = index
                        winner_result = outcomes[index].result
                        winner_event.set()

                if winner_index is not None:
                    break

                completed_schedule_tasks = [task for task in done if task in schedule_tasks]
                for task in completed_schedule_tasks:
                    index = schedule_tasks.pop(task)
                    try:
                        trigger_reason = task.result()
                    except asyncio.CancelledError:
                        continue
                    except Exception as exc:
                        schedule_errors.append(exc)
                        continue
                    if trigger_reason is None or winner_event.is_set():
                        continue
                    await start_attempt(index)

            if winner_index is not None and winner_result is not None:
                await self._allow_loser_grace(
                    started_at=started_at,
                    provider_tasks=provider_tasks,
                    outcomes=outcomes,
                )
                return winner_result

            errors = tuple(
                [outcome.error for outcome in outcomes if outcome.error is not None]
                + schedule_errors
            )
            raise LLMProviderRaceError(errors)
        finally:
            for task in schedule_tasks:
                await self._cancel_task(task)
            for task in provider_tasks.values():
                await self._cancel_task(task)

    async def close(self) -> None:
        async with self._close_lock:
            async with self._state_lock:
                self._closed = True
                inflight_tasks = list(self._inflight_tasks)
            for task in inflight_tasks:
                task.cancel()
            if inflight_tasks:
                await asyncio.gather(*inflight_tasks, return_exceptions=True)
            if self._providers_closed:
                return
            closed_ids: set[int] = set()
            for attempt in self.attempts:
                provider = attempt.provider
                if id(provider) in closed_ids:
                    continue
                closed_ids.add(id(provider))
                with contextlib.suppress(Exception):
                    await provider.close()
            self._providers_closed = True

    async def _wait_for_attempt_start(
        self,
        attempt: LLMProviderAttempt,
        *,
        primary_error: asyncio.Event,
        winner_event: asyncio.Event,
    ) -> str | None:
        delay_task = await self._create_tracked_task(self._sleep(attempt.start_after_ms / 1000.0))
        error_task = (
            await self._create_tracked_task(primary_error.wait())
            if attempt.start_on_primary_error
            else None
        )
        waiters = {delay_task}
        if error_task is not None:
            waiters.add(error_task)
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if winner_event.is_set():
                return None
            if error_task is not None and error_task in done:
                return "primary_error"
            return "timeout"
        finally:
            for task in waiters:
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _sleep(self, delay_s: float) -> None:
        if self.sleeper is None:
            await asyncio.sleep(delay_s)
            return
        await self.sleeper(delay_s)

    async def _allow_loser_grace(
        self,
        *,
        started_at: float,
        provider_tasks: dict[int, asyncio.Task[object]],
        outcomes: list[_BranchOutcome],
    ) -> None:
        pending = set(provider_tasks.values())
        if pending and self.loser_grace_ms > 0:
            done, _ = await asyncio.wait(
                pending,
                timeout=self.loser_grace_ms / 1000.0,
            )
            for index, task in tuple(provider_tasks.items()):
                if task in done and not outcomes[index].resolved:
                    await self._capture_outcome(
                        task=task,
                        outcome=outcomes[index],
                        started_at=started_at,
                    )
        for index, task in tuple(provider_tasks.items()):
            if outcomes[index].resolved:
                continue
            outcomes[index].elapsed_ms = outcomes[index].elapsed_ms or self._elapsed_ms(started_at)
            await self._cancel_task(task)

    async def _capture_outcome(
        self,
        *,
        task: asyncio.Task[object],
        outcome: _BranchOutcome,
        started_at: float,
    ) -> None:
        if outcome.resolved:
            return
        if task.cancelled():
            raise asyncio.CancelledError
        outcome.elapsed_ms = self._elapsed_ms(started_at)
        try:
            outcome.result = task.result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            outcome.error = exc

    async def _create_tracked_task(self, awaitable: Awaitable[object]) -> asyncio.Task[object]:
        task = asyncio.create_task(awaitable)
        async with self._state_lock:
            if self._closed:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                raise asyncio.CancelledError
            self._inflight_tasks.add(task)
        task.add_done_callback(self._inflight_tasks.discard)
        return task

    async def _cancel_task(self, task: asyncio.Task[object] | None) -> None:
        if task is None:
            return
        if task.done():
            self._consume_task_result(task)
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def _elapsed_ms(self, started_at: float) -> int:
        return max(0, int(round((self.clock() - started_at) * 1000)))

    def _emit_attempt_started(self, index: int) -> None:
        if index == 0 or self.runtime_logging is None:
            return
        fields = ["[LLM][Fallback] started", f"stage={index}"]
        summary = self.attempts[index].log_summary
        if summary:
            fields.append(summary)
        with contextlib.suppress(Exception):
            self.runtime_logging.emit_basic(", ".join(fields))

    @staticmethod
    def _consume_task_result(task: asyncio.Task[object]) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()
