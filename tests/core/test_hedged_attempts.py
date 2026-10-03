from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.llm import FallbackRacingLLMProvider
from puripuly_heart.core.llm.fallback_racing import LLMProviderAttempt, LLMProviderRaceError
from puripuly_heart.core.llm.latency import current_attempt, observe_request
from puripuly_heart.core.llm.provider import LLMProvider, SemaphoreLLMProvider
from puripuly_heart.domain.models import Translation


def _kwargs() -> dict[str, object]:
    return {
        "utterance_id": uuid4(),
        "text": "hello",
        "system_prompt": "translate",
        "source_language": "en",
        "target_language": "ko",
    }


@dataclass(slots=True)
class FakeProvider(LLMProvider):
    name: str
    result_text: str | None = None
    error: Exception | None = None
    gate: asyncio.Event | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    close_calls: int = 0

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            if self.error is not None:
                raise self.error
            return Translation(
                utterance_id=utterance_id,
                text=self.result_text or self.name,
                source_text=text,
                source_language=source_language,
                target_language=target_language,
            )
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def close(self) -> None:
        self.close_calls += 1


class ControlledSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []
        self.waiters: dict[float, list[asyncio.Event]] = {}

    async def __call__(self, delay_s: float) -> None:
        event = asyncio.Event()
        self.calls.append(delay_s)
        self.waiters.setdefault(delay_s, []).append(event)
        await event.wait()

    def release(self, delay_s: float) -> None:
        self.waiters[delay_s].pop(0).set()


@pytest.mark.asyncio
async def test_fast_primary_does_not_start_scheduled_attempts() -> None:
    sleeper = ControlledSleeper()
    primary = FakeProvider("primary", result_text="primary")
    fallback = FakeProvider("fallback", result_text="fallback")
    provider = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(primary),
            LLMProviderAttempt(fallback, start_after_ms=1300, start_on_primary_error=True),
        ),
        sleeper=sleeper,
    )

    result = await provider.translate(**_kwargs())

    assert result.text == "primary"
    assert primary.started.is_set()
    assert not fallback.started.is_set()
    await provider.close()


@pytest.mark.asyncio
async def test_primary_error_starts_first_fallback_without_waiting_for_delay() -> None:
    sleeper = ControlledSleeper()
    primary = FakeProvider("primary", error=RuntimeError("primary"))
    fallback = FakeProvider("fallback", result_text="fallback")
    provider = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(primary),
            LLMProviderAttempt(
                fallback,
                start_after_ms=1300,
                start_on_primary_error=True,
            ),
        ),
        sleeper=sleeper,
    )

    result = await provider.translate(**_kwargs())

    assert result.text == "fallback"
    assert fallback.started.is_set()
    assert sleeper.calls == [1.3]
    await provider.close()


@pytest.mark.asyncio
async def test_emergency_attempt_waits_for_schedule_after_earlier_errors() -> None:
    sleeper = ControlledSleeper()
    primary = FakeProvider("primary", error=RuntimeError("primary"))
    fallback = FakeProvider("fallback", error=RuntimeError("fallback"))
    emergency = FakeProvider("emergency", result_text="emergency")
    provider = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(primary),
            LLMProviderAttempt(
                fallback,
                start_after_ms=1300,
                start_on_primary_error=True,
            ),
            LLMProviderAttempt(
                emergency,
                start_after_ms=4400,
            ),
        ),
        sleeper=sleeper,
    )
    task = asyncio.create_task(provider.translate(**_kwargs()))

    while len(sleeper.calls) < 2:
        await asyncio.sleep(0)
    await asyncio.wait_for(fallback.started.wait(), timeout=0.2)
    assert not emergency.started.is_set()

    sleeper.release(4.4)
    result = await asyncio.wait_for(task, timeout=0.2)

    assert result.text == "emergency"
    assert emergency.started.is_set()
    await provider.close()


@pytest.mark.asyncio
async def test_loser_grace_cancels_slow_attempt_and_close_is_not_duplicated() -> None:
    primary = FakeProvider("primary", gate=asyncio.Event())
    winner = FakeProvider("winner", result_text="winner")
    provider = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(primary),
            LLMProviderAttempt(winner, start_after_ms=0),
        ),
        loser_grace_ms=1,
    )

    result = await asyncio.wait_for(provider.translate(**_kwargs()), timeout=0.2)

    assert result.text == "winner"
    await asyncio.wait_for(primary.cancelled.wait(), timeout=0.2)
    await provider.close()
    await provider.close()
    assert primary.close_calls == 1
    assert winner.close_calls == 1


@pytest.mark.asyncio
async def test_total_failure_preserves_errors_without_false_winner() -> None:
    primary = FakeProvider("primary", error=RuntimeError("private primary payload"))
    fallback = FakeProvider("fallback", error=RuntimeError("private fallback payload"))
    provider = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(primary),
            LLMProviderAttempt(fallback, start_after_ms=0, start_on_primary_error=True),
        ),
    )

    with pytest.raises(LLMProviderRaceError) as failure:
        await provider.translate(**_kwargs())
    assert failure.value.errors == (primary.error, fallback.error)
    assert "private primary payload" not in str(failure.value)
    assert "private fallback payload" not in str(failure.value)

    await provider.close()


@pytest.mark.asyncio
async def test_caller_cancellation_after_hedge_launch_propagates() -> None:
    primary = FakeProvider("primary", gate=asyncio.Event())
    fallback = FakeProvider("fallback", gate=asyncio.Event())
    provider = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(primary),
            LLMProviderAttempt(fallback, start_after_ms=0),
        ),
    )
    task = asyncio.create_task(provider.translate(**_kwargs()))
    await asyncio.wait_for(fallback.started.wait(), timeout=0.2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await provider.close()


@dataclass(slots=True)
class FakeExecution:
    branches: tuple[FakeProvider, ...]
    params: dict[str, object] = field(default_factory=dict)
    used: set[int] = field(default_factory=set)
    observed_indices: list[int] = field(default_factory=list)
    returned: set[int] = field(default_factory=set)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    close_calls: int = 0

    @property
    def attempt_count(self) -> int:
        return len(self.branches)

    async def translate_attempt(self, attempt_index: int) -> Translation:
        assert not self.closed.is_set()
        assert attempt_index not in self.used
        self.used.add(attempt_index)
        observation = current_attempt()
        if observation is not None:
            self.observed_indices.append(observation.attempt)
        return await self.branches[attempt_index].translate(**self.params)

    async def close(self) -> None:
        self.close_calls += 1
        self.returned.update(set(range(self.attempt_count)) - self.used)
        self.closed.set()


@dataclass(slots=True)
class FakeAdmission:
    executions: list[FakeExecution]
    gates: dict[int, asyncio.Event] = field(default_factory=dict)
    limits: list[int] = field(default_factory=list)
    entered: asyncio.Queue[int] = field(default_factory=asyncio.Queue)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_after_grant: bool = False

    async def admit_request(self, *, max_attempts: int = 2, **params: object) -> FakeExecution:
        index = len(self.limits)
        self.limits.append(max_attempts)
        self.entered.put_nowait(index)
        try:
            if index in self.gates:
                await self.gates[index].wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        execution = self.executions[index]
        assert execution.attempt_count <= max_attempts
        execution.params = dict(params)
        if self.cancel_after_grant:
            operation = asyncio.current_task()
            assert operation is not None
            operation.cancel()
        return execution


@dataclass(slots=True)
class LatencySink:
    messages: list[str] = field(default_factory=list)

    def emit_translation_latency(self, message: str) -> bool:
        self.messages.append(message)
        return True


def _admitted_racer(
    admission: FakeAdmission,
    *,
    recover: bool = True,
    loser_grace_ms: int = 0,
    sleeper: ControlledSleeper | None = None,
) -> FallbackRacingLLMProvider:
    provider = FakeProvider("direct")
    return FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(provider, provider_name="chatgpt"),
            LLMProviderAttempt(
                provider,
                start_on_primary_error=recover,
                provider_name="chatgpt",
            ),
        ),
        request_admission=admission,
        loser_grace_ms=loser_grace_ms,
        sleeper=sleeper,
    )


def _request_observation(sink: LatencySink | None, params: dict[str, object]):
    return observe_request(
        sink=sink,
        utterance_id=params["utterance_id"],
        channel="self",
        kind="translation",
        source_language="en",
        target_language="ko",
        provider_generation=0,
        input_chars=5,
        prompt_chars=9,
        context_chars=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt_count", [1, 2])
@pytest.mark.parametrize("diagnostics", [False, True])
async def test_admitted_attempts_start_immediately_and_publish_first_winner(
    attempt_count: int, diagnostics: bool
) -> None:
    branches = tuple(FakeProvider(str(index), gate=asyncio.Event()) for index in range(2))
    execution = FakeExecution(branches[:attempt_count])
    spare = FakeExecution((FakeProvider("late-duplicate"),))
    admission = FakeAdmission([execution, spare])
    sleeper = ControlledSleeper()
    provider = _admitted_racer(admission, sleeper=sleeper)
    params = _kwargs()
    sink = LatencySink() if diagnostics else None

    with _request_observation(sink, params) as request:
        task = asyncio.create_task(provider.translate(**params))
        for branch in execution.branches:
            await asyncio.wait_for(branch.started.wait(), timeout=1)
        checkpoint = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(checkpoint.set_result, None)
        await checkpoint
        assert not task.done()
        assert admission.limits == [2]
        assert sleeper.calls == []
        assert not spare.branches[0].started.is_set()
        if attempt_count == 1:
            assert not branches[1].started.is_set()
        winner_index = attempt_count - 1
        branches[winner_index].gate.set()
        result = await asyncio.wait_for(task, timeout=1)
        assert result.text == str(winner_index)
        if request is not None:
            assert request.winner_attempt == winner_index
            assert execution.observed_indices == list(range(attempt_count))
        await provider.close()

    assert execution.close_calls == 1
    assert not spare.closed.is_set()
    assert admission.limits == [2]
    if attempt_count == 2:
        assert branches[0].cancelled.is_set()
    assert provider.primary.close_calls == 1


@pytest.mark.asyncio
async def test_single_admission_failure_waits_for_single_recovery_and_observes_global_index() -> (
    None
):
    first = FakeExecution((FakeProvider("first", error=RuntimeError("first failed")),))
    recovery = FakeExecution((FakeProvider("recovered"),))
    gate = asyncio.Event()
    admission = FakeAdmission([first, recovery], gates={1: gate})
    provider = _admitted_racer(admission)
    params = _kwargs()

    with _request_observation(LatencySink(), params) as request:
        task = asyncio.create_task(provider.translate(**params))
        assert await asyncio.wait_for(admission.entered.get(), timeout=1) == 0
        assert await asyncio.wait_for(admission.entered.get(), timeout=1) == 1
        assert admission.limits == [2, 1]
        assert not recovery.branches[0].started.is_set()
        assert not task.done()
        gate.set()
        result = await asyncio.wait_for(task, timeout=1)
        assert result.text == "recovered"
        assert request.winner_attempt == 1
        assert first.observed_indices == [0]
        assert recovery.observed_indices == [1]
        await provider.close()

    assert first.close_calls == recovery.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt_count,recover", [(1, False), (2, False), (2, True)])
async def test_terminal_admitted_failures_never_add_an_unconfigured_or_third_attempt(
    attempt_count: int, recover: bool
) -> None:
    branches = tuple(
        FakeProvider(str(index), error=RuntimeError(str(index))) for index in range(attempt_count)
    )
    execution = FakeExecution(branches)
    extra = FakeExecution((FakeProvider("unexpected"),))
    admission = FakeAdmission([execution, extra])
    provider = _admitted_racer(admission, recover=recover)

    with pytest.raises(LLMProviderRaceError) as failure:
        await provider.translate(**_kwargs())
    assert failure.value.errors == tuple(branch.error for branch in branches)
    await provider.close()
    assert admission.limits == [2]
    assert execution.closed.is_set()
    assert not extra.branches[0].started.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_admission_wait_is_owned_and_cancelled_without_sending(shutdown: bool) -> None:
    execution = FakeExecution((FakeProvider("unused"),))
    admission = FakeAdmission([execution], gates={0: asyncio.Event()})
    provider = _admitted_racer(admission)
    semaphore = asyncio.Semaphore(1)
    outer = SemaphoreLLMProvider(provider, semaphore)
    task = asyncio.create_task(outer.translate(**_kwargs()))
    await asyncio.wait_for(admission.entered.get(), timeout=1)

    if shutdown:
        await asyncio.wait_for(outer.close(), timeout=1)
    else:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await outer.close()
    await asyncio.wait_for(semaphore.acquire(), timeout=1)
    assert admission.cancelled.is_set()
    assert not execution.branches[0].started.is_set()
    assert not execution.closed.is_set()
    assert provider.primary.close_calls == 1
    semaphore.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt_count", [1, 2])
async def test_cancellation_after_grant_returns_every_unused_reservation(
    attempt_count: int,
) -> None:
    execution = FakeExecution(tuple(FakeProvider(str(index)) for index in range(attempt_count)))
    admission = FakeAdmission([execution], cancel_after_grant=True)
    provider = _admitted_racer(admission)
    task = asyncio.create_task(provider.translate(**_kwargs()))

    with pytest.raises(asyncio.CancelledError):
        await task
    await provider.close()
    assert execution.close_calls == 1
    assert execution.returned == set(range(attempt_count))
    assert not any(branch.started.is_set() for branch in execution.branches)
    assert admission.limits == [2]


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_started_admitted_attempts_are_cancelled_and_closed_on_shutdown_or_caller_cancel(
    shutdown: bool,
) -> None:
    branches = tuple(FakeProvider(str(index), gate=asyncio.Event()) for index in range(2))
    execution = FakeExecution(branches)
    admission = FakeAdmission([execution])
    provider = _admitted_racer(admission)
    task = asyncio.create_task(provider.translate(**_kwargs()))
    for branch in branches:
        await asyncio.wait_for(branch.started.wait(), timeout=1)

    if shutdown:
        await asyncio.wait_for(provider.close(), timeout=1)
    else:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await provider.close()
    assert all(branch.cancelled.is_set() for branch in branches)
    assert execution.close_calls == 1
    assert execution.returned == set()
    assert admission.limits == [2]
    assert provider.primary.close_calls == 1


@pytest.mark.asyncio
async def test_cancellation_during_single_recovery_admission_closes_the_first_execution() -> None:
    first = FakeExecution((FakeProvider("failed", error=RuntimeError("failed")),))
    recovery = FakeExecution((FakeProvider("unused"),))
    admission = FakeAdmission([first, recovery], gates={1: asyncio.Event()})
    provider = _admitted_racer(admission)
    task = asyncio.create_task(provider.translate(**_kwargs()))
    assert await asyncio.wait_for(admission.entered.get(), timeout=1) == 0
    assert await asyncio.wait_for(admission.entered.get(), timeout=1) == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await provider.close()
    assert first.close_calls == 1
    assert not recovery.closed.is_set()
    assert not recovery.branches[0].started.is_set()
    assert admission.limits == [2, 1]


class CancelledExecution(FakeExecution):
    async def translate_attempt(self, attempt_index: int) -> Translation:
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_cancelled_single_lease_never_enters_recovery() -> None:
    execution = CancelledExecution((FakeProvider("retired"),))
    recovery = FakeExecution((FakeProvider("unexpected"),))
    admission = FakeAdmission([execution, recovery])
    provider = _admitted_racer(admission)

    with pytest.raises(asyncio.CancelledError):
        await provider.translate(**_kwargs())
    await provider.close()
    assert execution.returned == {0}
    assert execution.close_calls == 1
    assert not execution.branches[0].started.is_set()
    assert not recovery.branches[0].started.is_set()
    assert admission.limits == [2]


@dataclass(slots=True)
class DrainingProvider(FakeProvider):
    physical_task: asyncio.Task[bool] | None = field(default=None, init=False)

    async def translate(self, **params: object) -> Translation:
        assert self.gate is not None
        self.physical_task = asyncio.create_task(self.gate.wait())
        self.started.set()
        try:
            await asyncio.shield(self.physical_task)
            return await FakeProvider.translate(self, **params)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.mark.asyncio
async def test_admitted_winner_and_semaphore_cleanup_do_not_wait_for_physical_loser_drain() -> None:
    loser = DrainingProvider("loser", gate=asyncio.Event())
    winner = FakeProvider("winner", gate=asyncio.Event())
    execution = FakeExecution((loser, winner))
    provider = _admitted_racer(FakeAdmission([execution]))
    semaphore = asyncio.Semaphore(1)
    outer = SemaphoreLLMProvider(provider, semaphore)
    task = asyncio.create_task(outer.translate(**_kwargs()))
    await asyncio.wait_for(loser.started.wait(), timeout=1)
    await asyncio.wait_for(winner.started.wait(), timeout=1)

    winner.gate.set()
    result = await asyncio.wait_for(task, timeout=1)
    assert result.text == "winner"
    await asyncio.wait_for(execution.closed.wait(), timeout=1)
    await asyncio.wait_for(semaphore.acquire(), timeout=1)
    assert loser.cancelled.is_set()
    assert loser.physical_task is not None
    assert not loser.physical_task.done()
    assert execution.close_calls == 1
    loser.gate.set()
    await loser.physical_task
    semaphore.release()
    await outer.close()


@pytest.mark.asyncio
async def test_single_recovery_failure_preserves_both_errors_and_never_readmits() -> None:
    first_error = RuntimeError("first")
    recovery_error = RuntimeError("recovery")
    first = FakeExecution((FakeProvider("first", error=first_error),))
    recovery = FakeExecution((FakeProvider("recovery", error=recovery_error),))
    extra = FakeExecution((FakeProvider("unexpected"),))
    admission = FakeAdmission([first, recovery, extra])
    provider = _admitted_racer(admission)

    with pytest.raises(LLMProviderRaceError) as failure:
        await provider.translate(**_kwargs())
    assert failure.value.errors == (first_error, recovery_error)
    await provider.close()
    assert admission.limits == [2, 1]
    assert first.close_calls == recovery.close_calls == 1
    assert not extra.branches[0].started.is_set()
