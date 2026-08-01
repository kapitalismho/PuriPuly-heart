from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.llm import FallbackRacingLLMProvider
from puripuly_heart.core.llm.fallback_racing import LLMProviderAttempt, LLMProviderRaceError
from puripuly_heart.core.llm.provider import LLMProvider
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


@dataclass(slots=True)
class FakeRuntimeLogging:
    messages: list[str] = field(default_factory=list)

    def emit_basic(self, message: str, **_kwargs: object) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_fast_primary_does_not_start_scheduled_attempts() -> None:
    sleeper = ControlledSleeper()
    runtime_logging = FakeRuntimeLogging()
    primary = FakeProvider("primary", result_text="primary")
    fallback = FakeProvider("fallback", result_text="fallback")
    provider = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(primary),
            LLMProviderAttempt(fallback, start_after_ms=1300, start_on_primary_error=True),
        ),
        sleeper=sleeper,
        runtime_logging=runtime_logging,
    )

    result = await provider.translate(**_kwargs())

    assert result.text == "primary"
    assert primary.started.is_set()
    assert not fallback.started.is_set()
    assert runtime_logging.messages == []
    await provider.close()


@pytest.mark.asyncio
async def test_primary_error_starts_first_fallback_without_waiting_for_delay() -> None:
    sleeper = ControlledSleeper()
    runtime_logging = FakeRuntimeLogging()
    primary = FakeProvider("primary", error=RuntimeError("primary"))
    fallback = FakeProvider("fallback", result_text="fallback")
    provider = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(primary),
            LLMProviderAttempt(
                fallback,
                start_after_ms=1300,
                start_on_primary_error=True,
                log_summary=(
                    "provider=openrouter, model=google/gemma-4-31b-a4b-it, "
                    "mode=latency, route=gemma4_31b_latency, delay=1300ms"
                ),
            ),
        ),
        sleeper=sleeper,
        runtime_logging=runtime_logging,
    )

    result = await provider.translate(**_kwargs())

    assert result.text == "fallback"
    assert fallback.started.is_set()
    assert sleeper.calls == [1.3]
    assert runtime_logging.messages == [
        "[LLM][Fallback] started, stage=1, provider=openrouter, "
        "model=google/gemma-4-31b-a4b-it, mode=latency, "
        "route=gemma4_31b_latency, delay=1300ms"
    ]
    await provider.close()


@pytest.mark.asyncio
async def test_emergency_attempt_waits_for_schedule_after_earlier_errors() -> None:
    sleeper = ControlledSleeper()
    runtime_logging = FakeRuntimeLogging()
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
                log_summary=(
                    "provider=openrouter, model=google/gemma-4-31b-a4b-it, "
                    "mode=latency, route=gemma4_31b_latency, delay=1300ms"
                ),
            ),
            LLMProviderAttempt(
                emergency,
                start_after_ms=4500,
                log_summary=(
                    "provider=openrouter, model=google/gemma-4-31b-a4b-it, "
                    "mode=latency, route=gemma4_31b_cerebras_only, delay=4500ms"
                ),
            ),
        ),
        sleeper=sleeper,
        runtime_logging=runtime_logging,
    )
    task = asyncio.create_task(provider.translate(**_kwargs()))

    while len(sleeper.calls) < 2:
        await asyncio.sleep(0)
    await asyncio.wait_for(fallback.started.wait(), timeout=0.2)
    assert not emergency.started.is_set()

    sleeper.release(4.5)
    result = await asyncio.wait_for(task, timeout=0.2)

    assert result.text == "emergency"
    assert emergency.started.is_set()
    assert runtime_logging.messages == [
        "[LLM][Fallback] started, stage=1, provider=openrouter, "
        "model=google/gemma-4-31b-a4b-it, mode=latency, "
        "route=gemma4_31b_latency, delay=1300ms",
        "[LLM][Fallback] started, stage=2, provider=openrouter, "
        "model=google/gemma-4-31b-a4b-it, mode=latency, "
        "route=gemma4_31b_cerebras_only, delay=4500ms",
    ]
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
    assert primary.cancelled.is_set()
    await provider.close()
    await provider.close()
    assert primary.close_calls == 1
    assert winner.close_calls == 1


@pytest.mark.asyncio
async def test_total_failure_preserves_all_attempt_errors() -> None:
    primary = FakeProvider("primary", error=RuntimeError("primary"))
    fallback = FakeProvider("fallback", error=RuntimeError("fallback"))
    provider = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(primary),
            LLMProviderAttempt(fallback, start_after_ms=0, start_on_primary_error=True),
        )
    )

    with pytest.raises(LLMProviderRaceError, match="primary failed.*fallback failed"):
        await provider.translate(**_kwargs())
    await provider.close()
