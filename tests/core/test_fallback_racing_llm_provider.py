from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.llm import FallbackRacingLLMProvider
from puripuly_heart.core.llm.fallback_racing import LLMProviderRaceError
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.domain.models import Translation


def _translation_kwargs(*, utterance_id: UUID) -> dict[str, object]:
    return {
        "utterance_id": utterance_id,
        "text": "안녕",
        "system_prompt": "PROMPT",
        "source_language": "ko",
        "target_language": "en",
        "context": "ctx",
    }


@dataclass(slots=True)
class FakeLLM(LLMProvider):
    translated_text: str = "translated"
    delay_s: float = 0.0
    gate: asyncio.Event | None = None
    error: Exception | None = None
    translate_calls: list[dict[str, object]] = field(default_factory=list)
    close_calls: int = 0
    translate_started: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    translate_cancelled: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    translate_finished: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

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
        self.translate_calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "system_prompt": system_prompt,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
            }
        )
        self.translate_started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.error is not None:
                raise self.error
            translation = Translation(
                utterance_id=utterance_id,
                text=self.translated_text,
                source_text=text,
                source_language=source_language,
                target_language=target_language,
            )
            self.translate_finished.set()
            return translation
        except asyncio.CancelledError:
            self.translate_cancelled.set()
            raise

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_fallback_racer_starts_primary_immediately_and_returns_primary_before_timeout() -> (
    None
):
    primary_gate = asyncio.Event()
    primary = FakeLLM(translated_text="primary", gate=primary_gate)
    fallback = FakeLLM(translated_text="fallback")
    provider = FallbackRacingLLMProvider(
        primary=primary,
        fallback=fallback,
        fallback_timeout_ms=50,
    )
    utterance_id = uuid4()

    translate_task = asyncio.create_task(
        provider.translate(**_translation_kwargs(utterance_id=utterance_id))
    )
    await asyncio.wait_for(primary.translate_started.wait(), timeout=0.2)

    assert primary.translate_calls == [
        {
            "utterance_id": utterance_id,
            "text": "안녕",
            "system_prompt": "PROMPT",
            "source_language": "ko",
            "target_language": "en",
            "context": "ctx",
        }
    ]
    assert fallback.translate_calls == []

    primary_gate.set()
    result = await asyncio.wait_for(translate_task, timeout=0.2)

    assert result.text == "primary"
    assert fallback.translate_calls == []
    await provider.close()


@pytest.mark.asyncio
async def test_fallback_racer_returns_fallback_after_timeout() -> None:
    primary = FakeLLM(translated_text="primary", gate=asyncio.Event())
    fallback = FakeLLM(translated_text="fallback")
    provider = FallbackRacingLLMProvider(
        primary=primary,
        fallback=fallback,
        fallback_timeout_ms=10,
    )

    result = await asyncio.wait_for(
        provider.translate(**_translation_kwargs(utterance_id=uuid4())), timeout=0.2
    )

    assert result.text == "fallback"
    assert primary.translate_started.is_set()
    assert fallback.translate_started.is_set()
    assert primary.translate_cancelled.is_set()
    await provider.close()


@pytest.mark.asyncio
async def test_fallback_racer_starts_fallback_after_primary_exception() -> None:
    primary = FakeLLM(error=RuntimeError("primary failed"))
    fallback = FakeLLM(translated_text="fallback")
    provider = FallbackRacingLLMProvider(
        primary=primary,
        fallback=fallback,
        fallback_timeout_ms=100,
    )

    result = await asyncio.wait_for(
        provider.translate(**_translation_kwargs(utterance_id=uuid4())), timeout=0.2
    )

    assert result.text == "fallback"
    assert fallback.translate_started.is_set()
    await provider.close()


@pytest.mark.asyncio
async def test_fallback_racer_can_still_return_primary_after_fallback_starts() -> None:
    primary_gate = asyncio.Event()
    primary = FakeLLM(translated_text="primary", gate=primary_gate)
    fallback = FakeLLM(translated_text="fallback", delay_s=0.01)
    provider = FallbackRacingLLMProvider(
        primary=primary,
        fallback=fallback,
        fallback_timeout_ms=1,
        loser_grace_ms=50,
    )
    translate_task = asyncio.create_task(
        provider.translate(**_translation_kwargs(utterance_id=uuid4()))
    )

    await asyncio.wait_for(primary.translate_started.wait(), timeout=0.2)
    await asyncio.wait_for(fallback.translate_started.wait(), timeout=0.2)
    primary_gate.set()

    result = await asyncio.wait_for(translate_task, timeout=0.3)

    assert result.text == "primary"
    assert fallback.translate_finished.is_set()
    await provider.close()


@pytest.mark.asyncio
async def test_fallback_racer_preserves_primary_when_fallback_fails() -> None:
    primary_gate = asyncio.Event()
    primary = FakeLLM(translated_text="primary", gate=primary_gate)
    fallback = FakeLLM(error=RuntimeError("fallback unavailable"))
    provider = FallbackRacingLLMProvider(
        primary=primary,
        fallback=fallback,
        fallback_timeout_ms=1,
    )
    translate_task = asyncio.create_task(
        provider.translate(**_translation_kwargs(utterance_id=uuid4()))
    )

    await asyncio.wait_for(primary.translate_started.wait(), timeout=0.2)
    await asyncio.wait_for(fallback.translate_started.wait(), timeout=0.2)
    primary_gate.set()

    result = await asyncio.wait_for(translate_task, timeout=0.3)

    assert result.text == "primary"
    await provider.close()


@pytest.mark.asyncio
async def test_fallback_racer_close_cancels_inflight_branches() -> None:
    primary = FakeLLM(gate=asyncio.Event())
    fallback = FakeLLM(gate=asyncio.Event())
    provider = FallbackRacingLLMProvider(
        primary=primary,
        fallback=fallback,
        fallback_timeout_ms=10,
    )

    translate_task = asyncio.create_task(
        provider.translate(**_translation_kwargs(utterance_id=uuid4()))
    )
    await asyncio.wait_for(primary.translate_started.wait(), timeout=0.2)
    await asyncio.wait_for(fallback.translate_started.wait(), timeout=0.2)

    await provider.close()

    with pytest.raises(asyncio.CancelledError):
        await translate_task
    assert primary.translate_cancelled.is_set()
    assert fallback.translate_cancelled.is_set()
    assert primary.close_calls == 1
    assert fallback.close_calls == 1


@pytest.mark.asyncio
async def test_fallback_racer_caller_cancel_cancels_inflight_branches_without_close() -> None:
    primary = FakeLLM(gate=asyncio.Event())
    fallback = FakeLLM(gate=asyncio.Event())
    provider = FallbackRacingLLMProvider(
        primary=primary,
        fallback=fallback,
        fallback_timeout_ms=10,
    )

    translate_task = asyncio.create_task(
        provider.translate(**_translation_kwargs(utterance_id=uuid4()))
    )
    await asyncio.wait_for(primary.translate_started.wait(), timeout=0.2)
    await asyncio.wait_for(fallback.translate_started.wait(), timeout=0.2)

    translate_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await translate_task
    await asyncio.wait_for(primary.translate_cancelled.wait(), timeout=0.2)
    await asyncio.wait_for(fallback.translate_cancelled.wait(), timeout=0.2)
    assert provider._inflight_tasks == set()
    assert primary.close_calls == 0
    assert fallback.close_calls == 0
    await provider.close()


@pytest.mark.asyncio
async def test_fallback_racer_waits_for_fallback_when_primary_fails_after_fallback_started() -> (
    None
):
    primary_gate = asyncio.Event()
    primary = FakeLLM(error=RuntimeError("primary boom"), gate=primary_gate)
    fallback = FakeLLM(translated_text="fallback", delay_s=0.01)
    provider = FallbackRacingLLMProvider(
        primary=primary,
        fallback=fallback,
        fallback_timeout_ms=1,
    )
    translate_task = asyncio.create_task(
        provider.translate(**_translation_kwargs(utterance_id=uuid4()))
    )

    await asyncio.wait_for(primary.translate_started.wait(), timeout=0.2)
    await asyncio.wait_for(fallback.translate_started.wait(), timeout=0.2)
    primary_gate.set()

    result = await asyncio.wait_for(translate_task, timeout=0.3)

    assert result.text == "fallback"
    await provider.close()


@pytest.mark.asyncio
async def test_fallback_racer_preserves_both_errors_when_both_branches_fail() -> None:
    primary_gate = asyncio.Event()
    primary = FakeLLM(error=RuntimeError("primary boom"), gate=primary_gate)
    fallback = FakeLLM(error=RuntimeError("fallback boom"))
    provider = FallbackRacingLLMProvider(
        primary=primary,
        fallback=fallback,
        fallback_timeout_ms=1,
    )
    translate_task = asyncio.create_task(
        provider.translate(**_translation_kwargs(utterance_id=uuid4()))
    )

    await asyncio.wait_for(primary.translate_started.wait(), timeout=0.2)
    await asyncio.wait_for(fallback.translate_started.wait(), timeout=0.2)
    primary_gate.set()

    with pytest.raises(
        LLMProviderRaceError,
        match="primary failed: RuntimeError: primary boom; fallback failed: RuntimeError: fallback boom",
    ):
        await asyncio.wait_for(translate_task, timeout=0.3)

    await provider.close()
