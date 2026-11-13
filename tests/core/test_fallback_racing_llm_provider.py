from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.diagnostics import (
    DIAGNOSTIC_RECORD_ATTRIBUTE,
    deserialize_diagnostics,
    serialize_diagnostics,
)
from puripuly_heart.core.error_messages import provider_failure_report
from puripuly_heart.core.llm import FallbackRacingLLMProvider
from puripuly_heart.core.llm.fallback_racing import LLMProviderRaceError
from puripuly_heart.core.llm.provider import LLMProvider, SemaphoreLLMProvider
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.openrouter import OpenRouterResponseError


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
        scene_participant_count: int | None = None,
    ) -> Translation:
        self.translate_calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "system_prompt": system_prompt,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
                "scene_participant_count": scene_participant_count,
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


@dataclass(slots=True)
class HeldCancelLLM(FakeLLM):
    cancel_release: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    async def translate(self, **kwargs: object) -> Translation:
        try:
            return await super().translate(**kwargs)
        except asyncio.CancelledError:
            while not self.cancel_release.is_set():
                try:
                    await self.cancel_release.wait()
                except asyncio.CancelledError:
                    continue
            if self.error is not None:
                raise self.error
            raise


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
            "scene_participant_count": None,
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
    assert result.recovered_failures == ()
    assert primary.translate_started.is_set()
    assert fallback.translate_started.is_set()
    await provider.close()
    assert primary.translate_cancelled.is_set()


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
    await asyncio.wait_for(fallback.translate_finished.wait(), timeout=0.2)
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
    assert primary.close_calls == 0
    assert fallback.close_calls == 0
    await provider.close()
    assert provider._inflight_tasks == set()


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

    with pytest.raises(LLMProviderRaceError) as failure:
        await asyncio.wait_for(translate_task, timeout=0.3)

    assert failure.value.errors == (primary.error, fallback.error)
    assert "primary boom" not in str(failure.value)
    assert "fallback boom" not in str(failure.value)

    await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary", "fallback", "key", "status", "retry"),
    [
        (
            OpenRouterResponseError(402, limit_source="openrouter_key_limit", retry_after_ms=9_000),
            OpenRouterResponseError(402, limit_source="openrouter_key_limit", retry_after_ms=9_000),
            "provider.openrouter.key_limit",
            402,
            9_000,
        ),
        (
            OpenRouterResponseError(402, limit_source="openrouter_credits", retry_after_ms=1_000),
            OpenRouterResponseError(402, limit_source="openrouter_key_limit", retry_after_ms=2_000),
            "provider.openrouter.payment_required",
            402,
            None,
        ),
        (
            OpenRouterResponseError(402, limit_source="openrouter_key_limit"),
            RuntimeError("fallback provider unavailable"),
            "provider.failure",
            None,
            None,
        ),
    ],
)
async def test_race_reports_only_diagnosis_shared_by_every_failed_branch(
    primary: Exception, fallback: Exception, key: str, status: int | None, retry: int | None
) -> None:
    provider = FallbackRacingLLMProvider(
        primary=FakeLLM(error=primary),
        fallback=FakeLLM(error=fallback),
        fallback_timeout_ms=0,
    )
    try:
        with pytest.raises(LLMProviderRaceError) as failure:
            await provider.translate(**_translation_kwargs(utterance_id=uuid4()))
        report = provider_failure_report(failure.value, provider="llm", operation="translate")
        assert report.message.key == key
        assert report.diagnostics.status_code == status
        assert report.diagnostics.retry_after_ms == retry
        assert ("limit_source" in report.diagnostics.fields) == (
            key == "provider.openrouter.key_limit"
        )
        assert report.diagnostics.category != "invalid_response"
        if status is None:
            assert report.message.params["provider"] == "llm"
        else:
            assert report.message.params["provider"] == "openrouter"
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_openrouter_payment_failure_does_not_replace_successful_fallback() -> None:
    provider = FallbackRacingLLMProvider(
        primary=FakeLLM(error=OpenRouterResponseError(402, limit_source="openrouter_credits")),
        fallback=FakeLLM(translated_text="fallback"),
        fallback_timeout_ms=0,
    )
    try:
        assert (await provider.translate(**_translation_kwargs(utterance_id=uuid4()))).text == (
            "fallback"
        )
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_winner_reaches_consumer_before_loser_cleanup_and_close_drains() -> None:

    loser = HeldCancelLLM(gate=asyncio.Event())
    winner = FakeLLM(translated_text="winner", gate=asyncio.Event())
    provider = FallbackRacingLLMProvider(
        primary=loser, fallback=winner, fallback_timeout_ms=0, loser_grace_ms=1
    )
    operation = asyncio.create_task(provider.translate(**_translation_kwargs(utterance_id=uuid4())))
    await asyncio.wait_for(loser.translate_started.wait(), 0.2)
    await asyncio.wait_for(winner.translate_started.wait(), 0.2)
    winner.gate.set()
    try:
        await asyncio.sleep(0.01)
        assert operation.done(), "winner must not wait for held loser cancellation"
        result = operation.result()
        assert result.text == "winner"
        assert provider._inflight_tasks
        closing = asyncio.create_task(provider.close())
        await asyncio.sleep(0.01)
        assert not closing.done()
        assert loser.close_calls == 0
    finally:
        loser.cancel_release.set()
        await asyncio.wait_for(provider.close(), 0.3)
        await asyncio.gather(operation, return_exceptions=True)
    assert loser.close_calls == winner.close_calls == 1


@pytest.mark.asyncio
async def test_racing_semaphore_counts_held_loser_until_cleanup() -> None:
    loser = HeldCancelLLM(gate=asyncio.Event())
    winner = FakeLLM(translated_text="winner", gate=asyncio.Event())
    inner = FallbackRacingLLMProvider(
        primary=loser, fallback=winner, fallback_timeout_ms=0, loser_grace_ms=1
    )
    provider = SemaphoreLLMProvider(inner=inner, semaphore=asyncio.Semaphore(1))
    first = asyncio.create_task(provider.translate(**_translation_kwargs(utterance_id=uuid4())))
    await asyncio.wait_for(winner.translate_started.wait(), 0.2)
    winner.gate.set()
    try:
        assert (await asyncio.wait_for(first, 0.2)).text == "winner"
        queued = asyncio.create_task(
            provider.translate(**_translation_kwargs(utterance_id=uuid4()))
        )
        await asyncio.sleep(0.01)
        assert not queued.done()
        assert len(loser.translate_calls) == 1
        assert provider.semaphore.locked()
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert provider.semaphore.locked()
    finally:
        loser.cancel_release.set()
        await asyncio.wait_for(provider.close(), 0.3)
    assert provider.semaphore._value == 1
    assert loser.close_calls == winner.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_close_retains_cleanup_and_rejects_waiting_admission() -> None:
    loser = HeldCancelLLM(gate=asyncio.Event())
    winner = FakeLLM(translated_text="winner", gate=asyncio.Event())
    inner = FallbackRacingLLMProvider(
        primary=loser, fallback=winner, fallback_timeout_ms=0, loser_grace_ms=1
    )
    provider = SemaphoreLLMProvider(inner=inner, semaphore=asyncio.Semaphore(1))
    first = asyncio.create_task(provider.translate(**_translation_kwargs(utterance_id=uuid4())))
    await asyncio.wait_for(loser.translate_started.wait(), 0.2)
    await asyncio.wait_for(winner.translate_started.wait(), 0.2)
    winner.gate.set()
    assert (await asyncio.wait_for(first, 0.2)).text == "winner"
    closing = asyncio.create_task(provider.close())
    waiting = asyncio.create_task(provider.translate(**_translation_kwargs(utterance_id=uuid4())))
    try:
        await asyncio.wait_for(loser.translate_cancelled.wait(), 0.2)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert loser.close_calls == 0
        assert not waiting.done()
        assert len(loser.translate_calls) == 1
    finally:
        loser.cancel_release.set()
        await asyncio.wait_for(provider.close(), 0.3)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiting, 0.2)
    await asyncio.wait_for(provider.semaphore.acquire(), 0.2)
    provider.semaphore.release()
    assert loser.close_calls == winner.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_caller_keeps_both_branches_accounted_until_drained() -> None:
    primary = HeldCancelLLM(gate=asyncio.Event())
    fallback = HeldCancelLLM(gate=asyncio.Event())
    inner = FallbackRacingLLMProvider(
        primary=primary, fallback=fallback, fallback_timeout_ms=0, loser_grace_ms=1
    )
    provider = SemaphoreLLMProvider(inner=inner, semaphore=asyncio.Semaphore(1))
    operation = asyncio.create_task(provider.translate(**_translation_kwargs(utterance_id=uuid4())))
    await asyncio.wait_for(primary.translate_started.wait(), 0.2)
    await asyncio.wait_for(fallback.translate_started.wait(), 0.2)
    operation.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 0.2)
        assert provider.semaphore.locked()
        closing = asyncio.create_task(provider.close())
        await asyncio.sleep(0.01)
        assert not closing.done()
        assert primary.close_calls == fallback.close_calls == 0
    finally:
        primary.cancel_release.set()
        fallback.cancel_release.set()
        await asyncio.wait_for(provider.close(), 0.3)
    await asyncio.wait_for(provider.semaphore.acquire(), 0.2)
    provider.semaphore.release()
    assert primary.close_calls == fallback.close_calls == 1


@pytest.mark.asyncio
async def test_late_loser_error_is_consumed_without_replacing_winner() -> None:
    loser = HeldCancelLLM(error=RuntimeError("late provider failure"), gate=asyncio.Event())
    winner = FakeLLM(translated_text="winner", gate=asyncio.Event())
    provider = FallbackRacingLLMProvider(
        primary=loser, fallback=winner, fallback_timeout_ms=0, loser_grace_ms=1
    )
    operation = asyncio.create_task(provider.translate(**_translation_kwargs(utterance_id=uuid4())))
    await asyncio.wait_for(loser.translate_started.wait(), 0.2)
    await asyncio.wait_for(winner.translate_started.wait(), 0.2)
    winner.gate.set()
    try:
        assert (await asyncio.wait_for(operation, 0.2)).text == "winner"
        assert loser.close_calls == 0
    finally:
        loser.cancel_release.set()
        await asyncio.wait_for(provider.close(), 0.3)
    assert loser.close_calls == winner.close_calls == 1


@pytest.mark.asyncio
async def test_successful_fallback_carries_original_evidence_when_sink_fails(monkeypatch):
    class BrokenLogger:
        def log(self, *_args, **_kwargs):
            raise OSError("PRIVATE_SECRET")

    error = OpenRouterResponseError(403)
    provider = FallbackRacingLLMProvider(
        primary=FakeLLM(error=error),
        fallback=FakeLLM(translated_text="winner"),
        fallback_timeout_ms=100,
    )
    monkeypatch.setattr("puripuly_heart.core.llm.fallback_racing.logger", BrokenLogger())
    turn_id = uuid4()
    result = await provider.translate(**_translation_kwargs(utterance_id=turn_id))
    await provider.close()

    assert result.text == "winner"
    assert len(result.recovered_failures) == 1
    recovered = deserialize_diagnostics(result.recovered_failures[0])
    assert recovered.diagnostic_id == error.failure_diagnostics.diagnostic_id
    assert recovered.turn_id == str(turn_id)
    assert recovered.status_code == 403
    assert recovered.category == "auth"
    with pytest.raises(TypeError):
        result.recovered_failures[0]["fields"]["provider"] = "PRIVATE_SECRET"


@pytest.mark.asyncio
async def test_cleanup_failure_is_degraded_without_replacing_success(caplog):
    class CleanupFailure(FakeLLM):
        async def close(self):
            raise PermissionError(13, "PRIVATE_SECRET", "C:/Users/private/file")

    provider = FallbackRacingLLMProvider(primary=CleanupFailure(translated_text="winner"))
    result = await provider.translate(**_translation_kwargs(utterance_id=uuid4()))
    await provider.close()
    records = [
        getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
        for record in caplog.records
        if hasattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
    ]

    assert result.text == "winner" and result.recovered_failures == ()
    assert len(records) == 1
    failure, severity, disposition, stage, _ = records[0]
    assert failure.code == "provider.cleanup_failed"
    assert failure.fields["errno"] == 13
    assert (severity, disposition, stage) == ("warning", "degraded", "cleanup")
    assert "PRIVATE_SECRET" not in repr(failure)


@pytest.mark.asyncio
async def test_cancelled_request_keeps_late_failure_as_history_not_terminal_outcome(caplog):
    primary = HeldCancelLLM(error=OpenRouterResponseError(403), gate=asyncio.Event())
    provider = FallbackRacingLLMProvider(primary=primary)
    operation = asyncio.create_task(provider.translate(**_translation_kwargs(utterance_id=uuid4())))
    await asyncio.wait_for(primary.translate_started.wait(), 0.2)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    await asyncio.wait_for(primary.translate_cancelled.wait(), 0.2)
    primary.cancel_release.set()
    await asyncio.wait_for(provider.close(), 0.3)
    records = [
        getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
        for record in caplog.records
        if hasattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
    ]
    assert len(records) == 1
    failure, _, disposition, _, _ = records[0]
    assert failure.status_code == 403
    assert disposition == "degraded"


@pytest.mark.asyncio
async def test_fallback_preserves_winner_secondary_evidence_and_collection_loss_flags():
    recovered = OpenRouterResponseError(502).failure_diagnostics
    secondary = OpenRouterResponseError(503).failure_diagnostics
    primary_error = OpenRouterResponseError(403)

    class EvidenceWinner(FakeLLM):
        async def translate(self, **kwargs):
            return Translation(
                utterance_id=kwargs["utterance_id"],
                text="winner",
                recovered_failures=(serialize_diagnostics(recovered),),
                secondary_diagnostics=(serialize_diagnostics(secondary),),
                recovered_failures_truncated=True,
                secondary_diagnostics_truncated=True,
            )

    provider = FallbackRacingLLMProvider(
        primary=FakeLLM(error=primary_error),
        fallback=EvidenceWinner(),
        fallback_timeout_ms=100,
    )
    result = await provider.translate(**_translation_kwargs(utterance_id=uuid4()))
    await provider.close()
    assert result.text == "winner"
    assert [item["diagnostic_id"] for item in result.recovered_failures] == [
        recovered.diagnostic_id,
        primary_error.failure_diagnostics.diagnostic_id,
    ]
    assert [item["diagnostic_id"] for item in result.secondary_diagnostics] == [
        secondary.diagnostic_id
    ]
    assert result.recovered_failures_truncated is True
    assert result.secondary_diagnostics_truncated is True
