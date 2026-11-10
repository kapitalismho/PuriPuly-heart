from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.stt.backend import (
    STTIndependentRecognitionSession,
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
    STTScopedTurnSession,
    scoped_event_for_delivery,
)
from puripuly_heart.core.stt.scoped_engine import (
    STTProbeSessionFactory,
    STTRecognitionWatchdogs,
)
from puripuly_heart.domain.recognition import RecognitionStreamIdentity


@dataclass(frozen=True, slots=True)
class RecognitionProbeOutcome:
    result: str
    failure_category: str | None = None
    final_event: str | None = None
    exercised_provider: str | None = None


async def recognize_probe_audio(
    factory: STTProbeSessionFactory,
    settings: AudioSegmentSettingsSnapshot,
    channel: Literal["self", "peer"],
    pcm16le: bytes,
    watchdogs: STTRecognitionWatchdogs,
    retained_sessions: list[STTScopedTurnSession],
) -> RecognitionProbeOutcome:
    epoch = uuid4().hex
    identity = STTProviderTurnIdentity(
        AudioSegmentIdentity(0, 1, uuid4(), 0),
        epoch,
        uuid4().hex,
        (settings.provider_id, settings.provider_signature, settings.runtime_signature),
    )
    stream = RecognitionStreamIdentity(channel, 0, 0, epoch, identity.settings_scope)
    session = None
    scope = LifecycleScope(f"recognition-probe:{epoch}")
    try:
        async with asyncio.timeout(watchdogs.readiness_timeout_s):
            session, exercised_provider = await factory(settings, epoch)

        def terminal_outcome(
            result: str, failure_category: str | None = None, final_event: str | None = None
        ) -> RecognitionProbeOutcome:
            return RecognitionProbeOutcome(
                result, failure_category, final_event, exercised_provider
            )

        independent = (
            session
            if isinstance(session, STTIndependentRecognitionSession)
            and session.accepts_stream_input
            and session.independent_recognition_units
            else None
        )
        async with asyncio.timeout(watchdogs.write_timeout_s):
            if independent is not None:
                await independent.begin_stream(stream)
            else:
                await session.begin_turn(STTProviderTurnRequest(identity, settings, channel))
        count = len(pcm16le) // 2
        span = AudioCaptureSpan(
            0,
            0,
            settings.target_sample_rate_hz,
            0,
            count,
            0.0,
            count / settings.target_sample_rate_hz,
            normalized_sample_rate_hz=settings.target_sample_rate_hz,
            normalized_start_sample=0,
            normalized_end_sample=count,
        )
        chunk_samples = max(1, settings.target_sample_rate_hz // 10)
        for sequence, start in enumerate(range(0, count, chunk_samples), 1):
            end = min(count, start + chunk_samples)
            ranges = (span.slice_normalized(start, end),)
            async with asyncio.timeout(watchdogs.write_timeout_s):
                if independent is not None:
                    await independent.send_stream_audio(
                        pcm16le[start * 2 : end * 2], source_ranges=ranges
                    )
                else:
                    await session.send_turn_audio(
                        identity,
                        pcm16le[start * 2 : end * 2],
                        payload_sequence=sequence,
                        source_ranges=ranges,
                        context_only=False,
                    )
        async with asyncio.timeout(watchdogs.write_timeout_s + watchdogs.drain_timeout_s):
            if independent is not None:
                await independent.end_stream(reason="source_eof")
            else:
                await session.seal_turn(
                    identity,
                    sealed_content_ranges=(span,),
                    seal_reason="source_eof",
                    observed_trailing_silence_ms=None,
                )
        async with asyncio.timeout(watchdogs.final_timeout_s):
            async for raw_event in session.turn_events():
                event = scoped_event_for_delivery(raw_event)
                if (
                    isinstance(event, STTProviderInputTerminal)
                    and event.identity == identity
                    and event.outcome != "submitted"
                ):
                    return terminal_outcome(
                        "failed",
                        event.failure_diagnostics.category if event.failure_diagnostics else None,
                    )
                if isinstance(event, STTProviderTurnTerminal) and event.identity == identity:
                    if (
                        event.outcome == "final"
                        and event.text_authority == "authoritative"
                        and event.text.strip()
                    ):
                        return terminal_outcome("succeeded", final_event="authoritative_turn_final")
                    return terminal_outcome(
                        (
                            "empty"
                            if event.outcome == "empty"
                            or (event.outcome == "final" and not event.text.strip())
                            else "failed"
                        ),
                        event.failure_diagnostics.category if event.failure_diagnostics else None,
                    )
                if isinstance(event, STTRecognitionUnit) and event.identity.stream == stream:
                    if event.text.strip():
                        return terminal_outcome("succeeded", final_event="final_recognition_unit")
                if (
                    isinstance(event, STTRecognitionUnitTerminal)
                    and event.unit.identity.stream == stream
                ):
                    if event.outcome == "final" and event.unit.text.strip():
                        return terminal_outcome("succeeded", final_event="final_recognition_unit")
                    if event.outcome in {"failed", "cancelled"}:
                        return terminal_outcome("failed")
                if isinstance(event, STTProviderEpochEnded) and event.provider_epoch_id == epoch:
                    return terminal_outcome(
                        "no_result",
                        event.failure_diagnostics.category if event.failure_diagnostics else None,
                    )
        return terminal_outcome("no_result")
    finally:
        if session is not None:

            async def close_session():
                try:
                    await session.close()
                    return None
                except (Exception, asyncio.CancelledError) as exc:
                    retained_sessions.append(session)
                    return exc

            closing = start_lifecycle_task(scope, close_session(), name="close-session")
            cancelled = False
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    cancelled = True
            failure = closing.result()
            await scope.close()
            if cancelled:
                raise asyncio.CancelledError
            if failure is not None:
                raise RuntimeError("recognition_probe_cleanup_failed") from failure
        else:
            await scope.close()
