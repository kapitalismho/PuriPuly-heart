from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Literal
from uuid import uuid4

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.diagnostics import exception_diagnostics
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
from puripuly_heart.core.stt.diagnostics import recognition_failure_diagnostics
from puripuly_heart.core.stt.scoped_engine import (
    STTProbeSessionFactory,
    STTRecognitionWatchdogs,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.domain.recognition import RecognitionStreamIdentity


@dataclass(frozen=True, slots=True)
class RecognitionProbeOutcome:
    result: str
    failure_category: str | None = None
    final_event: str | None = None
    exercised_provider: str | None = None
    diagnostics: ErrorDiagnostics | None = None
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    cleanup_complete: bool = True


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
    outcome = None
    primary_error = None
    cleanup_diagnostics = None
    scope = LifecycleScope(f"recognition-probe:{epoch}")
    try:
        async with asyncio.timeout(watchdogs.readiness_timeout_s):
            session, exercised_provider = await factory(settings, epoch)

        def terminal_outcome(
            result: str,
            failure_category: str | None = None,
            final_event: str | None = None,
            diagnostics: ErrorDiagnostics | None = None,
        ) -> RecognitionProbeOutcome:
            return RecognitionProbeOutcome(
                result, failure_category, final_event, exercised_provider, diagnostics
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
                    outcome = terminal_outcome(
                        "failed",
                        event.failure_diagnostics.category if event.failure_diagnostics else None,
                        diagnostics=event.failure_diagnostics,
                    )
                    break
                if isinstance(event, STTProviderTurnTerminal) and event.identity == identity:
                    if (
                        event.outcome == "final"
                        and event.text_authority == "authoritative"
                        and event.text.strip()
                    ):
                        outcome = terminal_outcome(
                            "succeeded", final_event="authoritative_turn_final"
                        )
                        break
                    outcome = terminal_outcome(
                        (
                            "empty"
                            if event.outcome == "empty"
                            or (event.outcome == "final" and not event.text.strip())
                            else "failed"
                        ),
                        event.failure_diagnostics.category if event.failure_diagnostics else None,
                        diagnostics=event.failure_diagnostics,
                    )
                    break
                if isinstance(event, STTRecognitionUnit) and event.identity.stream == stream:
                    if event.text.strip():
                        outcome = terminal_outcome(
                            "succeeded", final_event="final_recognition_unit"
                        )
                        break
                if (
                    isinstance(event, STTRecognitionUnitTerminal)
                    and event.unit.identity.stream == stream
                ):
                    if event.outcome == "final" and event.unit.text.strip():
                        outcome = terminal_outcome(
                            "succeeded", final_event="final_recognition_unit"
                        )
                        break
                    if event.outcome in {"failed", "cancelled"}:
                        outcome = terminal_outcome(
                            "failed",
                            diagnostics=recognition_failure_diagnostics(
                                None,
                                provider=exercised_provider or settings.provider_id,
                                channel=channel,
                                operation="recognition_probe",
                                reason=event.failure_reason or "provider_protocol_error",
                                identity=identity,
                            ),
                        )
                        break
                if isinstance(event, STTProviderEpochEnded) and event.provider_epoch_id == epoch:
                    outcome = terminal_outcome(
                        "no_result",
                        event.failure_diagnostics.category if event.failure_diagnostics else None,
                        diagnostics=event.failure_diagnostics,
                    )
                    break
        if outcome is None:
            outcome = terminal_outcome("no_result")
    except BaseException as exc:
        primary_error = exc
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
                cancellation = (
                    primary_error
                    if isinstance(primary_error, asyncio.CancelledError)
                    else asyncio.CancelledError()
                )
                if primary_error is not None and not isinstance(primary_error, asyncio.CancelledError):
                    cancellation.failure_diagnostics = exception_diagnostics(
                        primary_error,
                        component="recognition_probe",
                        operation="recognition_probe",
                        code="recognition_probe_failed",
                    )
                elif outcome is not None and outcome.diagnostics is not None:
                    cancellation.failure_diagnostics = outcome.diagnostics
                primary_error = cancellation
            if failure is not None:
                cleanup_diagnostics = exception_diagnostics(
                    failure,
                    component="recognition_probe",
                    operation="recognition_probe",
                    code="recognition_probe_cleanup_failed",
                )
        else:
            await scope.close()
    if primary_error is not None:
        if cleanup_diagnostics is not None:
            primary_error.secondary_diagnostics = (cleanup_diagnostics,)
        raise primary_error
    if cleanup_diagnostics is not None:
        return replace(
            outcome,
            diagnostics=outcome.diagnostics or cleanup_diagnostics,
            secondary_diagnostics=(
                (cleanup_diagnostics,) if outcome.diagnostics is not None else ()
            ),
            cleanup_complete=False,
        )
    return outcome
