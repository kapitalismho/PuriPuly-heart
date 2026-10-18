from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol
from uuid import UUID, uuid4

from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentTerminalReceipt,
    OwnedStreamInput,
    OwnedVadEvent,
)
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.diagnostics import current_diagnostic_operation
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.local_asr.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
)
from puripuly_heart.core.messages import UserErrorReport, UserMessageRef
from puripuly_heart.core.orchestrator.channel_runtime import (
    ChannelRuntime,
    _MergeBuffer,
)
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
    TranslationRuntimeConfigSnapshotPort,
)
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    LatencyInheritanceDiagnostic,
    LatencyStageDiagnostic,
    LatencyTimelineDiagnostic,
    RuntimeDiagnostic,
    SttEventLoopFailureDiagnostic,
    SttTurnFailureDiagnostic,
    TranslationFailureDiagnostic,
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import (
    ActiveSelfProjection,
    TranslationOutputProjectionOwner,
    TranslationUiMessage,
)
from puripuly_heart.core.orchestrator.translation_request import (
    PreparedTranslationRequest,
    TranslationProcessRequest,
    TranslationRequestPort,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TerminalTranslationFailure,
    TranslationOutputSubmission,
    TranslationTurnChild,
    TranslationTurnKind,
    TranslationTurnLifecycleOwner,
    TranslationTurnOutcome,
    TranslationTurnProcessResult,
    TranslationTurnRequest,
)
from puripuly_heart.core.runtime.output import SELF_SPEECH_TYPING_REASON
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    scoped_event_for_delivery,
)
from puripuly_heart.core.stt.diagnostics import recognition_cause, recognition_failure_diagnostics
from puripuly_heart.core.stt.recognition_units import recognition_transcript
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadEvent
from puripuly_heart.domain.events import (
    SelfTurnPresentation,
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionState,
    STTSessionStateEvent,
    UIErrorPayload,
    UIEventType,
)
from puripuly_heart.domain.models import Transcript, Translation
from puripuly_heart.domain.recognition import RecognitionTextOrigin

_PROMO_INTERVAL_SEC = 300.0


@dataclass(frozen=True, slots=True)
class TextSubmissionAdmission:
    status: Literal["admitted", "rejected"]
    turn_id: UUID
    child_turn_ids: tuple[UUID, ...]


class SelfTranslationChannelPort(Protocol):
    async def handle_vad_event(self, event: VadEvent) -> None: ...

    async def handle_stt_event(self, event: object) -> None: ...

    async def handle_retired_stt_event(self, event: object) -> None: ...

    async def handle_stt_event_loop_exception(self, exc: Exception) -> None: ...

    async def submit_text(self, text: str, *, source: str = "You") -> TextSubmissionAdmission: ...

    def mark_promo_eligible(self) -> None: ...

    async def reset_provider_channel(self, channel: str = "self") -> None: ...

    async def clear_language_runtime_state(self) -> None: ...

    async def open_ingress(self) -> None: ...

    async def close_ingress(self) -> None: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class _AcousticContinuation:
    utterance_id: UUID
    started_at: float
    recognition_scope: tuple[object, ...] | None
    chunk_count: int = 0
    ended_at: float | None = None
    timeout_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class SelfTranslationChannelOwner:
    runtime: ChannelRuntime = field(repr=False)
    config_snapshot: TranslationRuntimeConfigSnapshotPort = field(repr=False)
    translation_turns: TranslationTurnLifecycleOwner = field(repr=False)
    local_asr_runtime: LocalASRProviderRuntimePort = field(repr=False)
    translation_requests: TranslationRequestPort = field(repr=False)
    output_projection: TranslationOutputProjectionOwner = field(repr=False)
    diagnostics: TranslationLatencyDiagnosticsOwner = field(repr=False)
    clock: Clock
    _last_promo_time: float | None = field(init=False, default=None)
    _promo_eligible: bool = field(init=False, default=False)
    _accepting_events: bool = field(init=False, default=True)
    _admitted_requests: dict[UUID, PreparedTranslationRequest] = field(default_factory=dict)
    _stt_consumption: STTContributionConsumptionLedger = field(
        default_factory=STTContributionConsumptionLedger
    )
    _scoped_publication_ids: dict[STTProviderTurnIdentity, UUID] = field(default_factory=dict)
    _scoped_publication_text: dict[STTProviderTurnIdentity, str] = field(default_factory=dict)
    _scoped_publication_origins: dict[
        STTProviderTurnIdentity, tuple[RecognitionTextOrigin, ...]
    ] = field(default_factory=dict)
    _scoped_endpoint_publication_ids: dict[UUID, UUID] = field(default_factory=dict)
    _retired_local_vad_ids: OrderedDict[UUID, None] = field(default_factory=OrderedDict)
    _scoped_terminal_reports: OrderedDict[STTProviderTurnIdentity, None] = field(
        default_factory=OrderedDict
    )
    _acoustic_continuations: dict[AudioSegmentIdentity | UUID, _AcousticContinuation] = field(
        default_factory=dict
    )
    _task_scope: LifecycleScope = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.runtime.channel != "self":
            raise ValueError("Self translation owner requires the Self channel runtime")
        self._task_scope = LifecycleScope("SelfTranslationChannelOwner")
        self.translation_turns.on_draft_updated = self._on_draft_updated
        self.translation_turns.prepare_draft_child = self._prepare_draft_child

    @property
    def merge_buffer(self) -> _MergeBuffer | None:
        return self.runtime.merge_buffer

    @merge_buffer.setter
    def merge_buffer(self, value: _MergeBuffer | None) -> None:
        self.runtime.merge_buffer = value

    @property
    def accepting_events(self) -> bool:
        return self._accepting_events

    def set_clock(self, clock: Clock) -> None:
        self.clock = clock

    async def open_ingress(self) -> None:
        self._accepting_events = True

    async def close_ingress(self) -> None:
        self._accepting_events = False

    async def close(self) -> None:
        self._accepting_events = False
        self._admitted_requests.clear()
        self._retired_local_vad_ids.clear()
        self._clear_scoped_recognition_state()
        await self.runtime.reset_runtime_state()
        await self._task_scope.close()
        self.diagnostics.clear_latency_state(channel="self")

    async def reset_provider_channel(self, channel: str = "self") -> None:
        if channel != "self":
            raise ValueError("Self translation owner cannot reset a non-Self channel")
        self.output_projection.retire_self_ui_speech()
        await self.translation_turns.cancel_pending(channel="self", turn_kinds=frozenset({"self"}))
        await self.output_projection.reset_overlay_preview()
        await self.runtime.clear_self_speech_state()
        self._clear_scoped_recognition_state()
        self._retired_local_vad_ids.clear()
        self.diagnostics.clear_latency_state(channel="self")

    async def clear_language_runtime_state(self) -> None:
        await self.translation_turns.cancel_pending(channel="self")
        self._admitted_requests.clear()
        await self.runtime.clear_live_translation_state()
        self.diagnostics.clear_latency_state(channel="self")
        self._clear_scoped_recognition_state()
        self._retired_local_vad_ids.clear()
        await self.output_projection.reset_overlay_preview()

    def mark_promo_eligible(self) -> None:
        self._promo_eligible = True

    async def observe_source_activity(
        self,
        *,
        speech_observed: bool,
        observed_at_monotonic_s: float,
    ) -> None:
        self._require_ingress()
        await self.local_asr_runtime.observe_source_activity(
            "self",
            speech_observed=speech_observed,
            observed_at_monotonic_s=observed_at_monotonic_s,
        )

    async def observe_pending_source_work(self, *, pending: bool) -> None:
        self._require_ingress()
        await self.local_asr_runtime.observe_pending_source_work("self", pending=pending)

    async def handle_stream_input(self, owned: OwnedStreamInput) -> None:
        self._require_ingress()
        await self.local_asr_runtime.handle_stream_input("self", owned)

    async def handle_vad_event(self, event: VadEvent | OwnedVadEvent) -> None:
        self._require_ingress()
        owned = event if isinstance(event, OwnedVadEvent) else None
        vad_event = owned.event if owned is not None else event
        if not isinstance(vad_event, SpeechStart | SpeechChunk | SpeechEnd):
            raise TypeError("Self translation owner received an invalid VAD event")
        resume_overlay_resync_buffer: _MergeBuffer | None = None
        low_latency_mode = self.config_snapshot().value.low_latency_mode
        source_identity = owned.segment.identity if owned is not None else None
        recognition_scope = (
            (
                owned.segment.settings.provider_id,
                owned.segment.settings.provider_signature,
                owned.segment.settings.runtime_signature,
            )
            if owned is not None
            else None
        )
        if isinstance(vad_event, SpeechStart) and low_latency_mode:
            self._mark_resume_pending(
                vad_event,
                source_identity=source_identity,
                recognition_scope=recognition_scope,
            )
        if isinstance(vad_event, SpeechChunk) and low_latency_mode:
            resume_overlay_resync_buffer = self._maybe_confirm_resume(
                vad_event,
                source_identity=source_identity,
            )
        if (
            isinstance(vad_event, SpeechEnd)
            and vad_event.utterance_id not in self._retired_local_vad_ids
        ):
            speech_end_at = self.clock.now()
            sealed_at = (
                owned.segment.sealed_at_monotonic_s
                if owned is not None and owned.segment.sealed_at_monotonic_s is not None
                else speech_end_at
            )
            source_end_at = (
                owned.segment.content_ranges[-1].source_end_monotonic_s
                if owned is not None and owned.segment.content_ranges
                else sealed_at
            )
            self._record_latency_stage(
                utterance_id=vad_event.utterance_id,
                stage="last_speech",
                timestamp=source_end_at - vad_event.trailing_silence_ms / 1000.0,
                publish_now=False,
            )
            self.output_projection.set_self_chatbox_typing_reason(
                SELF_SPEECH_TYPING_REASON,
                True,
            )
            self.runtime.utterance_start_times[vad_event.utterance_id] = speech_end_at
            self.runtime.speech_ended_ids.add(vad_event.utterance_id)
            self._record_latency_stage(
                utterance_id=vad_event.utterance_id,
                stage="speech_end",
                timestamp=sealed_at,
                publish_now=not low_latency_mode,
            )
            self._inherit_scoped_publication_end(
                vad_event.utterance_id,
                speech_end_at,
            )
            if low_latency_mode:
                self._maybe_update_buffer_end_time(vad_event.utterance_id)
                self._maybe_start_finalize_wait(vad_event.utterance_id)
                await self._maybe_clear_resume_on_end(
                    vad_event,
                    source_identity=source_identity,
                )
                buffer = self.merge_buffer
                if buffer is not None:
                    await self._evaluate_speculative_next_action(
                        buffer,
                        reason="speech_end",
                    )
        if owned is None:
            await self.local_asr_runtime.handle_vad_event("self", vad_event)
        else:
            await self.local_asr_runtime.handle_owned_vad_event("self", owned)
        if isinstance(vad_event, SpeechEnd):
            await self.local_asr_runtime.commit_handoff("self")
        if (
            resume_overlay_resync_buffer is not None
            and self.merge_buffer is resume_overlay_resync_buffer
        ):
            await self._sync_overlay_active_self(resume_overlay_resync_buffer)

    async def reject_owned_segment(
        self,
        event: object,
        *,
        reason: str,
        outcome: str,
    ) -> None:
        self._require_ingress()
        if not isinstance(event, OwnedVadEvent):
            raise TypeError("Self recognition rejection requires OwnedVadEvent")
        await self.local_asr_runtime.reject_owned_segment(
            "self",
            event,
            reason=reason,
            outcome=outcome,
        )

    async def fail_owned_segment(self, event: object, *, reason: str) -> None:
        self._require_ingress()
        if not isinstance(event, OwnedVadEvent):
            raise TypeError("Self recognition failure requires OwnedVadEvent")
        await self.local_asr_runtime.fail_owned_segment(
            "self",
            event,
            reason=reason,
        )

    async def handle_source_segment_retired(self, receipt: AudioSegmentTerminalReceipt) -> None:
        if receipt.outcome not in {"cancelled", "failed", "expired"}:
            return
        if receipt.identity not in self._acoustic_continuations:
            return
        self._settle_acoustic_continuation(receipt.identity)
        buffer = self.merge_buffer
        if buffer is not None:
            await self._evaluate_speculative_next_action(buffer, reason="source_segment_retired")

    def release_input_segment(self, event: STTProviderInputTerminal) -> None:
        segment_id = event.identity.segment.segment_id
        if event.independent_recognition:
            observation = self._acoustic_continuations.get(event.identity.segment)
            if (
                observation is not None
                and observation.recognition_scope == event.identity.settings_scope
            ):
                self._settle_acoustic_continuation(event.identity.segment)
        self._clear_runtime_latency_bookkeeping(segment_id)
        self._scoped_endpoint_publication_ids.pop(segment_id, None)
        self._retired_local_vad_ids[segment_id] = None
        self._retired_local_vad_ids.move_to_end(segment_id)
        if len(self._retired_local_vad_ids) > 4096:
            self._retired_local_vad_ids.popitem(last=False)

    async def submit_text(self, text: str, *, source: str = "You") -> TextSubmissionAdmission:
        self._require_ingress()
        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")
        operation_identity = current_diagnostic_operation()
        utterance_id = uuid4()
        self.runtime.remember_source(utterance_id, source)
        transcript = Transcript(
            utterance_id=utterance_id,
            text=text,
            is_final=True,
            created_at=self.clock.now(),
        )
        await self._handle_transcript(
            transcript, is_final=True, source=source, **operation_identity
        )
        child_ids = await self._ensure_translation(
            transcript,
            turn_kind="manual",
            wait_for_parent=(
                not self.translation_requests.provider_available
                or not self.config_snapshot().value.translation_enabled
            ),
            **operation_identity,
        )
        return TextSubmissionAdmission(
            status="admitted" if child_ids else "rejected",
            turn_id=utterance_id,
            child_turn_ids=child_ids,
        )

    async def handle_recognition_unit(self, unit: STTRecognitionUnit) -> None:
        self._require_ingress()
        if unit.identity.stream.channel != "self":
            raise ValueError("Self translation owner received non-Self recognition unit")
        transcript = recognition_transcript(unit, created_at=self.clock.now())
        self.diagnostics.record_recognition_latency(unit)
        await self._handle_transcript(transcript, is_final=True, source="Mic")
        await self._ensure_translation(
            transcript,
            turn_kind="self",
            wait_for_parent=(
                not self.translation_requests.provider_available
                or not self.translation_requests.translation_enabled_for("self")
            ),
        )

    async def handle_stt_event(self, event: object) -> None:
        self._require_ingress()
        utterance_id = getattr(event, "utterance_id", None)
        self.diagnostics.record_stt_ingress(
            "stt_handler_start",
            channel="self",
            event_type=type(event).__name__,
            utterance_id=None if utterance_id is None else str(utterance_id),
        )
        low_latency_mode = self.config_snapshot().value.low_latency_mode
        event = scoped_event_for_delivery(event)
        if event is None:
            return
        if isinstance(event, STTProviderTurnUpdate | STTProviderTurnTerminal):
            await self._handle_scoped_stt_event(event, low_latency_mode=low_latency_mode)
            return
        if isinstance(event, STTProviderEpochEnded):
            return
        if isinstance(event, STTSessionStateEvent):
            if event.channel != "self":
                raise ValueError("Self translation owner received a non-Self session event")
            self._emit_basic(
                "[Translation] STT state: channel=%s state=%s",
                event.channel,
                event.state.name,
            )
            await self.output_projection.publish_ui(
                TranslationUiMessage(
                    event_type=UIEventType.SESSION_STATE_CHANGED,
                    payload=event.state,
                    channel=event.channel,
                )
            )
            if event.state == STTSessionState.STREAMING:
                self._send_stt_connected_notification()
            return
        if isinstance(event, STTErrorEvent):
            if event.channel != "self":
                raise ValueError("Self translation owner received a non-Self error event")
            await self.output_projection.publish_ui(
                TranslationUiMessage(
                    event_type=UIEventType.ERROR,
                    payload=self._stt_error_event_payload(event),
                    source="Mic",
                    channel="self",
                    runtime_log_handled=event.runtime_log_handled,
                )
            )
            return
        if isinstance(event, STTPartialEvent):
            if event.channel != "self":
                raise ValueError("Self translation owner received a non-Self partial event")
            self._send_stt_connected_notification()
            if low_latency_mode:
                return
            self._emit_diagnostic(
                "[Translation] STT Partial: channel=%s utterance_id=%s text_len=%s",
                event.channel,
                event.transcript.utterance_id,
                len(event.transcript.text),
                fallback_level=logging.DEBUG,
            )
            await self._handle_transcript(event.transcript, is_final=False, source="Mic")
            return
        if isinstance(event, STTFinalEvent):
            if event.channel != "self":
                raise ValueError("Self translation owner received a non-Self final event")
            self._send_stt_connected_notification()
            if low_latency_mode:
                await self._handle_low_latency_final(event.transcript)
                return
            self._record_latency_stage(
                utterance_id=event.transcript.utterance_id,
                stage="stt_final",
            )
            await self._handle_transcript(event.transcript, is_final=True, source="Mic")
            await self._ensure_translation(
                event.transcript,
                turn_kind="self",
                wait_for_parent=(
                    not self.translation_requests.provider_available
                    or not self.translation_requests.translation_enabled_for("self")
                ),
            )

    async def _handle_scoped_stt_event(
        self,
        event: STTProviderTurnUpdate | STTProviderTurnTerminal,
        *,
        low_latency_mode: bool,
    ) -> None:
        if isinstance(event, STTProviderTurnUpdate):
            if event.stability != "stable":
                return
            text, origins = self._stt_consumption.consume_with_origins(event)
            if not text:
                return
            if low_latency_mode:
                await self._apply_scoped_contribution(event.identity, text, origins)
            return

        text, origins = self._stt_consumption.consume_with_origins(event)
        try:
            if low_latency_mode:
                if text:
                    await self._apply_scoped_contribution(event.identity, text, origins)
                await self._settle_scoped_terminal(event.identity)
            elif event.text and event.text_authority != "none":
                utterance_id = event.identity.segment.segment_id
                transcript = Transcript(
                    utterance_id=utterance_id,
                    text=event.text,
                    is_final=True,
                    created_at=self.clock.now(),
                    channel="self",
                    final_language_runs=event.final_language_runs,
                    final_speaker_runs=event.final_speaker_runs,
                    recognition_origins=event.recognition_origins,
                )
                self._record_latency_stage(utterance_id=utterance_id, stage="stt_final")
                await self._handle_transcript(transcript, is_final=True, source="Mic")
                await self._ensure_translation(
                    transcript,
                    turn_kind="self",
                    wait_for_parent=(
                        not self.translation_requests.provider_available
                        or not self.translation_requests.translation_enabled_for("self")
                    ),
                )
            elif event.outcome not in {"final", "degraded"} or not event.text:
                self._emit_diagnostic(
                    "[Pipeline] turn_result channel=self utterance_id=%s "
                    "origin=self recognition=%s provider_epoch=%s provider_turn=%s "
                    "cause=%s",
                    event.identity.segment.segment_id,
                    event.outcome,
                    event.identity.provider_epoch_id,
                    event.identity.provider_turn_id,
                    recognition_cause(event.failure_reason),
                    level=(logging.ERROR if event.outcome == "failed" else logging.INFO),
                )
            if (
                self._scoped_terminal_requires_user_error(event)
                and event.identity not in self._scoped_terminal_reports
            ):
                self._scoped_terminal_reports[event.identity] = None
                if len(self._scoped_terminal_reports) > 4096:
                    self._scoped_terminal_reports.popitem(last=False)
                scope_provider = (
                    event.identity.settings_scope[0] if event.identity.settings_scope else "stt"
                )
                report = self.diagnostics.record_stt_turn_failure(
                    SttTurnFailureDiagnostic(
                        exception=None,
                        failure_diagnostics=event.failure_diagnostics
                        or recognition_failure_diagnostics(
                            None,
                            provider=str(scope_provider),
                            channel="self",
                            operation="turn_terminal",
                            reason=event.failure_reason or event.outcome,
                            identity=event.identity,
                        ),
                        provider=scope_provider,
                        channel="self",
                    )
                )
                if not event.recovery_pending:
                    await self.output_projection.publish_ui(
                        TranslationUiMessage(
                            event_type=UIEventType.ERROR,
                            payload=report,
                            source="Mic",
                            channel="self",
                        )
                    )
        finally:
            self._stt_consumption.retire(event.identity)
            self._scoped_publication_ids.pop(event.identity, None)
            self._scoped_publication_text.pop(event.identity, None)
            self._scoped_publication_origins.pop(event.identity, None)

    async def _apply_scoped_contribution(
        self,
        identity: STTProviderTurnIdentity,
        text: str,
        origins: tuple[RecognitionTextOrigin, ...] = (),
    ) -> None:
        recognition_scope = self._recognition_scope(identity)
        self._settle_acoustic_continuation(identity.segment)
        buffer = self.merge_buffer
        if (
            buffer is not None
            and buffer.recognition_scope is not None
            and buffer.recognition_scope != recognition_scope
        ):
            await self._commit_recognition_configuration_barrier(buffer)
        elif buffer is not None and buffer.recognition_scope is None and not buffer.parts:
            buffer.recognition_scope = recognition_scope
        publication_id = self._scoped_publication_ids.get(identity)
        if (
            publication_id is None
            or publication_id in self.runtime.low_latency_committed_utterance_ids
        ):
            publication_id = identity.segment.segment_id if publication_id is None else uuid4()
            self._scoped_publication_ids[identity] = publication_id
            self._scoped_publication_text[identity] = ""
            self._scoped_publication_origins[identity] = ()
        acoustic_id = identity.segment.segment_id
        self._scoped_endpoint_publication_ids[acoustic_id] = publication_id
        cumulative = self._scoped_publication_text.get(identity, "") + text
        offset = len(self._scoped_publication_text.get(identity, ""))
        self._scoped_publication_origins[identity] = self._scoped_publication_origins.get(
            identity, ()
        ) + tuple(
            replace(
                origin, text_start=origin.text_start + offset, text_end=origin.text_end + offset
            )
            for origin in origins
        )
        self._scoped_publication_text[identity] = cumulative
        self._inherit_latency_for_output(
            output_utterance_id=publication_id,
            source_utterance_ids=(acoustic_id,),
        )
        if acoustic_id in self.runtime.speech_ended_ids:
            self.runtime.speech_ended_ids.add(publication_id)
            end_time = self.runtime.utterance_start_times.get(acoustic_id)
            if end_time is not None:
                self.runtime.utterance_start_times[publication_id] = end_time
        await self._handle_low_latency_final(
            Transcript(
                utterance_id=publication_id,
                text=cumulative,
                is_final=True,
                created_at=self.clock.now(),
                channel="self",
                recognition_origins=self._scoped_publication_origins[identity],
            ),
            recognition_scope=recognition_scope,
            source_identity=identity.segment,
        )

    async def _settle_scoped_terminal(self, identity: STTProviderTurnIdentity) -> None:
        acoustic_id = identity.segment.segment_id
        self._settle_acoustic_continuation(identity.segment)
        buffer = self.merge_buffer
        if buffer is None:
            return
        if buffer.resume_utterance_id == acoustic_id:
            self._clear_resume_state(buffer)
        publication_id = self._scoped_publication_ids.get(identity)
        if publication_id is None:
            await self._evaluate_speculative_next_action(
                buffer,
                reason="recognition_terminal",
            )
            return
        if acoustic_id in self.runtime.speech_ended_ids:
            self.runtime.speech_ended_ids.add(publication_id)
            end_time = self.runtime.utterance_start_times.get(acoustic_id)
            if end_time is not None:
                self.runtime.utterance_start_times[publication_id] = end_time
            self._maybe_start_finalize_wait(publication_id)
            await self._evaluate_speculative_next_action(
                buffer,
                reason="recognition_terminal",
            )

    @staticmethod
    def _recognition_scope(identity: STTProviderTurnIdentity) -> tuple[object, ...]:
        return identity.settings_scope

    async def _commit_recognition_configuration_barrier(
        self,
        buffer: _MergeBuffer,
    ) -> None:
        for key, observation in tuple(self._acoustic_continuations.items()):
            if observation.recognition_scope == buffer.recognition_scope:
                self._settle_acoustic_continuation(key)
        self._cancel_awaiting_vad_timeout(buffer)
        self._cancel_finalize_wait(buffer)
        self._clear_resume_state(buffer)
        buffer.awaiting_vad_end = False
        buffer.awaiting_vad_utterance_id = None
        await self._commit_merge(buffer, reason="recognition_configuration_barrier")

    def _inherit_scoped_publication_end(
        self,
        acoustic_id: UUID,
        end_time: float,
    ) -> None:
        publication_id = self._scoped_endpoint_publication_ids.pop(acoustic_id, None)
        if publication_id is None:
            return
        self.runtime.speech_ended_ids.add(publication_id)
        self.runtime.utterance_start_times[publication_id] = end_time
        self._inherit_latency_for_output(
            output_utterance_id=publication_id,
            source_utterance_ids=(acoustic_id,),
        )
        self._maybe_update_buffer_end_time(publication_id)
        self._maybe_start_finalize_wait(publication_id)

    @staticmethod
    def _scoped_terminal_requires_user_error(event: STTProviderTurnTerminal) -> bool:
        if event.outcome in ("final", "empty", "suppressed"):
            return False
        if event.outcome == "cancelled" and event.failure_reason in {
            "cancelled",
            "closed",
            "stopped",
            "toggle_off",
            "stream_input_ownership_changed",
        }:
            return False
        return True

    def _clear_scoped_recognition_state(self) -> None:
        for key in tuple(self._acoustic_continuations):
            self._settle_acoustic_continuation(key)
        for identity in tuple(self._scoped_publication_ids):
            self._stt_consumption.retire(identity)
        self._scoped_publication_ids.clear()
        self._scoped_publication_text.clear()
        self._scoped_publication_origins.clear()
        self._scoped_endpoint_publication_ids.clear()

    async def handle_retired_stt_event(self, event: object) -> None:
        if isinstance(event, STTProviderTurnUpdate | STTProviderTurnTerminal):
            await self.handle_stt_event(event)

    async def handle_stt_event_loop_exception(self, exc: Exception) -> None:
        self.diagnostics.record_stt_event_loop_failure(
            SttEventLoopFailureDiagnostic(
                exception=exc,
                provider=None,
                default_channel="self",
            )
        )

    async def on_child_created(self, child: TranslationTurnChild) -> None:
        if child.channel != "self":
            raise ValueError("Self translation owner received a non-Self child")
        if child.draft:
            return
        if child.target_index != 0:
            self.runtime.get_or_create_bundle(child.utterance_id).with_transcript(child.transcript)
            self.runtime.remember_source(child.utterance_id, child.source)
            return
        if child.utterance_id != child.parent_utterance_id:
            await self._handle_transcript(
                child.transcript,
                is_final=True,
                source=child.source,
                parent_utterance_id=child.parent_utterance_id,
                operation_id=child.operation_id,
                request_id=child.request_id,
            )

    async def on_parent_admitted(
        self,
        children: tuple[TranslationTurnChild, ...],
    ) -> None:
        if any(child.channel != "self" for child in children):
            raise ValueError("Self translation owner received a non-Self parent")
        if children and children[0].draft:
            if not self.translation_requests.provider_available:
                return
            for child in children:
                prepared = child.prepared_request
                if prepared is None or not child.config_snapshot.value.translation_enabled:
                    continue
                self.translation_requests.remember_context(
                    child.transcript.text,
                    self.clock.now(),
                    channel="self",
                    config_snapshot=child.config_snapshot,
                    source_language=prepared.source_language,
                    target_language=child.target_language,
                    origin="self",
                )
            return
        if not self.output_projection.admit_self_turn(children):
            raise RuntimeError("Self translation projection rejected parent admission")
        if not await self.output_projection.admit_translation_parent(children):
            raise RuntimeError("Self translation output admission rejected parent")
        admitted = self.translation_requests.admit(
            tuple(self._process_request_for_child(child) for child in children)
        )
        self._admitted_requests.update(admitted)

    async def process_child(
        self,
        child: TranslationTurnChild,
        cancellation_requested: Callable[[], bool],
    ) -> TranslationTurnProcessResult:
        if child.channel != "self":
            raise ValueError("Self translation owner received a non-Self child")
        if cancellation_requested():
            raise asyncio.CancelledError
        request = self._process_request_for_child(child)
        target_language = request.target_language
        prepared = self._admitted_requests.pop(child.utterance_id, None)
        if child.draft:
            prepared = child.prepared_request
            if child.preparation_failure is not None:
                raise child.preparation_failure
            request = replace(
                request,
                expected_provider_generation=(
                    self.translation_requests.provider_generation
                    if child.canonical_attempt
                    else child.provider_generation
                ),
                prestarted=not child.canonical_attempt,
            )
            if child.canonical_attempt and prepared is not None:
                prepared = replace(
                    prepared, provider_generation=self.translation_requests.provider_generation
                )
        if child.precomputed_translation is not None:
            if prepared is None:
                self.translation_requests.remember_context(
                    child.transcript.text,
                    self.clock.now(),
                    channel="self",
                    config_snapshot=child.config_snapshot,
                    source_language=child.precomputed_translation.source_language,
                    target_language=target_language,
                    origin=child.turn_kind,
                )
            return TranslationTurnProcessResult(
                "translated",
                TranslationOutputSubmission(
                    parent_utterance_id=child.parent_utterance_id,
                    child_utterance_id=child.utterance_id,
                    sequence=child.sequence,
                    channel="self",
                    source=child.source,
                    source_text=child.transcript.text,
                    source_language=child.detected_language,
                    target_language=target_language,
                    outcome="translated",
                    config_snapshot=child.config_snapshot,
                    translation=child.precomputed_translation,
                    target_index=child.target_index,
                    turn_generation=child.turn_generation,
                    turn_order=child.turn_order,
                    turn_kind=child.turn_kind,
                    parent_output_count=child.parent_output_count,
                    context_texts=None if prepared is None else prepared.context_texts,
                ),
            )
        result = await self.translation_requests.process(
            request,
            cancellation_requested=cancellation_requested,
            prepared=prepared,
        )
        if cancellation_requested():
            raise asyncio.CancelledError
        return result

    async def on_child_started(
        self,
        child: TranslationTurnChild,
        task: asyncio.Task[TranslationTurnProcessResult],
    ) -> None:
        if child.channel != "self":
            raise ValueError("Self translation owner received a non-Self child")
        if child.draft:
            return
        self.runtime.translation_tasks[child.utterance_id] = task
        if len(child.config_snapshot.value.self_target_languages) == 2:
            presentation_revision = self.output_projection.self_turn_presentation_revision(child)
            self.diagnostics.emit(
                RuntimeDiagnostic(
                    message=(
                        "[Diagnostic][Translation] translation_target_started "
                        "parent_utterance_id=%s turn_generation=%s turn_order=%s "
                        "target_index=%s target_language=%s presentation_revision=%s "
                        "precomputed=%s"
                    ),
                    args=(
                        child.parent_utterance_id,
                        child.turn_generation,
                        child.turn_order,
                        child.target_index,
                        child.target_language,
                        presentation_revision,
                        child.precomputed_translation is not None,
                    ),
                    diagnostic_only=True,
                )
            )

    async def on_child_terminal(
        self,
        child: TranslationTurnChild,
        outcome: TranslationTurnOutcome,
    ) -> None:
        if child.channel != "self":
            raise ValueError("Self translation owner received a non-Self child")
        if child.draft:
            if child.canonical_attempt and not self.translation_turns.child_output_was_submitted(
                child.utterance_id
            ):
                self.output_projection.record_child_terminal_conversation(child, outcome)
            self._clear_runtime_latency_bookkeeping(child.utterance_id)
            return
        prepared = self._admitted_requests.pop(child.utterance_id, None)
        self.runtime.translation_tasks.pop(child.utterance_id, None)
        if not self.translation_turns.child_output_was_submitted(child.utterance_id):
            self.output_projection.record_child_terminal_conversation(
                child,
                outcome,
                context_texts=None if prepared is None else prepared.context_texts,
            )
        await self.output_projection.complete_self_target(child, outcome)
        dual_target = len(child.config_snapshot.value.self_target_languages) == 2
        await self.output_projection.complete_translation_parent_output(
            parent_utterance_id=child.parent_utterance_id,
            channel=child.channel,
            turn_kind=child.turn_kind,
            sequence=child.sequence,
            target_index=child.target_index,
            dual_target_self=dual_target,
        )
        if outcome != "cancelled":
            return
        if len(child.config_snapshot.value.self_target_languages) == 2:
            async with self.output_projection.self_primary_presentation(
                parent_utterance_id=child.parent_utterance_id,
                target_index=child.target_index,
                turn_generation=child.turn_generation,
                turn_order=child.turn_order,
            ) as present_self_result:
                if not present_self_result:
                    return
                finalized = await self.output_projection.close_overlay_utterance(
                    utterance_id=child.utterance_id,
                    channel="self",
                    is_final=False,
                    finalize_latency=False,
                )
        else:
            finalized = await self.output_projection.close_overlay_utterance(
                utterance_id=child.utterance_id,
                channel="self",
                is_final=False,
                finalize_latency=False,
            )
        if finalized:
            self._clear_runtime_latency_bookkeeping(child.utterance_id)

    def _with_prepared_context(
        self,
        submission: TranslationOutputSubmission,
    ) -> TranslationOutputSubmission:
        if submission.context_texts is not None:
            return submission
        prepared = self._admitted_requests.get(submission.child_utterance_id)
        if prepared is None:
            return submission
        return replace(submission, context_texts=prepared.context_texts)

    async def submit_translation_output(self, submission: TranslationOutputSubmission) -> None:
        if submission.channel != "self":
            raise ValueError("Self translation owner received non-Self output")
        if self.translation_turns.is_draft_parent(submission.parent_utterance_id):
            self.output_projection.record_final_self_submission(submission)
            return
        submission = self._with_prepared_context(submission)
        translation = submission.translation
        dual_target = len(submission.config_snapshot.value.self_target_languages) == 2
        admitted_destinations = await self.output_projection.await_translation_parent_output(
            submission
        )
        try:
            if translation is not None and not dual_target:
                self.runtime.get_or_create_bundle(submission.child_utterance_id).with_translation(
                    translation
                )
            receipt = await self.output_projection.project_translation_result(
                submission,
                admitted_destinations=admitted_destinations,
            )
            if translation is not None and dual_target and receipt.record_runtime_translation:
                self.runtime.get_or_create_bundle(submission.child_utterance_id).with_translation(
                    translation
                )
            if receipt.clear_runtime_latency_bookkeeping:
                self._clear_runtime_latency_bookkeeping(submission.child_utterance_id)
        finally:
            await self.output_projection.complete_translation_parent_output(
                parent_utterance_id=submission.parent_utterance_id,
                channel=submission.channel,
                turn_kind=submission.turn_kind or submission.channel,
                sequence=submission.sequence,
                target_index=submission.target_index,
                dual_target_self=dual_target,
                destinations=admitted_destinations,
            )

    async def translate_and_enqueue(
        self,
        utterance_id: UUID,
        text: str,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> None:
        config_snapshot = self.config_snapshot()
        if cancellation_requested is not None and cancellation_requested():
            return
        source = self.runtime.get_source(utterance_id) or "Mic"
        target_languages = config_snapshot.value.self_target_languages or (
            self.translation_requests.target_language_for(
                "self",
                config_snapshot.value,
            ),
        )
        await self.translation_turns.submit(
            TranslationTurnRequest(
                transcript=Transcript(
                    utterance_id=utterance_id,
                    text=text,
                    is_final=True,
                    created_at=self.clock.now(),
                    channel="self",
                ),
                source=source,
                turn_kind="self",
                target_languages=target_languages,
                config_snapshot=config_snapshot,
            ),
            wait_for_parent=True,
        )

    async def _handle_transcript(
        self,
        transcript: Transcript,
        *,
        is_final: bool,
        source: str | None,
        parent_utterance_id: UUID | None = None,
        operation_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        if transcript.channel != "self":
            raise ValueError("Self translation owner received a non-Self transcript")
        self.runtime.get_or_create_bundle(transcript.utterance_id).with_transcript(transcript)
        self.runtime.remember_source(transcript.utterance_id, source)
        configuration = self.config_snapshot().value
        async with self.output_projection.self_transcript_presentation(
            parent_utterance_id=parent_utterance_id or transcript.utterance_id,
        ):
            await self.output_projection.publish_ui(
                TranslationUiMessage(
                    event_type=(
                        UIEventType.TRANSCRIPT_FINAL if is_final else UIEventType.TRANSCRIPT_PARTIAL
                    ),
                    utterance_id=transcript.utterance_id,
                    payload=transcript,
                    source=source,
                    parent_utterance_id=parent_utterance_id,
                    operation_id=operation_id,
                    request_id=request_id,
                )
            )
            if not is_final:
                return
            turn_kind = "manual" if source not in {None, "Mic"} else "self"
            recognition = "not_applicable" if turn_kind == "manual" else "completed"
            self._emit_diagnostic(
                "[Pipeline] turn_result channel=self utterance_id=%s "
                "origin=%s recognition=%s source_language=%s",
                transcript.utterance_id,
                turn_kind,
                recognition,
                self.translation_requests.source_language_for(
                    "self",
                    configuration,
                ),
            )
            finalized = await self.output_projection.project_self_final_transcript(
                transcript=transcript,
                source_language=self.translation_requests.source_language_for(
                    "self",
                    configuration,
                ),
                target_language=self.translation_requests.target_language_for(
                    "self",
                    configuration,
                ),
                translation_will_follow=self._overlay_translation_will_follow(configuration),
            )
        if is_final and finalized:
            self._clear_runtime_latency_bookkeeping(transcript.utterance_id)

    async def _ensure_translation(
        self,
        transcript: Transcript,
        *,
        turn_kind: TranslationTurnKind = "self",
        precomputed_translation: Translation | None = None,
        precomputed_failure: TerminalTranslationFailure | None = None,
        wait_for_parent: bool = False,
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
        operation_id: str | None = None,
        request_id: str | None = None,
    ) -> tuple[UUID, ...]:
        config_snapshot = config_snapshot or self.config_snapshot()
        source = self.runtime.get_source(transcript.utterance_id) or "Mic"
        child_ids = await self.translation_turns.submit(
            TranslationTurnRequest(
                transcript=transcript,
                source=source,
                turn_kind=turn_kind,
                target_languages=config_snapshot.value.self_target_languages,
                precomputed_translation=precomputed_translation,
                precomputed_failure=precomputed_failure,
                config_snapshot=config_snapshot,
                operation_id=operation_id,
                request_id=request_id,
            ),
            wait_for_parent=wait_for_parent,
        )
        return child_ids

    def _process_request_for_child(
        self,
        child: TranslationTurnChild,
    ) -> TranslationProcessRequest:
        target_language = (
            self.translation_requests.target_language_for("self", child.config_snapshot.value)
            if child.target_language == "und"
            else child.target_language
        )
        return TranslationProcessRequest(
            parent_utterance_id=child.parent_utterance_id,
            utterance_id=child.utterance_id,
            sequence=child.sequence,
            text=child.transcript.text,
            channel="self",
            source=child.source,
            target_language=target_language,
            context_policy=child.context_policy,
            detected_language=child.detected_language,
            config_snapshot=child.config_snapshot,
            target_index=child.target_index,
            turn_generation=child.turn_generation,
            turn_order=child.turn_order,
            turn_kind=child.turn_kind,
            parent_output_count=child.parent_output_count,
            terminal_failure=child.precomputed_failure,
            operation_id=child.operation_id,
            request_id=child.request_id,
        )

    def _send_stt_connected_notification(self) -> None:
        if not self._promo_eligible:
            return
        self._promo_eligible = False
        now = self.clock.now()
        if self._last_promo_time is not None:
            if now - self._last_promo_time < _PROMO_INTERVAL_SEC:
                return
        result = self.output_projection.publish_system_immediate("PuriPuly ON!")
        if result.decision.decision == "published":
            self._last_promo_time = now

    def _overlay_translation_will_follow(
        self,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> bool:
        return (
            self.output_projection.has_overlay_destination
            and self.translation_requests.provider_available
            and self.translation_requests.translation_enabled_for("self", configuration)
        )

    def _require_ingress(self) -> None:
        if not self._accepting_events:
            raise RuntimeError("Self translation ingress is closed")

    def _emit_basic(
        self,
        message: str,
        *args: object,
        level: int = logging.INFO,
        fallback_level: int | None = None,
    ) -> None:
        self.diagnostics.emit(
            RuntimeDiagnostic(
                message=message,
                args=args,
                level=level,
                fallback_level=fallback_level,
            )
        )

    def _emit_diagnostic(
        self,
        message: str,
        *args: object,
        level: int = logging.INFO,
        fallback_level: int | None = None,
    ) -> bool:
        return self.diagnostics.emit(
            RuntimeDiagnostic(
                message=message,
                args=args,
                level=level,
                fallback_level=fallback_level,
                diagnostic_only=True,
            )
        )

    def _emit_metric(self, message: str, *args: object) -> None:
        return

    def _record_latency_stage(
        self,
        *,
        utterance_id: UUID,
        stage: str,
        timestamp: float | None = None,
        overwrite: bool = True,
        publish_now: bool = True,
    ) -> None:
        self.diagnostics.record_latency_stage(
            LatencyStageDiagnostic(
                channel="self",
                utterance_id=utterance_id,
                stage=stage,
                timestamp=timestamp,
                overwrite=overwrite,
                publish_now=publish_now,
            )
        )

    def _inherit_latency_for_output(
        self,
        *,
        output_utterance_id: UUID,
        source_utterance_ids: tuple[UUID, ...],
    ) -> None:
        self.diagnostics.inherit_latency(
            LatencyInheritanceDiagnostic(
                channel="self",
                output_utterance_id=output_utterance_id,
                source_utterance_ids=source_utterance_ids,
            )
        )

    def _clear_latency_timeline(self, utterance_id: UUID) -> None:
        self.diagnostics.clear_latency_timeline(channel="self", utterance_id=utterance_id)

    def _clear_runtime_latency_bookkeeping(self, utterance_id: UUID) -> None:
        self.runtime.utterance_start_times.pop(utterance_id, None)
        self.runtime.speech_ended_ids.discard(utterance_id)
        self._clear_latency_timeline(utterance_id)

    def _finalize_latency_timeline(self, utterance_id: UUID) -> None:
        self.diagnostics.publish_latency(
            LatencyTimelineDiagnostic(channel="self", utterance_id=utterance_id)
        )

    def _log_translation_failure(
        self,
        *,
        stage: str,
        exc: Exception,
        diagnostic_only: bool = False,
    ) -> UserErrorReport:
        return self.diagnostics.record_translation_failure(
            TranslationFailureDiagnostic(
                stage=stage,
                channel="self",
                exception=exc,
                diagnostic_only=diagnostic_only,
            )
        )

    @staticmethod
    def _stt_error_event_payload(event: STTErrorEvent) -> UIErrorPayload | None:
        if isinstance(event.message, UserMessageRef) and event.diagnostics is not None:
            return UserErrorReport(message=event.message, diagnostics=event.diagnostics)
        return event.message

    def _merge_text(self, parts: list[str], *, preserve_text: bool = False) -> str:
        merged = ""
        for part in parts:
            addition = part if preserve_text else part.strip()
            if not addition:
                continue
            if not merged:
                merged = addition
            elif self._needs_space(merged, addition):
                merged = f"{merged} {addition}"
            else:
                merged = f"{merged}{addition}"
        return merged

    def _merged_origins(self, buffer: _MergeBuffer) -> tuple[RecognitionTextOrigin, ...]:
        text = ""
        origins: tuple[RecognitionTextOrigin, ...] = ()
        for utterance_id, part in zip(buffer.utterance_ids, buffer.parts, strict=True):
            if text and self._needs_space(text, part):
                text += " "
            offset = len(text)
            origins += tuple(
                replace(
                    origin, text_start=origin.text_start + offset, text_end=origin.text_end + offset
                )
                for origin in buffer.recognition_origins.get(utterance_id, ())
            )
            text += part
        return origins

    @staticmethod
    def _needs_space(left: str, right: str) -> bool:
        if not left or not right:
            return False
        left_ch = left[-1]
        right_ch = right[0]
        if ord(left_ch) < 128 and ord(right_ch) < 128:
            return left_ch.isalnum() and right_ch.isalnum()
        return (" " in left or " " in right) and left_ch.isalnum() and right_ch.isalnum()

    def _upsert_merge_part(
        self,
        buffer: _MergeBuffer,
        utterance_id: UUID,
        text: str,
    ) -> None:
        if not text:
            return
        for idx in range(len(buffer.utterance_ids) - 1, -1, -1):
            if buffer.utterance_ids[idx] != utterance_id:
                continue
            existing = buffer.parts[idx]
            if existing == text:
                return
            buffer.parts[idx] = text
            self._emit_metric(
                "[Metric] final_update id=%s index=%s text_len=%s",
                str(buffer.merge_id)[:8],
                idx,
                len(text),
            )
            return
        buffer.parts.append(text)
        buffer.utterance_ids.append(utterance_id)

    def _settle_acoustic_continuation(self, key: AudioSegmentIdentity | UUID) -> None:
        observation = self._acoustic_continuations.pop(key, None)
        if observation is None:
            return
        task = observation.timeout_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        self._refresh_resume_state()

    def _refresh_resume_state(self) -> None:
        buffer = self.merge_buffer
        if buffer is None:
            return
        self._clear_resume_state(buffer)
        observation = next(reversed(self._acoustic_continuations.values()), None)
        if observation is None:
            return
        buffer.resume_pending = True
        buffer.resume_confirmed = observation.chunk_count >= 3
        buffer.resume_utterance_id = observation.utterance_id
        buffer.resume_chunk_count = observation.chunk_count
        buffer.resume_started_at = observation.started_at
        if observation.timeout_task is not None:
            buffer.resume_end_utterance_id = observation.utterance_id
            buffer.resume_end_timeout_task = observation.timeout_task

    def _clear_resume_state(self, buffer: _MergeBuffer) -> None:
        buffer.resume_pending = False
        buffer.resume_confirmed = False
        buffer.resume_utterance_id = None
        buffer.resume_chunk_count = 0
        buffer.resume_started_at = None
        self._cancel_resume_end_timeout(buffer)

    def _maybe_update_buffer_end_time(self, utterance_id: UUID) -> None:
        buffer = self.merge_buffer
        if buffer is None or utterance_id not in buffer.utterance_ids:
            return
        end_time = self.runtime.utterance_start_times.get(utterance_id)
        if end_time is None:
            return
        if buffer.start_time is None or end_time < buffer.start_time:
            buffer.start_time = end_time
        if buffer.last_end_time is None or end_time > buffer.last_end_time:
            buffer.last_end_time = end_time

    def _cancel_finalize_wait(self, buffer: _MergeBuffer) -> None:
        task = buffer.finalize_wait_task
        if task is not None and task is not asyncio.current_task():
            if not task.done():
                task.cancel()
        buffer.finalize_wait_task = None
        buffer.finalize_wait_started_at = None

    def _maybe_start_finalize_wait(self, utterance_id: UUID) -> None:
        buffer = self.merge_buffer
        if buffer is None:
            return
        if not buffer.awaiting_vad_end or buffer.awaiting_vad_utterance_id != utterance_id:
            return
        buffer.awaiting_vad_end = False
        buffer.awaiting_vad_utterance_id = None
        self._cancel_awaiting_vad_timeout(buffer)
        self._restart_post_end_grace(buffer)

    def _cancel_awaiting_vad_timeout(self, buffer: _MergeBuffer) -> None:
        task = buffer.awaiting_vad_timeout_task
        if task is not None and task is not asyncio.current_task():
            if not task.done():
                task.cancel()
        buffer.awaiting_vad_timeout_task = None

    def _start_awaiting_vad_timeout(self, buffer: _MergeBuffer) -> None:
        timeout_s = self.config_snapshot().value.low_latency_awaiting_vad_timeout_s
        if timeout_s <= 0:
            return
        self._cancel_awaiting_vad_timeout(buffer)
        buffer.awaiting_vad_timeout_task = asyncio.create_task(
            self._awaiting_vad_timeout(buffer.merge_id, timeout_s)
        )

    async def _awaiting_vad_timeout(self, merge_id: UUID, timeout_s: float) -> None:
        try:
            await asyncio.sleep(timeout_s)
        except asyncio.CancelledError:
            return
        buffer = self.merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        if not buffer.awaiting_vad_end:
            return
        self._emit_metric(
            "[Metric] awaiting_vad_timeout id=%s timeout_s=%s",
            str(merge_id)[:8],
            timeout_s,
        )
        buffer.awaiting_vad_end = False
        buffer.awaiting_vad_utterance_id = None
        buffer.awaiting_vad_timeout_task = None
        self._restart_post_end_grace(buffer)
        await self._evaluate_speculative_next_action(
            buffer,
            reason="awaiting_vad_timeout",
        )

    def _cancel_resume_end_timeout(self, buffer: _MergeBuffer) -> None:
        task = buffer.resume_end_timeout_task
        acoustic_owned = task is not None and any(
            observation.timeout_task is task
            for observation in self._acoustic_continuations.values()
        )
        if not acoustic_owned and task is not None and task is not asyncio.current_task():
            if not task.done():
                task.cancel()
        buffer.resume_end_timeout_task = None
        buffer.resume_end_utterance_id = None

    def _start_resume_end_timeout(
        self,
        key: AudioSegmentIdentity | UUID,
    ) -> None:
        observation = self._acoustic_continuations.get(key)
        timeout_s = self.config_snapshot().value.low_latency_awaiting_vad_timeout_s
        if observation is None or observation.timeout_task is not None:
            return
        observation.timeout_task = start_lifecycle_task(
            self._task_scope,
            self._resume_end_timeout(key, timeout_s),
            name=f"self-resume-end:{observation.utterance_id}",
        )
        self._refresh_resume_state()

    async def _resume_end_timeout(
        self,
        key: AudioSegmentIdentity | UUID,
        timeout_s: float,
    ) -> None:
        try:
            await asyncio.sleep(timeout_s)
        except asyncio.CancelledError:
            return
        observation = self._acoustic_continuations.get(key)
        buffer = self.merge_buffer
        if observation is None or observation.timeout_task is not asyncio.current_task():
            return
        utterance_id = observation.utterance_id
        self._settle_acoustic_continuation(key)
        self._emit_metric(
            "[Metric] resume_end_timeout id=%s vad_id=%s timeout_s=%s",
            str(buffer.merge_id)[:8] if buffer is not None else "none",
            str(utterance_id)[:8],
            timeout_s,
        )
        if buffer is not None:
            self._cancel_finalize_wait(buffer)
            await self._evaluate_speculative_next_action(
                buffer,
                reason="resume_end_timeout",
            )

    def _restart_post_end_grace(self, buffer: _MergeBuffer) -> None:
        wait_ms = self.config_snapshot().value.low_latency_finalize_wait_ms
        if wait_ms <= 0:
            self._cancel_finalize_wait(buffer)
            return
        self._cancel_finalize_wait(buffer)
        buffer.finalize_wait_started_at = self.clock.now()
        buffer.finalize_wait_task = asyncio.create_task(
            self._finalize_wait_timeout(
                buffer.merge_id,
                buffer.finalize_wait_started_at,
                wait_ms,
            )
        )
        self._emit_metric(
            "[Metric] post_end_grace_start id=%s wait_ms=%s",
            str(buffer.merge_id)[:8],
            wait_ms,
        )

    async def _finalize_wait_timeout(
        self,
        merge_id: UUID,
        started_at: float,
        wait_ms: int,
    ) -> None:
        try:
            await asyncio.sleep(wait_ms / 1000.0)
        except asyncio.CancelledError:
            return
        buffer = self.merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        if buffer.finalize_wait_started_at != started_at:
            return
        buffer.finalize_wait_task = None
        buffer.finalize_wait_started_at = None
        self._emit_metric(
            "[Metric] post_end_grace_timeout id=%s wait_ms=%s",
            str(merge_id)[:8],
            wait_ms,
        )
        await self._evaluate_speculative_next_action(
            buffer,
            reason="post_end_grace",
        )

    def _mark_resume_pending(
        self,
        event: SpeechStart,
        *,
        source_identity: AudioSegmentIdentity | None = None,
        recognition_scope: tuple[object, ...] | None = None,
    ) -> None:
        key = source_identity or event.utterance_id
        if key in self._acoustic_continuations:
            return
        self._acoustic_continuations[key] = _AcousticContinuation(
            utterance_id=event.utterance_id,
            started_at=self.clock.now(),
            recognition_scope=recognition_scope,
        )
        self._refresh_resume_state()
        buffer = self.merge_buffer
        if buffer is not None:
            self._emit_metric(
                "[Metric] resume_pending id=%s vad_id=%s",
                str(buffer.merge_id)[:8],
                str(event.utterance_id)[:8],
            )

    def _maybe_confirm_resume(
        self,
        event: SpeechChunk,
        *,
        source_identity: AudioSegmentIdentity | None = None,
    ) -> _MergeBuffer | None:
        observation = self._acoustic_continuations.get(source_identity or event.utterance_id)
        if observation is None or observation.ended_at is not None or observation.chunk_count >= 3:
            return None
        observation.chunk_count += 1
        self._refresh_resume_state()
        buffer = self.merge_buffer
        if observation.chunk_count < 3 or buffer is None:
            return None
        self._emit_metric(
            "[Metric] resume_confirmed id=%s confirm_ms=%s chunk_count=%s",
            str(buffer.merge_id)[:8],
            int((self.clock.now() - observation.started_at) * 1000),
            observation.chunk_count,
        )
        return buffer

    async def _maybe_clear_resume_on_end(
        self,
        event: SpeechEnd,
        *,
        source_identity: AudioSegmentIdentity | None = None,
    ) -> None:
        key = source_identity or event.utterance_id
        observation = self._acoustic_continuations.get(key)
        buffer = self.merge_buffer
        if observation is None:
            return
        if observation.ended_at is None:
            observation.ended_at = self.clock.now()
        if observation.chunk_count >= 3:
            self._start_resume_end_timeout(key)
            return
        false_ms = int((self.clock.now() - observation.started_at) * 1000)
        chunk_count = observation.chunk_count
        self._settle_acoustic_continuation(key)
        if buffer is not None:
            self._emit_metric(
                "[Metric] resume_false_start id=%s false_ms=%s chunk_count=%s",
                str(buffer.merge_id)[:8],
                false_ms,
                chunk_count,
            )
            await self._evaluate_speculative_next_action(
                buffer,
                reason="resume_false_start",
            )

    async def _handle_low_latency_final(
        self,
        transcript: Transcript,
        *,
        recognition_scope: tuple[object, ...] | None = None,
        source_identity: AudioSegmentIdentity | None = None,
    ) -> None:
        text = transcript.text if transcript.recognition_origins else transcript.text.strip()
        if not text:
            return
        if transcript.utterance_id in self.runtime.low_latency_committed_utterance_ids:
            self._emit_metric(
                "[Metric] final_duplicate_ignored vad_id=%s",
                str(transcript.utterance_id)[:8],
            )
            return

        self._record_latency_stage(
            utterance_id=transcript.utterance_id,
            stage="stt_final",
            publish_now=False,
        )

        now = self.clock.now()
        self._settle_acoustic_continuation(source_identity or transcript.utterance_id)
        buffer = self.merge_buffer
        if buffer is None:
            buffer = _MergeBuffer(merge_id=uuid4(), start_time=now, last_final_at=now)
            self.merge_buffer = buffer
            self._refresh_resume_state()
        if recognition_scope is not None:
            if buffer.recognition_scope is None:
                buffer.recognition_scope = recognition_scope
            elif buffer.recognition_scope != recognition_scope:
                raise RuntimeError("recognition configuration barrier was not applied")
        acoustic_id = (
            source_identity.segment_id if source_identity is not None else transcript.utterance_id
        )
        if buffer.resume_utterance_id == acoustic_id:
            self._clear_resume_state(buffer)
        self._upsert_merge_part(buffer, transcript.utterance_id, text)
        if transcript.recognition_origins:
            buffer.recognition_origins[transcript.utterance_id] = transcript.recognition_origins
        buffer.last_final_at = now

        end_time = self.runtime.utterance_start_times.get(transcript.utterance_id)
        speech_already_ended = transcript.utterance_id in self.runtime.speech_ended_ids

        if end_time is None and not speech_already_ended:
            buffer.awaiting_vad_end = True
            buffer.awaiting_vad_utterance_id = transcript.utterance_id
            self._cancel_finalize_wait(buffer)
            self._start_awaiting_vad_timeout(buffer)
            self._emit_metric(
                "[Metric] final_phase id=%s phase=pre_end vad_id=%s",
                str(buffer.merge_id)[:8],
                str(transcript.utterance_id)[:8],
            )
        else:
            self._maybe_update_buffer_end_time(transcript.utterance_id)
            if (
                buffer.awaiting_vad_end
                and buffer.awaiting_vad_utterance_id == transcript.utterance_id
            ):
                buffer.awaiting_vad_end = False
                buffer.awaiting_vad_utterance_id = None
            self._restart_post_end_grace(buffer)
            self._emit_metric(
                "[Metric] final_phase id=%s phase=post_end vad_id=%s",
                str(buffer.merge_id)[:8],
                str(transcript.utterance_id)[:8],
            )

        await self._maybe_restart_spec(buffer)
        await self._evaluate_speculative_next_action(
            buffer,
            reason="final_reconciled",
        )

    async def _commit_merge(self, buffer: _MergeBuffer, *, reason: str) -> None:
        if self.merge_buffer is not buffer:
            return
        blocker = self._continuation_blocker(buffer)
        if blocker is not None:
            self._emit_continuation_blocker(buffer, blocker=blocker, reason=reason)
            return
        await self._maybe_restart_spec(buffer)
        if buffer.last_end_time is not None:
            self.runtime.utterance_start_times[buffer.merge_id] = buffer.last_end_time
        elif buffer.start_time is not None:
            self.runtime.utterance_start_times[buffer.merge_id] = buffer.start_time
        self._inherit_latency_for_output(
            output_utterance_id=buffer.merge_id,
            source_utterance_ids=tuple(buffer.utterance_ids),
        )
        self._cancel_finalize_wait(buffer)
        self._cancel_awaiting_vad_timeout(buffer)
        buffer.awaiting_vad_end = False
        buffer.awaiting_vad_utterance_id = None
        for utterance_id in buffer.utterance_ids:
            self.runtime.utterance_start_times.pop(utterance_id, None)
            self.runtime.speech_ended_ids.discard(utterance_id)
            self.runtime.remember_low_latency_committed_utterance(utterance_id)
            self._clear_latency_timeline(utterance_id)
        self.merge_buffer = None
        await self.translation_turns.seal_draft(buffer.merge_id)

    def _continuation_blocker(self, buffer: _MergeBuffer) -> str | None:
        if self._acoustic_continuations or buffer.resume_pending or buffer.resume_confirmed:
            return "resume"
        if buffer.awaiting_vad_end:
            return "await_vad_end"
        if buffer.finalize_wait_task is not None:
            return "post_end_grace"
        return None

    def _emit_continuation_blocker(
        self, buffer: _MergeBuffer, *, blocker: str, reason: str
    ) -> None:
        self._emit_metric(
            "[Metric] commit_blocked id=%s blocker=%s reason=%s",
            str(buffer.merge_id)[:8],
            blocker,
            reason,
        )

    async def _maybe_restart_spec(self, buffer: _MergeBuffer) -> None:
        text = self._merge_text(buffer.parts, preserve_text=bool(buffer.recognition_origins))
        if not text:
            return
        self._inherit_latency_for_output(
            output_utterance_id=buffer.merge_id,
            source_utterance_ids=tuple(buffer.utterance_ids),
        )
        await self.translation_turns.revise_draft(
            TranslationTurnRequest(
                transcript=Transcript(
                    utterance_id=buffer.merge_id,
                    text=text,
                    is_final=True,
                    created_at=(
                        buffer.start_time if buffer.start_time is not None else self.clock.now()
                    ),
                    channel="self",
                    recognition_origins=self._merged_origins(buffer),
                ),
                source="Mic",
                turn_kind="self",
                target_languages=self.config_snapshot().value.self_target_languages,
                config_snapshot=self.config_snapshot(),
            ),
            provider_generation=self.translation_requests.provider_generation,
            compatible=lambda left, right: self.output_projection.soft_reuse_mode(left, right)
            is not None,
            configuration_compatible=self._translation_config_matches,
        )

    def _prepare_draft_child(self, child: TranslationTurnChild) -> TranslationTurnChild:
        if (
            not self.translation_requests.provider_available
            or not child.config_snapshot.value.translation_enabled
        ):
            return replace(child, prepared_request=None, preparation_failure=None)
        try:
            prepared = self.translation_requests.prepare(
                child.transcript.text,
                channel="self",
                target_language=child.target_language,
                context_policy=child.context_policy,
                config_snapshot=child.config_snapshot,
                parent_utterance_id=child.parent_utterance_id,
                target_index=child.target_index,
            )
        except Exception as exc:
            return replace(child, prepared_request=None, preparation_failure=exc)
        return replace(
            child,
            prepared_request=replace(prepared, provider_generation=child.provider_generation),
            preparation_failure=None,
        )

    @staticmethod
    def _translation_config_matches(
        left: TranslationRuntimeConfig,
        right: TranslationRuntimeConfig,
    ) -> bool:
        return (
            left.source_language,
            left.target_language,
            left.self_target_languages,
            left.peer_source_language,
            left.peer_target_language,
            left.peer_source_mode,
            left.system_prompt,
            left.translation_enabled,
            left.peer_translation_enabled,
            left.integrated_context_enabled,
            left.context_time_window_s,
            left.context_max_entries,
            left.integrated_context_time_window_s,
            left.integrated_context_max_entries,
        ) == (
            right.source_language,
            right.target_language,
            right.self_target_languages,
            right.peer_source_language,
            right.peer_target_language,
            right.peer_source_mode,
            right.system_prompt,
            right.translation_enabled,
            right.peer_translation_enabled,
            right.integrated_context_enabled,
            right.context_time_window_s,
            right.context_max_entries,
            right.integrated_context_time_window_s,
            right.integrated_context_max_entries,
        )

    async def _evaluate_speculative_next_action(self, buffer: _MergeBuffer, *, reason: str) -> None:
        if self.merge_buffer is not buffer:
            return
        blocker = self._continuation_blocker(buffer)
        if blocker == "post_end_grace" or blocker is None:
            self.translation_turns.prestart_draft_secondary(buffer.merge_id)
        if blocker is not None:
            self._emit_continuation_blocker(buffer, blocker=blocker, reason=reason)
            return
        await self._commit_merge(buffer, reason=reason)

    async def _sync_overlay_active_self(
        self, buffer: _MergeBuffer, *, created_at: float | None = None
    ) -> None:
        configuration = self.config_snapshot().value
        await self.output_projection.sync_active_self(
            ActiveSelfProjection(
                merge_id=buffer.merge_id,
                active_text=self._merge_text(
                    buffer.parts, preserve_text=bool(buffer.recognition_origins)
                ),
                spec_text=None,
                spec_translation=None,
                source_language=configuration.source_language,
                target_language=configuration.target_language,
                resume_pending=buffer.resume_pending,
                resume_confirmed=buffer.resume_confirmed,
                created_at=created_at,
            ),
            handoff=True,
        )

    async def _on_draft_updated(
        self, snapshot: SelfTurnPresentation, request: TranslationTurnRequest
    ) -> None:
        await self.output_projection.project_self_turn(
            snapshot, request.config_snapshot, source=request.source
        )


__all__ = [
    "SelfTranslationChannelOwner",
    "SelfTranslationChannelPort",
]
