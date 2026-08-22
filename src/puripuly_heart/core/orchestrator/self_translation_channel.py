from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID, uuid4

from puripuly_heart.core.clock import Clock
from puripuly_heart.core.local_asr_provider_runtime import LocalASRProviderRuntimePort
from puripuly_heart.core.messages import UserErrorReport, UserMessageRef
from puripuly_heart.core.orchestrator.channel_runtime import (
    ChannelRuntime,
    _MergeBuffer,
    _SpeculativeAttempt,
    _SpeculativeAttemptStatus,
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
    TranslationFailureDiagnostic,
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import (
    ActiveSelfProjection,
    TranslationOutputProjectionOwner,
    TranslationUiMessage,
)
from puripuly_heart.core.orchestrator.translation_request import (
    DirectTranslationRequest,
    StaleProviderCompletion,
    TranslationProcessRequest,
    TranslationRequestPort,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationOutputSubmission,
    TranslationTurnChild,
    TranslationTurnKind,
    TranslationTurnLifecycleOwner,
    TranslationTurnOutcome,
    TranslationTurnProcessResult,
    TranslationTurnRequest,
)
from puripuly_heart.core.runtime.output import SELF_SPEECH_TYPING_REASON
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadEvent
from puripuly_heart.domain.events import (
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionState,
    STTSessionStateEvent,
    UIErrorPayload,
    UIEventType,
)
from puripuly_heart.domain.models import Transcript, Translation

_PROMO_INTERVAL_SEC = 300.0
_RELAXED_OVERLAP_MIN_CHARS = 3
_BOUNDARY_PUNCT = {".", ",", ";", ":", "!", "?"}


class SelfTranslationChannelPort(Protocol):
    async def handle_vad_event(self, event: VadEvent) -> None: ...

    async def handle_stt_event(self, event: object) -> None: ...

    async def handle_retired_stt_event(self, event: object) -> None: ...

    async def handle_stt_event_loop_exception(self, exc: Exception) -> None: ...

    async def submit_text(self, text: str, *, source: str = "You") -> UUID: ...

    def mark_promo_eligible(self) -> None: ...

    async def reset_provider_channel(self, channel: str = "self") -> None: ...

    async def clear_language_runtime_state(self) -> None: ...

    async def open_ingress(self) -> None: ...

    async def close_ingress(self) -> None: ...

    async def close(self) -> None: ...


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

    def __post_init__(self) -> None:
        if self.runtime.channel != "self":
            raise ValueError("Self translation owner requires the Self channel runtime")

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
        await self.runtime.reset_runtime_state()

    async def reset_provider_channel(self, channel: str = "self") -> None:
        if channel != "self":
            raise ValueError("Self translation owner cannot reset a non-Self channel")
        await self.translation_turns.cancel_pending(channel="self")
        await self.output_projection.reset_overlay_preview()
        await self.runtime.reset_runtime_state()
        self.diagnostics.clear_latency_state(channel="self")

    async def clear_language_runtime_state(self) -> None:
        await self.translation_turns.cancel_pending(channel="self")
        await self.runtime.clear_live_translation_state()
        self.diagnostics.clear_latency_state(channel="self")
        await self.output_projection.reset_overlay_preview()

    def mark_promo_eligible(self) -> None:
        self._promo_eligible = True

    async def handle_vad_event(self, event: VadEvent) -> None:
        self._require_ingress()
        resume_overlay_resync_buffer: _MergeBuffer | None = None
        low_latency_mode = self.config_snapshot().value.low_latency_mode
        if isinstance(event, SpeechStart) and low_latency_mode:
            self._mark_resume_pending(event)
        if isinstance(event, SpeechChunk) and low_latency_mode:
            resume_overlay_resync_buffer = self._maybe_confirm_resume(event)
        if isinstance(event, SpeechEnd):
            speech_end_at = self.clock.now()
            self.output_projection.set_self_chatbox_typing_reason(
                SELF_SPEECH_TYPING_REASON,
                True,
            )
            self.runtime.utterance_start_times[event.utterance_id] = speech_end_at
            self.runtime.speech_ended_ids.add(event.utterance_id)
            self._record_latency_stage(
                utterance_id=event.utterance_id,
                stage="speech_end",
                timestamp=speech_end_at,
                publish_now=not low_latency_mode,
            )
            if low_latency_mode:
                self._maybe_update_buffer_end_time(event.utterance_id)
                self._maybe_start_finalize_wait(event.utterance_id)
                await self._maybe_clear_resume_on_end(event)
                buffer = self.merge_buffer
                if buffer is not None:
                    await self._evaluate_speculative_next_action(
                        buffer,
                        reason="speech_end",
                    )
        await self.local_asr_runtime.handle_vad_event("self", event)
        if isinstance(event, SpeechEnd):
            await self.local_asr_runtime.commit_handoff("self")
        if (
            resume_overlay_resync_buffer is not None
            and self.merge_buffer is resume_overlay_resync_buffer
        ):
            await self._sync_overlay_active_self(resume_overlay_resync_buffer)

    async def submit_text(self, text: str, *, source: str = "You") -> UUID:
        self._require_ingress()
        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")
        utterance_id = uuid4()
        self.runtime.remember_source(utterance_id, source)
        transcript = Transcript(
            utterance_id=utterance_id,
            text=text,
            is_final=True,
            created_at=self.clock.now(),
        )
        await self._handle_transcript(transcript, is_final=True, source=source)
        await self._ensure_translation(
            transcript,
            turn_kind="manual",
            wait_for_parent=(
                not self.translation_requests.provider_available
                or not self.config_snapshot().value.translation_enabled
            ),
        )
        return utterance_id

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
            self._emit_detailed(
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

    async def handle_retired_stt_event(self, event: object) -> None:
        if isinstance(event, STTFinalEvent) and event.channel == "self":
            if self.config_snapshot().value.low_latency_mode:
                return
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
        if child.utterance_id != child.parent_utterance_id:
            await self._handle_transcript(child.transcript, is_final=True, source=child.source)

    async def process_child(
        self,
        child: TranslationTurnChild,
        cancellation_requested: Callable[[], bool],
    ) -> TranslationTurnProcessResult:
        if child.channel != "self":
            raise ValueError("Self translation owner received a non-Self child")
        if cancellation_requested():
            raise asyncio.CancelledError
        target_language = (
            self.translation_requests.target_language_for("self", child.config_snapshot.value)
            if child.target_language == "und"
            else child.target_language
        )
        if child.precomputed_translation is not None:
            self.translation_requests.remember_context(
                child.transcript.text,
                self.clock.now(),
                channel="self",
                config_snapshot=child.config_snapshot,
                source_language=child.precomputed_translation.source_language,
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
                ),
            )
        result = await self.translation_requests.process(
            TranslationProcessRequest(
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
            ),
            cancellation_requested=cancellation_requested,
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
        self.runtime.translation_tasks[child.utterance_id] = task

    async def on_child_terminal(
        self,
        child: TranslationTurnChild,
        outcome: TranslationTurnOutcome,
    ) -> None:
        if child.channel != "self":
            raise ValueError("Self translation owner received a non-Self child")
        self.runtime.translation_tasks.pop(child.utterance_id, None)
        if outcome != "cancelled":
            return
        finalized = await self.output_projection.close_overlay_utterance(
            utterance_id=child.utterance_id,
            channel="self",
            is_final=False,
            finalize_latency=False,
        )
        if finalized:
            self._clear_runtime_latency_bookkeeping(child.utterance_id)

    async def submit_translation_output(self, submission: TranslationOutputSubmission) -> None:
        if submission.channel != "self":
            raise ValueError("Self translation owner received non-Self output")
        translation = submission.translation
        if translation is not None:
            self.runtime.get_or_create_bundle(submission.child_utterance_id).with_translation(
                translation
            )
        receipt = await self.output_projection.project_translation_result(submission)
        if receipt.clear_runtime_latency_bookkeeping:
            self._clear_runtime_latency_bookkeeping(submission.child_utterance_id)

    async def translate_and_enqueue(
        self,
        utterance_id: UUID,
        text: str,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> None:
        config_snapshot = self.config_snapshot()
        source = self.runtime.get_source(utterance_id) or "Mic"
        result = await self.translation_requests.process(
            TranslationProcessRequest(
                parent_utterance_id=utterance_id,
                utterance_id=utterance_id,
                sequence=0,
                text=text,
                channel="self",
                source=source,
                target_language=self.translation_requests.target_language_for(
                    "self",
                    config_snapshot.value,
                ),
                context_policy=self.translation_turns.policy.context_policy,
                config_snapshot=config_snapshot,
            ),
            cancellation_requested=cancellation_requested,
        )
        if result.output is not None:
            await self.submit_translation_output(result.output)

    async def _handle_transcript(
        self,
        transcript: Transcript,
        *,
        is_final: bool,
        source: str | None,
    ) -> None:
        if transcript.channel != "self":
            raise ValueError("Self translation owner received a non-Self transcript")
        self.runtime.get_or_create_bundle(transcript.utterance_id).with_transcript(transcript)
        self.runtime.remember_source(transcript.utterance_id, source)
        await self.output_projection.publish_ui(
            TranslationUiMessage(
                event_type=(
                    UIEventType.TRANSCRIPT_FINAL if is_final else UIEventType.TRANSCRIPT_PARTIAL
                ),
                utterance_id=transcript.utterance_id,
                payload=transcript,
                source=source,
            )
        )
        if not is_final:
            return
        configuration = self.config_snapshot().value
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
        if finalized:
            self._clear_runtime_latency_bookkeeping(transcript.utterance_id)

    async def _ensure_translation(
        self,
        transcript: Transcript,
        *,
        turn_kind: TranslationTurnKind = "self",
        precomputed_translation: Translation | None = None,
        wait_for_parent: bool = False,
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    ) -> None:
        config_snapshot = config_snapshot or self.config_snapshot()
        source = self.runtime.get_source(transcript.utterance_id) or "Mic"
        await self.translation_turns.submit(
            TranslationTurnRequest(
                transcript=transcript,
                source=source,
                turn_kind=turn_kind,
                target_languages=(
                    self.translation_requests.target_language_for(
                        "self",
                        config_snapshot.value,
                    ),
                ),
                precomputed_translation=precomputed_translation,
                config_snapshot=config_snapshot,
            ),
            wait_for_parent=wait_for_parent,
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

    def _emit_detailed(
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
                detailed=True,
            )
        )

    def _emit_metric(self, message: str, *args: object) -> None:
        self.diagnostics.emit_metric(message, *args)

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
        detailed: bool = False,
    ) -> UserErrorReport:
        return self.diagnostics.record_translation_failure(
            TranslationFailureDiagnostic(
                stage=stage,
                channel="self",
                exception=exc,
                detailed=detailed,
            )
        )

    @staticmethod
    def _stt_error_event_payload(event: STTErrorEvent) -> UIErrorPayload | None:
        if isinstance(event.message, UserMessageRef) and event.diagnostics is not None:
            return UserErrorReport(message=event.message, diagnostics=event.diagnostics)
        return event.message

    def _merge_text(self, parts: list[str]) -> str:
        merged = ""
        for part in parts:
            part_clean = part.strip()
            if not part_clean:
                continue
            if not merged:
                merged = part_clean
                continue
            merged = self._merge_with_overlap(merged, part_clean)
        return merged.strip()

    def _merge_with_overlap(self, existing: str, addition: str) -> str:
        if not existing:
            return addition
        if not addition:
            return existing
        if existing.endswith(addition):
            return existing

        max_overlap = min(len(existing), len(addition))
        overlap_len = 0
        for i in range(1, max_overlap + 1):
            if existing[-i:] == addition[:i]:
                overlap_len = i
        if overlap_len:
            return existing + addition[overlap_len:]

        relaxed_merge = self._relaxed_overlap_merge(existing, addition)
        if relaxed_merge is not None:
            return relaxed_merge

        if self._needs_space(existing, addition):
            return f"{existing} {addition}"
        return f"{existing}{addition}"

    def _relaxed_overlap_merge(self, existing: str, addition: str) -> str | None:
        if not existing or not addition:
            return None

        left_trimmed, left_trimmed_len = self._strip_trailing_boundary(existing)
        right_trimmed, right_trimmed_len = self._strip_leading_boundary(addition)
        if left_trimmed_len == 0 and right_trimmed_len == 0:
            return None
        if not left_trimmed or not right_trimmed:
            return None

        max_overlap = min(len(left_trimmed), len(right_trimmed))
        overlap_len = 0
        for i in range(1, max_overlap + 1):
            if left_trimmed[-i:] == right_trimmed[:i]:
                overlap_len = i

        if overlap_len < _RELAXED_OVERLAP_MIN_CHARS:
            return None

        cut = right_trimmed_len + overlap_len
        if cut <= 0 or cut > len(addition):
            return None

        base = existing[:-left_trimmed_len] if left_trimmed_len else existing
        if cut >= len(addition):
            return base
        return f"{base}{addition[cut:]}"

    def _strip_trailing_boundary(self, text: str) -> tuple[str, int]:
        idx = len(text)
        while idx > 0 and self._is_boundary_char(text[idx - 1]):
            idx -= 1
        return text[:idx], len(text) - idx

    def _strip_leading_boundary(self, text: str) -> tuple[str, int]:
        idx = 0
        while idx < len(text) and self._is_boundary_char(text[idx]):
            idx += 1
        return text[idx:], idx

    def _is_boundary_char(self, ch: str) -> bool:
        return ch.isspace() or ch in _BOUNDARY_PUNCT

    def _needs_space(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        left_ch = left[-1]
        right_ch = right[0]
        if self._is_ascii_alnum(left_ch) and self._is_ascii_alnum(right_ch):
            return True
        if (" " in left or " " in right) and left_ch.isalnum() and right_ch.isalnum():
            return True
        return False

    def _is_ascii_alnum(self, ch: str) -> bool:
        return ord(ch) < 128 and ch.isalnum()

    def _upsert_merge_part(
        self,
        buffer: _MergeBuffer,
        utterance_id: UUID,
        text: str,
    ) -> None:
        if not text:
            return
        for idx in range(len(buffer.utterance_ids) - 1, -1, -1):
            if buffer.utterance_ids[idx] == utterance_id:
                existing = buffer.parts[idx]
                if existing == text:
                    return
                if text in existing:
                    return
                if existing in text:
                    merged = text
                else:
                    merged = self._merge_with_overlap(existing, text)
                if merged != existing:
                    buffer.parts[idx] = merged
                    self._emit_metric(
                        "[Metric] final_update id=%s index=%s text_len=%s",
                        str(buffer.merge_id)[:8],
                        idx,
                        len(merged),
                    )
                return
        buffer.parts.append(text)
        buffer.utterance_ids.append(utterance_id)

    def _clear_resume_state(self, buffer: _MergeBuffer) -> None:
        buffer.resume_pending = False
        buffer.resume_confirmed = False
        buffer.resume_utterance_id = None
        buffer.resume_chunk_count = 0
        buffer.resume_started_at = None
        self._cancel_resume_end_timeout(buffer)

    def _clear_spec_latency_state(self, buffer: _MergeBuffer) -> None:
        attempt = buffer.speculative_attempt
        if attempt is not None:
            attempt.latency_stage_times.clear()

    def _record_spec_latency_stage(
        self,
        buffer: _MergeBuffer,
        *,
        stage: str,
        timestamp: float | None = None,
    ) -> None:
        attempt = buffer.speculative_attempt
        if attempt is None:
            return
        attempt.latency_stage_times[stage] = self.clock.now() if timestamp is None else timestamp

    def _promote_spec_latency_to_output(self, buffer: _MergeBuffer) -> None:
        attempt = buffer.speculative_attempt
        if attempt is None or not attempt.latency_stage_times:
            return
        for stage in ("llm_request_start", "llm_first_chunk", "llm_done"):
            timestamp = attempt.latency_stage_times.get(stage)
            if timestamp is None:
                continue
            self._record_latency_stage(
                utterance_id=buffer.merge_id,
                stage=stage,
                timestamp=timestamp,
                publish_now=False,
            )
        self._clear_spec_latency_state(buffer)
        self._finalize_latency_timeline(buffer.merge_id)

    def _clear_spec_state(self, buffer: _MergeBuffer, *, reason: str) -> bool:
        attempt = buffer.speculative_attempt
        if attempt is None:
            return False
        attempt.status = _SpeculativeAttemptStatus.CANCELLED
        if (
            attempt.task is not None
            and not attempt.task.done()
            and attempt.task is not asyncio.current_task()
        ):
            attempt.task.cancel()
            self._emit_metric(
                "[Metric] spec_cancel id=%s reason=%s",
                str(buffer.merge_id)[:8],
                reason,
            )
        elif attempt.result is not None:
            self._emit_metric(
                "[Metric] spec_cancel id=%s reason=%s",
                str(buffer.merge_id)[:8],
                reason,
            )
        attempt.latency_stage_times.clear()
        buffer.speculative_attempt = None
        return True

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
        if task is not None and task is not asyncio.current_task():
            if not task.done():
                task.cancel()
        buffer.resume_end_timeout_task = None
        buffer.resume_end_utterance_id = None

    def _start_resume_end_timeout(
        self,
        buffer: _MergeBuffer,
        utterance_id: UUID,
    ) -> None:
        self._cancel_resume_end_timeout(buffer)
        buffer.resume_end_utterance_id = utterance_id
        timeout_s = self.config_snapshot().value.low_latency_awaiting_vad_timeout_s
        buffer.resume_end_timeout_task = asyncio.create_task(
            self._resume_end_timeout(buffer.merge_id, utterance_id, timeout_s)
        )

    async def _resume_end_timeout(
        self,
        merge_id: UUID,
        utterance_id: UUID,
        timeout_s: float,
    ) -> None:
        try:
            await asyncio.sleep(timeout_s)
        except asyncio.CancelledError:
            return
        buffer = self.merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        if buffer.resume_end_utterance_id != utterance_id:
            return
        if not buffer.resume_confirmed:
            return
        self._emit_metric(
            "[Metric] resume_end_timeout id=%s vad_id=%s timeout_s=%s",
            str(merge_id)[:8],
            str(utterance_id)[:8],
            timeout_s,
        )
        self._clear_resume_state(buffer)
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

    def _mark_resume_pending(self, event: SpeechStart) -> None:
        buffer = self.merge_buffer
        if buffer is None:
            return
        if buffer.resume_pending and buffer.resume_utterance_id == event.utterance_id:
            return
        self._cancel_resume_end_timeout(buffer)
        buffer.resume_pending = True
        buffer.resume_confirmed = False
        buffer.resume_utterance_id = event.utterance_id
        buffer.resume_chunk_count = 0
        buffer.resume_started_at = self.clock.now()
        self._emit_metric(
            "[Metric] resume_pending id=%s vad_id=%s",
            str(buffer.merge_id)[:8],
            str(event.utterance_id)[:8],
        )

    def _maybe_confirm_resume(self, event: SpeechChunk) -> _MergeBuffer | None:
        buffer = self.merge_buffer
        if buffer is None or not buffer.resume_pending:
            return None
        if buffer.resume_utterance_id != event.utterance_id:
            return None
        if buffer.resume_confirmed:
            return None
        buffer.resume_chunk_count += 1
        if buffer.resume_chunk_count < 3:
            return None
        buffer.resume_confirmed = True
        confirm_ms = 0
        if buffer.resume_started_at is not None:
            confirm_ms = int((self.clock.now() - buffer.resume_started_at) * 1000)
        self._emit_metric(
            "[Metric] resume_confirmed id=%s confirm_ms=%s chunk_count=%s",
            str(buffer.merge_id)[:8],
            confirm_ms,
            buffer.resume_chunk_count,
        )
        cleared_spec_state = self._clear_spec_state(buffer, reason="resume_confirmed")
        if not cleared_spec_state:
            return None
        return buffer

    async def _maybe_clear_resume_on_end(self, event: SpeechEnd) -> None:
        buffer = self.merge_buffer
        if buffer is None:
            return
        if buffer.resume_utterance_id != event.utterance_id:
            return
        if buffer.resume_confirmed:
            self._start_resume_end_timeout(buffer, event.utterance_id)
            return
        if not buffer.resume_pending:
            return
        false_ms = 0
        if buffer.resume_started_at is not None:
            false_ms = int((self.clock.now() - buffer.resume_started_at) * 1000)
        self._emit_metric(
            "[Metric] resume_false_start id=%s false_ms=%s chunk_count=%s",
            str(buffer.merge_id)[:8],
            false_ms,
            buffer.resume_chunk_count,
        )
        self._clear_resume_state(buffer)
        await self._evaluate_speculative_next_action(
            buffer,
            reason="resume_false_start",
        )

    async def _handle_low_latency_final(self, transcript: Transcript) -> None:
        text = transcript.text.strip()
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
        buffer = self.merge_buffer
        if buffer is None:
            buffer = _MergeBuffer(merge_id=uuid4(), start_time=now, last_final_at=now)
            self.merge_buffer = buffer
        if buffer.resume_pending or buffer.resume_confirmed:
            self._clear_resume_state(buffer)
        self._upsert_merge_part(buffer, transcript.utterance_id, text)
        buffer.last_final_at = now
        await self._sync_overlay_active_self(buffer, created_at=transcript.created_at)

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
        attempt = buffer.speculative_attempt
        blocker = self._continuation_blocker(buffer)
        if blocker is not None:
            self._emit_continuation_blocker(buffer, blocker=blocker, reason=reason)
            return
        self._cancel_finalize_wait(buffer)
        buffer.awaiting_vad_end = False
        buffer.awaiting_vad_utterance_id = None
        for utterance_id in buffer.utterance_ids:
            self.runtime.utterance_start_times.pop(utterance_id, None)
            self.runtime.speech_ended_ids.discard(utterance_id)
            self.runtime.remember_low_latency_committed_utterance(utterance_id)
        if self.merge_buffer is buffer:
            self.merge_buffer = None

        final_text = self._merge_text(buffer.parts)
        if not final_text:
            await self.output_projection.reset_overlay_preview()
            return

        current_config_snapshot = self.config_snapshot()
        reuse_mode = None
        if (
            attempt is not None
            and attempt.status is _SpeculativeAttemptStatus.READY
            and isinstance(attempt.result, Translation)
            and attempt.provider_generation == self.translation_requests.provider_generation
            and self._translation_config_matches(
                attempt.config_snapshot.value,
                current_config_snapshot.value,
            )
        ):
            reuse_mode = self.output_projection.soft_reuse_mode(
                attempt.source_text,
                final_text,
            )

        if self.output_projection.should_blank_stale_active_secondary(
            final_text=final_text,
            reuse_mode=reuse_mode,
        ):
            configuration = self.config_snapshot().value
            source_language, target_language = (
                self.output_projection.self_overlay_languages_for_utterance(
                    utterance_id=buffer.merge_id,
                    source_language=configuration.source_language,
                    target_language=configuration.target_language,
                )
            )
            await self.output_projection.blank_active_self(
                utterance_id=buffer.merge_id,
                text=final_text,
                source_language=source_language,
                target_language=target_language,
                created_at=self.clock.now(),
            )

        if (
            attempt is not None
            and attempt.task is not None
            and not attempt.task.done()
            and attempt.task is not asyncio.current_task()
        ):
            attempt.status = _SpeculativeAttemptStatus.CANCELLED
            attempt.task.cancel()

        if buffer.last_end_time is not None:
            self.runtime.utterance_start_times[buffer.merge_id] = buffer.last_end_time
        elif buffer.start_time is not None:
            self.runtime.utterance_start_times[buffer.merge_id] = buffer.start_time
        self._inherit_latency_for_output(
            output_utterance_id=buffer.merge_id,
            source_utterance_ids=tuple(buffer.utterance_ids),
        )
        for utterance_id in buffer.utterance_ids:
            self._clear_latency_timeline(utterance_id)

        transcript = Transcript(
            utterance_id=buffer.merge_id,
            text=final_text,
            is_final=True,
            created_at=self.clock.now(),
        )
        await self._handle_transcript(transcript, is_final=True, source="Mic")
        config_snapshot = (
            attempt.config_snapshot
            if reuse_mode is not None and attempt is not None and attempt.result is not None
            else current_config_snapshot
        )

        if (
            not self.translation_requests.provider_available
            or not config_snapshot.value.translation_enabled
        ):
            await self._ensure_translation(
                transcript,
                turn_kind="self",
                wait_for_parent=True,
                config_snapshot=config_snapshot,
            )
            return

        reuse_spec = reuse_mode is not None
        commit_delay_ms = 0
        if buffer.start_time is not None:
            commit_delay_ms = int((self.clock.now() - buffer.start_time) * 1000)
        self._emit_metric(
            "[Metric] merge_commit id=%s used_spec=%s parts=%s text_len=%s commit_delay_ms=%s reason=%s",
            str(buffer.merge_id)[:8],
            reuse_spec,
            len(buffer.parts),
            len(final_text),
            commit_delay_ms,
            reason,
        )
        if reuse_spec:
            translation = attempt.result if attempt is not None else None
            if isinstance(translation, Translation):
                attempt.terminal_action_started = True
                self._promote_spec_latency_to_output(buffer)
                self._emit_metric(
                    "[Metric] spec_reuse id=%s translation_len=%s after_final=%s",
                    str(buffer.merge_id)[:8],
                    len(translation.text),
                    True,
                )
                await self._ensure_translation(
                    transcript,
                    turn_kind="self",
                    precomputed_translation=translation,
                    wait_for_parent=True,
                    config_snapshot=config_snapshot,
                )
                return

        if attempt is not None and attempt.result is not None and reuse_mode is None:
            self._clear_spec_latency_state(buffer)
            self._emit_metric(
                "[Metric] spec_cancel id=%s reason=final_mismatch",
                str(buffer.merge_id)[:8],
            )

        await self._ensure_translation(
            transcript,
            turn_kind="self",
            wait_for_parent=True,
            config_snapshot=config_snapshot,
        )

    def _continuation_blocker(self, buffer: _MergeBuffer) -> str | None:
        if buffer.resume_pending or buffer.resume_confirmed:
            return "resume"
        if buffer.awaiting_vad_end:
            return "await_vad_end"
        if buffer.finalize_wait_task is not None:
            return "post_end_grace"
        return None

    def _emit_continuation_blocker(
        self,
        buffer: _MergeBuffer,
        *,
        blocker: str,
        reason: str,
    ) -> None:
        attempt = buffer.speculative_attempt
        hold_ms = 0
        if blocker == "resume" and attempt is not None and attempt.completed_at is not None:
            hold_ms = int((self.clock.now() - attempt.completed_at) * 1000)
        elif buffer.finalize_wait_started_at is not None:
            hold_ms = int((self.clock.now() - buffer.finalize_wait_started_at) * 1000)
        self._emit_metric(
            "[Metric] commit_blocked id=%s blocker=%s reason=%s hold_ms=%s",
            str(buffer.merge_id)[:8],
            blocker,
            reason,
            hold_ms,
        )

    async def _maybe_restart_spec(self, buffer: _MergeBuffer) -> None:
        config_snapshot = self.config_snapshot()
        if (
            not self.translation_requests.provider_available
            or not config_snapshot.value.translation_enabled
        ):
            return

        merged_text = self._merge_text(buffer.parts)
        if not merged_text:
            return

        attempt = buffer.speculative_attempt
        if attempt is not None:
            normalized_text = self.output_projection.normalize_soft_reuse_text(merged_text)
            if attempt.source_text == merged_text:
                reuse_mode = "exact"
            elif attempt.normalized_text and attempt.normalized_text == normalized_text:
                reuse_mode = "soft_boundary"
            else:
                reuse_mode = None
            config_matches = self._translation_config_matches(
                attempt.config_snapshot.value,
                config_snapshot.value,
            )
            provider_matches = (
                attempt.provider_generation == self.translation_requests.provider_generation
            )
            if reuse_mode is not None and config_matches and provider_matches:
                self._emit_metric(
                    "[Metric] spec_preserve id=%s reason=%s status=%s attempt=%s",
                    str(buffer.merge_id)[:8],
                    reuse_mode,
                    attempt.status.value,
                    attempt.sequence,
                )
                return
            if not provider_matches:
                restart_reason = "provider_changed"
            elif not config_matches:
                restart_reason = "config_changed"
            else:
                restart_reason = "source_changed"
            self._clear_spec_state(buffer, reason=restart_reason)

        buffer.speculative_sequence += 1
        attempt = _SpeculativeAttempt(
            source_text=merged_text,
            normalized_text=self.output_projection.normalize_soft_reuse_text(merged_text),
            config_snapshot=config_snapshot,
            provider_generation=self.translation_requests.provider_generation,
            sequence=buffer.speculative_sequence,
            started_at=self.clock.now(),
        )
        buffer.speculative_attempt = attempt
        self._emit_metric(
            "[Metric] spec_start id=%s text_len=%s attempt=%s",
            str(buffer.merge_id)[:8],
            len(merged_text),
            attempt.sequence,
        )
        attempt.task = asyncio.create_task(
            self._run_spec_translation(
                buffer.merge_id,
                merged_text,
                attempt.sequence,
            )
        )

    @staticmethod
    def _translation_config_matches(
        left: TranslationRuntimeConfig,
        right: TranslationRuntimeConfig,
    ) -> bool:
        return (
            left.source_language,
            left.target_language,
            left.peer_source_language,
            left.peer_target_language,
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
            right.peer_source_language,
            right.peer_target_language,
            right.system_prompt,
            right.translation_enabled,
            right.peer_translation_enabled,
            right.integrated_context_enabled,
            right.context_time_window_s,
            right.context_max_entries,
            right.integrated_context_time_window_s,
            right.integrated_context_max_entries,
        )

    async def _run_spec_translation(
        self,
        merge_id: UUID,
        text: str,
        attempt: int,
        *,
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    ) -> None:
        if not self.translation_requests.provider_available:
            return
        buffer = self.merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        current_attempt = buffer.speculative_attempt
        if (
            current_attempt is None
            or current_attempt.source_text != text
            or current_attempt.sequence != attempt
        ):
            return
        config_snapshot = (
            config_snapshot or current_attempt.config_snapshot or self.config_snapshot()
        )
        current_attempt.config_snapshot = config_snapshot
        self._record_spec_latency_stage(buffer, stage="llm_request_start")
        try:
            translation = await self.translation_requests.translate(
                DirectTranslationRequest(
                    utterance_id=merge_id,
                    text=text,
                    record_latency=False,
                    config_snapshot=config_snapshot,
                )
            )
        except asyncio.CancelledError:
            return
        except StaleProviderCompletion:
            await self._handle_stale_spec_translation(merge_id, text, attempt)
            return
        except Exception as exc:
            self._log_translation_failure(
                stage="spec",
                exc=exc,
                detailed=True,
            )
            buffer = self.merge_buffer
            if buffer is None or buffer.merge_id != merge_id:
                return
            current_attempt = buffer.speculative_attempt
            if (
                current_attempt is None
                or current_attempt.source_text != text
                or current_attempt.sequence != attempt
            ):
                return
            self._clear_spec_latency_state(buffer)
            current_attempt.status = _SpeculativeAttemptStatus.FAILED
            current_attempt.completed_at = self.clock.now()
            await self._evaluate_speculative_next_action(
                buffer,
                reason="spec_failed",
            )
            return

        buffer = self.merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        current_attempt = buffer.speculative_attempt
        if (
            current_attempt is None
            or current_attempt.source_text != text
            or current_attempt.sequence != attempt
        ):
            return

        self._record_spec_latency_stage(buffer, stage="llm_done")
        current_attempt.result = translation
        current_attempt.status = _SpeculativeAttemptStatus.READY
        current_attempt.completed_at = self.clock.now()
        if current_attempt.started_at is None:
            latency_ms = 0
        else:
            latency_ms = int((self.clock.now() - current_attempt.started_at) * 1000)
        self._emit_metric(
            "[Metric] spec_done id=%s spec_latency_ms=%s translation_len=%s",
            str(merge_id)[:8],
            latency_ms,
            len(translation.text),
        )
        await self._sync_overlay_active_self(buffer, created_at=translation.created_at)
        await self._evaluate_speculative_next_action(
            buffer,
            reason="spec_done",
        )

    async def _handle_stale_spec_translation(
        self,
        merge_id: UUID,
        text: str,
        attempt: int,
    ) -> None:
        buffer = self.merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        current_attempt = buffer.speculative_attempt
        if (
            current_attempt is None
            or current_attempt.source_text != text
            or current_attempt.sequence != attempt
        ):
            return
        self._clear_spec_latency_state(buffer)
        current_attempt.result = None
        current_attempt.status = _SpeculativeAttemptStatus.STALE
        current_attempt.completed_at = self.clock.now()
        await self._evaluate_speculative_next_action(
            buffer,
            reason="spec_stale",
        )

    async def _evaluate_speculative_next_action(
        self,
        buffer: _MergeBuffer,
        *,
        reason: str,
    ) -> None:
        if self.merge_buffer is None or self.merge_buffer is not buffer:
            return
        attempt = buffer.speculative_attempt
        blocker = self._continuation_blocker(buffer)
        if blocker is not None:
            self._emit_continuation_blocker(buffer, blocker=blocker, reason=reason)
            return

        final_text = self._merge_text(buffer.parts)
        if not final_text:
            return

        config_snapshot = self.config_snapshot()
        translation_active = (
            self.translation_requests.provider_available
            and config_snapshot.value.translation_enabled
        )
        if attempt is None:
            await self._commit_merge(buffer, reason=reason)
            return

        if attempt.terminal_action_started:
            self._emit_metric(
                "[Metric] spec_terminal_duplicate_prevented id=%s reason=%s status=%s attempt=%s",
                str(buffer.merge_id)[:8],
                reason,
                attempt.status.value,
                attempt.sequence,
            )
            return

        if translation_active and attempt.status is _SpeculativeAttemptStatus.RUNNING:
            return

        reuse_mode = self.output_projection.soft_reuse_mode(
            attempt.source_text,
            final_text,
        )
        config_matches = self._translation_config_matches(
            attempt.config_snapshot.value,
            config_snapshot.value,
        )
        provider_matches = (
            attempt.provider_generation == self.translation_requests.provider_generation
        )
        reuse_ready = (
            translation_active
            and attempt.status is _SpeculativeAttemptStatus.READY
            and isinstance(attempt.result, Translation)
            and reuse_mode is not None
            and config_matches
            and provider_matches
        )
        action = "reuse" if reuse_ready else "fallback"
        if not translation_active:
            action = "source_only"
        attempt.terminal_action_started = True
        self._emit_metric(
            "[Metric] spec_terminal id=%s action=%s reason=%s status=%s attempt=%s",
            str(buffer.merge_id)[:8],
            action,
            reason,
            attempt.status.value,
            attempt.sequence,
        )
        await self._commit_merge(buffer, reason=reason)

    async def _sync_overlay_active_self(
        self,
        buffer: _MergeBuffer,
        *,
        created_at: float | None = None,
    ) -> None:
        active_text = self._merge_text(buffer.parts)
        if not active_text:
            return
        attempt = buffer.speculative_attempt
        configuration = self.config_snapshot().value
        await self.output_projection.sync_active_self(
            ActiveSelfProjection(
                merge_id=buffer.merge_id,
                active_text=active_text,
                spec_text=attempt.source_text if attempt is not None else None,
                spec_translation=(
                    attempt.result
                    if attempt is not None and isinstance(attempt.result, Translation)
                    else None
                ),
                source_language=self.translation_requests.source_language_for(
                    "self",
                    configuration,
                ),
                target_language=self.translation_requests.target_language_for(
                    "self",
                    configuration,
                ),
                resume_pending=buffer.resume_pending,
                resume_confirmed=buffer.resume_confirmed,
                created_at=created_at,
            )
        )


__all__ = [
    "SelfTranslationChannelOwner",
    "SelfTranslationChannelPort",
]
