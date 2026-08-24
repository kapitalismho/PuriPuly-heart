from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.local_asr_provider_runtime import LocalASRProviderRuntimePort
from puripuly_heart.core.messages import (
    UserErrorReport,
    UserMessageRef,
)
from puripuly_heart.core.orchestrator.channel_runtime import (
    ChannelRuntime,
)
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
    TranslationRuntimeConfigSnapshotPort,
)
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    LatencyInheritanceDiagnostic,
    LatencyStageDiagnostic,
    RuntimeDiagnostic,
    SttEventLoopFailureDiagnostic,
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import (
    TranslationOutputProjectionOwner,
    TranslationUiMessage,
)
from puripuly_heart.core.orchestrator.translation_request import (
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
from puripuly_heart.core.vad.gating import SpeechEnd, VadEvent
from puripuly_heart.domain.events import (
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionStateEvent,
    UIErrorPayload,
    UIEventType,
)
from puripuly_heart.domain.models import (
    ChannelId,
    Transcript,
    Translation,
)


@dataclass(slots=True)
class PeerTranslationChannelOwner:
    runtime: ChannelRuntime = field(repr=False)
    config_snapshot: TranslationRuntimeConfigSnapshotPort = field(repr=False)
    translation_turns: TranslationTurnLifecycleOwner = field(repr=False)
    local_asr_runtime: LocalASRProviderRuntimePort = field(repr=False)
    translation_requests: TranslationRequestPort = field(repr=False)
    output_projection: TranslationOutputProjectionOwner = field(repr=False)
    diagnostics: TranslationLatencyDiagnosticsOwner = field(repr=False)
    clock: Clock = field(default_factory=SystemClock)
    _peer_turn_parent_ids: dict[UUID, UUID] = field(default_factory=dict)
    _peer_parent_turn_ids: dict[UUID, set[UUID]] = field(default_factory=dict)
    _peer_completed_turn_ids: set[UUID] = field(default_factory=set)
    _peer_parent_speech_end_times: dict[UUID, float] = field(default_factory=dict)
    _peer_translation_parent_ids: set[UUID] = field(default_factory=set)
    _accepting_events: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        if self.runtime.channel != "peer":
            raise ValueError("Peer translation owner requires the Peer channel runtime")

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
        await self.translation_turns.cancel_pending(channel="peer")
        await self.runtime.reset_runtime_state()
        self._clear_peer_logical_turn_state()
        self.diagnostics.clear_latency_state(channel="peer")

    def translation_runtime_config_snapshot(self) -> TranslationRuntimeConfigSnapshot:
        return self.config_snapshot()

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

    def _record_latency_stage(
        self,
        *,
        channel: ChannelId,
        utterance_id: UUID,
        stage: str,
        timestamp: float | None = None,
        overwrite: bool = True,
        publish_now: bool = True,
    ) -> None:
        self.diagnostics.record_latency_stage(
            LatencyStageDiagnostic(
                channel=channel,
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
        channel: ChannelId,
        output_utterance_id: UUID,
        source_utterance_ids: list[UUID],
    ) -> None:
        self.diagnostics.inherit_latency(
            LatencyInheritanceDiagnostic(
                channel=channel,
                output_utterance_id=output_utterance_id,
                source_utterance_ids=tuple(source_utterance_ids),
            )
        )

    def _clear_latency_timeline(self, *, channel: ChannelId, utterance_id: UUID) -> None:
        self.diagnostics.clear_latency_timeline(channel, utterance_id)

    def _clear_latency_state(self, *, channel: ChannelId | None = None) -> None:
        self.diagnostics.clear_latency_state(channel)

    def _clear_runtime_latency_bookkeeping(self, *, channel: ChannelId, utterance_id: UUID) -> None:
        if channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer channel")
        self.runtime.utterance_start_times.pop(utterance_id, None)
        self.runtime.speech_ended_ids.discard(utterance_id)

    def _finalize_latency_timeline(self, *, channel: ChannelId, utterance_id: UUID) -> None:
        self._clear_runtime_latency_bookkeeping(channel=channel, utterance_id=utterance_id)
        self._clear_latency_timeline(channel=channel, utterance_id=utterance_id)

    def _clear_peer_logical_turn_state(self) -> None:
        self._peer_turn_parent_ids.clear()
        self._peer_parent_turn_ids.clear()
        self._peer_completed_turn_ids.clear()
        self._peer_parent_speech_end_times.clear()
        self._peer_translation_parent_ids.clear()

    def _peer_parent_speech_end_time(self, parent_utterance_id: UUID) -> float | None:
        parent_end_time = self.runtime.utterance_start_times.get(parent_utterance_id)
        if parent_end_time is not None:
            return parent_end_time
        return self._peer_parent_speech_end_times.get(parent_utterance_id)

    def _peer_parent_speech_ended(self, parent_utterance_id: UUID) -> bool:
        return (
            parent_utterance_id in self.runtime.speech_ended_ids
            or parent_utterance_id in self._peer_parent_speech_end_times
        )

    def _register_peer_logical_turn(
        self,
        *,
        parent_utterance_id: UUID,
        peer_turn_id: UUID,
    ) -> None:
        self._peer_turn_parent_ids[peer_turn_id] = parent_utterance_id
        self._peer_parent_turn_ids.setdefault(parent_utterance_id, set()).add(peer_turn_id)
        self._inherit_peer_parent_vad_bookkeeping(
            parent_utterance_id=parent_utterance_id,
            peer_turn_id=peer_turn_id,
        )

    def _inherit_peer_parent_vad_bookkeeping(
        self,
        *,
        parent_utterance_id: UUID,
        peer_turn_id: UUID,
    ) -> None:
        runtime = self.runtime
        parent_end_time = self._peer_parent_speech_end_time(parent_utterance_id)
        if parent_end_time is not None:
            runtime.utterance_start_times[peer_turn_id] = parent_end_time
            self._record_latency_stage(
                channel="peer",
                utterance_id=peer_turn_id,
                stage="speech_end",
                timestamp=parent_end_time,
                overwrite=False,
            )
        if self._peer_parent_speech_ended(parent_utterance_id):
            runtime.speech_ended_ids.add(peer_turn_id)
        self._inherit_latency_for_output(
            channel="peer",
            output_utterance_id=peer_turn_id,
            source_utterance_ids=[parent_utterance_id],
        )

    def _clear_peer_parent_vad_bookkeeping(
        self,
        parent_utterance_id: UUID,
        *,
        preserve_parent_speech_end_time: bool = False,
    ) -> None:
        peer_turn_ids = self._peer_parent_turn_ids.pop(parent_utterance_id, set())
        for peer_turn_id in peer_turn_ids:
            self._peer_turn_parent_ids.pop(peer_turn_id, None)
            self._peer_completed_turn_ids.discard(peer_turn_id)
        self.runtime.utterance_start_times.pop(parent_utterance_id, None)
        self.runtime.speech_ended_ids.discard(parent_utterance_id)
        if not preserve_parent_speech_end_time:
            self._peer_parent_speech_end_times.pop(parent_utterance_id, None)
        self._clear_latency_timeline(channel="peer", utterance_id=parent_utterance_id)

    def _maybe_clear_completed_peer_parent(
        self,
        parent_utterance_id: UUID,
        *,
        preserve_parent_speech_end_time: bool = False,
    ) -> None:
        peer_turn_ids = self._peer_parent_turn_ids.get(parent_utterance_id)
        if not peer_turn_ids:
            self._clear_peer_parent_vad_bookkeeping(
                parent_utterance_id,
                preserve_parent_speech_end_time=preserve_parent_speech_end_time,
            )
            return
        if not self._peer_parent_speech_ended(parent_utterance_id):
            return
        if peer_turn_ids.issubset(self._peer_completed_turn_ids):
            self._clear_peer_parent_vad_bookkeeping(
                parent_utterance_id,
                preserve_parent_speech_end_time=preserve_parent_speech_end_time,
            )

    def _complete_peer_logical_turn(
        self,
        peer_turn_id: UUID,
        *,
        preserve_parent_speech_end_time: bool = False,
    ) -> None:
        parent_utterance_id = self._peer_turn_parent_ids.get(peer_turn_id)
        if parent_utterance_id is None:
            return
        self._peer_completed_turn_ids.add(peer_turn_id)
        self._maybe_clear_completed_peer_parent(
            parent_utterance_id,
            preserve_parent_speech_end_time=preserve_parent_speech_end_time,
        )

    def _emit_exception_summary(
        self,
        message: str,
        *args: object,
        level: int = logging.ERROR,
    ) -> None:
        self.diagnostics.emit(
            RuntimeDiagnostic(
                message=message,
                args=args,
                level=level,
                safe_exceptions=True,
            )
        )

    def _emit_stt_event_loop_failure(
        self,
        exc: Exception,
        *,
        provider: object | None = None,
        channel: ChannelId = "peer",
    ) -> None:
        self.diagnostics.record_stt_event_loop_failure(
            SttEventLoopFailureDiagnostic(
                exception=exc,
                provider=provider,
                default_channel=channel,
            )
        )

    @staticmethod
    def _stt_error_event_payload(event: STTErrorEvent) -> UIErrorPayload | None:
        if isinstance(event.message, UserMessageRef) and event.diagnostics is not None:
            return UserErrorReport(message=event.message, diagnostics=event.diagnostics)
        return event.message

    async def reset_provider_channel(self, channel: ChannelId) -> None:
        if channel != "peer":
            raise ValueError("Peer translation owner cannot reset a non-Peer channel")
        await self.translation_turns.cancel_pending(channel="peer")
        await self.runtime.reset_runtime_state()
        self._clear_peer_logical_turn_state()
        self._clear_latency_state(channel="peer")

    def _remember_context_entry(
        self,
        text: str,
        timestamp: float,
        *,
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
        runtime: ChannelRuntime | None = None,
        source_language: str | None = None,
    ) -> None:
        runtime = runtime or self.runtime
        if runtime.channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer runtime")
        self.translation_requests.remember_context(
            text,
            timestamp,
            channel=runtime.channel,
            config_snapshot=config_snapshot,
            source_language=source_language,
        )

    async def handle_peer_vad_event(self, event: VadEvent) -> None:
        self._require_ingress()
        if isinstance(event, SpeechEnd) and not self.translation_turns.is_parent_closed(
            event.utterance_id
        ):
            speech_end_at = self.clock.now()
            self.runtime.utterance_start_times[event.utterance_id] = speech_end_at
            self.runtime.speech_ended_ids.add(event.utterance_id)
            self._peer_parent_speech_end_times[event.utterance_id] = speech_end_at
            self._record_latency_stage(
                channel="peer",
                utterance_id=event.utterance_id,
                stage="speech_end",
                timestamp=speech_end_at,
            )
            for peer_turn_id in tuple(self._peer_parent_turn_ids.get(event.utterance_id, set())):
                if peer_turn_id in self._peer_completed_turn_ids:
                    continue
                self._inherit_peer_parent_vad_bookkeeping(
                    parent_utterance_id=event.utterance_id,
                    peer_turn_id=peer_turn_id,
                )
            if event.utterance_id in self._peer_parent_turn_ids:
                self._maybe_clear_completed_peer_parent(event.utterance_id)
        await self.local_asr_runtime.handle_vad_event("peer", event)
        if isinstance(event, SpeechEnd):
            await self.local_asr_runtime.commit_handoff("peer")

    async def clear_language_runtime_state(self, *, channel: ChannelId) -> None:
        if channel != "peer":
            raise ValueError("Peer translation owner cannot clear a non-Peer channel")
        await self.translation_turns.cancel_pending(channel="peer")
        await self.runtime.clear_live_translation_state()
        self._clear_peer_logical_turn_state()
        self._clear_latency_state(channel="peer")

    async def handle_stt_event_loop_exception(
        self,
        exc: Exception,
        *,
        channel: ChannelId = "peer",
    ) -> None:
        if channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer exception")
        self._emit_stt_event_loop_failure(exc, channel=channel)

    async def handle_stt_event(self, event: object) -> None:
        self._require_ingress()
        utterance_id = getattr(event, "utterance_id", None)
        self.diagnostics.record_stt_ingress(
            "stt_handler_start",
            channel="peer",
            event_type=type(event).__name__,
            utterance_id=None if utterance_id is None else str(utterance_id),
        )
        if isinstance(event, STTSessionStateEvent):
            if event.channel != "peer":
                raise ValueError("Peer translation owner received a non-Peer session event")
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
            return

        if isinstance(event, STTErrorEvent):
            if event.channel != "peer":
                raise ValueError("Peer translation owner received a non-Peer error event")
            await self.output_projection.publish_ui(
                TranslationUiMessage(
                    event_type=UIEventType.ERROR,
                    payload=self._stt_error_event_payload(event),
                    source="Peer",
                    channel="peer",
                    runtime_log_handled=event.runtime_log_handled,
                )
            )
            return

        if isinstance(event, STTPartialEvent):
            if event.channel != "peer":
                raise ValueError("Peer translation owner received a non-Peer partial event")
            return

        if isinstance(event, STTFinalEvent):
            if event.channel != "peer":
                raise ValueError("Peer translation owner received a non-Peer final event")
            await self._ensure_translation(
                event.transcript,
                turn_kind="peer",
                wait_for_parent=(
                    not self.translation_requests.provider_available
                    or not self._translation_enabled_for_runtime(self.runtime)
                ),
            )

    async def handle_retired_stt_event(self, event: object) -> None:
        if isinstance(event, STTFinalEvent) and event.channel == "peer":
            await self.handle_stt_event(event)

    def _require_ingress(self) -> None:
        if not self._accepting_events:
            raise RuntimeError("Peer translation ingress is closed")

    async def _handle_transcript(
        self, transcript: Transcript, *, is_final: bool, source: str | None
    ) -> None:
        if transcript.channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer transcript")
        bundle = self.runtime.get_or_create_bundle(transcript.utterance_id)
        bundle.with_transcript(transcript)
        self._remember_source(transcript.utterance_id, source, channel="peer")
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
        deny_peer_chatbox_attempt = self.output_projection.chatbox_is_denied("peer")
        peer_terminal_work_will_follow = self._peer_terminal_work_will_follow(self.runtime)
        if self._overlay_translation_will_follow(self.runtime):
            await self._ensure_translation(transcript, turn_kind="peer")
        elif self.output_projection.has_overlay_destination:
            configuration = self.translation_runtime_config_snapshot().value
            finalized = await self.output_projection.project_peer_source_only(
                transcript=transcript,
                source_language=self._source_language_for(
                    self.runtime,
                    configuration,
                ),
                target_language=self._target_language_for(
                    self.runtime,
                    configuration,
                ),
                close_is_final=True,
                finalize_latency=not peer_terminal_work_will_follow,
            )
            if finalized:
                self._clear_runtime_latency_bookkeeping(
                    channel="peer",
                    utterance_id=transcript.utterance_id,
                )
            if deny_peer_chatbox_attempt:
                await self.output_projection.publish_peer_chatbox_denial(transcript.utterance_id)
                self._clear_runtime_latency_bookkeeping(
                    channel="peer",
                    utterance_id=transcript.utterance_id,
                )
        elif deny_peer_chatbox_attempt:
            await self.output_projection.publish_peer_chatbox_denial(transcript.utterance_id)
            self._clear_runtime_latency_bookkeeping(
                channel="peer",
                utterance_id=transcript.utterance_id,
            )
        elif not peer_terminal_work_will_follow:
            self._finalize_latency_timeline(
                channel="peer",
                utterance_id=transcript.utterance_id,
            )

    async def _handle_peer_final_transcript(
        self,
        transcript: Transcript,
        *,
        parent_utterance_id: UUID,
        source: str,
    ) -> None:
        _ = parent_utterance_id
        runtime = self.runtime
        bundle = runtime.get_or_create_bundle(transcript.utterance_id)
        bundle.with_transcript(transcript)
        self._remember_source(transcript.utterance_id, source, channel="peer")
        await self.output_projection.publish_ui(
            TranslationUiMessage(
                event_type=UIEventType.TRANSCRIPT_FINAL,
                utterance_id=transcript.utterance_id,
                payload=transcript,
                source=source,
            )
        )
        self._record_latency_stage(
            channel="peer",
            utterance_id=transcript.utterance_id,
            stage="stt_final",
        )

    async def on_child_created(self, child: TranslationTurnChild) -> None:
        if child.channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer child")
        self._register_peer_logical_turn(
            parent_utterance_id=child.parent_utterance_id,
            peer_turn_id=child.utterance_id,
        )
        await self._handle_peer_final_transcript(
            child.transcript,
            parent_utterance_id=child.parent_utterance_id,
            source=child.source,
        )

    async def process_child(
        self,
        child: TranslationTurnChild,
        cancellation_requested: Callable[[], bool],
    ) -> TranslationTurnProcessResult:
        if child.channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer child")
        runtime = self.runtime
        config_snapshot = child.config_snapshot
        if cancellation_requested():
            raise asyncio.CancelledError
        target_language = (
            self._target_language_for(runtime, config_snapshot.value)
            if child.target_language == "und"
            else child.target_language
        )
        if child.precomputed_translation is not None:
            self._remember_context_entry(
                child.transcript.text,
                self.clock.now(),
                config_snapshot=config_snapshot,
                runtime=runtime,
                source_language=child.precomputed_translation.source_language,
            )
            return TranslationTurnProcessResult(
                "translated",
                TranslationOutputSubmission(
                    parent_utterance_id=child.parent_utterance_id,
                    child_utterance_id=child.utterance_id,
                    sequence=child.sequence,
                    channel=child.channel,
                    source=child.source,
                    source_text=child.transcript.text,
                    source_language=child.detected_language,
                    target_language=target_language,
                    outcome="translated",
                    config_snapshot=config_snapshot,
                    translation=child.precomputed_translation,
                ),
            )
        result = await self.translation_requests.process(
            TranslationProcessRequest(
                parent_utterance_id=child.parent_utterance_id,
                utterance_id=child.utterance_id,
                sequence=child.sequence,
                text=child.transcript.text,
                channel=runtime.channel,
                source=child.source,
                target_language=target_language,
                context_policy=child.context_policy,
                detected_language=child.detected_language,
                config_snapshot=config_snapshot,
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
        if child.channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer child")
        self.runtime.translation_tasks[child.utterance_id] = task

    async def on_child_terminal(
        self,
        child: TranslationTurnChild,
        outcome: TranslationTurnOutcome,
    ) -> None:
        if child.channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer child")
        runtime = self.runtime
        runtime.translation_tasks.pop(child.utterance_id, None)
        if outcome == "cancelled":
            configuration = child.config_snapshot.value
            await self.output_projection.project_peer_source_only(
                transcript=child.transcript,
                source_language=self._source_language_for(runtime, configuration),
                target_language=self._target_language_for(runtime, configuration),
                close_is_final=False,
                finalize_latency=True,
            )
            await self.output_projection.publish_peer_chatbox_denial(child.utterance_id)
            self._clear_runtime_latency_bookkeeping(
                channel="peer",
                utterance_id=child.utterance_id,
            )
        self._complete_peer_logical_turn(
            child.utterance_id,
            preserve_parent_speech_end_time=True,
        )

    async def on_parent_closed(self, parent_utterance_id: UUID) -> None:
        if parent_utterance_id in self._peer_translation_parent_ids:
            self._peer_translation_parent_ids.discard(parent_utterance_id)
            self._clear_peer_parent_vad_bookkeeping(parent_utterance_id)

    async def on_parent_rejected(self, parent_utterance_id: UUID) -> None:
        if parent_utterance_id in self._peer_translation_parent_ids:
            try:
                await self.output_projection.publish_peer_chatbox_denial(parent_utterance_id)
                self._clear_runtime_latency_bookkeeping(
                    channel="peer",
                    utterance_id=parent_utterance_id,
                )
            finally:
                if not self.translation_turns.is_parent_active(parent_utterance_id):
                    self._peer_translation_parent_ids.discard(parent_utterance_id)

    def _overlay_translation_will_follow(self, runtime: ChannelRuntime) -> bool:
        return (
            self.output_projection.has_overlay_destination
            and self.translation_requests.provider_available
            and self._translation_enabled_for_runtime(runtime)
        )

    def _peer_terminal_work_will_follow(self, runtime: ChannelRuntime) -> bool:
        if runtime.channel != "peer":
            return False
        return (
            self.translation_requests.provider_available
            and self._translation_enabled_for_runtime(runtime)
        ) or self.output_projection.chatbox_is_denied(runtime.channel)

    def _remember_source(
        self,
        utterance_id: UUID,
        source: str | None,
        *,
        channel: ChannelId = "peer",
    ) -> None:
        if channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer source")
        self.runtime.remember_source(utterance_id, source)

    def _get_source(self, utterance_id: UUID, *, channel: ChannelId = "peer") -> str | None:
        if channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer source")
        return self.runtime.get_source(utterance_id)

    def _source_language_for(
        self,
        runtime: ChannelRuntime,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> str:
        return self.translation_requests.source_language_for(runtime.channel, configuration)

    def _target_language_for(
        self,
        runtime: ChannelRuntime,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> str:
        return self.translation_requests.target_language_for(runtime.channel, configuration)

    def _translation_enabled_for_runtime(
        self,
        runtime: ChannelRuntime,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> bool:
        return self.translation_requests.translation_enabled_for(
            runtime.channel,
            configuration,
        )

    async def _ensure_translation(
        self,
        transcript: Transcript,
        *,
        turn_kind: TranslationTurnKind | None = None,
        precomputed_translation: Translation | None = None,
        wait_for_parent: bool = False,
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    ) -> None:
        if transcript.channel != "peer":
            raise ValueError("Peer translation owner received a non-Peer transcript")
        runtime = self.runtime
        config_snapshot = config_snapshot or self.translation_runtime_config_snapshot()
        resolved_kind = turn_kind or "peer"
        if resolved_kind != "peer":
            raise ValueError("Peer translation owner received a non-Peer turn")
        self._peer_translation_parent_ids.add(transcript.utterance_id)
        source = self._get_source(transcript.utterance_id, channel="peer")
        if source is None:
            source = "Peer"
        await self.translation_turns.submit(
            TranslationTurnRequest(
                transcript=transcript,
                source=source,
                turn_kind=resolved_kind,
                target_languages=(self._target_language_for(runtime, config_snapshot.value),),
                precomputed_translation=precomputed_translation,
                config_snapshot=config_snapshot,
            ),
            wait_for_parent=wait_for_parent,
        )

    async def submit_translation_output(self, submission: TranslationOutputSubmission) -> None:
        if submission.channel != "peer":
            raise ValueError("Peer translation owner received non-Peer output")
        await self._publish_translation_result(submission)

    async def _publish_translation_result(
        self,
        submission: TranslationOutputSubmission,
    ) -> None:
        if submission.channel != "peer":
            raise ValueError("Peer translation owner received non-Peer output")
        runtime = self.runtime
        utterance_id = submission.child_utterance_id
        translation = submission.translation
        if translation is not None:
            runtime.get_or_create_bundle(utterance_id).with_translation(translation)
        receipt = await self.output_projection.project_translation_result(submission)
        if receipt.clear_runtime_latency_bookkeeping:
            self._clear_runtime_latency_bookkeeping(
                channel=runtime.channel,
                utterance_id=utterance_id,
            )
        if receipt.complete_peer_logical_turn:
            self._complete_peer_logical_turn(utterance_id)


__all__ = ["PeerTranslationChannelOwner"]
