from __future__ import annotations

import asyncio
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

from puripuly_heart.config.prompts import render_translation_prompt_template, warm_prompt_cache
from puripuly_heart.config.vad_defaults import DEFAULT_STABLE_VAD_HANGOVER_MS
from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.language import get_llm_language_name
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseDiagnostics,
    ManagedOpenRouterUserFacingError,
    format_managed_openrouter_diagnostics,
)
from puripuly_heart.core.orchestrator.channel_runtime import (
    ChannelRuntime,
    ContextEntry,
    _MergeBuffer,
)
from puripuly_heart.core.orchestrator.context import ContextMode, ContextResolver
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.core.overlay.sink import (
    OverlayEventAdapter,
    OverlaySink,
)
from puripuly_heart.core.runtime_logging import (
    LATENCY_CAUSE_E2E_THRESHOLD_MS,
    LATENCY_DOMINANT_STAGE_NORMAL,
    SessionLoggingMode,
    SessionRuntimeLoggingService,
    compute_latency_dominant_stage,
    format_basic_latency_summary,
    format_detailed_latency_breakdown,
    format_detailed_latency_trace,
    format_latency_cause_metric,
    format_translation_ready_for_output,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadEvent
from puripuly_heart.domain.events import (
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionState,
    STTSessionStateEvent,
    UIEvent,
    UIEventType,
)
from puripuly_heart.domain.models import (
    ChannelId,
    OSCMessage,
    Transcript,
    Translation,
    UtteranceBundle,
)


class STTProvider(Protocol):
    async def handle_vad_event(self, event: VadEvent) -> None: ...
    async def close(self) -> None: ...
    def events(self): ...


_PROMO_INTERVAL_SEC: float = 300.0  # 5 minutes
_RELAXED_OVERLAP_MIN_CHARS: int = 3
_BOUNDARY_PUNCT = {".", ",", ";", ":", "!", "?"}
_SOFT_REUSE_PUNCT = {".", ",", "…", "。", "，", "、"}
_SELF_RUNTIME_FIELDS = {
    "stt": "stt",
    "_stt_task": "stt_task",
    "_utterances": "utterances",
    "_translation_tasks": "translation_tasks",
    "_utterance_sources": "utterance_sources",
    "_utterance_start_times": "utterance_start_times",
    "_translation_history": "translation_history",
    "_speech_ended_ids": "speech_ended_ids",
    "_merge_buffer": "merge_buffer",
}
_LATENCY_TRACE_ORDER = (
    "speech_end",
    "stt_final",
    "llm_request_start",
    "llm_first_chunk",
    "llm_done",
    "self_chatbox_enqueue",
    "peer_overlay_first_emit",
    "peer_overlay_first_render",
)
_LATENCY_SUMMARY_OUTPUT_STAGES = {"self_chatbox_enqueue", "peer_overlay_first_emit"}
_SELF_SPEECH_TYPING_REASON = "self_speech_pending"


@dataclass(slots=True)
class _LatencyTimeline:
    channel: ChannelId
    stage_times: dict[str, float] = field(default_factory=dict)
    emitted_trace_points: set[str] = field(default_factory=set)
    basic_summary_emitted: bool = False


@dataclass(slots=True)
class ClientHub:
    stt: STTProvider | None
    llm: LLMProvider | None
    osc: ChatboxPaginator
    peer_stt: STTProvider | None = None
    overlay_sink: OverlaySink | None = None
    overlay_diagnostics: OverlayDiagnosticsRecorder | None = None
    clock: Clock = SystemClock()
    runtime_logging: SessionRuntimeLoggingService | None = None

    source_language: str = "ko"
    target_language: str = "en"
    peer_source_language: str = ""
    peer_target_language: str = ""
    system_prompt: str = ""
    chatbox_include_source: bool = True
    fallback_transcript_only: bool = False
    translation_enabled: bool = True
    peer_translation_enabled: bool = False
    integrated_context_enabled: bool = False
    # Self VAD hangover in seconds for user-facing E2E latency.
    hangover_s: float = DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
    peer_hangover_s: float = 0.6  # Peer VAD hangover in seconds for user-facing E2E latency.

    # Context memory settings
    context_time_window_s: float = 30.0  # Only include entries within this time window
    context_max_entries: int = 3  # Maximum number of context entries to include
    integrated_context_time_window_s: float = 40.0
    integrated_context_max_entries: int = 4
    low_latency_mode: bool = False
    low_latency_merge_gap_ms: int = 600
    low_latency_spec_retry_max: int = 1
    low_latency_finalize_wait_ms: int = 400
    low_latency_awaiting_vad_timeout_s: float = 3.0  # Timeout for awaiting_vad_end state

    ui_events: asyncio.Queue[UIEvent] = field(default_factory=asyncio.Queue)

    _utterances: dict[UUID, UtteranceBundle] = field(default_factory=dict)
    _translation_tasks: dict[UUID, asyncio.Task[None]] = field(default_factory=dict)
    _utterance_sources: dict[UUID, str] = field(default_factory=dict)
    _utterance_start_times: dict[UUID, float] = field(
        default_factory=dict
    )  # For E2E latency tracking
    _translation_history: list[ContextEntry] = field(default_factory=list)  # Context memory
    _speech_ended_ids: set[UUID] = field(default_factory=set)  # Track SpeechEnd arrivals
    _stt_task: asyncio.Task[None] | None = None
    _peer_stt_task: asyncio.Task[None] | None = None
    _osc_flush_task: asyncio.Task[None] | None = None
    _running: bool = False
    _last_promo_time: float | None = None
    _promo_eligible: bool = False
    _merge_buffer: _MergeBuffer | None = None
    self_runtime: ChannelRuntime = field(init=False)
    peer_runtime: ChannelRuntime = field(init=False)
    _peer_turn_parent_ids: dict[UUID, UUID] = field(default_factory=dict)
    _peer_parent_turn_ids: dict[UUID, set[UUID]] = field(default_factory=dict)
    _peer_completed_turn_ids: set[UUID] = field(default_factory=set)
    _peer_parent_speech_end_times: dict[UUID, float] = field(default_factory=dict)
    context_resolver: ContextResolver = field(init=False)
    active_chatbox_channel: ChannelId = field(init=False, default="self")
    overlay_event_adapter: OverlayEventAdapter = field(init=False)
    _last_logged_context_modes: dict[ChannelId, ContextMode | None] = field(
        init=False,
        default_factory=lambda: {"self": None, "peer": None},
    )
    overlay_stream_coalesce_ms: int = 300
    last_error_source: str | None = None
    _last_overlay_secondary_runtime_signature: tuple[object, ...] | None = field(
        init=False,
        default=None,
    )
    _last_overlay_secondary_diagnostics_signature: tuple[object, ...] | None = field(
        init=False,
        default=None,
    )
    _latency_timelines: dict[tuple[ChannelId, UUID], _LatencyTimeline] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self.overlay_event_adapter = OverlayEventAdapter(clock=self.clock)
        self.self_runtime = ChannelRuntime(
            channel="self",
            stt=self.stt,
            stt_task=self._stt_task,
            utterances=self._utterances,
            translation_tasks=self._translation_tasks,
            utterance_sources=self._utterance_sources,
            utterance_start_times=self._utterance_start_times,
            translation_history=self._translation_history,
            speech_ended_ids=self._speech_ended_ids,
            merge_buffer=self._merge_buffer,
            alias_target=self,
        )
        self.peer_runtime = ChannelRuntime(channel="peer", stt=self.peer_stt)
        self.context_resolver = ContextResolver(
            clock=self.clock,
            local_time_window_s=self.context_time_window_s,
            local_max_entries=self.context_max_entries,
            integrated_time_window_s=self.integrated_context_time_window_s,
            integrated_max_entries=self.integrated_context_max_entries,
        )
        warm_prompt_cache()
        self._sync_self_runtime_aliases()

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name in {
            "clock",
            "context_time_window_s",
            "context_max_entries",
            "integrated_context_time_window_s",
            "integrated_context_max_entries",
        }:
            try:
                resolver = object.__getattribute__(self, "context_resolver")
            except AttributeError:
                resolver = None
            try:
                overlay_event_adapter = object.__getattribute__(self, "overlay_event_adapter")
            except AttributeError:
                overlay_event_adapter = None
            if resolver is not None:
                if name == "clock":
                    resolver.clock = value  # type: ignore[assignment]
                elif name == "context_time_window_s":
                    resolver.local_time_window_s = value  # type: ignore[assignment]
                elif name == "context_max_entries":
                    resolver.local_max_entries = value  # type: ignore[assignment]
                elif name == "integrated_context_time_window_s":
                    resolver.integrated_time_window_s = value  # type: ignore[assignment]
                elif name == "integrated_context_max_entries":
                    resolver.integrated_max_entries = value  # type: ignore[assignment]
            if name == "clock" and overlay_event_adapter is not None:
                overlay_event_adapter.clock = value  # type: ignore[assignment]
        runtime_field = _SELF_RUNTIME_FIELDS.get(name)
        if runtime_field is None:
            return
        try:
            runtime = object.__getattribute__(self, "self_runtime")
        except AttributeError:
            return
        object.__setattr__(runtime, runtime_field, value)

    def _sync_self_runtime_aliases(self) -> None:
        self._stt_task = self.self_runtime.stt_task
        self._utterances = self.self_runtime.utterances
        self._translation_tasks = self.self_runtime.translation_tasks
        self._utterance_sources = self.self_runtime.utterance_sources
        self._utterance_start_times = self.self_runtime.utterance_start_times
        self._translation_history = self.self_runtime.translation_history
        self._speech_ended_ids = self.self_runtime.speech_ended_ids
        self._merge_buffer = self.self_runtime.merge_buffer

    @staticmethod
    def _format_log_message(message: str, *args: object) -> str:
        return message % args if args else message

    def _emit_basic(
        self,
        message: str,
        *args: object,
        level: int = logging.INFO,
        fallback_level: int | None = None,
    ) -> None:
        formatted = self._format_log_message(message, *args)
        if self.runtime_logging is not None:
            self.runtime_logging.emit_basic(formatted, level=level)
            return
        logger.log(level if fallback_level is None else fallback_level, formatted)

    def _emit_detailed(
        self,
        message: str,
        *args: object,
        level: int = logging.INFO,
        fallback_level: int | None = None,
    ) -> bool:
        if self.runtime_logging is not None:
            return self.runtime_logging.emit_detailed_lazy(
                lambda: self._format_log_message(message, *args),
                level=level,
            )
        _ = fallback_level
        return False

    def _emit_metric(self, message: str, *args: object) -> None:
        self._emit_detailed(message, *args, fallback_level=logging.DEBUG)

    @staticmethod
    def _latency_key(channel: ChannelId, utterance_id: UUID) -> tuple[ChannelId, UUID]:
        return channel, utterance_id

    def _get_latency_timeline(
        self,
        *,
        channel: ChannelId,
        utterance_id: UUID,
        create: bool = False,
    ) -> _LatencyTimeline | None:
        key = self._latency_key(channel, utterance_id)
        timeline = self._latency_timelines.get(key)
        if timeline is None and create:
            timeline = _LatencyTimeline(channel=channel)
            self._latency_timelines[key] = timeline
        return timeline

    @staticmethod
    def _elapsed_latency_ms(start_at: float | None, end_at: float | None) -> int | None:
        if start_at is None or end_at is None:
            return None
        return max(0, int(round((end_at - start_at) * 1000)))

    def _latency_hangover_ms(self, channel: ChannelId) -> int:
        hangover_s = self.peer_hangover_s if channel == "peer" else self.hangover_s
        return max(0, int(round(hangover_s * 1000)))

    def _emit_latency_trace_if_ready(
        self,
        *,
        channel: ChannelId,
        utterance_id: UUID,
        stage: str,
    ) -> None:
        timeline = self._get_latency_timeline(channel=channel, utterance_id=utterance_id)
        if timeline is None or stage in timeline.emitted_trace_points:
            return
        speech_end_at = timeline.stage_times.get("speech_end")
        stage_at = timeline.stage_times.get(stage)
        elapsed_ms = self._elapsed_latency_ms(speech_end_at, stage_at)
        if elapsed_ms is None:
            return
        emitted = self._emit_detailed(
            format_detailed_latency_trace(
                channel=channel,
                utterance_id=str(utterance_id)[:8],
                stage=stage,
                elapsed_ms=elapsed_ms,
            )
        )
        if emitted:
            timeline.emitted_trace_points.add(stage)

    def _emit_latency_summary_if_ready(
        self,
        *,
        channel: ChannelId,
        utterance_id: UUID,
        final_output_stage: str,
    ) -> None:
        timeline = self._get_latency_timeline(channel=channel, utterance_id=utterance_id)
        if timeline is None or timeline.basic_summary_emitted:
            return
        speech_end_at = timeline.stage_times.get("speech_end")
        final_output_at = timeline.stage_times.get(final_output_stage)
        measured_speech_end_to_final_output_ms = self._elapsed_latency_ms(
            speech_end_at, final_output_at
        )
        if measured_speech_end_to_final_output_ms is None:
            return
        e2e_ms = measured_speech_end_to_final_output_ms + self._latency_hangover_ms(channel)

        stt_final_at = timeline.stage_times.get("stt_final")
        speech_end_to_stt_final_ms = self._elapsed_latency_ms(speech_end_at, stt_final_at)
        stt_reference_at = None
        if speech_end_at is not None and stt_final_at is not None:
            stt_reference_at = max(speech_end_at, stt_final_at)
        stt_final_to_final_output_ms = self._elapsed_latency_ms(stt_reference_at, final_output_at)

        self._emit_basic(
            format_basic_latency_summary(
                channel=channel,
                e2e_ms=e2e_ms,
            )
        )
        dominant_stage = compute_latency_dominant_stage(
            speech_end_to_stt_final_ms=speech_end_to_stt_final_ms,
            stt_final_to_final_output_ms=stt_final_to_final_output_ms,
        )
        self._emit_detailed(
            format_detailed_latency_breakdown(
                channel=channel,
                e2e_ms=e2e_ms,
                speech_end_to_stt_final_ms=speech_end_to_stt_final_ms,
                stt_final_to_final_output_ms=stt_final_to_final_output_ms,
                dominant_stage=dominant_stage,
            )
        )
        if (
            dominant_stage != LATENCY_DOMINANT_STAGE_NORMAL
            or e2e_ms >= LATENCY_CAUSE_E2E_THRESHOLD_MS
        ):
            self._emit_basic(
                format_latency_cause_metric(
                    channel=channel,
                    e2e_ms=e2e_ms,
                    dominant_stage=dominant_stage,
                    speech_end_to_stt_final_ms=speech_end_to_stt_final_ms,
                    stt_final_to_final_output_ms=stt_final_to_final_output_ms,
                )
            )
        timeline.basic_summary_emitted = True

    def _emit_latency_contract_if_ready(
        self,
        *,
        channel: ChannelId,
        utterance_id: UUID,
    ) -> None:
        for trace_stage in _LATENCY_TRACE_ORDER:
            self._emit_latency_trace_if_ready(
                channel=channel,
                utterance_id=utterance_id,
                stage=trace_stage,
            )
        for output_stage in _LATENCY_SUMMARY_OUTPUT_STAGES:
            self._emit_latency_summary_if_ready(
                channel=channel,
                utterance_id=utterance_id,
                final_output_stage=output_stage,
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
        timeline = self._get_latency_timeline(
            channel=channel, utterance_id=utterance_id, create=True
        )
        assert timeline is not None
        if not overwrite and stage in timeline.stage_times:
            return
        timeline.stage_times[stage] = self.clock.now() if timestamp is None else timestamp

        if not publish_now:
            return

        self._emit_latency_contract_if_ready(
            channel=channel,
            utterance_id=utterance_id,
        )

    def _inherit_latency_for_output(
        self,
        *,
        channel: ChannelId,
        output_utterance_id: UUID,
        source_utterance_ids: list[UUID],
    ) -> None:
        output_timeline = self._get_latency_timeline(
            channel=channel,
            utterance_id=output_utterance_id,
            create=True,
        )
        assert output_timeline is not None
        for source_utterance_id in source_utterance_ids:
            source_timeline = self._get_latency_timeline(
                channel=channel,
                utterance_id=source_utterance_id,
            )
            if source_timeline is None:
                continue
            for stage in ("speech_end", "stt_final"):
                source_time = source_timeline.stage_times.get(stage)
                if source_time is None:
                    continue
                existing_time = output_timeline.stage_times.get(stage)
                if existing_time is None:
                    output_timeline.stage_times[stage] = source_time
                else:
                    output_timeline.stage_times[stage] = max(existing_time, source_time)
        self._emit_latency_contract_if_ready(
            channel=channel,
            utterance_id=output_utterance_id,
        )

    def _clear_latency_timeline(self, *, channel: ChannelId, utterance_id: UUID) -> None:
        self._latency_timelines.pop(self._latency_key(channel, utterance_id), None)

    def _clear_latency_state(self, *, channel: ChannelId | None = None) -> None:
        if channel is None:
            self._latency_timelines.clear()
            return
        keys_to_remove = [key for key in self._latency_timelines if key[0] == channel]
        for key in keys_to_remove:
            self._latency_timelines.pop(key, None)

    def _clear_runtime_latency_bookkeeping(self, *, channel: ChannelId, utterance_id: UUID) -> None:
        runtime = self._runtime_for_channel(channel)
        runtime.utterance_start_times.pop(utterance_id, None)
        runtime.speech_ended_ids.discard(utterance_id)

    def _finalize_latency_timeline(self, *, channel: ChannelId, utterance_id: UUID) -> None:
        self._clear_runtime_latency_bookkeeping(channel=channel, utterance_id=utterance_id)
        self._clear_latency_timeline(channel=channel, utterance_id=utterance_id)

    def _clear_peer_logical_turn_state(self) -> None:
        self._peer_turn_parent_ids.clear()
        self._peer_parent_turn_ids.clear()
        self._peer_completed_turn_ids.clear()
        self._peer_parent_speech_end_times.clear()

    def _peer_parent_speech_end_time(self, parent_utterance_id: UUID) -> float | None:
        parent_end_time = self.peer_runtime.utterance_start_times.get(parent_utterance_id)
        if parent_end_time is not None:
            return parent_end_time
        return self._peer_parent_speech_end_times.get(parent_utterance_id)

    def _peer_parent_speech_ended(self, parent_utterance_id: UUID) -> bool:
        return (
            parent_utterance_id in self.peer_runtime.speech_ended_ids
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
        runtime = self.peer_runtime
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
        self.peer_runtime.utterance_start_times.pop(parent_utterance_id, None)
        self.peer_runtime.speech_ended_ids.discard(parent_utterance_id)
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

    def _peer_logical_turn_transcript(self, transcript: Transcript) -> tuple[UUID, Transcript]:
        parent_utterance_id = transcript.utterance_id
        peer_turn_id = uuid4()
        self._register_peer_logical_turn(
            parent_utterance_id=parent_utterance_id,
            peer_turn_id=peer_turn_id,
        )
        return parent_utterance_id, Transcript(
            utterance_id=peer_turn_id,
            text=transcript.text,
            is_final=True,
            created_at=transcript.created_at,
            channel="peer",
        )

    def _emit_exception_summary(
        self,
        message: str,
        *args: object,
        level: int = logging.ERROR,
    ) -> None:
        formatted = self._format_log_message(message, *args)
        if self.runtime_logging is not None:
            self.runtime_logging.emit_basic(formatted, level=level)
            detail = "".join(traceback.format_exception(*sys.exc_info())).rstrip()
            if detail:
                self.runtime_logging.emit_detailed(detail, level=level)
            return
        logger.exception(formatted)

    def _translation_skip_reason(self, runtime: ChannelRuntime) -> str:
        if self.llm is None:
            return "llm unavailable"
        if not self.translation_enabled:
            return "translation disabled"
        if runtime.channel == "peer" and not self.peer_translation_enabled:
            return "peer translation disabled"
        return "translation disabled"

    def _log_translation_skipped(
        self,
        *,
        stage: str,
        runtime: ChannelRuntime,
        publish_chatbox: bool,
    ) -> None:
        self._emit_detailed(
            "[Hub] Translation skipped (stage=%s, channel=%s, publish_chatbox=%s): %s",
            stage,
            runtime.channel,
            publish_chatbox,
            self._translation_skip_reason(runtime),
            fallback_level=logging.INFO,
        )

    def _log_translation_failure(
        self,
        *,
        stage: str,
        runtime: ChannelRuntime,
        exc: Exception,
        detailed: bool = False,
    ) -> None:
        emit = self._emit_detailed if detailed else self._emit_basic
        message = str(exc)
        diagnostics = self._managed_openrouter_diagnostics(exc)
        diagnostics_text = format_managed_openrouter_diagnostics(diagnostics)
        if diagnostics_text:
            message = f"{message} [{diagnostics_text}]"
        emit(
            "[Hub] Translation failed (stage=%s, channel=%s): %s",
            stage,
            runtime.channel,
            message,
            level=logging.ERROR,
            fallback_level=logging.ERROR,
        )

    def _managed_openrouter_diagnostics(
        self, exc: Exception
    ) -> ManagedOpenRouterReleaseDiagnostics | None:
        diagnostics = getattr(exc, "diagnostics", None)
        if isinstance(diagnostics, ManagedOpenRouterReleaseDiagnostics):
            return diagnostics
        return None

    async def start(self, *, auto_flush_osc: bool = False) -> None:
        if self._running:
            return
        self._running = True
        if self.stt is not None:
            self._stt_task = asyncio.create_task(self._run_stt_event_loop(self.stt))
        if self.peer_stt is not None:
            self._peer_stt_task = asyncio.create_task(self._run_stt_event_loop(self.peer_stt))
        if auto_flush_osc:
            self._osc_flush_task = asyncio.create_task(self._run_osc_flush_loop())

    async def stop(self) -> None:
        if not self._running:
            self._clear_osc_typing_reasons()
            return
        self._running = False
        self._clear_osc_typing_reasons()

        if self._osc_flush_task:
            self._osc_flush_task.cancel()
            await asyncio.gather(self._osc_flush_task, return_exceptions=True)
            self._osc_flush_task = None

        await self._stop_stt_event_loop()
        await self.reset_overlay_preview()
        await self._reset_stt_runtime_state()

        if self.stt is not None:
            await self.stt.close()
        if self.peer_stt is not None:
            await self.peer_stt.close()

        if self.llm is not None:
            await self.llm.close()

    async def replace_stt_provider(self, stt: STTProvider | None) -> None:
        old_stt = self.stt
        await self._stop_stt_task("_stt_task")
        await self.reset_overlay_preview()
        await self.self_runtime.reset_runtime_state()
        self._clear_latency_state(channel="self")
        self._sync_self_runtime_aliases()

        if old_stt is not None:
            await old_stt.close()

        self.stt = stt
        self.self_runtime.stt = stt
        if self._running and self.stt is not None:
            self._stt_task = asyncio.create_task(self._run_stt_event_loop(self.stt))

    async def replace_peer_stt_provider(self, stt: STTProvider | None) -> None:
        old_stt = self.peer_stt
        await self._stop_stt_task("_peer_stt_task")
        await self.peer_runtime.reset_runtime_state()
        self._clear_peer_logical_turn_state()
        self._clear_latency_state(channel="peer")

        if old_stt is not None:
            await old_stt.close()

        self.peer_stt = stt
        self.peer_runtime.stt = stt
        if self._running and self.peer_stt is not None:
            self._peer_stt_task = asyncio.create_task(self._run_stt_event_loop(self.peer_stt))

    def mark_promo_eligible(self) -> None:
        """Mark that user clicked STT button. Next STREAMING state will send promo."""
        self._promo_eligible = True

    def clear_context(self) -> None:
        """Clear the translation context history."""
        self.self_runtime.clear_context()
        self.peer_runtime.clear_context()
        self._emit_basic("[Hub] Context history cleared")

    def _get_valid_context(self) -> list[ContextEntry]:
        """Get context entries within time window and max entries limit."""
        return self.context_resolver.get_local_entries(
            runtime=self.self_runtime,
            source_language=self._source_language_for(self.self_runtime),
            target_language=self._target_language_for(self.self_runtime),
        )

    def _format_context_for_llm(self, context: list[ContextEntry]) -> str:
        """Format context entries as a string for LLM prompt."""
        return self.context_resolver.format_local(context)

    def _remember_context_entry(
        self,
        text: str,
        timestamp: float,
        *,
        runtime: ChannelRuntime | None = None,
    ) -> None:
        runtime = runtime or self.self_runtime
        runtime.remember_context(
            text,
            timestamp=timestamp,
            source_language=self._source_language_for(runtime),
            target_language=self._target_language_for(runtime),
            max_entries=max(self.context_max_entries, self.integrated_context_max_entries),
        )

    def _log_context_mode_change(
        self,
        *,
        runtime: ChannelRuntime,
        applied_mode: ContextMode,
    ) -> None:
        last_mode = self._last_logged_context_modes.get(runtime.channel)
        if last_mode == applied_mode:
            return
        self._last_logged_context_modes[runtime.channel] = applied_mode
        self._emit_basic("[Hub] Context mode: channel=%s mode=%s", runtime.channel, applied_mode)

    def _log_context_application(
        self,
        *,
        text: str,
        runtime: ChannelRuntime,
        context: str,
    ) -> None:
        context_lines = context.splitlines() if context else []
        applied_mode = self._last_logged_context_modes.get(runtime.channel)
        if runtime.channel == "peer" and applied_mode in (None, "local"):
            peer_entries = len(context_lines)
            self_entries = 0
        else:
            peer_entries = sum(
                1
                for line in context_lines
                if line.startswith("- [peer,") or line.startswith("- [others,")
            )
            self_entries = len(context_lines) - peer_entries
        self._emit_basic(
            "[Hub] Context apply: channel=%s mode=%s request_chars=%s "
            "entries=%s self_entries=%s peer_entries=%s context_chars=%s",
            runtime.channel,
            applied_mode,
            len(text),
            len(context_lines),
            self_entries,
            peer_entries,
            len(context),
        )

    async def handle_vad_event(self, event: VadEvent) -> None:
        resume_overlay_resync_buffer: _MergeBuffer | None = None

        if isinstance(event, SpeechStart):
            if self.low_latency_mode:
                self._mark_resume_pending(event)

        if isinstance(event, SpeechChunk):
            if self.low_latency_mode:
                resume_overlay_resync_buffer = self._maybe_confirm_resume(event)

        # Record start time for E2E latency tracking (from speech end)
        if isinstance(event, SpeechEnd):
            speech_end_at = self.clock.now()
            self._set_osc_typing_reason(_SELF_SPEECH_TYPING_REASON, True)
            self._utterance_start_times[event.utterance_id] = speech_end_at
            self._speech_ended_ids.add(event.utterance_id)
            self._record_latency_stage(
                channel="self",
                utterance_id=event.utterance_id,
                stage="speech_end",
                timestamp=speech_end_at,
                publish_now=not self.low_latency_mode,
            )
            if self.low_latency_mode:
                self._maybe_update_buffer_end_time(event.utterance_id)
                self._maybe_start_finalize_wait(event.utterance_id)
                await self._maybe_clear_resume_on_end(event)

        if self.stt is not None:
            await self.stt.handle_vad_event(event)

        if (
            resume_overlay_resync_buffer is not None
            and self._merge_buffer is resume_overlay_resync_buffer
        ):
            await self._sync_overlay_active_self(resume_overlay_resync_buffer)

    async def handle_peer_vad_event(self, event: VadEvent) -> None:
        if isinstance(event, SpeechEnd):
            speech_end_at = self.clock.now()
            self.peer_runtime.utterance_start_times[event.utterance_id] = speech_end_at
            self.peer_runtime.speech_ended_ids.add(event.utterance_id)
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
        if self.peer_stt is not None:
            await self.peer_stt.handle_vad_event(event)

    async def submit_text(self, text: str, *, source: str = "You") -> UUID:
        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")

        utterance_id = uuid4()
        self._remember_source(utterance_id, source)

        transcript = Transcript(
            utterance_id=utterance_id,
            text=text,
            is_final=True,
            created_at=self.clock.now(),
        )
        await self._handle_transcript(transcript, is_final=True, source=source)

        if self.llm is None or not self.translation_enabled:
            await self._enqueue_osc(utterance_id, transcript_text=text, translation_text=None)
        else:
            await self._ensure_translation(transcript)

        return utterance_id

    def _runtime_for_channel(self, channel: ChannelId) -> ChannelRuntime:
        return self.self_runtime if channel == "self" else self.peer_runtime

    async def clear_language_runtime_state(self, *, channel: ChannelId) -> None:
        runtime = self._runtime_for_channel(channel)
        await runtime.clear_live_translation_state()
        if channel == "peer":
            self._clear_peer_logical_turn_state()
        self._clear_latency_state(channel=channel)
        if channel == "self":
            await self.reset_overlay_preview()
            self._sync_self_runtime_aliases()

    def _runtime_for_utterance(
        self, utterance_id: UUID, *, default_channel: ChannelId = "self"
    ) -> ChannelRuntime:
        if utterance_id in self.self_runtime.utterances:
            return self.self_runtime
        if utterance_id in self.peer_runtime.utterances:
            return self.peer_runtime
        return self._runtime_for_channel(default_channel)

    def get_or_create_bundle(
        self, utterance_id: UUID, *, channel: ChannelId = "self"
    ) -> UtteranceBundle:
        return self._runtime_for_utterance(
            utterance_id, default_channel=channel
        ).get_or_create_bundle(utterance_id)

    async def _run_stt_event_loop(self, provider: STTProvider) -> None:
        try:
            async for ev in provider.events():
                await self._handle_stt_event(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit_exception_summary(
                "[Hub] STT event loop crashed: %s",
                exc,
                level=logging.ERROR,
            )
            raise

    async def _stop_stt_event_loop(self) -> None:
        await self._stop_stt_task("_stt_task")
        await self._stop_stt_task("_peer_stt_task")

    async def _stop_stt_task(self, attr_name: str) -> None:
        task = getattr(self, attr_name)
        if task is None:
            return
        setattr(self, attr_name, None)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _reset_stt_runtime_state(self) -> None:
        await self.self_runtime.reset_runtime_state()
        await self.peer_runtime.reset_runtime_state()
        self._clear_peer_logical_turn_state()
        self._clear_latency_state()
        self._sync_self_runtime_aliases()

    async def _handle_stt_event(self, event: object) -> None:
        if isinstance(event, STTSessionStateEvent):
            self._emit_basic(
                "[Hub] STT state: channel=%s state=%s",
                event.channel,
                event.state.name,
            )
            await self.ui_events.put(
                UIEvent(
                    type=UIEventType.SESSION_STATE_CHANGED,
                    payload=event.state,
                    channel=event.channel,
                )
            )
            if event.state == STTSessionState.STREAMING and event.channel == "self":
                self._send_stt_connected_notification()
            return

        if isinstance(event, STTErrorEvent):
            await self.ui_events.put(
                UIEvent(
                    type=UIEventType.ERROR,
                    payload=event.message,
                    source="Peer" if event.channel == "peer" else "Mic",
                    channel=event.channel,
                    runtime_log_handled=event.runtime_log_handled,
                )
            )
            return

        if isinstance(event, STTPartialEvent):
            if event.channel == "peer":
                return
            self._send_stt_connected_notification()
            if self.low_latency_mode:
                return
            self._emit_detailed(
                f"[Hub] STT Partial: '{event.transcript.text[:50]}...' id={str(event.transcript.utterance_id)[:8]}",
                fallback_level=logging.DEBUG,
            )
            await self._handle_transcript(event.transcript, is_final=False, source="Mic")
            return

        if isinstance(event, STTFinalEvent):
            runtime = self._runtime_for_channel(event.channel)
            source = "Peer" if runtime.channel == "peer" else "Mic"
            if runtime.channel == "peer":
                parent_utterance_id, peer_transcript = self._peer_logical_turn_transcript(
                    event.transcript
                )
                await self._handle_peer_final_transcript(
                    peer_transcript,
                    parent_utterance_id=parent_utterance_id,
                    source=source,
                )
                return
            if runtime.channel == "self":
                self._send_stt_connected_notification()
            if self.low_latency_mode and runtime.channel == "self":
                await self._handle_low_latency_final(event.transcript)
                return
            self._record_latency_stage(
                channel=runtime.channel,
                utterance_id=event.transcript.utterance_id,
                stage="stt_final",
            )
            await self._handle_transcript(event.transcript, is_final=True, source=source)
            if self.llm is None or not self._translation_enabled_for_runtime(runtime):
                self._log_translation_skipped(
                    stage="final",
                    runtime=runtime,
                    publish_chatbox=self._should_publish_to_chatbox(runtime),
                )
                if self._should_publish_to_chatbox(runtime):
                    await self._enqueue_osc(
                        event.transcript.utterance_id,
                        transcript_text=event.transcript.text,
                        translation_text=None,
                    )
                else:
                    self._finalize_latency_timeline(
                        channel=runtime.channel,
                        utterance_id=event.transcript.utterance_id,
                    )
            else:
                await self._ensure_translation(event.transcript)
            return

    def _send_stt_connected_notification(self) -> None:
        """Send promo message when STT connects (only if user clicked button)."""
        if not self._promo_eligible:
            return  # Skip if not triggered by user button click
        self._promo_eligible = False

        now = self.clock.now()
        if self._last_promo_time is not None:
            if now - self._last_promo_time < _PROMO_INTERVAL_SEC:
                return
        if self.osc.send_immediate("PuriPuly ON!"):
            self._last_promo_time = now

    async def _handle_transcript(
        self, transcript: Transcript, *, is_final: bool, source: str | None
    ) -> None:
        runtime = self._runtime_for_channel(transcript.channel)
        bundle = self.get_or_create_bundle(transcript.utterance_id, channel=transcript.channel)
        bundle.with_transcript(transcript)
        self._remember_source(transcript.utterance_id, source, channel=transcript.channel)
        await self.ui_events.put(
            UIEvent(
                type=UIEventType.TRANSCRIPT_FINAL if is_final else UIEventType.TRANSCRIPT_PARTIAL,
                utterance_id=transcript.utterance_id,
                payload=transcript,
                source=source,
            )
        )
        if is_final:
            if runtime.channel == "peer":
                peer_terminal_work_will_follow = self._peer_terminal_work_will_follow(runtime)
                if self._overlay_translation_will_follow(runtime):
                    await self._ensure_translation(transcript)
                elif self.overlay_sink is not None:
                    await self._finalize_peer_source_only(
                        transcript,
                        close_is_final=True,
                        finalize_latency=not peer_terminal_work_will_follow,
                    )
                elif not peer_terminal_work_will_follow:
                    self._finalize_latency_timeline(
                        channel=transcript.channel,
                        utterance_id=transcript.utterance_id,
                    )
                return
            await self._emit_final_transcript_to_overlay(transcript)
            if not self._overlay_translation_will_follow(runtime):
                await self._emit_overlay_utterance_closed(
                    utterance_id=transcript.utterance_id,
                    channel=transcript.channel,
                    is_final=True,
                )

    async def _handle_peer_final_transcript(
        self,
        transcript: Transcript,
        *,
        parent_utterance_id: UUID,
        source: str,
    ) -> None:
        _ = parent_utterance_id
        runtime = self.peer_runtime
        bundle = runtime.get_or_create_bundle(transcript.utterance_id)
        bundle.with_transcript(transcript)
        self._remember_source(transcript.utterance_id, source, channel="peer")
        await self.ui_events.put(
            UIEvent(
                type=UIEventType.TRANSCRIPT_FINAL,
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
        if self.llm is None or not self._translation_enabled_for_runtime(runtime):
            self._log_translation_skipped(
                stage="final",
                runtime=runtime,
                publish_chatbox=self._should_publish_to_chatbox(runtime),
            )
            await self._finalize_peer_source_only(
                transcript,
                close_is_final=True,
                finalize_latency=not self._should_publish_to_chatbox(runtime),
                preserve_parent_speech_end_time=True,
            )
            if self._should_publish_to_chatbox(runtime):
                await self._enqueue_osc(
                    transcript.utterance_id,
                    transcript_text=transcript.text,
                    translation_text=None,
                )
            return
        await self._ensure_translation(transcript)

    async def _emit_final_transcript_to_overlay(self, transcript: Transcript) -> None:
        if self.overlay_sink is None:
            return
        source_language, target_language = self._self_overlay_languages_for_utterance(
            transcript.utterance_id
        )
        await self._emit_overlay_event(
            self.overlay_event_adapter.transcript_final(
                transcript,
                source_language=source_language,
                target_language=target_language,
            )
        )

    async def _finalize_peer_source_only(
        self,
        transcript: Transcript,
        *,
        close_is_final: bool,
        finalize_latency: bool,
        preserve_parent_speech_end_time: bool = False,
    ) -> None:
        if self.overlay_sink is not None:
            self._record_overlay_emit(
                event_kind="peer_transcript_final",
                utterance_id=transcript.utterance_id,
                channel="peer",
                secondary_len=len(transcript.text.strip()),
            )
            self._record_latency_stage(
                channel="peer",
                utterance_id=transcript.utterance_id,
                stage="peer_overlay_first_emit",
                overwrite=False,
            )
            await self._emit_overlay_event(
                self.overlay_event_adapter.transcript_final(
                    transcript,
                    source_language=self._source_language_for(self.peer_runtime),
                    target_language=self._target_language_for(self.peer_runtime),
                )
            )
        await self._emit_overlay_utterance_closed(
            utterance_id=transcript.utterance_id,
            channel="peer",
            is_final=close_is_final,
            finalize_latency=finalize_latency,
        )
        self._complete_peer_logical_turn(
            transcript.utterance_id,
            preserve_parent_speech_end_time=preserve_parent_speech_end_time,
        )

    async def _emit_overlay_utterance_closed(
        self,
        *,
        utterance_id: UUID,
        channel: ChannelId,
        is_final: bool,
        finalize_latency: bool | None = None,
    ) -> None:
        if self.overlay_sink is None:
            if finalize_latency is True or (finalize_latency is None and channel == "peer"):
                self._finalize_latency_timeline(channel=channel, utterance_id=utterance_id)
            return
        await self._emit_overlay_event(
            self.overlay_event_adapter.utterance_closed(
                utterance_id=utterance_id,
                channel=channel,
                is_final=is_final,
            )
        )
        if finalize_latency is True or (finalize_latency is None and channel == "peer"):
            self._finalize_latency_timeline(channel=channel, utterance_id=utterance_id)

    def _overlay_translation_will_follow(self, runtime: ChannelRuntime) -> bool:
        return (
            self.overlay_sink is not None
            and self.llm is not None
            and self._translation_enabled_for_runtime(runtime)
        )

    def _peer_terminal_work_will_follow(self, runtime: ChannelRuntime) -> bool:
        if runtime.channel != "peer":
            return False
        return (self.llm is not None and self._translation_enabled_for_runtime(runtime)) or (
            self._should_publish_to_chatbox(runtime)
        )

    @staticmethod
    def _translation_overlay_metadata(translation: Translation) -> dict[str, object]:
        return {
            "update_id": translation.update_id,
            "origin_wall_clock_ms": translation.origin_wall_clock_ms,
            "session_scope": translation.session_scope,
            "source_text_hash": translation.source_text_hash,
            "source_text_len": translation.source_text_len,
            "logical_turn_key": translation.logical_turn_key,
        }

    @staticmethod
    def _language_or_fallback(language: str | None, fallback: str) -> str:
        if language is not None and language.strip():
            return language
        return fallback

    @staticmethod
    def _metadata_language(metadata: object | None, field_name: str) -> str | None:
        value = getattr(metadata, field_name, None)
        if not isinstance(value, str):
            return None
        return value

    def _active_self_display_languages_for_utterance(
        self,
        utterance_id: UUID,
    ) -> tuple[str | None, str | None]:
        metadata = self._current_active_self_metadata()
        if metadata is None:
            return None, None
        if getattr(metadata, "utterance_id", None) != utterance_id:
            return None, None
        if getattr(metadata, "occupant_key", None) != f"self:{utterance_id}":
            return None, None
        return (
            self._metadata_language(metadata, "primary_language"),
            self._metadata_language(metadata, "secondary_language"),
        )

    def _self_overlay_languages_for_utterance(self, utterance_id: UUID) -> tuple[str, str]:
        primary_language, secondary_language = self._active_self_display_languages_for_utterance(
            utterance_id
        )
        return (
            self._language_or_fallback(primary_language, self.source_language),
            self._language_or_fallback(secondary_language, self.target_language),
        )

    def _current_active_self_metadata(self) -> object | None:
        provider = getattr(self.overlay_sink, "active_self_overlay_metadata", None)
        if not callable(provider):
            return None
        return provider()

    @staticmethod
    def _active_self_translation_metadata(metadata: object | None) -> dict[str, object]:
        if metadata is None:
            return {
                "update_id": None,
                "origin_wall_clock_ms": None,
                "session_scope": None,
                "source_text_hash": None,
                "source_text_len": None,
                "logical_turn_key": None,
            }
        return {
            "update_id": getattr(metadata, "update_id", None),
            "origin_wall_clock_ms": getattr(metadata, "origin_wall_clock_ms", None),
            "session_scope": getattr(metadata, "session_scope", None),
            "source_text_hash": getattr(metadata, "source_text_hash", None),
            "source_text_len": getattr(metadata, "source_text_len", None),
            "logical_turn_key": getattr(metadata, "logical_turn_key", None),
        }

    def _cached_active_self_secondary_text(self) -> str:
        metadata = self._current_active_self_metadata()
        if metadata is None:
            return ""
        return str(getattr(metadata, "secondary_text", "") or "")

    def _overlay_secondary_translation_metadata(
        self,
        *,
        buffer: _MergeBuffer,
        source: str,
        secondary_text: str,
    ) -> dict[str, object]:
        if not secondary_text:
            return self._active_self_translation_metadata(None)
        if source == "spec" and isinstance(buffer.spec_translation, Translation):
            return self._translation_overlay_metadata(buffer.spec_translation)
        metadata = self._current_active_self_metadata()
        if (
            source == "sticky_cache"
            and metadata is not None
            and getattr(metadata, "utterance_id", None) == buffer.merge_id
        ):
            return self._active_self_translation_metadata(metadata)
        return self._active_self_translation_metadata(None)

    def _active_self_overlay_languages(
        self,
        *,
        buffer: _MergeBuffer,
        source: str,
        secondary_text: str,
        current_metadata: object | None,
    ) -> tuple[str, str]:
        source_language = self._source_language_for(self.self_runtime)
        target_language = self._target_language_for(self.self_runtime)
        if source == "spec" and isinstance(buffer.spec_translation, Translation):
            return (
                self._language_or_fallback(
                    buffer.spec_translation.source_language,
                    source_language,
                ),
                self._language_or_fallback(
                    buffer.spec_translation.target_language,
                    target_language,
                ),
            )
        metadata_matches_active_self = (
            current_metadata is not None
            and getattr(current_metadata, "utterance_id", None) == buffer.merge_id
            and getattr(current_metadata, "occupant_key", None)
            == self._active_self_occupant_key(buffer)
        )
        if secondary_text and source == "sticky_cache" and metadata_matches_active_self:
            return (
                self._language_or_fallback(
                    self._metadata_language(current_metadata, "primary_language"),
                    source_language,
                ),
                self._language_or_fallback(
                    self._metadata_language(current_metadata, "secondary_language"),
                    target_language,
                ),
            )
        if not secondary_text and metadata_matches_active_self:
            return (
                self._language_or_fallback(
                    self._metadata_language(current_metadata, "primary_language"),
                    source_language,
                ),
                target_language,
            )
        return source_language, target_language

    def _translation_ready_elapsed_ms(
        self,
        *,
        channel: ChannelId,
        utterance_id: UUID,
    ) -> int | None:
        timeline = self._get_latency_timeline(channel=channel, utterance_id=utterance_id)
        if timeline is None:
            return None
        ready_at = timeline.stage_times.get("llm_done")
        if ready_at is None:
            return None
        return self._elapsed_latency_ms(timeline.stage_times.get("speech_end"), ready_at)

    def _emit_translation_ready_for_output(
        self,
        *,
        translation: Translation,
        runtime: ChannelRuntime,
    ) -> bool:
        if self.runtime_logging is None:
            return False
        return self.runtime_logging.emit_detailed_lazy(
            lambda: format_translation_ready_for_output(
                channel=runtime.channel,
                utterance_id=str(translation.utterance_id),
                update_id=translation.update_id,
                origin_wall_clock_ms=translation.origin_wall_clock_ms,
                session_scope=translation.session_scope,
                source_text_hash=translation.source_text_hash,
                source_text_len=translation.source_text_len,
                logical_turn_key=translation.logical_turn_key,
                translation_len=len(translation.text),
                elapsed_ms=self._translation_ready_elapsed_ms(
                    channel=runtime.channel,
                    utterance_id=translation.utterance_id,
                ),
            )
        )

    async def _emit_translation_to_overlay(
        self,
        *,
        translation: Translation,
        applied_context_mode: ContextMode | None,
    ) -> None:
        if self.overlay_sink is None:
            return

        self._record_overlay_emit(
            event_kind="translation_final",
            utterance_id=translation.utterance_id,
            channel=translation.channel,
            secondary_len=len(translation.text.strip()),
        )
        await self._emit_overlay_event(
            self.overlay_event_adapter.translation_final(
                utterance_id=translation.utterance_id,
                channel=translation.channel,
                text=translation.text,
                source_language=self._language_or_fallback(
                    translation.source_language,
                    self.source_language,
                ),
                target_language=self._language_or_fallback(
                    translation.target_language,
                    self.target_language,
                ),
                applied_context_mode=applied_context_mode,
                created_at=translation.created_at,
                **self._translation_overlay_metadata(translation),
            )
        )

    async def _emit_peer_translation_to_overlay(
        self,
        *,
        translation: Translation,
        runtime: ChannelRuntime,
        applied_context_mode: ContextMode | None,
    ) -> None:
        if self.overlay_sink is None:
            return

        self._record_overlay_emit(
            event_kind="translation_final",
            utterance_id=translation.utterance_id,
            channel=translation.channel,
            secondary_len=len(translation.text.strip()),
        )
        self._record_latency_stage(
            channel=runtime.channel,
            utterance_id=translation.utterance_id,
            stage="peer_overlay_first_emit",
            overwrite=False,
        )
        await self._emit_overlay_event(
            self.overlay_event_adapter.translation_final(
                utterance_id=translation.utterance_id,
                channel=translation.channel,
                text=translation.text,
                source_text=translation.source_text,
                source_language=self._language_or_fallback(
                    translation.source_language,
                    self._source_language_for(runtime),
                ),
                target_language=self._language_or_fallback(
                    translation.target_language,
                    self._target_language_for(runtime),
                ),
                applied_context_mode=applied_context_mode,
                created_at=translation.created_at,
                **self._translation_overlay_metadata(translation),
            )
        )

    async def _emit_overlay_event(self, event: object) -> None:
        if self.overlay_sink is None:
            return
        detailed_mode = (
            self.runtime_logging is not None
            and self.runtime_logging.mode is SessionLoggingMode.DETAILED
        )
        start = time.perf_counter() if detailed_mode else 0.0
        try:
            await self.overlay_sink.emit(event)  # type: ignore[arg-type]
        except Exception as exc:
            self.last_error_source = "overlay_sink"
            self._emit_exception_summary(
                "[Hub] Overlay sink emit failed: %s",
                exc,
                level=logging.ERROR,
            )
            return
        if detailed_mode:
            elapsed_ms = max(0, int((time.perf_counter() - start) * 1000))
            event_type = type(event).__name__
            channel = getattr(event, "channel", None)
            utterance_id = getattr(event, "utterance_id", None)
            update_id = getattr(event, "update_id", None)
            self.runtime_logging.emit_detailed_lazy(
                lambda: (
                    "[Detailed][Hub] overlay_sink_emit_duration "
                    f"event_type={event_type} "
                    f"channel={channel} "
                    f"utterance_id={utterance_id} "
                    f"update_id={update_id} "
                    f"elapsed_ms={elapsed_ms}"
                )
            )

    async def _emit_self_active_overlay_event(self, event: object) -> None:
        await self._emit_overlay_event(event)

    def _active_self_secondary_decision(
        self,
        buffer: _MergeBuffer,
    ) -> tuple[str, str, str | None]:
        translation = buffer.spec_translation
        active_text = self._merge_text(buffer.parts)
        if not active_text:
            return "", "blank", None
        reuse_mode = None
        if isinstance(translation, Translation):
            reuse_mode = self._soft_reuse_mode(buffer.spec_text, active_text)
            if reuse_mode is not None:
                return translation.text.strip(), "spec", reuse_mode
        sticky_secondary = self._cached_active_self_secondary_text().strip()
        if sticky_secondary:
            return sticky_secondary, "sticky_cache", reuse_mode
        return "", "blank", reuse_mode

    def _active_self_occupant_key(self, buffer: _MergeBuffer) -> str:
        return f"self:{buffer.merge_id}"

    async def _sync_overlay_active_self(
        self, buffer: _MergeBuffer | None, *, created_at: float | None = None
    ) -> None:
        if self.overlay_sink is None or buffer is None:
            return

        active_text = self._merge_text(buffer.parts)
        if not active_text:
            return
        secondary_text, source, reuse_mode = self._active_self_secondary_decision(buffer)
        self._record_active_self_secondary_decision(
            buffer=buffer,
            active_text=active_text,
            secondary_text=secondary_text,
            source=source,
            reuse_mode=reuse_mode,
        )
        current_metadata = self._current_active_self_metadata()
        translation_metadata = self._overlay_secondary_translation_metadata(
            buffer=buffer,
            source=source,
            secondary_text=secondary_text,
        )
        current_translation_metadata = self._active_self_translation_metadata(current_metadata)
        occupant_key = self._active_self_occupant_key(buffer)
        source_language, target_language = self._active_self_overlay_languages(
            buffer=buffer,
            source=source,
            secondary_text=secondary_text,
            current_metadata=current_metadata,
        )
        primary_language = source_language.strip() or None
        secondary_language = (target_language.strip() or None) if secondary_text.strip() else None
        if (
            current_metadata is not None
            and buffer.merge_id == getattr(current_metadata, "utterance_id", None)
            and occupant_key == getattr(current_metadata, "occupant_key", None)
            and active_text == getattr(current_metadata, "text", None)
            and secondary_text == getattr(current_metadata, "secondary_text", "")
            and primary_language == getattr(current_metadata, "primary_language", None)
            and secondary_language == getattr(current_metadata, "secondary_language", None)
            and translation_metadata == current_translation_metadata
        ):
            return

        self._record_overlay_emit(
            event_kind="active_self",
            utterance_id=buffer.merge_id,
            channel="self",
            secondary_len=len(secondary_text),
        )
        await self._emit_self_active_overlay_event(
            self.overlay_event_adapter.self_active_update(
                text=active_text,
                utterance_id=buffer.merge_id,
                secondary_text=secondary_text,
                occupant_key=occupant_key,
                source_language=source_language,
                target_language=target_language,
                created_at=created_at,
                **translation_metadata,
            )
        )

    async def reset_overlay_preview(self) -> None:
        if self._current_active_self_metadata() is None:
            return
        if self.overlay_sink is None:
            return
        await self._emit_self_active_overlay_event(self.overlay_event_adapter.self_active_clear())

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

    def _soft_reuse_mode(self, spec_text: str | None, final_text: str) -> str | None:
        if spec_text is None:
            return None
        if spec_text == final_text:
            return "exact"

        normalized_spec = self._normalize_soft_reuse_text(spec_text)
        normalized_final = self._normalize_soft_reuse_text(final_text)
        if not normalized_spec or not normalized_final:
            return None
        if normalized_spec == normalized_final:
            return "soft_boundary"
        return None

    def _normalize_soft_reuse_text(self, text: str) -> str:
        start = 0
        end = len(text)
        while start < end and self._is_soft_reuse_boundary_char(text[start]):
            start += 1
        while end > start and self._is_soft_reuse_boundary_char(text[end - 1]):
            end -= 1
        return text[start:end]

    def _record_active_self_secondary_decision(
        self,
        *,
        buffer: _MergeBuffer,
        active_text: str,
        secondary_text: str,
        source: str,
        reuse_mode: str | None,
    ) -> None:
        signature = (
            buffer.merge_id,
            active_text,
            secondary_text,
            source,
            reuse_mode,
            buffer.resume_pending,
            buffer.resume_confirmed,
        )
        self._maybe_emit_active_self_secondary_runtime_log(
            buffer=buffer,
            active_text=active_text,
            secondary_text=secondary_text,
            source=source,
            reuse_mode=reuse_mode,
            signature=signature,
        )
        if self.overlay_diagnostics is None:
            return
        if signature == self._last_overlay_secondary_diagnostics_signature:
            return
        self._last_overlay_secondary_diagnostics_signature = signature
        spec_translation_len = 0
        if isinstance(buffer.spec_translation, Translation):
            spec_translation_len = len(buffer.spec_translation.text.strip())
        self.overlay_diagnostics.record_hub(
            "active_self_secondary",
            merge_id=str(buffer.merge_id),
            source=source,
            active_text_len=len(active_text),
            secondary_len=len(secondary_text),
            spec_text_len=len((buffer.spec_text or "").strip()),
            spec_translation_len=spec_translation_len,
            cached_secondary_len=len(self._cached_active_self_secondary_text().strip()),
            reuse_mode=reuse_mode,
            resume_pending=buffer.resume_pending,
            resume_confirmed=buffer.resume_confirmed,
        )

    def _maybe_emit_active_self_secondary_runtime_log(
        self,
        *,
        buffer: _MergeBuffer,
        active_text: str,
        secondary_text: str,
        source: str,
        reuse_mode: str | None,
        signature: tuple[object, ...],
    ) -> None:
        if signature == self._last_overlay_secondary_runtime_signature:
            return
        spec_translation_len = 0
        if isinstance(buffer.spec_translation, Translation):
            spec_translation_len = len(buffer.spec_translation.text.strip())
        emitted = self._emit_detailed(
            "[Hub] active_self_secondary merge_id=%s source=%s active_len=%s secondary_len=%s spec_text_len=%s spec_translation_len=%s cached_secondary_len=%s reuse_mode=%s resume_pending=%s resume_confirmed=%s",
            str(buffer.merge_id)[:8],
            source,
            len(active_text),
            len(secondary_text),
            len((buffer.spec_text or "").strip()),
            spec_translation_len,
            len(self._cached_active_self_secondary_text().strip()),
            reuse_mode,
            buffer.resume_pending,
            buffer.resume_confirmed,
            fallback_level=logging.INFO,
        )
        if emitted:
            self._last_overlay_secondary_runtime_signature = signature

    def _should_blank_stale_active_secondary_before_finalizing(
        self,
        *,
        final_text: str,
        reuse_mode: str | None,
    ) -> bool:
        # Presenter promotion preserves active secondary text for the same occupant.
        # Blank the active row first when speculative reuse is unsafe so stale
        # secondary text cannot be promoted into the finalized row.
        metadata = self._current_active_self_metadata()
        return (
            reuse_mode is None
            and self.overlay_sink is not None
            and metadata is not None
            and getattr(metadata, "text", None) == final_text
            and str(getattr(metadata, "secondary_text", "") or "").strip() != ""
        )

    def _record_overlay_emit(
        self,
        *,
        event_kind: str,
        utterance_id: UUID,
        channel: ChannelId,
        secondary_len: int,
    ) -> None:
        if self.overlay_diagnostics is None:
            return
        self.overlay_diagnostics.record_hub(
            "overlay_emit",
            event_kind=event_kind,
            utterance_id=str(utterance_id),
            channel=channel,
            secondary_len=secondary_len,
            sink_type=type(self.overlay_sink).__name__ if self.overlay_sink is not None else None,
        )

    def _is_soft_reuse_boundary_char(self, ch: str) -> bool:
        return ch.isspace() or ch in _SOFT_REUSE_PUNCT

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

    def _upsert_merge_part(self, buffer: _MergeBuffer, utterance_id: UUID, text: str) -> None:
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
        buffer.spec_latency_stage_times.clear()

    def _record_spec_latency_stage(
        self,
        buffer: _MergeBuffer,
        *,
        stage: str,
        timestamp: float | None = None,
    ) -> None:
        buffer.spec_latency_stage_times[stage] = (
            self.clock.now() if timestamp is None else timestamp
        )

    def _promote_spec_latency_to_output(self, buffer: _MergeBuffer) -> None:
        if not buffer.spec_latency_stage_times:
            return
        for stage in ("llm_request_start", "llm_first_chunk", "llm_done"):
            timestamp = buffer.spec_latency_stage_times.get(stage)
            if timestamp is None:
                continue
            self._record_latency_stage(
                channel="self",
                utterance_id=buffer.merge_id,
                stage=stage,
                timestamp=timestamp,
                publish_now=False,
            )
        self._clear_spec_latency_state(buffer)
        self._emit_latency_contract_if_ready(channel="self", utterance_id=buffer.merge_id)

    def _clear_spec_state(self, buffer: _MergeBuffer, *, reason: str) -> bool:
        had_spec_state = any(
            value is not None
            for value in (
                buffer.spec_task,
                buffer.spec_translation,
                buffer.spec_text,
                buffer.spec_started_at,
                buffer.spec_done_at,
            )
        ) or bool(buffer.spec_latency_stage_times)
        if not had_spec_state:
            return False
        if buffer.spec_task is not None and not buffer.spec_task.done():
            buffer.spec_task.cancel()
            self._emit_metric(
                "[Metric] spec_cancel id=%s reason=%s",
                str(buffer.merge_id)[:8],
                reason,
            )
        elif buffer.spec_translation is not None:
            self._emit_metric(
                "[Metric] spec_cancel id=%s reason=%s",
                str(buffer.merge_id)[:8],
                reason,
            )
        self._clear_spec_latency_state(buffer)
        buffer.spec_task = None
        buffer.spec_translation = None
        buffer.spec_text = None
        buffer.spec_started_at = None
        buffer.spec_done_at = None
        return True

    def _maybe_update_buffer_end_time(self, utterance_id: UUID) -> None:
        buffer = self._merge_buffer
        if buffer is None or utterance_id not in buffer.utterance_ids:
            return
        end_time = self._utterance_start_times.get(utterance_id)
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
        buffer = self._merge_buffer
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
        if self.low_latency_awaiting_vad_timeout_s <= 0:
            return
        self._cancel_awaiting_vad_timeout(buffer)
        buffer.awaiting_vad_timeout_task = asyncio.create_task(
            self._awaiting_vad_timeout(buffer.merge_id)
        )

    async def _awaiting_vad_timeout(self, merge_id: UUID) -> None:
        try:
            await asyncio.sleep(self.low_latency_awaiting_vad_timeout_s)
        except asyncio.CancelledError:
            return
        buffer = self._merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        if not buffer.awaiting_vad_end:
            return
        self._emit_metric(
            "[Metric] awaiting_vad_timeout id=%s timeout_s=%s",
            str(merge_id)[:8],
            self.low_latency_awaiting_vad_timeout_s,
        )
        buffer.awaiting_vad_end = False
        buffer.awaiting_vad_utterance_id = None
        buffer.awaiting_vad_timeout_task = None
        self._restart_post_end_grace(buffer)

    def _cancel_resume_end_timeout(self, buffer: _MergeBuffer) -> None:
        task = buffer.resume_end_timeout_task
        if task is not None and task is not asyncio.current_task():
            if not task.done():
                task.cancel()
        buffer.resume_end_timeout_task = None
        buffer.resume_end_utterance_id = None

    def _start_resume_end_timeout(self, buffer: _MergeBuffer, utterance_id: UUID) -> None:
        self._cancel_resume_end_timeout(buffer)
        buffer.resume_end_utterance_id = utterance_id
        buffer.resume_end_timeout_task = asyncio.create_task(
            self._resume_end_timeout(buffer.merge_id, utterance_id)
        )

    async def _resume_end_timeout(self, merge_id: UUID, utterance_id: UUID) -> None:
        try:
            await asyncio.sleep(self.low_latency_awaiting_vad_timeout_s)
        except asyncio.CancelledError:
            return
        buffer = self._merge_buffer
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
            self.low_latency_awaiting_vad_timeout_s,
        )
        self._clear_resume_state(buffer)
        self._cancel_finalize_wait(buffer)
        await self._try_commit_after_spec(buffer, reason="resume_end_timeout", allow_fallback=True)

    def _restart_post_end_grace(self, buffer: _MergeBuffer) -> None:
        if self.low_latency_finalize_wait_ms <= 0:
            self._cancel_finalize_wait(buffer)
            return
        self._cancel_finalize_wait(buffer)
        buffer.finalize_wait_started_at = self.clock.now()
        buffer.finalize_wait_task = asyncio.create_task(
            self._finalize_wait_timeout(buffer.merge_id, buffer.finalize_wait_started_at)
        )
        self._emit_metric(
            "[Metric] post_end_grace_start id=%s wait_ms=%s",
            str(buffer.merge_id)[:8],
            self.low_latency_finalize_wait_ms,
        )

    async def _finalize_wait_timeout(self, merge_id: UUID, started_at: float) -> None:
        try:
            await asyncio.sleep(self.low_latency_finalize_wait_ms / 1000.0)
        except asyncio.CancelledError:
            return
        buffer = self._merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        if buffer.finalize_wait_started_at != started_at:
            return
        buffer.finalize_wait_task = None
        buffer.finalize_wait_started_at = None
        self._emit_metric(
            "[Metric] post_end_grace_timeout id=%s wait_ms=%s",
            str(merge_id)[:8],
            self.low_latency_finalize_wait_ms,
        )
        if self.llm is None or not self.translation_enabled:
            await self._commit_merge(buffer, reason="post_end_grace")
            return
        await self._try_commit_after_spec(buffer, reason="post_end_grace", allow_fallback=False)

    def _mark_resume_pending(self, event: SpeechStart) -> None:
        buffer = self._merge_buffer
        if buffer is None:
            return
        if buffer.resume_pending and buffer.resume_utterance_id == event.utterance_id:
            return
        # 새 resume 시작 시 이전 타임아웃 취소
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
        buffer = self._merge_buffer
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
        buffer = self._merge_buffer
        if buffer is None:
            return
        if buffer.resume_utterance_id != event.utterance_id:
            return
        if buffer.resume_confirmed:
            # resume_confirmed 상태에서 SpeechEnd → STT Final 대기 타임아웃 시작
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
        await self._try_commit_after_spec(buffer, reason="resume_false_start", allow_fallback=True)

    async def _handle_low_latency_final(self, transcript: Transcript) -> None:
        text = transcript.text.strip()
        if not text:
            return

        self._record_latency_stage(
            channel="self",
            utterance_id=transcript.utterance_id,
            stage="stt_final",
            publish_now=False,
        )

        now = self.clock.now()
        buffer = self._merge_buffer
        if buffer is None:
            buffer = _MergeBuffer(merge_id=uuid4(), start_time=now, last_final_at=now)
            self._merge_buffer = buffer
        if buffer.resume_pending or buffer.resume_confirmed:
            self._clear_resume_state(buffer)
        self._upsert_merge_part(buffer, transcript.utterance_id, text)
        buffer.last_final_at = now
        await self._sync_overlay_active_self(buffer, created_at=transcript.created_at)

        end_time = self._utterance_start_times.get(transcript.utterance_id)
        speech_already_ended = transcript.utterance_id in self._speech_ended_ids

        if end_time is None and not speech_already_ended:
            # SpeechEnd has not arrived yet - wait for it
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
            # SpeechEnd already arrived (or end_time exists) - proceed to post_end
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

        if self.llm is None or not self.translation_enabled:
            await self._commit_merge(buffer, reason="final_no_llm")
            return

        await self._maybe_restart_spec(buffer)

    async def _commit_merge(self, buffer: _MergeBuffer, *, reason: str) -> None:
        if buffer.resume_pending or buffer.resume_confirmed:
            hold_ms = 0
            if buffer.spec_done_at is not None:
                hold_ms = int((self.clock.now() - buffer.spec_done_at) * 1000)
            self._emit_metric(
                "[Metric] commit_blocked id=%s reason=%s hold_ms=%s",
                str(buffer.merge_id)[:8],
                reason,
                hold_ms,
            )
            return
        if buffer.awaiting_vad_end:
            hold_ms = 0
            if buffer.finalize_wait_started_at is not None:
                hold_ms = int((self.clock.now() - buffer.finalize_wait_started_at) * 1000)
            self._emit_metric(
                "[Metric] commit_blocked id=%s reason=await_vad_end hold_ms=%s",
                str(buffer.merge_id)[:8],
                hold_ms,
            )
            return
        if buffer.finalize_wait_task is not None:
            hold_ms = 0
            if buffer.finalize_wait_started_at is not None:
                hold_ms = int((self.clock.now() - buffer.finalize_wait_started_at) * 1000)
            self._emit_metric(
                "[Metric] commit_deferred id=%s reason=post_end_grace hold_ms=%s",
                str(buffer.merge_id)[:8],
                hold_ms,
            )
            return
        self._cancel_finalize_wait(buffer)
        buffer.awaiting_vad_end = False
        buffer.awaiting_vad_utterance_id = None
        for utterance_id in buffer.utterance_ids:
            self._utterance_start_times.pop(utterance_id, None)
            self._speech_ended_ids.discard(utterance_id)
        if self._merge_buffer is buffer:
            self._merge_buffer = None

        final_text = self._merge_text(buffer.parts)
        if not final_text:
            await self.reset_overlay_preview()
            return

        reuse_mode = None
        if buffer.spec_translation is not None:
            reuse_mode = self._soft_reuse_mode(buffer.spec_text, final_text)

        if self._should_blank_stale_active_secondary_before_finalizing(
            final_text=final_text,
            reuse_mode=reuse_mode,
        ):
            source_language, target_language = self._self_overlay_languages_for_utterance(
                buffer.merge_id
            )
            self._record_overlay_emit(
                event_kind="active_self",
                utterance_id=buffer.merge_id,
                channel="self",
                secondary_len=0,
            )
            await self._emit_self_active_overlay_event(
                self.overlay_event_adapter.self_active_update(
                    text=final_text,
                    utterance_id=buffer.merge_id,
                    secondary_text="",
                    occupant_key=self._active_self_occupant_key(buffer),
                    source_language=source_language,
                    target_language=target_language,
                    created_at=self.clock.now(),
                )
            )

        if buffer.spec_task is not None and not buffer.spec_task.done():
            buffer.spec_task.cancel()

        if buffer.last_end_time is not None:
            self._utterance_start_times[buffer.merge_id] = buffer.last_end_time
        elif buffer.start_time is not None:
            self._utterance_start_times[buffer.merge_id] = buffer.start_time
        self._inherit_latency_for_output(
            channel="self",
            output_utterance_id=buffer.merge_id,
            source_utterance_ids=buffer.utterance_ids,
        )
        for utterance_id in buffer.utterance_ids:
            self._clear_latency_timeline(channel="self", utterance_id=utterance_id)

        transcript = Transcript(
            utterance_id=buffer.merge_id,
            text=final_text,
            is_final=True,
            created_at=self.clock.now(),
        )
        await self._handle_transcript(transcript, is_final=True, source="Mic")

        if self.llm is None or not self.translation_enabled:
            self._log_translation_skipped(
                stage="final",
                runtime=self.self_runtime,
                publish_chatbox=True,
            )
            await self._enqueue_osc(
                buffer.merge_id, transcript_text=final_text, translation_text=None
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
            translation = buffer.spec_translation
            if translation is not None:
                self._promote_spec_latency_to_output(buffer)
                self._emit_metric(
                    "[Metric] spec_reuse id=%s translation_len=%s after_final=%s",
                    str(buffer.merge_id)[:8],
                    len(translation.text),
                    True,
                )
                bundle = self.get_or_create_bundle(buffer.merge_id)
                bundle.with_translation(translation)
                bundle.with_translation(translation)
                self._emit_translation_ready_for_output(
                    translation=translation,
                    runtime=self.self_runtime,
                )
                self._remember_context_entry(final_text, self.clock.now())
                await self.ui_events.put(
                    UIEvent(
                        type=UIEventType.TRANSLATION_DONE,
                        utterance_id=buffer.merge_id,
                        payload=translation,
                        source=self._get_source(buffer.merge_id),
                    )
                )
                await self._emit_translation_to_overlay(
                    translation=translation,
                    applied_context_mode=None,
                )
                await self._emit_overlay_utterance_closed(
                    utterance_id=buffer.merge_id,
                    channel="self",
                    is_final=True,
                )
                await self._enqueue_osc(
                    buffer.merge_id,
                    transcript_text=final_text,
                    translation_text=translation.text,
                )
                return

        if buffer.spec_translation is not None and reuse_mode is None:
            self._clear_spec_latency_state(buffer)
            self._emit_metric(
                "[Metric] spec_cancel id=%s reason=final_mismatch", str(buffer.merge_id)[:8]
            )

        await self._translate_and_enqueue(buffer.merge_id, final_text)

    async def _maybe_restart_spec(self, buffer: _MergeBuffer) -> None:
        if self.llm is None or not self.translation_enabled:
            return

        self._clear_spec_state(buffer, reason="spec_retry")

        merged_text = self._merge_text(buffer.parts)
        if not merged_text:
            return

        buffer.spec_attempts += 1
        buffer.spec_text = merged_text
        buffer.spec_started_at = self.clock.now()
        self._emit_metric(
            "[Metric] spec_start id=%s text_len=%s attempt=%s",
            str(buffer.merge_id)[:8],
            len(merged_text),
            buffer.spec_attempts,
        )
        buffer.spec_task = asyncio.create_task(
            self._run_spec_translation(buffer.merge_id, merged_text, buffer.spec_attempts)
        )

    async def _run_spec_translation(self, merge_id: UUID, text: str, attempt: int) -> None:
        if self.llm is None:
            return
        buffer = self._merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        if buffer.spec_text != text or buffer.spec_attempts != attempt:
            return
        self._record_spec_latency_stage(buffer, stage="llm_request_start")
        try:
            translation = await self._translate_text(merge_id, text, record_latency=False)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._log_translation_failure(
                stage="spec",
                runtime=self.self_runtime,
                exc=exc,
                detailed=True,
            )
            buffer = self._merge_buffer
            if buffer is None or buffer.merge_id != merge_id:
                return
            if buffer.spec_text != text or buffer.spec_attempts != attempt:
                return
            self._clear_spec_latency_state(buffer)
            buffer.spec_done_at = self.clock.now()
            await self._try_commit_after_spec(buffer, reason="spec_failed", allow_fallback=True)
            return

        buffer = self._merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        if buffer.spec_text != text or buffer.spec_attempts != attempt:
            return

        self._record_spec_latency_stage(buffer, stage="llm_done")
        buffer.spec_translation = translation
        buffer.spec_done_at = self.clock.now()
        if buffer.spec_started_at is None:
            latency_ms = 0
        else:
            latency_ms = int((self.clock.now() - buffer.spec_started_at) * 1000)
        self._emit_metric(
            "[Metric] spec_done id=%s spec_latency_ms=%s translation_len=%s",
            str(merge_id)[:8],
            latency_ms,
            len(translation.text),
        )
        await self._sync_overlay_active_self(buffer, created_at=translation.created_at)
        await self._try_commit_after_spec(buffer, reason="spec_done", allow_fallback=False)

    async def _try_commit_after_spec(
        self, buffer: _MergeBuffer, *, reason: str, allow_fallback: bool
    ) -> None:
        if self._merge_buffer is None or self._merge_buffer is not buffer:
            return
        if buffer.resume_pending or buffer.resume_confirmed:
            hold_ms = 0
            if buffer.spec_done_at is not None:
                hold_ms = int((self.clock.now() - buffer.spec_done_at) * 1000)
            self._emit_metric(
                "[Metric] commit_blocked id=%s reason=%s hold_ms=%s",
                str(buffer.merge_id)[:8],
                reason,
                hold_ms,
            )
            return
        if buffer.awaiting_vad_end:
            hold_ms = 0
            if buffer.finalize_wait_started_at is not None:
                hold_ms = int((self.clock.now() - buffer.finalize_wait_started_at) * 1000)
            self._emit_metric(
                "[Metric] commit_blocked id=%s reason=await_vad_end hold_ms=%s",
                str(buffer.merge_id)[:8],
                hold_ms,
            )
            return
        if buffer.finalize_wait_task is not None:
            hold_ms = 0
            if buffer.finalize_wait_started_at is not None:
                hold_ms = int((self.clock.now() - buffer.finalize_wait_started_at) * 1000)
            self._emit_metric(
                "[Metric] commit_deferred id=%s reason=post_end_grace hold_ms=%s",
                str(buffer.merge_id)[:8],
                hold_ms,
            )
            return

        final_text = self._merge_text(buffer.parts)
        if not final_text:
            return

        if buffer.spec_translation is None:
            if not allow_fallback:
                return
            await self._commit_merge(buffer, reason=reason)
            return

        if self._soft_reuse_mode(buffer.spec_text, final_text) is None:
            return

        await self._commit_merge(buffer, reason=reason)

    def _remember_source(
        self,
        utterance_id: UUID,
        source: str | None,
        *,
        channel: ChannelId = "self",
    ) -> None:
        self._runtime_for_utterance(utterance_id, default_channel=channel).remember_source(
            utterance_id, source
        )

    def _get_source(self, utterance_id: UUID, *, channel: ChannelId = "self") -> str | None:
        runtime = self._runtime_for_utterance(utterance_id, default_channel=channel)
        source = runtime.get_source(utterance_id)
        if source is not None:
            return source
        other_runtime = self.peer_runtime if runtime is self.self_runtime else self.self_runtime
        return other_runtime.get_source(utterance_id)

    def _source_language_for(self, runtime: ChannelRuntime) -> str:
        if runtime.channel == "peer" and self.peer_source_language:
            return self.peer_source_language
        return self.source_language

    def _target_language_for(self, runtime: ChannelRuntime) -> str:
        if runtime.channel == "peer" and self.peer_target_language:
            return self.peer_target_language
        return self.target_language

    def _format_system_prompt(self, runtime: ChannelRuntime | None = None) -> str:
        runtime = runtime or self.self_runtime
        return render_translation_prompt_template(
            self.system_prompt,
            source_name=get_llm_language_name(self._source_language_for(runtime)),
            target_name=get_llm_language_name(self._target_language_for(runtime)),
        )

    def _other_runtime(self, runtime: ChannelRuntime) -> ChannelRuntime:
        return self.peer_runtime if runtime is self.self_runtime else self.self_runtime

    def _should_publish_to_chatbox(self, runtime: ChannelRuntime) -> bool:
        return runtime.channel == self.active_chatbox_channel

    def _translation_enabled_for_runtime(self, runtime: ChannelRuntime) -> bool:
        if runtime.channel == "peer":
            return self.translation_enabled and self.peer_translation_enabled
        return self.translation_enabled

    def _prepare_llm_request(
        self,
        text: str,
        *,
        runtime: ChannelRuntime | None = None,
    ) -> tuple[str, str, float]:
        formatted_prompt, context_str, now, _ = self._prepare_llm_request_with_mode(
            text,
            runtime=runtime,
        )
        return formatted_prompt, context_str, now

    def _prepare_llm_request_with_mode(
        self,
        text: str,
        *,
        runtime: ChannelRuntime | None = None,
    ) -> tuple[str, str, float, ContextMode]:
        _ = text
        runtime = runtime or self.self_runtime
        requested_mode: ContextMode = "integrated" if self.integrated_context_enabled else "local"
        now = self.clock.now()
        other_runtime = self._other_runtime(runtime)
        context_str, applied_mode = self.context_resolver.resolve_for_request(
            runtime=runtime,
            other_runtime=other_runtime,
            requested_mode=requested_mode,
            peer_translation_enabled=self.peer_translation_enabled,
            source_language=self._source_language_for(runtime),
            target_language=self._target_language_for(runtime),
            other_source_language=self._source_language_for(other_runtime),
            other_target_language=self._target_language_for(other_runtime),
        )
        self._log_context_mode_change(runtime=runtime, applied_mode=applied_mode)
        self._log_context_application(text=text, runtime=runtime, context=context_str)
        formatted_prompt = self._format_system_prompt(runtime)
        return formatted_prompt, context_str, now, applied_mode

    def _normalize_translation(
        self,
        translation: Translation,
        *,
        runtime: ChannelRuntime,
        text: str,
        source_language: str,
        target_language: str,
    ) -> Translation:
        return Translation(
            utterance_id=translation.utterance_id,
            translated_text=translation.text,
            source_text=text,
            source_language=self._language_or_fallback(
                translation.source_language,
                source_language,
            ),
            target_language=self._language_or_fallback(
                translation.target_language,
                target_language,
            ),
            channel=runtime.channel,
            created_at=translation.created_at,
            update_id=translation.update_id,
            origin_wall_clock_ms=translation.origin_wall_clock_ms,
            session_scope=translation.session_scope,
            source_text_hash=translation.source_text_hash,
            source_text_len=translation.source_text_len,
            logical_turn_key=f"{runtime.channel}:{translation.utterance_id}",
        )

    async def _translate_text(
        self,
        utterance_id: UUID,
        text: str,
        *,
        runtime: ChannelRuntime | None = None,
        record_latency: bool = True,
    ) -> Translation:
        if self.llm is None:
            raise RuntimeError("LLM is not configured")

        runtime = runtime or self.self_runtime
        formatted_prompt, context_str, _ = self._prepare_llm_request(
            text,
            runtime=runtime,
        )
        if record_latency:
            self._record_latency_stage(
                channel=runtime.channel,
                utterance_id=utterance_id,
                stage="llm_request_start",
            )
        request_source_language = self._source_language_for(runtime)
        request_target_language = self._target_language_for(runtime)
        translation = await self.llm.translate(
            utterance_id=utterance_id,
            text=text,
            system_prompt=formatted_prompt,
            source_language=request_source_language,
            target_language=request_target_language,
            context=context_str,
        )
        if record_latency:
            self._record_latency_stage(
                channel=runtime.channel,
                utterance_id=utterance_id,
                stage="llm_done",
            )
        return self._normalize_translation(
            translation,
            runtime=runtime,
            text=text,
            source_language=request_source_language,
            target_language=request_target_language,
        )

    async def _ensure_translation(self, transcript: Transcript) -> None:
        if self.llm is None:
            return
        runtime = self._runtime_for_channel(transcript.channel)
        if not self._translation_enabled_for_runtime(runtime):
            return
        utterance_id = transcript.utterance_id
        if utterance_id in runtime.translation_tasks:
            return
        task = asyncio.create_task(
            self._translate_and_enqueue(
                utterance_id,
                transcript.text,
                runtime=runtime,
            )
        )
        runtime.translation_tasks[utterance_id] = task
        task.add_done_callback(lambda _t: runtime.translation_tasks.pop(utterance_id, None))

    async def _translate_and_enqueue(
        self,
        utterance_id: UUID,
        text: str,
        *,
        runtime: ChannelRuntime | None = None,
    ) -> None:
        if self.llm is None:
            return
        runtime = runtime or self.self_runtime
        applied_mode: ContextMode | None = None
        peer_overlay_active = runtime.channel == "peer" and self.overlay_sink is not None
        try:
            formatted_prompt, context_str, now, applied_mode = self._prepare_llm_request_with_mode(
                text,
                runtime=runtime,
            )

            # Add current text to context history at REQUEST time
            self._remember_context_entry(
                text,
                now,
                runtime=runtime,
            )
            self._record_latency_stage(
                channel=runtime.channel,
                utterance_id=utterance_id,
                stage="llm_request_start",
            )

            request_source_language = self._source_language_for(runtime)
            request_target_language = self._target_language_for(runtime)
            raw_translation = await self.llm.translate(
                utterance_id=utterance_id,
                text=text,
                system_prompt=formatted_prompt,
                source_language=request_source_language,
                target_language=request_target_language,
                context=context_str,
            )
            translation = self._normalize_translation(
                raw_translation,
                runtime=runtime,
                text=text,
                source_language=request_source_language,
                target_language=request_target_language,
            )
            self._record_latency_stage(
                channel=runtime.channel,
                utterance_id=utterance_id,
                stage="llm_done",
            )
        except asyncio.CancelledError:
            if runtime.channel == "self":
                await self._emit_overlay_utterance_closed(
                    utterance_id=utterance_id,
                    channel=runtime.channel,
                    is_final=False,
                    finalize_latency=not self._should_publish_to_chatbox(runtime),
                )
            elif runtime.channel == "peer":
                await self._finalize_peer_source_only(
                    Transcript(
                        utterance_id=utterance_id,
                        text=text,
                        is_final=True,
                        created_at=self.clock.now(),
                        channel="peer",
                    ),
                    close_is_final=False,
                    finalize_latency=True,
                )
            else:
                self._finalize_latency_timeline(channel=runtime.channel, utterance_id=utterance_id)
            raise
        except Exception as exc:
            self._log_translation_failure(stage="final", runtime=runtime, exc=exc)
            fallback_to_chatbox = self.fallback_transcript_only and self._should_publish_to_chatbox(
                runtime
            )
            payload: object = exc if isinstance(exc, ManagedOpenRouterUserFacingError) else str(exc)
            await self.ui_events.put(
                UIEvent(
                    type=UIEventType.ERROR,
                    utterance_id=utterance_id,
                    payload=payload,
                    source=self._get_source(utterance_id, channel=runtime.channel),
                    channel=runtime.channel,
                    runtime_log_handled=True,
                )
            )
            if runtime.channel == "self":
                await self._emit_overlay_utterance_closed(
                    utterance_id=utterance_id,
                    channel=runtime.channel,
                    is_final=False,
                    finalize_latency=not (
                        self.fallback_transcript_only and self._should_publish_to_chatbox(runtime)
                    ),
                )
            elif runtime.channel == "peer":
                await self._finalize_peer_source_only(
                    Transcript(
                        utterance_id=utterance_id,
                        text=text,
                        is_final=True,
                        created_at=self.clock.now(),
                        channel="peer",
                    ),
                    close_is_final=False,
                    finalize_latency=not fallback_to_chatbox,
                )
            if fallback_to_chatbox:
                await self._enqueue_osc(
                    utterance_id,
                    transcript_text=text,
                    translation_text=None,
                )
            elif runtime.channel != "peer":
                self._finalize_latency_timeline(channel=runtime.channel, utterance_id=utterance_id)
            return

        publish_to_chatbox = self._should_publish_to_chatbox(runtime)
        bundle = self.get_or_create_bundle(utterance_id, channel=runtime.channel)
        bundle.with_translation(translation)
        self._emit_translation_ready_for_output(
            translation=translation,
            runtime=runtime,
        )
        if peer_overlay_active:
            await self._emit_peer_translation_to_overlay(
                translation=translation,
                runtime=runtime,
                applied_context_mode=applied_mode,
            )
            await self._emit_overlay_utterance_closed(
                utterance_id=utterance_id,
                channel=runtime.channel,
                is_final=True,
                finalize_latency=not publish_to_chatbox,
            )
        await self.ui_events.put(
            UIEvent(
                type=UIEventType.TRANSLATION_DONE,
                utterance_id=utterance_id,
                payload=translation,
                source=self._get_source(utterance_id, channel=runtime.channel),
            )
        )
        if runtime.channel == "self":
            await self._emit_translation_to_overlay(
                translation=translation,
                applied_context_mode=applied_mode,
            )
            await self._emit_overlay_utterance_closed(
                utterance_id=utterance_id,
                channel=runtime.channel,
                is_final=True,
                finalize_latency=not self._should_publish_to_chatbox(runtime),
            )
        if publish_to_chatbox:
            await self._enqueue_osc(
                utterance_id,
                transcript_text=text,
                translation_text=translation.text,
            )
        else:
            self._finalize_latency_timeline(channel=runtime.channel, utterance_id=utterance_id)
        if runtime.channel == "peer":
            self._complete_peer_logical_turn(utterance_id)

    async def handle_peer_transcript_final_for_test(
        self,
        text: str,
        source: str = "Peer",
    ) -> UUID:
        _ = source
        parent_utterance_id = uuid4()
        before_event_count = 0
        if hasattr(self.overlay_sink, "events"):
            before_event_count = len(self.overlay_sink.events)  # type: ignore[attr-defined]
        existing_peer_utterance_ids = set(self.peer_runtime.utterances)
        await self._handle_stt_event(
            STTFinalEvent(
                utterance_id=parent_utterance_id,
                transcript=Transcript(
                    utterance_id=parent_utterance_id,
                    text=text,
                    is_final=True,
                    created_at=self.clock.now(),
                    channel="peer",
                ),
            )
        )
        if hasattr(self.overlay_sink, "events"):
            new_events = self.overlay_sink.events[before_event_count:]  # type: ignore[attr-defined]
            for event in new_events:
                if getattr(event, "type", None) == "peer_active_update":
                    return event.utterance_id
        for utterance_id, bundle in self.peer_runtime.utterances.items():
            if utterance_id in existing_peer_utterance_ids:
                continue
            if bundle.final is not None and bundle.final.text == text:
                return utterance_id
        raise AssertionError("peer test helper did not produce a peer logical turn")

    async def translate_peer_text_for_test(
        self,
        text: str,
    ) -> UUID:
        utterance_id = await self.handle_peer_transcript_final_for_test(
            text=text,
        )
        if self.peer_runtime.translation_tasks:
            await asyncio.gather(
                *self.peer_runtime.translation_tasks.values(), return_exceptions=True
            )
        return utterance_id

    async def _enqueue_osc(
        self,
        utterance_id: UUID,
        *,
        transcript_text: str,
        translation_text: str | None,
    ) -> None:
        if translation_text is None:
            merged = transcript_text
        elif self.chatbox_include_source:
            merged = f"{transcript_text} ({translation_text})"
        else:
            merged = translation_text

        msg = OSCMessage(utterance_id=utterance_id, text=merged, created_at=self.clock.now())
        runtime = self._runtime_for_utterance(utterance_id)

        self._emit_detailed(
            "[Hub] OSC enqueue preview: channel=%s text=%r",
            runtime.channel,
            merged,
            fallback_level=logging.INFO,
        )
        if runtime.channel == "self":
            self._record_latency_stage(
                channel=runtime.channel,
                utterance_id=utterance_id,
                stage="self_chatbox_enqueue",
            )

        runtime.utterance_start_times.pop(utterance_id, None)
        runtime.speech_ended_ids.discard(utterance_id)

        self.osc.enqueue(msg)

        if runtime.channel == "self":
            self._set_osc_typing_reason(_SELF_SPEECH_TYPING_REASON, False)

        await self.ui_events.put(
            UIEvent(
                type=UIEventType.OSC_SENT,
                utterance_id=utterance_id,
                payload=msg,
                source=self._get_source(utterance_id),
                channel=runtime.channel,
            )
        )
        self._clear_latency_timeline(channel=runtime.channel, utterance_id=utterance_id)

    def enqueue_peer_translation_disclosure(self, text: str) -> None:
        msg = OSCMessage(utterance_id=uuid4(), text=text, created_at=self.clock.now())
        self._emit_detailed(
            "[Hub] OSC disclosure enqueue: channel=peer text_len=%s",
            len(text),
            fallback_level=logging.INFO,
        )
        self.osc.enqueue(msg)

    def _set_osc_typing_reason(self, reason: str, active: bool) -> None:
        set_reason = getattr(self.osc, "set_typing_reason", None)
        if callable(set_reason):
            set_reason(reason, active)
            return
        self.osc.send_typing(active)

    def _clear_osc_typing_reasons(self) -> None:
        clear_reasons = getattr(self.osc, "clear_typing_reasons", None)
        if callable(clear_reasons):
            clear_reasons()
            return
        self.osc.send_typing(False)

    async def _run_osc_flush_loop(self) -> None:
        try:
            while True:
                self.osc.process_due()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
