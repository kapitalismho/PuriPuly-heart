from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast
from uuid import UUID

from puripuly_heart.core.clock import Clock
from puripuly_heart.core.error_messages import (
    format_error_report_for_log,
    provider_failure_report,
    stt_failure_report,
)
from puripuly_heart.core.messages import UserErrorReport
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
)
from puripuly_heart.core.orchestrator.context import ContextMode
from puripuly_heart.core.orchestrator.ports import (
    TranslationRuntimeLoggingPort,
    format_basic_latency_summary,
    format_detailed_latency_breakdown,
    format_detailed_latency_trace,
    format_latency_cause_metric,
    format_translation_ready_for_output,
    runtime_logging_mode_is_detailed,
)
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.domain.models import ChannelId

_LATENCY_TRACE_ORDER = (
    "last_speech",
    "speech_end",
    "stt_final",
    "llm_request_start",
    "llm_first_chunk",
    "llm_done",
    "self_chatbox_send",
    "peer_overlay_applied",
    "dashboard_translation_applied",
    "peer_overlay_first_render",
)
_LATENCY_SUMMARY_OUTPUT_STAGES = (
    ("self", "self_chatbox_send", "chatbox_send"),
    ("peer", "peer_overlay_applied", "overlay_applied"),
)


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostic:
    message: str
    args: tuple[object, ...] = ()
    level: int = logging.INFO
    fallback_level: int | None = None
    detailed: bool = False
    safe_exceptions: bool = False


@dataclass(frozen=True, slots=True)
class SttEventLoopFailureDiagnostic:
    exception: Exception
    provider: object | None
    default_channel: ChannelId


@dataclass(frozen=True, slots=True)
class SttTurnFailureDiagnostic:
    exception: Exception
    provider: object | None
    channel: ChannelId


@dataclass(frozen=True, slots=True)
class TranslationSkipDiagnostic:
    stage: str
    channel: ChannelId
    publish_chatbox: bool
    llm_available: bool
    configuration: TranslationRuntimeConfig
    parent_utterance_id: UUID | None = None
    target_index: int | None = None
    target_language: str | None = None
    cause: str | None = None
    segment_count: int | None = None


@dataclass(frozen=True, slots=True)
class TranslationFailureDiagnostic:
    stage: str
    channel: ChannelId
    exception: Exception
    detailed: bool = False
    parent_utterance_id: UUID | None = None
    target_index: int | None = None
    target_language: str | None = None


@dataclass(frozen=True, slots=True)
class ContextModeDiagnostic:
    channel: ChannelId
    applied_mode: ContextMode
    parent_utterance_id: UUID | None = None
    target_index: int | None = None
    target_language: str | None = None


@dataclass(frozen=True, slots=True)
class ContextApplicationDiagnostic:
    channel: ChannelId
    request_chars: int
    context_lines: tuple[str, ...]
    context_chars: int
    parent_utterance_id: UUID | None = None
    target_index: int | None = None
    target_language: str | None = None


@dataclass(frozen=True, slots=True)
class LatencyStageDiagnostic:
    channel: ChannelId
    utterance_id: UUID
    stage: str
    timestamp: float | None = None
    overwrite: bool = True
    publish_now: bool = True
    parent_utterance_id: UUID | None = None
    target_index: int | None = None
    target_language: str | None = None
    turn_generation: int | None = None
    turn_order: int | None = None


@dataclass(frozen=True, slots=True)
class LatencyInheritanceDiagnostic:
    channel: ChannelId
    output_utterance_id: UUID
    source_utterance_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class LatencyTimelineDiagnostic:
    channel: ChannelId
    utterance_id: UUID


@dataclass(frozen=True, slots=True)
class TranslationReadyDiagnostic:
    channel: ChannelId
    utterance_id: UUID
    update_id: str
    origin_wall_clock_ms: int | None
    session_scope: str | None
    source_text_hash: str | None
    source_text_len: int | None
    logical_turn_key: str | None
    translation_len: int
    parent_utterance_id: UUID | None = None
    target_index: int | None = None
    target_language: str | None = None
    turn_generation: int | None = None
    turn_order: int | None = None


@dataclass(frozen=True, slots=True)
class SelfOverlayDecisionDiagnostic:
    merge_id: UUID
    source: str
    active_text: str = field(repr=False)
    secondary_text: str = field(repr=False)
    active_text_len: int
    secondary_len: int
    spec_text_len: int
    spec_translation_len: int
    cached_secondary_len: int
    reuse_mode: str | None
    resume_pending: bool
    resume_confirmed: bool

    @classmethod
    def create(
        cls,
        *,
        merge_id: UUID,
        source: str,
        active_text: str,
        secondary_text: str,
        spec_text_len: int,
        spec_translation_len: int,
        cached_secondary_len: int,
        reuse_mode: str | None,
        resume_pending: bool,
        resume_confirmed: bool,
    ) -> SelfOverlayDecisionDiagnostic:
        return cls(
            merge_id=merge_id,
            source=source,
            active_text=active_text,
            secondary_text=secondary_text,
            active_text_len=len(active_text),
            secondary_len=len(secondary_text),
            spec_text_len=spec_text_len,
            spec_translation_len=spec_translation_len,
            cached_secondary_len=cached_secondary_len,
            reuse_mode=reuse_mode,
            resume_pending=resume_pending,
            resume_confirmed=resume_confirmed,
        )


@dataclass(frozen=True, slots=True)
class OverlayEmitDiagnostic:
    event_kind: str
    utterance_id: UUID
    channel: ChannelId
    secondary_len: int
    sink_type: str | None


@dataclass(frozen=True, slots=True)
class OverlaySinkDurationDiagnostic:
    event_type: str
    channel: object
    utterance_id: object
    update_id: object
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class TranslationLatencyDiagnosticsSnapshot:
    timeline_keys: frozenset[tuple[ChannelId, UUID]]
    last_error_source: str | None
    context_modes: tuple[tuple[ChannelId, ContextMode | None], ...]
    overlay_diagnostics_attached: bool


@dataclass(slots=True)
class _LatencyTimeline:
    channel: ChannelId
    stage_times: dict[str, float] = field(default_factory=dict)
    emitted_trace_points: set[str] = field(default_factory=set)
    basic_summary_emitted: bool = False
    latency_cause_emitted: bool = False
    parent_utterance_id: UUID | None = None
    target_index: int | None = None
    target_language: str | None = None
    turn_generation: int | None = None
    turn_order: int | None = None
    pending_sources: set[UUID] = field(default_factory=set)
    clear_requested: bool = False
    awaiting_speech_end: bool = False


@dataclass(slots=True)
class TranslationLatencyDiagnosticsOwner:
    clock: Clock
    config_snapshot: Callable[[], TranslationRuntimeConfigSnapshot]
    runtime_logging: TranslationRuntimeLoggingPort | None = None
    overlay_diagnostics: OverlayDiagnosticsRecorder | None = None
    fallback_logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("puripuly_heart.core.orchestrator.translation"),
        repr=False,
    )
    _last_context_modes: dict[ChannelId, ContextMode | None] = field(
        init=False,
        default_factory=lambda: {"self": None, "peer": None},
        repr=False,
    )
    _last_target_context_modes: dict[tuple[ChannelId, str], ContextMode] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _last_overlay_runtime_signature: tuple[object, ...] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _last_overlay_diagnostics_signature: tuple[object, ...] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _timelines: OrderedDict[tuple[ChannelId, UUID], _LatencyTimeline] = field(
        init=False,
        default_factory=OrderedDict,
        repr=False,
    )
    _awaiting_output: set[tuple[ChannelId, UUID]] = field(
        init=False,
        default_factory=set,
        repr=False,
    )
    _source_outputs: dict[tuple[ChannelId, UUID], set[UUID]] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _last_error_source: str | None = field(init=False, default=None, repr=False)

    def snapshot(self) -> TranslationLatencyDiagnosticsSnapshot:
        return TranslationLatencyDiagnosticsSnapshot(
            timeline_keys=frozenset(self._timelines),
            last_error_source=self._last_error_source,
            context_modes=tuple(self._last_context_modes.items()),
            overlay_diagnostics_attached=self.overlay_diagnostics is not None,
        )

    @property
    def detailed_enabled(self) -> bool:
        logging_port = self.runtime_logging
        return logging_port is not None and runtime_logging_mode_is_detailed(logging_port.mode)

    def emit(self, diagnostic: RuntimeDiagnostic) -> bool:
        args = diagnostic.args
        if diagnostic.safe_exceptions:
            args = tuple(self._safe_log_arg(arg) for arg in args)
        if diagnostic.detailed:
            if self.runtime_logging is None:
                return False
            return self.runtime_logging.emit_detailed_lazy(
                lambda: self._format_log_message(diagnostic.message, *args),
                level=diagnostic.level,
            )
        formatted = self._format_log_message(diagnostic.message, *args)
        if self.runtime_logging is not None:
            self.runtime_logging.emit_basic(formatted, level=diagnostic.level)
            return True
        level = diagnostic.level if diagnostic.fallback_level is None else diagnostic.fallback_level
        self.fallback_logger.log(level, formatted)
        return True

    def emit_metric(self, message: str, *args: object) -> bool:
        return self.emit(
            RuntimeDiagnostic(
                message=message,
                args=args,
                fallback_level=logging.DEBUG,
                detailed=True,
            )
        )

    def record_stt_event_loop_failure(
        self,
        diagnostic: SttEventLoopFailureDiagnostic,
    ) -> None:
        if self.runtime_logging is None:
            self.emit(
                RuntimeDiagnostic(
                    message="[Translation] STT event loop crashed: %s",
                    args=(diagnostic.exception,),
                    level=logging.ERROR,
                    safe_exceptions=True,
                )
            )
            return
        provider, channel = self._stt_failure_context(
            diagnostic.provider,
            default_channel=diagnostic.default_channel,
        )
        report = stt_failure_report(
            diagnostic.exception,
            provider=provider,
            operation="event_loop",
            channel=channel,
        )
        self.emit(
            RuntimeDiagnostic(
                message="[Translation] STT event loop crashed: %s",
                args=(format_error_report_for_log(report),),
                level=logging.ERROR,
            )
        )

    def record_stt_turn_failure(
        self,
        diagnostic: SttTurnFailureDiagnostic,
    ) -> UserErrorReport:
        provider, channel = self._stt_failure_context(
            diagnostic.provider,
            default_channel=diagnostic.channel,
        )
        report = stt_failure_report(
            diagnostic.exception,
            provider=provider,
            operation="turn_terminal",
            channel=channel,
        )
        self.emit(
            RuntimeDiagnostic(
                message="[Translation] scoped STT turn failed: %s",
                args=(format_error_report_for_log(report),),
                level=logging.ERROR,
            )
        )
        return report

    def record_translation_skip(self, diagnostic: TranslationSkipDiagnostic) -> None:
        parts = [
            "[Translation] turn_result",
            f"channel={diagnostic.channel}",
            f"parent_utterance_id={diagnostic.parent_utterance_id}",
            f"target_index={diagnostic.target_index}",
            f"target_language={diagnostic.target_language}",
            "translation=skipped",
            f"destination_chatbox={'intended' if diagnostic.publish_chatbox else 'disabled'}",
            f"cause={self._translation_skip_reason(diagnostic)}",
        ]
        if diagnostic.segment_count is not None:
            parts.append(f"segment_count={diagnostic.segment_count}")
        self.emit(
            RuntimeDiagnostic(
                message=" ".join(parts),
                fallback_level=logging.INFO,
            )
        )
        if diagnostic.target_language is not None:
            self.emit(
                RuntimeDiagnostic(
                    message=(
                        "[Detailed][Translation] translation_target_skipped "
                        "parent_utterance_id=%s target_index=%s target_language=%s"
                    ),
                    args=(
                        diagnostic.parent_utterance_id,
                        diagnostic.target_index,
                        diagnostic.target_language,
                    ),
                    detailed=True,
                )
            )

    def record_translation_failure(
        self,
        diagnostic: TranslationFailureDiagnostic,
    ) -> UserErrorReport:
        report = provider_failure_report(
            diagnostic.exception,
            provider="llm",
            operation="translate",
        )
        self.emit(
            RuntimeDiagnostic(
                message=(
                    "[Translation] turn_result channel=%s parent_utterance_id=%s "
                    "target_index=%s target_language=%s translation=failed stage=%s cause=%s"
                ),
                args=(
                    diagnostic.channel,
                    diagnostic.parent_utterance_id,
                    diagnostic.target_index,
                    diagnostic.target_language,
                    diagnostic.stage,
                    format_error_report_for_log(report),
                ),
                level=logging.ERROR,
                fallback_level=logging.ERROR,
                detailed=diagnostic.detailed,
            )
        )
        if diagnostic.target_language is not None:
            self.emit(
                RuntimeDiagnostic(
                    message=(
                        "[Detailed][Translation] translation_target_failed "
                        "parent_utterance_id=%s target_index=%s target_language=%s"
                    ),
                    args=(
                        diagnostic.parent_utterance_id,
                        diagnostic.target_index,
                        diagnostic.target_language,
                    ),
                    detailed=True,
                )
            )
        return report

    def record_context_mode(self, diagnostic: ContextModeDiagnostic) -> None:
        target_key = (
            (diagnostic.channel, diagnostic.target_language)
            if diagnostic.target_language is not None
            else None
        )
        legacy_changed = self._last_context_modes.get(diagnostic.channel) != diagnostic.applied_mode
        target_changed = (
            target_key is not None
            and self._last_target_context_modes.get(target_key) != diagnostic.applied_mode
        )
        if legacy_changed:
            self._last_context_modes[diagnostic.channel] = diagnostic.applied_mode
            self.emit(
                RuntimeDiagnostic(
                    message="[Translation] Context mode: channel=%s mode=%s",
                    args=(diagnostic.channel, diagnostic.applied_mode),
                )
            )
        if target_key is not None:
            self._last_target_context_modes[target_key] = diagnostic.applied_mode
        if target_changed:
            self.emit(
                RuntimeDiagnostic(
                    message=(
                        "[Detailed][Translation] context_mode_target "
                        "parent_utterance_id=%s target_index=%s target_language=%s mode=%s"
                    ),
                    args=(
                        diagnostic.parent_utterance_id,
                        diagnostic.target_index,
                        diagnostic.target_language,
                        diagnostic.applied_mode,
                    ),
                    detailed=True,
                )
            )

    def record_context_application(
        self,
        diagnostic: ContextApplicationDiagnostic,
    ) -> None:
        applied_mode = (
            self._last_target_context_modes.get((diagnostic.channel, diagnostic.target_language))
            if diagnostic.target_language is not None
            else self._last_context_modes.get(diagnostic.channel)
        )
        peer_entries = sum(
            1
            for line in diagnostic.context_lines
            if line.startswith("- [peer]") or line.startswith("- [others]")
        )
        self_entries = len(diagnostic.context_lines) - peer_entries
        if diagnostic.target_language is None:
            self.emit(
                RuntimeDiagnostic(
                    message=(
                        "[Detailed][Translation] context_apply channel=%s mode=%s "
                        "request_chars=%s entries=%s self_entries=%s "
                        "peer_entries=%s context_chars=%s"
                    ),
                    args=(
                        diagnostic.channel,
                        applied_mode,
                        diagnostic.request_chars,
                        len(diagnostic.context_lines),
                        self_entries,
                        peer_entries,
                        diagnostic.context_chars,
                    ),
                    detailed=True,
                )
            )
        if diagnostic.target_language is not None:
            self.emit(
                RuntimeDiagnostic(
                    message=(
                        "[Detailed][Translation] context_apply_target "
                        "parent_utterance_id=%s target_index=%s target_language=%s "
                        "mode=%s request_chars=%s entries=%s self_entries=%s "
                        "peer_entries=%s context_chars=%s"
                    ),
                    args=(
                        diagnostic.parent_utterance_id,
                        diagnostic.target_index,
                        diagnostic.target_language,
                        applied_mode,
                        diagnostic.request_chars,
                        len(diagnostic.context_lines),
                        self_entries,
                        peer_entries,
                        diagnostic.context_chars,
                    ),
                    detailed=True,
                )
            )

    def record_chatbox_stage(self, event: str, **fields: object) -> None:
        if event in {"chatbox_revision_replaced", "chatbox_older_turn_pruned"}:
            self.emit(
                RuntimeDiagnostic(
                    message=(
                        "[Detailed][Translation] %s parent_utterance_id=%s "
                        "turn_generation=%s turn_order=%s target_indexes=%s "
                        "target_languages=%s presentation_revision=%s "
                        "previous_revision=%s location=%s dropped_pages=%s "
                        "pruned_messages=%s pruned_pages=%s"
                    ),
                    args=(
                        event,
                        fields.get("utterance_id"),
                        fields.get("turn_generation"),
                        fields.get("turn_order"),
                        fields.get("target_indexes"),
                        fields.get("target_languages"),
                        fields.get("presentation_revision"),
                        fields.get("previous_revision"),
                        fields.get("location"),
                        fields.get("dropped_pages"),
                        fields.get("pruned_messages"),
                        fields.get("pruned_pages"),
                    ),
                    detailed=True,
                )
            )
        raw_utterance_id = fields.get("utterance_id")
        if event in {"page_send", "message_terminal"} and isinstance(raw_utterance_id, str):
            try:
                utterance_id = UUID(raw_utterance_id)
            except ValueError:
                pass
            else:
                if event == "page_send":
                    self.record_output_latency_stage(
                        LatencyStageDiagnostic(
                            channel="self",
                            utterance_id=utterance_id,
                            stage="self_chatbox_send",
                            overwrite=False,
                        )
                    )
                else:
                    self.abandon_pending_latency_output("self", utterance_id)
        recorder = self.overlay_diagnostics
        if recorder is None:
            return
        recorder.record_chatbox(event, **fields)

    def record_translation_wait(self, event: str, fields: dict[str, object]) -> None:
        recorder = self.overlay_diagnostics
        if recorder is None:
            return
        recorder.record_translation(event, **fields)

    def record_stt_ingress(self, event: str, **fields: object) -> None:
        recorder = self.overlay_diagnostics
        if recorder is None:
            return
        recorder.record_stt(event, **fields)

    def record_latency_stage(self, diagnostic: LatencyStageDiagnostic) -> None:
        timeline = self._get_timeline(
            diagnostic.channel,
            diagnostic.utterance_id,
            create=True,
        )
        assert timeline is not None
        if diagnostic.parent_utterance_id is not None:
            timeline.parent_utterance_id = diagnostic.parent_utterance_id
        if diagnostic.target_index is not None:
            timeline.target_index = diagnostic.target_index
        if diagnostic.target_language is not None:
            timeline.target_language = diagnostic.target_language
        if diagnostic.turn_generation is not None:
            timeline.turn_generation = diagnostic.turn_generation
        if diagnostic.turn_order is not None:
            timeline.turn_order = diagnostic.turn_order
        if not diagnostic.overwrite and diagnostic.stage in timeline.stage_times:
            return
        timeline.stage_times[diagnostic.stage] = (
            self.clock.now() if diagnostic.timestamp is None else diagnostic.timestamp
        )
        if diagnostic.stage == "speech_end":
            timeline.awaiting_speech_end = False
        if diagnostic.publish_now:
            self._emit_latency_contract(
                diagnostic.channel,
                diagnostic.utterance_id,
            )
        if diagnostic.stage == "speech_end":
            for output_id in tuple(
                self._source_outputs.get((diagnostic.channel, diagnostic.utterance_id), ())
            ):
                self._emit_latency_contract(diagnostic.channel, output_id)
            if timeline.clear_requested:
                self.clear_latency_timeline(diagnostic.channel, diagnostic.utterance_id)

    def record_output_latency_stage(self, diagnostic: LatencyStageDiagnostic) -> bool:
        timeline = self._get_timeline(diagnostic.channel, diagnostic.utterance_id)
        if timeline is None:
            return False
        self.record_latency_stage(diagnostic)
        if "last_speech" not in timeline.stage_times and not timeline.pending_sources:
            self.abandon_pending_latency_output(diagnostic.channel, diagnostic.utterance_id)
        return True

    def inherit_latency(self, diagnostic: LatencyInheritanceDiagnostic) -> None:
        channel = diagnostic.channel
        output_id = diagnostic.output_utterance_id
        output_key = (channel, output_id)
        output_timeline = self._get_timeline(channel, output_id)
        for source_id in diagnostic.source_utterance_ids:
            if source_id == output_id:
                continue
            self._resolve_latency_sources(channel, source_id)
            source_timeline = self._get_timeline(channel, source_id)
            if source_timeline is None:
                continue
            if output_timeline is None:
                output_timeline = self._get_timeline(channel, output_id, create=True)
                assert output_timeline is not None
                if self._timelines.get((channel, source_id)) is not source_timeline:
                    self._discard_latency_timeline(channel, output_id)
                    return
            if self._timelines.get(output_key) is not output_timeline:
                return
            self._inherit_stage_times(output_timeline, source_timeline)
            pending_sources = source_timeline.pending_sources
            if not pending_sources and "speech_end" not in source_timeline.stage_times:
                pending_sources = {source_id}
            for pending_id in pending_sources:
                if pending_id == output_id:
                    continue
                source = self._timelines[(channel, pending_id)]
                source.awaiting_speech_end = True
                output_timeline.pending_sources.add(pending_id)
                self._source_outputs.setdefault((channel, pending_id), set()).add(output_id)
        self._emit_latency_contract(channel, output_id)

    @staticmethod
    def _inherit_stage_times(output: _LatencyTimeline, source: _LatencyTimeline) -> None:
        for stage in ("last_speech", "speech_end", "stt_final"):
            source_time = source.stage_times.get(stage)
            if source_time is None:
                continue
            existing_time = output.stage_times.get(stage)
            output.stage_times[stage] = (
                source_time if existing_time is None else max(existing_time, source_time)
            )

    def _resolve_latency_sources(self, channel: ChannelId, utterance_id: UUID) -> None:
        timeline = self._timelines.get((channel, utterance_id))
        if timeline is None:
            return
        for source_id in tuple(timeline.pending_sources):
            source = self._timelines.get((channel, source_id))
            if source is None:
                self._discard_latency_timeline(channel, utterance_id)
                return
            self._inherit_stage_times(timeline, source)
            if "speech_end" in source.stage_times:
                timeline.pending_sources.remove(source_id)
                self._release_latency_source(channel, source_id, utterance_id)

    def _release_latency_source(self, channel: ChannelId, source_id: UUID, output_id: UUID) -> None:
        key = (channel, source_id)
        outputs = self._source_outputs.get(key)
        if outputs is None:
            return
        outputs.discard(output_id)
        if outputs:
            return
        self._source_outputs.pop(key, None)
        source = self._timelines.get(key)
        if (
            source is not None
            and source.clear_requested
            and not source.awaiting_speech_end
            and key not in self._awaiting_output
        ):
            self._discard_latency_timeline(channel, source_id)

    def _discard_latency_timeline(self, channel: ChannelId, utterance_id: UUID) -> None:
        key = (channel, utterance_id)
        timeline = self._timelines.pop(key, None)
        self._awaiting_output.discard(key)
        if timeline is None:
            return
        for source_id in timeline.pending_sources:
            self._release_latency_source(channel, source_id, utterance_id)
        for output_id in self._source_outputs.pop(key, ()):
            self._discard_latency_timeline(channel, output_id)

    def retain_latency_until_output(self, channel: ChannelId, utterance_id: UUID) -> None:
        key = (channel, utterance_id)
        timeline = self._timelines.get(key)
        if timeline is not None:
            self._awaiting_output.add(key)

    def abandon_pending_latency_output(
        self,
        channel: ChannelId,
        utterance_id: UUID,
    ) -> None:
        key = (channel, utterance_id)
        if key not in self._awaiting_output:
            return
        self._awaiting_output.discard(key)
        self.clear_latency_timeline(channel, utterance_id)

    def clear_latency_timeline(self, channel: ChannelId, utterance_id: UUID) -> None:
        key = (channel, utterance_id)
        timeline = self._timelines.get(key)
        if timeline is None:
            return
        timeline.clear_requested = True
        if (
            key in self._awaiting_output
            or self._source_outputs.get(key)
            or timeline.awaiting_speech_end
        ):
            return
        self._discard_latency_timeline(channel, utterance_id)

    def clear_latency_state(self, channel: ChannelId | None = None) -> None:
        if channel is None:
            self._timelines.clear()
            self._awaiting_output.clear()
            self._source_outputs.clear()
            return
        keys = [key for key in self._timelines if key[0] == channel]
        for _, utterance_id in keys:
            self._discard_latency_timeline(channel, utterance_id)

    def publish_latency(self, diagnostic: LatencyTimelineDiagnostic) -> None:
        self._emit_latency_contract(diagnostic.channel, diagnostic.utterance_id)

    def emit_translation_ready(self, diagnostic: TranslationReadyDiagnostic) -> bool:
        logging_port = self.runtime_logging
        if logging_port is None:
            return False
        timeline = self._get_timeline(diagnostic.channel, diagnostic.utterance_id)
        elapsed_ms = None
        if timeline is not None:
            elapsed_ms = self._elapsed_ms(
                timeline.stage_times.get("last_speech"),
                timeline.stage_times.get("llm_done"),
            )
        return logging_port.emit_detailed_lazy(
            lambda: format_translation_ready_for_output(
                channel=diagnostic.channel,
                utterance_id=str(diagnostic.utterance_id),
                update_id=diagnostic.update_id,
                origin_wall_clock_ms=diagnostic.origin_wall_clock_ms,
                session_scope=diagnostic.session_scope,
                source_text_hash=diagnostic.source_text_hash,
                source_text_len=diagnostic.source_text_len,
                logical_turn_key=diagnostic.logical_turn_key,
                translation_len=diagnostic.translation_len,
                elapsed_ms=elapsed_ms,
                parent_utterance_id=(
                    str(diagnostic.parent_utterance_id)
                    if diagnostic.parent_utterance_id is not None
                    else None
                ),
                target_index=diagnostic.target_index,
                target_language=diagnostic.target_language,
                turn_generation=diagnostic.turn_generation,
                turn_order=diagnostic.turn_order,
            )
        )

    def record_self_overlay_decision(
        self,
        diagnostic: SelfOverlayDecisionDiagnostic,
    ) -> None:
        signature = (
            diagnostic.merge_id,
            diagnostic.active_text,
            diagnostic.secondary_text,
            diagnostic.source,
            diagnostic.reuse_mode,
            diagnostic.resume_pending,
            diagnostic.resume_confirmed,
        )
        if signature != self._last_overlay_runtime_signature:
            emitted = self.emit(
                RuntimeDiagnostic(
                    message=(
                        "[Translation] active_self_secondary merge_id=%s source=%s "
                        "active_len=%s secondary_len=%s spec_text_len=%s "
                        "spec_translation_len=%s cached_secondary_len=%s "
                        "reuse_mode=%s resume_pending=%s resume_confirmed=%s"
                    ),
                    args=(
                        str(diagnostic.merge_id)[:8],
                        diagnostic.source,
                        diagnostic.active_text_len,
                        diagnostic.secondary_len,
                        diagnostic.spec_text_len,
                        diagnostic.spec_translation_len,
                        diagnostic.cached_secondary_len,
                        diagnostic.reuse_mode,
                        diagnostic.resume_pending,
                        diagnostic.resume_confirmed,
                    ),
                    fallback_level=logging.INFO,
                    detailed=True,
                )
            )
            if emitted:
                self._last_overlay_runtime_signature = signature
        recorder = self.overlay_diagnostics
        if recorder is None or signature == self._last_overlay_diagnostics_signature:
            return
        self._last_overlay_diagnostics_signature = signature
        recorder.record_translation(
            "active_self_secondary",
            merge_id=str(diagnostic.merge_id),
            source=diagnostic.source,
            active_text_len=diagnostic.active_text_len,
            secondary_len=diagnostic.secondary_len,
            spec_text_len=diagnostic.spec_text_len,
            spec_translation_len=diagnostic.spec_translation_len,
            cached_secondary_len=diagnostic.cached_secondary_len,
            reuse_mode=diagnostic.reuse_mode,
            resume_pending=diagnostic.resume_pending,
            resume_confirmed=diagnostic.resume_confirmed,
        )

    def record_overlay_emit(self, diagnostic: OverlayEmitDiagnostic) -> None:
        recorder = self.overlay_diagnostics
        if recorder is None:
            return
        recorder.record_translation(
            "overlay_emit",
            event_kind=diagnostic.event_kind,
            utterance_id=str(diagnostic.utterance_id),
            channel=diagnostic.channel,
            secondary_len=diagnostic.secondary_len,
            sink_type=diagnostic.sink_type,
        )

    def record_overlay_sink_failure(self, error_type: object) -> None:
        self._last_error_source = "overlay_sink"
        self.emit(
            RuntimeDiagnostic(
                message="[Translation] Overlay sink emit failed: %s",
                args=(error_type,),
                level=logging.ERROR,
            )
        )

    def record_overlay_sink_duration(
        self,
        diagnostic: OverlaySinkDurationDiagnostic,
    ) -> bool:
        return self.emit(
            RuntimeDiagnostic(
                message=(
                    "[Detailed][Translation] overlay_sink_emit_duration "
                    "event_type=%s channel=%s utterance_id=%s "
                    "update_id=%s elapsed_ms=%s"
                ),
                args=(
                    diagnostic.event_type,
                    diagnostic.channel,
                    diagnostic.utterance_id,
                    diagnostic.update_id,
                    diagnostic.elapsed_ms,
                ),
                detailed=True,
            )
        )

    def replace_overlay_diagnostics(
        self,
        diagnostics: OverlayDiagnosticsRecorder | None,
        *,
        expected_current: OverlayDiagnosticsRecorder | None = None,
        require_match: bool = False,
    ) -> bool:
        if require_match and self.overlay_diagnostics is not expected_current:
            return False
        self.overlay_diagnostics = diagnostics
        return True

    @staticmethod
    def _stt_failure_context(
        provider: object | None,
        *,
        default_channel: ChannelId,
    ) -> tuple[str, ChannelId]:
        provider_label = "stt"
        channel = default_channel
        if provider is None:
            return provider_label, channel
        if isinstance(provider, str) and provider.strip():
            return provider, channel
        provider_name = getattr(provider, "stt_provider_name", None)
        provider_name_value = getattr(provider_name, "value", None)
        if isinstance(provider_name_value, str) and provider_name_value.strip():
            provider_label = provider_name_value
        elif isinstance(provider_name, str) and provider_name.strip():
            provider_label = provider_name
        provider_channel = getattr(provider, "channel", None)
        if provider_channel in ("self", "peer"):
            channel = cast(ChannelId, provider_channel)
        return provider_label, channel

    @staticmethod
    def _translation_skip_reason(diagnostic: TranslationSkipDiagnostic) -> str:
        if diagnostic.cause is not None:
            return diagnostic.cause
        if not diagnostic.llm_available:
            return "provider_unavailable"
        if not diagnostic.configuration.translation_enabled:
            return "translation_disabled"
        if diagnostic.channel == "peer" and not diagnostic.configuration.peer_translation_enabled:
            return "peer_translation_disabled"
        return "translation_disabled"

    def _get_timeline(
        self,
        channel: ChannelId,
        utterance_id: UUID,
        *,
        create: bool = False,
    ) -> _LatencyTimeline | None:
        key = (channel, utterance_id)
        timeline = self._timelines.get(key)
        if timeline is not None:
            self._timelines.move_to_end(key)
        if timeline is None and create:
            timeline = _LatencyTimeline(channel=channel)
            self._timelines[key] = timeline
            while len(self._timelines) > 4096:
                evicted_channel, evicted_id = next(iter(self._timelines))
                self._discard_latency_timeline(evicted_channel, evicted_id)
        return timeline

    @staticmethod
    def _elapsed_ms(start_at: float | None, end_at: float | None) -> int | None:
        if start_at is None or end_at is None:
            return None
        return max(0, int(round((end_at - start_at) * 1000)))

    def _emit_latency_trace(
        self,
        channel: ChannelId,
        utterance_id: UUID,
        stage: str,
    ) -> None:
        timeline = self._get_timeline(channel, utterance_id)
        if timeline is None or timeline.pending_sources or stage in timeline.emitted_trace_points:
            return
        elapsed_ms = self._elapsed_ms(
            timeline.stage_times.get("last_speech"),
            timeline.stage_times.get(stage),
        )
        if elapsed_ms is None:
            return
        emitted = self.emit(
            RuntimeDiagnostic(
                message=format_detailed_latency_trace(
                    channel=channel,
                    utterance_id=str(utterance_id)[:8],
                    stage=stage,
                    elapsed_ms=elapsed_ms,
                    parent_utterance_id=(
                        str(timeline.parent_utterance_id)
                        if timeline.parent_utterance_id is not None
                        else None
                    ),
                    target_index=timeline.target_index,
                    target_language=timeline.target_language,
                    turn_generation=timeline.turn_generation,
                    turn_order=timeline.turn_order,
                ),
                detailed=True,
            )
        )
        if emitted:
            timeline.emitted_trace_points.add(stage)

    def _emit_latency_summary(
        self,
        channel: ChannelId,
        utterance_id: UUID,
        final_output_stage: str,
        endpoint: str,
    ) -> None:
        timeline = self._get_timeline(channel, utterance_id)
        if timeline is None or timeline.basic_summary_emitted or timeline.pending_sources:
            return
        last_speech_at = timeline.stage_times.get("last_speech")
        final_output_at = timeline.stage_times.get(final_output_stage)
        elapsed_ms = self._elapsed_ms(last_speech_at, final_output_at)
        if elapsed_ms is None:
            return
        if final_output_at < last_speech_at:
            timeline.basic_summary_emitted = True
            self._awaiting_output.discard((channel, utterance_id))
            self.clear_latency_timeline(channel, utterance_id)
            return
        speech_end_at = timeline.stage_times.get("speech_end")
        stt_final_at = timeline.stage_times.get("stt_final")
        stt_reference_at = None
        if speech_end_at is not None and stt_final_at is not None:
            stt_reference_at = max(speech_end_at, stt_final_at)
        self.emit(
            RuntimeDiagnostic(
                message=format_basic_latency_summary(
                    channel=channel,
                    endpoint=endpoint,
                    elapsed_ms=elapsed_ms,
                )
            )
        )
        self.emit(
            RuntimeDiagnostic(
                message=format_detailed_latency_breakdown(
                    channel=channel,
                    endpoint=endpoint,
                    elapsed_ms=elapsed_ms,
                    last_speech_to_speech_end_ms=self._elapsed_ms(
                        last_speech_at,
                        speech_end_at,
                    ),
                    speech_end_to_stt_final_ms=self._elapsed_ms(
                        speech_end_at,
                        stt_final_at,
                    ),
                    stt_final_to_final_output_ms=self._elapsed_ms(
                        stt_reference_at,
                        final_output_at,
                    ),
                ),
                detailed=True,
            )
        )
        self._emit_latency_cause(channel, utterance_id, final_output_stage, endpoint)
        timeline.basic_summary_emitted = True
        key = (channel, utterance_id)
        self._awaiting_output.discard(key)
        self.clear_latency_timeline(channel, utterance_id)

    def _emit_latency_cause(
        self,
        channel: ChannelId,
        utterance_id: UUID,
        final_output_stage: str,
        endpoint: str,
    ) -> None:
        timeline = self._get_timeline(channel, utterance_id)
        if timeline is None or timeline.latency_cause_emitted:
            return
        stages = timeline.stage_times
        last_speech_at = stages.get("last_speech")
        speech_end_at = stages.get("speech_end")
        stt_final_at = stages.get("stt_final")
        llm_request_start_at = stages.get("llm_request_start")
        llm_first_chunk_at = stages.get("llm_first_chunk")
        llm_done_at = stages.get("llm_done")
        final_output_at = stages.get(final_output_stage)
        message = format_latency_cause_metric(
            channel=channel,
            provider="llm" if llm_request_start_at is not None else "stt",
            utterance_id=str(utterance_id)[:8],
            parent_utterance_id=(
                str(timeline.parent_utterance_id)
                if timeline.parent_utterance_id is not None
                else None
            ),
            target_index=timeline.target_index,
            target_language=timeline.target_language,
            turn_generation=timeline.turn_generation,
            turn_order=timeline.turn_order,
            stage_durations_ms={
                "last_speech_to_speech_end": self._elapsed_ms(
                    last_speech_at,
                    speech_end_at,
                ),
                "speech_end_to_stt_final": self._elapsed_ms(
                    speech_end_at,
                    stt_final_at,
                ),
                "stt_final_to_llm_request_start": self._elapsed_ms(
                    stt_final_at,
                    llm_request_start_at,
                ),
                "llm_request_to_first_chunk": self._elapsed_ms(
                    llm_request_start_at,
                    llm_first_chunk_at,
                ),
                "llm_request_to_llm_done": self._elapsed_ms(
                    llm_request_start_at,
                    llm_done_at,
                ),
                f"stt_final_to_{endpoint}": (
                    self._elapsed_ms(stt_final_at, final_output_at)
                    if llm_request_start_at is None
                    else None
                ),
                f"llm_done_to_{endpoint}": (
                    self._elapsed_ms(llm_done_at, final_output_at)
                    if llm_request_start_at is not None
                    else None
                ),
            },
        )
        if message is None:
            return
        if self.emit(
            RuntimeDiagnostic(
                message=message,
                fallback_level=logging.DEBUG,
                detailed=True,
            )
        ):
            timeline.latency_cause_emitted = True

    def _emit_latency_contract(self, channel: ChannelId, utterance_id: UUID) -> None:
        self._resolve_latency_sources(channel, utterance_id)
        for stage in _LATENCY_TRACE_ORDER:
            self._emit_latency_trace(channel, utterance_id, stage)
        for summary_channel, stage, endpoint in _LATENCY_SUMMARY_OUTPUT_STAGES:
            if channel == summary_channel:
                self._emit_latency_summary(channel, utterance_id, stage, endpoint)

    @staticmethod
    def _format_log_message(message: str, *args: object) -> str:
        return message % args if args else message

    @staticmethod
    def _safe_log_arg(value: object) -> object:
        if isinstance(value, BaseException):
            return type(value).__name__
        return value


__all__ = [
    "ContextApplicationDiagnostic",
    "ContextModeDiagnostic",
    "LatencyInheritanceDiagnostic",
    "LatencyStageDiagnostic",
    "LatencyTimelineDiagnostic",
    "OverlayEmitDiagnostic",
    "OverlaySinkDurationDiagnostic",
    "RuntimeDiagnostic",
    "SelfOverlayDecisionDiagnostic",
    "SttEventLoopFailureDiagnostic",
    "SttTurnFailureDiagnostic",
    "TranslationFailureDiagnostic",
    "TranslationLatencyDiagnosticsOwner",
    "TranslationLatencyDiagnosticsSnapshot",
    "TranslationReadyDiagnostic",
    "TranslationSkipDiagnostic",
]
