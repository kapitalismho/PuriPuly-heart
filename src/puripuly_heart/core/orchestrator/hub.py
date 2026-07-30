from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field, replace
from typing import Protocol, cast
from uuid import UUID, uuid4

from puripuly_heart.config.prompts import render_translation_prompt_template, warm_prompt_cache
from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.language import (
    DetectedLanguageForLLM,
    get_llm_language_name,
    map_detected_language_for_llm,
)
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.local_asr_provider_runtime import LocalASRProviderRuntimePort
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterUserFacingError,
)
from puripuly_heart.core.messages import (
    SEVERITY_ERROR,
    SafeMessageParam,
    UserErrorReport,
    UserMessageRef,
)
from puripuly_heart.core.orchestrator.channel_runtime import (
    ChannelRuntime,
    ContextEntry,
    _MergeBuffer,
)
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.context import ContextMode, ContextResolver
from puripuly_heart.core.orchestrator.ports import (
    HubChatboxPort,
    HubOverlayEventFactoryPort,
    HubOverlaySinkPort,
)
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    ContextApplicationDiagnostic,
    ContextModeDiagnostic,
    LatencyInheritanceDiagnostic,
    LatencyStageDiagnostic,
    LatencyTimelineDiagnostic,
    OverlayEmitDiagnostic,
    OverlaySinkDurationDiagnostic,
    RuntimeDiagnostic,
    SelfOverlayDecisionDiagnostic,
    SttEventLoopFailureDiagnostic,
    TranslationFailureDiagnostic,
    TranslationLatencyDiagnosticsOwner,
    TranslationReadyDiagnostic,
    TranslationSkipDiagnostic,
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
from puripuly_heart.core.overlay.sink import OverlayEventUnion
from puripuly_heart.core.runtime.output import (
    SELF_SPEECH_TYPING_REASON,
    OutputPublicationResult,
    OutputRuntime,
)
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.translation_policy import TranslationContextPolicy
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadEvent
from puripuly_heart.domain.events import (
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionState,
    STTSessionStateEvent,
    UIErrorPayload,
    UIEvent,
    UIEventType,
)
from puripuly_heart.domain.models import (
    ChannelId,
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
_TRANSLATION_RUNTIME_CONFIG_FIELDS = frozenset(
    {
        "source_language",
        "target_language",
        "peer_source_language",
        "peer_target_language",
        "system_prompt",
        "chatbox_include_source",
        "fallback_transcript_only",
        "translation_enabled",
        "peer_translation_enabled",
        "integrated_context_enabled",
        "hangover_s",
        "peer_hangover_s",
        "context_time_window_s",
        "context_max_entries",
        "integrated_context_time_window_s",
        "integrated_context_max_entries",
        "low_latency_mode",
        "low_latency_merge_gap_ms",
        "low_latency_spec_retry_max",
        "low_latency_finalize_wait_ms",
        "low_latency_awaiting_vad_timeout_s",
    }
)


class _StaleProviderCompletion(Exception):
    """Internal signal for provider calls completed by a replaced provider handle."""


class _UnmappedDetectedLanguage(Exception):
    pass


def _safe_user_message_params(params: Mapping[str, object]) -> dict[str, SafeMessageParam]:
    safe_params: dict[str, SafeMessageParam] = {}
    for key, value in params.items():
        if not isinstance(key, str) or len(key) > 64:
            continue
        if value is None or isinstance(value, str | int | float | bool):
            safe_params[key] = value
    return safe_params


@dataclass(slots=True)
class ClientHub:
    osc: HubChatboxPort
    translation_runtime_configuration: TranslationRuntimeConfigurationOwner
    ui_events: asyncio.Queue[UIEvent]
    direct_output_runtime: InitVar[OutputRuntime]
    direct_self_runtime: InitVar[ChannelRuntime]
    direct_peer_runtime: InitVar[ChannelRuntime]
    direct_translation_turns: InitVar[TranslationTurnLifecycleOwner]
    direct_local_asr_runtime: InitVar[LocalASRProviderRuntimePort]
    direct_llm_runtime: InitVar[ProviderRuntimeHandle]
    direct_context_resolver: InitVar[ContextResolver]
    direct_translation_diagnostics: InitVar[TranslationLatencyDiagnosticsOwner]
    overlay_sink: HubOverlaySinkPort | None = None
    clock: Clock = SystemClock()
    source_language: InitVar[str | None] = None
    target_language: InitVar[str | None] = None
    peer_source_language: InitVar[str | None] = None
    peer_target_language: InitVar[str | None] = None
    system_prompt: InitVar[str | None] = None
    chatbox_include_source: InitVar[bool | None] = None
    fallback_transcript_only: InitVar[bool | None] = None
    translation_enabled: InitVar[bool | None] = None
    peer_translation_enabled: InitVar[bool | None] = None
    integrated_context_enabled: InitVar[bool | None] = None
    hangover_s: InitVar[float | None] = None
    peer_hangover_s: InitVar[float | None] = None
    context_time_window_s: InitVar[float | None] = None
    context_max_entries: InitVar[int | None] = None
    integrated_context_time_window_s: InitVar[float | None] = None
    integrated_context_max_entries: InitVar[int | None] = None
    low_latency_mode: InitVar[bool | None] = None
    low_latency_merge_gap_ms: InitVar[int | None] = None
    low_latency_spec_retry_max: InitVar[int | None] = None
    low_latency_finalize_wait_ms: InitVar[int | None] = None
    low_latency_awaiting_vad_timeout_s: InitVar[float | None] = None

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
    _last_promo_time: float | None = None
    _promo_eligible: bool = False
    _merge_buffer: _MergeBuffer | None = None
    self_runtime: ChannelRuntime = field(init=False)
    peer_runtime: ChannelRuntime = field(init=False)
    translation_turns: TranslationTurnLifecycleOwner = field(init=False)
    peer_final_runs: TranslationTurnLifecycleOwner = field(init=False)
    _peer_turn_parent_ids: dict[UUID, UUID] = field(default_factory=dict)
    _peer_parent_turn_ids: dict[UUID, set[UUID]] = field(default_factory=dict)
    _peer_completed_turn_ids: set[UUID] = field(default_factory=set)
    _peer_parent_speech_end_times: dict[UUID, float] = field(default_factory=dict)
    _peer_translation_parent_ids: set[UUID] = field(default_factory=set)
    context_resolver: ContextResolver = field(init=False)
    active_chatbox_channel: ChannelId = field(init=False, default="self")
    output_runtime: OutputRuntime = field(init=False)
    overlay_event_adapter: HubOverlayEventFactoryPort = field(init=False)
    translation_diagnostics: TranslationLatencyDiagnosticsOwner = field(init=False)
    _local_asr_provider_runtime: LocalASRProviderRuntimePort | None = field(
        init=False,
        default=None,
    )
    _llm_provider_runtime: ProviderRuntimeHandle = field(init=False)

    def __post_init__(
        self,
        direct_output_runtime: OutputRuntime,
        direct_self_runtime: ChannelRuntime,
        direct_peer_runtime: ChannelRuntime,
        direct_translation_turns: TranslationTurnLifecycleOwner,
        direct_local_asr_runtime: LocalASRProviderRuntimePort,
        direct_llm_runtime: ProviderRuntimeHandle,
        direct_context_resolver: ContextResolver,
        direct_translation_diagnostics: TranslationLatencyDiagnosticsOwner,
        source_language: str | None,
        target_language: str | None,
        peer_source_language: str | None,
        peer_target_language: str | None,
        system_prompt: str | None,
        chatbox_include_source: bool | None,
        fallback_transcript_only: bool | None,
        translation_enabled: bool | None,
        peer_translation_enabled: bool | None,
        integrated_context_enabled: bool | None,
        hangover_s: float | None,
        peer_hangover_s: float | None,
        context_time_window_s: float | None,
        context_max_entries: int | None,
        integrated_context_time_window_s: float | None,
        integrated_context_max_entries: int | None,
        low_latency_mode: bool | None,
        low_latency_merge_gap_ms: int | None,
        low_latency_spec_retry_max: int | None,
        low_latency_finalize_wait_ms: int | None,
        low_latency_awaiting_vad_timeout_s: float | None,
    ) -> None:
        config_overrides = {
            name: value
            for name, value in (
                ("source_language", source_language),
                ("target_language", target_language),
                ("peer_source_language", peer_source_language),
                ("peer_target_language", peer_target_language),
                ("system_prompt", system_prompt),
                ("chatbox_include_source", chatbox_include_source),
                ("fallback_transcript_only", fallback_transcript_only),
                ("translation_enabled", translation_enabled),
                ("peer_translation_enabled", peer_translation_enabled),
                ("integrated_context_enabled", integrated_context_enabled),
                ("hangover_s", hangover_s),
                ("peer_hangover_s", peer_hangover_s),
                ("context_time_window_s", context_time_window_s),
                ("context_max_entries", context_max_entries),
                (
                    "integrated_context_time_window_s",
                    integrated_context_time_window_s,
                ),
                ("integrated_context_max_entries", integrated_context_max_entries),
                ("low_latency_mode", low_latency_mode),
                ("low_latency_merge_gap_ms", low_latency_merge_gap_ms),
                ("low_latency_spec_retry_max", low_latency_spec_retry_max),
                ("low_latency_finalize_wait_ms", low_latency_finalize_wait_ms),
                (
                    "low_latency_awaiting_vad_timeout_s",
                    low_latency_awaiting_vad_timeout_s,
                ),
            )
            if value is not None
        }
        config_owner = self.translation_runtime_configuration
        if config_overrides:
            config_owner.replace(replace(config_owner.snapshot().value, **config_overrides))
        self.output_runtime = direct_output_runtime
        assert self.output_runtime.overlay_event_adapter is not None
        self.overlay_event_adapter = self.output_runtime.overlay_event_adapter
        self.self_runtime = direct_self_runtime
        self.self_runtime.alias_target = self
        self._sync_self_runtime_aliases()
        self.peer_runtime = direct_peer_runtime
        self.translation_turns = direct_translation_turns
        self.peer_final_runs = self.translation_turns
        self._local_asr_provider_runtime = direct_local_asr_runtime
        self._llm_provider_runtime = direct_llm_runtime
        self.context_resolver = direct_context_resolver
        self.translation_diagnostics = direct_translation_diagnostics
        warm_prompt_cache()
        self._sync_self_runtime_aliases()

    def __getattribute__(self, name: str) -> object:
        if name in _TRANSLATION_RUNTIME_CONFIG_FIELDS:
            owner = object.__getattribute__(self, "translation_runtime_configuration")
            if owner is None:
                raise RuntimeError("translation runtime configuration is unavailable")
            return getattr(owner.snapshot().value, name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "translation_runtime_configuration":
            try:
                current_owner = object.__getattribute__(
                    self,
                    "translation_runtime_configuration",
                )
            except AttributeError:
                current_owner = None
            if current_owner is not None and value is not current_owner:
                raise RuntimeError("translation runtime configuration owner is fixed")
        if name in _TRANSLATION_RUNTIME_CONFIG_FIELDS:
            owner = object.__getattribute__(self, "translation_runtime_configuration")
            if owner is None:
                raise RuntimeError("translation runtime configuration is unavailable")
            owner.transform(lambda current: replace(current, **{name: value}))
            return
        object.__setattr__(self, name, value)
        if name == "clock":
            try:
                resolver = object.__getattribute__(self, "context_resolver")
            except AttributeError:
                resolver = None
            try:
                overlay_event_adapter = object.__getattribute__(self, "overlay_event_adapter")
            except AttributeError:
                overlay_event_adapter = None
            try:
                output_runtime = object.__getattribute__(self, "output_runtime")
            except AttributeError:
                output_runtime = None
            try:
                translation_diagnostics = object.__getattribute__(
                    self,
                    "translation_diagnostics",
                )
            except AttributeError:
                translation_diagnostics = None
            if resolver is not None:
                resolver.clock = value  # type: ignore[assignment]
            if overlay_event_adapter is not None:
                overlay_event_adapter.clock = value  # type: ignore[assignment]
            if output_runtime is not None:
                output_runtime.clock = value  # type: ignore[assignment]
            if translation_diagnostics is not None:
                translation_diagnostics.clock = value  # type: ignore[assignment]
        if name == "osc":
            try:
                output_runtime = object.__getattribute__(self, "output_runtime")
            except AttributeError:
                output_runtime = None
            if output_runtime is not None:
                output_runtime.chatbox = value  # type: ignore[assignment]
        runtime_field = _SELF_RUNTIME_FIELDS.get(name)
        if runtime_field is None:
            return
        try:
            runtime = object.__getattribute__(self, "self_runtime")
        except AttributeError:
            return
        object.__setattr__(runtime, runtime_field, value)

    def translation_runtime_config_snapshot(self) -> TranslationRuntimeConfigSnapshot:
        owner = self.translation_runtime_configuration
        if owner is None:
            raise RuntimeError("translation runtime configuration is unavailable")
        return owner.snapshot()

    def _sync_self_runtime_aliases(self) -> None:
        self._stt_task = self.self_runtime.stt_task
        self._utterances = self.self_runtime.utterances
        self._translation_tasks = self.self_runtime.translation_tasks
        self._utterance_sources = self.self_runtime.utterance_sources
        self._utterance_start_times = self.self_runtime.utterance_start_times
        self._translation_history = self.self_runtime.translation_history
        self._speech_ended_ids = self.self_runtime.speech_ended_ids
        self._merge_buffer = self.self_runtime.merge_buffer

    def _emit_basic(
        self,
        message: str,
        *args: object,
        level: int = logging.INFO,
        fallback_level: int | None = None,
    ) -> None:
        self.translation_diagnostics.emit(
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
        return self.translation_diagnostics.emit(
            RuntimeDiagnostic(
                message=message,
                args=args,
                level=level,
                fallback_level=fallback_level,
                detailed=True,
            )
        )

    def _emit_metric(self, message: str, *args: object) -> None:
        self.translation_diagnostics.emit_metric(message, *args)

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
        self.translation_diagnostics.record_latency_stage(
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
        self.translation_diagnostics.inherit_latency(
            LatencyInheritanceDiagnostic(
                channel=channel,
                output_utterance_id=output_utterance_id,
                source_utterance_ids=tuple(source_utterance_ids),
            )
        )

    def _clear_latency_timeline(self, *, channel: ChannelId, utterance_id: UUID) -> None:
        self.translation_diagnostics.clear_latency_timeline(channel, utterance_id)

    def _clear_latency_state(self, *, channel: ChannelId | None = None) -> None:
        self.translation_diagnostics.clear_latency_state(channel)

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

    def _emit_exception_summary(
        self,
        message: str,
        *args: object,
        level: int = logging.ERROR,
    ) -> None:
        self.translation_diagnostics.emit(
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
        provider: STTProvider | None = None,
        channel: ChannelId = "self",
    ) -> None:
        self.translation_diagnostics.record_stt_event_loop_failure(
            SttEventLoopFailureDiagnostic(
                exception=exc,
                provider=provider,
                default_channel=channel,
            )
        )

    def _log_translation_skipped(
        self,
        *,
        stage: str,
        runtime: ChannelRuntime,
        publish_chatbox: bool,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> None:
        resolved_configuration = (
            self.translation_runtime_config_snapshot().value
            if configuration is None
            else configuration
        )
        self.translation_diagnostics.record_translation_skip(
            TranslationSkipDiagnostic(
                stage=stage,
                channel=runtime.channel,
                publish_chatbox=publish_chatbox,
                llm_available=self._llm_provider_runtime.provider is not None,
                configuration=resolved_configuration,
            )
        )

    def _log_translation_failure(
        self,
        *,
        stage: str,
        runtime: ChannelRuntime,
        exc: Exception,
        detailed: bool = False,
    ) -> UserErrorReport:
        return self.translation_diagnostics.record_translation_failure(
            TranslationFailureDiagnostic(
                stage=stage,
                channel=runtime.channel,
                exception=exc,
                detailed=detailed,
            )
        )

    @staticmethod
    def _translation_error_payload(exc: Exception, report: UserErrorReport) -> UserErrorReport:
        if not isinstance(exc, ManagedOpenRouterUserFacingError):
            return report
        return UserErrorReport(
            message=UserMessageRef(
                key=exc.message_key,
                params=_safe_user_message_params(exc.message_kwargs),
                severity=SEVERITY_ERROR,
            ),
            diagnostics=report.diagnostics,
        )

    @staticmethod
    def _stt_error_event_payload(event: STTErrorEvent) -> UIErrorPayload | None:
        if isinstance(event.message, UserMessageRef) and event.diagnostics is not None:
            return UserErrorReport(message=event.message, diagnostics=event.diagnostics)
        return event.message

    async def reset_provider_channel(self, channel: ChannelId) -> None:
        await self.translation_turns.cancel_pending(channel=channel)
        runtime = self._runtime_for_channel(channel)
        if channel == "self":
            await self.reset_overlay_preview()
        await runtime.reset_runtime_state()
        if channel == "peer":
            self._clear_peer_logical_turn_state()
        self._clear_latency_state(channel=channel)
        if channel == "self":
            self._sync_self_runtime_aliases()

    def _require_local_asr_provider_runtime(self) -> LocalASRProviderRuntimePort:
        runtime = self._local_asr_provider_runtime
        if runtime is None:
            raise RuntimeError("local ASR provider runtime is not configured")
        return runtime

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
        configuration = self.translation_runtime_config_snapshot().value
        return self.context_resolver.get_local_entries(
            runtime=self.self_runtime,
            source_language=self._source_language_for(
                self.self_runtime,
                configuration,
            ),
            target_language=self._target_language_for(
                self.self_runtime,
                configuration,
            ),
            configuration=configuration,
        )

    def _format_context_for_llm(self, context: list[ContextEntry]) -> str:
        """Format context entries as a string for LLM prompt."""
        return self.context_resolver.format_local(context)

    def _remember_context_entry(
        self,
        text: str,
        timestamp: float,
        *,
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
        runtime: ChannelRuntime | None = None,
        source_language: str | None = None,
    ) -> None:
        runtime = runtime or self.self_runtime
        config_snapshot = config_snapshot or self.translation_runtime_config_snapshot()
        configuration = config_snapshot.value
        runtime.remember_context(
            text,
            timestamp=timestamp,
            source_language=source_language or self._source_language_for(runtime, configuration),
            target_language=self._target_language_for(runtime, configuration),
            max_entries=max(
                configuration.context_max_entries,
                configuration.integrated_context_max_entries,
            ),
        )

    def _log_context_mode_change(
        self,
        *,
        runtime: ChannelRuntime,
        applied_mode: ContextMode,
    ) -> None:
        self.translation_diagnostics.record_context_mode(
            ContextModeDiagnostic(
                channel=runtime.channel,
                applied_mode=applied_mode,
            )
        )

    def _log_context_application(
        self,
        *,
        text: str,
        runtime: ChannelRuntime,
        context: str,
    ) -> None:
        self.translation_diagnostics.record_context_application(
            ContextApplicationDiagnostic(
                channel=runtime.channel,
                request_chars=len(text),
                context_lines=tuple(context.splitlines()) if context else (),
                context_chars=len(context),
            )
        )

    async def handle_vad_event(self, event: VadEvent) -> None:
        resume_overlay_resync_buffer: _MergeBuffer | None = None
        low_latency_mode = self.translation_runtime_config_snapshot().value.low_latency_mode

        if isinstance(event, SpeechStart):
            if low_latency_mode:
                self._mark_resume_pending(event)

        if isinstance(event, SpeechChunk):
            if low_latency_mode:
                resume_overlay_resync_buffer = self._maybe_confirm_resume(event)

        # Record start time for E2E latency tracking (from speech end)
        if isinstance(event, SpeechEnd):
            speech_end_at = self.clock.now()
            self.set_self_chatbox_typing_reason(SELF_SPEECH_TYPING_REASON, True)
            self._utterance_start_times[event.utterance_id] = speech_end_at
            self._speech_ended_ids.add(event.utterance_id)
            self._record_latency_stage(
                channel="self",
                utterance_id=event.utterance_id,
                stage="speech_end",
                timestamp=speech_end_at,
                publish_now=not low_latency_mode,
            )
            if low_latency_mode:
                self._maybe_update_buffer_end_time(event.utterance_id)
                self._maybe_start_finalize_wait(event.utterance_id)
                await self._maybe_clear_resume_on_end(event)

        await self._require_local_asr_provider_runtime().handle_vad_event("self", event)

        if isinstance(event, SpeechEnd):
            await self._require_local_asr_provider_runtime().commit_handoff("self")

        if (
            resume_overlay_resync_buffer is not None
            and self._merge_buffer is resume_overlay_resync_buffer
        ):
            await self._sync_overlay_active_self(resume_overlay_resync_buffer)

    async def handle_peer_vad_event(self, event: VadEvent) -> None:
        if isinstance(event, SpeechEnd) and not self.translation_turns.is_parent_closed(
            event.utterance_id
        ):
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
        await self._require_local_asr_provider_runtime().handle_vad_event("peer", event)
        if isinstance(event, SpeechEnd):
            await self._require_local_asr_provider_runtime().commit_handoff("peer")

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

        await self._ensure_translation(
            transcript,
            turn_kind="manual",
            wait_for_parent=(
                self._llm_provider_runtime.provider is None
                or not self.translation_runtime_config_snapshot().value.translation_enabled
            ),
        )

        return utterance_id

    def _runtime_for_channel(self, channel: ChannelId) -> ChannelRuntime:
        return self.self_runtime if channel == "self" else self.peer_runtime

    async def clear_language_runtime_state(self, *, channel: ChannelId) -> None:
        runtime = self._runtime_for_channel(channel)
        await self.translation_turns.cancel_pending(channel=channel)
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
            self._emit_stt_event_loop_failure(exc, provider=provider)
            raise

    async def _handle_stt_event_loop_exception(
        self,
        exc: Exception,
        *,
        channel: ChannelId = "self",
    ) -> None:
        self._emit_stt_event_loop_failure(exc, channel=channel)

    async def _stop_stt_event_loop(self) -> None:
        return

    async def _stop_stt_task(self, attr_name: str) -> None:
        if attr_name in {"_stt_task", "_peer_stt_task"}:
            return
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
        low_latency_mode = self.translation_runtime_config_snapshot().value.low_latency_mode
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
                    payload=self._stt_error_event_payload(event),
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
            if low_latency_mode:
                return
            self._emit_detailed(
                "[Hub] STT Partial: channel=%s utterance_id=%s text_len=%s",
                event.channel,
                event.transcript.utterance_id,
                len(event.transcript.text),
                fallback_level=logging.DEBUG,
            )
            await self._handle_transcript(event.transcript, is_final=False, source="Mic")
            return

        if isinstance(event, STTFinalEvent):
            runtime = self._runtime_for_channel(event.channel)
            source = "Peer" if runtime.channel == "peer" else "Mic"
            if runtime.channel == "peer":
                await self._ensure_translation(
                    event.transcript,
                    turn_kind="peer",
                    wait_for_parent=(
                        self._llm_provider_runtime.provider is None
                        or not self._translation_enabled_for_runtime(runtime)
                    ),
                )
                return
            if runtime.channel == "self":
                self._send_stt_connected_notification()
            if low_latency_mode and runtime.channel == "self":
                await self._handle_low_latency_final(event.transcript)
                return
            self._record_latency_stage(
                channel=runtime.channel,
                utterance_id=event.transcript.utterance_id,
                stage="stt_final",
            )
            await self._handle_transcript(event.transcript, is_final=True, source=source)
            await self._ensure_translation(
                event.transcript,
                turn_kind="self",
                wait_for_parent=(
                    self._llm_provider_runtime.provider is None
                    or not self._translation_enabled_for_runtime(runtime)
                ),
            )
            return

    async def _handle_retired_stt_event(self, event: object) -> None:
        if isinstance(event, STTFinalEvent):
            await self._handle_stt_event(event)

    def _send_stt_connected_notification(self) -> None:
        """Send promo message when STT connects (only if user clicked button)."""
        if not self._promo_eligible:
            return  # Skip if not triggered by user button click
        self._promo_eligible = False

        now = self.clock.now()
        if self._last_promo_time is not None:
            if now - self._last_promo_time < _PROMO_INTERVAL_SEC:
                return
        result = self.output_runtime.publish_system_immediate_chatbox(text="PuriPuly ON!")
        if result.decision.decision == "published":
            self._last_promo_time = now

    def set_self_chatbox_typing_reason(
        self,
        reason: str,
        active: bool,
    ) -> OutputPublicationResult:
        return self.output_runtime.set_self_chatbox_typing_reason(reason, active)

    def clear_self_chatbox_typing_reasons(self) -> OutputPublicationResult:
        return self.output_runtime.clear_self_chatbox_typing_reasons()

    async def replace_overlay_sink(
        self,
        overlay_sink: HubOverlaySinkPort | None,
        *,
        expected_current: HubOverlaySinkPort | None = None,
        require_match: bool = False,
    ) -> bool:
        replaced = await self.output_runtime.replace_overlay_sink(
            overlay_sink,
            expected_current=expected_current,
            require_match=require_match,
        )
        if replaced:
            object.__setattr__(self, "overlay_sink", overlay_sink)
        return replaced

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
                deny_peer_chatbox_attempt = self.output_runtime.chatbox_is_denied(runtime.channel)
                peer_terminal_work_will_follow = self._peer_terminal_work_will_follow(runtime)
                if self._overlay_translation_will_follow(runtime):
                    await self._ensure_translation(transcript, turn_kind="peer")
                elif self.output_runtime.has_overlay_destination:
                    await self._finalize_peer_source_only(
                        transcript,
                        close_is_final=True,
                        finalize_latency=not peer_terminal_work_will_follow,
                    )
                    if deny_peer_chatbox_attempt:
                        await self._publish_peer_chatbox_candidate(transcript.utterance_id)
                elif deny_peer_chatbox_attempt:
                    await self._publish_peer_chatbox_candidate(transcript.utterance_id)
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

    async def _on_peer_final_run_child_created(self, child: TranslationTurnChild) -> None:
        if child.channel == "peer":
            self._register_peer_logical_turn(
                parent_utterance_id=child.parent_utterance_id,
                peer_turn_id=child.utterance_id,
            )
            await self._handle_peer_final_transcript(
                child.transcript,
                parent_utterance_id=child.parent_utterance_id,
                source=child.source,
            )
        elif child.utterance_id != child.parent_utterance_id:
            await self._handle_transcript(child.transcript, is_final=True, source=child.source)

    async def _process_peer_final_run_child(
        self,
        child: TranslationTurnChild,
        cancellation_requested: Callable[[], bool],
    ) -> TranslationTurnProcessResult:
        runtime = self._runtime_for_channel(child.channel)
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
        result = await self._build_translation_process_result(
            parent_utterance_id=child.parent_utterance_id,
            utterance_id=child.utterance_id,
            sequence=child.sequence,
            text=child.transcript.text,
            runtime=runtime,
            source=child.source,
            target_language=target_language,
            context_policy=child.context_policy,
            detected_language=child.detected_language,
            cancellation_requested=cancellation_requested,
            config_snapshot=config_snapshot,
        )
        if cancellation_requested():
            raise asyncio.CancelledError
        return result

    async def _on_peer_final_run_child_started(
        self,
        child: TranslationTurnChild,
        task: asyncio.Task[TranslationTurnProcessResult],
    ) -> None:
        self._runtime_for_channel(child.channel).translation_tasks[child.utterance_id] = task

    async def _on_peer_final_run_child_terminal(
        self,
        child: TranslationTurnChild,
        outcome: TranslationTurnOutcome,
    ) -> None:
        runtime = self._runtime_for_channel(child.channel)
        runtime.translation_tasks.pop(child.utterance_id, None)
        if outcome == "cancelled" and child.channel == "peer":
            await self._finalize_peer_source_only(
                child.transcript,
                close_is_final=False,
                finalize_latency=True,
                preserve_parent_speech_end_time=True,
            )
            await self._publish_peer_chatbox_candidate(child.utterance_id)
        elif outcome == "cancelled":
            await self._emit_overlay_utterance_closed(
                utterance_id=child.utterance_id,
                channel=child.channel,
                is_final=False,
                finalize_latency=not self.output_runtime.chatbox_is_eligible(child.channel),
            )
        if child.channel == "peer":
            self._complete_peer_logical_turn(
                child.utterance_id,
                preserve_parent_speech_end_time=True,
            )

    async def _on_peer_final_run_parent_closed(self, parent_utterance_id: UUID) -> None:
        if parent_utterance_id in self._peer_translation_parent_ids:
            self._peer_translation_parent_ids.discard(parent_utterance_id)
            self._clear_peer_parent_vad_bookkeeping(parent_utterance_id)

    async def _on_peer_final_run_parent_rejected(self, parent_utterance_id: UUID) -> None:
        if parent_utterance_id in self._peer_translation_parent_ids:
            try:
                await self._publish_peer_chatbox_candidate(parent_utterance_id)
            finally:
                if not self.translation_turns.is_parent_active(parent_utterance_id):
                    self._peer_translation_parent_ids.discard(parent_utterance_id)

    async def _emit_final_transcript_to_overlay(self, transcript: Transcript) -> None:
        if not self.output_runtime.has_overlay_destination:
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
        if self.output_runtime.has_overlay_destination:
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

    async def _emit_overlay_utterance_closed(
        self,
        *,
        utterance_id: UUID,
        channel: ChannelId,
        is_final: bool,
        finalize_latency: bool | None = None,
    ) -> None:
        if not self.output_runtime.has_overlay_destination:
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
            self.output_runtime.has_overlay_destination
            and self._llm_provider_runtime.provider is not None
            and self._translation_enabled_for_runtime(runtime)
        )

    def _peer_terminal_work_will_follow(self, runtime: ChannelRuntime) -> bool:
        if runtime.channel != "peer":
            return False
        return (
            self._llm_provider_runtime.provider is not None
            and self._translation_enabled_for_runtime(runtime)
        ) or self.output_runtime.chatbox_is_denied(runtime.channel)

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
        configuration = self.translation_runtime_config_snapshot().value
        return (
            self._language_or_fallback(
                primary_language,
                configuration.source_language,
            ),
            self._language_or_fallback(
                secondary_language,
                configuration.target_language,
            ),
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

    def _emit_translation_ready_for_output(
        self,
        *,
        translation: Translation,
        runtime: ChannelRuntime,
    ) -> bool:
        return self.translation_diagnostics.emit_translation_ready(
            TranslationReadyDiagnostic(
                channel=runtime.channel,
                utterance_id=translation.utterance_id,
                update_id=translation.update_id,
                origin_wall_clock_ms=translation.origin_wall_clock_ms,
                session_scope=translation.session_scope,
                source_text_hash=translation.source_text_hash,
                source_text_len=translation.source_text_len,
                logical_turn_key=translation.logical_turn_key,
                translation_len=len(translation.text),
            )
        )

    async def _emit_translation_to_overlay(
        self,
        *,
        translation: Translation,
        applied_context_mode: ContextMode | None,
    ) -> None:
        if not self.output_runtime.has_overlay_destination:
            return
        configuration = self.translation_runtime_config_snapshot().value

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
                    configuration.source_language,
                ),
                target_language=self._language_or_fallback(
                    translation.target_language,
                    configuration.target_language,
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
        if not self.output_runtime.has_overlay_destination:
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

    async def _emit_overlay_event(self, event: OverlayEventUnion) -> None:
        if not self.output_runtime.has_overlay_destination:
            return
        detailed_mode = self.translation_diagnostics.detailed_enabled
        start = time.perf_counter() if detailed_mode else 0.0
        result = await self.output_runtime.publish_overlay_event(event)
        if result.decision.reason == "destination_publish_failed":
            self.translation_diagnostics.record_overlay_sink_failure(
                result.decision.metadata.get("error_type", "Exception")
            )
            return
        if detailed_mode and result.decision.decision == "published":
            elapsed_ms = max(0, int((time.perf_counter() - start) * 1000))
            self.translation_diagnostics.record_overlay_sink_duration(
                OverlaySinkDurationDiagnostic(
                    event_type=type(event).__name__,
                    channel=getattr(event, "channel", None),
                    utterance_id=getattr(event, "utterance_id", None),
                    update_id=getattr(event, "update_id", None),
                    elapsed_ms=elapsed_ms,
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
        if not self.output_runtime.has_overlay_destination or buffer is None:
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
        if not self.output_runtime.has_overlay_destination:
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
        spec_translation_len = 0
        if isinstance(buffer.spec_translation, Translation):
            spec_translation_len = len(buffer.spec_translation.text.strip())
        self.translation_diagnostics.record_self_overlay_decision(
            SelfOverlayDecisionDiagnostic.create(
                merge_id=buffer.merge_id,
                source=source,
                active_text=active_text,
                secondary_text=secondary_text,
                spec_text_len=len((buffer.spec_text or "").strip()),
                spec_translation_len=spec_translation_len,
                cached_secondary_len=len(self._cached_active_self_secondary_text().strip()),
                reuse_mode=reuse_mode,
                resume_pending=buffer.resume_pending,
                resume_confirmed=buffer.resume_confirmed,
            )
        )

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
            and self.output_runtime.has_overlay_destination
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
        self.translation_diagnostics.record_overlay_emit(
            OverlayEmitDiagnostic(
                event_kind=event_kind,
                utterance_id=utterance_id,
                channel=channel,
                secondary_len=secondary_len,
                sink_type=(
                    type(self.output_runtime.overlay_sink).__name__
                    if self.output_runtime.has_overlay_destination
                    else None
                ),
            )
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
        self.translation_diagnostics.publish_latency(
            LatencyTimelineDiagnostic(
                channel="self",
                utterance_id=buffer.merge_id,
            )
        )

    def _clear_spec_state(self, buffer: _MergeBuffer, *, reason: str) -> bool:
        had_spec_state = any(
            value is not None
            for value in (
                buffer.spec_task,
                buffer.spec_translation,
                buffer.spec_text,
                buffer.spec_config_snapshot,
                buffer.spec_started_at,
                buffer.spec_done_at,
            )
        ) or bool(buffer.spec_latency_stage_times)
        if not had_spec_state:
            return False
        if (
            buffer.spec_task is not None
            and not buffer.spec_task.done()
            and buffer.spec_task is not asyncio.current_task()
        ):
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
        buffer.spec_config_snapshot = None
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
        timeout_s = (
            self.translation_runtime_config_snapshot().value.low_latency_awaiting_vad_timeout_s
        )
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
        buffer = self._merge_buffer
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
        timeout_s = (
            self.translation_runtime_config_snapshot().value.low_latency_awaiting_vad_timeout_s
        )
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
            timeout_s,
        )
        self._clear_resume_state(buffer)
        self._cancel_finalize_wait(buffer)
        await self._try_commit_after_spec(buffer, reason="resume_end_timeout", allow_fallback=True)

    def _restart_post_end_grace(self, buffer: _MergeBuffer) -> None:
        wait_ms = self.translation_runtime_config_snapshot().value.low_latency_finalize_wait_ms
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
            wait_ms,
        )
        if (
            self._llm_provider_runtime.provider is None
            or not self.translation_runtime_config_snapshot().value.translation_enabled
        ):
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

        if (
            self._llm_provider_runtime.provider is None
            or not self.translation_runtime_config_snapshot().value.translation_enabled
        ):
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

        if (
            buffer.spec_task is not None
            and not buffer.spec_task.done()
            and buffer.spec_task is not asyncio.current_task()
        ):
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
        config_snapshot = (
            buffer.spec_config_snapshot
            if reuse_mode is not None
            and buffer.spec_translation is not None
            and buffer.spec_config_snapshot is not None
            else self.translation_runtime_config_snapshot()
        )

        if (
            self._llm_provider_runtime.provider is None
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
            translation = buffer.spec_translation
            if translation is not None:
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

        if buffer.spec_translation is not None and reuse_mode is None:
            self._clear_spec_latency_state(buffer)
            self._emit_metric(
                "[Metric] spec_cancel id=%s reason=final_mismatch", str(buffer.merge_id)[:8]
            )

        await self._ensure_translation(
            transcript,
            turn_kind="self",
            wait_for_parent=True,
            config_snapshot=config_snapshot,
        )

    async def _maybe_restart_spec(self, buffer: _MergeBuffer) -> None:
        config_snapshot = self.translation_runtime_config_snapshot()
        if (
            self._llm_provider_runtime.provider is None
            or not config_snapshot.value.translation_enabled
        ):
            return

        self._clear_spec_state(buffer, reason="spec_retry")

        merged_text = self._merge_text(buffer.parts)
        if not merged_text:
            return

        buffer.spec_attempts += 1
        buffer.spec_text = merged_text
        buffer.spec_config_snapshot = config_snapshot
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

    async def _run_spec_translation(
        self,
        merge_id: UUID,
        text: str,
        attempt: int,
        *,
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    ) -> None:
        if self._llm_provider_runtime.provider is None:
            return
        buffer = self._merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        if buffer.spec_text != text or buffer.spec_attempts != attempt:
            return
        config_snapshot = (
            config_snapshot
            or buffer.spec_config_snapshot
            or self.translation_runtime_config_snapshot()
        )
        buffer.spec_config_snapshot = config_snapshot
        self._record_spec_latency_stage(buffer, stage="llm_request_start")
        try:
            translation = await self._translate_text(
                merge_id,
                text,
                record_latency=False,
                config_snapshot=config_snapshot,
            )
        except asyncio.CancelledError:
            return
        except _StaleProviderCompletion:
            await self._handle_stale_spec_translation(merge_id, text, attempt)
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

    async def _handle_stale_spec_translation(
        self,
        merge_id: UUID,
        text: str,
        attempt: int,
    ) -> None:
        buffer = self._merge_buffer
        if buffer is None or buffer.merge_id != merge_id:
            return
        if buffer.spec_text != text or buffer.spec_attempts != attempt:
            return
        self._clear_spec_latency_state(buffer)
        buffer.spec_translation = None
        buffer.spec_done_at = self.clock.now()
        await self._try_commit_after_spec(buffer, reason="spec_stale", allow_fallback=True)

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

    def _source_language_for(
        self,
        runtime: ChannelRuntime,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> str:
        configuration = (
            self.translation_runtime_config_snapshot().value
            if configuration is None
            else configuration
        )
        if runtime.channel == "peer" and configuration.peer_source_language:
            return configuration.peer_source_language
        return configuration.source_language

    def _target_language_for(
        self,
        runtime: ChannelRuntime,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> str:
        configuration = (
            self.translation_runtime_config_snapshot().value
            if configuration is None
            else configuration
        )
        if runtime.channel == "peer" and configuration.peer_target_language:
            return configuration.peer_target_language
        return configuration.target_language

    def _format_system_prompt(
        self,
        runtime: ChannelRuntime | None = None,
        *,
        source_name: str | None = None,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> str:
        runtime = runtime or self.self_runtime
        configuration = (
            self.translation_runtime_config_snapshot().value
            if configuration is None
            else configuration
        )
        return render_translation_prompt_template(
            configuration.system_prompt,
            source_name=source_name
            or get_llm_language_name(self._source_language_for(runtime, configuration)),
            target_name=get_llm_language_name(self._target_language_for(runtime, configuration)),
        )

    def _detected_language_for_llm(
        self,
        detected_language: str | None,
    ) -> DetectedLanguageForLLM | None:
        if detected_language is None:
            return None
        return map_detected_language_for_llm(detected_language)

    def _request_source_language(
        self,
        runtime: ChannelRuntime,
        *,
        detected_language: str | None = None,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> tuple[str, str] | None:
        detected = self._detected_language_for_llm(detected_language)
        if detected_language is not None:
            if detected is None:
                return None
            return detected.code, detected.name
        source_language = self._source_language_for(runtime, configuration)
        return source_language, get_llm_language_name(source_language)

    def _other_runtime(self, runtime: ChannelRuntime) -> ChannelRuntime:
        return self.peer_runtime if runtime is self.self_runtime else self.self_runtime

    def _translation_enabled_for_runtime(
        self,
        runtime: ChannelRuntime,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> bool:
        configuration = (
            self.translation_runtime_config_snapshot().value
            if configuration is None
            else configuration
        )
        if runtime.channel == "peer":
            return configuration.translation_enabled and configuration.peer_translation_enabled
        return configuration.translation_enabled

    def _capture_llm_provider_request(self) -> tuple[LLMProvider, int] | None:
        provider, generation = self._llm_provider_runtime.current_provider_generation()
        if provider is None:
            return None
        return cast(LLMProvider, provider), generation

    def _raise_if_stale_llm_provider_request(
        self,
        provider: LLMProvider,
        generation: int,
    ) -> None:
        if not self._llm_provider_runtime.is_current_provider_generation(
            provider=provider,
            generation=generation,
        ):
            raise _StaleProviderCompletion

    def _prepare_llm_request(
        self,
        text: str,
        *,
        runtime: ChannelRuntime | None = None,
        detected_language: str | None = None,
        context_policy: TranslationContextPolicy = "integrated_preferred",
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    ) -> tuple[str, str, float]:
        formatted_prompt, context_str, now, _ = self._prepare_llm_request_with_mode(
            text,
            runtime=runtime,
            detected_language=detected_language,
            context_policy=context_policy,
            config_snapshot=config_snapshot,
        )
        return formatted_prompt, context_str, now

    def _prepare_llm_request_with_mode(
        self,
        text: str,
        *,
        runtime: ChannelRuntime | None = None,
        detected_language: str | None = None,
        context_policy: TranslationContextPolicy = "integrated_preferred",
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    ) -> tuple[str, str, float, ContextMode]:
        _ = text
        runtime = runtime or self.self_runtime
        config_snapshot = config_snapshot or self.translation_runtime_config_snapshot()
        configuration = config_snapshot.value
        request_source = self._request_source_language(
            runtime,
            detected_language=detected_language,
            configuration=configuration,
        )
        if request_source is None:
            raise _UnmappedDetectedLanguage
        source_language, source_name = request_source
        if context_policy != "integrated_preferred":
            raise ValueError("unsupported translation context policy")
        requested_mode: ContextMode = "integrated"
        now = self.clock.now()
        other_runtime = self._other_runtime(runtime)
        context_str, applied_mode = self.context_resolver.resolve_for_request(
            runtime=runtime,
            other_runtime=other_runtime,
            requested_mode=requested_mode,
            peer_translation_enabled=configuration.peer_translation_enabled,
            source_language=source_language,
            target_language=self._target_language_for(runtime, configuration),
            other_source_language=self._source_language_for(
                other_runtime,
                configuration,
            ),
            other_target_language=self._target_language_for(
                other_runtime,
                configuration,
            ),
            configuration=configuration,
        )
        self._log_context_mode_change(runtime=runtime, applied_mode=applied_mode)
        self._log_context_application(text=text, runtime=runtime, context=context_str)
        formatted_prompt = self._format_system_prompt(
            runtime,
            source_name=source_name,
            configuration=configuration,
        )
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
        detected_language: str | None = None,
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    ) -> Translation:
        config_snapshot = config_snapshot or self.translation_runtime_config_snapshot()
        configuration = config_snapshot.value
        llm_request = self._capture_llm_provider_request()
        if llm_request is None:
            raise RuntimeError("LLM is not configured")
        llm, llm_generation = llm_request

        runtime = runtime or self.self_runtime
        formatted_prompt, context_str, _ = self._prepare_llm_request(
            text,
            runtime=runtime,
            detected_language=detected_language,
            config_snapshot=config_snapshot,
        )
        if record_latency:
            self._record_latency_stage(
                channel=runtime.channel,
                utterance_id=utterance_id,
                stage="llm_request_start",
            )
        request_source = self._request_source_language(
            runtime,
            detected_language=detected_language,
            configuration=configuration,
        )
        if request_source is None:
            raise _UnmappedDetectedLanguage
        request_source_language, _ = request_source
        request_target_language = self._target_language_for(runtime, configuration)
        try:
            translation = await llm.translate(
                utterance_id=utterance_id,
                text=text,
                system_prompt=formatted_prompt,
                source_language=request_source_language,
                target_language=request_target_language,
                context=context_str,
            )
        except Exception:
            self._raise_if_stale_llm_provider_request(llm, llm_generation)
            raise
        self._raise_if_stale_llm_provider_request(llm, llm_generation)
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

    async def _ensure_translation(
        self,
        transcript: Transcript,
        *,
        turn_kind: TranslationTurnKind | None = None,
        precomputed_translation: Translation | None = None,
        wait_for_parent: bool = False,
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    ) -> None:
        runtime = self._runtime_for_channel(transcript.channel)
        config_snapshot = config_snapshot or self.translation_runtime_config_snapshot()
        resolved_kind = turn_kind or ("peer" if runtime.channel == "peer" else "self")
        if runtime.channel == "peer":
            self._peer_translation_parent_ids.add(transcript.utterance_id)
        source = self._get_source(transcript.utterance_id, channel=runtime.channel)
        if source is None:
            source = "Peer" if runtime.channel == "peer" else "Mic"
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

    async def _cleanup_dropped_translation(
        self,
        utterance_id: UUID,
        text: str,
        *,
        runtime: ChannelRuntime,
    ) -> None:
        if runtime.channel == "peer":
            await self._publish_peer_chatbox_candidate(utterance_id)
            self._complete_peer_logical_turn(utterance_id)
            return
        await self._emit_overlay_utterance_closed(
            utterance_id=utterance_id,
            channel=runtime.channel,
            is_final=False,
            finalize_latency=True,
        )

    async def _build_translation_process_result(
        self,
        *,
        parent_utterance_id: UUID,
        utterance_id: UUID,
        sequence: int,
        text: str,
        runtime: ChannelRuntime,
        source: str,
        target_language: str,
        context_policy: TranslationContextPolicy,
        detected_language: str | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
        config_snapshot: TranslationRuntimeConfigSnapshot,
    ) -> TranslationTurnProcessResult:
        configuration = config_snapshot.value
        llm_request = self._capture_llm_provider_request()
        if llm_request is None or not self._translation_enabled_for_runtime(
            runtime,
            configuration,
        ):
            self._log_translation_skipped(
                stage="final",
                runtime=runtime,
                publish_chatbox=self.output_runtime.chatbox_is_eligible(runtime.channel),
                configuration=configuration,
            )
            return TranslationTurnProcessResult(
                "source_only",
                TranslationOutputSubmission(
                    parent_utterance_id=parent_utterance_id,
                    child_utterance_id=utterance_id,
                    sequence=sequence,
                    channel=runtime.channel,
                    source=source,
                    source_text=text,
                    source_language=detected_language,
                    target_language=target_language,
                    outcome="source_only",
                    config_snapshot=config_snapshot,
                    failure_code="translation_unavailable",
                ),
            )
        llm, llm_generation = llm_request
        request_source = self._request_source_language(
            runtime,
            detected_language=detected_language,
            configuration=configuration,
        )
        if request_source is None:
            outcome: TranslationTurnOutcome = (
                "source_only" if runtime.channel == "peer" else "failed"
            )
            if outcome == "failed":
                exc = _UnmappedDetectedLanguage()
                report = self._log_translation_failure(stage="final", runtime=runtime, exc=exc)
                await self.ui_events.put(
                    UIEvent(
                        type=UIEventType.ERROR,
                        utterance_id=utterance_id,
                        payload=report,
                        source=source,
                        channel=runtime.channel,
                        runtime_log_handled=True,
                    )
                )
            return TranslationTurnProcessResult(
                outcome,
                TranslationOutputSubmission(
                    parent_utterance_id=parent_utterance_id,
                    child_utterance_id=utterance_id,
                    sequence=sequence,
                    channel=runtime.channel,
                    source=source,
                    source_text=text,
                    source_language=detected_language,
                    target_language=target_language,
                    outcome=outcome,
                    config_snapshot=config_snapshot,
                    failure_code="unsupported_source_language",
                ),
            )

        request_source_language, _ = request_source
        applied_mode: ContextMode | None = None
        try:
            formatted_prompt, context_str, now, applied_mode = self._prepare_llm_request_with_mode(
                text,
                runtime=runtime,
                detected_language=detected_language,
                context_policy=context_policy,
                config_snapshot=config_snapshot,
            )
            self._remember_context_entry(
                text,
                now,
                config_snapshot=config_snapshot,
                runtime=runtime,
                source_language=request_source_language,
            )
            self._record_latency_stage(
                channel=runtime.channel,
                utterance_id=utterance_id,
                stage="llm_request_start",
            )
            try:
                raw_translation = await llm.translate(
                    utterance_id=utterance_id,
                    text=text,
                    system_prompt=formatted_prompt,
                    source_language=request_source_language,
                    target_language=target_language,
                    context=context_str,
                )
            except Exception:
                self._raise_if_stale_llm_provider_request(llm, llm_generation)
                raise
            self._raise_if_stale_llm_provider_request(llm, llm_generation)
            if cancellation_requested is not None and cancellation_requested():
                raise asyncio.CancelledError
            translation = self._normalize_translation(
                raw_translation,
                runtime=runtime,
                text=text,
                source_language=request_source_language,
                target_language=target_language,
            )
            self._record_latency_stage(
                channel=runtime.channel,
                utterance_id=utterance_id,
                stage="llm_done",
            )
        except asyncio.CancelledError:
            raise
        except _StaleProviderCompletion:
            return TranslationTurnProcessResult(
                "failed",
                TranslationOutputSubmission(
                    parent_utterance_id=parent_utterance_id,
                    child_utterance_id=utterance_id,
                    sequence=sequence,
                    channel=runtime.channel,
                    source=source,
                    source_text=text,
                    source_language=request_source_language,
                    target_language=target_language,
                    outcome="failed",
                    config_snapshot=config_snapshot,
                    failure_code="stale_provider_completion",
                ),
            )
        except Exception as exc:
            report = self._log_translation_failure(stage="final", runtime=runtime, exc=exc)
            await self.ui_events.put(
                UIEvent(
                    type=UIEventType.ERROR,
                    utterance_id=utterance_id,
                    payload=self._translation_error_payload(exc, report),
                    source=source,
                    channel=runtime.channel,
                    runtime_log_handled=True,
                )
            )
            return TranslationTurnProcessResult(
                "failed",
                TranslationOutputSubmission(
                    parent_utterance_id=parent_utterance_id,
                    child_utterance_id=utterance_id,
                    sequence=sequence,
                    channel=runtime.channel,
                    source=source,
                    source_text=text,
                    source_language=request_source_language,
                    target_language=target_language,
                    outcome="failed",
                    config_snapshot=config_snapshot,
                    failure_code="provider_error",
                ),
            )

        return TranslationTurnProcessResult(
            "translated",
            TranslationOutputSubmission(
                parent_utterance_id=parent_utterance_id,
                child_utterance_id=utterance_id,
                sequence=sequence,
                channel=runtime.channel,
                source=source,
                source_text=text,
                source_language=request_source_language,
                target_language=target_language,
                outcome="translated",
                config_snapshot=config_snapshot,
                translation=translation,
                applied_context_mode=applied_mode,
            ),
        )

    async def submit_translation_output(self, submission: TranslationOutputSubmission) -> None:
        await self._publish_translation_result(submission)

    async def _publish_translation_result(
        self,
        submission: TranslationOutputSubmission,
    ) -> None:
        runtime = self._runtime_for_channel(submission.channel)
        config_snapshot = submission.config_snapshot
        configuration = config_snapshot.value
        utterance_id = submission.child_utterance_id
        text = submission.source_text
        if submission.outcome == "source_only":
            if runtime.channel == "peer":
                await self._finalize_peer_source_only(
                    Transcript(
                        utterance_id=utterance_id,
                        text=text,
                        is_final=True,
                        created_at=self.clock.now(),
                        channel="peer",
                    ),
                    close_is_final=True,
                    finalize_latency=True,
                    preserve_parent_speech_end_time=True,
                )
                await self._publish_peer_chatbox_candidate(utterance_id)
            elif self.output_runtime.chatbox_is_eligible(runtime.channel):
                await self._publish_chatbox_candidate(
                    utterance_id,
                    transcript_text=text,
                    translation_text=None,
                    config_snapshot=config_snapshot,
                )
            else:
                self._finalize_latency_timeline(
                    channel=runtime.channel,
                    utterance_id=utterance_id,
                )
            return

        if submission.outcome == "failed":
            if submission.failure_code == "stale_provider_completion":
                await self._cleanup_dropped_translation(utterance_id, text, runtime=runtime)
                return
            deny_peer_chatbox_attempt = self.output_runtime.chatbox_is_denied(runtime.channel)
            fallback_to_chatbox = (
                configuration.fallback_transcript_only
                and self.output_runtime.chatbox_is_eligible(runtime.channel)
            )
            denied_fallback_to_chatbox = (
                configuration.fallback_transcript_only and deny_peer_chatbox_attempt
            )
            if runtime.channel == "self":
                await self._emit_overlay_utterance_closed(
                    utterance_id=utterance_id,
                    channel=runtime.channel,
                    is_final=False,
                    finalize_latency=not fallback_to_chatbox,
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
                    finalize_latency=not denied_fallback_to_chatbox,
                    preserve_parent_speech_end_time=True,
                )
            if fallback_to_chatbox:
                await self._publish_chatbox_candidate(
                    utterance_id,
                    transcript_text=text,
                    translation_text=None,
                    config_snapshot=config_snapshot,
                )
            elif deny_peer_chatbox_attempt:
                await self._publish_peer_chatbox_candidate(utterance_id)
            elif runtime.channel != "peer":
                self._finalize_latency_timeline(
                    channel=runtime.channel,
                    utterance_id=utterance_id,
                )
            return

        translation = submission.translation
        if translation is None:
            raise ValueError("translated submission requires a translation")
        publish_to_chatbox = self.output_runtime.chatbox_is_eligible(runtime.channel)
        deny_peer_chatbox_attempt = self.output_runtime.chatbox_is_denied(runtime.channel)
        bundle = self.get_or_create_bundle(utterance_id, channel=runtime.channel)
        bundle.with_translation(translation)
        self._emit_translation_ready_for_output(
            translation=translation,
            runtime=runtime,
        )
        if runtime.channel == "peer" and self.output_runtime.has_overlay_destination:
            await self._emit_peer_translation_to_overlay(
                translation=translation,
                runtime=runtime,
                applied_context_mode=submission.applied_context_mode,
            )
            await self._emit_overlay_utterance_closed(
                utterance_id=utterance_id,
                channel=runtime.channel,
                is_final=True,
                finalize_latency=not (publish_to_chatbox or deny_peer_chatbox_attempt),
            )
        await self.ui_events.put(
            UIEvent(
                type=UIEventType.TRANSLATION_DONE,
                utterance_id=utterance_id,
                payload=translation,
                source=submission.source,
            )
        )
        if runtime.channel == "self":
            await self._emit_translation_to_overlay(
                translation=translation,
                applied_context_mode=submission.applied_context_mode,
            )
            await self._emit_overlay_utterance_closed(
                utterance_id=utterance_id,
                channel=runtime.channel,
                is_final=True,
                finalize_latency=not self.output_runtime.chatbox_is_eligible(runtime.channel),
            )
        if publish_to_chatbox:
            await self._publish_chatbox_candidate(
                utterance_id,
                transcript_text=text,
                translation_text=translation.text,
                config_snapshot=config_snapshot,
            )
        elif deny_peer_chatbox_attempt:
            await self._publish_peer_chatbox_candidate(utterance_id)
        else:
            self._finalize_latency_timeline(
                channel=runtime.channel,
                utterance_id=utterance_id,
            )

    async def _translate_and_enqueue(
        self,
        utterance_id: UUID,
        text: str,
        *,
        runtime: ChannelRuntime | None = None,
        detected_language: str | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> None:
        runtime = runtime or self.self_runtime
        config_snapshot = self.translation_runtime_config_snapshot()
        source = self._get_source(utterance_id, channel=runtime.channel)
        if source is None:
            source = "Peer" if runtime.channel == "peer" else "Mic"
        result = await self._build_translation_process_result(
            parent_utterance_id=utterance_id,
            utterance_id=utterance_id,
            sequence=0,
            text=text,
            runtime=runtime,
            source=source,
            target_language=self._target_language_for(
                runtime,
                config_snapshot.value,
            ),
            context_policy=self.translation_turns.policy.context_policy,
            detected_language=detected_language,
            cancellation_requested=cancellation_requested,
            config_snapshot=config_snapshot,
        )
        if result.output is not None:
            await self.submit_translation_output(result.output)

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
        if (
            self._llm_provider_runtime.provider is None
            or not self._translation_enabled_for_runtime(self.peer_runtime)
        ):
            await self.translation_turns.wait_for_idle()
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
        await self.translation_turns.wait_for_idle()
        return utterance_id

    async def _publish_chatbox_candidate(
        self,
        utterance_id: UUID,
        *,
        transcript_text: str,
        translation_text: str | None,
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    ) -> OutputPublicationResult:
        runtime = self._runtime_for_utterance(utterance_id)
        config_snapshot = config_snapshot or self.translation_runtime_config_snapshot()
        include_source = config_snapshot.value.chatbox_include_source
        result = await self.output_runtime.publish_chatbox(
            publication_id=utterance_id,
            channel=runtime.channel,
            transcript_text=transcript_text,
            translation_text=translation_text,
            include_source=include_source,
        )

        if result.decision.decision != "published":
            self._emit_detailed(
                "[Hub] OSC enqueue skipped: channel=%s route=%s reason=%s",
                runtime.channel,
                result.decision.route,
                result.decision.reason,
                fallback_level=logging.INFO,
            )
            runtime.utterance_start_times.pop(utterance_id, None)
            runtime.speech_ended_ids.discard(utterance_id)
            self._clear_latency_timeline(channel=runtime.channel, utterance_id=utterance_id)
            return result

        msg = result.message
        assert msg is not None
        merged = msg.text

        self._emit_detailed(
            "[Hub] OSC enqueue preview: channel=%s text_len=%s translation_text_present=%s include_source=%s",
            runtime.channel,
            len(merged),
            translation_text is not None,
            include_source,
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
        return result

    async def _publish_peer_chatbox_candidate(
        self,
        utterance_id: UUID,
    ) -> OutputPublicationResult:
        result = await self.output_runtime.publish_chatbox(
            publication_id=utterance_id,
            channel="peer",
            transcript_text="",
            translation_text=None,
            include_source=False,
        )
        self._emit_detailed(
            "[Hub] OSC enqueue skipped: channel=%s route=%s reason=%s",
            "peer",
            result.decision.route,
            result.decision.reason,
            fallback_level=logging.INFO,
        )
        self.peer_runtime.utterance_start_times.pop(utterance_id, None)
        self.peer_runtime.speech_ended_ids.discard(utterance_id)
        self._clear_latency_timeline(channel="peer", utterance_id=utterance_id)
        return result

    def enqueue_peer_translation_disclosure(self, text: str) -> None:
        self._emit_detailed(
            "[Hub] OSC disclosure enqueue: channel=peer text_len=%s",
            len(text),
            fallback_level=logging.INFO,
        )
        self.output_runtime.publish_system_disclosure_chatbox(text=text)
