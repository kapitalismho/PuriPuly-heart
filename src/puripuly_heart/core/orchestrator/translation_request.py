from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol, cast
from uuid import UUID

from puripuly_heart.config.prompts import (
    UNSPECIFIED_SOURCE_TEXT_REF,
    render_translation_prompt_template,
    warm_prompt_cache,
)
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.diagnostics import emit_failure
from puripuly_heart.core.language import get_llm_language_name, map_detected_language_for_llm
from puripuly_heart.core.llm.latency import observe_request
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterUserFacingError
from puripuly_heart.core.messages import (
    SEVERITY_ERROR,
    ErrorDiagnostics,
    SafeMessageParam,
    UserErrorReport,
    UserMessageRef,
)
from puripuly_heart.core.orchestrator.channel_runtime import ChannelRuntime, ContextEntry
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
    TranslationRuntimeConfigSnapshotPort,
)
from puripuly_heart.core.orchestrator.context import ContextMode, ContextResolver
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    LatencyStageDiagnostic,
    RuntimeDiagnostic,
    TranslationFailureDiagnostic,
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import TranslationUiMessage
from puripuly_heart.core.orchestrator.translation_turn import (
    TerminalTranslationFailure,
    TranslationOutputSubmission,
    TranslationTurnKind,
    TranslationTurnOutcome,
    TranslationTurnProcessResult,
)
from puripuly_heart.core.translation_backend import (
    LlmTranslationBackend,
    TranslationBackend,
    TranslationBackendRequest,
)
from puripuly_heart.core.translation_policy import TranslationContextPolicy
from puripuly_heart.core.vrchat_scene import SceneSnapshotProvider, VrchatSceneSnapshot
from puripuly_heart.domain.events import UIEventType
from puripuly_heart.domain.models import ChannelId, Translation

_PEER_TURN_CONTEXT_INSTRUCTION = (
    "Current peer turn is reference only. Translate only <input>, returning only "
    "the translation of that segment."
)


def render_translation_system_prompt(
    template: str,
    *,
    source_language: str,
    target_language: str,
    source_name: str | None = None,
    input_channel: ChannelId = "self",
    source_specified: bool = True,
) -> str:
    return render_translation_prompt_template(
        template,
        source_name=source_name or get_llm_language_name(source_language),
        target_name=get_llm_language_name(target_language),
        input_channel=input_channel,
        source_specified=source_specified,
    )


class TranslationProviderGenerationPort(Protocol):
    @property
    def provider(self) -> object | None: ...

    def current_provider_generation(self) -> tuple[object | None, int]: ...

    def is_current_provider_generation(
        self,
        *,
        provider: object,
        generation: int,
    ) -> bool: ...


class TranslationRequestPresentationPort(Protocol):
    def chatbox_is_eligible(self, channel: ChannelId) -> bool: ...

    async def publish_ui(self, message: TranslationUiMessage) -> None: ...


class TranslationRequestPort(Protocol):
    @property
    def provider_available(self) -> bool: ...

    @property
    def provider_generation(self) -> int: ...

    def set_clock(self, clock: Clock) -> None: ...

    def clear_context(self) -> None: ...

    def source_language_for(
        self,
        channel: ChannelId,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> str: ...

    def target_language_for(
        self,
        channel: ChannelId,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> str: ...

    def translation_enabled_for(
        self,
        channel: ChannelId,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> bool: ...

    def get_valid_context(self) -> list[ContextEntry]: ...

    def format_context(self, context: list[ContextEntry]) -> str: ...

    def remember_context(
        self,
        text: str,
        timestamp: float,
        *,
        channel: ChannelId = "self",
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
        source_language: str | None = None,
        target_language: str | None = None,
        origin: str | None = None,
    ) -> None: ...

    def prepare(
        self,
        text: str,
        *,
        channel: ChannelId = "self",
        detected_language: str | None = None,
        target_language: str | None = None,
        context_policy: TranslationContextPolicy = "integrated_preferred",
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
        parent_utterance_id: UUID | None = None,
        target_index: int | None = None,
    ) -> PreparedTranslationRequest: ...

    def admit(
        self,
        requests: tuple[TranslationProcessRequest, ...],
    ) -> Mapping[UUID, PreparedTranslationRequest]: ...

    def capture_terminal_failure(
        self,
        exception: Exception,
        *,
        provider_generation: int,
        utterance_id: UUID,
        text: str,
        channel: ChannelId,
        source_language: str,
        target_language: str,
        configuration: TranslationRuntimeConfig,
    ) -> TerminalTranslationFailure | None: ...

    async def translate(self, request: DirectTranslationRequest) -> Translation: ...

    async def process(
        self,
        request: TranslationProcessRequest,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
        prepared: PreparedTranslationRequest | None = None,
    ) -> TranslationTurnProcessResult: ...

    def admit_peer(
        self,
        requests: tuple[TranslationProcessRequest, ...],
    ) -> Mapping[UUID, PreparedTranslationRequest]: ...


@dataclass(frozen=True, slots=True)
class PreparedTranslationRequest:
    system_prompt: str
    context: str
    requested_at: float
    applied_context_mode: ContextMode
    source_language: str
    target_language: str
    scene_snapshot: VrchatSceneSnapshot = field(default_factory=VrchatSceneSnapshot)
    provider_generation: int | None = None
    context_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DirectTranslationRequest:
    utterance_id: UUID
    text: str
    channel: ChannelId = "self"
    record_latency: bool = True
    detected_language: str | None = None
    target_language: str | None = None
    config_snapshot: TranslationRuntimeConfigSnapshot | None = None
    expected_provider_generation: int | None = None


@dataclass(frozen=True, slots=True)
class TranslationProcessRequest:
    parent_utterance_id: UUID
    utterance_id: UUID
    sequence: int
    text: str
    channel: ChannelId
    source: str
    target_language: str
    context_policy: TranslationContextPolicy
    config_snapshot: TranslationRuntimeConfigSnapshot
    detected_language: str | None = None
    target_index: int = 0
    turn_generation: int | None = None
    turn_order: int | None = None
    publication_generation: int | None = None
    source_order: int | None = None
    turn_kind: TranslationTurnKind | None = None
    parent_output_count: int = 1
    expected_provider_generation: int | None = None
    prestarted: bool = False
    operation_id: str | None = None
    request_id: str | None = None
    terminal_failure: TerminalTranslationFailure | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (self.turn_generation is None) != (self.turn_order is None):
            raise ValueError("turn generation and order must be provided together")
        if self.prestarted and self.expected_provider_generation is None:
            raise ValueError("prestarted translation requires an expected provider generation")
        for name, value in (
            ("turn_generation", self.turn_generation),
            ("turn_order", self.turn_order),
            ("expected_provider_generation", self.expected_provider_generation),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (self.publication_generation is None) != (self.source_order is None):
            raise ValueError("publication generation and source order must be provided together")
        if self.channel == "peer" and self.publication_generation is None:
            raise ValueError("Peer requests require publication identity")
        if self.channel != "peer" and self.publication_generation is not None:
            raise ValueError("publication identity is only valid for Peer requests")
        for name, value, minimum in (
            ("publication_generation", self.publication_generation, 0),
            ("source_order", self.source_order, 1),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError("invalid Peer publication identity")


class StaleProviderCompletion(Exception):
    pass


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


def _scene_participant_count(snapshot: VrchatSceneSnapshot) -> int | None:
    if snapshot.status != "ready":
        return None
    return snapshot.participant_count


@dataclass(slots=True)
class TranslationRequestOwner:
    config_snapshot: TranslationRuntimeConfigSnapshotPort = field(repr=False)
    self_runtime: ChannelRuntime = field(repr=False)
    peer_runtime: ChannelRuntime = field(repr=False)
    context_resolver: ContextResolver = field(repr=False)
    provider_runtime: TranslationProviderGenerationPort = field(repr=False)
    diagnostics: TranslationLatencyDiagnosticsOwner = field(repr=False)
    presentation: TranslationRequestPresentationPort = field(repr=False)
    clock: Clock
    scene_provider: SceneSnapshotProvider | None = None
    settled_failure_sink: Callable[[BaseException], None] | None = None

    def __post_init__(self) -> None:
        warm_prompt_cache()

    def set_clock(self, clock: Clock) -> None:
        self.clock = clock
        self.context_resolver.clock = clock

    def clear_context(self) -> None:
        self.self_runtime.clear_context()
        self.peer_runtime.clear_context()
        self.diagnostics.emit(RuntimeDiagnostic(message="[Translation] Context history cleared"))

    @property
    def provider_available(self) -> bool:
        return self.provider_runtime.provider is not None

    @property
    def provider_generation(self) -> int:
        return self.provider_runtime.current_provider_generation()[1]

    def capture_terminal_failure(
        self,
        exception: Exception,
        *,
        provider_generation: int,
        utterance_id: UUID,
        text: str,
        channel: ChannelId,
        source_language: str,
        target_language: str,
        configuration: TranslationRuntimeConfig,
    ) -> TerminalTranslationFailure | None:
        provider_request = self._capture_provider_request()
        if provider_request is None:
            return None
        backend, generation = provider_request
        if (
            generation == provider_generation
            and isinstance(backend, LlmTranslationBackend)
            and isinstance(backend.provider, SemaphoreLLMProvider)
            and backend.provider.provider_name == "index_translate"
        ):
            return TerminalTranslationFailure(
                exception,
                generation,
                utterance_id,
                text,
                channel,
                source_language,
                target_language,
                configuration,
            )
        return None

    def runtime_for_channel(self, channel: ChannelId) -> ChannelRuntime:
        return self.peer_runtime if channel == "peer" else self.self_runtime

    def source_language_for(
        self,
        channel: ChannelId,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> str:
        configuration = configuration or self.config_snapshot().value
        if channel == "peer" and configuration.peer_source_language:
            return configuration.peer_source_language
        return configuration.source_language

    def target_language_for(
        self,
        channel: ChannelId,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> str:
        configuration = configuration or self.config_snapshot().value
        if channel == "peer" and configuration.peer_target_language:
            return configuration.peer_target_language
        return configuration.target_language

    def translation_enabled_for(
        self,
        channel: ChannelId,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> bool:
        configuration = configuration or self.config_snapshot().value
        if channel == "peer":
            return configuration.translation_enabled and configuration.peer_translation_enabled
        return configuration.translation_enabled

    def get_valid_context(self) -> list[ContextEntry]:
        configuration = self.config_snapshot().value
        return self.context_resolver.get_entries(
            runtime=self.self_runtime,
            source_language=self.source_language_for("self", configuration),
            target_language=self.target_language_for("self", configuration),
            configuration=configuration,
        )

    def format_context(self, context: list[ContextEntry]) -> str:
        return self.context_resolver.format_entries(context)

    def remember_context(
        self,
        text: str,
        timestamp: float,
        *,
        channel: ChannelId = "self",
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
        source_language: str | None = None,
        target_language: str | None = None,
        origin: str | None = None,
    ) -> None:
        runtime = self.runtime_for_channel(channel)
        config_snapshot = config_snapshot or self.config_snapshot()
        configuration = config_snapshot.value
        runtime.remember_context(
            text,
            timestamp=timestamp,
            source_language=source_language or self.source_language_for(channel, configuration),
            target_language=target_language or self.target_language_for(channel, configuration),
            origin=origin,
            max_entries=max(
                configuration.context_max_entries,
                configuration.integrated_context_max_entries,
            ),
        )

    def prepare(
        self,
        text: str,
        *,
        channel: ChannelId = "self",
        detected_language: str | None = None,
        target_language: str | None = None,
        context_policy: TranslationContextPolicy = "integrated_preferred",
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
        parent_utterance_id: UUID | None = None,
        target_index: int | None = None,
    ) -> PreparedTranslationRequest:
        config_snapshot = config_snapshot or self.config_snapshot()
        configuration = config_snapshot.value
        request_source = self._request_source_language(
            channel,
            detected_language=detected_language,
            configuration=configuration,
        )
        if request_source is None:
            raise _UnmappedDetectedLanguage
        source_language, source_name = request_source
        source_specified = source_name != UNSPECIFIED_SOURCE_TEXT_REF
        if context_policy != "integrated_preferred":
            raise ValueError("unsupported translation context policy")
        runtime = self.runtime_for_channel(channel)
        target_language = target_language or self.target_language_for(channel, configuration)
        other_channel: ChannelId = "self" if channel == "peer" else "peer"
        other_runtime = self.runtime_for_channel(other_channel)
        context_texts, context, applied_mode = self.context_resolver.resolve_selected_for_request(
            runtime=runtime,
            other_runtime=other_runtime,
            source_language=source_language,
            target_language=target_language,
            other_source_language=self.source_language_for(other_channel, configuration),
            other_target_language=self.target_language_for(other_channel, configuration),
            configuration=configuration,
        )
        return PreparedTranslationRequest(
            system_prompt=render_translation_system_prompt(
                configuration.system_prompt,
                source_language=source_language,
                target_language=target_language,
                source_name=source_name,
                input_channel=channel,
                source_specified=source_specified,
            ),
            context=context,
            requested_at=self.clock.now(),
            applied_context_mode=applied_mode,
            source_language=source_language,
            target_language=target_language,
            scene_snapshot=self._scene_snapshot(),
            context_texts=context_texts,
        )

    def admit(
        self,
        requests: tuple[TranslationProcessRequest, ...],
    ) -> Mapping[UUID, PreparedTranslationRequest]:
        prepared_requests: list[tuple[TranslationProcessRequest, PreparedTranslationRequest]] = []
        for request in requests:
            configuration = request.config_snapshot.value
            if not self.provider_available or not self.translation_enabled_for(
                request.channel,
                configuration,
            ):
                continue
            try:
                prepared = self.prepare(
                    request.text,
                    channel=request.channel,
                    detected_language=request.detected_language,
                    target_language=request.target_language,
                    context_policy=request.context_policy,
                    config_snapshot=request.config_snapshot,
                    parent_utterance_id=request.parent_utterance_id,
                    target_index=request.target_index,
                )
            except _UnmappedDetectedLanguage:
                continue
            prepared_requests.append((request, prepared))
        for request, prepared in prepared_requests:
            self.remember_context(
                request.text,
                prepared.requested_at,
                channel=request.channel,
                config_snapshot=request.config_snapshot,
                source_language=prepared.source_language,
                target_language=prepared.target_language,
                origin=request.turn_kind,
            )
        return {request.utterance_id: prepared for request, prepared in prepared_requests}

    def admit_peer(
        self,
        requests: tuple[TranslationProcessRequest, ...],
    ) -> Mapping[UUID, PreparedTranslationRequest]:
        if not requests:
            return {}
        first = requests[0]
        if any(
            request.channel != "peer"
            or request.parent_utterance_id != first.parent_utterance_id
            or request.target_language != first.target_language
            or request.config_snapshot != first.config_snapshot
            for request in requests
        ):
            raise ValueError("Peer requests must share one parent, target and configuration")
        provider_request = self._capture_provider_request()
        if provider_request is None:
            return {}
        backend, generation = provider_request
        prepared = self.admit(requests)
        if not prepared:
            return prepared
        parent_prepared = next(iter(prepared.values()))
        reference = ""
        if isinstance(backend, LlmTranslationBackend) and len(requests) > 1:
            reference = "\n".join(
                f"{request.sequence} / language={request.detected_language or 'unknown'}: "
                f"{json.dumps(request.text, ensure_ascii=False)}"
                for request in requests
            )
        result: dict[UUID, PreparedTranslationRequest] = {}
        for request in requests:
            item = prepared.get(request.utterance_id)
            if item is None:
                continue
            context = item.context
            system_prompt = item.system_prompt
            if reference:
                context = (
                    f"{context}\n\n[Current peer turn; target segment={request.sequence}]\n"
                    f"{reference}"
                ).lstrip()
                system_prompt = f"{system_prompt.rstrip()}\n\n{_PEER_TURN_CONTEXT_INSTRUCTION}"
            result[request.utterance_id] = replace(
                item,
                context=context,
                system_prompt=system_prompt,
                requested_at=parent_prepared.requested_at,
                scene_snapshot=parent_prepared.scene_snapshot,
                provider_generation=generation,
            )
        return result

    async def translate(self, request: DirectTranslationRequest) -> Translation:
        config_snapshot = request.config_snapshot or self.config_snapshot()
        provider_request = self._capture_provider_request()
        if provider_request is None:
            raise RuntimeError("translation backend is not configured")
        backend, generation = provider_request
        if (
            request.expected_provider_generation is not None
            and generation != request.expected_provider_generation
        ):
            raise StaleProviderCompletion
        prepared = self.prepare(
            request.text,
            channel=request.channel,
            detected_language=request.detected_language,
            target_language=request.target_language,
            config_snapshot=config_snapshot,
        )
        if request.record_latency:
            self._record_latency(request.channel, request.utterance_id, "llm_request_start")
        with observe_request(
            sink=self.diagnostics.runtime_logging,
            utterance_id=request.utterance_id,
            channel=request.channel,
            kind=request.channel if request.record_latency else "speculative",
            source_language=prepared.source_language,
            target_language=prepared.target_language,
            provider_generation=generation,
            input_chars=len(request.text),
            prompt_chars=len(prepared.system_prompt),
            context_chars=len(prepared.context),
            clock=self.clock.now,
        ) as observation:
            try:
                try:
                    raw_translation = await backend.translate(
                        TranslationBackendRequest(
                            utterance_id=request.utterance_id,
                            text=request.text,
                            system_prompt=prepared.system_prompt,
                            source_language=prepared.source_language,
                            target_language=prepared.target_language,
                            context=prepared.context,
                            scene_participant_count=_scene_participant_count(
                                prepared.scene_snapshot
                            ),
                        )
                    )
                except Exception as exc:
                    self._raise_if_stale_provider_request(backend, generation)
                    if request.record_latency and self.settled_failure_sink is not None:
                        self.settled_failure_sink(exc)
                    raise
                self._raise_if_stale_provider_request(backend, generation)
                if request.record_latency:
                    self._record_latency(request.channel, request.utterance_id, "llm_done")
                return self._normalize_translation(
                    raw_translation,
                    channel=request.channel,
                    text=request.text,
                    source_language=prepared.source_language,
                    target_language=prepared.target_language,
                )
            except StaleProviderCompletion:
                if observation is not None:
                    observation.status = "stale"
                raise

    async def process(
        self,
        request: TranslationProcessRequest,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
        prepared: PreparedTranslationRequest | None = None,
    ) -> TranslationTurnProcessResult:
        configuration = request.config_snapshot.value
        prepared_context_texts = None if prepared is None else prepared.context_texts
        provider_request = self._capture_provider_request()
        if provider_request is None or not self.translation_enabled_for(
            request.channel,
            configuration,
        ):
            return self._result(
                request,
                "source_only",
                "translation_unavailable",
                source_language=request.detected_language,
                context_texts=prepared_context_texts,
            )
        backend, generation = provider_request
        if (
            request.expected_provider_generation is not None
            and generation != request.expected_provider_generation
        ) or (
            prepared is not None
            and prepared.provider_generation is not None
            and generation != prepared.provider_generation
        ):
            return self._result(
                request,
                "failed",
                "stale_provider_completion",
                source_language=request.detected_language,
                context_texts=prepared_context_texts,
                failure_diagnostics=self._record_stale_provider_completion(request),
            )
        request_source = self._request_source_language(
            request.channel,
            detected_language=request.detected_language,
            configuration=configuration,
        )
        if request_source is None:
            outcome: TranslationTurnOutcome = (
                "source_only" if request.channel == "peer" else "failed"
            )
            if outcome == "failed":
                exc = _UnmappedDetectedLanguage()
                report = self._record_failure(request, exc)
                if not request.prestarted:
                    await self._publish_failure(request, report)
            return self._result(
                request,
                outcome,
                "unsupported_source_language",
                source_language=request.detected_language,
                context_texts=prepared_context_texts,
            )
        source_language, _ = request_source
        terminal_failure = request.terminal_failure
        if terminal_failure is not None and (
            terminal_failure.provider_generation != generation
            or terminal_failure.utterance_id != request.utterance_id
            or terminal_failure.text != request.text
            or terminal_failure.channel != request.channel
            or terminal_failure.source_language != source_language
            or terminal_failure.target_language != request.target_language
            or terminal_failure.configuration != configuration
        ):
            terminal_failure = None
        applied_mode: ContextMode | None = None
        context_texts: tuple[str, ...] | None = None
        try:
            if prepared is None:
                prepared = self.prepare(
                    request.text,
                    channel=request.channel,
                    detected_language=request.detected_language,
                    target_language=request.target_language,
                    context_policy=request.context_policy,
                    config_snapshot=request.config_snapshot,
                    parent_utterance_id=request.parent_utterance_id,
                    target_index=request.target_index,
                )
                self.remember_context(
                    request.text,
                    prepared.requested_at,
                    channel=request.channel,
                    config_snapshot=request.config_snapshot,
                    source_language=source_language,
                    target_language=request.target_language,
                    origin=request.turn_kind,
                )
            elif prepared.target_language != request.target_language:
                raise ValueError("prepared translation target mismatch")
            applied_mode = prepared.applied_context_mode
            context_texts = prepared.context_texts
            self._record_latency(
                request.channel,
                request.utterance_id,
                "llm_request_start",
                parent_utterance_id=request.parent_utterance_id,
                target_index=request.target_index,
                turn_generation=request.turn_generation,
                turn_order=request.turn_order,
                target_language=request.target_language,
            )
            with observe_request(
                sink=self.diagnostics.runtime_logging,
                utterance_id=request.utterance_id,
                parent_utterance_id=request.parent_utterance_id,
                channel=request.channel,
                kind=request.turn_kind or request.channel,
                source_language=source_language,
                target_language=request.target_language,
                provider_generation=generation,
                input_chars=len(request.text),
                prompt_chars=len(prepared.system_prompt),
                context_chars=len(prepared.context),
                clock=self.clock.now,
            ) as observation:
                try:
                    try:
                        if terminal_failure is not None:
                            raise terminal_failure.exception
                        raw_translation = await backend.translate(
                            TranslationBackendRequest(
                                utterance_id=request.utterance_id,
                                text=request.text,
                                system_prompt=prepared.system_prompt,
                                source_language=source_language,
                                target_language=request.target_language,
                                context=prepared.context,
                                scene_participant_count=_scene_participant_count(
                                    prepared.scene_snapshot
                                ),
                            )
                        )
                    except Exception:
                        self._raise_if_stale_provider_request(backend, generation)
                        raise
                    self._raise_if_stale_provider_request(backend, generation)
                    if cancellation_requested is not None and cancellation_requested():
                        raise asyncio.CancelledError
                    translation = self._normalize_translation(
                        raw_translation,
                        channel=request.channel,
                        text=request.text,
                        source_language=source_language,
                        target_language=request.target_language,
                    )
                except StaleProviderCompletion:
                    if observation is not None:
                        observation.status = "stale"
                    raise
            self._record_latency(
                request.channel,
                request.utterance_id,
                "llm_done",
                parent_utterance_id=request.parent_utterance_id,
                target_index=request.target_index,
                turn_generation=request.turn_generation,
                turn_order=request.turn_order,
                target_language=request.target_language,
            )
        except asyncio.CancelledError:
            raise
        except StaleProviderCompletion:
            return self._result(
                request,
                "failed",
                "stale_provider_completion",
                source_language=source_language,
                context_texts=context_texts,
                failure_diagnostics=self._record_stale_provider_completion(request),
            )
        except Exception as exc:
            report = self._record_failure(request, exc)
            terminal_failure = self.capture_terminal_failure(
                exc,
                provider_generation=generation,
                utterance_id=request.utterance_id,
                text=request.text,
                channel=request.channel,
                source_language=source_language,
                target_language=request.target_language,
                configuration=configuration,
            )
            if not request.prestarted:
                await self._publish_failure(request, self._translation_error_payload(exc, report))
            if (
                self.settled_failure_sink is not None
                and (not request.prestarted or terminal_failure is None)
                and self.provider_runtime.is_current_provider_generation(
                    provider=backend, generation=generation
                )
                and not (cancellation_requested is not None and cancellation_requested())
            ):
                self.settled_failure_sink(exc)
            return self._result(
                request,
                "failed",
                "provider_error",
                failure_diagnostics=report.diagnostics,
                source_language=source_language,
                context_texts=context_texts,
                terminal_failure=terminal_failure if request.prestarted else None,
            )
        return TranslationTurnProcessResult(
            "translated",
            TranslationOutputSubmission(
                parent_utterance_id=request.parent_utterance_id,
                child_utterance_id=request.utterance_id,
                sequence=request.sequence,
                channel=request.channel,
                source=request.source,
                source_text=request.text,
                source_language=source_language,
                target_language=request.target_language,
                outcome="translated",
                config_snapshot=request.config_snapshot,
                translation=translation,
                applied_context_mode=applied_mode,
                target_index=request.target_index,
                turn_generation=request.turn_generation,
                turn_order=request.turn_order,
                publication_generation=request.publication_generation,
                source_order=request.source_order,
                turn_kind=request.turn_kind,
                parent_output_count=request.parent_output_count,
                context_texts=context_texts,
                operation_id=request.operation_id,
                request_id=request.request_id,
            ),
        )

    @staticmethod
    def _record_stale_provider_completion(request: TranslationProcessRequest) -> ErrorDiagnostics:
        diagnostics = ErrorDiagnostics(
            component="translation.turn",
            operation="provider_generation",
            code="translation.stale_provider_completion",
            category="lifecycle",
            visibility="basic",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            channel=request.channel,
            turn_id=str(request.utterance_id),
            generation=request.turn_generation,
            operation_id=request.operation_id,
            request_id=request.request_id,
            fields={
                "parent_utterance_id": str(request.parent_utterance_id),
                "target_index": request.target_index,
                "outcome": "failed",
            },
        )
        emit_failure(
            logging.getLogger(__name__),
            diagnostics,
            severity="info",
            disposition="superseded",
            stage="provider_generation",
        )
        return diagnostics

    def _capture_provider_request(self) -> tuple[TranslationBackend, int] | None:
        backend, generation = self.provider_runtime.current_provider_generation()
        if backend is None:
            return None
        return cast(TranslationBackend, backend), generation

    def _scene_snapshot(self) -> VrchatSceneSnapshot:
        provider = self.scene_provider
        if provider is None:
            return VrchatSceneSnapshot()
        return provider.snapshot()

    def _raise_if_stale_provider_request(
        self,
        provider: object,
        generation: int,
    ) -> None:
        if not self.provider_runtime.is_current_provider_generation(
            provider=provider,
            generation=generation,
        ):
            raise StaleProviderCompletion

    def _request_source_language(
        self,
        channel: ChannelId,
        *,
        detected_language: str | None,
        configuration: TranslationRuntimeConfig,
    ) -> tuple[str, str] | None:
        normalized_detected = (detected_language or "").strip() or None
        if normalized_detected is not None:
            detected = map_detected_language_for_llm(normalized_detected)
            if detected is None:
                return None
            return detected.code, detected.name
        if channel == "peer" and configuration.peer_source_mode == "auto":
            return "auto", UNSPECIFIED_SOURCE_TEXT_REF
        source_language = self.source_language_for(channel, configuration)
        return source_language, get_llm_language_name(source_language)

    @staticmethod
    def _normalize_translation(
        translation: Translation,
        *,
        channel: ChannelId,
        text: str,
        source_language: str,
        target_language: str,
    ) -> Translation:
        return Translation(
            utterance_id=translation.utterance_id,
            translated_text=translation.text,
            source_text=text,
            source_language=TranslationRequestOwner._language_or_fallback(
                translation.source_language,
                source_language,
            ),
            target_language=TranslationRequestOwner._language_or_fallback(
                translation.target_language,
                target_language,
            ),
            channel=channel,
            created_at=translation.created_at,
            update_id=translation.update_id,
            origin_wall_clock_ms=translation.origin_wall_clock_ms,
            session_scope=translation.session_scope,
            source_text_hash=translation.source_text_hash,
            source_text_len=translation.source_text_len,
            logical_turn_key=f"{channel}:{translation.utterance_id}",
        )

    @staticmethod
    def _language_or_fallback(language: str | None, fallback: str) -> str:
        if language is not None and language.strip():
            return language
        return fallback

    def _record_latency(
        self,
        channel: ChannelId,
        utterance_id: UUID,
        stage: str,
        *,
        parent_utterance_id: UUID | None = None,
        target_index: int | None = None,
        target_language: str | None = None,
        turn_generation: int | None = None,
        turn_order: int | None = None,
    ) -> None:
        self.diagnostics.record_latency_stage(
            LatencyStageDiagnostic(
                channel=channel,
                utterance_id=utterance_id,
                stage=stage,
                parent_utterance_id=parent_utterance_id,
                target_index=target_index,
                target_language=target_language,
                turn_generation=turn_generation,
                turn_order=turn_order,
            )
        )

    def _record_failure(
        self,
        request: TranslationProcessRequest,
        exc: Exception,
    ) -> UserErrorReport:
        return self.diagnostics.record_translation_failure(
            TranslationFailureDiagnostic(
                stage="final",
                channel=request.channel,
                exception=exc,
                parent_utterance_id=request.parent_utterance_id,
                target_index=request.target_index,
                target_language=request.target_language,
                utterance_id=request.utterance_id,
                turn_generation=request.turn_generation,
                operation_id=request.operation_id,
                request_id=request.request_id,
            )
        )

    async def _publish_failure(
        self,
        request: TranslationProcessRequest,
        payload: UserErrorReport,
    ) -> None:
        await self.presentation.publish_ui(
            TranslationUiMessage(
                event_type=UIEventType.ERROR,
                utterance_id=request.utterance_id,
                payload=payload,
                source=request.source,
                channel=request.channel,
                runtime_log_handled=True,
                publication_generation=request.publication_generation,
                source_order=request.source_order,
                parent_utterance_id=request.parent_utterance_id,
            )
        )

    def _result(
        self,
        request: TranslationProcessRequest,
        outcome: TranslationTurnOutcome,
        failure_code: str,
        *,
        source_language: str | None = None,
        context_texts: tuple[str, ...] | None = None,
        failure_diagnostics: ErrorDiagnostics | None = None,
        terminal_failure: TerminalTranslationFailure | None = None,
    ) -> TranslationTurnProcessResult:
        return TranslationTurnProcessResult(
            outcome,
            TranslationOutputSubmission(
                parent_utterance_id=request.parent_utterance_id,
                child_utterance_id=request.utterance_id,
                sequence=request.sequence,
                channel=request.channel,
                source=request.source,
                source_text=request.text,
                source_language=source_language,
                target_language=request.target_language,
                outcome=outcome,
                config_snapshot=request.config_snapshot,
                failure_code=failure_code,
                target_index=request.target_index,
                turn_generation=request.turn_generation,
                turn_order=request.turn_order,
                publication_generation=request.publication_generation,
                source_order=request.source_order,
                turn_kind=request.turn_kind,
                parent_output_count=request.parent_output_count,
                context_texts=context_texts,
                operation_id=request.operation_id,
                request_id=request.request_id,
                failure_diagnostics=failure_diagnostics,
            ),
            failure_diagnostics=failure_diagnostics,
            terminal_failure=terminal_failure,
        )

    @staticmethod
    def _translation_error_payload(
        exc: Exception,
        report: UserErrorReport,
    ) -> UserErrorReport:
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


__all__ = [
    "DirectTranslationRequest",
    "PreparedTranslationRequest",
    "StaleProviderCompletion",
    "TranslationProcessRequest",
    "TranslationProviderGenerationPort",
    "TranslationRequestOwner",
    "TranslationRequestPort",
    "TranslationRequestPresentationPort",
    "render_translation_system_prompt",
]
