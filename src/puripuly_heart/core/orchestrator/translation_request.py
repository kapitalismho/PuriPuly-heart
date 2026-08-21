from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, cast
from uuid import UUID

from puripuly_heart.config.prompts import render_translation_prompt_template, warm_prompt_cache
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.language import get_llm_language_name, map_detected_language_for_llm
from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterUserFacingError
from puripuly_heart.core.messages import (
    SEVERITY_ERROR,
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
    ContextApplicationDiagnostic,
    ContextModeDiagnostic,
    LatencyStageDiagnostic,
    RuntimeDiagnostic,
    TranslationFailureDiagnostic,
    TranslationLatencyDiagnosticsOwner,
    TranslationSkipDiagnostic,
)
from puripuly_heart.core.orchestrator.translation_output_projection import TranslationUiMessage
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationOutputSubmission,
    TranslationTurnOutcome,
    TranslationTurnProcessResult,
)
from puripuly_heart.core.translation_backend import (
    TranslationBackend,
    TranslationBackendRequest,
)
from puripuly_heart.core.translation_policy import TranslationContextPolicy
from puripuly_heart.domain.events import UIEventType
from puripuly_heart.domain.models import ChannelId, Translation


def render_translation_system_prompt(
    template: str,
    *,
    source_language: str,
    target_language: str,
    source_name: str | None = None,
) -> str:
    return render_translation_prompt_template(
        template,
        source_name=source_name or get_llm_language_name(source_language),
        target_name=get_llm_language_name(target_language),
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
    ) -> None: ...

    def prepare(
        self,
        text: str,
        *,
        channel: ChannelId = "self",
        detected_language: str | None = None,
        context_policy: TranslationContextPolicy = "integrated_preferred",
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
    ) -> PreparedTranslationRequest: ...

    async def translate(self, request: DirectTranslationRequest) -> Translation: ...

    async def process(
        self,
        request: TranslationProcessRequest,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> TranslationTurnProcessResult: ...


@dataclass(frozen=True, slots=True)
class PreparedTranslationRequest:
    system_prompt: str
    context: str
    requested_at: float
    applied_context_mode: ContextMode
    source_language: str
    target_language: str


@dataclass(frozen=True, slots=True)
class DirectTranslationRequest:
    utterance_id: UUID
    text: str
    channel: ChannelId = "self"
    record_latency: bool = True
    detected_language: str | None = None
    config_snapshot: TranslationRuntimeConfigSnapshot | None = None


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
        return self.context_resolver.get_local_entries(
            runtime=self.self_runtime,
            source_language=self.source_language_for("self", configuration),
            target_language=self.target_language_for("self", configuration),
            configuration=configuration,
        )

    def format_context(self, context: list[ContextEntry]) -> str:
        return self.context_resolver.format_local(context)

    def remember_context(
        self,
        text: str,
        timestamp: float,
        *,
        channel: ChannelId = "self",
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
        source_language: str | None = None,
    ) -> None:
        runtime = self.runtime_for_channel(channel)
        config_snapshot = config_snapshot or self.config_snapshot()
        configuration = config_snapshot.value
        runtime.remember_context(
            text,
            timestamp=timestamp,
            source_language=source_language or self.source_language_for(channel, configuration),
            target_language=self.target_language_for(channel, configuration),
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
        context_policy: TranslationContextPolicy = "integrated_preferred",
        config_snapshot: TranslationRuntimeConfigSnapshot | None = None,
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
        if context_policy != "integrated_preferred":
            raise ValueError("unsupported translation context policy")
        runtime = self.runtime_for_channel(channel)
        other_channel: ChannelId = "self" if channel == "peer" else "peer"
        other_runtime = self.runtime_for_channel(other_channel)
        context, applied_mode = self.context_resolver.resolve_for_request(
            runtime=runtime,
            other_runtime=other_runtime,
            requested_mode="integrated",
            peer_translation_enabled=configuration.peer_translation_enabled,
            source_language=source_language,
            target_language=self.target_language_for(channel, configuration),
            other_source_language=self.source_language_for(other_channel, configuration),
            other_target_language=self.target_language_for(other_channel, configuration),
            configuration=configuration,
        )
        self.diagnostics.record_context_mode(
            ContextModeDiagnostic(channel=channel, applied_mode=applied_mode)
        )
        self.diagnostics.record_context_application(
            ContextApplicationDiagnostic(
                channel=channel,
                request_chars=len(text),
                context_lines=tuple(context.splitlines()) if context else (),
                context_chars=len(context),
            )
        )
        target_language = self.target_language_for(channel, configuration)
        return PreparedTranslationRequest(
            system_prompt=render_translation_system_prompt(
                configuration.system_prompt,
                source_language=source_language,
                target_language=target_language,
                source_name=source_name,
            ),
            context=context,
            requested_at=self.clock.now(),
            applied_context_mode=applied_mode,
            source_language=source_language,
            target_language=target_language,
        )

    async def translate(self, request: DirectTranslationRequest) -> Translation:
        config_snapshot = request.config_snapshot or self.config_snapshot()
        provider_request = self._capture_provider_request()
        if provider_request is None:
            raise RuntimeError("translation backend is not configured")
        backend, generation = provider_request
        prepared = self.prepare(
            request.text,
            channel=request.channel,
            detected_language=request.detected_language,
            config_snapshot=config_snapshot,
        )
        if request.record_latency:
            self._record_latency(request.channel, request.utterance_id, "llm_request_start")
        try:
            raw_translation = await backend.translate(
                TranslationBackendRequest(
                    utterance_id=request.utterance_id,
                    text=request.text,
                    system_prompt=prepared.system_prompt,
                    source_language=prepared.source_language,
                    target_language=prepared.target_language,
                    context=prepared.context,
                )
            )
        except Exception:
            self._raise_if_stale_provider_request(backend, generation)
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

    async def process(
        self,
        request: TranslationProcessRequest,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> TranslationTurnProcessResult:
        configuration = request.config_snapshot.value
        provider_request = self._capture_provider_request()
        if provider_request is None or not self.translation_enabled_for(
            request.channel,
            configuration,
        ):
            self.diagnostics.record_translation_skip(
                TranslationSkipDiagnostic(
                    stage="final",
                    channel=request.channel,
                    publish_chatbox=self.presentation.chatbox_is_eligible(request.channel),
                    llm_available=self.provider_runtime.provider is not None,
                    configuration=configuration,
                )
            )
            return self._result(
                request,
                "source_only",
                "translation_unavailable",
                source_language=request.detected_language,
            )
        backend, generation = provider_request
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
                report = self._record_failure(request.channel, exc)
                await self._publish_failure(request, report)
            return self._result(
                request,
                outcome,
                "unsupported_source_language",
                source_language=request.detected_language,
            )
        source_language, _ = request_source
        applied_mode: ContextMode | None = None
        try:
            prepared = self.prepare(
                request.text,
                channel=request.channel,
                detected_language=request.detected_language,
                context_policy=request.context_policy,
                config_snapshot=request.config_snapshot,
            )
            applied_mode = prepared.applied_context_mode
            self.remember_context(
                request.text,
                prepared.requested_at,
                channel=request.channel,
                config_snapshot=request.config_snapshot,
                source_language=source_language,
            )
            self._record_latency(request.channel, request.utterance_id, "llm_request_start")
            try:
                raw_translation = await backend.translate(
                    TranslationBackendRequest(
                        utterance_id=request.utterance_id,
                        text=request.text,
                        system_prompt=prepared.system_prompt,
                        source_language=source_language,
                        target_language=request.target_language,
                        context=prepared.context,
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
            self._record_latency(request.channel, request.utterance_id, "llm_done")
        except asyncio.CancelledError:
            raise
        except StaleProviderCompletion:
            return self._result(
                request,
                "failed",
                "stale_provider_completion",
                source_language=source_language,
            )
        except Exception as exc:
            report = self._record_failure(request.channel, exc)
            await self._publish_failure(request, self._translation_error_payload(exc, report))
            return self._result(
                request,
                "failed",
                "provider_error",
                source_language=source_language,
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
            ),
        )

    def _capture_provider_request(self) -> tuple[TranslationBackend, int] | None:
        backend, generation = self.provider_runtime.current_provider_generation()
        if backend is None:
            return None
        return cast(TranslationBackend, backend), generation

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
        if detected_language is not None:
            detected = map_detected_language_for_llm(detected_language)
            if detected is None:
                return None
            return detected.code, detected.name
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

    def _record_latency(self, channel: ChannelId, utterance_id: UUID, stage: str) -> None:
        self.diagnostics.record_latency_stage(
            LatencyStageDiagnostic(
                channel=channel,
                utterance_id=utterance_id,
                stage=stage,
            )
        )

    def _record_failure(self, channel: ChannelId, exc: Exception) -> UserErrorReport:
        return self.diagnostics.record_translation_failure(
            TranslationFailureDiagnostic(
                stage="final",
                channel=channel,
                exception=exc,
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
            )
        )

    def _result(
        self,
        request: TranslationProcessRequest,
        outcome: TranslationTurnOutcome,
        failure_code: str,
        *,
        source_language: str | None = None,
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
            ),
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
