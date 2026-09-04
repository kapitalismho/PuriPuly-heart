from __future__ import annotations

from dataclasses import replace

from puripuly_heart.app.ports.translation_runtime_configuration import (
    TranslationRuntimeSettingsValues,
)
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigChange,
    TranslationRuntimeConfigurationPort,
)


def build_translation_runtime_config(
    settings: TranslationRuntimeSettingsValues,
    *,
    current: TranslationRuntimeConfig | None = None,
    fallback_transcript_only: bool | None = None,
    translation_enabled: bool | None = None,
    peer_translation_enabled: bool | None = None,
    integrated_context_enabled: bool | None = None,
) -> TranslationRuntimeConfig:
    value = current or TranslationRuntimeConfig()
    return replace(
        value,
        source_language=settings.source_language,
        target_language=settings.target_language,
        self_target_languages=settings.self_target_languages,
        peer_source_language=settings.peer_source_language,
        peer_target_language=settings.peer_target_language,
        peer_source_mode=settings.peer_source_mode,
        system_prompt=settings.system_prompt,
        chatbox_include_source=settings.chatbox_include_source,
        fallback_transcript_only=(
            value.fallback_transcript_only
            if fallback_transcript_only is None
            else fallback_transcript_only
        ),
        translation_enabled=(
            value.translation_enabled if translation_enabled is None else translation_enabled
        ),
        peer_translation_enabled=(
            value.peer_translation_enabled
            if peer_translation_enabled is None
            else peer_translation_enabled
        ),
        integrated_context_enabled=(
            value.integrated_context_enabled
            if integrated_context_enabled is None
            else integrated_context_enabled
        ),
        hangover_s=(settings.hangover_s),
        peer_hangover_s=settings.peer_hangover_s,
        low_latency_mode=settings.low_latency_mode,
        low_latency_merge_gap_ms=settings.low_latency_merge_gap_ms,
        low_latency_spec_retry_max=settings.low_latency_spec_retry_max,
    )


def replace_translation_runtime_settings(
    owner: TranslationRuntimeConfigurationPort,
    settings: TranslationRuntimeSettingsValues,
    *,
    peer_translation_enabled: bool | None = None,
    integrated_context_enabled: bool | None = None,
) -> TranslationRuntimeConfigChange:
    return owner.transform(
        lambda current: build_translation_runtime_config(
            settings,
            current=current,
            peer_translation_enabled=peer_translation_enabled,
            integrated_context_enabled=integrated_context_enabled,
        )
    )


def replace_translation_runtime_effective_flags(
    owner: TranslationRuntimeConfigurationPort,
    *,
    peer_translation_enabled: bool,
    integrated_context_enabled: bool,
) -> TranslationRuntimeConfigChange:
    return owner.transform(
        lambda current: replace(
            current,
            peer_translation_enabled=peer_translation_enabled,
            integrated_context_enabled=integrated_context_enabled,
        )
    )


def replace_translation_runtime_enabled(
    owner: TranslationRuntimeConfigurationPort,
    enabled: bool,
) -> TranslationRuntimeConfigChange:
    return owner.transform(lambda current: replace(current, translation_enabled=bool(enabled)))
