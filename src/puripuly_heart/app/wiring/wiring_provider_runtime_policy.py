from __future__ import annotations

import hashlib
import json
from typing import Protocol

from puripuly_heart.config.provider_values import (
    LLMProviderName,
    OpenRouterCredentialSource,
    STTProviderName,
)
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
)
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.openrouter_routing import OpenRouterProviderRouting


class _TranslationFallbackView(Protocol):
    enabled: bool
    model: TranslationModel
    connection: TranslationConnection


class _TranslationView(Protocol):
    model: TranslationModel
    connection: TranslationConnection
    http_extension_id: str | None
    fallback: _TranslationFallbackView
    gpu_device_id: str


class _ProviderSelectionView(Protocol):
    llm: LLMProviderName
    stt: STTProviderName
    peer_stt: STTProviderName


class _OpenRouterView(Protocol):
    llm_model: object
    routing_mode: object
    provider_routing: OpenRouterProviderRouting
    selected_source: OpenRouterCredentialSource
    selection_alias: object
    broker_base_url: str


class _ManagedIdentityView(Protocol):
    installation_id: str
    release_token: str | None
    release_token_expires_at: str | None
    verified_hardware_hash: str | None
    verified_hardware_hash_salt_version: int | None
    active_managed_credential_ref: str | None
    active_managed_expires_at: str | None
    referral_id: str | None


class _ConcurrencyView(Protocol):
    concurrency_limit: int


class _NamedModelView(Protocol):
    llm_model: object
    region: object


class _LocalLlmView(Protocol):
    backend: object
    base_url: str
    model: str
    extra_body: object


class _LanguageView(Protocol):
    source_language: str
    target_language: str


class _SttDeviceView(Protocol):
    gpu_device_id: str


class LlmProviderSignatureSettings(Protocol):
    translation: _TranslationView
    provider: _ProviderSelectionView
    llm: _ConcurrencyView
    gemini: _NamedModelView
    openrouter: _OpenRouterView
    qwen: _NamedModelView
    deepseek: _NamedModelView
    local_llm: _LocalLlmView
    languages: _LanguageView
    system_prompt: str
    managed_identity: _ManagedIdentityView


class SttGpuRestartSettings(Protocol):
    stt: _SttDeviceView
    provider: _ProviderSelectionView


def build_llm_provider_signature(
    settings: LlmProviderSignatureSettings,
    *,
    http_extensions: HttpExtensionRegistry | None = None,
) -> tuple[object, ...]:
    managed_gemma_selected = settings.translation.model in (
        TranslationModel.MANAGED_GEMMA,
        TranslationModel.MANAGED_GEMMA_12B,
    )
    primary_uses_openrouter = settings.provider.llm == LLMProviderName.OPENROUTER
    fallback_uses_openrouter = bool(
        not managed_gemma_selected
        and settings.translation.fallback.enabled
        and settings.translation.fallback.connection
        in (
            TranslationConnection.OPENROUTER,
            TranslationConnection.MANAGED,
            TranslationConnection.MANAGED_CHINA,
        )
    )
    uses_openrouter = primary_uses_openrouter or fallback_uses_openrouter
    uses_managed_openrouter = bool(
        (
            primary_uses_openrouter
            and settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
        )
        or (
            not managed_gemma_selected
            and settings.translation.fallback.enabled
            and settings.translation.fallback.connection
            in (TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA)
        )
    )
    extension_signature: tuple[object, ...] | None = None
    if settings.translation.model == TranslationModel.CUSTOM_HTTP:
        selected_id = settings.translation.http_extension_id
        selected = (
            http_extensions.snapshot.get(selected_id) if http_extensions is not None else None
        )
        extension_signature = (
            selected_id,
            selected.fingerprint if selected is not None else None,
        )
    return (
        settings.translation.model,
        settings.translation.connection,
        settings.translation.http_extension_id,
        extension_signature,
        settings.provider.llm,
        settings.llm.concurrency_limit,
        settings.gemini.llm_model if settings.provider.llm == LLMProviderName.GEMINI else None,
        settings.openrouter.llm_model if primary_uses_openrouter else None,
        settings.openrouter.routing_mode if uses_openrouter else None,
        (
            settings.openrouter.provider_routing
            if uses_openrouter
            else OpenRouterProviderRouting.DEFAULT
        ),
        settings.openrouter.selected_source if primary_uses_openrouter else None,
        settings.openrouter.selection_alias if primary_uses_openrouter else None,
        (
            (False, None, None)
            if managed_gemma_selected
            else (
                settings.translation.fallback.enabled,
                settings.translation.fallback.model,
                settings.translation.fallback.connection,
            )
        ),
        settings.openrouter.broker_base_url if uses_openrouter else None,
        _managed_openrouter_identity_signature(settings) if uses_managed_openrouter else None,
        settings.qwen.llm_model if settings.provider.llm == LLMProviderName.QWEN else None,
        settings.qwen.region if settings.provider.llm == LLMProviderName.QWEN else None,
        (
            settings.deepseek.llm_model
            if settings.provider.llm == LLMProviderName.DEEPSEEK
            else None
        ),
        (
            (
                settings.local_llm.backend,
                settings.local_llm.base_url,
                settings.local_llm.model,
                _canonical_json_signature(settings.local_llm.extra_body),
            )
            if settings.provider.llm == LLMProviderName.LOCAL_LLM
            else None
        ),
        (
            (
                settings.languages.source_language,
                settings.languages.target_language,
                settings.system_prompt,
                settings.translation.gpu_device_id,
            )
            if managed_gemma_selected
            else None
        ),
    )


def provider_runtime_requires_gpu_restart(
    current_settings: object,
    next_settings: object,
) -> bool:
    current = _as_stt_gpu_restart_settings(current_settings)
    nxt = _as_stt_gpu_restart_settings(next_settings)
    if current is None or nxt is None:
        return False
    return current.stt.gpu_device_id != nxt.stt.gpu_device_id and (
        current.provider.stt == STTProviderName.LOCAL_QWEN_GPU
        or current.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
        or nxt.provider.stt == STTProviderName.LOCAL_QWEN_GPU
        or nxt.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
    )


def _as_stt_gpu_restart_settings(settings: object) -> SttGpuRestartSettings | None:
    try:
        stt = settings.stt
        provider = settings.provider
        _ = (stt.gpu_device_id, provider.stt, provider.peer_stt)
    except AttributeError:
        return None
    return settings  # type: ignore[return-value]


def _canonical_json_signature(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sensitive_optional_text_signature(value: str | None) -> tuple[int, str] | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return (len(normalized), digest)


def _managed_openrouter_identity_signature(
    settings: LlmProviderSignatureSettings,
) -> tuple[object, ...]:
    identity = settings.managed_identity
    return (
        identity.installation_id,
        _sensitive_optional_text_signature(identity.release_token),
        identity.release_token_expires_at,
        identity.verified_hardware_hash,
        identity.verified_hardware_hash_salt_version,
        identity.active_managed_credential_ref,
        identity.active_managed_expires_at,
        identity.referral_id,
    )


__all__ = [
    "LlmProviderSignatureSettings",
    "SttGpuRestartSettings",
    "build_llm_provider_signature",
    "provider_runtime_requires_gpu_restart",
]
