from __future__ import annotations

import hashlib
import json

from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterProviderRouting,
    STTProviderName,
    TranslationConnection,
    TranslationModel,
)
from puripuly_heart.core.http_extensions import HttpExtensionRegistry


def build_llm_provider_signature(
    settings: AppSettings,
    *,
    http_extensions: HttpExtensionRegistry | None = None,
) -> tuple[object, ...]:
    managed_gemma_selected = settings.translation.model == TranslationModel.MANAGED_GEMMA
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
            )
            if managed_gemma_selected
            else None
        ),
    )


def provider_runtime_requires_gpu_restart(
    current_settings: object,
    next_settings: object,
) -> bool:
    if not isinstance(current_settings, AppSettings) or not isinstance(
        next_settings,
        AppSettings,
    ):
        return False
    return current_settings.stt.gpu_device_id != next_settings.stt.gpu_device_id and (
        current_settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
        or current_settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
        or next_settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
        or next_settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
    )


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
    settings: AppSettings,
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
    "build_llm_provider_signature",
    "provider_runtime_requires_gpu_restart",
]
