from __future__ import annotations

import hashlib
import json

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.openrouter_routing import OpenRouterProviderRouting

_OPENROUTER_FALLBACK_CONNECTIONS = frozenset({"openrouter", "managed", "managed_china"})
_MANAGED_OPENROUTER_CONNECTIONS = frozenset({"managed", "managed_china"})
_MANAGED_GEMMA_MODELS = frozenset({"managed_gemma", "managed_gemma_12b"})


def provider_llm_for_translation(model: str, connection: str) -> str:
    if model in _MANAGED_GEMMA_MODELS:
        return "managed_gemma"
    if model == "local_llm":
        return "local_llm"
    if model == "gemma4_31b_cerebras" or (model == "gemma4_31b" and connection == "cerebras"):
        return "cerebras"
    if model in {"gemini37_flash", "gemini31_flash_lite"}:
        if connection == "openrouter":
            return "openrouter"
        return "gemini"
    if model in {"deepseek_v4_flash", "deepseek_v4_pro"} and connection == "official_byok":
        return "deepseek"
    if model == "qwen35_plus":
        return "qwen"
    return "openrouter"


def build_llm_provider_signature(
    settings: AppSettingsVNext,
    *,
    http_extensions: HttpExtensionRegistry | None = None,
) -> tuple[object, ...]:
    translation = settings.intent.translation
    local_llm = settings.intent.local_llm
    languages = settings.intent.languages
    provider_llm = provider_llm_for_translation(translation.model, translation.connection)
    managed_gemma_selected = translation.model in _MANAGED_GEMMA_MODELS
    primary_uses_openrouter = provider_llm == "openrouter"
    fallback_uses_openrouter = bool(
        not managed_gemma_selected
        and translation.fallback.enabled
        and translation.fallback.connection in _OPENROUTER_FALLBACK_CONNECTIONS
    )
    uses_openrouter = primary_uses_openrouter or fallback_uses_openrouter
    uses_managed_openrouter = bool(
        (primary_uses_openrouter and translation.openrouter_selected_source == "managed")
        or (
            not managed_gemma_selected
            and translation.fallback.enabled
            and translation.fallback.connection in _MANAGED_OPENROUTER_CONNECTIONS
        )
    )
    extension_signature: tuple[object, ...] | None = None
    if translation.model == "custom_http":
        selected_id = translation.http_extension_id
        selected = (
            http_extensions.snapshot.get(selected_id) if http_extensions is not None else None
        )
        extension_signature = (
            selected_id,
            selected.fingerprint if selected is not None else None,
        )
    return (
        translation.model,
        translation.connection,
        translation.http_extension_id,
        extension_signature,
        provider_llm,
        translation.concurrency_limit,
        translation.gemini.llm_model if provider_llm == "gemini" else None,
        translation.openrouter_model if primary_uses_openrouter else None,
        translation.openrouter_routing_mode if uses_openrouter else None,
        (
            translation.openrouter_provider_routing
            if uses_openrouter
            else OpenRouterProviderRouting.DEFAULT.value
        ),
        translation.openrouter_selected_source if primary_uses_openrouter else None,
        translation.openrouter_selection_alias if primary_uses_openrouter else None,
        (
            (False, None, None)
            if managed_gemma_selected
            else (
                translation.fallback.enabled,
                translation.fallback.model,
                translation.fallback.connection,
            )
        ),
        translation.openrouter_broker_base_url if uses_openrouter else None,
        _managed_openrouter_identity_signature(settings) if uses_managed_openrouter else None,
        translation.qwen.llm_model if provider_llm == "qwen" else None,
        translation.qwen.region if provider_llm == "qwen" else None,
        translation.deepseek.llm_model if provider_llm == "deepseek" else None,
        (
            (
                local_llm.backend,
                local_llm.base_url,
                local_llm.model,
                _canonical_json_signature(local_llm.extra_body),
            )
            if provider_llm == "local_llm"
            else None
        ),
        (
            (
                languages.source_language,
                languages.target_language,
                settings.intent.prompts.system_prompt,
                translation.gpu_device_id,
            )
            if managed_gemma_selected
            else None
        ),
    )


def provider_runtime_requires_gpu_restart(
    current_settings: AppSettingsVNext,
    next_settings: AppSettingsVNext,
) -> bool:
    gpu = STTProviderName.LOCAL_QWEN_GPU.value
    return current_settings.intent.stt.gpu_device_id != next_settings.intent.stt.gpu_device_id and (
        current_settings.intent.stt.provider == gpu
        or current_settings.intent.peer_stt.provider == gpu
        or next_settings.intent.stt.provider == gpu
        or next_settings.intent.peer_stt.provider == gpu
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
    settings: AppSettingsVNext,
) -> tuple[object, ...]:
    identity = settings.state.managed_connection
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
    "provider_llm_for_translation",
    "provider_runtime_requires_gpu_restart",
]
