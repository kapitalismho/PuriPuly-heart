from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from puripuly_heart.config.llm_profiles import (
    LLM_PROVIDER_GEMINI,
    LLM_PROVIDER_OPENROUTER,
    openrouter_alias_for_fields,
    profile_for_alias,
    resolve_openrouter_fallback_model,
)
from puripuly_heart.config.settings import (
    STT_INTERNAL_SAMPLE_RATE_HZ,
    AppSettings,
    CerebrasLLMModel,
    DeepSeekLLMModel,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterFallbackSelectionAlias,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterSelectionAlias,
    QwenRegion,
    SecretsBackend,
    SecretsSettings,
    STTProviderName,
    TranslationFallbackSelectionAlias,
)
from puripuly_heart.core.llm import FallbackRacingLLMProvider
from puripuly_heart.core.llm.provider import LLMProvider, SemaphoreLLMProvider
from puripuly_heart.core.openrouter_credentials import (
    load_managed_openrouter_user_identifier,
    require_openrouter_execution_api_key,
    resolve_openrouter_credentials,
)
from puripuly_heart.core.runtime_logging import SessionRuntimeLoggingService
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    KeyringSecretStore,
    SecretStore,
)
from puripuly_heart.core.stt.backend import STTBackend
from puripuly_heart.core.stt.custom_vocab import get_effective_custom_terms
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.cerebras import CerebrasLLMProvider
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
from puripuly_heart.providers.llm.local_openai import LocalOpenAICompatibleLLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.llm.qwen import QwenLLMProvider
from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider

SECRETS_PASSPHRASE_ENV = "PURIPULY_HEART_SECRETS_PASSPHRASE"
MANAGED_OPENROUTER_RELEASE_SERVICE_REQUIRED_ERROR = (
    "OpenRouter managed mode requires a managed release service; "
    "CLI/headless paths are not wired for managed OpenRouter mode yet"
)


@dataclass(slots=True)
class _LazyFactoryLLMProvider(LLMProvider):
    factory: Callable[[], LLMProvider]
    _delegate: LLMProvider | None = field(init=False, default=None, repr=False)
    _delegate_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    async def _ensure_delegate(self) -> LLMProvider:
        if self._delegate is not None:
            return self._delegate

        async with self._delegate_lock:
            if self._delegate is None:
                self._delegate = self.factory()
            return self._delegate

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        delegate = await self._ensure_delegate()
        return await delegate.translate(
            utterance_id=utterance_id,
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )

    async def close(self) -> None:
        if self._delegate is not None:
            await self._delegate.close()


def _resolve_primary_openrouter_alias(settings: AppSettings) -> str:
    if settings.openrouter.selection_alias is not None:
        return settings.openrouter.selection_alias.value
    if settings.openrouter.selected_source == OpenRouterCredentialSource.NONE:
        raise ValueError("OpenRouter selected source must not be `none` for execution")
    return openrouter_alias_for_fields(
        model=settings.openrouter.llm_model.value,
        source=settings.openrouter.selected_source.value,
    )


def _settings_for_openrouter_alias(settings: AppSettings, *, alias: str) -> AppSettings:
    profile = profile_for_alias(alias)
    if profile.openrouter_model is None:
        raise ValueError(f"LLM selection alias `{alias}` is not an OpenRouter profile")
    canonical_alias = OpenRouterSelectionAlias(
        openrouter_alias_for_fields(
            model=profile.openrouter_model,
            source=profile.openrouter_source,
        )
    )
    return replace(
        settings,
        openrouter=replace(
            settings.openrouter,
            llm_model=OpenRouterLLMModel(profile.openrouter_model),
            selected_source=OpenRouterCredentialSource(profile.openrouter_source),
            selection_alias=canonical_alias,
        ),
    )


def _settings_for_openrouter_fallback_model(
    settings: AppSettings,
    *,
    fallback_model: str,
    provider_routing: OpenRouterProviderRouting | None = None,
) -> AppSettings:
    resolved_settings = replace(settings)
    resolved_settings.openrouter = replace(settings.openrouter)
    resolved_settings.openrouter.llm_model = OpenRouterLLMModel(fallback_model)
    resolved_settings.openrouter.selection_alias = None
    if provider_routing is not None:
        resolved_settings.openrouter.provider_routing = provider_routing
    return resolved_settings


def _provider_routing_for_openrouter_fallback(
    fallback_selection_alias: OpenRouterFallbackSelectionAlias,
) -> OpenRouterProviderRouting:
    if fallback_selection_alias == OpenRouterFallbackSelectionAlias.DEEPSEEK_V4_FLASH_CHINA:
        return OpenRouterProviderRouting.DEEPSEEK_ONLY
    return OpenRouterProviderRouting.DEFAULT


def _create_llm_provider_from_alias_profile(
    settings: AppSettings,
    *,
    alias: str,
    secrets: SecretStore,
    managed_release_service: object | None,
    managed_delegate_ready: Callable[[], object] | None,
    runtime_logging: SessionRuntimeLoggingService | None,
) -> LLMProvider:
    profile = profile_for_alias(alias)
    if profile.provider == LLM_PROVIDER_GEMINI:
        api_key = require_secret(secrets, key="google_api_key", env_var="GOOGLE_API_KEY")
        return GeminiLLMProvider(
            api_key=api_key,
            model=profile.gemini_model or settings.gemini.llm_model.value,
            runtime_logging=runtime_logging,
        )
    if profile.provider != LLM_PROVIDER_OPENROUTER:
        raise ValueError(f"Unsupported LLM selection alias: {alias}")

    alias_settings = _settings_for_openrouter_alias(settings, alias=alias)
    alias_managed_release_service = _managed_release_service_for_alias(
        managed_release_service,
        alias_settings=alias_settings,
    )
    if (
        alias_settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
        and alias_managed_release_service is None
    ):
        raise ValueError(MANAGED_OPENROUTER_RELEASE_SERVICE_REQUIRED_ERROR)

    resolution = resolve_openrouter_credentials(alias_settings, secrets=secrets)
    if (
        alias_settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
        and resolution.api_key is None
        and alias_managed_release_service is not None
    ):
        from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterLLMProvider

        return ManagedOpenRouterLLMProvider(
            release_service=alias_managed_release_service,
            delegate_factory=lambda api_key: OpenRouterLLMProvider(
                api_key=api_key,
                user_identifier=load_managed_openrouter_user_identifier(
                    alias_settings,
                    secrets=secrets,
                ),
                model=alias_settings.openrouter.llm_model.value,
                routing_mode=settings.openrouter.routing_mode,
                provider_routing=alias_settings.openrouter.provider_routing,
                runtime_logging=runtime_logging,
            ),
            on_delegate_ready=managed_delegate_ready,
        )

    api_key = require_openrouter_execution_api_key(alias_settings, secrets=secrets)
    user_identifier = None
    if alias_settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED:
        user_identifier = load_managed_openrouter_user_identifier(alias_settings, secrets=secrets)
    return OpenRouterLLMProvider(
        api_key=api_key,
        user_identifier=user_identifier,
        model=alias_settings.openrouter.llm_model.value,
        routing_mode=settings.openrouter.routing_mode,
        provider_routing=alias_settings.openrouter.provider_routing,
        runtime_logging=runtime_logging,
    )


def _managed_release_service_for_alias(
    managed_release_service: object | None,
    *,
    alias_settings: AppSettings,
) -> object | None:
    if managed_release_service is None:
        return None

    from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterReleaseService

    if not isinstance(managed_release_service, ManagedOpenRouterReleaseService):
        return managed_release_service

    if (
        managed_release_service.settings.openrouter.selection_alias
        == alias_settings.openrouter.selection_alias
    ):
        return managed_release_service

    return ManagedOpenRouterReleaseService(
        settings=alias_settings,
        secrets=managed_release_service.secrets,
        client=managed_release_service.client,
        persist_settings=lambda _updated: managed_release_service.persist_settings(
            managed_release_service.settings
        ),
        app_version=managed_release_service.app_version,
        raw_hardware_fingerprint_provider=managed_release_service.raw_hardware_fingerprint_provider,
        hardware_hash_provider=managed_release_service._legacy_hardware_hash_provider,
        signed_at_provider=managed_release_service.signed_at_provider,
        monotonic_ms_provider=managed_release_service.monotonic_ms_provider,
    )


def _create_openrouter_fallback_provider(
    *,
    settings: AppSettings,
    secrets: SecretStore,
    managed_release_service: object | None,
    managed_delegate_ready: Callable[[], object] | None,
    runtime_logging: SessionRuntimeLoggingService | None,
) -> LLMProvider:
    fallback_model = resolve_openrouter_fallback_model(
        settings.openrouter.fallback_selection_alias.value
    )
    if fallback_model is None:
        raise ValueError("OpenRouter fallback selection must resolve to a model")

    resolved_settings = _settings_for_openrouter_fallback_model(
        settings,
        fallback_model=fallback_model,
        provider_routing=_provider_routing_for_openrouter_fallback(
            settings.openrouter.fallback_selection_alias
        ),
    )

    if settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED:
        if managed_release_service is None:
            raise ValueError(MANAGED_OPENROUTER_RELEASE_SERVICE_REQUIRED_ERROR)

        from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterLLMProvider

        fallback_managed_release_service = _managed_release_service_for_alias(
            managed_release_service,
            alias_settings=resolved_settings,
        )

        return ManagedOpenRouterLLMProvider(
            release_service=fallback_managed_release_service,
            delegate_factory=lambda api_key: OpenRouterLLMProvider(
                api_key=api_key,
                user_identifier=load_managed_openrouter_user_identifier(
                    resolved_settings,
                    secrets=secrets,
                ),
                model=resolved_settings.openrouter.llm_model.value,
                routing_mode=resolved_settings.openrouter.routing_mode,
                provider_routing=resolved_settings.openrouter.provider_routing,
                runtime_logging=runtime_logging,
            ),
            on_delegate_ready=managed_delegate_ready,
        )

    api_key = require_openrouter_execution_api_key(resolved_settings, secrets=secrets)
    return OpenRouterLLMProvider(
        api_key=api_key,
        model=resolved_settings.openrouter.llm_model.value,
        routing_mode=resolved_settings.openrouter.routing_mode,
        provider_routing=resolved_settings.openrouter.provider_routing,
        runtime_logging=runtime_logging,
    )


def _translation_fallback_openrouter_source(settings: AppSettings) -> OpenRouterCredentialSource:
    if settings.provider.llm == LLMProviderName.OPENROUTER:
        if settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED:
            return OpenRouterCredentialSource.MANAGED
        return OpenRouterCredentialSource.BYOK
    return OpenRouterCredentialSource.BYOK


def _openrouter_settings_for_translation_fallback(
    settings: AppSettings,
    *,
    model: OpenRouterLLMModel,
) -> AppSettings:
    resolved_settings = _settings_for_openrouter_fallback_model(
        settings,
        fallback_model=model.value,
        provider_routing=OpenRouterProviderRouting.DEFAULT,
    )
    resolved_settings.openrouter.selected_source = _translation_fallback_openrouter_source(settings)
    resolved_settings.openrouter.selection_alias = None
    return resolved_settings


def _create_openrouter_translation_fallback_provider(
    *,
    settings: AppSettings,
    secrets: SecretStore,
    model: OpenRouterLLMModel,
    managed_release_service: object | None,
    managed_delegate_ready: Callable[[], object] | None,
    runtime_logging: SessionRuntimeLoggingService | None,
) -> LLMProvider:
    resolved_settings = _openrouter_settings_for_translation_fallback(settings, model=model)
    if resolved_settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED:
        if managed_release_service is None:
            raise ValueError(MANAGED_OPENROUTER_RELEASE_SERVICE_REQUIRED_ERROR)

        from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterLLMProvider

        fallback_managed_release_service = _managed_release_service_for_alias(
            managed_release_service,
            alias_settings=resolved_settings,
        )
        return ManagedOpenRouterLLMProvider(
            release_service=fallback_managed_release_service,
            delegate_factory=lambda api_key: OpenRouterLLMProvider(
                api_key=api_key,
                user_identifier=load_managed_openrouter_user_identifier(
                    resolved_settings,
                    secrets=secrets,
                ),
                model=resolved_settings.openrouter.llm_model.value,
                routing_mode=resolved_settings.openrouter.routing_mode,
                provider_routing=resolved_settings.openrouter.provider_routing,
                runtime_logging=runtime_logging,
            ),
            on_delegate_ready=managed_delegate_ready,
        )

    api_key = require_openrouter_execution_api_key(resolved_settings, secrets=secrets)
    return OpenRouterLLMProvider(
        api_key=api_key,
        model=resolved_settings.openrouter.llm_model.value,
        routing_mode=resolved_settings.openrouter.routing_mode,
        provider_routing=resolved_settings.openrouter.provider_routing,
        runtime_logging=runtime_logging,
    )


def _create_translation_fallback_provider(
    *,
    settings: AppSettings,
    secrets: SecretStore,
    alias: TranslationFallbackSelectionAlias,
    managed_release_service: object | None,
    managed_delegate_ready: Callable[[], object] | None,
    runtime_logging: SessionRuntimeLoggingService | None,
) -> LLMProvider:
    if alias == TranslationFallbackSelectionAlias.DEEPSEEK_V4_FLASH_OFFICIAL:
        return DeepSeekLLMProvider(
            api_key=require_secret(
                secrets,
                key="deepseek_api_key",
                env_var="DEEPSEEK_API_KEY",
            ),
            model=DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value,
            runtime_logging=runtime_logging,
        )
    if alias == TranslationFallbackSelectionAlias.CEREBRAS_GEMMA4_31B:
        return CerebrasLLMProvider(
            api_key=require_secret(
                secrets,
                key="cerebras_api_key",
                env_var="CEREBRAS_API_KEY",
            ),
            model=CerebrasLLMModel.GEMMA_4_31B.value,
            runtime_logging=runtime_logging,
        )
    if alias == TranslationFallbackSelectionAlias.OPENROUTER_DEEPSEEK_V4_FLASH:
        return _create_openrouter_translation_fallback_provider(
            settings=settings,
            secrets=secrets,
            model=OpenRouterLLMModel.DEEPSEEK_V4_FLASH,
            managed_release_service=managed_release_service,
            managed_delegate_ready=managed_delegate_ready,
            runtime_logging=runtime_logging,
        )
    if alias == TranslationFallbackSelectionAlias.OPENROUTER_GEMMA4_26B_A4B:
        return _create_openrouter_translation_fallback_provider(
            settings=settings,
            secrets=secrets,
            model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            managed_release_service=managed_release_service,
            managed_delegate_ready=managed_delegate_ready,
            runtime_logging=runtime_logging,
        )
    raise ValueError(f"Unsupported translation fallback selection: {alias}")


def _translation_fallback_matches_primary(
    settings: AppSettings,
    alias: TranslationFallbackSelectionAlias,
) -> bool:
    if alias == TranslationFallbackSelectionAlias.DEEPSEEK_V4_FLASH_OFFICIAL:
        return (
            settings.provider.llm == LLMProviderName.DEEPSEEK
            and settings.deepseek.llm_model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH
        )
    if alias == TranslationFallbackSelectionAlias.CEREBRAS_GEMMA4_31B:
        return (
            settings.provider.llm == LLMProviderName.CEREBRAS
            and settings.cerebras.llm_model == CerebrasLLMModel.GEMMA_4_31B
        )
    if alias == TranslationFallbackSelectionAlias.OPENROUTER_DEEPSEEK_V4_FLASH:
        return (
            settings.provider.llm == LLMProviderName.OPENROUTER
            and settings.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
        )
    if alias == TranslationFallbackSelectionAlias.OPENROUTER_GEMMA4_26B_A4B:
        return (
            settings.provider.llm == LLMProviderName.OPENROUTER
            and settings.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
        )
    return False


def _emit_translation_fallback_noop(
    *,
    runtime_logging: SessionRuntimeLoggingService | None,
    alias: TranslationFallbackSelectionAlias,
) -> None:
    if runtime_logging is None:
        return
    emit = getattr(runtime_logging, "emit_basic", None)
    if callable(emit):
        with contextlib.suppress(Exception):
            emit(f"[Fallback] Skipped matching fallback target: {alias.value}")


def _shared_managed_release_service_for_fallback(
    primary: LLMProvider,
    managed_release_service: object | None,
) -> object | None:
    from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterLLMProvider

    if isinstance(primary, ManagedOpenRouterLLMProvider):
        return primary.release_service
    return managed_release_service


@dataclass(frozen=True, slots=True)
class ResolvedPeerSTTConfig:
    provider: STTProviderName
    source_language: str
    sample_rate_hz: int
    keyterms: tuple[str, ...]
    deepgram_model: str | None = None
    qwen_model: str | None = None
    qwen_region: QwenRegion | None = None
    soniox_model: str | None = None
    soniox_endpoint: str | None = None
    soniox_keepalive_interval_s: float | None = None
    soniox_trailing_silence_ms: int | None = None


def create_secret_store(
    settings: SecretsSettings,
    *,
    config_path: Path,
    passphrase: str | None = None,
) -> SecretStore:
    passphrase = passphrase or os.getenv(SECRETS_PASSPHRASE_ENV)

    if settings.backend == SecretsBackend.KEYRING:
        return KeyringSecretStore()

    if settings.backend == SecretsBackend.ENCRYPTED_FILE:
        if not passphrase:
            raise ValueError(
                "encrypted_file secrets backend requires a passphrase; "
                f"set {SECRETS_PASSPHRASE_ENV} or pass passphrase explicitly"
            )
        path = Path(settings.encrypted_file_path)
        if not path.is_absolute():
            path = config_path.parent / path
        return EncryptedFileSecretStore(path=path, passphrase=passphrase)

    raise ValueError(f"Unsupported secrets backend: {settings.backend}")


def _get_secret(
    secrets: SecretStore,
    *,
    key: str,
    env_var: str,
) -> str | None:
    value = secrets.get(key)
    if value:
        return value
    env = os.getenv(env_var)
    if env:
        return env
    return None


def _get_secret_any(
    secrets: SecretStore,
    *,
    key: str,
    env_vars: tuple[str, ...],
    legacy_keys: tuple[str, ...] = (),
) -> str | None:
    value = secrets.get(key)
    if value:
        return value
    for legacy_key in legacy_keys:
        legacy_value = secrets.get(legacy_key)
        if legacy_value:
            # Backfill to the new key so subsequent runs do not rely on fallback.
            with contextlib.suppress(Exception):
                secrets.set(key, legacy_value)
            return legacy_value
    for env_var in env_vars:
        env = os.getenv(env_var)
        if env:
            return env
    return None


def require_secret_any(
    secrets: SecretStore,
    *,
    key: str,
    env_vars: tuple[str, ...],
    legacy_keys: tuple[str, ...] = (),
) -> str:
    value = _get_secret_any(secrets, key=key, env_vars=env_vars, legacy_keys=legacy_keys)
    if value:
        return value
    env_list = ", ".join(env_vars)
    raise ValueError(f"Missing secret `{key}` (or env vars {env_list})")


def require_secret(
    secrets: SecretStore,
    *,
    key: str,
    env_var: str,
) -> str:
    value = _get_secret(secrets, key=key, env_var=env_var)
    if value:
        return value
    raise ValueError(f"Missing secret `{key}` (or env var {env_var})")


def create_llm_provider(
    settings: AppSettings,
    *,
    secrets: SecretStore,
    managed_release_service: object | None = None,
    managed_delegate_ready: Callable[[], object] | None = None,
    runtime_logging: SessionRuntimeLoggingService | None = None,
) -> LLMProvider:
    if settings.provider.llm == LLMProviderName.GEMINI:
        api_key = require_secret(secrets, key="google_api_key", env_var="GOOGLE_API_KEY")
        base: LLMProvider = GeminiLLMProvider(
            api_key=api_key,
            model=settings.gemini.llm_model.value,
            runtime_logging=runtime_logging,
        )
    elif settings.provider.llm == LLMProviderName.OPENROUTER:
        primary_alias = _resolve_primary_openrouter_alias(settings)
        base = _create_llm_provider_from_alias_profile(
            settings,
            alias=primary_alias,
            secrets=secrets,
            managed_release_service=managed_release_service,
            managed_delegate_ready=managed_delegate_ready,
            runtime_logging=runtime_logging,
        )
    elif settings.provider.llm == LLMProviderName.QWEN:
        from puripuly_heart.config.settings import QwenRegion

        if settings.qwen.region == QwenRegion.BEIJING:
            api_key = require_secret_any(
                secrets,
                key="alibaba_api_key_beijing",
                env_vars=("ALIBABA_API_KEY_BEIJING", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"),
                legacy_keys=("alibaba_api_key",),
            )
        else:
            api_key = require_secret_any(
                secrets,
                key="alibaba_api_key_singapore",
                env_vars=("ALIBABA_API_KEY_SINGAPORE", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"),
                legacy_keys=("alibaba_api_key",),
            )
        if settings.stt.low_latency_mode:
            # Low-latency mode: use httpx async client for immediate cancellation
            base_url = settings.qwen.get_llm_base_url()
            # Convert SDK URL to OpenAI-compatible URL
            async_base_url = base_url.replace("/api/v1", "/compatible-mode/v1")
            base = AsyncQwenLLMProvider(
                api_key=api_key,
                base_url=async_base_url,
                model=settings.qwen.llm_model.value,
                runtime_logging=runtime_logging,
            )
        else:
            # Standard mode: use DashScope SDK
            base = QwenLLMProvider(
                api_key=api_key,
                base_url=settings.qwen.get_llm_base_url(),
                model=settings.qwen.llm_model.value,
                runtime_logging=runtime_logging,
            )
    elif settings.provider.llm == LLMProviderName.DEEPSEEK:
        api_key = require_secret(
            secrets,
            key="deepseek_api_key",
            env_var="DEEPSEEK_API_KEY",
        )
        base = DeepSeekLLMProvider(
            api_key=api_key,
            model=settings.deepseek.llm_model.value,
            runtime_logging=runtime_logging,
        )
    elif settings.provider.llm == LLMProviderName.CEREBRAS:
        api_key = require_secret(
            secrets,
            key="cerebras_api_key",
            env_var="CEREBRAS_API_KEY",
        )
        base = CerebrasLLMProvider(
            api_key=api_key,
            model=settings.cerebras.llm_model.value,
            runtime_logging=runtime_logging,
        )
    elif settings.provider.llm == LLMProviderName.LOCAL_LLM:
        api_key = (secrets.get("local_llm_api_key") or "").strip()
        base = LocalOpenAICompatibleLLMProvider(
            base_url=settings.local_llm.base_url,
            model=settings.local_llm.model,
            extra_body=settings.local_llm.extra_body,
            api_key=api_key,
            runtime_logging=runtime_logging,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.provider.llm}")

    translation_fallback_alias = settings.translation.fallback_selection_alias
    if translation_fallback_alias != TranslationFallbackSelectionAlias.NONE:
        if _translation_fallback_matches_primary(settings, translation_fallback_alias):
            _emit_translation_fallback_noop(
                runtime_logging=runtime_logging,
                alias=translation_fallback_alias,
            )
        else:
            fallback_managed_release_service = _shared_managed_release_service_for_fallback(
                base,
                managed_release_service,
            )
            base = FallbackRacingLLMProvider(
                primary=base,
                fallback=_LazyFactoryLLMProvider(
                    factory=lambda: _create_translation_fallback_provider(
                        settings=settings,
                        secrets=secrets,
                        alias=translation_fallback_alias,
                        managed_release_service=fallback_managed_release_service,
                        managed_delegate_ready=managed_delegate_ready,
                        runtime_logging=runtime_logging,
                    )
                ),
                runtime_logging=runtime_logging,
            )
    elif (
        settings.provider.llm == LLMProviderName.OPENROUTER
        and settings.openrouter.fallback_selection_alias != OpenRouterFallbackSelectionAlias.NONE
        and settings.openrouter.provider_routing != OpenRouterProviderRouting.DEEPSEEK_ONLY
    ):
        fallback_managed_release_service = _shared_managed_release_service_for_fallback(
            base,
            managed_release_service,
        )
        base = FallbackRacingLLMProvider(
            primary=base,
            fallback=_LazyFactoryLLMProvider(
                factory=lambda: _create_openrouter_fallback_provider(
                    settings=settings,
                    secrets=secrets,
                    managed_release_service=fallback_managed_release_service,
                    managed_delegate_ready=managed_delegate_ready,
                    runtime_logging=runtime_logging,
                )
            ),
            runtime_logging=runtime_logging,
        )

    return SemaphoreLLMProvider(
        inner=base,
        semaphore=asyncio.Semaphore(settings.llm.concurrency_limit),
    )


def create_stt_backend(
    settings: AppSettings,
    *,
    secrets: SecretStore,
    diagnostics_enabled: Callable[[], bool] | None = None,
) -> STTBackend:
    effective_terms = get_effective_custom_terms(settings, settings.languages.source_language)

    if settings.provider.stt == STTProviderName.LOCAL_QWEN:
        from puripuly_heart.core.language import get_local_qwen_language_hint
        from puripuly_heart.core.local_stt_assets import default_local_stt_model_dir
        from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend

        return LocalQwenSherpaSTTBackend(
            model_dir=default_local_stt_model_dir(),
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            stream_label="self",
            language_hint=get_local_qwen_language_hint(settings.languages.source_language),
            diagnostics_enabled=diagnostics_enabled,
        )

    if settings.provider.stt == STTProviderName.DEEPGRAM:
        api_key = require_secret(secrets, key="deepgram_api_key", env_var="DEEPGRAM_API_KEY")
        return _create_deepgram_stt_backend(
            settings=settings,
            api_key=api_key,
            keyterms=effective_terms,
        )

    if settings.provider.stt == STTProviderName.QWEN_ASR:
        from puripuly_heart.config.settings import QwenRegion
        from puripuly_heart.core.language import get_qwen_asr_language
        from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend

        if settings.qwen.region == QwenRegion.BEIJING:
            api_key = require_secret_any(
                secrets,
                key="alibaba_api_key_beijing",
                env_vars=("ALIBABA_API_KEY_BEIJING", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"),
                legacy_keys=("alibaba_api_key",),
            )
        else:
            api_key = require_secret_any(
                secrets,
                key="alibaba_api_key_singapore",
                env_vars=("ALIBABA_API_KEY_SINGAPORE", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"),
                legacy_keys=("alibaba_api_key",),
            )
        endpoint = settings.qwen.get_asr_endpoint()
        return QwenASRRealtimeSTTBackend(
            api_key=api_key,
            model=settings.qwen_asr_stt.model,
            endpoint=endpoint,
            language=get_qwen_asr_language(settings.languages.source_language),
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
        )

    if settings.provider.stt == STTProviderName.SONIOX:
        from puripuly_heart.core.language import get_soniox_language_hints
        from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

        api_key = require_secret(secrets, key="soniox_api_key", env_var="SONIOX_API_KEY")
        return SonioxRealtimeSTTBackend(
            api_key=api_key,
            model=settings.soniox_stt.model,
            endpoint=settings.soniox_stt.endpoint,
            language_hints=get_soniox_language_hints(settings.languages.source_language),
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            keepalive_interval_s=settings.soniox_stt.keepalive_interval_s,
            trailing_silence_ms=settings.soniox_stt.trailing_silence_ms,
            context_terms=effective_terms,
        )

    raise ValueError(f"Unsupported STT provider: {settings.provider.stt}")


def resolve_peer_stt_config(settings: AppSettings) -> ResolvedPeerSTTConfig:
    peer_source_language = settings.languages.effective_peer_source
    keyterms: tuple[str, ...] = ()
    provider = settings.provider.peer_stt

    if provider == STTProviderName.DEEPGRAM:
        return ResolvedPeerSTTConfig(
            provider=provider,
            source_language=peer_source_language,
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            keyterms=keyterms,
            deepgram_model=settings.deepgram_stt.model,
        )

    if provider == STTProviderName.QWEN_ASR:
        return ResolvedPeerSTTConfig(
            provider=provider,
            source_language=peer_source_language,
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            keyterms=keyterms,
            qwen_model=settings.qwen_asr_stt.model,
            qwen_region=settings.qwen.region,
        )

    if provider == STTProviderName.SONIOX:
        return ResolvedPeerSTTConfig(
            provider=provider,
            source_language=peer_source_language,
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            keyterms=keyterms,
            soniox_model=settings.soniox_stt.model,
            soniox_endpoint=settings.soniox_stt.endpoint,
            soniox_keepalive_interval_s=settings.soniox_stt.keepalive_interval_s,
            soniox_trailing_silence_ms=settings.soniox_stt.trailing_silence_ms,
        )

    if provider == STTProviderName.LOCAL_QWEN:
        return ResolvedPeerSTTConfig(
            provider=provider,
            source_language=peer_source_language,
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            keyterms=(),
        )

    raise ValueError(f"Unsupported peer STT provider: {provider}")


def build_peer_stt_provider_signature(settings: AppSettings) -> tuple[object, ...]:
    resolved = resolve_peer_stt_config(settings)
    return (
        resolved.provider,
        resolved.source_language,
        resolved.sample_rate_hz,
        resolved.deepgram_model,
        resolved.qwen_model,
        resolved.qwen_region,
        resolved.soniox_model,
        resolved.soniox_endpoint,
        resolved.soniox_keepalive_interval_s,
        resolved.soniox_trailing_silence_ms,
        resolved.keyterms,
    )


def create_peer_stt_backend(
    settings: AppSettings,
    *,
    secrets: SecretStore,
    diagnostics_enabled: Callable[[], bool] | None = None,
) -> STTBackend:
    resolved = resolve_peer_stt_config(settings)

    if resolved.provider == STTProviderName.DEEPGRAM:
        api_key = require_secret(secrets, key="deepgram_api_key", env_var="DEEPGRAM_API_KEY")
        return _create_deepgram_stt_backend(
            settings=settings,
            api_key=api_key,
            keyterms=resolved.keyterms,
            source_language=resolved.source_language,
            stream_label="peer",
            model=resolved.deepgram_model,
        )

    if resolved.provider == STTProviderName.QWEN_ASR:
        from puripuly_heart.core.language import get_qwen_asr_language
        from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend

        if resolved.qwen_region == QwenRegion.BEIJING:
            api_key = require_secret_any(
                secrets,
                key="alibaba_api_key_beijing",
                env_vars=("ALIBABA_API_KEY_BEIJING", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"),
                legacy_keys=("alibaba_api_key",),
            )
            endpoint = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
        else:
            api_key = require_secret_any(
                secrets,
                key="alibaba_api_key_singapore",
                env_vars=("ALIBABA_API_KEY_SINGAPORE", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"),
                legacy_keys=("alibaba_api_key",),
            )
            endpoint = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"

        return QwenASRRealtimeSTTBackend(
            api_key=api_key,
            model=resolved.qwen_model,
            endpoint=endpoint,
            language=get_qwen_asr_language(resolved.source_language),
            sample_rate_hz=resolved.sample_rate_hz,
        )

    if resolved.provider == STTProviderName.SONIOX:
        from puripuly_heart.core.language import get_soniox_language_hints
        from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

        api_key = require_secret(secrets, key="soniox_api_key", env_var="SONIOX_API_KEY")
        return SonioxRealtimeSTTBackend(
            api_key=api_key,
            model=resolved.soniox_model,
            endpoint=resolved.soniox_endpoint,
            language_hints=get_soniox_language_hints(resolved.source_language),
            sample_rate_hz=resolved.sample_rate_hz,
            keepalive_interval_s=resolved.soniox_keepalive_interval_s,
            trailing_silence_ms=resolved.soniox_trailing_silence_ms,
            context_terms=resolved.keyterms,
        )

    if resolved.provider == STTProviderName.LOCAL_QWEN:
        from puripuly_heart.core.language import get_local_qwen_language_hint
        from puripuly_heart.core.local_stt_assets import default_local_stt_model_dir
        from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend

        return LocalQwenSherpaSTTBackend(
            model_dir=default_local_stt_model_dir(),
            sample_rate_hz=resolved.sample_rate_hz,
            stream_label="peer",
            language_hint=get_local_qwen_language_hint(resolved.source_language),
            diagnostics_enabled=diagnostics_enabled,
        )

    raise ValueError(f"Unsupported peer STT provider: {resolved.provider}")


def _create_deepgram_stt_backend(
    *,
    settings: AppSettings,
    api_key: str,
    keyterms: tuple[str, ...] | list[str],
    source_language: str | None = None,
    stream_label: str | None = None,
    model: str | None = None,
) -> STTBackend:
    from puripuly_heart.core.language import get_deepgram_language
    from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend

    source_language = source_language or settings.languages.source_language
    return DeepgramRealtimeSTTBackend(
        api_key=api_key,
        model=model or settings.deepgram_stt.model,
        language=get_deepgram_language(source_language),
        sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
        keyterms=keyterms,
        stream_label=stream_label,
    )
