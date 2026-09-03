from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app import wiring as wiring_module
from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.wiring import (
    ManagedIdentityStateAdapter,
    ResolvedPeerSTTConfig,
    _LazyFactoryLLMProvider,
    build_openrouter_release_runtime_config,
    build_peer_stt_provider_signature,
    build_peer_stt_provider_signature_from_vnext,
    build_self_stt_runtime_signature,
    build_self_stt_runtime_signature_from_vnext,
    create_llm_provider_from_resolved_config,
    resolve_peer_stt_config,
    resolve_peer_stt_runtime_config_from_vnext,
)
from puripuly_heart.app.wiring import wiring_llm_factory as wiring_llm_factory_module
from puripuly_heart.app.wiring.wiring_llm_factory import (
    create_llm_provider as create_llm_provider_from_runtime_input,
)
from puripuly_heart.app.wiring.wiring_llm_factory import (
    llm_factory_extras_from_vnext,
    runtime_resolution_input_from_vnext,
)
from puripuly_heart.app.wiring.wiring_stt_factory import (
    create_peer_stt_backend_from_resolved_config,
    create_stt_backend_from_resolved_config,
    peer_stt_runtime_intent_from_vnext,
    resolve_peer_stt_runtime_config,
    resolve_self_stt_runtime_config_from_vnext,
    self_stt_runtime_intent_from_vnext,
)
from puripuly_heart.config.llm_profiles import get_openrouter_llm_profile
from puripuly_heart.config.provider_values import (
    CerebrasLLMModel,
    DeepSeekLLMModel,
    GeminiLLMModel,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
    QwenLLMModel,
    QwenRegion,
    STTProviderName,
)
from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_NONE,
    CREDENTIAL_SOURCE_SECRET_STORE,
    ResolvedCredentialRequirement,
    ResolvedLLMConfig,
    ResolvedLLMFallbackPlan,
    ResolvedLLMTarget,
    ResolvedSTTConfig,
)
from puripuly_heart.config.runtime_resolution import (
    TranslationFallbackRuntimeIntent,
    resolve_stt_config,
)
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    DesktopFletOverlayIntent,
    DesktopFletOverlayPositionIntent,
    DesktopFletOverlayVisualIntent,
    TranslationFallbackIntent,
)
from puripuly_heart.core.language import (
    get_deepgram_language,
    get_qwen_asr_language,
)
from puripuly_heart.core.llm import FallbackRacingLLMProvider
from puripuly_heart.core.llm.provider import LLMProvider, SemaphoreLLMProvider
from puripuly_heart.core.openrouter_routing import (
    OpenRouterProviderRouting,
    OpenRouterRoutingMode,
)


class _ConcurrencyProbeProvider(LLMProvider):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def translate(self, **_kwargs: object) -> object:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.active -= 1
        return object()

    async def close(self) -> None:
        return None


def assert_bounded_concurrency(provider: SemaphoreLLMProvider, limit: int) -> None:
    probe = _ConcurrencyProbeProvider()

    async def run() -> None:
        wrapped = SemaphoreLLMProvider(inner=probe, semaphore=provider.semaphore)
        await asyncio.gather(
            *(
                wrapped.translate(
                    utterance_id=uuid.uuid4(),
                    text="t",
                    system_prompt="s",
                    source_language="en",
                    target_language="ko",
                )
                for _ in range(limit * 2 + 1)
            ),
        )

    asyncio.run(run())
    assert probe.max_active == limit


from puripuly_heart.core.local_asr.local_stt_assets import default_local_stt_model_dir
from puripuly_heart.core.openrouter.managed_openrouter_release import (
    ManagedOpenRouterLLMProvider,
    ManagedOpenRouterReleaseService,
    _resolve_managed_issue_model,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore, SecretStore
from puripuly_heart.core.stt.backend import STTBackend
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.providers.llm.cerebras import CerebrasLLMProvider
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
from puripuly_heart.providers.llm.local_openai import LocalOpenAICompatibleLLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from puripuly_heart.providers.stt.local_cpu import LocalCPUAutoSTTBackend
from puripuly_heart.providers.stt.local_parakeet_sherpa import (
    LocalParakeetJapaneseSherpaSTTBackend,
    LocalParakeetV3SherpaSTTBackend,
)
from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend
from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

_LLM_DEFAULTS: dict[str, tuple[str, str]] = {
    "gemini": ("gemini37_flash", "official_byok"),
    "qwen": ("qwen38_flash", "official_byok"),
    "deepseek": ("deepseek_v4_flash", "official_byok"),
    "cerebras": ("gemma4_31b", "cerebras"),
    "local_llm": ("local_llm", "ollama"),
    "openrouter": ("gemma4", "openrouter"),
    "managed_gemma": ("managed_gemma", "cpu"),
}
_OPENROUTER_ALIAS_DEFAULTS: dict[str, tuple[str, str]] = {
    OpenRouterSelectionAlias.QWEN35_FLASH_BYOK.value: (
        "openrouter_qwen35_flash",
        "openrouter",
    ),
    OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED.value: (
        "openrouter_qwen35_flash",
        "managed",
    ),
    OpenRouterSelectionAlias.GEMMA4_BYOK.value: ("gemma4", "openrouter"),
    OpenRouterSelectionAlias.GEMMA4_MANAGED.value: ("gemma4", "managed"),
    OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK.value: (
        "deepseek_v4_flash",
        "openrouter",
    ),
    OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED.value: (
        "deepseek_v4_flash",
        "managed",
    ),
    OpenRouterSelectionAlias.GEMINI37_FLASH_BYOK.value: (
        "gemini37_flash",
        "openrouter",
    ),
}


def _vnext(
    *,
    llm: str | None = None,
    model: str | None = None,
    connection: str | None = None,
    concurrency_limit: int | None = None,
    fallback_alias: str | None = None,
    openrouter_model: str | None = None,
    openrouter_source: str | None = None,
    openrouter_alias: str | None = None,
    openrouter_routing: str | None = None,
    openrouter_routing_mode: str | None = None,
    gemini_model: str | None = None,
    qwen_model: str | None = None,
    qwen_region: str | None = None,
    cerebras_model: str | None = None,
    deepseek_model: str | None = None,
    stt_provider: str | None = None,
    peer_stt_provider: str | None = None,
    low_latency: bool | None = None,
    local_base_url: str | None = None,
    local_model: str | None = None,
    local_extra_body: dict[str, object] | None = None,
    http_extension_id: str | None = None,
    managed_credential_ref: str | None = None,
    deepgram_model: str | None = None,
    source_language: str | None = None,
    peer_source_language: str | None = None,
    peer_source_mode: str | None = None,
    peer_expected_languages: list[str] | None = None,
    custom_vocabulary_enabled: bool | None = None,
    custom_terms: dict[str, list[str]] | None = None,
    soniox_model: str | None = None,
    soniox_endpoint: str | None = None,
    soniox_keepalive_interval_s: float | None = None,
    soniox_trailing_silence_ms: int | None = None,
    qwen_asr_model: str | None = None,
    **stt_fields: object,
) -> AppSettingsVNext:
    settings = AppSettingsVNext()
    translation = settings.intent.translation
    if openrouter_alias is not None:
        mapped = _OPENROUTER_ALIAS_DEFAULTS.get(openrouter_alias)
        if mapped is not None:
            model = model or mapped[0]
            connection = connection or mapped[1]
            if openrouter_model is None:
                profile = get_openrouter_llm_profile(openrouter_alias)
                if profile is not None and profile.openrouter_model is not None:
                    openrouter_model = profile.openrouter_model
    if openrouter_source == "managed" and connection is None:
        connection = "managed"
    elif openrouter_source == "byok" and connection is None:
        connection = "openrouter"
    if llm is not None:
        default_model, default_connection = _LLM_DEFAULTS[llm]
        translation = replace(
            translation,
            model=model or default_model,
            connection=connection or default_connection,
        )
    elif model is not None or connection is not None:
        translation = replace(
            translation,
            model=model or translation.model,
            connection=connection or translation.connection,
        )
    if concurrency_limit is not None:
        translation = replace(translation, concurrency_limit=concurrency_limit)
    translation = replace(
        translation,
        fallback=TranslationFallbackIntent(selection_alias=fallback_alias or "none"),
    )
    if openrouter_routing_mode is not None:
        translation = replace(translation, openrouter_routing_mode=openrouter_routing_mode)
    if gemini_model is not None:
        translation = replace(
            translation,
            gemini=replace(translation.gemini, llm_model=gemini_model),
        )
    if qwen_model is not None or qwen_region is not None:
        translation = replace(
            translation,
            qwen=replace(
                translation.qwen,
                llm_model=qwen_model or translation.qwen.llm_model,
                region=qwen_region or translation.qwen.region,
            ),
        )
    if cerebras_model is not None:
        translation = replace(
            translation,
            cerebras=replace(translation.cerebras, llm_model=cerebras_model),
        )
    if deepseek_model is not None:
        translation = replace(
            translation,
            deepseek=replace(translation.deepseek, llm_model=deepseek_model),
        )
    if http_extension_id is not None:
        translation = replace(translation, http_extension_id=http_extension_id)
    stt = settings.intent.stt
    if stt_provider is not None:
        stt = replace(stt, provider=stt_provider)
    if low_latency is not None:
        stt = replace(stt, low_latency_mode=low_latency)
    if custom_vocabulary_enabled is not None:
        stt = replace(stt, custom_vocabulary_enabled=custom_vocabulary_enabled)
    if custom_terms is not None:
        stt = replace(stt, custom_terms=custom_terms)
    if deepgram_model is not None:
        stt = replace(stt, deepgram=replace(stt.deepgram, model=deepgram_model))
    if qwen_asr_model is not None:
        stt = replace(stt, qwen_asr=replace(stt.qwen_asr, model=qwen_asr_model))
    if (
        soniox_model is not None
        or soniox_endpoint is not None
        or soniox_keepalive_interval_s is not None
        or soniox_trailing_silence_ms is not None
    ):
        stt = replace(
            stt,
            soniox=replace(
                stt.soniox,
                model=soniox_model or stt.soniox.model,
                endpoint=soniox_endpoint or stt.soniox.endpoint,
                keepalive_interval_s=(
                    stt.soniox.keepalive_interval_s
                    if soniox_keepalive_interval_s is None
                    else soniox_keepalive_interval_s
                ),
                trailing_silence_ms=(
                    stt.soniox.trailing_silence_ms
                    if soniox_trailing_silence_ms is None
                    else soniox_trailing_silence_ms
                ),
            ),
        )
    if stt_fields:
        stt = replace(stt, **stt_fields)
    peer_stt = settings.intent.peer_stt
    if peer_stt_provider is not None:
        peer_stt = replace(peer_stt, provider=peer_stt_provider)
    languages = settings.intent.languages
    if source_language is not None:
        languages = replace(languages, source_language=source_language)
    if peer_source_language is not None:
        languages = replace(languages, peer_source_language=peer_source_language)
    if peer_source_mode is not None or peer_expected_languages is not None:
        languages = replace(
            languages,
            peer_source_mode=peer_source_mode or languages.peer_source_mode,
            peer_expected_languages=(
                languages.peer_expected_languages
                if peer_expected_languages is None
                else peer_expected_languages
            ),
        )
    local_llm = settings.intent.local_llm
    if local_base_url is not None or local_model is not None or local_extra_body is not None:
        local_llm = replace(
            local_llm,
            base_url=local_base_url or local_llm.base_url,
            model=local_model or local_llm.model,
            extra_body=local_extra_body if local_extra_body is not None else local_llm.extra_body,
        )
    state = settings.state
    if managed_credential_ref is not None:
        state = replace(
            state,
            managed_connection=replace(
                state.managed_connection,
                active_managed_credential_ref=managed_credential_ref,
            ),
        )
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            translation=translation,
            stt=stt,
            peer_stt=peer_stt,
            languages=languages,
            local_llm=local_llm,
        ),
        state=state,
    )
    settings = materialize_canonical_translation_settings(settings)
    translation = settings.intent.translation
    overlay: dict[str, object] = {}
    if openrouter_model is not None:
        overlay["openrouter_model"] = openrouter_model
    if openrouter_alias is not None:
        overlay["openrouter_selection_alias"] = openrouter_alias
    if openrouter_source is not None:
        overlay["openrouter_selected_source"] = openrouter_source
    if openrouter_routing is not None:
        overlay["openrouter_provider_routing"] = openrouter_routing
    if qwen_model is not None or qwen_region is not None:
        overlay["qwen"] = replace(
            translation.qwen,
            llm_model=qwen_model or translation.qwen.llm_model,
            region=qwen_region or translation.qwen.region,
        )
    if overlay:
        settings = replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(translation, **overlay),
            ),
        )
    return settings


def create_llm_provider(
    settings: AppSettingsVNext,
    *,
    secrets: SecretStore,
    **kwargs: object,
) -> LLMProvider:
    extras = kwargs.pop("extras", None)
    fallback_model = kwargs.pop("fallback_model", None)
    fallback_connection = kwargs.pop("fallback_connection", None)
    if settings.intent.translation.openrouter_selected_source == "none":
        raise ValueError("OpenRouter selected source must not be `none` for execution")
    runtime_input = runtime_resolution_input_from_vnext(settings)
    fallback = settings.intent.translation.fallback
    if fallback_model is not None and fallback_connection is not None:
        runtime_input = replace(
            runtime_input,
            translation_fallback=TranslationFallbackRuntimeIntent(
                enabled=True,
                model=str(fallback_model),
                connection=str(fallback_connection),
            ),
        )
    elif fallback.enabled:
        runtime_input = replace(
            runtime_input,
            translation_fallback=TranslationFallbackRuntimeIntent(
                enabled=True,
                model=fallback.model,
                connection=fallback.connection,
            ),
        )
    return create_llm_provider_from_runtime_input(
        runtime_input,
        secrets=secrets,
        extras=extras if extras is not None else llm_factory_extras_from_vnext(settings),
        **kwargs,
    )


def create_stt_backend(
    settings: AppSettingsVNext,
    *,
    secrets: SecretStore,
    **kwargs: object,
) -> STTBackend:
    return create_stt_backend_from_resolved_config(
        resolve_self_stt_runtime_config_from_vnext(settings),
        secrets=secrets,
        gpu_device_id=settings.intent.stt.gpu_device_id,
        **kwargs,
    )


def create_peer_stt_backend(
    settings: AppSettingsVNext,
    *,
    secrets: SecretStore,
    **kwargs: object,
) -> STTBackend:
    return create_peer_stt_backend_from_resolved_config(
        resolve_peer_stt_runtime_config_from_vnext(settings),
        secrets=secrets,
        gpu_device_id=settings.intent.stt.gpu_device_id,
        **kwargs,
    )


def _unwrap_release_service(release_service: object) -> object:
    while hasattr(release_service, "release_service") and not isinstance(
        release_service,
        ManagedOpenRouterReleaseService,
    ):
        release_service = getattr(release_service, "release_service")
    return release_service


def _resolved_stt_config(
    *,
    channel: str = "self",
    provider: str = "deepgram",
    source_language: str = "ko-KR",
    model: str | None = "nova-3",
    endpoint: str | None = None,
    region: str | None = None,
    credential_reference: str | None = "deepgram:stt",
    input_host_api: str | None = "Windows WASAPI",
    input_device: str | None = "Microphone Array",
    output_device: str | None = None,
    sample_rate_hz: int = 16000,
    custom_vocabulary_enabled: bool = False,
    custom_terms: dict[str, tuple[str, ...]] | None = None,
    provider_options: dict[str, object] | None = None,
) -> ResolvedSTTConfig:
    credential = (
        ResolvedCredentialRequirement(
            source=CREDENTIAL_SOURCE_SECRET_STORE,
            required=True,
            reference=credential_reference,
        )
        if credential_reference is not None
        else ResolvedCredentialRequirement(
            source=CREDENTIAL_SOURCE_NONE,
            required=False,
            reference=None,
        )
    )
    return ResolvedSTTConfig(
        channel=channel,
        source_language=source_language,
        provider=provider,
        model=model,
        endpoint=endpoint,
        region=region,
        credential=credential,
        input_host_api=input_host_api,
        input_device=input_device,
        output_device=output_device,
        sample_rate_hz=sample_rate_hz,
        channels=1,
        ring_buffer_ms=500,
        drain_timeout_s=2.0,
        vad_speech_threshold=0.5,
        vad_hangover_ms=600,
        vad_pre_roll_ms=500,
        low_latency_enabled=True,
        low_latency_merge_gap_ms=600,
        low_latency_spec_retry_max=10,
        custom_vocabulary_enabled=custom_vocabulary_enabled,
        custom_terms={} if custom_terms is None else custom_terms,
        provider_options={} if provider_options is None else provider_options,
    )


def test_legacy_resolved_peer_stt_config_constructor_exposes_old_fields() -> None:
    resolved = ResolvedPeerSTTConfig(
        provider=STTProviderName.SONIOX,
        source_language="zh-CN",
        sample_rate_hz=16000,
        keyterms=("Airi", "Shinano"),
        deepgram_model="nova-peer",
        qwen_model="qwen-peer",
        qwen_region=QwenRegion.SINGAPORE,
        soniox_model="stt-rt-v4-peer",
        soniox_endpoint="wss://peer-soniox.example/realtime",
        soniox_keepalive_interval_s=12.5,
        soniox_trailing_silence_ms=700,
    )

    assert resolved.provider is STTProviderName.SONIOX
    assert resolved.source_language == "zh-CN"
    assert resolved.sample_rate_hz == 16000
    assert resolved.keyterms == ("Airi", "Shinano")
    assert resolved.deepgram_model == "nova-peer"
    assert resolved.qwen_model == "qwen-peer"
    assert resolved.qwen_region is QwenRegion.SINGAPORE
    assert resolved.soniox_model == "stt-rt-v4-peer"
    assert resolved.soniox_endpoint == "wss://peer-soniox.example/realtime"
    assert resolved.soniox_keepalive_interval_s == 12.5
    assert resolved.soniox_trailing_silence_ms == 700


def test_create_llm_provider_gemini_uses_secret_and_concurrency_limit() -> None:
    settings = _vnext(llm="gemini", concurrency_limit=3)
    secrets = InMemorySecretStore()
    secrets.set("google_api_key", "k")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, GeminiLLMProvider)
    assert provider.inner.api_key == "k"
    assert provider.inner.model == "gemini-3.7-flash"
    assert_bounded_concurrency(provider, 3)


def test_create_llm_provider_gemini_uses_selected_model() -> None:
    settings = _vnext(llm="gemini", gemini_model=GeminiLLMModel.GEMINI_37_FLASH.value)
    secrets = InMemorySecretStore()
    secrets.set("google_api_key", "k")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, GeminiLLMProvider)
    assert provider.inner.model == "gemini-3.7-flash"


def test_create_llm_provider_gemini_passes_runtime_logging() -> None:
    settings = _vnext(llm="gemini")
    secrets = InMemorySecretStore()
    secrets.set("google_api_key", "k")
    runtime_logging = object()

    provider = create_llm_provider(settings, secrets=secrets, runtime_logging=runtime_logging)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, GeminiLLMProvider)
    assert provider.inner.runtime_logging is runtime_logging


def test_create_llm_provider_qwen_uses_secret() -> None:
    settings = _vnext(llm="qwen")
    secrets = InMemorySecretStore()
    # Default region is Beijing, so we need alibaba_api_key_beijing
    secrets.set("alibaba_api_key_beijing", "k2")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.api_key == "k2"
    assert provider.inner.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert provider.inner.model == "qwen3.8-flash"
    assert_bounded_concurrency(provider, 5)


def test_create_llm_provider_qwen_low_latency_passes_runtime_logging() -> None:
    settings = _vnext(llm="qwen")
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "k2")
    runtime_logging = object()

    provider = create_llm_provider(settings, secrets=secrets, runtime_logging=runtime_logging)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.runtime_logging is runtime_logging


def test_create_llm_provider_qwen_uses_singapore_region() -> None:
    settings = _vnext(
        llm="qwen",
        qwen_region=QwenRegion.SINGAPORE.value,
        qwen_model=QwenLLMModel.QWEN_38_FLASH.value,
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "k3")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.api_key == "k3"
    assert provider.inner.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert provider.inner.model == "qwen3.8-flash"


def test_create_llm_provider_qwen_uses_legacy_alibaba_secret_key() -> None:
    settings = _vnext(llm="qwen")
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key", "legacy-k2")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.api_key == "legacy-k2"
    # Legacy key should be backfilled to region-specific key for future runs.
    assert secrets.get("alibaba_api_key_beijing") == "legacy-k2"


def test_create_llm_provider_qwen_historical_false_still_uses_async_provider() -> None:
    settings = _vnext(
        llm="qwen",
        low_latency=False,
        qwen_model=QwenLLMModel.QWEN_38_FLASH.value,
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "k2")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.api_key == "k2"
    assert provider.inner.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert provider.inner.model == "qwen3.8-flash"


def test_create_llm_provider_qwen_historical_false_passes_runtime_logging() -> None:
    settings = _vnext(llm="qwen", low_latency=False)
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "k2")
    runtime_logging = object()

    provider = create_llm_provider(settings, secrets=secrets, runtime_logging=runtime_logging)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.runtime_logging is runtime_logging


def test_create_llm_provider_qwen_historical_false_uses_async_singapore() -> None:
    settings = _vnext(
        llm="qwen",
        qwen_region=QwenRegion.SINGAPORE.value,
        qwen_model=QwenLLMModel.QWEN_35_FLASH.value,
        low_latency=False,
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "k3")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.api_key == "k3"
    assert provider.inner.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert provider.inner.model == "qwen3.5-flash"


def test_create_llm_provider_deepseek_uses_secret_and_model() -> None:
    settings = _vnext(llm="deepseek", concurrency_limit=4)
    secrets = InMemorySecretStore()
    secrets.set("deepseek_api_key", "ds-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, DeepSeekLLMProvider)
    assert provider.inner.api_key == "ds-key"
    assert provider.inner.model == "deepseek-v4-flash"
    assert provider.inner.base_url == "https://api.deepseek.com"
    assert_bounded_concurrency(provider, 4)


def test_create_llm_provider_deepseek_uses_v4_flash_model() -> None:
    settings = _vnext(llm="deepseek", deepseek_model=DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value)
    secrets = InMemorySecretStore()
    secrets.set("deepseek_api_key", "ds-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, DeepSeekLLMProvider)
    assert provider.inner.model == "deepseek-v4-flash"


def test_create_llm_provider_deepseek_passes_runtime_logging() -> None:
    settings = _vnext(llm="deepseek")
    secrets = InMemorySecretStore()
    secrets.set("deepseek_api_key", "ds-key")
    runtime_logging = object()

    provider = create_llm_provider(settings, secrets=secrets, runtime_logging=runtime_logging)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, DeepSeekLLMProvider)
    assert provider.inner.runtime_logging is runtime_logging


def test_create_llm_provider_cerebras_uses_secret_and_model() -> None:
    settings = _vnext(
        llm="cerebras",
        cerebras_model=CerebrasLLMModel.GEMMA_4_31B.value,
        concurrency_limit=6,
    )
    secrets = InMemorySecretStore()
    secrets.set("cerebras_api_key", "cerebras-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, CerebrasLLMProvider)
    assert provider.inner.api_key == "cerebras-key"
    assert provider.inner.model == "gemma-4-31b"
    assert_bounded_concurrency(provider, 6)


def test_create_llm_provider_cerebras_from_resolved_config_uses_dto_and_secret_store() -> None:
    resolved = ResolvedLLMConfig(
        primary=ResolvedLLMTarget(
            provider="cerebras",
            model="gemma-4-31b",
            credential=ResolvedCredentialRequirement(
                source=CREDENTIAL_SOURCE_SECRET_STORE,
                required=True,
                reference="cerebras:byok",
            ),
        ),
        concurrency_limit=2,
    )
    secrets = InMemorySecretStore()
    secrets.set("cerebras_api_key", "dto-cerebras-key")

    provider = create_llm_provider_from_resolved_config(resolved, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, CerebrasLLMProvider)
    assert provider.inner.api_key == "dto-cerebras-key"
    assert provider.inner.model == "gemma-4-31b"
    assert_bounded_concurrency(provider, 2)


def test_create_llm_provider_local_llm_uses_settings_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _vnext(
        llm="local_llm",
        local_base_url="http://127.0.0.1:11434/v1",
        local_model="llama3.1:8b",
        local_extra_body={"think": False},
        concurrency_limit=2,
    )
    secrets = InMemorySecretStore()
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, LocalOpenAICompatibleLLMProvider)
    assert provider.inner.base_url == "http://127.0.0.1:11434/v1"
    assert provider.inner.model == "llama3.1:8b"
    assert provider.inner.api_key == ""
    assert provider.inner.extra_body == {"think": False}
    assert_bounded_concurrency(provider, 2)


def test_create_llm_provider_local_llm_ignores_optional_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _vnext(llm="local_llm")
    secrets = InMemorySecretStore()
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-secret")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, LocalOpenAICompatibleLLMProvider)
    assert provider.inner.api_key == ""


def test_create_llm_provider_local_llm_uses_secret_store_key_even_when_env_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _vnext(llm="local_llm")
    secrets = InMemorySecretStore()
    secrets.set("local_llm_api_key", "store-secret")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "env-secret")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, LocalOpenAICompatibleLLMProvider)
    assert provider.inner.api_key == "store-secret"


def test_create_llm_provider_from_resolved_local_llm_uses_dto_values_and_optional_secret() -> None:
    legacy_settings = _vnext(
        llm="local_llm",
        local_base_url="http://legacy.local/v1",
        local_model="legacy-model",
        local_extra_body={"legacy": True},
        concurrency_limit=1,
    )
    resolved = ResolvedLLMConfig(
        primary=ResolvedLLMTarget(
            provider="local_llm",
            model="dto-model",
            credential=ResolvedCredentialRequirement(
                source=CREDENTIAL_SOURCE_NONE,
                required=False,
                reference=None,
            ),
            base_url="http://dto.local/v1",
            provider_options={"extra_body": {"think": False}},
        ),
        concurrency_limit=7,
    )
    secrets = InMemorySecretStore()
    secrets.set("local_llm_api_key", "dto-local-secret")

    provider = create_llm_provider_from_resolved_config(
        resolved,
        secrets=secrets,
        extras=llm_factory_extras_from_vnext(legacy_settings),
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, LocalOpenAICompatibleLLMProvider)
    assert provider.inner.base_url == "http://dto.local/v1"
    assert provider.inner.model == "dto-model"
    assert provider.inner.api_key == "dto-local-secret"
    assert provider.inner.extra_body == {"think": False}
    assert_bounded_concurrency(provider, 7)


def test_create_llm_provider_openrouter_uses_secret_and_model() -> None:
    settings = _vnext(
        llm="openrouter",
        concurrency_limit=4,
        openrouter_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
        openrouter_routing_mode=OpenRouterRoutingMode.LATENCY.value,
        fallback_alias="none",
        openrouter_source="byok",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "or-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "or-key"
    assert provider.inner.model == "google/gemma-4-26b-a4b-it"
    assert provider.inner.base_url == "https://openrouter.ai/api/v1"
    assert provider.inner.routing_mode == OpenRouterRoutingMode.LATENCY
    assert_bounded_concurrency(provider, 4)


def test_create_llm_provider_from_resolved_openrouter_gemini_byok_uses_google_latency_routing() -> (
    None
):
    resolved = ResolvedLLMConfig(
        primary=ResolvedLLMTarget(
            provider="openrouter",
            model=OpenRouterLLMModel.GEMINI_37_FLASH.value,
            credential=ResolvedCredentialRequirement(
                source=CREDENTIAL_SOURCE_SECRET_STORE,
                required=True,
                reference="openrouter:byok",
            ),
            routing_mode="latency",
            provider_routing="google_gemini_latency",
        ),
        concurrency_limit=3,
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "gemini-byok-key")

    provider = create_llm_provider_from_resolved_config(resolved, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "gemini-byok-key"
    assert provider.inner.model == OpenRouterLLMModel.GEMINI_37_FLASH.value
    assert provider.inner.routing_mode == OpenRouterRoutingMode.LATENCY
    assert provider.inner.provider_routing == OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY
    assert_bounded_concurrency(provider, 3)


def test_create_llm_provider_openrouter_byok_still_uses_user_owned_secret_after_pkce_storage() -> (
    None
):
    settings = _vnext(
        llm="openrouter",
        openrouter_source="byok",
        openrouter_alias=OpenRouterSelectionAlias.GEMMA4_BYOK.value,
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "pkce-user-key")
    secrets.set("openrouter_managed_api_key", "managed-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "pkce-user-key"


def test_create_llm_provider_openrouter_qwen_byok_alias_uses_resolved_qwen_model() -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_source="byok",
        openrouter_alias=OpenRouterSelectionAlias.QWEN35_FLASH_BYOK.value,
        fallback_alias="none",
        openrouter_routing_mode="latency",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "qwen-byok-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "qwen-byok-key"
    assert provider.inner.model == OpenRouterLLMModel.QWEN_35_FLASH_02_23.value
    assert provider.inner.routing_mode == OpenRouterRoutingMode.LATENCY
    assert provider.inner.provider_routing == OpenRouterProviderRouting.DEFAULT


def test_create_llm_provider_openrouter_qwen_byok_deepseek_only_skips_fallback_racing() -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_source="byok",
        openrouter_alias=OpenRouterSelectionAlias.QWEN35_FLASH_BYOK.value,
        fallback_alias="none",
        openrouter_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY.value,
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "qwen-byok-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert not isinstance(provider.inner, FallbackRacingLLMProvider)
    assert provider.inner.model == OpenRouterLLMModel.QWEN_35_FLASH_02_23.value
    assert provider.inner.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY


def test_create_llm_provider_openrouter_passes_runtime_logging() -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_source="byok",
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "or-key")
    runtime_logging = object()

    provider = create_llm_provider(settings, secrets=secrets, runtime_logging=runtime_logging)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.runtime_logging is runtime_logging


def test_create_llm_provider_openrouter_uses_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-or-key")
    settings = _vnext(
        llm="openrouter",
        openrouter_source="byok",
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "env-or-key"


def test_create_llm_provider_openrouter_uses_selected_managed_key() -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_source="managed",
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "byok-key")
    secrets.set("openrouter_managed_api_key", "managed-key")
    managed_release_service = object()

    provider = create_llm_provider(
        settings,
        secrets=secrets,
        managed_release_service=managed_release_service,
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "managed-key"


def test_create_llm_provider_openrouter_deepseek_only_skips_openrouter_fallback_racing() -> None:
    settings = _vnext(
        llm="openrouter",
        model="deepseek_v4_flash",
        connection="managed_china",
        openrouter_model=OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value,
        openrouter_source="managed",
        openrouter_alias=OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED.value,
        fallback_alias="none",
        openrouter_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY.value,
        managed_credential_ref="managed-ref-qq",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_managed_qq_api_key", "managed-qq-key")

    provider = create_llm_provider(
        settings,
        secrets=secrets,
        managed_release_service=object(),
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    assert provider.inner.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY


def test_create_llm_provider_openrouter_deepseek_byok_deepseek_only_skips_fallback_racing() -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_model=OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value,
        openrouter_source="byok",
        openrouter_alias=OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK.value,
        fallback_alias="none",
        openrouter_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY.value,
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "byok-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert not isinstance(provider.inner, FallbackRacingLLMProvider)
    assert provider.inner.api_key == "byok-key"
    assert provider.inner.model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    assert provider.inner.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY


def test_create_llm_provider_deepseek_flash_official_fallback_uses_flash_model() -> None:
    settings = _vnext(
        llm="deepseek",
        deepseek_model=DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value,
        fallback_alias="deepseek_v4_flash_official",
    )
    secrets = InMemorySecretStore()
    secrets.set("deepseek_api_key", "deepseek-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, FallbackRacingLLMProvider)
    assert isinstance(provider.inner.primary, DeepSeekLLMProvider)
    assert provider.inner.primary.model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value
    assert isinstance(provider.inner.fallback, _LazyFactoryLLMProvider)

    fallback_delegate = provider.inner.fallback.factory()

    assert isinstance(fallback_delegate, DeepSeekLLMProvider)
    assert fallback_delegate.model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value


def test_create_llm_provider_openrouter_deepseek_china_fallback_uses_deepseek_only_routing() -> (
    None
):
    settings = _vnext(
        llm="openrouter",
        openrouter_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
        openrouter_source="byok",
        openrouter_alias=OpenRouterSelectionAlias.GEMMA4_BYOK.value,
        openrouter_routing="default",
        fallback_alias="deepseek_v4_flash_china",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "byok-key")
    secrets.set("openrouter_managed_qq_api_key", "managed-qq-key")

    provider = create_llm_provider(settings, secrets=secrets, managed_release_service=object())

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, FallbackRacingLLMProvider)
    assert isinstance(provider.inner.primary, OpenRouterLLMProvider)
    assert provider.inner.primary.provider_routing == OpenRouterProviderRouting.GEMMA4_26B_LATENCY
    assert isinstance(provider.inner.fallback, _LazyFactoryLLMProvider)

    fallback_provider = provider.inner.fallback.factory()

    assert isinstance(fallback_provider, ManagedOpenRouterLLMProvider)
    fallback_delegate = fallback_provider.delegate_factory("managed-qq-key")
    assert isinstance(fallback_delegate, OpenRouterLLMProvider)
    assert fallback_delegate.model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    assert fallback_delegate.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY


def test_create_llm_provider_from_resolved_openrouter_fallback_uses_resolved_routing() -> None:
    resolved = ResolvedLLMConfig(
        primary=ResolvedLLMTarget(
            provider="openrouter",
            model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
            credential=ResolvedCredentialRequirement(
                source=CREDENTIAL_SOURCE_SECRET_STORE,
                required=True,
                reference="openrouter:byok",
            ),
            routing_mode=OpenRouterRoutingMode.LATENCY.value,
            provider_routing=OpenRouterProviderRouting.DEFAULT.value,
        ),
        fallback=ResolvedLLMFallbackPlan(
            target=ResolvedLLMTarget(
                provider="openrouter",
                model=OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value,
                credential=ResolvedCredentialRequirement(
                    source=CREDENTIAL_SOURCE_SECRET_STORE,
                    required=True,
                    reference="openrouter:byok",
                ),
                routing_mode=OpenRouterRoutingMode.LATENCY.value,
                provider_routing="deepseek_only",
            )
        ),
        concurrency_limit=3,
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "or-key")
    runtime_logging = object()

    provider = create_llm_provider_from_resolved_config(
        resolved,
        secrets=secrets,
        runtime_logging=runtime_logging,
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, FallbackRacingLLMProvider)
    assert isinstance(provider.inner.primary, OpenRouterLLMProvider)
    assert provider.inner.primary.model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    assert provider.inner.primary.routing_mode == OpenRouterRoutingMode.LATENCY
    assert provider.inner.primary.provider_routing == OpenRouterProviderRouting.DEFAULT
    assert isinstance(provider.inner.fallback, _LazyFactoryLLMProvider)
    assert provider.inner.runtime_logging is runtime_logging
    assert provider.inner.attempts[1].log_summary == (
        "provider=openrouter, model=deepseek/deepseek-v4-flash-0731, mode=latency, "
        "route=deepseek_only, delay=1300ms"
    )

    fallback_provider = provider.inner.fallback.factory()

    assert isinstance(fallback_provider, OpenRouterLLMProvider)
    assert fallback_provider.model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    assert fallback_provider.routing_mode == OpenRouterRoutingMode.LATENCY
    assert fallback_provider.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY
    assert_bounded_concurrency(provider, 3)


def test_create_llm_provider_from_resolved_cerebras_fallback_uses_resolved_secret() -> None:
    resolved = ResolvedLLMConfig(
        primary=ResolvedLLMTarget(
            provider="deepseek",
            model=DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value,
            credential=ResolvedCredentialRequirement(
                source=CREDENTIAL_SOURCE_SECRET_STORE,
                required=True,
                reference="deepseek:byok",
            ),
        ),
        fallback=ResolvedLLMFallbackPlan(
            target=ResolvedLLMTarget(
                provider="cerebras",
                model=CerebrasLLMModel.GEMMA_4_31B.value,
                credential=ResolvedCredentialRequirement(
                    source=CREDENTIAL_SOURCE_SECRET_STORE,
                    required=True,
                    reference="cerebras:byok",
                ),
            )
        ),
    )
    secrets = InMemorySecretStore()
    secrets.set("deepseek_api_key", "deepseek-key")
    secrets.set("cerebras_api_key", "cerebras-key")

    provider = create_llm_provider_from_resolved_config(resolved, secrets=secrets)

    assert isinstance(provider.inner, FallbackRacingLLMProvider)
    fallback_provider = provider.inner.fallback.factory()
    assert isinstance(fallback_provider, CerebrasLLMProvider)
    assert fallback_provider.api_key == "cerebras-key"
    assert fallback_provider.model == CerebrasLLMModel.GEMMA_4_31B.value


def test_create_llm_provider_openrouter_direct_managed_reuse_forwards_cached_user_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_source="managed",
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_managed_api_key", "managed-key")
    calls: list[OpenRouterCredentialSource] = []

    def fake_load_managed_openrouter_user_identifier(
        loaded_config: object,
        *,
        secrets: InMemorySecretStore,
    ) -> str:
        _ = secrets
        calls.append(loaded_config.selected_source)
        return "managed-user-123"

    monkeypatch.setattr(
        wiring_llm_factory_module,
        "load_managed_openrouter_user_identifier",
        fake_load_managed_openrouter_user_identifier,
        raising=False,
    )

    provider = create_llm_provider(
        settings,
        secrets=secrets,
        managed_release_service=object(),
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "managed-key"
    assert provider.inner.user_identifier == "managed-user-123"
    assert calls == [OpenRouterCredentialSource.MANAGED]


def test_create_llm_provider_openrouter_requires_release_service_for_managed_mode() -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_source="managed",
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "byok-key")
    secrets.set("openrouter_managed_api_key", "managed-key")

    with pytest.raises(ValueError, match="managed release service"):
        create_llm_provider(settings, secrets=secrets)


def test_create_llm_provider_openrouter_uses_managed_wrapper_when_release_service_is_available() -> (
    None
):
    settings = _vnext(
        llm="openrouter",
        openrouter_source="managed",
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    managed_release_service = object()
    runtime_logging = object()

    provider = create_llm_provider(
        settings,
        secrets=secrets,
        managed_release_service=managed_release_service,
        runtime_logging=runtime_logging,
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, ManagedOpenRouterLLMProvider)
    assert _unwrap_release_service(provider.inner.release_service) is managed_release_service
    delegate = provider.inner.delegate_factory("delegate-key")
    assert isinstance(delegate, OpenRouterLLMProvider)
    assert delegate.runtime_logging is runtime_logging


def test_create_llm_provider_openrouter_managed_delegate_factory_loads_user_identifier_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_source="managed",
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    current_user_identifier: str | None = None
    load_calls = 0

    def fake_load_managed_openrouter_user_identifier(
        loaded_settings: AppSettingsVNext,
        *,
        secrets: InMemorySecretStore,
    ) -> str | None:
        nonlocal load_calls
        _ = loaded_settings, secrets
        load_calls += 1
        return current_user_identifier

    monkeypatch.setattr(
        wiring_llm_factory_module,
        "load_managed_openrouter_user_identifier",
        fake_load_managed_openrouter_user_identifier,
        raising=False,
    )

    provider = create_llm_provider(
        settings,
        secrets=secrets,
        managed_release_service=object(),
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, ManagedOpenRouterLLMProvider)
    assert load_calls == 0

    current_user_identifier = "managed-user-123"
    delegate = provider.inner.delegate_factory("delegate-key")

    assert isinstance(delegate, OpenRouterLLMProvider)
    assert delegate.user_identifier == "managed-user-123"
    assert load_calls == 1


def test_create_llm_provider_openrouter_wraps_primary_with_source_locked_openrouter_fallback() -> (
    None
):
    settings = _vnext(
        llm="openrouter",
        openrouter_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
        openrouter_source="byok",
        openrouter_routing_mode="latency",
        openrouter_alias=OpenRouterSelectionAlias.GEMMA4_BYOK.value,
        fallback_alias="openrouter_deepseek_v4_flash",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "or-key")
    runtime_logging = object()

    provider = create_llm_provider(
        settings,
        secrets=secrets,
        runtime_logging=runtime_logging,
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, FallbackRacingLLMProvider)
    assert isinstance(provider.inner.primary, OpenRouterLLMProvider)
    assert provider.inner.primary.api_key == "or-key"
    assert provider.inner.primary.model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    assert provider.inner.primary.routing_mode == OpenRouterRoutingMode.LATENCY
    assert provider.inner.primary.runtime_logging is runtime_logging
    assert isinstance(provider.inner.fallback, _LazyFactoryLLMProvider)

    fallback_delegate = provider.inner.fallback.factory()

    assert isinstance(fallback_delegate, OpenRouterLLMProvider)
    assert fallback_delegate.api_key == "or-key"
    assert fallback_delegate.model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    assert fallback_delegate.routing_mode == OpenRouterRoutingMode.LATENCY
    assert fallback_delegate.runtime_logging is runtime_logging


def test_create_llm_provider_openrouter_byok_paths_omit_managed_user_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
        openrouter_source="byok",
        openrouter_routing_mode="latency",
        openrouter_alias=OpenRouterSelectionAlias.GEMMA4_BYOK.value,
        fallback_alias="openrouter_deepseek_v4_flash",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "or-key")

    def unexpected_load_managed_openrouter_user_identifier(
        loaded_settings: AppSettingsVNext,
        *,
        secrets: InMemorySecretStore,
    ) -> str:
        _ = loaded_settings, secrets
        raise AssertionError("managed user identifier should not be loaded for BYOK paths")

    monkeypatch.setattr(
        wiring_llm_factory_module,
        "load_managed_openrouter_user_identifier",
        unexpected_load_managed_openrouter_user_identifier,
        raising=False,
    )

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, FallbackRacingLLMProvider)
    assert isinstance(provider.inner.primary, OpenRouterLLMProvider)
    assert provider.inner.primary.user_identifier is None
    assert isinstance(provider.inner.fallback, _LazyFactoryLLMProvider)

    fallback_delegate = provider.inner.fallback.factory()

    assert isinstance(fallback_delegate, OpenRouterLLMProvider)
    assert fallback_delegate.user_identifier is None


def test_create_llm_provider_openrouter_legacy_qwen_fallback_alias_is_ignored() -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
        openrouter_source="managed",
        openrouter_routing_mode="latency",
        openrouter_alias=OpenRouterSelectionAlias.GEMMA4_MANAGED.value,
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    managed_release_service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(
            SimpleNamespace(**asdict(settings.state.managed_connection)),
            lambda _updated: None,
        ),
        secrets=secrets,
        client=object(),
        app_version="2.0.0",
        raw_hardware_fingerprint_provider=lambda: "raw-hardware-fingerprint-test",
    )

    provider = create_llm_provider(
        settings,
        secrets=secrets,
        managed_release_service=managed_release_service,
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, ManagedOpenRouterLLMProvider)
    assert _unwrap_release_service(provider.inner.release_service) is managed_release_service
    assert (
        settings.intent.translation.openrouter_selection_alias
        == OpenRouterSelectionAlias.GEMMA4_MANAGED.value
    )
    assert (
        settings.intent.translation.openrouter_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    )


def test_create_llm_provider_openrouter_managed_deepseek_fallback_uses_fallback_specific_release_service() -> (
    None
):
    deepseek_model = getattr(OpenRouterLLMModel, "DEEPSEEK_V4_FLASH", None)

    assert deepseek_model is not None

    settings = _vnext(
        llm="openrouter",
        openrouter_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
        openrouter_source="managed",
        openrouter_routing_mode="latency",
        openrouter_alias=OpenRouterSelectionAlias.GEMMA4_MANAGED.value,
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    managed_release_service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(
            SimpleNamespace(**asdict(settings.state.managed_connection)),
            lambda _updated: None,
        ),
        secrets=secrets,
        client=object(),
        app_version="2.0.0",
        raw_hardware_fingerprint_provider=lambda: "raw-hardware-fingerprint-test",
    )

    provider = create_llm_provider(
        settings,
        secrets=secrets,
        managed_release_service=managed_release_service,
        fallback_model="deepseek_v4_flash",
        fallback_connection="managed",
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, FallbackRacingLLMProvider)
    assert isinstance(provider.inner.primary, ManagedOpenRouterLLMProvider)
    assert (
        _unwrap_release_service(provider.inner.primary.release_service) is managed_release_service
    )
    assert isinstance(provider.inner.fallback, _LazyFactoryLLMProvider)

    fallback_delegate = provider.inner.fallback.factory()

    assert isinstance(fallback_delegate, ManagedOpenRouterLLMProvider)
    fallback_release_service = _unwrap_release_service(fallback_delegate.release_service)
    assert isinstance(fallback_release_service, ManagedOpenRouterReleaseService)
    assert fallback_release_service is not managed_release_service
    assert fallback_release_service.openrouter_config.selection_alias is None
    assert fallback_release_service.openrouter_config.llm_model == deepseek_model
    assert (
        _resolve_managed_issue_model(fallback_release_service.openrouter_config)
        == deepseek_model.value
    )
    assert (
        settings.intent.translation.openrouter_selection_alias
        == OpenRouterSelectionAlias.GEMMA4_MANAGED.value
    )
    assert (
        settings.intent.translation.openrouter_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    )

    fallback_openrouter_delegate = fallback_delegate.delegate_factory("managed-key")

    assert isinstance(fallback_openrouter_delegate, OpenRouterLLMProvider)
    assert fallback_openrouter_delegate.model == deepseek_model.value
    assert fallback_openrouter_delegate.routing_mode == OpenRouterRoutingMode.LATENCY


def test_create_llm_provider_openrouter_managed_fallback_delegate_factory_loads_user_identifier_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_source="managed",
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    current_user_identifier: str | None = None
    load_calls = 0

    def fake_load_managed_openrouter_user_identifier(
        loaded_settings: AppSettingsVNext,
        *,
        secrets: InMemorySecretStore,
    ) -> str | None:
        nonlocal load_calls
        _ = loaded_settings, secrets
        load_calls += 1
        return current_user_identifier

    monkeypatch.setattr(
        wiring_llm_factory_module,
        "load_managed_openrouter_user_identifier",
        fake_load_managed_openrouter_user_identifier,
        raising=False,
    )

    provider = create_llm_provider(
        settings,
        secrets=secrets,
        managed_release_service=object(),
        fallback_model="deepseek_v4_flash",
        fallback_connection="managed",
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, FallbackRacingLLMProvider)
    assert isinstance(provider.inner.fallback, _LazyFactoryLLMProvider)
    assert load_calls == 0

    fallback_provider = provider.inner.fallback.factory()

    assert isinstance(fallback_provider, ManagedOpenRouterLLMProvider)
    assert load_calls == 0

    current_user_identifier = "managed-user-456"
    fallback_delegate = fallback_provider.delegate_factory("delegate-key")

    assert isinstance(fallback_delegate, OpenRouterLLMProvider)
    assert fallback_delegate.user_identifier == "managed-user-456"
    assert load_calls == 1


def test_create_llm_provider_openrouter_managed_deepseek_fallback_clears_primary_alias_for_issue_identity() -> (
    None
):
    settings = _vnext(
        llm="openrouter",
        openrouter_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
        openrouter_source="managed",
        openrouter_routing_mode="latency",
        openrouter_alias=OpenRouterSelectionAlias.GEMMA4_MANAGED.value,
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    managed_release_service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(
            SimpleNamespace(**asdict(settings.state.managed_connection)),
            lambda _updated: None,
        ),
        secrets=secrets,
        client=object(),
        app_version="2.0.0",
        raw_hardware_fingerprint_provider=lambda: "raw-hardware-fingerprint-test",
    )

    provider = create_llm_provider(
        settings,
        secrets=secrets,
        managed_release_service=managed_release_service,
        fallback_model="deepseek_v4_flash",
        fallback_connection="managed",
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, FallbackRacingLLMProvider)
    assert isinstance(provider.inner.fallback, _LazyFactoryLLMProvider)

    fallback_delegate = provider.inner.fallback.factory()

    assert isinstance(fallback_delegate, ManagedOpenRouterLLMProvider)
    fallback_release_service = _unwrap_release_service(fallback_delegate.release_service)
    assert isinstance(fallback_release_service, ManagedOpenRouterReleaseService)
    assert fallback_release_service is not managed_release_service
    assert fallback_release_service.openrouter_config.selection_alias is None
    assert (
        fallback_release_service.openrouter_config.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    )
    assert (
        _resolve_managed_issue_model(fallback_release_service.openrouter_config)
        == OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    )
    assert (
        settings.intent.translation.openrouter_selection_alias
        == OpenRouterSelectionAlias.GEMMA4_MANAGED.value
    )
    assert (
        settings.intent.translation.openrouter_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    )

    fallback_openrouter_delegate = fallback_delegate.delegate_factory("managed-key")

    assert isinstance(fallback_openrouter_delegate, OpenRouterLLMProvider)
    assert fallback_openrouter_delegate.model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    assert fallback_openrouter_delegate.routing_mode == OpenRouterRoutingMode.LATENCY


def test_create_llm_provider_openrouter_rejects_none_selected_source_even_with_keys() -> None:
    settings = _vnext(
        llm="openrouter",
        openrouter_source="none",
        fallback_alias="none",
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "byok-key")
    secrets.set("openrouter_managed_api_key", "managed-key")

    with pytest.raises(ValueError, match="selected source"):
        create_llm_provider(settings, secrets=secrets)


def test_create_llm_provider_requires_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = _vnext(llm="gemini")
    secrets = InMemorySecretStore()
    with pytest.raises(ValueError):
        create_llm_provider(settings, secrets=secrets)


def test_create_stt_backend_from_resolved_deepgram_uses_dto_values_and_secret() -> None:
    resolved = _resolved_stt_config(
        provider="deepgram",
        source_language="ko-KR",
        model="nova-3-general",
        custom_vocabulary_enabled=True,
        custom_terms={"ko-KR": ("Puripuly", "VRChat")},
    )
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "dto-deepgram-key")

    backend = wiring_module.create_stt_backend_from_resolved_config(resolved, secrets=secrets)

    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert backend.api_key == "dto-deepgram-key"
    assert backend.model == "nova-3-general"
    assert backend.language == get_deepgram_language("ko-KR")
    assert backend.sample_rate_hz == 16000
    assert list(backend.keyterms) == ["Puripuly", "VRChat"]
    assert backend.stream_label == "self"


def test_create_stt_backend_from_resolved_qwen_uses_endpoint_region_and_secret_ref() -> None:
    resolved = _resolved_stt_config(
        provider="qwen_asr",
        source_language="ja",
        model="qwen3-asr-dto",
        endpoint="wss://dto-qwen.example/realtime",
        region="singapore",
        credential_reference="qwen:singapore",
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "dto-qwen-key")

    backend = wiring_module.create_stt_backend_from_resolved_config(resolved, secrets=secrets)

    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.api_key == "dto-qwen-key"
    assert backend.model == "qwen3-asr-dto"
    assert backend.endpoint == "wss://dto-qwen.example/realtime"
    assert backend.language == get_qwen_asr_language("ja")


def test_create_stt_backend_from_resolved_qwen_uses_region_when_endpoint_missing() -> None:
    resolved = _resolved_stt_config(
        provider="qwen_asr",
        source_language="ja",
        model="qwen3-asr-dto",
        endpoint=None,
        region="singapore",
        credential_reference="qwen:singapore",
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "dto-qwen-key")

    backend = wiring_module.create_stt_backend_from_resolved_config(resolved, secrets=secrets)

    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.endpoint == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"


def test_create_stt_backend_from_resolved_soniox_uses_options_and_custom_terms() -> None:
    resolved = _resolved_stt_config(
        provider="soniox",
        source_language="zh-CN",
        model="stt-rt-v4-dto",
        endpoint="wss://dto-soniox.example/realtime",
        credential_reference="soniox:stt",
        custom_vocabulary_enabled=True,
        custom_terms={"zh-CN": ("Airi", "Shinano")},
        provider_options={"keepalive_interval_s": 12.5, "trailing_silence_ms": 450},
    )
    secrets = InMemorySecretStore()
    secrets.set("soniox_api_key", "dto-soniox-key")

    backend = wiring_module.create_stt_backend_from_resolved_config(resolved, secrets=secrets)

    assert isinstance(backend, SonioxRealtimeSTTBackend)
    assert backend.api_key == "dto-soniox-key"
    assert backend.model == "stt-rt-v4-dto"
    assert backend.endpoint == "wss://dto-soniox.example/realtime"
    assert backend.sample_rate_hz == 16000
    assert backend.keepalive_interval_s == 12.5
    assert backend.trailing_silence_ms == 450
    assert list(backend.context_terms) == ["Airi", "Shinano"]


def test_create_stt_backend_from_resolved_soniox_falls_back_to_realtime_v5_model() -> None:
    resolved = _resolved_stt_config(
        provider="soniox",
        source_language="en",
        model=None,
        endpoint=None,
        credential_reference="soniox:stt",
    )
    secrets = InMemorySecretStore()
    secrets.set("soniox_api_key", "dto-soniox-key")

    backend = wiring_module.create_stt_backend_from_resolved_config(resolved, secrets=secrets)

    assert isinstance(backend, SonioxRealtimeSTTBackend)
    assert backend.model == "stt-rt-v5"


def test_create_stt_backend_from_resolved_local_qwen_uses_channel_language_and_no_secret() -> None:
    resolved = _resolved_stt_config(
        provider="local_qwen",
        source_language="zh-CN",
        model=None,
        credential_reference=None,
    )
    secrets = InMemorySecretStore()

    backend = wiring_module.create_stt_backend_from_resolved_config(resolved, secrets=secrets)

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert backend.model_dir == default_local_stt_model_dir()
    assert backend.sample_rate_hz == 16000
    assert backend.stream_label == "self"
    assert backend.language_hint == "zh"


@pytest.mark.parametrize(
    ("provider", "backend_type"),
    [
        ("local_parakeet_v3", LocalParakeetV3SherpaSTTBackend),
        ("local_parakeet_ja", LocalParakeetJapaneseSherpaSTTBackend),
    ],
)
def test_create_stt_backend_from_resolved_direct_parakeet_provider(
    provider: str,
    backend_type: type,
) -> None:
    resolved = _resolved_stt_config(
        provider=provider,
        source_language="ja",
        model=None,
        credential_reference=None,
    )

    backend = wiring_module.create_stt_backend_from_resolved_config(
        resolved,
        secrets=InMemorySecretStore(),
    )

    assert isinstance(backend, backend_type)
    assert backend.sample_rate_hz == 16000
    assert backend.stream_label == "self"


def test_create_stt_backend_from_resolved_cpu_auto_provider() -> None:
    resolved = _resolved_stt_config(
        provider="local_cpu_auto",
        source_language="ja",
        model=None,
        credential_reference=None,
    )

    backend = wiring_module.create_stt_backend_from_resolved_config(
        resolved,
        secrets=InMemorySecretStore(),
    )

    assert isinstance(backend, LocalCPUAutoSTTBackend)
    assert backend.source_language == "ja"
    assert backend.stream_label == "self"


def test_create_stt_backend_from_resolved_gpu_provider_fails_without_cpu_fallback() -> None:
    resolved = _resolved_stt_config(
        provider="local_qwen_gpu",
        source_language="ja",
        model=None,
        credential_reference=None,
    )

    with pytest.raises(RuntimeError, match="Vulkan ASR worker is not available"):
        wiring_module.create_stt_backend_from_resolved_config(
            resolved,
            secrets=InMemorySecretStore(),
        )


def test_peer_qwen_gpu_auto_omits_hint_while_manual_uses_qwen_code(tmp_path: Path) -> None:
    runtime = object()

    automatic = _resolved_stt_config(
        channel="peer",
        provider="local_qwen_gpu",
        source_language="ja",
        model=None,
        credential_reference=None,
    )
    automatic = replace(automatic, source_mode="auto")
    manual = replace(automatic, source_mode="manual")

    automatic_backend = wiring_module.create_stt_backend_from_resolved_config(
        automatic,
        secrets=InMemorySecretStore(),
        gpu_runtime=runtime,
        gpu_model_path=tmp_path / "model.gguf",
    )
    manual_backend = wiring_module.create_stt_backend_from_resolved_config(
        manual,
        secrets=InMemorySecretStore(),
        gpu_runtime=runtime,
        gpu_model_path=tmp_path / "model.gguf",
    )

    assert automatic_backend.source_mode == "auto"
    assert automatic_backend.language_hint is None
    assert manual_backend.source_mode == "manual"
    assert manual_backend.language_hint == "ja"


def test_create_peer_stt_backend_from_resolved_uses_peer_dto_without_raw_self_settings() -> None:
    resolved = _resolved_stt_config(
        channel="peer",
        provider="deepgram",
        source_language="zh-CN",
        model="dto-peer-model",
        input_host_api=None,
        input_device=None,
        output_device="Steam Streaming Speakers",
    )
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "peer-key")

    backend = wiring_module.create_peer_stt_backend_from_resolved_config(
        resolved,
        secrets=secrets,
    )

    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert backend.api_key == "peer-key"
    assert backend.model == "dto-peer-model"
    assert backend.language == get_deepgram_language("zh-CN")
    assert backend.stream_label == "peer"


def test_resolve_overlay_config_maps_desktop_flet_to_resolved_desktop_options() -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            overlay=replace(
                settings.intent.overlay,
                target="desktop",
                show_translation=False,
                show_peer_original=True,
                calibration=replace(settings.intent.overlay.calibration, distance=2.5),
                desktop_flet=DesktopFletOverlayIntent(
                    size_preset="large",
                    position=DesktopFletOverlayPositionIntent(x=111, y=222),
                    visual=DesktopFletOverlayVisualIntent(background_alpha=0.44),
                ),
            ),
        ),
    )

    resolved = wiring_module.resolve_overlay_config_from_vnext(
        settings,
        enabled=True,
        locked=True,
    )

    assert resolved.enabled is True
    assert resolved.target == "desktop"
    assert resolved.show_translation is False
    assert resolved.show_peer_original is True
    assert resolved.calibration["distance"] == 2.5
    assert resolved.desktop_overlay_options == {
        "size_preset": "large",
        "position": {"x": 111, "y": 222},
        "locked": True,
        "swap_caption_languages": False,
        "visual": {
            "text_scale": 1.0,
            "background_alpha": 0.44,
            "outline_width": None,
        },
    }
    assert "desktop_flet" not in resolved.desktop_overlay_options


def test_create_stt_backend_deepgram_uses_settings_and_secret() -> None:
    settings = _vnext(stt_provider="deepgram", deepgram_model="nova-3")
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "k3")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert backend.api_key == "k3"
    assert backend.model == "nova-3"
    assert backend.sample_rate_hz == 16000
    assert backend.language == get_deepgram_language(settings.intent.languages.source_language)
    assert list(backend.keyterms) == []


def test_create_stt_backend_deepgram_passes_effective_custom_terms() -> None:
    settings = _vnext(
        stt_provider="deepgram",
        deepgram_model="nova-3",
        custom_vocabulary_enabled=True,
        custom_terms={"ko": [" Puripuly ", "", "VRChat", "Puripuly"]},
    )
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "k3")

    backend = create_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert list(backend.keyterms) == ["Puripuly", "VRChat"]


def test_create_stt_backend_local_qwen_uses_shared_model_path_without_secret() -> None:
    settings = _vnext(stt_provider="local_qwen")
    secrets = InMemorySecretStore()

    backend = create_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert backend.model_dir == default_local_stt_model_dir()
    assert backend.sample_rate_hz == 16000
    assert backend.stream_label == "self"


def test_create_stt_backend_local_qwen_passes_diagnostics_enabled_predicate() -> None:
    settings = _vnext(stt_provider="local_qwen")
    secrets = InMemorySecretStore()

    def diagnostics_enabled() -> bool:
        return True

    backend = create_stt_backend(
        settings,
        secrets=secrets,
        diagnostics_enabled=diagnostics_enabled,
    )

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert backend.diagnostics_enabled is diagnostics_enabled


def test_create_stt_backend_local_qwen_passes_language_hint_without_hotwords() -> None:
    settings = _vnext(
        stt_provider="local_qwen",
        source_language="ko-KR",
        custom_vocabulary_enabled=True,
        custom_terms={
            "ko": ["Puripuly", "VRChat, Japan", *[f"term-{i:02d}" for i in range(20)]],
        },
    )
    secrets = InMemorySecretStore()

    backend = create_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert getattr(backend, "language_hint", None) == "ko"
    assert getattr(backend, "hotwords", ()) == ()


def test_create_stt_backend_rejects_invalid_compatibility_provider() -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            stt=replace(settings.intent.stt, provider="corrupt-self-stt-provider"),
        ),
    )

    with pytest.raises(ValueError, match="Unsupported STT provider"):
        resolve_self_stt_runtime_config_from_vnext(settings)


def test_create_peer_stt_backend_uses_dedicated_deepgram_configuration_without_hint_terms() -> None:
    settings = _vnext(
        stt_provider="soniox",
        peer_stt_provider="deepgram",
        deepgram_model="nova-3",
    )
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "peer-k")

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert backend.api_key == "peer-k"
    assert backend.model == "nova-3"
    assert backend.sample_rate_hz == 16000
    assert backend.language == get_deepgram_language(
        settings.intent.languages.effective_peer_source
    )
    assert list(backend.keyterms) == []
    assert backend.stream_label == "peer"


def test_create_peer_stt_backend_uses_effective_peer_source_language_without_hint_terms() -> None:
    settings = _vnext(
        stt_provider="soniox",
        peer_stt_provider="deepgram",
        deepgram_model="nova-3",
        source_language="ko",
        peer_source_language="zh-CN",
    )
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "peer-k")

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert backend.language == get_deepgram_language(
        settings.intent.languages.effective_peer_source
    )
    assert list(backend.keyterms) == []


def test_self_stt_provider_setting_does_not_change_peer_backend_choice() -> None:
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "peer-k")

    soniox_settings = _vnext(stt_provider="soniox", peer_stt_provider="deepgram")
    qwen_settings = _vnext(stt_provider="qwen_asr", peer_stt_provider="deepgram")

    soniox_backend = create_peer_stt_backend(soniox_settings, secrets=secrets)
    qwen_backend = create_peer_stt_backend(qwen_settings, secrets=secrets)

    assert isinstance(soniox_backend, DeepgramRealtimeSTTBackend)
    assert isinstance(qwen_backend, DeepgramRealtimeSTTBackend)


def test_resolve_peer_stt_config_always_uses_self_deepgram_model() -> None:
    settings = _vnext(peer_stt_provider="deepgram", deepgram_model="nova-3-general")

    resolved = resolve_peer_stt_config(settings)

    assert resolved.provider == STTProviderName.DEEPGRAM
    assert resolved.model == "nova-3-general"


def test_resolve_peer_stt_config_exposes_legacy_provider_specific_fields() -> None:
    settings = _vnext(
        peer_stt_provider="soniox",
        peer_source_language="zh-CN",
        soniox_model="stt-rt-v4-peer",
        soniox_endpoint="wss://peer-soniox.example/realtime",
        soniox_keepalive_interval_s=12.5,
        soniox_trailing_silence_ms=700,
    )

    resolved = resolve_peer_stt_config(settings)

    assert isinstance(resolved, ResolvedPeerSTTConfig)
    assert resolved.provider is STTProviderName.SONIOX
    assert resolved.source_language == "zh-CN"
    assert resolved.sample_rate_hz == 16000
    assert resolved.keyterms == ()
    assert resolved.deepgram_model is None
    assert resolved.qwen_model is None
    assert resolved.qwen_region is None
    assert resolved.soniox_model == "stt-rt-v4-peer"
    assert resolved.soniox_endpoint == "wss://peer-soniox.example/realtime"
    assert resolved.soniox_keepalive_interval_s == 12.5
    assert resolved.soniox_trailing_silence_ms == 700


def test_resolve_peer_stt_config_rejects_invalid_compatibility_provider() -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            peer_stt=replace(settings.intent.peer_stt, provider="corrupt-peer-stt-provider"),
        ),
    )

    with pytest.raises(ValueError, match="Unsupported peer STT provider"):
        resolve_peer_stt_config(settings)


def test_create_peer_stt_backend_uses_peer_selected_soniox_provider() -> None:
    settings = _vnext(
        peer_stt_provider="soniox",
        peer_source_language="ko",
        soniox_model="stt-rt-v4",
    )
    secrets = InMemorySecretStore()
    secrets.set("soniox_api_key", "peer-soniox")

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, SonioxRealtimeSTTBackend)
    assert backend.api_key == "peer-soniox"
    assert backend.model == "stt-rt-v4"


def test_create_peer_stt_backend_uses_shared_qwen_region_for_endpoint_and_secret() -> None:
    settings = _vnext(peer_stt_provider="qwen_asr", qwen_region="singapore")
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "peer-qwen")

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.api_key == "peer-qwen"
    assert backend.endpoint == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"


def test_self_stt_runtime_signature_from_vnext_matches_bag_restart_fields() -> None:
    cases = (
        _vnext(
            stt_provider="deepgram",
            custom_vocabulary_enabled=True,
            custom_terms={"ko": [" Puripuly ", "a,b", "VRChat"]},
        ),
        _vnext(
            stt_provider="local_qwen",
            custom_vocabulary_enabled=True,
            custom_terms={"ko": [f"term{i}, extra" for i in range(20)]},
        ),
        _vnext(
            stt_provider="local_qwen_gpu",
            custom_vocabulary_enabled=True,
            custom_terms={"ko": ["gpu-term"]},
        ),
        _vnext(
            stt_provider="soniox",
            custom_vocabulary_enabled=True,
            custom_terms={"ko": ["soniox-term"]},
        ),
        _vnext(stt_provider="qwen_asr", qwen_region="beijing"),
        _vnext(stt_provider="qwen_asr", qwen_region="singapore"),
        _vnext(stt_provider="custom"),
    )
    for settings in cases:
        canonical = settings
        assert build_self_stt_runtime_signature_from_vnext(
            canonical
        ) == build_self_stt_runtime_signature(canonical)


def test_build_peer_stt_provider_signature_includes_backend_affecting_values() -> None:
    settings = _vnext(
        peer_stt_provider="soniox",
        peer_source_language="zh-CN",
        soniox_model="stt-rt-v4",
        soniox_trailing_silence_ms=350,
    )

    signature = build_peer_stt_provider_signature(settings)

    assert STTProviderName.SONIOX in signature
    assert "zh-CN" in signature
    assert "stt-rt-v4" in signature
    assert 350 in signature


def test_peer_auto_detection_keeps_self_language_restriction_separate() -> None:
    settings = _vnext(
        stt_provider="soniox",
        peer_stt_provider="soniox",
        peer_source_mode="auto",
        peer_expected_languages=["ja", "zh-TW"],
    )

    peer = resolve_peer_stt_runtime_config(settings)
    self_config = resolve_stt_config(self_stt_runtime_intent_from_vnext(settings))

    assert peer.provider_options["enable_language_identification"] is True
    assert peer.provider_options["language_hints"] == ("ja", "zh")
    assert "enable_language_identification" not in self_config.provider_options
    assert self_config.provider_options["language_hints"] == ("ko",)
    assert self_config.provider_options["language_hints_strict"] is True
    assert "language_hints_strict" not in peer.provider_options


def test_peer_auto_detection_falls_back_to_manual_configuration_for_other_providers() -> None:
    settings = _vnext(
        peer_stt_provider="deepgram",
        peer_source_mode="auto",
        peer_expected_languages=["ja"],
    )

    peer = resolve_peer_stt_runtime_config(settings)

    assert peer.source_language == settings.intent.languages.effective_peer_source
    assert peer.provider_options == {}


def test_vnext_peer_runtime_resolution_and_signature_use_canonical_auto_intent() -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            peer_stt=replace(settings.intent.peer_stt, provider="soniox"),
            languages=replace(
                settings.intent.languages,
                peer_source_mode="auto",
                peer_expected_languages=["ja", "zh-TW"],
            ),
        ),
    )

    automatic = resolve_peer_stt_runtime_config_from_vnext(settings)
    automatic_signature = build_peer_stt_provider_signature_from_vnext(settings)
    manual_settings = replace(
        settings,
        intent=replace(
            settings.intent,
            languages=replace(settings.intent.languages, peer_source_mode="manual"),
        ),
    )

    manual = resolve_peer_stt_runtime_config_from_vnext(manual_settings)

    assert automatic.provider_options["enable_language_identification"] is True
    assert automatic.provider_options["language_hints"] == ("ja", "zh")
    assert automatic_signature != build_peer_stt_provider_signature_from_vnext(manual_settings)
    assert "enable_language_identification" not in manual.provider_options
    assert manual.provider_options["language_hints"] == ("en",)
    assert "language_hints_strict" not in manual.provider_options
    assert "language_hints_strict" not in automatic.provider_options


def test_vnext_peer_auto_without_expected_languages_omits_hints() -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            peer_stt=replace(settings.intent.peer_stt, provider="soniox"),
            languages=replace(
                settings.intent.languages,
                peer_source_mode="auto",
                peer_expected_languages=[],
            ),
        ),
    )

    automatic = resolve_peer_stt_runtime_config_from_vnext(settings)

    assert automatic.provider_options["enable_language_identification"] is True
    assert "language_hints" not in automatic.provider_options
    assert "language_hints_strict" not in automatic.provider_options


def test_vnext_peer_runtime_keeps_self_and_non_soniox_paths_manual() -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            stt=replace(settings.intent.stt, provider="soniox"),
            peer_stt=replace(settings.intent.peer_stt, provider="deepgram"),
            languages=replace(
                settings.intent.languages,
                peer_source_mode="auto",
                peer_expected_languages=["ja"],
            ),
        ),
    )

    peer = resolve_peer_stt_runtime_config_from_vnext(settings)
    self_config = resolve_stt_config(self_stt_runtime_intent_from_vnext(AppSettingsVNext()))

    assert peer.provider == "deepgram"
    assert peer.provider_options == {}
    assert self_config.provider_options.get("enable_language_identification") is None


def test_self_soniox_is_strict_while_manual_peer_uses_a_soft_hint() -> None:
    settings = _vnext(stt_provider="soniox", peer_stt_provider="soniox")
    secrets = InMemorySecretStore()
    secrets.set("soniox_api_key", "soniox-key")

    self_backend = create_stt_backend(settings, secrets=secrets)
    peer_backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(self_backend, SonioxRealtimeSTTBackend)
    assert isinstance(peer_backend, SonioxRealtimeSTTBackend)
    assert self_backend.language_hints == ["ko"]
    assert self_backend.language_hints_strict is True
    assert peer_backend.language_hints == ["en"]
    assert peer_backend.language_hints_strict is False


def test_build_peer_stt_provider_signature_uses_fixed_16khz_runtime_contract() -> None:
    settings = _vnext(peer_stt_provider="qwen_asr")

    signature = build_peer_stt_provider_signature(settings)

    assert signature[2] == 16000


def test_resolve_peer_stt_config_uses_provider_owned_qwen_model() -> None:
    settings = _vnext(peer_stt_provider="qwen_asr", qwen_asr_model="self-qwen-asr")

    resolved = resolve_peer_stt_config(settings)

    assert resolved.model == "qwen3-asr-flash-realtime"


def test_mixed_qwen_cloud_providers_resolve_independent_models() -> None:
    settings = _vnext(
        stt_provider="qwen_asr",
        peer_stt_provider="qwen_audio",
        qwen_asr_model="self-qwen-asr",
    )

    self_intent = self_stt_runtime_intent_from_vnext(settings)
    peer_intent = peer_stt_runtime_intent_from_vnext(settings)

    assert self_intent.provider == "qwen_asr"
    assert self_intent.qwen_asr_model == "qwen3-asr-flash-realtime"
    assert peer_intent.provider == "qwen_asr"
    assert peer_intent.qwen_asr_model == "qwen-audio-3.0-asr-flash-streaming"


def test_qwen_audio_auto_mode_survives_peer_runtime_normalization() -> None:
    settings = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="auto",
        peer_expected_languages=["ja", "zh-TW"],
    )

    intent = peer_stt_runtime_intent_from_vnext(settings)

    assert intent.provider == "qwen_asr"
    assert intent.qwen_asr_model == "qwen-audio-3.0-asr-flash-streaming"
    assert intent.source_mode == "auto"
    assert intent.qwen_audio_language_hints == ("ja", "zh")


def test_non_audio_qwen_asr_runtime_does_not_inherit_auto_detection() -> None:
    settings = _vnext(
        peer_stt_provider="qwen_asr",
        qwen_asr_model="qwen3-asr-flash-realtime",
        peer_source_mode="auto",
        peer_expected_languages=["ja"],
    )

    intent = peer_stt_runtime_intent_from_vnext(settings)

    assert intent.source_mode == "manual"
    assert intent.qwen_audio_language_hints is None
    assert intent.soniox_language_hints is None


def test_qwen_audio_manual_mode_keeps_single_hint_contract() -> None:
    settings = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="manual",
        peer_source_language="ja",
        peer_expected_languages=["ko", "ja"],
    )

    resolved = resolve_peer_stt_runtime_config(settings)

    assert resolved.source_mode == "manual"
    assert resolved.provider_options == {}


def test_qwen_audio_auto_without_expected_languages_omits_hints() -> None:
    settings = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="auto",
        peer_expected_languages=[],
    )

    resolved = resolve_peer_stt_runtime_config(settings)

    assert resolved.source_mode == "auto"
    assert resolved.provider_options["language_hints"] == ()


def test_qwen_audio_auto_sends_ordered_mapped_deduped_hints() -> None:
    settings = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="auto",
        peer_expected_languages=["ko", "ja", "zh-CN", "zh-TW", "en"],
    )

    resolved = resolve_peer_stt_runtime_config(settings)

    assert resolved.source_mode == "auto"
    assert resolved.provider_options["language_hints"] == ("ko", "ja", "zh", "en")


def test_gemini_transcribe_auto_sends_expected_language_codes_up_to_32() -> None:
    settings = _vnext(
        peer_stt_provider="gemini_transcribe",
        peer_source_mode="auto",
        peer_expected_languages=["ko", "ja", "zh-TW", "en", "xx"],
    )

    resolved = resolve_peer_stt_runtime_config(settings)

    assert resolved.source_mode == "auto"
    assert resolved.provider_options["language_codes"] == ("ko", "ja", "zh-TW", "en")
    assert resolved.provider_options["auto_language"] is True


def test_gemini_transcribe_auto_without_expected_languages_omits_codes() -> None:
    settings = _vnext(
        peer_stt_provider="gemini_transcribe",
        peer_source_mode="auto",
        peer_expected_languages=[],
    )

    resolved = resolve_peer_stt_runtime_config(settings)

    assert resolved.source_mode == "auto"
    assert resolved.provider_options["language_codes"] == ()
    assert resolved.provider_options["auto_language"] is True


def test_qwen_audio_auto_drops_unsupported_expected_languages() -> None:
    settings = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="auto",
        peer_expected_languages=["ko", "xx", "et", "ja"],
    )

    resolved = resolve_peer_stt_runtime_config(settings)

    assert resolved.provider_options["language_hints"] == ("ko", "ja")


def test_qwen_audio_provider_handoff_caps_hints_at_four() -> None:
    from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend

    settings = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="auto",
        peer_expected_languages=["ko", "ja", "zh-CN", "zh-TW", "en", "fr"],
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "k-audio")

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, QwenAudioStreamingSTTBackend)
    assert backend.language_hints == ("ko", "ja", "zh", "en")


def test_qwen_audio_manual_handoff_sends_single_mapped_hint() -> None:
    from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend

    settings = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="manual",
        peer_source_language="ja",
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "k-audio")

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, QwenAudioStreamingSTTBackend)
    assert backend.language_hints == ("ja",)


def test_peer_expected_languages_remain_untruncated_for_qwen_audio() -> None:
    settings = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="auto",
        peer_expected_languages=["ko", "ja", "zh-CN", "zh-TW", "en", "fr"],
    )

    _ = resolve_peer_stt_runtime_config(settings)

    assert settings.intent.languages.peer_expected_languages == [
        "ko",
        "ja",
        "zh-CN",
        "zh-TW",
        "en",
        "fr",
    ]


def test_peer_stt_provider_signature_tracks_qwen_audio_language_hints() -> None:
    auto_settings = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="auto",
        peer_expected_languages=["ko", "ja"],
    )
    other_hints = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="auto",
        peer_expected_languages=["ko", "en"],
    )
    manual_settings = _vnext(
        peer_stt_provider="qwen_audio",
        peer_source_mode="manual",
        peer_source_language="ja",
    )

    assert build_peer_stt_provider_signature_from_vnext(
        auto_settings
    ) != build_peer_stt_provider_signature_from_vnext(other_hints)
    assert build_peer_stt_provider_signature_from_vnext(
        auto_settings
    ) != build_peer_stt_provider_signature_from_vnext(manual_settings)


def test_create_peer_stt_backend_uses_peer_local_qwen_provider_and_fixed_sample_rate() -> None:
    settings = _vnext(peer_stt_provider="local_qwen")
    secrets = InMemorySecretStore()

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert backend.model_dir == default_local_stt_model_dir()
    assert backend.sample_rate_hz == 16000
    assert backend.stream_label == "peer"


def test_create_peer_stt_backend_local_qwen_passes_diagnostics_enabled_predicate() -> None:
    settings = _vnext(peer_stt_provider="local_qwen")
    secrets = InMemorySecretStore()

    def diagnostics_enabled() -> bool:
        return True

    backend = create_peer_stt_backend(
        settings,
        secrets=secrets,
        diagnostics_enabled=diagnostics_enabled,
    )

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert backend.diagnostics_enabled is diagnostics_enabled


def test_managed_stt_provider_rejects_legacy_8khz_runtime_sample_rate() -> None:
    with pytest.raises(ValueError, match="16000"):
        ManagedSTTProvider(backend=None, sample_rate_hz=8000)  # type: ignore[arg-type]


def test_create_peer_stt_backend_local_qwen_uses_peer_language_without_hotwords() -> None:
    settings = _vnext(
        peer_stt_provider="local_qwen",
        source_language="ko",
        peer_source_language="zh-CN",
        custom_vocabulary_enabled=True,
        custom_terms={
            "zh-CN": ["airi", "shinano", *[f"term-{i:02d}" for i in range(20)]],
        },
    )
    secrets = InMemorySecretStore()

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert getattr(backend, "language_hint", None) == "zh"
    assert getattr(backend, "hotwords", ()) == ()


def test_resolve_peer_stt_config_uses_shared_soniox_endpoint_keepalive_and_trailing_silence() -> (
    None
):
    settings = _vnext(
        peer_stt_provider="soniox",
        soniox_model="self-soniox",
        soniox_endpoint="wss://self-soniox.example/realtime",
        soniox_keepalive_interval_s=12.5,
        soniox_trailing_silence_ms=900,
    )

    resolved = resolve_peer_stt_config(settings)

    assert resolved.model == "self-soniox"
    assert resolved.endpoint == "wss://self-soniox.example/realtime"
    assert resolved.provider_options["keepalive_interval_s"] == 12.5
    assert resolved.provider_options["trailing_silence_ms"] == 900


def test_create_stt_backend_qwen_asr_uses_settings_and_secret() -> None:
    settings = _vnext(stt_provider="qwen_asr")
    secrets = InMemorySecretStore()
    # Default region is Beijing, so we need alibaba_api_key_beijing
    secrets.set("alibaba_api_key_beijing", "k4")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.api_key == "k4"
    assert backend.model == "qwen3-asr-flash-realtime"
    # Endpoint is derived from region (Beijing default)
    assert backend.endpoint == "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    assert backend.sample_rate_hz == 16000
    assert backend.language == get_qwen_asr_language(settings.intent.languages.source_language)


def test_create_stt_backend_qwen_asr_ignores_custom_terms() -> None:
    settings = _vnext(
        stt_provider="qwen_asr",
        custom_vocabulary_enabled=True,
        custom_terms={"ko": ["Puripuly", "VRChat"]},
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "k4")

    backend = create_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.api_key == "k4"
    assert backend.model == "qwen3-asr-flash-realtime"
    assert backend.language == get_qwen_asr_language(settings.intent.languages.source_language)
    assert not hasattr(backend, "keyterms")
    assert not hasattr(backend, "context_terms")


def test_create_stt_backend_qwen_asr_uses_singapore_region() -> None:
    settings = _vnext(stt_provider="qwen_asr", qwen_region="singapore")
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "k5")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.endpoint == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"


def test_create_stt_backend_qwen_asr_uses_legacy_alibaba_secret_key() -> None:
    settings = _vnext(stt_provider="qwen_asr")
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key", "legacy-k4")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.api_key == "legacy-k4"
    # Legacy key should be backfilled to region-specific key for future runs.
    assert secrets.get("alibaba_api_key_beijing") == "legacy-k4"


def test_create_stt_backend_soniox_uses_secret() -> None:
    settings = _vnext(stt_provider="soniox")
    secrets = InMemorySecretStore()
    secrets.set("soniox_api_key", "k6")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, SonioxRealtimeSTTBackend)
    assert backend.api_key == "k6"
    assert list(backend.context_terms) == []


def test_create_stt_backend_soniox_passes_effective_custom_terms() -> None:
    settings = _vnext(
        stt_provider="soniox",
        custom_vocabulary_enabled=True,
        custom_terms={"ko": [" Puripuly ", "VRChat", "Puripuly", " "]},
    )
    secrets = InMemorySecretStore()
    secrets.set("soniox_api_key", "k6")

    backend = create_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, SonioxRealtimeSTTBackend)
    assert list(backend.context_terms) == ["Puripuly", "VRChat"]
