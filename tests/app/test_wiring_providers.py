from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from puripuly_heart.app import wiring as wiring_module
from puripuly_heart.app.wiring import (
    ManagedIdentityStateAdapter,
    ResolvedPeerSTTConfig,
    _LazyFactoryLLMProvider,
    build_openrouter_release_runtime_config,
    build_peer_stt_provider_signature,
    build_peer_stt_provider_signature_from_vnext,
    create_llm_provider,
    create_llm_provider_from_resolved_config,
    create_peer_stt_backend,
    create_stt_backend,
    resolve_peer_stt_config,
    resolve_peer_stt_runtime_config_from_vnext,
)
from puripuly_heart.app.wiring import root as wiring_root
from puripuly_heart.app.wiring.wiring_stt_factory import (
    _self_stt_runtime_intent_from_compatibility_settings,
    resolve_peer_stt_runtime_config,
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
from puripuly_heart.config.runtime_resolution import resolve_stt_config
from puripuly_heart.config.settings import (
    AppSettings,
    CerebrasLLMModel,
    CerebrasSettings,
    DeepgramSTTSettings,
    DeepSeekLLMModel,
    DeepSeekSettings,
    GeminiLLMModel,
    GeminiSettings,
    LLMProviderName,
    LLMSettings,
    OpenRouterCredentialSource,
    OpenRouterFallbackSelectionAlias,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterRoutingMode,
    OpenRouterSelectionAlias,
    OpenRouterSettings,
    ProviderSettings,
    QwenASRSTTSettings,
    QwenLLMModel,
    QwenRegion,
    QwenSettings,
    SonioxSTTSettings,
    STTProviderName,
    STTSettings,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
    TranslationSettings,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.language import (
    get_deepgram_language,
    get_qwen_asr_language,
)
from puripuly_heart.core.llm import FallbackRacingLLMProvider
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.local_asr.local_stt_assets import default_local_stt_model_dir
from puripuly_heart.core.openrouter.managed_openrouter_release import (
    ManagedOpenRouterLLMProvider,
    ManagedOpenRouterReleaseService,
    _resolve_managed_issue_model,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore
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


def _translation_with_fallback(
    *,
    model: TranslationModel,
    connection: TranslationConnection,
) -> TranslationSettings:
    return TranslationSettings(
        fallback=TranslationFallbackSettings(
            enabled=True,
            model=model,
            connection=connection,
        )
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.GEMINI),
        llm=LLMSettings(concurrency_limit=3),
    )
    secrets = InMemorySecretStore()
    secrets.set("google_api_key", "k")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, GeminiLLMProvider)
    assert provider.inner.api_key == "k"
    assert provider.inner.model == "gemini-3.1-flash-lite"
    assert provider.semaphore._value == 3  # type: ignore[attr-defined]


def test_create_llm_provider_gemini_uses_selected_model() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.GEMINI),
        gemini=GeminiSettings(llm_model=GeminiLLMModel.GEMINI_31_FLASH_LITE),
    )
    secrets = InMemorySecretStore()
    secrets.set("google_api_key", "k")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, GeminiLLMProvider)
    assert provider.inner.model == "gemini-3.1-flash-lite"


def test_create_llm_provider_gemini_passes_runtime_logging() -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.GEMINI))
    secrets = InMemorySecretStore()
    secrets.set("google_api_key", "k")
    runtime_logging = object()

    provider = create_llm_provider(settings, secrets=secrets, runtime_logging=runtime_logging)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, GeminiLLMProvider)
    assert provider.inner.runtime_logging is runtime_logging


def test_create_llm_provider_qwen_uses_secret() -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.QWEN))
    secrets = InMemorySecretStore()
    # Default region is Beijing, so we need alibaba_api_key_beijing
    secrets.set("alibaba_api_key_beijing", "k2")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.api_key == "k2"
    assert provider.inner.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert provider.inner.model == "qwen3.5-plus"
    assert provider.semaphore._value == 5  # type: ignore[attr-defined]


def test_create_llm_provider_qwen_low_latency_passes_runtime_logging() -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.QWEN))
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "k2")
    runtime_logging = object()

    provider = create_llm_provider(settings, secrets=secrets, runtime_logging=runtime_logging)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.runtime_logging is runtime_logging


def test_create_llm_provider_qwen_uses_singapore_region() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.QWEN),
        qwen=QwenSettings(region=QwenRegion.SINGAPORE, llm_model=QwenLLMModel.QWEN_35_PLUS),
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "k3")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.api_key == "k3"
    assert provider.inner.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert provider.inner.model == "qwen3.5-plus"


def test_create_llm_provider_qwen_uses_legacy_alibaba_secret_key() -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.QWEN))
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key", "legacy-k2")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.api_key == "legacy-k2"
    # Legacy key should be backfilled to region-specific key for future runs.
    assert secrets.get("alibaba_api_key_beijing") == "legacy-k2"


def test_create_llm_provider_qwen_historical_false_still_uses_async_provider() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.QWEN),
        stt=STTSettings(low_latency_mode=False),
        qwen=QwenSettings(llm_model=QwenLLMModel.QWEN_35_PLUS),
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "k2")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.api_key == "k2"
    assert provider.inner.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert provider.inner.model == "qwen3.5-plus"


def test_create_llm_provider_qwen_historical_false_passes_runtime_logging() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.QWEN),
        stt=STTSettings(low_latency_mode=False),
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "k2")
    runtime_logging = object()

    provider = create_llm_provider(settings, secrets=secrets, runtime_logging=runtime_logging)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, AsyncQwenLLMProvider)
    assert provider.inner.runtime_logging is runtime_logging


def test_create_llm_provider_qwen_historical_false_uses_async_singapore() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.QWEN),
        qwen=QwenSettings(region=QwenRegion.SINGAPORE, llm_model=QwenLLMModel.QWEN_35_FLASH),
        stt=STTSettings(low_latency_mode=False),
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.DEEPSEEK),
        llm=LLMSettings(concurrency_limit=4),
    )
    secrets = InMemorySecretStore()
    secrets.set("deepseek_api_key", "ds-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, DeepSeekLLMProvider)
    assert provider.inner.api_key == "ds-key"
    assert provider.inner.model == "deepseek-v4-flash"
    assert provider.inner.base_url == "https://api.deepseek.com"
    assert provider.semaphore._value == 4  # type: ignore[attr-defined]


def test_create_llm_provider_deepseek_uses_v4_flash_model() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.DEEPSEEK),
    )
    settings.deepseek.llm_model = DeepSeekLLMModel.DEEPSEEK_V4_FLASH
    secrets = InMemorySecretStore()
    secrets.set("deepseek_api_key", "ds-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, DeepSeekLLMProvider)
    assert provider.inner.model == "deepseek-v4-flash"


def test_create_llm_provider_deepseek_passes_runtime_logging() -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.DEEPSEEK))
    secrets = InMemorySecretStore()
    secrets.set("deepseek_api_key", "ds-key")
    runtime_logging = object()

    provider = create_llm_provider(settings, secrets=secrets, runtime_logging=runtime_logging)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, DeepSeekLLMProvider)
    assert provider.inner.runtime_logging is runtime_logging


def test_create_llm_provider_cerebras_uses_secret_and_model() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.CEREBRAS),
        cerebras=CerebrasSettings(llm_model=CerebrasLLMModel.GEMMA_4_31B),
        llm=LLMSettings(concurrency_limit=6),
    )
    secrets = InMemorySecretStore()
    secrets.set("cerebras_api_key", "cerebras-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, CerebrasLLMProvider)
    assert provider.inner.api_key == "cerebras-key"
    assert provider.inner.model == "gemma-4-31b"
    assert provider.semaphore._value == 6  # type: ignore[attr-defined]


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
    assert provider.semaphore._value == 2  # type: ignore[attr-defined]


def test_create_llm_provider_local_llm_uses_settings_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    settings.local_llm.base_url = "http://127.0.0.1:11434/v1"
    settings.local_llm.model = "llama3.1:8b"
    settings.local_llm.extra_body = {"think": False}
    settings.llm.concurrency_limit = 2
    secrets = InMemorySecretStore()
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, LocalOpenAICompatibleLLMProvider)
    assert provider.inner.base_url == "http://127.0.0.1:11434/v1"
    assert provider.inner.model == "llama3.1:8b"
    assert provider.inner.api_key == ""
    assert provider.inner.extra_body == {"think": False}
    assert provider.semaphore._value == 2  # type: ignore[attr-defined]


def test_create_llm_provider_local_llm_ignores_optional_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    secrets = InMemorySecretStore()
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-secret")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, LocalOpenAICompatibleLLMProvider)
    assert provider.inner.api_key == ""


def test_create_llm_provider_local_llm_uses_secret_store_key_even_when_env_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    secrets = InMemorySecretStore()
    secrets.set("local_llm_api_key", "store-secret")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "env-secret")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, LocalOpenAICompatibleLLMProvider)
    assert provider.inner.api_key == "store-secret"


def test_create_llm_provider_from_resolved_local_llm_uses_dto_values_and_optional_secret() -> None:
    legacy_settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    legacy_settings.local_llm.base_url = "http://legacy.local/v1"
    legacy_settings.local_llm.model = "legacy-model"
    legacy_settings.local_llm.extra_body = {"legacy": True}
    legacy_settings.llm.concurrency_limit = 1
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
        compatibility_settings=legacy_settings,
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, LocalOpenAICompatibleLLMProvider)
    assert provider.inner.base_url == "http://dto.local/v1"
    assert provider.inner.model == "dto-model"
    assert provider.inner.api_key == "dto-local-secret"
    assert provider.inner.extra_body == {"think": False}
    assert provider.semaphore._value == 7  # type: ignore[attr-defined]


def test_create_llm_provider_legacy_facade_uses_runtime_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.GEMINI),
        gemini=GeminiSettings(llm_model=GeminiLLMModel.GEMINI_37_FLASH),
        llm=LLMSettings(concurrency_limit=2),
    )
    settings.local_llm.base_url = "http://legacy.local/v1"
    settings.local_llm.model = "legacy-local-model"
    settings.local_llm.extra_body = {"legacy": True}
    resolved = ResolvedLLMConfig(
        primary=ResolvedLLMTarget(
            provider="local_llm",
            model="resolved-local-model",
            credential=ResolvedCredentialRequirement(
                source=CREDENTIAL_SOURCE_NONE,
                required=False,
                reference=None,
            ),
            base_url="http://resolved.local/v1",
            provider_options={"extra_body": {"resolved": True}},
        ),
        concurrency_limit=9,
    )
    captured_inputs: list[object] = []

    def fake_resolve_llm_config(runtime_input: object) -> ResolvedLLMConfig:
        captured_inputs.append(runtime_input)
        return resolved

    monkeypatch.setattr(
        wiring_root,
        "resolve_llm_config",
        fake_resolve_llm_config,
        raising=False,
    )
    secrets = InMemorySecretStore()
    secrets.set("google_api_key", "raw-gemini-key")
    secrets.set("local_llm_api_key", "resolved-local-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert captured_inputs, "legacy facade must call runtime resolution"
    runtime_input = captured_inputs[0]
    assert runtime_input.direct.local_llm_base_url == "http://legacy.local/v1"  # type: ignore[attr-defined]
    assert runtime_input.direct.local_llm_model == "legacy-local-model"  # type: ignore[attr-defined]
    assert runtime_input.translation.concurrency_limit == 2  # type: ignore[attr-defined]
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, LocalOpenAICompatibleLLMProvider)
    assert provider.inner.base_url == "http://resolved.local/v1"
    assert provider.inner.model == "resolved-local-model"
    assert provider.inner.api_key == "resolved-local-key"
    assert provider.inner.extra_body == {"resolved": True}
    assert provider.semaphore._value == 9  # type: ignore[attr-defined]


def test_create_llm_provider_openrouter_uses_secret_and_model() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        llm=LLMSettings(concurrency_limit=4),
        openrouter=OpenRouterSettings(
            llm_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            routing_mode=OpenRouterRoutingMode.LATENCY,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
            selected_source=OpenRouterCredentialSource.BYOK,
        ),
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
    assert provider.semaphore._value == 4  # type: ignore[attr-defined]


def test_create_llm_provider_from_resolved_openrouter_gemini_byok_uses_google_latency_routing() -> (
    None
):
    resolved = ResolvedLLMConfig(
        primary=ResolvedLLMTarget(
            provider="openrouter",
            model=OpenRouterLLMModel.GEMINI_31_FLASH_LITE.value,
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
    assert provider.inner.model == OpenRouterLLMModel.GEMINI_31_FLASH_LITE.value
    assert provider.inner.routing_mode == OpenRouterRoutingMode.LATENCY
    assert provider.inner.provider_routing == OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY
    assert provider.semaphore._value == 3  # type: ignore[attr-defined]


def test_create_llm_provider_openrouter_byok_still_uses_user_owned_secret_after_pkce_storage() -> (
    None
):
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.BYOK,
            selection_alias=OpenRouterSelectionAlias.GEMMA4_BYOK,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "pkce-user-key")
    secrets.set("openrouter_managed_api_key", "managed-key")

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "pkce-user-key"


def test_create_llm_provider_openrouter_qwen_byok_alias_uses_resolved_qwen_model() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.BYOK,
            selection_alias=OpenRouterSelectionAlias.QWEN35_FLASH_BYOK,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
            routing_mode=OpenRouterRoutingMode.LATENCY,
        ),
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.BYOK,
            selection_alias=OpenRouterSelectionAlias.QWEN35_FLASH_BYOK,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
            provider_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY,
        ),
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.BYOK,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.BYOK,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
    )
    secrets = InMemorySecretStore()

    provider = create_llm_provider(settings, secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "env-or-key"


def test_create_llm_provider_openrouter_uses_selected_managed_key() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.MANAGED,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            llm_model=OpenRouterLLMModel.DEEPSEEK_V4_FLASH,
            selected_source=OpenRouterCredentialSource.MANAGED,
            selection_alias=OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
            provider_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY,
        ),
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_managed_qq_api_key", "managed-qq-key")
    settings.managed_identity.active_managed_credential_ref = "managed-ref-qq"

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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            llm_model=OpenRouterLLMModel.DEEPSEEK_V4_FLASH,
            selected_source=OpenRouterCredentialSource.BYOK,
            selection_alias=OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
            provider_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY,
        ),
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.DEEPSEEK),
        deepseek=DeepSeekSettings(llm_model=DeepSeekLLMModel.DEEPSEEK_V4_FLASH),
        translation=_translation_with_fallback(
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OFFICIAL_BYOK,
        ),
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            llm_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            selected_source=OpenRouterCredentialSource.BYOK,
            selection_alias=OpenRouterSelectionAlias.GEMMA4_BYOK,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
            provider_routing=OpenRouterProviderRouting.DEFAULT,
        ),
        translation=_translation_with_fallback(
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.MANAGED_CHINA,
        ),
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
    assert provider.semaphore._value == 3  # type: ignore[attr-defined]


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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.MANAGED,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
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
        wiring_root,
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.MANAGED,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "byok-key")
    secrets.set("openrouter_managed_api_key", "managed-key")

    with pytest.raises(ValueError, match="managed release service"):
        create_llm_provider(settings, secrets=secrets)


def test_create_llm_provider_openrouter_uses_managed_wrapper_when_release_service_is_available() -> (
    None
):
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.MANAGED,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.MANAGED,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
    )
    secrets = InMemorySecretStore()
    current_user_identifier: str | None = None
    load_calls = 0

    def fake_load_managed_openrouter_user_identifier(
        loaded_settings: AppSettings,
        *,
        secrets: InMemorySecretStore,
    ) -> str | None:
        nonlocal load_calls
        _ = loaded_settings, secrets
        load_calls += 1
        return current_user_identifier

    monkeypatch.setattr(
        wiring_root,
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            llm_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            selected_source=OpenRouterCredentialSource.BYOK,
            routing_mode=OpenRouterRoutingMode.LATENCY,
            selection_alias=OpenRouterSelectionAlias.GEMMA4_BYOK,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
        translation=_translation_with_fallback(
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OPENROUTER,
        ),
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
    assert provider.inner.fallback._delegate is None

    fallback_delegate = provider.inner.fallback.factory()

    assert isinstance(fallback_delegate, OpenRouterLLMProvider)
    assert fallback_delegate.api_key == "or-key"
    assert fallback_delegate.model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    assert fallback_delegate.routing_mode == OpenRouterRoutingMode.LATENCY
    assert fallback_delegate.runtime_logging is runtime_logging


def test_create_llm_provider_openrouter_byok_paths_omit_managed_user_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            llm_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            selected_source=OpenRouterCredentialSource.BYOK,
            routing_mode=OpenRouterRoutingMode.LATENCY,
            selection_alias=OpenRouterSelectionAlias.GEMMA4_BYOK,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
        translation=_translation_with_fallback(
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OPENROUTER,
        ),
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "or-key")

    def unexpected_load_managed_openrouter_user_identifier(
        loaded_settings: AppSettings,
        *,
        secrets: InMemorySecretStore,
    ) -> str:
        _ = loaded_settings, secrets
        raise AssertionError("managed user identifier should not be loaded for BYOK paths")

    monkeypatch.setattr(
        wiring_root,
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            llm_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            selected_source=OpenRouterCredentialSource.MANAGED,
            routing_mode=OpenRouterRoutingMode.LATENCY,
            selection_alias=OpenRouterSelectionAlias.GEMMA4_MANAGED,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.QWEN35_FLASH,
        ),
    )
    secrets = InMemorySecretStore()
    managed_release_service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(settings, lambda _updated: None),
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
    assert settings.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_MANAGED
    assert settings.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT


def test_create_llm_provider_openrouter_managed_deepseek_fallback_uses_fallback_specific_release_service() -> (
    None
):
    deepseek_model = getattr(OpenRouterLLMModel, "DEEPSEEK_V4_FLASH", None)

    assert deepseek_model is not None

    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            llm_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            selected_source=OpenRouterCredentialSource.MANAGED,
            routing_mode=OpenRouterRoutingMode.LATENCY,
            selection_alias=OpenRouterSelectionAlias.GEMMA4_MANAGED,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
        translation=_translation_with_fallback(
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.MANAGED,
        ),
    )
    secrets = InMemorySecretStore()
    managed_release_service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(settings, lambda _updated: None),
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
    assert fallback_release_service.managed_state is not managed_release_service.managed_state
    assert fallback_release_service.openrouter_config.selection_alias is None
    assert fallback_release_service.openrouter_config.llm_model == deepseek_model
    assert (
        _resolve_managed_issue_model(fallback_release_service.openrouter_config)
        == deepseek_model.value
    )
    assert settings.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_MANAGED
    assert settings.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT

    fallback_openrouter_delegate = fallback_delegate.delegate_factory("managed-key")

    assert isinstance(fallback_openrouter_delegate, OpenRouterLLMProvider)
    assert fallback_openrouter_delegate.model == deepseek_model.value
    assert fallback_openrouter_delegate.routing_mode == OpenRouterRoutingMode.LATENCY


def test_create_llm_provider_openrouter_managed_fallback_delegate_factory_loads_user_identifier_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.MANAGED,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
        translation=_translation_with_fallback(
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.MANAGED,
        ),
    )
    secrets = InMemorySecretStore()
    current_user_identifier: str | None = None
    load_calls = 0

    def fake_load_managed_openrouter_user_identifier(
        loaded_settings: AppSettings,
        *,
        secrets: InMemorySecretStore,
    ) -> str | None:
        nonlocal load_calls
        _ = loaded_settings, secrets
        load_calls += 1
        return current_user_identifier

    monkeypatch.setattr(
        wiring_root,
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
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            llm_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            selected_source=OpenRouterCredentialSource.MANAGED,
            routing_mode=OpenRouterRoutingMode.LATENCY,
            selection_alias=OpenRouterSelectionAlias.GEMMA4_MANAGED,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
        translation=_translation_with_fallback(
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.MANAGED,
        ),
    )
    secrets = InMemorySecretStore()
    managed_release_service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(settings, lambda _updated: None),
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
    assert settings.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_MANAGED
    assert settings.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT

    fallback_openrouter_delegate = fallback_delegate.delegate_factory("managed-key")

    assert isinstance(fallback_openrouter_delegate, OpenRouterLLMProvider)
    assert fallback_openrouter_delegate.model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    assert fallback_openrouter_delegate.routing_mode == OpenRouterRoutingMode.LATENCY


def test_create_llm_provider_openrouter_rejects_none_selected_source_even_with_keys() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.NONE,
            selection_alias=None,
        ),
    )
    secrets = InMemorySecretStore()
    secrets.set("openrouter_api_key", "byok-key")
    secrets.set("openrouter_managed_api_key", "managed-key")

    with pytest.raises(ValueError, match="selected source"):
        create_llm_provider(settings, secrets=secrets)


def test_create_llm_provider_requires_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.GEMINI))
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
    settings = AppSettings()
    settings.ui.overlay_enabled = True
    settings.overlay.target = "desktop"
    settings.overlay.show_translation = False
    settings.overlay.show_peer_original = True
    settings.overlay.calibration.distance = 2.5
    settings.overlay.desktop_flet.size_preset = "large"
    settings.overlay.desktop_flet.position.x = 111
    settings.overlay.desktop_flet.position.y = 222
    settings.overlay.desktop_flet.locked = True
    settings.overlay.desktop_flet.visual.background_alpha = 0.44

    resolved = wiring_module.resolve_overlay_config(settings)

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
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.DEEPGRAM),
        deepgram_stt=DeepgramSTTSettings(model="nova-3"),
    )
    settings.audio.internal_sample_rate_hz = 8000
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "k3")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert backend.api_key == "k3"
    assert backend.model == "nova-3"
    assert backend.sample_rate_hz == 16000
    assert backend.language == get_deepgram_language(settings.languages.source_language)
    assert list(backend.keyterms) == []


def test_create_stt_backend_deepgram_passes_effective_custom_terms() -> None:
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.DEEPGRAM),
        deepgram_stt=DeepgramSTTSettings(model="nova-3"),
        stt=STTSettings(
            custom_vocabulary_enabled=True,
            custom_terms={"ko": [" Puripuly ", "", "VRChat", "Puripuly"]},
        ),
    )
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "k3")

    backend = create_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert list(backend.keyterms) == ["Puripuly", "VRChat"]


def test_create_stt_backend_local_qwen_uses_shared_model_path_without_secret() -> None:
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.LOCAL_QWEN),
    )
    settings.audio.internal_sample_rate_hz = 8000
    secrets = InMemorySecretStore()

    backend = create_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert backend.model_dir == default_local_stt_model_dir()
    assert backend.sample_rate_hz == 16000
    assert backend.stream_label == "self"


def test_create_stt_backend_local_qwen_passes_diagnostics_enabled_predicate() -> None:
    settings = AppSettings(provider=ProviderSettings(stt=STTProviderName.LOCAL_QWEN))
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
    settings = AppSettings(provider=ProviderSettings(stt=STTProviderName.LOCAL_QWEN))
    settings.languages.source_language = "ko-KR"
    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {
        "ko": ["Puripuly", "VRChat, Japan", *[f"term-{i:02d}" for i in range(20)]],
    }
    secrets = InMemorySecretStore()

    backend = create_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert getattr(backend, "language_hint", None) == "ko"
    assert getattr(backend, "hotwords", ()) == ()


def test_create_stt_backend_rejects_invalid_compatibility_provider() -> None:
    settings = AppSettings()
    settings.provider.stt = "corrupt-self-stt-provider"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unsupported STT provider"):
        create_stt_backend(settings, secrets=InMemorySecretStore())


def test_create_peer_stt_backend_uses_dedicated_deepgram_configuration_without_hint_terms() -> None:
    settings = AppSettings(
        provider=ProviderSettings(
            stt=STTProviderName.SONIOX,
            peer_stt=STTProviderName.DEEPGRAM,
        ),
        deepgram_stt=DeepgramSTTSettings(model="nova-3"),
    )
    settings.audio.internal_sample_rate_hz = 8000
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "peer-k")

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert backend.api_key == "peer-k"
    assert backend.model == "nova-3"
    assert backend.sample_rate_hz == 16000
    assert backend.language == get_deepgram_language(settings.languages.effective_peer_source)
    assert list(backend.keyterms) == []
    assert backend.stream_label == "peer"


def test_create_peer_stt_backend_uses_effective_peer_source_language_without_hint_terms() -> None:
    settings = AppSettings(
        provider=ProviderSettings(
            stt=STTProviderName.SONIOX,
            peer_stt=STTProviderName.DEEPGRAM,
        ),
        deepgram_stt=DeepgramSTTSettings(model="nova-3"),
    )
    settings.languages.source_language = "ko"
    settings.languages.peer_source_language = "zh-CN"
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "peer-k")

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert backend.language == get_deepgram_language(settings.languages.effective_peer_source)
    assert list(backend.keyterms) == []


def test_self_stt_provider_setting_does_not_change_peer_backend_choice() -> None:
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "peer-k")

    soniox_settings = AppSettings(
        provider=ProviderSettings(
            stt=STTProviderName.SONIOX,
            peer_stt=STTProviderName.DEEPGRAM,
        )
    )
    qwen_settings = AppSettings(
        provider=ProviderSettings(
            stt=STTProviderName.QWEN_ASR,
            peer_stt=STTProviderName.DEEPGRAM,
        )
    )

    soniox_backend = create_peer_stt_backend(soniox_settings, secrets=secrets)
    qwen_backend = create_peer_stt_backend(qwen_settings, secrets=secrets)

    assert isinstance(soniox_backend, DeepgramRealtimeSTTBackend)
    assert isinstance(qwen_backend, DeepgramRealtimeSTTBackend)


def test_resolve_peer_stt_config_always_uses_self_deepgram_model() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.DEEPGRAM
    settings.deepgram_stt.model = "nova-3-general"

    resolved = resolve_peer_stt_config(settings)

    assert resolved.provider == STTProviderName.DEEPGRAM
    assert resolved.model == "nova-3-general"


def test_resolve_peer_stt_config_exposes_legacy_provider_specific_fields() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.SONIOX
    settings.languages.peer_source_language = "zh-CN"
    settings.soniox_stt.model = "stt-rt-v4-peer"
    settings.soniox_stt.endpoint = "wss://peer-soniox.example/realtime"
    settings.soniox_stt.keepalive_interval_s = 12.5
    settings.soniox_stt.trailing_silence_ms = 700

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
    settings = AppSettings()
    settings.provider.peer_stt = "corrupt-peer-stt-provider"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unsupported peer STT provider"):
        resolve_peer_stt_config(settings)


def test_create_peer_stt_backend_rejects_invalid_compatibility_provider() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = "corrupt-peer-stt-provider"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unsupported peer STT provider"):
        create_peer_stt_backend(settings, secrets=InMemorySecretStore())


def test_create_peer_stt_backend_uses_peer_selected_soniox_provider() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.SONIOX
    settings.languages.peer_source_language = "ko"
    settings.soniox_stt.model = "stt-rt-v4"
    secrets = InMemorySecretStore()
    secrets.set("soniox_api_key", "peer-soniox")

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, SonioxRealtimeSTTBackend)
    assert backend.api_key == "peer-soniox"
    assert backend.model == "stt-rt-v4"


def test_create_peer_stt_backend_uses_shared_qwen_region_for_endpoint_and_secret() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.QWEN_ASR
    settings.qwen.region = QwenRegion.SINGAPORE
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "peer-qwen")

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.api_key == "peer-qwen"
    assert backend.endpoint == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"


def test_build_peer_stt_provider_signature_includes_backend_affecting_values() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.SONIOX
    settings.languages.peer_source_language = "zh-CN"
    settings.soniox_stt.model = "stt-rt-v4"
    settings.soniox_stt.trailing_silence_ms = 350

    signature = build_peer_stt_provider_signature(settings)

    assert STTProviderName.SONIOX in signature
    assert "zh-CN" in signature
    assert "stt-rt-v4" in signature
    assert 350 in signature


def test_peer_auto_detection_keeps_self_language_restriction_separate() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.SONIOX
    settings.provider.peer_stt = STTProviderName.SONIOX
    settings.languages.peer_source_mode = "auto"
    settings.languages.peer_expected_languages = ["ja", "zh-TW"]

    peer = resolve_peer_stt_runtime_config(settings)
    self_config = resolve_stt_config(_self_stt_runtime_intent_from_compatibility_settings(settings))

    assert peer.provider_options["enable_language_identification"] is True
    assert peer.provider_options["language_hints"] == ("ja", "zh")
    assert "enable_language_identification" not in self_config.provider_options
    assert self_config.provider_options["language_hints"] == ("ko",)
    assert self_config.provider_options["language_hints_strict"] is True
    assert "language_hints_strict" not in peer.provider_options


def test_peer_auto_detection_falls_back_to_manual_configuration_for_other_providers() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.DEEPGRAM
    settings.languages.peer_source_mode = "auto"
    settings.languages.peer_expected_languages = ["ja"]

    peer = resolve_peer_stt_runtime_config(settings)

    assert peer.source_language == settings.languages.effective_peer_source
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
    self_config = resolve_stt_config(
        _self_stt_runtime_intent_from_compatibility_settings(AppSettings())
    )

    assert peer.provider == "deepgram"
    assert peer.provider_options == {}
    assert self_config.provider_options.get("enable_language_identification") is None


def test_self_soniox_is_strict_while_manual_peer_uses_a_soft_hint() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.SONIOX
    settings.provider.peer_stt = STTProviderName.SONIOX
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
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.QWEN_ASR
    settings.audio.internal_sample_rate_hz = 8000

    signature = build_peer_stt_provider_signature(settings)

    assert signature[2] == 16000


def test_resolve_peer_stt_config_uses_shared_qwen_model_only() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.QWEN_ASR
    settings.qwen_asr_stt.model = "self-qwen-asr"

    resolved = resolve_peer_stt_config(settings)

    assert resolved.model == "self-qwen-asr"


def test_create_peer_stt_backend_uses_peer_local_qwen_provider_and_fixed_sample_rate() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.LOCAL_QWEN
    settings.audio.internal_sample_rate_hz = 8000
    secrets = InMemorySecretStore()

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert backend.model_dir == default_local_stt_model_dir()
    assert backend.sample_rate_hz == 16000
    assert backend.stream_label == "peer"


def test_create_peer_stt_backend_local_qwen_passes_diagnostics_enabled_predicate() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.LOCAL_QWEN
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
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.LOCAL_QWEN
    settings.languages.source_language = "ko"
    settings.languages.peer_source_language = "zh-CN"
    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {
        "zh-CN": ["airi", "shinano", *[f"term-{i:02d}" for i in range(20)]],
    }
    secrets = InMemorySecretStore()

    backend = create_peer_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, LocalQwenSherpaSTTBackend)
    assert getattr(backend, "language_hint", None) == "zh"
    assert getattr(backend, "hotwords", ()) == ()


def test_resolve_peer_stt_config_uses_shared_soniox_endpoint_keepalive_and_trailing_silence() -> (
    None
):
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.SONIOX
    settings.soniox_stt.model = "self-soniox"
    settings.soniox_stt.endpoint = "wss://self-soniox.example/realtime"
    settings.soniox_stt.keepalive_interval_s = 12.5
    settings.soniox_stt.trailing_silence_ms = 900

    resolved = resolve_peer_stt_config(settings)

    assert resolved.model == "self-soniox"
    assert resolved.endpoint == "wss://self-soniox.example/realtime"
    assert resolved.provider_options["keepalive_interval_s"] == 12.5
    assert resolved.provider_options["trailing_silence_ms"] == 900


def test_create_stt_backend_qwen_asr_uses_settings_and_secret() -> None:
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.QWEN_ASR),
        qwen_asr_stt=QwenASRSTTSettings(
            model="qwen3-asr-flash-realtime",
        ),
    )
    settings.audio.internal_sample_rate_hz = 8000
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
    assert backend.language == get_qwen_asr_language(settings.languages.source_language)


def test_create_stt_backend_qwen_asr_ignores_custom_terms() -> None:
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.QWEN_ASR),
        stt=STTSettings(
            custom_vocabulary_enabled=True,
            custom_terms={"ko": ["Puripuly", "VRChat"]},
        ),
        qwen_asr_stt=QwenASRSTTSettings(model="qwen3-asr-flash-realtime"),
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "k4")

    backend = create_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.api_key == "k4"
    assert backend.model == "qwen3-asr-flash-realtime"
    assert backend.language == get_qwen_asr_language(settings.languages.source_language)
    assert not hasattr(backend, "keyterms")
    assert not hasattr(backend, "context_terms")


def test_create_stt_backend_qwen_asr_uses_singapore_region() -> None:
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.QWEN_ASR),
        qwen=QwenSettings(region=QwenRegion.SINGAPORE),
        qwen_asr_stt=QwenASRSTTSettings(model="qwen3-asr-flash-realtime"),
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "k5")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.endpoint == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"


def test_create_stt_backend_qwen_asr_uses_legacy_alibaba_secret_key() -> None:
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.QWEN_ASR),
        qwen_asr_stt=QwenASRSTTSettings(model="qwen3-asr-flash-realtime"),
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key", "legacy-k4")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.api_key == "legacy-k4"
    # Legacy key should be backfilled to region-specific key for future runs.
    assert secrets.get("alibaba_api_key_beijing") == "legacy-k4"


def test_create_stt_backend_soniox_uses_secret() -> None:
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.SONIOX),
        soniox_stt=SonioxSTTSettings(model="stt-rt-v4"),
    )
    secrets = InMemorySecretStore()
    secrets.set("soniox_api_key", "k6")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, SonioxRealtimeSTTBackend)
    assert backend.api_key == "k6"
    assert list(backend.context_terms) == []


def test_create_stt_backend_soniox_passes_effective_custom_terms() -> None:
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.SONIOX),
        soniox_stt=SonioxSTTSettings(model="stt-rt-v4"),
        stt=STTSettings(
            custom_vocabulary_enabled=True,
            custom_terms={"ko": [" Puripuly ", "VRChat", "Puripuly", " "]},
        ),
    )
    secrets = InMemorySecretStore()
    secrets.set("soniox_api_key", "k6")

    backend = create_stt_backend(settings, secrets=secrets)

    assert isinstance(backend, SonioxRealtimeSTTBackend)
    assert list(backend.context_terms) == ["Puripuly", "VRChat"]
