from __future__ import annotations

import pytest

from puripuly_heart.app.wiring import create_llm_provider, create_stt_backend
from puripuly_heart.config.settings import (
    AppSettings,
    DeepgramSTTSettings,
    LLMProviderName,
    LLMSettings,
    ProviderSettings,
    QwenASRSTTSettings,
    STTProviderName,
)
from puripuly_heart.core.language import get_deepgram_language, get_qwen_asr_language
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
from puripuly_heart.providers.llm.qwen import QwenLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend


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
    assert provider.semaphore._value == 3  # type: ignore[attr-defined]


def test_create_llm_provider_qwen_uses_secret() -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.QWEN))
    secrets = InMemorySecretStore()
    # Default region is Beijing, so we need alibaba_api_key_beijing
    secrets.set("alibaba_api_key_beijing", "k2")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, QwenLLMProvider)
    assert provider.inner.api_key == "k2"
    assert provider.inner.base_url == "https://dashscope.aliyuncs.com/api/v1"


def test_create_llm_provider_requires_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.GEMINI))
    secrets = InMemorySecretStore()
    with pytest.raises(ValueError):
        create_llm_provider(settings, secrets=secrets)


def test_create_stt_backend_deepgram_uses_settings_and_secret() -> None:
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.DEEPGRAM),
        deepgram_stt=DeepgramSTTSettings(model="nova-3"),
    )
    secrets = InMemorySecretStore()
    secrets.set("deepgram_api_key", "k3")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, DeepgramRealtimeSTTBackend)
    assert backend.api_key == "k3"
    assert backend.model == "nova-3"
    assert backend.sample_rate_hz == settings.audio.internal_sample_rate_hz
    assert backend.language == get_deepgram_language(settings.languages.source_language)


def test_create_stt_backend_qwen_asr_uses_settings_and_secret() -> None:
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.QWEN_ASR),
        qwen_asr_stt=QwenASRSTTSettings(
            model="qwen3-asr-flash-realtime",
        ),
    )
    secrets = InMemorySecretStore()
    # Default region is Beijing, so we need alibaba_api_key_beijing
    secrets.set("alibaba_api_key_beijing", "k4")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, QwenASRRealtimeSTTBackend)
    assert backend.api_key == "k4"
    assert backend.model == "qwen3-asr-flash-realtime"
    # Endpoint is derived from region (Beijing default)
    assert backend.endpoint == "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    assert backend.sample_rate_hz == settings.audio.internal_sample_rate_hz
    assert backend.language == get_qwen_asr_language(settings.languages.source_language)
