from __future__ import annotations

import pytest

from ajebal_daera_translator.app.wiring import create_llm_provider, create_stt_backend
from ajebal_daera_translator.config.settings import (
    AlibabaSTTSettings,
    AppSettings,
    GoogleSpeechSettings,
    LLMProviderName,
    LLMSettings,
    ProviderSettings,
    STTProviderName,
)
from ajebal_daera_translator.core.llm.provider import SemaphoreLLMProvider
from ajebal_daera_translator.core.storage.secrets import InMemorySecretStore
from ajebal_daera_translator.providers.llm.gemini import GeminiLLMProvider
from ajebal_daera_translator.providers.llm.qwen import QwenLLMProvider
from ajebal_daera_translator.providers.stt.alibaba_model_studio import AlibabaModelStudioRealtimeSTTBackend
from ajebal_daera_translator.providers.stt.google_speech_v2 import GoogleSpeechV2Backend


def test_create_llm_provider_gemini_uses_secret_and_concurrency_limit():
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


def test_create_llm_provider_qwen_uses_secret():
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.QWEN))
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key", "k2")

    provider = create_llm_provider(settings, secrets=secrets)
    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, QwenLLMProvider)
    assert provider.inner.api_key == "k2"


def test_create_llm_provider_requires_secret():
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.GEMINI))
    secrets = InMemorySecretStore()
    with pytest.raises(ValueError):
        create_llm_provider(settings, secrets=secrets)


def test_create_stt_backend_google_uses_settings():
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.GOOGLE),
        google_speech=GoogleSpeechSettings(recognizer="projects/p/locations/l/recognizers/r"),
    )
    secrets = InMemorySecretStore()

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, GoogleSpeechV2Backend)
    assert backend.recognizer == "projects/p/locations/l/recognizers/r"
    assert backend.sample_rate_hz == settings.audio.internal_sample_rate_hz
    assert backend.language_codes == (settings.languages.source_language,)


def test_create_stt_backend_alibaba_uses_settings_and_secret():
    settings = AppSettings(
        provider=ProviderSettings(stt=STTProviderName.ALIBABA),
        alibaba_stt=AlibabaSTTSettings(model="m", endpoint="wss://example"),
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key", "k3")

    backend = create_stt_backend(settings, secrets=secrets)
    assert isinstance(backend, AlibabaModelStudioRealtimeSTTBackend)
    assert backend.api_key == "k3"
    assert backend.model == "m"
    assert backend.endpoint == "wss://example"
    assert backend.sample_rate_hz == settings.audio.internal_sample_rate_hz
