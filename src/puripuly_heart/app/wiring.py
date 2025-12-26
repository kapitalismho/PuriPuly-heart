from __future__ import annotations

import asyncio
import os
from pathlib import Path

from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    SecretsBackend,
    SecretsSettings,
    STTProviderName,
)
from puripuly_heart.core.llm.provider import LLMProvider, SemaphoreLLMProvider
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    KeyringSecretStore,
    SecretStore,
)
from puripuly_heart.core.stt.backend import STTBackend
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
from puripuly_heart.providers.llm.qwen import QwenLLMProvider

SECRETS_PASSPHRASE_ENV = "PURIPULY_HEART_SECRETS_PASSPHRASE"


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


def create_llm_provider(settings: AppSettings, *, secrets: SecretStore) -> LLMProvider:
    if settings.provider.llm == LLMProviderName.GEMINI:
        api_key = require_secret(secrets, key="google_api_key", env_var="GOOGLE_API_KEY")
        base: LLMProvider = GeminiLLMProvider(api_key=api_key)
    elif settings.provider.llm == LLMProviderName.QWEN:
        api_key = require_secret(secrets, key="alibaba_api_key", env_var="ALIBABA_API_KEY")
        base = QwenLLMProvider(api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.provider.llm}")

    return SemaphoreLLMProvider(
        inner=base,
        semaphore=asyncio.Semaphore(settings.llm.concurrency_limit),
    )


def create_stt_backend(settings: AppSettings, *, secrets: SecretStore) -> STTBackend:
    if settings.provider.stt == STTProviderName.DEEPGRAM:
        from puripuly_heart.core.language import get_deepgram_language
        from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend

        api_key = require_secret(secrets, key="deepgram_api_key", env_var="DEEPGRAM_API_KEY")
        return DeepgramRealtimeSTTBackend(
            api_key=api_key,
            model=settings.deepgram_stt.model,
            language=get_deepgram_language(settings.languages.source_language),
            sample_rate_hz=settings.audio.internal_sample_rate_hz,
        )

    if settings.provider.stt == STTProviderName.QWEN_ASR:
        from puripuly_heart.core.language import get_qwen_asr_language
        from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend

        api_key = require_secret(secrets, key="alibaba_api_key", env_var="ALIBABA_API_KEY")
        return QwenASRRealtimeSTTBackend(
            api_key=api_key,
            model=settings.qwen_asr_stt.model,
            endpoint=settings.qwen_asr_stt.endpoint,
            language=get_qwen_asr_language(settings.languages.source_language),
            sample_rate_hz=settings.audio.internal_sample_rate_hz,
        )

    raise ValueError(f"Unsupported STT provider: {settings.provider.stt}")
