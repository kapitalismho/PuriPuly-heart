from __future__ import annotations

from enum import Enum
from urllib.parse import urlsplit, urlunsplit

from puripuly_heart.config.llm_profiles import (
    OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_CHINA,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_26B_31B,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_31B,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_NONE,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_QWEN35_FLASH,
    OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
    OPENROUTER_MODEL_GEMINI_37_FLASH,
    OPENROUTER_MODEL_GEMMA_4_31B_IT,
    OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_BYOK,
    OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_MANAGED,
    OPENROUTER_SELECTION_ALIAS_GEMINI37_FLASH_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_MANAGED,
    OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_BYOK,
    OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_MANAGED,
)

MAX_CUSTOM_VOCAB_TERMS = 100
STT_INTERNAL_SAMPLE_RATE_HZ = 16000
LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "max_tokens",
    }
)
LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS = frozenset(
    {"api_key", "authorization", "headers", "token", "secret", "password"}
)
REFERRAL_ID_LENGTH = 6
REFERRAL_ID_ALPHABET = frozenset("23456789ABCDEFGHJKMNPQRSTUVWXYZ")


class STTProviderName(str, Enum):
    LOCAL_CPU_AUTO = "local_cpu_auto"
    LOCAL_PARAKEET_V3 = "local_parakeet_v3"
    LOCAL_PARAKEET_JAPANESE = "local_parakeet_ja"
    LOCAL_QWEN = "local_qwen"
    LOCAL_QWEN_GPU = "local_qwen_gpu"
    DEEPGRAM = "deepgram"
    GEMINI_TRANSCRIBE = "gemini_transcribe"
    ELEVENLABS_SCRIBE = "elevenlabs_scribe"
    QWEN_ASR = "qwen_asr"
    QWEN_AUDIO = "qwen_audio"
    SONIOX = "soniox"
    ROLLING_FREE = "rolling_free"
    CUSTOM = "custom"
    CUSTOM_OFFLINE = "custom_offline"
    CUSTOM_REALTIME = "custom_realtime"


CLOUD_FREE_TIER_STT_PROVIDERS: tuple[STTProviderName, ...] = (
    STTProviderName.GEMINI_TRANSCRIBE,
    STTProviderName.ELEVENLABS_SCRIBE,
    STTProviderName.DEEPGRAM,
)
DEFAULT_CLOUD_FREE_TIER_STT_PROVIDERS: tuple[STTProviderName, ...] = (
    STTProviderName.GEMINI_TRANSCRIBE,
)


def normalize_cloud_free_tier_providers(values: object) -> tuple[STTProviderName, ...]:
    allowed = {provider.value: provider for provider in CLOUD_FREE_TIER_STT_PROVIDERS}
    selected: list[STTProviderName] = []
    seen: set[STTProviderName] = set()
    if isinstance(values, list | tuple):
        for item in values:
            raw = getattr(item, "value", item)
            if not isinstance(raw, str):
                continue
            provider = allowed.get(raw.strip())
            if provider is None or provider in seen:
                continue
            selected.append(provider)
            seen.add(provider)
    if not selected:
        return DEFAULT_CLOUD_FREE_TIER_STT_PROVIDERS
    order = {provider: index for index, provider in enumerate(CLOUD_FREE_TIER_STT_PROVIDERS)}
    selected.sort(key=lambda provider: order[provider])
    return tuple(selected)


_CUSTOM_STT_PROVIDER_VALUES = frozenset(
    {
        STTProviderName.CUSTOM.value,
        STTProviderName.CUSTOM_OFFLINE.value,
        STTProviderName.CUSTOM_REALTIME.value,
    }
)


def is_custom_stt_provider(provider: STTProviderName | str | None) -> bool:
    if provider is None:
        return False
    value = provider.value if isinstance(provider, STTProviderName) else str(provider)
    return value in _CUSTOM_STT_PROVIDER_VALUES


def display_stt_provider(
    provider: STTProviderName,
    *,
    custom_mode: str = "offline",
    qwen_asr_model: str | None = None,
) -> STTProviderName:
    if provider is STTProviderName.CUSTOM:
        if custom_mode == "realtime":
            return STTProviderName.CUSTOM_REALTIME
        return STTProviderName.CUSTOM_OFFLINE
    return provider


def is_qwen_cloud_stt_provider(provider: STTProviderName | str | None) -> bool:
    if provider is None:
        return False
    value = provider.value if isinstance(provider, STTProviderName) else str(provider)
    return value in {STTProviderName.QWEN_ASR.value, STTProviderName.QWEN_AUDIO.value}


def qwen_cloud_stt_model_for_provider(provider: STTProviderName | str) -> str | None:
    value = provider.value if isinstance(provider, STTProviderName) else str(provider)
    if value == STTProviderName.QWEN_AUDIO.value:
        return QwenASRSTTModel.AUDIO_STREAMING.value
    if value == STTProviderName.QWEN_ASR.value:
        return QwenASRSTTModel.REALTIME.value
    return None


def custom_stt_selection_for_provider(
    provider: STTProviderName | str,
    *,
    stored_mode: str,
    stored_compatibility: str,
) -> tuple[str, str]:
    value = provider.value if isinstance(provider, STTProviderName) else str(provider)
    if value == STTProviderName.CUSTOM_REALTIME.value:
        return "realtime", "openai_realtime"
    if value == STTProviderName.CUSTOM_OFFLINE.value:
        return "offline", "openai_transcription"
    return stored_mode, stored_compatibility


class LLMProviderName(str, Enum):
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    CEREBRAS = "cerebras"
    MANAGED_GEMMA = "managed_gemma"
    LOCAL_LLM = "local_llm"


class QwenRegion(str, Enum):
    BEIJING = "beijing"
    SINGAPORE = "singapore"


class QwenLLMModel(str, Enum):
    QWEN_35_FLASH = "qwen3.5-flash"
    QWEN_38_FLASH = "qwen3.8-flash"


class QwenASRSTTModel(str, Enum):
    REALTIME = "qwen3-asr-flash-realtime"
    AUDIO_STREAMING = "qwen-audio-3.0-asr-flash-streaming"


class SecretsBackend(str, Enum):
    KEYRING = "keyring"
    ENCRYPTED_FILE = "encrypted_file"


class OpenRouterLLMModel(str, Enum):
    GEMMA_4_26B_A4B_IT = "google/gemma-4-26b-a4b-it"
    GEMMA_4_31B_IT = OPENROUTER_MODEL_GEMMA_4_31B_IT
    QWEN_35_FLASH_02_23 = "qwen/qwen3.5-flash-02-23"
    DEEPSEEK_V4_FLASH = OPENROUTER_MODEL_DEEPSEEK_V4_FLASH
    GEMINI_37_FLASH = OPENROUTER_MODEL_GEMINI_37_FLASH


class OpenRouterCredentialSource(str, Enum):
    NONE = "none"
    MANAGED = "managed"
    BYOK = "byok"


class GeminiLLMModel(str, Enum):
    GEMINI_37_FLASH = "gemini-3.7-flash"


class DeepSeekLLMModel(str, Enum):
    DEEPSEEK_V4_FLASH = "deepseek-v4-flash"


class CerebrasLLMModel(str, Enum):
    GEMMA_4_31B = "gemma-4-31b"


class LocalLLMBackend(str, Enum):
    OLLAMA = "ollama"


class OpenRouterFallbackSelectionAlias(str, Enum):
    NONE = OPENROUTER_FALLBACK_SELECTION_ALIAS_NONE
    QWEN35_FLASH = OPENROUTER_FALLBACK_SELECTION_ALIAS_QWEN35_FLASH
    DEEPSEEK_V4_FLASH = OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH
    DEEPSEEK_V4_FLASH_CHINA = OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_CHINA
    GEMMA4_26B_31B = OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_26B_31B
    GEMMA4_31B = OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_31B


class OpenRouterSelectionAlias(str, Enum):
    GEMMA4_26B_31B_MANAGED = OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED
    GEMMA4_26B_31B_BYOK = OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK
    GEMMA4_31B_MANAGED = OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED
    GEMMA4_31B_BYOK = OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK
    GEMMA4_MANAGED = OPENROUTER_SELECTION_ALIAS_GEMMA4_MANAGED
    GEMMA4_BYOK = OPENROUTER_SELECTION_ALIAS_GEMMA4_BYOK
    QWEN35_FLASH_MANAGED = OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_MANAGED
    QWEN35_FLASH_BYOK = OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_BYOK
    DEEPSEEK_V4_FLASH_MANAGED = OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_MANAGED
    DEEPSEEK_V4_FLASH_BYOK = OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_BYOK
    GEMINI37_FLASH_BYOK = OPENROUTER_SELECTION_ALIAS_GEMINI37_FLASH_BYOK


def normalize_owned_referral_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if len(normalized) != REFERRAL_ID_LENGTH:
        return None
    if any(char not in REFERRAL_ID_ALPHABET for char in normalized):
        return None
    return normalized


def normalize_local_llm_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid local llm base url")
    try:
        parsed = urlsplit(value.strip())
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("invalid local llm base url") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("invalid local llm base url")
    if not parsed.hostname:
        raise ValueError("invalid local llm base url")
    if (
        "@" in parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid local llm base url")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


__all__ = [
    "CerebrasLLMModel",
    "DeepSeekLLMModel",
    "GeminiLLMModel",
    "LLMProviderName",
    "LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS",
    "LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS",
    "LocalLLMBackend",
    "MAX_CUSTOM_VOCAB_TERMS",
    "OpenRouterCredentialSource",
    "OpenRouterFallbackSelectionAlias",
    "OpenRouterLLMModel",
    "OpenRouterSelectionAlias",
    "QwenLLMModel",
    "QwenASRSTTModel",
    "QwenRegion",
    "STTProviderName",
    "CLOUD_FREE_TIER_STT_PROVIDERS",
    "DEFAULT_CLOUD_FREE_TIER_STT_PROVIDERS",
    "normalize_cloud_free_tier_providers",
    "SecretsBackend",
    "STT_INTERNAL_SAMPLE_RATE_HZ",
    "custom_stt_selection_for_provider",
    "display_stt_provider",
    "is_custom_stt_provider",
    "is_qwen_cloud_stt_provider",
    "normalize_owned_referral_id",
    "qwen_cloud_stt_model_for_provider",
    "normalize_local_llm_base_url",
]
