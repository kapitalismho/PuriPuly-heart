from __future__ import annotations

import copy
import json
import locale
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from puripuly_heart.config.audio_host_api import (
    WINDOWS_DIRECTSOUND_HOST_API,
    WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
)
from puripuly_heart.config.llm_profiles import (
    LEGACY_OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_CHINA,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_26B_31B,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_31B,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_NONE,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_QWEN35_FLASH,
    OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
    OPENROUTER_MODEL_GEMINI_31_FLASH_LITE,
    OPENROUTER_MODEL_GEMINI_37_FLASH,
    OPENROUTER_MODEL_GEMMA_4_31B_IT,
    OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_BYOK,
    OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_MANAGED,
    OPENROUTER_SELECTION_ALIAS_GEMINI31_FLASH_LITE_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMINI37_FLASH_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_MANAGED,
    OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_BYOK,
    OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_MANAGED,
    get_openrouter_llm_profile,
    get_openrouter_selection_alias_for_model_and_source,
    normalize_openrouter_fallback_selection_alias,
    openrouter_alias_for_fields,
)
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.settings_vnext.schema import (  # noqa: F401
    AppSettingsVNext,
    new_anonymous_telemetry_identifier,
)

SETTINGS_SCHEMA_VERSION = 24
MANAGED_AUTH_CLAIM_SOURCE_DISCORD = "discord"
MANAGED_AUTH_CLAIM_SOURCE_QQ = "qq"
MANAGED_AUTH_CLAIM_SOURCES = (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    MANAGED_AUTH_CLAIM_SOURCE_QQ,
)
STT_INTERNAL_SAMPLE_RATE_HZ = 16000
DEFAULT_DESKTOP_AUDIO_VAD_HANGOVER_MS = 500
LEGACY_LOW_LATENCY_VAD_HANGOVER_MS = 600
DEFAULT_LOW_LATENCY_VAD_HANGOVER_MS = 500
MAX_CUSTOM_VOCAB_TERMS = 100
DEFAULT_OPENROUTER_BROKER_BASE_URL = "https://puripuly-heart-broker.kapitalismho.workers.dev"
TELEMETRY_CONSENT_VALUES = frozenset({"unknown", "allow", "decline"})
REFERRAL_ID_LENGTH = 6
REFERRAL_ID_ALPHABET = frozenset("23456789ABCDEFGHJKMNPQRSTUVWXYZ")
OVERLAY_TARGET_STEAMVR = "steamvr"
OVERLAY_TARGET_DESKTOP = "desktop"
OVERLAY_TARGET_VALUES = frozenset({OVERLAY_TARGET_STEAMVR, OVERLAY_TARGET_DESKTOP})
DESKTOP_FLET_MIN_WIDTH = 480
DESKTOP_FLET_MIN_HEIGHT = 160
DESKTOP_FLET_DEFAULT_TEXT_SCALE = 1.0
DESKTOP_FLET_MIN_TEXT_SCALE = 0.75
DESKTOP_FLET_MAX_TEXT_SCALE = 1.5
DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA = 0.6
DESKTOP_FLET_MIN_BACKGROUND_ALPHA = 0.0
DESKTOP_FLET_MAX_BACKGROUND_ALPHA = 1.0
DESKTOP_FLET_MIN_OUTLINE_WIDTH = 0.5
DESKTOP_FLET_MAX_OUTLINE_WIDTH = 8.0
DESKTOP_FLET_SIZE_PRESET_ORDER = ("tiny", "xsmall", "small", "medium", "large", "xlarge")
DESKTOP_FLET_SIZE_PRESET_DISPLAY_ORDER = tuple(reversed(DESKTOP_FLET_SIZE_PRESET_ORDER))
DESKTOP_FLET_DEFAULT_SIZE_PRESET = "medium"
DESKTOP_FLET_SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "tiny": (640, 160),
    "xsmall": (960, 240),
    "small": (1152, 288),
    "medium": (1344, 336),
    "large": (1600, 400),
    "xlarge": (1792, 448),
}
DESKTOP_FLET_DEFAULT_WIDTH = DESKTOP_FLET_SIZE_PRESETS[DESKTOP_FLET_DEFAULT_SIZE_PRESET][0]
DESKTOP_FLET_DEFAULT_HEIGHT = DESKTOP_FLET_SIZE_PRESETS[DESKTOP_FLET_DEFAULT_SIZE_PRESET][1]
DEFAULT_CUSTOM_VOCAB_TERMS: dict[str, tuple[str, ...]] = {}
LEGACY_QWEN_DEFAULT_PROMPT = (
    "VRChat social voice chat interpretation. Use spoken, conversational language and mirror "
    "the speaker's tone and formality. Fix voice recognition errors like missing punctuation "
    "and typos."
)
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


def _default_local_llm_extra_body() -> dict[str, object]:
    return {"reasoning_effort": "none"}


def _default_custom_terms() -> dict[str, list[str]]:
    return {language: list(terms) for language, terms in DEFAULT_CUSTOM_VOCAB_TERMS.items()}


def normalize_owned_referral_id(value: object) -> str | None:
    """Normalize an owned Referral ID for app persistence/display, or return None."""

    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if len(normalized) != REFERRAL_ID_LENGTH:
        return None
    if any(char not in REFERRAL_ID_ALPHABET for char in normalized):
        return None
    return normalized


def normalize_managed_claim_sources(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates: tuple[object, ...] = (value,)
    elif isinstance(value, (list, tuple, set, frozenset)):
        candidates = tuple(value)
    else:
        candidates = ()

    normalized = {
        item.strip().lower()
        for item in candidates
        if isinstance(item, str) and item.strip().lower() in MANAGED_AUTH_CLAIM_SOURCES
    }
    return tuple(source for source in MANAGED_AUTH_CLAIM_SOURCES if source in normalized)


class STTProviderName(str, Enum):
    LOCAL_CPU_AUTO = "local_cpu_auto"
    LOCAL_PARAKEET_V3 = "local_parakeet_v3"
    LOCAL_PARAKEET_JAPANESE = "local_parakeet_ja"
    LOCAL_QWEN = "local_qwen"
    LOCAL_QWEN_GPU = "local_qwen_gpu"
    DEEPGRAM = "deepgram"
    QWEN_ASR = "qwen_asr"
    SONIOX = "soniox"
    CUSTOM = "custom"
    CUSTOM_OFFLINE = "custom_offline"
    CUSTOM_REALTIME = "custom_realtime"


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
) -> STTProviderName:
    if provider is not STTProviderName.CUSTOM:
        return provider
    if custom_mode == "realtime":
        return STTProviderName.CUSTOM_REALTIME
    return STTProviderName.CUSTOM_OFFLINE


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
    LOCAL_LLM = "local_llm"


class SecretsBackend(str, Enum):
    KEYRING = "keyring"
    ENCRYPTED_FILE = "encrypted_file"


class QwenRegion(str, Enum):
    BEIJING = "beijing"
    SINGAPORE = "singapore"


class GeminiLLMModel(str, Enum):
    GEMINI_37_FLASH = "gemini-3.7-flash"
    GEMINI_31_FLASH_LITE = "gemini-3.1-flash-lite"


class QwenLLMModel(str, Enum):
    QWEN_35_FLASH = "qwen3.5-flash"
    QWEN_35_PLUS = "qwen3.5-plus"


class DeepSeekLLMModel(str, Enum):
    DEEPSEEK_V4_FLASH = "deepseek-v4-flash"


class CerebrasLLMModel(str, Enum):
    GEMMA_4_31B = "gemma-4-31b"


class LocalLLMBackend(str, Enum):
    OLLAMA = "ollama"


class OpenRouterLLMModel(str, Enum):
    GEMMA_4_26B_A4B_IT = "google/gemma-4-26b-a4b-it"
    GEMMA_4_31B_IT = OPENROUTER_MODEL_GEMMA_4_31B_IT
    QWEN_35_FLASH_02_23 = "qwen/qwen3.5-flash-02-23"
    DEEPSEEK_V4_FLASH = OPENROUTER_MODEL_DEEPSEEK_V4_FLASH
    GEMINI_37_FLASH = OPENROUTER_MODEL_GEMINI_37_FLASH
    GEMINI_31_FLASH_LITE = OPENROUTER_MODEL_GEMINI_31_FLASH_LITE


class OpenRouterRoutingMode(str, Enum):
    LATENCY = "latency"


class OpenRouterProviderRouting(str, Enum):
    DEFAULT = "default"
    DEEPSEEK_ONLY = "deepseek_only"
    GOOGLE_GEMINI_LATENCY = "google_gemini_latency"
    GEMMA4_26B_31B_LATENCY = "gemma4_26b_31b_latency"
    GEMMA4_31B_LATENCY = "gemma4_31b_latency"
    GEMMA4_26B_LATENCY = "gemma4_26b_latency"
    DEEPSEEK_V4_FLASH_LATENCY = "deepseek_v4_flash_latency"
    GEMMA4_31B_CEREBRAS_ONLY = "gemma4_31b_cerebras_only"


class OpenRouterCredentialSource(str, Enum):
    NONE = "none"
    MANAGED = "managed"
    BYOK = "byok"


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
    GEMINI31_FLASH_LITE_BYOK = OPENROUTER_SELECTION_ALIAS_GEMINI31_FLASH_LITE_BYOK


class OpenRouterFallbackSelectionAlias(str, Enum):
    NONE = OPENROUTER_FALLBACK_SELECTION_ALIAS_NONE
    QWEN35_FLASH = OPENROUTER_FALLBACK_SELECTION_ALIAS_QWEN35_FLASH
    DEEPSEEK_V4_FLASH = OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH
    DEEPSEEK_V4_FLASH_CHINA = OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_CHINA
    GEMMA4_26B_31B = OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_26B_31B
    GEMMA4_31B = OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_31B


class TranslationModel(str, Enum):
    GEMMA4_26B_31B = "gemma4_26b_31b"
    GEMMA4_31B = "gemma4_31b"
    GEMMA4 = "gemma4"
    DEEPSEEK_V4_FLASH = "deepseek_v4_flash"
    GEMINI_37_FLASH = "gemini37_flash"
    GEMINI_31_FLASH_LITE = "gemini31_flash_lite"
    QWEN_35_PLUS = "qwen35_plus"
    LOCAL_LLM = "local_llm"
    CUSTOM_HTTP = "custom_http"


class TranslationConnection(str, Enum):
    MANAGED = "managed"
    MANAGED_CHINA = "managed_china"
    OPENROUTER = "openrouter"
    CEREBRAS = "cerebras"
    OFFICIAL_BYOK = "official_byok"
    OLLAMA = "ollama"
    CUSTOM_HTTP = "custom_http"


@dataclass(slots=True)
class TranslationFallbackSettings:
    enabled: bool = False
    model: TranslationModel = TranslationModel.DEEPSEEK_V4_FLASH
    connection: TranslationConnection = TranslationConnection.OFFICIAL_BYOK

    def validate(self) -> None:
        self.enabled = bool(self.enabled)
        if not isinstance(self.model, TranslationModel):
            raise ValueError("invalid translation fallback model")
        if not isinstance(self.connection, TranslationConnection):
            raise ValueError("invalid translation fallback connection")
        if self.model == TranslationModel.CUSTOM_HTTP:
            raise ValueError("custom HTTP translation cannot be used as fallback")
        if self.connection not in _supported_translation_connections(self.model):
            raise ValueError("translation fallback connection is not supported for model")


@dataclass(slots=True)
class TranslationSettings:
    model: TranslationModel = TranslationModel.GEMMA4_26B_31B
    connection: TranslationConnection = TranslationConnection.MANAGED
    connection_history: dict[str, TranslationConnection] = field(
        default_factory=lambda: _default_translation_connection_history()
    )
    fallback: TranslationFallbackSettings = field(default_factory=TranslationFallbackSettings)
    http_extension_id: str | None = None
    previous_llm_model: TranslationModel | None = None

    def validate(self) -> None:
        if not isinstance(self.model, TranslationModel):
            raise ValueError("invalid translation model")
        if not isinstance(self.connection, TranslationConnection):
            raise ValueError("invalid translation connection")
        if self.connection not in _supported_translation_connections(self.model):
            raise ValueError("translation connection is not supported for model")
        if not isinstance(self.connection_history, dict):
            raise ValueError("translation connection_history must be a dict")
        for model_value, connection in self.connection_history.items():
            model = _parse_translation_model(model_value)
            if model is None:
                raise ValueError("invalid translation connection_history model")
            if not isinstance(connection, TranslationConnection):
                raise ValueError("invalid translation connection_history connection")
            if connection not in _supported_translation_connections(model):
                raise ValueError("translation connection_history connection is not supported")
        if self.http_extension_id is not None and (
            not isinstance(self.http_extension_id, str)
            or not self.http_extension_id.strip()
            or len(self.http_extension_id.strip()) > 64
        ):
            raise ValueError("invalid HTTP http_extension_id")
        if self.previous_llm_model is not None and not isinstance(
            self.previous_llm_model,
            TranslationModel,
        ):
            raise ValueError("invalid previous LLM translation model")
        if self.model == TranslationModel.CUSTOM_HTTP:
            if self.connection != TranslationConnection.CUSTOM_HTTP:
                raise ValueError("custom HTTP translation requires custom_http connection")
            if self.previous_llm_model == TranslationModel.CUSTOM_HTTP:
                raise ValueError("invalid previous LLM translation model")
        elif self.previous_llm_model is not None:
            raise ValueError("previous LLM translation model is only valid for custom HTTP")
        self.fallback.validate()


TRANSLATION_CONNECTIONS_BY_MODEL: dict[TranslationModel, tuple[TranslationConnection, ...]] = {
    TranslationModel.GEMMA4_26B_31B: (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.GEMMA4_31B: (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
        TranslationConnection.CEREBRAS,
    ),
    TranslationModel.GEMMA4: (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.DEEPSEEK_V4_FLASH: (
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
        TranslationConnection.OPENROUTER,
        TranslationConnection.OFFICIAL_BYOK,
    ),
    TranslationModel.GEMINI_37_FLASH: (
        TranslationConnection.OFFICIAL_BYOK,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.GEMINI_31_FLASH_LITE: (
        TranslationConnection.OFFICIAL_BYOK,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.QWEN_35_PLUS: (TranslationConnection.OFFICIAL_BYOK,),
    TranslationModel.LOCAL_LLM: (TranslationConnection.OLLAMA,),
    TranslationModel.CUSTOM_HTTP: (TranslationConnection.CUSTOM_HTTP,),
}
TRANSLATION_CONNECTION_PRIORITY: tuple[TranslationConnection, ...] = (
    TranslationConnection.MANAGED,
    TranslationConnection.OPENROUTER,
    TranslationConnection.OFFICIAL_BYOK,
)


def supported_translation_connections(
    model: TranslationModel,
) -> tuple[TranslationConnection, ...]:
    return TRANSLATION_CONNECTIONS_BY_MODEL[model]


def default_translation_connection(model: TranslationModel) -> TranslationConnection:
    if model == TranslationModel.CUSTOM_HTTP:
        return TranslationConnection.CUSTOM_HTTP
    if model in (TranslationModel.GEMINI_37_FLASH, TranslationModel.GEMINI_31_FLASH_LITE):
        return TranslationConnection.OFFICIAL_BYOK
    supported_connections = supported_translation_connections(model)
    for connection in TRANSLATION_CONNECTION_PRIORITY:
        if connection in supported_connections:
            return connection
    return supported_connections[0]


def _supported_translation_connections(
    model: TranslationModel,
) -> tuple[TranslationConnection, ...]:
    return supported_translation_connections(model)


def _default_translation_connection(model: TranslationModel) -> TranslationConnection:
    return default_translation_connection(model)


def _default_translation_connection_history() -> dict[str, TranslationConnection]:
    return {TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.MANAGED}


def _parse_translation_model(value: object) -> TranslationModel | None:
    if isinstance(value, TranslationModel):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized == "gemma4_31b_cerebras":
            return TranslationModel.GEMMA4_31B
        if normalized == "deepseek_v4_pro":
            return TranslationModel.DEEPSEEK_V4_FLASH
        if normalized == "gemini3_flash":
            return TranslationModel.GEMINI_37_FLASH
        try:
            return TranslationModel(normalized)
        except ValueError:
            pass
    return None


def _parse_translation_connection(value: object) -> TranslationConnection | None:
    if isinstance(value, TranslationConnection):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        try:
            return TranslationConnection(normalized)
        except ValueError:
            pass
    return None


def _parse_translation_connection_history(value: object) -> dict[str, TranslationConnection]:
    if not isinstance(value, dict):
        return {}

    history: dict[str, TranslationConnection] = {}
    legacy_cerebras = False
    for raw_model, raw_connection in value.items():
        if str(raw_model).strip() == "gemma4_31b_cerebras":
            legacy_cerebras = True
            continue
        model = _parse_translation_model(raw_model)
        connection = _parse_translation_connection(raw_connection)
        if model is None or connection is None:
            continue
        if connection not in _supported_translation_connections(model):
            continue
        history[model.value] = connection
    if TranslationModel.GEMMA4_31B.value not in history and legacy_cerebras:
        history[TranslationModel.GEMMA4_31B.value] = TranslationConnection.CEREBRAS
    return history


def _parse_translation_selection(
    raw_model: object,
    raw_connection: object,
) -> tuple[TranslationModel | None, TranslationConnection | None]:
    if isinstance(raw_model, str) and raw_model.strip() == "gemma4_31b_cerebras":
        return TranslationModel.GEMMA4_31B, TranslationConnection.CEREBRAS
    return _parse_translation_model(raw_model), _parse_translation_connection(raw_connection)


def _parse_translation_fallback(value: object) -> TranslationFallbackSettings:
    if isinstance(value, TranslationFallbackSettings):
        fallback = copy.deepcopy(value)
        fallback.validate()
        return fallback
    if not isinstance(value, dict):
        return TranslationFallbackSettings()
    model, connection = _parse_translation_selection(value.get("model"), value.get("connection"))
    model = model or TranslationModel.DEEPSEEK_V4_FLASH
    if connection not in _supported_translation_connections(model):
        connection = _default_translation_connection(model)
    fallback = TranslationFallbackSettings(
        enabled=bool(value.get("enabled", False)),
        model=model,
        connection=connection,
    )
    fallback.validate()
    return fallback


def _normalize_translation_settings(
    *,
    model: TranslationModel | None,
    connection: TranslationConnection | None,
    fallback: object = None,
    history: object = None,
    http_extension_id: object = None,
    previous_llm_model: object = None,
) -> TranslationSettings:
    normalized_model = model or TranslationModel.GEMMA4_26B_31B
    normalized_history = _parse_translation_connection_history(history)
    if connection not in _supported_translation_connections(normalized_model):
        connection = _default_translation_connection(normalized_model)
    normalized_http_extension_id = (
        http_extension_id.strip()
        if isinstance(http_extension_id, str) and http_extension_id.strip()
        else None
    )
    normalized_previous_llm_model = _parse_translation_model(previous_llm_model)
    if normalized_previous_llm_model == TranslationModel.CUSTOM_HTTP:
        normalized_previous_llm_model = None
    if normalized_model != TranslationModel.CUSTOM_HTTP:
        normalized_previous_llm_model = None
    normalized_history[normalized_model.value] = connection
    return TranslationSettings(
        model=normalized_model,
        connection=connection,
        connection_history=normalized_history,
        fallback=_parse_translation_fallback(fallback),
        http_extension_id=normalized_http_extension_id,
        previous_llm_model=normalized_previous_llm_model,
    )


def _translation_data_has_valid_model(value: object) -> bool:
    return isinstance(value, dict) and _parse_translation_model(value.get("model")) is not None


def _translation_connection_from_openrouter_source(
    selected_source: OpenRouterCredentialSource,
    *,
    model: TranslationModel,
    provider_routing: OpenRouterProviderRouting = OpenRouterProviderRouting.DEFAULT,
) -> TranslationConnection:
    if selected_source == OpenRouterCredentialSource.MANAGED:
        if (
            model == TranslationModel.DEEPSEEK_V4_FLASH
            and provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY
        ):
            return TranslationConnection.MANAGED_CHINA
        return TranslationConnection.MANAGED
    if selected_source == OpenRouterCredentialSource.BYOK:
        return TranslationConnection.OPENROUTER
    return _default_translation_connection(model)


def _history_connection_or_default(
    model: TranslationModel,
    history: dict[str, TranslationConnection],
) -> TranslationConnection:
    connection = history.get(model.value)
    if connection in _supported_translation_connections(model):
        return connection
    return _default_translation_connection(model)


def _translation_settings_to_dict(settings: TranslationSettings) -> dict[str, Any]:
    data: dict[str, Any] = {
        "model": settings.model.value,
        "connection": settings.connection.value,
        "connection_history": {
            model: connection.value for model, connection in settings.connection_history.items()
        },
        "fallback": {
            "enabled": settings.fallback.enabled,
            "model": settings.fallback.model.value,
            "connection": settings.fallback.connection.value,
        },
    }
    if settings.model == TranslationModel.CUSTOM_HTTP or settings.http_extension_id is not None:
        data["http_extension_id"] = settings.http_extension_id
    if settings.previous_llm_model is not None:
        data["previous_llm_model"] = settings.previous_llm_model.value
    return data


def _default_translation_settings_dict() -> dict[str, Any]:
    return {
        "model": TranslationModel.GEMMA4_26B_31B.value,
        "connection": TranslationConnection.MANAGED.value,
        "connection_history": {
            TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.MANAGED.value,
        },
        "fallback": {
            "enabled": False,
            "model": TranslationModel.DEEPSEEK_V4_FLASH.value,
            "connection": TranslationConnection.OFFICIAL_BYOK.value,
        },
    }


def _translation_settings_is_exact_default(settings: TranslationSettings) -> bool:
    return _translation_settings_to_dict(settings) == _default_translation_settings_dict()


@dataclass(slots=True)
class LanguageSettings:
    source_language: str = "ko"
    target_language: str = "en"
    peer_source_language: str = "en"
    peer_target_language: str = "ko"
    peer_source_mode: str = "manual"
    peer_expected_languages: list[str] = field(default_factory=list)
    recent_source_languages: list[str] = field(default_factory=lambda: ["en", "zh-CN", "ja"])
    recent_target_languages: list[str] = field(default_factory=lambda: ["en", "zh-CN", "ja"])

    def validate(self) -> None:
        if not self.source_language:
            raise ValueError("source_language must be non-empty")
        if not self.target_language:
            raise ValueError("target_language must be non-empty")
        if self.peer_source_mode not in {"manual", "auto"}:
            raise ValueError("peer_source_mode must be manual or auto")

    @property
    def effective_peer_source(self) -> str:
        return self.peer_source_language or self.source_language

    @property
    def effective_peer_target(self) -> str:
        return self.peer_target_language or self.target_language


@dataclass(slots=True)
class AudioSettings:
    internal_sample_rate_hz: int = STT_INTERNAL_SAMPLE_RATE_HZ
    internal_channels: int = 1
    ring_buffer_ms: int = 500
    input_host_api: str = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    input_device: str = ""

    def validate(self) -> None:
        if self.internal_sample_rate_hz != STT_INTERNAL_SAMPLE_RATE_HZ:
            raise ValueError(f"internal_sample_rate_hz must be {STT_INTERNAL_SAMPLE_RATE_HZ}")
        if self.internal_channels != 1:
            raise ValueError("internal_channels must be 1 (mono)")
        if self.ring_buffer_ms <= 0:
            raise ValueError("ring_buffer_ms must be > 0")
        if self.input_host_api is None:
            raise ValueError("input_host_api must be a string")
        if self.input_device is None:
            raise ValueError("input_device must be a string")


@dataclass(slots=True)
class DesktopAudioSettings:
    output_device: str = ""
    vad_speech_threshold: float = 0.5
    vad_hangover_ms: int = DEFAULT_DESKTOP_AUDIO_VAD_HANGOVER_MS
    vad_pre_roll_ms: int = 500
    runtime_capture_target: object | None = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self.output_device is None:
            raise ValueError("output_device must be a string")
        if not (0.0 <= self.vad_speech_threshold <= 1.0):
            raise ValueError("vad_speech_threshold must be in 0.0..1.0")
        if self.vad_hangover_ms < 0:
            raise ValueError("vad_hangover_ms must be >= 0")
        if self.vad_pre_roll_ms < 0:
            raise ValueError("vad_pre_roll_ms must be >= 0")


@dataclass(slots=True)
class STTSettings:
    drain_timeout_s: float = 2.0
    vad_speech_threshold: float = 0.4
    low_latency_mode: bool = True
    low_latency_vad_hangover_ms: int = DEFAULT_LOW_LATENCY_VAD_HANGOVER_MS
    low_latency_merge_gap_ms: int = 600
    low_latency_spec_retry_max: int = 10
    custom_vocabulary_enabled: bool = True
    custom_terms: dict[str, list[str]] = field(default_factory=_default_custom_terms)
    gpu_device_id: str = "auto"

    def validate(self) -> None:
        if self.drain_timeout_s <= 0:
            raise ValueError("drain_timeout_s must be > 0")
        if not (0.0 <= self.vad_speech_threshold <= 1.0):
            raise ValueError("vad_speech_threshold must be in 0.0..1.0")
        if self.low_latency_vad_hangover_ms < 0:
            raise ValueError("low_latency_vad_hangover_ms must be >= 0")
        if self.low_latency_merge_gap_ms < 0:
            raise ValueError("low_latency_merge_gap_ms must be >= 0")
        if self.low_latency_spec_retry_max < 0:
            raise ValueError("low_latency_spec_retry_max must be >= 0")
        if not isinstance(self.custom_vocabulary_enabled, bool):
            raise ValueError("custom_vocabulary_enabled must be a bool")
        if not isinstance(self.custom_terms, dict):
            raise ValueError("custom_terms must be a dict[str, list[str]]")
        if not isinstance(self.gpu_device_id, str) or not self.gpu_device_id.strip():
            raise ValueError("gpu_device_id must be a non-empty string")
        for language, terms in self.custom_terms.items():
            if not isinstance(language, str):
                raise ValueError("custom_terms keys must be strings")
            if not isinstance(terms, list):
                raise ValueError("custom_terms values must be lists of strings")
            for term in terms:
                if not isinstance(term, str):
                    raise ValueError("custom_terms values must be lists of strings")


@dataclass(slots=True)
class DeepgramSTTSettings:
    model: str = "nova-3"

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")


@dataclass(slots=True)
class QwenASRSTTSettings:
    model: str = "qwen3-asr-flash-realtime"
    endpoint: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if not self.endpoint:
            raise ValueError("endpoint must be non-empty")


@dataclass(slots=True)
class SonioxSTTSettings:
    model: str = "stt-rt-v5"
    endpoint: str = "wss://stt-rt.soniox.com/transcribe-websocket"
    keepalive_interval_s: float = 10.0
    trailing_silence_ms: int = 100

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if not self.endpoint:
            raise ValueError("endpoint must be non-empty")
        if self.keepalive_interval_s <= 0:
            raise ValueError("keepalive_interval_s must be > 0")
        if self.trailing_silence_ms < 0:
            raise ValueError("trailing_silence_ms must be >= 0")


@dataclass(slots=True)
class CustomSTTSettings:
    mode: str = "offline"
    compatibility: str = "openai_transcription"
    endpoint: str = ""
    model: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        from puripuly_heart.core.stt.custom import (
            normalize_custom_stt_compatibility,
            normalize_custom_stt_endpoint,
            normalize_custom_stt_extra,
            normalize_custom_stt_mode,
            normalize_custom_stt_model,
            validate_mode_compatibility,
        )

        self.mode = normalize_custom_stt_mode(self.mode)
        self.compatibility = normalize_custom_stt_compatibility(
            self.compatibility,
            mode=self.mode,
        )
        self.endpoint = normalize_custom_stt_endpoint(self.endpoint)
        self.model = normalize_custom_stt_model(self.model)
        self.extra = normalize_custom_stt_extra(self.extra)
        validate_mode_compatibility(self.mode, self.compatibility)


@dataclass(slots=True)
class PeerQwenASRSTTSettings:
    model: str | None = None
    region: QwenRegion | None = None

    def validate(self) -> None:
        if self.model is not None and not self.model:
            raise ValueError("peer qwen asr model override must be non-empty")
        if self.region is not None and not isinstance(self.region, QwenRegion):
            raise ValueError("invalid peer qwen asr region")


@dataclass(slots=True)
class PeerSonioxSTTSettings:
    model: str | None = None
    endpoint: str | None = None
    keepalive_interval_s: float | None = None
    trailing_silence_ms: int | None = None

    def validate(self) -> None:
        if self.model is not None and not self.model:
            raise ValueError("peer soniox model override must be non-empty")
        if self.endpoint is not None and not self.endpoint:
            raise ValueError("peer soniox endpoint override must be non-empty")
        if self.keepalive_interval_s is not None and self.keepalive_interval_s <= 0:
            raise ValueError("peer soniox keepalive override must be > 0")
        if self.trailing_silence_ms is not None and self.trailing_silence_ms < 0:
            raise ValueError("peer soniox trailing silence override must be >= 0")


@dataclass(slots=True)
class LLMSettings:
    concurrency_limit: int = 5

    def validate(self) -> None:
        if self.concurrency_limit <= 0:
            raise ValueError("concurrency_limit must be > 0")


@dataclass(slots=True)
class OSCSettings:
    host: str = "127.0.0.1"
    port: int = 9000
    send_port: int | None = None
    receive_port: int = 9001
    connection_mode: str = "automatic"
    chatbox_address: str = "/chatbox/input"
    chatbox_send: bool = True
    chatbox_clear: bool = False
    chatbox_max_chars: int = 144
    vrc_mic_intercept: bool = False
    chatbox_include_source: bool = False

    def __post_init__(self) -> None:
        if self.send_port is None:
            object.__setattr__(self, "send_port", self.port)
        else:
            object.__setattr__(self, "port", self.send_port)

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name == "port":
            object.__setattr__(self, "send_port", value)
        elif name == "send_port" and value is not None:
            object.__setattr__(self, "port", value)

    def validate(self) -> None:
        if not self.host:
            raise ValueError("host must be non-empty")
        if self.connection_mode not in {"automatic", "manual", "off"}:
            raise ValueError("connection_mode must be automatic, manual, or off")
        if not (0 < self.port <= 65535):
            raise ValueError("port must be in 1..65535")
        if self.send_port != self.port:
            raise ValueError("send_port must match port compatibility value")
        if not (0 < self.receive_port <= 65535):
            raise ValueError("receive_port must be in 1..65535")
        if not self.chatbox_address or not self.chatbox_address.startswith("/"):
            raise ValueError("chatbox_address must start with '/'")
        if self.chatbox_max_chars <= 0:
            raise ValueError("chatbox_max_chars must be > 0")


@dataclass(slots=True)
class ProviderSettings:
    stt: STTProviderName = STTProviderName.LOCAL_CPU_AUTO
    peer_stt: STTProviderName = STTProviderName.LOCAL_CPU_AUTO
    llm: LLMProviderName = LLMProviderName.OPENROUTER

    def validate(self) -> None:
        if not isinstance(self.stt, STTProviderName):
            raise ValueError("invalid stt provider")
        if not isinstance(self.peer_stt, STTProviderName):
            raise ValueError("invalid peer stt provider")
        if not isinstance(self.llm, LLMProviderName):
            raise ValueError("invalid llm provider")


@dataclass(slots=True)
class SecretsSettings:
    backend: SecretsBackend = SecretsBackend.KEYRING
    encrypted_file_path: str = "secrets.json"

    def validate(self) -> None:
        if not isinstance(self.backend, SecretsBackend):
            raise ValueError("invalid secrets backend")
        if self.backend == SecretsBackend.ENCRYPTED_FILE and not self.encrypted_file_path:
            raise ValueError("encrypted_file_path must be set for encrypted_file backend")


@dataclass(slots=True)
class GeminiSettings:
    llm_model: GeminiLLMModel = GeminiLLMModel.GEMINI_31_FLASH_LITE

    def validate(self) -> None:
        if not isinstance(self.llm_model, GeminiLLMModel):
            raise ValueError("invalid gemini llm model")


@dataclass(slots=True)
class QwenSettings:
    region: QwenRegion = QwenRegion.BEIJING
    llm_model: QwenLLMModel = QwenLLMModel.QWEN_35_PLUS

    def validate(self) -> None:
        if not isinstance(self.region, QwenRegion):
            raise ValueError("invalid qwen region")
        if not isinstance(self.llm_model, QwenLLMModel):
            raise ValueError("invalid qwen llm model")

    def get_llm_base_url(self) -> str:
        if self.region == QwenRegion.BEIJING:
            return "https://dashscope.aliyuncs.com/api/v1"
        return "https://dashscope-intl.aliyuncs.com/api/v1"

    def get_asr_endpoint(self) -> str:
        if self.region == QwenRegion.BEIJING:
            return "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
        return "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"


@dataclass(slots=True)
class DeepSeekSettings:
    llm_model: DeepSeekLLMModel = DeepSeekLLMModel.DEEPSEEK_V4_FLASH

    def validate(self) -> None:
        if not isinstance(self.llm_model, DeepSeekLLMModel):
            raise ValueError("invalid deepseek llm model")


@dataclass(slots=True)
class CerebrasSettings:
    llm_model: CerebrasLLMModel = CerebrasLLMModel.GEMMA_4_31B

    def validate(self) -> None:
        if not isinstance(self.llm_model, CerebrasLLMModel):
            raise ValueError("invalid cerebras llm model")


@dataclass(slots=True)
class LocalLLMSettings:
    backend: LocalLLMBackend = LocalLLMBackend.OLLAMA
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "llama3.1:8b"
    extra_body: dict[str, object] = field(default_factory=_default_local_llm_extra_body)

    def validate(self) -> None:
        if not isinstance(self.backend, LocalLLMBackend):
            raise ValueError("invalid local llm backend")
        self.base_url = _normalize_local_llm_base_url(self.base_url)
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("invalid local llm base url")
        self.model = _normalize_local_llm_model(self.model)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("invalid local llm model")
        if not isinstance(self.extra_body, dict):
            raise ValueError("invalid local llm extra body")
        normalized = {key: value for key, value in self.extra_body.items() if isinstance(key, str)}
        if len(normalized) != len(self.extra_body):
            raise ValueError("local llm extra body keys must be strings")
        lowered = {key.lower() for key in normalized}
        reserved = LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS.intersection(lowered)
        if reserved:
            key = sorted(reserved)[0]
            raise ValueError(f"reserved local llm extra_body key: {key}")
        sensitive = LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS.intersection(lowered)
        if sensitive:
            key = sorted(sensitive)[0]
            raise ValueError(f"sensitive local llm extra_body key: {key}")
        try:
            json.dumps(normalized, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("local llm extra body must be JSON serializable") from exc
        self.extra_body = copy.deepcopy(normalized)


@dataclass(slots=True)
class OpenRouterSettings:
    llm_model: OpenRouterLLMModel = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    routing_mode: OpenRouterRoutingMode = OpenRouterRoutingMode.LATENCY
    provider_routing: OpenRouterProviderRouting = OpenRouterProviderRouting.DEFAULT
    selected_source: OpenRouterCredentialSource = OpenRouterCredentialSource.MANAGED
    selection_alias: OpenRouterSelectionAlias | None = None
    fallback_selection_alias: OpenRouterFallbackSelectionAlias = (
        OpenRouterFallbackSelectionAlias.NONE
    )
    broker_base_url: str = DEFAULT_OPENROUTER_BROKER_BASE_URL

    def __post_init__(self) -> None:
        (
            self.llm_model,
            self.selected_source,
            self.selection_alias,
        ) = _resolve_openrouter_runtime_main_selection(
            selection_alias=self.selection_alias,
            llm_model=self.llm_model,
            selected_source=self.selected_source,
        )

    def validate(self) -> None:
        if not isinstance(self.llm_model, OpenRouterLLMModel):
            raise ValueError("invalid openrouter llm model")
        if not isinstance(self.routing_mode, OpenRouterRoutingMode):
            raise ValueError("invalid openrouter routing mode")
        if not isinstance(self.provider_routing, OpenRouterProviderRouting):
            raise ValueError("invalid openrouter provider routing")
        if not isinstance(self.selected_source, OpenRouterCredentialSource):
            raise ValueError("invalid openrouter credential source")
        if self.selection_alias is not None and not isinstance(
            self.selection_alias, OpenRouterSelectionAlias
        ):
            raise ValueError("invalid openrouter selection alias")
        if self.selection_alias is None and self.selected_source != OpenRouterCredentialSource.NONE:
            raise ValueError("openrouter selection alias is required for active sources")
        if not isinstance(self.fallback_selection_alias, OpenRouterFallbackSelectionAlias):
            raise ValueError("invalid openrouter fallback selection alias")
        if not isinstance(self.broker_base_url, str) or not self.broker_base_url.strip():
            raise ValueError("invalid openrouter broker base url")


def _default_openrouter_settings() -> OpenRouterSettings:
    return OpenRouterSettings(
        provider_routing=OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY,
        selection_alias=OpenRouterSelectionAlias.GEMMA4_26B_31B_MANAGED,
    )


@dataclass(slots=True)
class UiSettings:
    locale: str = "en"
    overlay_enabled: bool = False
    peer_translation_enabled: bool = False
    peer_translation_eula_accepted: bool = False
    integrated_context_enabled: bool = True
    integrated_context_bootstrapped: bool = False
    clipboard_auto_translate_enabled: bool = False
    github_star_prompt_clicked: bool = False
    github_star_prompt_last_shown_at: str | None = None
    github_star_prompt_show_count: int = 0
    github_star_prompt_translation_success_observed: bool = False
    github_star_prompt_eligible_launch_count: int = 0

    def validate(self) -> None:
        if not self.locale:
            raise ValueError("locale must be non-empty")
        if not isinstance(self.clipboard_auto_translate_enabled, bool):
            raise ValueError("clipboard_auto_translate_enabled must be a bool")
        if not isinstance(self.github_star_prompt_clicked, bool):
            raise ValueError("github_star_prompt_clicked must be a bool")
        self.github_star_prompt_last_shown_at = _parse_utc_iso8601_timestamp(
            self.github_star_prompt_last_shown_at
        )
        self.github_star_prompt_show_count = _parse_non_negative_int(
            self.github_star_prompt_show_count
        )
        if not isinstance(self.github_star_prompt_translation_success_observed, bool):
            raise ValueError("github_star_prompt_translation_success_observed must be a bool")
        self.github_star_prompt_eligible_launch_count = _parse_non_negative_int(
            self.github_star_prompt_eligible_launch_count
        )


@dataclass(slots=True)
class DesktopFletOverlayBounds:
    x: int | float | None = None
    y: int | float | None = None
    width: int | float = DESKTOP_FLET_DEFAULT_WIDTH
    height: int | float = DESKTOP_FLET_DEFAULT_HEIGHT

    def validate(self) -> None:
        self.x, self.y = _normalize_desktop_flet_bounds_position(self.x, self.y)
        self.width = _normalize_desktop_flet_dimension(
            self.width,
            default=DESKTOP_FLET_DEFAULT_WIDTH,
            minimum=DESKTOP_FLET_MIN_WIDTH,
        )
        self.height = _normalize_desktop_flet_dimension(
            self.height,
            default=DESKTOP_FLET_DEFAULT_HEIGHT,
            minimum=DESKTOP_FLET_MIN_HEIGHT,
        )


@dataclass(slots=True)
class DesktopFletOverlayPosition:
    x: int | float | None = None
    y: int | float | None = None

    def validate(self) -> None:
        self.x, self.y = _normalize_desktop_flet_position(self.x, self.y)


@dataclass(slots=True, init=False)
class DesktopFletOverlayVisualSettings:
    background_alpha: float = DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA

    def __init__(
        self,
        text_scale: object = None,
        background_alpha: object = DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
        outline_width: object = None,
    ) -> None:
        _ = (text_scale, outline_width)
        self.background_alpha = background_alpha

    def validate(self) -> None:
        self.background_alpha = _normalize_desktop_flet_range(
            self.background_alpha,
            default=DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
            minimum=DESKTOP_FLET_MIN_BACKGROUND_ALPHA,
            maximum=DESKTOP_FLET_MAX_BACKGROUND_ALPHA,
        )

    @property
    def text_scale(self) -> float:
        return DESKTOP_FLET_DEFAULT_TEXT_SCALE

    @text_scale.setter
    def text_scale(self, _value: object) -> None:
        return

    @property
    def outline_width(self) -> None:
        return None

    @outline_width.setter
    def outline_width(self, _value: object) -> None:
        return


@dataclass(slots=True)
class DesktopFletOverlaySettings:
    size_preset: str = DESKTOP_FLET_DEFAULT_SIZE_PRESET
    position: DesktopFletOverlayPosition = field(default_factory=DesktopFletOverlayPosition)
    locked: bool = False
    visual: DesktopFletOverlayVisualSettings = field(
        default_factory=DesktopFletOverlayVisualSettings
    )

    def validate(self) -> None:
        self.size_preset = _parse_desktop_flet_size_preset(self.size_preset)
        if not isinstance(self.position, DesktopFletOverlayPosition):
            self.position = DesktopFletOverlayPosition()
        if not isinstance(self.locked, bool):
            self.locked = False
        if not isinstance(self.visual, DesktopFletOverlayVisualSettings):
            self.visual = DesktopFletOverlayVisualSettings()
        self.position.validate()
        self.visual.validate()

    @property
    def bounds(self) -> DesktopFletOverlayBounds:
        width, height = _desktop_flet_dimensions_for_preset(self.size_preset)
        return DesktopFletOverlayBounds(
            x=self.position.x,
            y=self.position.y,
            width=width,
            height=height,
        )

    @bounds.setter
    def bounds(self, value: object) -> None:
        bounds = _parse_desktop_flet_bounds(value)
        self.size_preset = _nearest_desktop_flet_size_preset(bounds.width, bounds.height)
        self.position = DesktopFletOverlayPosition(x=bounds.x, y=bounds.y)


@dataclass(slots=True)
class OverlaySettings:
    target: str = OVERLAY_TARGET_STEAMVR
    show_translation: bool = True
    show_peer_original: bool = True
    calibration: OverlayCalibration = field(default_factory=OverlayCalibration)
    desktop_flet: DesktopFletOverlaySettings = field(default_factory=DesktopFletOverlaySettings)

    def validate(self) -> None:
        self.target = _parse_overlay_target(self.target)
        if not isinstance(self.show_translation, bool):
            raise ValueError("overlay show_translation must be a bool")
        if not isinstance(self.show_peer_original, bool):
            raise ValueError("overlay show_peer_original must be a bool")
        self.calibration.validate()
        if not isinstance(self.desktop_flet, DesktopFletOverlaySettings):
            self.desktop_flet = DesktopFletOverlaySettings()
        self.desktop_flet.validate()


@dataclass(slots=True)
class ApiKeyVerificationSettings:
    """Stores API key verification status for each provider."""

    deepgram: bool = False
    soniox: bool = False
    google: bool = False
    openrouter: bool = False
    deepseek: bool = False
    alibaba_beijing: bool = False
    alibaba_singapore: bool = False
    cerebras: bool = False

    def validate(self) -> None:
        pass  # No validation needed


@dataclass(slots=True)
class ManagedIdentitySettings:
    installation_id: str = ""
    release_token: str | None = None
    release_token_expires_at: str | None = None
    verified_hardware_hash: str | None = None
    verified_hardware_hash_salt_version: int | None = None
    active_managed_credential_ref: str | None = None
    active_managed_expires_at: str | None = None
    founder_letter_seen_credential_ref: str | None = None
    referral_id: str | None = None
    local_managed_claim_sources: tuple[str, ...] = field(default_factory=tuple)
    pending_delivery_ack_source: str | None = None
    pending_delivery_ack_delivery_id: str | None = None
    pending_delivery_ack_managed_credential_ref: str | None = None
    pending_delivery_ack_expires_at: str | None = None

    def validate(self) -> None:
        if not isinstance(self.installation_id, str):
            raise ValueError("managed installation_id must be a string")
        if self.release_token is not None and not isinstance(self.release_token, str):
            raise ValueError("managed release_token must be a string or None")
        if self.release_token_expires_at is not None and not isinstance(
            self.release_token_expires_at, str
        ):
            raise ValueError("managed release_token_expires_at must be a string or None")
        if self.verified_hardware_hash is not None and not isinstance(
            self.verified_hardware_hash, str
        ):
            raise ValueError("managed verified_hardware_hash must be a string or None")
        if isinstance(self.verified_hardware_hash_salt_version, bool) or (
            self.verified_hardware_hash_salt_version is not None
            and not isinstance(self.verified_hardware_hash_salt_version, int)
        ):
            raise ValueError("managed verified_hardware_hash_salt_version must be an int or None")
        if self.active_managed_credential_ref is not None and not isinstance(
            self.active_managed_credential_ref, str
        ):
            raise ValueError("managed active_managed_credential_ref must be a string or None")
        if self.active_managed_expires_at is not None and not isinstance(
            self.active_managed_expires_at, str
        ):
            raise ValueError("managed active_managed_expires_at must be a string or None")
        if self.founder_letter_seen_credential_ref is not None and not isinstance(
            self.founder_letter_seen_credential_ref, str
        ):
            raise ValueError("managed founder_letter_seen_credential_ref must be a string or None")
        self.referral_id = normalize_owned_referral_id(self.referral_id)
        self.local_managed_claim_sources = normalize_managed_claim_sources(
            self.local_managed_claim_sources
        )
        for key in (
            "pending_delivery_ack_source",
            "pending_delivery_ack_delivery_id",
            "pending_delivery_ack_managed_credential_ref",
            "pending_delivery_ack_expires_at",
        ):
            value = getattr(self, key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"managed {key} must be a string or None")


def _parse_telemetry_consent(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in TELEMETRY_CONSENT_VALUES:
            return normalized
    return "unknown"


def _normalize_telemetry_identifier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_telemetry_sent_dates(value: object) -> list[str]:
    if isinstance(value, str):
        candidates: tuple[object, ...] = (value,)
    elif isinstance(value, (list, tuple, set, frozenset)):
        candidates = tuple(value)
    else:
        candidates = ()
    normalized: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        text = candidate.strip()
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            continue
        date_text = parsed.strftime("%Y-%m-%d")
        if date_text not in normalized:
            normalized.append(date_text)
    return normalized


@dataclass(slots=True)
class TelemetrySettings:
    consent: str = "unknown"

    def validate(self) -> None:
        self.consent = _parse_telemetry_consent(self.consent)


@dataclass(slots=True)
class TelemetryStateSettings:
    anonymous_id: str | None = None
    sent_translation_success_dates_utc: list[str] = field(default_factory=list)

    def validate(self) -> None:
        self.anonymous_id = _normalize_telemetry_identifier(self.anonymous_id)
        self.sent_translation_success_dates_utc = _normalize_telemetry_sent_dates(
            self.sent_translation_success_dates_utc
        )


@dataclass(slots=True)
class AppSettings:
    settings_version: int = SETTINGS_SCHEMA_VERSION
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    languages: LanguageSettings = field(default_factory=LanguageSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    desktop_audio: DesktopAudioSettings = field(default_factory=DesktopAudioSettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    stt: STTSettings = field(default_factory=STTSettings)
    deepgram_stt: DeepgramSTTSettings = field(default_factory=DeepgramSTTSettings)
    qwen_asr_stt: QwenASRSTTSettings = field(default_factory=QwenASRSTTSettings)
    soniox_stt: SonioxSTTSettings = field(default_factory=SonioxSTTSettings)
    custom_stt: CustomSTTSettings = field(default_factory=CustomSTTSettings)
    peer_qwen_asr_stt: PeerQwenASRSTTSettings = field(default_factory=PeerQwenASRSTTSettings)
    peer_soniox_stt: PeerSonioxSTTSettings = field(default_factory=PeerSonioxSTTSettings)
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    openrouter: OpenRouterSettings = field(default_factory=_default_openrouter_settings)
    qwen: QwenSettings = field(default_factory=QwenSettings)
    deepseek: DeepSeekSettings = field(default_factory=DeepSeekSettings)
    cerebras: CerebrasSettings = field(default_factory=CerebrasSettings)
    local_llm: LocalLLMSettings = field(default_factory=LocalLLMSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    osc: OSCSettings = field(default_factory=OSCSettings)
    secrets: SecretsSettings = field(default_factory=SecretsSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    api_key_verified: ApiKeyVerificationSettings = field(default_factory=ApiKeyVerificationSettings)
    managed_identity: ManagedIdentitySettings = field(default_factory=ManagedIdentitySettings)
    telemetry: TelemetrySettings = field(default_factory=TelemetrySettings)
    telemetry_state: TelemetryStateSettings = field(default_factory=TelemetryStateSettings)
    system_prompt: str = ""
    system_prompts: dict[str, str] = field(default_factory=dict)

    @property
    def overlay_calibration(self) -> OverlayCalibration:
        return self.overlay.calibration

    @overlay_calibration.setter
    def overlay_calibration(self, value: OverlayCalibration) -> None:
        self.overlay.calibration = value

    def validate(self) -> None:
        if self.settings_version <= 0:
            raise ValueError("settings_version must be > 0")
        self.provider.validate()
        self.translation.validate()
        self.languages.validate()
        self.audio.validate()
        self.desktop_audio.validate()
        self.overlay.validate()
        self.stt.validate()
        self.deepgram_stt.validate()
        self.qwen_asr_stt.validate()
        self.soniox_stt.validate()
        self.custom_stt.validate()
        self.peer_qwen_asr_stt.validate()
        self.peer_soniox_stt.validate()
        self.gemini.validate()
        self.openrouter.validate()
        self.qwen.validate()
        self.deepseek.validate()
        self.cerebras.validate()
        self.local_llm.validate()
        self.llm.validate()
        self.osc.validate()
        self.secrets.validate()
        self.ui.validate()
        self.api_key_verified.validate()
        self.managed_identity.validate()
        self.telemetry.validate()
        self.telemetry_state.validate()
        for key, value in self.system_prompts.items():
            if not isinstance(key, str):
                raise ValueError("system_prompts keys must be strings")
            if not isinstance(value, str):
                raise ValueError("system_prompts values must be strings")


def _enum_to_value(obj: object) -> object:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _enum_to_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_enum_to_value(v) for v in obj]
    return obj


def _parse_overlay_target(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in OVERLAY_TARGET_VALUES:
            return normalized
    return OVERLAY_TARGET_STEAMVR


def _finite_non_bool_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number):
        return None
    return value


def _clamp_float(value: float, *, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _normalize_desktop_flet_bounds_position(
    x_value: object,
    y_value: object,
) -> tuple[int | float | None, int | float | None]:
    return _normalize_desktop_flet_position(x_value, y_value)


def _normalize_desktop_flet_position(
    x_value: object,
    y_value: object,
) -> tuple[int | float | None, int | float | None]:
    x = _finite_non_bool_number(x_value)
    y = _finite_non_bool_number(y_value)
    if x is None or y is None:
        return None, None
    return x, y


def _normalize_desktop_flet_dimension(
    value: object,
    *,
    default: int,
    minimum: int,
) -> int | float:
    number = _finite_non_bool_number(value)
    if number is None:
        return default
    return max(number, minimum)


def _normalize_desktop_flet_range(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    number = _finite_non_bool_number(value)
    if number is None:
        return default
    return _clamp_float(number, minimum=minimum, maximum=maximum)


def _parse_desktop_flet_size_preset(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in DESKTOP_FLET_SIZE_PRESET_ORDER:
            return normalized
    return DESKTOP_FLET_DEFAULT_SIZE_PRESET


def _desktop_flet_dimensions_for_preset(preset: object) -> tuple[int, int]:
    return DESKTOP_FLET_SIZE_PRESETS[_parse_desktop_flet_size_preset(preset)]


def _valid_desktop_flet_legacy_dimension(value: object) -> int | float | None:
    number = _finite_non_bool_number(value)
    if number is None or number <= 0:
        return None
    return number


def _nearest_desktop_flet_size_preset(width_value: object, height_value: object) -> str:
    width = _valid_desktop_flet_legacy_dimension(width_value)
    height = _valid_desktop_flet_legacy_dimension(height_value)
    if width is None or height is None:
        return DESKTOP_FLET_DEFAULT_SIZE_PRESET

    scores: list[tuple[str, float]] = []
    for preset in DESKTOP_FLET_SIZE_PRESET_ORDER:
        preset_width, preset_height = DESKTOP_FLET_SIZE_PRESETS[preset]
        score = (
            abs(width - preset_width) / preset_width + abs(height - preset_height) / preset_height
        )
        scores.append((preset, score))

    lowest_score = min(score for _preset, score in scores)
    tied = [
        preset
        for preset, score in scores
        if math.isclose(score, lowest_score, rel_tol=0.0, abs_tol=1e-12)
    ]
    if DESKTOP_FLET_DEFAULT_SIZE_PRESET in tied:
        return DESKTOP_FLET_DEFAULT_SIZE_PRESET
    return tied[0]


def _normalize_desktop_flet_outline_width(value: object) -> float | None:
    if value is None:
        return None
    number = _finite_non_bool_number(value)
    if number is None or number <= 0:
        return None
    return _clamp_float(
        number,
        minimum=DESKTOP_FLET_MIN_OUTLINE_WIDTH,
        maximum=DESKTOP_FLET_MAX_OUTLINE_WIDTH,
    )


def _parse_desktop_flet_bounds(value: object) -> DesktopFletOverlayBounds:
    if isinstance(value, DesktopFletOverlayBounds):
        bounds = copy.deepcopy(value)
        bounds.validate()
        return bounds
    data = value if isinstance(value, dict) else {}
    x, y = _normalize_desktop_flet_bounds_position(data.get("x"), data.get("y"))
    return DesktopFletOverlayBounds(
        x=x,
        y=y,
        width=_normalize_desktop_flet_dimension(
            data.get("width"),
            default=DESKTOP_FLET_DEFAULT_WIDTH,
            minimum=DESKTOP_FLET_MIN_WIDTH,
        ),
        height=_normalize_desktop_flet_dimension(
            data.get("height"),
            default=DESKTOP_FLET_DEFAULT_HEIGHT,
            minimum=DESKTOP_FLET_MIN_HEIGHT,
        ),
    )


def _parse_desktop_flet_position(value: object) -> DesktopFletOverlayPosition:
    if isinstance(value, DesktopFletOverlayPosition):
        position = copy.deepcopy(value)
        position.validate()
        return position
    data: dict[str, object]
    if isinstance(value, DesktopFletOverlayBounds):
        data = {"x": value.x, "y": value.y}
    else:
        data = value if isinstance(value, dict) else {}
    x, y = _normalize_desktop_flet_position(data.get("x"), data.get("y"))
    return DesktopFletOverlayPosition(x=x, y=y)


def _parse_desktop_flet_visual(value: object) -> DesktopFletOverlayVisualSettings:
    if isinstance(value, DesktopFletOverlayVisualSettings):
        visual = copy.deepcopy(value)
        visual.validate()
        return visual
    data = value if isinstance(value, dict) else {}
    return DesktopFletOverlayVisualSettings(
        background_alpha=_normalize_desktop_flet_range(
            data.get("background_alpha"),
            default=DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
            minimum=DESKTOP_FLET_MIN_BACKGROUND_ALPHA,
            maximum=DESKTOP_FLET_MAX_BACKGROUND_ALPHA,
        ),
    )


def _parse_desktop_flet_settings(value: object) -> DesktopFletOverlaySettings:
    if isinstance(value, DesktopFletOverlaySettings):
        settings = copy.deepcopy(value)
        settings.validate()
        return settings
    data = value if isinstance(value, dict) else {}
    bounds_data = data.get("bounds") if isinstance(data.get("bounds"), dict) else {}
    size_preset = (
        _parse_desktop_flet_size_preset(data.get("size_preset"))
        if "size_preset" in data
        else _nearest_desktop_flet_size_preset(
            bounds_data.get("width"),
            bounds_data.get("height"),
        )
    )
    position = (
        _parse_desktop_flet_position(data.get("position"))
        if "position" in data
        else _parse_desktop_flet_position(bounds_data)
    )
    return DesktopFletOverlaySettings(
        size_preset=size_preset,
        position=position,
        locked=False,
        visual=_parse_desktop_flet_visual(data.get("visual")),
    )


def _desktop_flet_bounds_to_dict(
    bounds: DesktopFletOverlayBounds,
) -> dict[str, int | float | None]:
    if not isinstance(bounds, DesktopFletOverlayBounds):
        bounds = DesktopFletOverlayBounds()
    bounds = copy.deepcopy(bounds)
    bounds.validate()
    return {
        "x": bounds.x,
        "y": bounds.y,
        "width": bounds.width,
        "height": bounds.height,
    }


def _desktop_flet_visual_to_dict(
    visual: DesktopFletOverlayVisualSettings,
) -> dict[str, float]:
    if not isinstance(visual, DesktopFletOverlayVisualSettings):
        visual = DesktopFletOverlayVisualSettings()
    visual = copy.deepcopy(visual)
    visual.validate()
    return {"background_alpha": visual.background_alpha}


def _desktop_flet_settings_to_dict(settings: DesktopFletOverlaySettings) -> dict[str, object]:
    if not isinstance(settings, DesktopFletOverlaySettings):
        settings = DesktopFletOverlaySettings()
    settings = copy.deepcopy(settings)
    settings.validate()
    return {
        "size_preset": settings.size_preset,
        "position": {"x": settings.position.x, "y": settings.position.y},
        "visual": _desktop_flet_visual_to_dict(settings.visual),
    }


def build_managed_openrouter_byok_target_settings(
    current_settings: AppSettings | None,
) -> AppSettings | None:
    if current_settings is None:
        return None
    if current_settings.provider.llm != LLMProviderName.OPENROUTER:
        return None
    if current_settings.openrouter.selected_source != OpenRouterCredentialSource.MANAGED:
        return None

    openrouter_model = None
    selection_alias = current_settings.openrouter.selection_alias
    if selection_alias is not None:
        profile = get_openrouter_llm_profile(selection_alias.value)
        if profile is not None:
            openrouter_model = profile.openrouter_model
    if openrouter_model is None:
        openrouter_model = current_settings.openrouter.llm_model.value

    alias_value = get_openrouter_selection_alias_for_model_and_source(
        openrouter_model,
        OpenRouterCredentialSource.BYOK.value,
    )
    if alias_value is None:
        return None

    target_settings = copy.deepcopy(current_settings)
    target_settings.provider.llm = LLMProviderName.OPENROUTER
    target_settings.openrouter.selection_alias = OpenRouterSelectionAlias(alias_value)
    target_settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    target_settings.openrouter.llm_model = OpenRouterLLMModel(openrouter_model)
    target_settings.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT
    target_settings.translation.connection = TranslationConnection.OPENROUTER
    target_settings.translation.connection_history[target_settings.translation.model.value] = (
        TranslationConnection.OPENROUTER
    )
    return target_settings


def to_dict(settings: AppSettings) -> dict[str, Any]:
    settings = copy.deepcopy(settings)
    if _translation_settings_is_exact_default(settings.translation):
        inferred_translation = _derive_translation_settings_from_runtime(
            settings,
            history=settings.translation.connection_history,
        )
        if not _translation_settings_is_exact_default(inferred_translation):
            settings.translation = inferred_translation
    materialize_translation_settings(settings)
    (
        normalized_openrouter_model,
        normalized_openrouter_selected_source,
        normalized_openrouter_selection_alias,
    ) = _resolve_openrouter_runtime_main_selection(
        selection_alias=settings.openrouter.selection_alias,
        llm_model=settings.openrouter.llm_model,
        selected_source=settings.openrouter.selected_source,
    )
    normalized_openrouter_selection_alias_value = (
        normalized_openrouter_selection_alias.value
        if normalized_openrouter_selection_alias is not None
        else None
    )

    data: dict[str, Any] = {
        "settings_version": settings.settings_version,
        "provider": {
            "stt": settings.provider.stt.value,
            "peer_stt": _parse_peer_stt_provider(settings.provider.peer_stt.value).value,
            "llm": settings.provider.llm.value,
        },
        "translation": _translation_settings_to_dict(settings.translation),
        "languages": {
            "source_language": settings.languages.source_language,
            "target_language": settings.languages.target_language,
            "peer_source_language": settings.languages.peer_source_language,
            "peer_target_language": settings.languages.peer_target_language,
            "peer_source_mode": settings.languages.peer_source_mode,
            "peer_expected_languages": settings.languages.peer_expected_languages,
            "recent_source_languages": settings.languages.recent_source_languages,
            "recent_target_languages": settings.languages.recent_target_languages,
        },
        "audio": {
            "internal_sample_rate_hz": settings.audio.internal_sample_rate_hz,
            "internal_channels": settings.audio.internal_channels,
            "ring_buffer_ms": settings.audio.ring_buffer_ms,
            "input_host_api": settings.audio.input_host_api,
            "input_device": settings.audio.input_device,
        },
        "desktop_audio": {
            "output_device": settings.desktop_audio.output_device,
            "vad_speech_threshold": settings.desktop_audio.vad_speech_threshold,
            "vad_hangover_ms": settings.desktop_audio.vad_hangover_ms,
            "vad_pre_roll_ms": settings.desktop_audio.vad_pre_roll_ms,
        },
        "overlay": {
            "target": _parse_overlay_target(settings.overlay.target),
            "show_translation": settings.overlay.show_translation,
            "show_peer_original": settings.overlay.show_peer_original,
            "calibration": settings.overlay.calibration.to_dict(),
            "desktop_flet": _desktop_flet_settings_to_dict(settings.overlay.desktop_flet),
        },
        "stt": {
            "drain_timeout_s": settings.stt.drain_timeout_s,
            "vad_speech_threshold": settings.stt.vad_speech_threshold,
            "low_latency_mode": settings.stt.low_latency_mode,
            "low_latency_vad_hangover_ms": settings.stt.low_latency_vad_hangover_ms,
            "low_latency_merge_gap_ms": settings.stt.low_latency_merge_gap_ms,
            "low_latency_spec_retry_max": settings.stt.low_latency_spec_retry_max,
            "custom_vocabulary_enabled": settings.stt.custom_vocabulary_enabled,
            "custom_terms": _parse_custom_terms(settings.stt.custom_terms),
            "gpu_device_id": settings.stt.gpu_device_id.strip(),
        },
        "deepgram_stt": {
            "model": settings.deepgram_stt.model,
        },
        "qwen_asr_stt": {
            "model": settings.qwen_asr_stt.model,
            "endpoint": settings.qwen.get_asr_endpoint(),
        },
        "soniox_stt": {
            "model": settings.soniox_stt.model,
            "endpoint": settings.soniox_stt.endpoint,
            "keepalive_interval_s": settings.soniox_stt.keepalive_interval_s,
            "trailing_silence_ms": settings.soniox_stt.trailing_silence_ms,
        },
        "custom_stt": {
            "mode": settings.custom_stt.mode,
            "compatibility": settings.custom_stt.compatibility,
            "endpoint": settings.custom_stt.endpoint,
            "model": settings.custom_stt.model,
            "extra": copy.deepcopy(settings.custom_stt.extra),
        },
        "gemini": {
            "llm_model": settings.gemini.llm_model.value,
        },
        "openrouter": {
            "llm_model": normalized_openrouter_model.value,
            "routing_mode": settings.openrouter.routing_mode.value,
            "provider_routing": settings.openrouter.provider_routing.value,
            "selected_source": normalized_openrouter_selected_source.value,
            "selection_alias": normalized_openrouter_selection_alias_value,
            "fallback_selection_alias": OpenRouterFallbackSelectionAlias.NONE.value,
            "broker_base_url": settings.openrouter.broker_base_url,
        },
        "qwen": {
            "region": settings.qwen.region.value,
            "llm_model": settings.qwen.llm_model.value,
        },
        "deepseek": {
            "llm_model": settings.deepseek.llm_model.value,
        },
        "cerebras": {
            "llm_model": settings.cerebras.llm_model.value,
        },
        "local_llm": {
            "backend": settings.local_llm.backend.value,
            "base_url": _parse_local_llm_base_url(settings.local_llm.base_url),
            "model": _parse_local_llm_model(settings.local_llm.model),
            "extra_body": _parse_local_llm_extra_body(settings.local_llm.extra_body),
        },
        "llm": {"concurrency_limit": settings.llm.concurrency_limit},
        "osc": {
            "host": settings.osc.host,
            "port": settings.osc.port,
            "chatbox_address": settings.osc.chatbox_address,
            "chatbox_send": settings.osc.chatbox_send,
            "chatbox_clear": settings.osc.chatbox_clear,
            "chatbox_max_chars": settings.osc.chatbox_max_chars,
            "vrc_mic_intercept": settings.osc.vrc_mic_intercept,
            "chatbox_include_source": settings.osc.chatbox_include_source,
        },
        "secrets": {
            "backend": settings.secrets.backend.value,
            "encrypted_file_path": settings.secrets.encrypted_file_path,
        },
        "ui": {
            "locale": settings.ui.locale,
            "peer_translation_eula_accepted": settings.ui.peer_translation_eula_accepted,
            "integrated_context_enabled": settings.ui.integrated_context_enabled,
            "integrated_context_bootstrapped": settings.ui.integrated_context_bootstrapped,
            "clipboard_auto_translate_enabled": settings.ui.clipboard_auto_translate_enabled,
            "github_star_prompt_clicked": settings.ui.github_star_prompt_clicked,
            "github_star_prompt_last_shown_at": _parse_utc_iso8601_timestamp(
                settings.ui.github_star_prompt_last_shown_at
            ),
            "github_star_prompt_show_count": _parse_non_negative_int(
                settings.ui.github_star_prompt_show_count
            ),
            "github_star_prompt_translation_success_observed": (
                settings.ui.github_star_prompt_translation_success_observed
            ),
            "github_star_prompt_eligible_launch_count": _parse_non_negative_int(
                settings.ui.github_star_prompt_eligible_launch_count
            ),
        },
        "telemetry": {"consent": _parse_telemetry_consent(settings.telemetry.consent)},
        "telemetry_state": {
            "anonymous_id": _normalize_telemetry_identifier(settings.telemetry_state.anonymous_id),
            "sent_translation_success_dates_utc": _normalize_telemetry_sent_dates(
                settings.telemetry_state.sent_translation_success_dates_utc
            ),
        },
        "api_key_verified": {
            "deepgram": settings.api_key_verified.deepgram,
            "soniox": settings.api_key_verified.soniox,
            "google": settings.api_key_verified.google,
            "openrouter": settings.api_key_verified.openrouter,
            "deepseek": settings.api_key_verified.deepseek,
            "alibaba_beijing": settings.api_key_verified.alibaba_beijing,
            "alibaba_singapore": settings.api_key_verified.alibaba_singapore,
            "cerebras": settings.api_key_verified.cerebras,
        },
        "managed_identity": {
            "installation_id": settings.managed_identity.installation_id,
            "release_token": settings.managed_identity.release_token,
            "release_token_expires_at": settings.managed_identity.release_token_expires_at,
            "verified_hardware_hash": settings.managed_identity.verified_hardware_hash,
            "verified_hardware_hash_salt_version": (
                settings.managed_identity.verified_hardware_hash_salt_version
            ),
            "active_managed_credential_ref": (
                settings.managed_identity.active_managed_credential_ref
            ),
            "active_managed_expires_at": settings.managed_identity.active_managed_expires_at,
            "founder_letter_seen_credential_ref": (
                settings.managed_identity.founder_letter_seen_credential_ref
            ),
            "referral_id": normalize_owned_referral_id(settings.managed_identity.referral_id),
            "local_managed_claim_sources": list(
                normalize_managed_claim_sources(
                    settings.managed_identity.local_managed_claim_sources
                )
            ),
            "pending_delivery_ack_source": settings.managed_identity.pending_delivery_ack_source,
            "pending_delivery_ack_delivery_id": (
                settings.managed_identity.pending_delivery_ack_delivery_id
            ),
            "pending_delivery_ack_managed_credential_ref": (
                settings.managed_identity.pending_delivery_ack_managed_credential_ref
            ),
            "pending_delivery_ack_expires_at": (
                settings.managed_identity.pending_delivery_ack_expires_at
            ),
        },
        "system_prompt": settings.system_prompt,
    }
    data["osc"].update(
        {
            "connection_mode": settings.osc.connection_mode,
            "send_port": settings.osc.send_port,
            "receive_port": settings.osc.receive_port,
        }
    )
    return _enum_to_value(data)  # type: ignore[return-value]


def with_telemetry_consent(
    settings: AppSettings,
    consent: str,
    *,
    identifier_factory: object = new_anonymous_telemetry_identifier,
) -> AppSettings:
    updated = copy.deepcopy(settings)
    normalized_consent = _parse_telemetry_consent(consent)
    updated.telemetry.consent = normalized_consent
    if normalized_consent == "decline":
        updated.telemetry_state.anonymous_id = None
        updated.telemetry_state.sent_translation_success_dates_utc = []
    elif normalized_consent == "allow":
        factory = (
            identifier_factory
            if callable(identifier_factory)
            else new_anonymous_telemetry_identifier
        )
        updated.telemetry_state.anonymous_id = _normalize_telemetry_identifier(
            updated.telemetry_state.anonymous_id
        ) or str(factory())
        updated.telemetry_state.sent_translation_success_dates_utc = (
            _normalize_telemetry_sent_dates(
                updated.telemetry_state.sent_translation_success_dates_utc
            )
        )
    updated.validate()
    return updated


def ensure_telemetry_default_allow(
    settings: AppSettings,
    *,
    identifier_factory: object = new_anonymous_telemetry_identifier,
) -> AppSettings:
    """Map unknown consent to allow and mint an anonymous id when needed."""
    consent = _parse_telemetry_consent(settings.telemetry.consent)
    if consent == "decline":
        return settings
    if consent == "allow" and _normalize_telemetry_identifier(
        settings.telemetry_state.anonymous_id
    ):
        return settings
    return with_telemetry_consent(settings, "allow", identifier_factory=identifier_factory)


def _parse_custom_stt_settings(value: object) -> CustomSTTSettings:
    from puripuly_heart.core.stt.custom import (
        CUSTOM_STT_COMPAT_OPENAI_TRANSCRIPTION,
        CUSTOM_STT_MODE_OFFLINE,
        normalize_custom_stt_compatibility,
        normalize_custom_stt_endpoint,
        normalize_custom_stt_extra,
        normalize_custom_stt_mode,
        normalize_custom_stt_model,
    )

    raw = value if isinstance(value, dict) else {}
    mode = normalize_custom_stt_mode(raw.get("mode"), default=CUSTOM_STT_MODE_OFFLINE)
    compatibility = normalize_custom_stt_compatibility(
        raw.get("compatibility"),
        mode=mode,
        default=CUSTOM_STT_COMPAT_OPENAI_TRANSCRIPTION if mode == CUSTOM_STT_MODE_OFFLINE else None,
    )
    settings = CustomSTTSettings(
        mode=mode,
        compatibility=compatibility,
        endpoint=normalize_custom_stt_endpoint(raw.get("endpoint")),
        model=normalize_custom_stt_model(raw.get("model")),
        extra=normalize_custom_stt_extra(raw.get("extra")),
    )
    settings.validate()
    return settings


def _parse_stt_provider(value: str) -> STTProviderName:
    """Parse STT provider, mapping legacy values to supported providers."""
    if value == "alibaba":
        return STTProviderName.QWEN_ASR
    try:
        return STTProviderName(value)
    except ValueError:
        return STTProviderName.DEEPGRAM


def _parse_peer_stt_provider(value: str) -> STTProviderName:
    return _parse_stt_provider(value)


def _parse_llm_provider(value: object) -> LLMProviderName:
    if isinstance(value, str):
        normalized = value.strip()
        try:
            return LLMProviderName(normalized)
        except ValueError:
            pass
    return LLMProviderName.GEMINI


def _parse_qwen_llm_model(value: object) -> QwenLLMModel:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized == "qwen-mt-flash":
            normalized = QwenLLMModel.QWEN_35_PLUS.value
        try:
            return QwenLLMModel(normalized)
        except ValueError:
            pass
    return QwenLLMModel.QWEN_35_PLUS


def _parse_gemini_llm_model(value: object) -> GeminiLLMModel:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in {"gemini-3-flash", "gemini-3-flash-preview"}:
            normalized = GeminiLLMModel.GEMINI_37_FLASH.value
        elif normalized in {"gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview"}:
            normalized = GeminiLLMModel.GEMINI_31_FLASH_LITE.value
        try:
            return GeminiLLMModel(normalized)
        except ValueError:
            pass
    return GeminiLLMModel.GEMINI_31_FLASH_LITE


def _parse_deepseek_llm_model(value: object) -> DeepSeekLLMModel:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized == "deepseek-chat":
            normalized = DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value
        if normalized == "deepseek-v4-pro":
            normalized = DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value
        try:
            return DeepSeekLLMModel(normalized)
        except ValueError:
            pass
    return DeepSeekLLMModel.DEEPSEEK_V4_FLASH


def _parse_cerebras_llm_model(value: object) -> CerebrasLLMModel:
    if isinstance(value, str):
        normalized = value.strip()
        try:
            return CerebrasLLMModel(normalized)
        except ValueError:
            pass
    return CerebrasLLMModel.GEMMA_4_31B


def _parse_openrouter_llm_model(value: object) -> OpenRouterLLMModel:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized == LEGACY_OPENROUTER_MODEL_DEEPSEEK_V4_FLASH:
            normalized = OPENROUTER_MODEL_DEEPSEEK_V4_FLASH
        if normalized == "google/gemini-3-flash-preview":
            normalized = OPENROUTER_MODEL_GEMINI_37_FLASH
        try:
            return OpenRouterLLMModel(normalized)
        except ValueError:
            pass
    return OpenRouterLLMModel.GEMMA_4_26B_A4B_IT


def _parse_openrouter_routing_mode(value: object) -> OpenRouterRoutingMode:
    if isinstance(value, str):
        normalized = value.strip()
        try:
            return OpenRouterRoutingMode(normalized)
        except ValueError:
            pass
    return OpenRouterRoutingMode.LATENCY


def _parse_openrouter_provider_routing(value: object) -> OpenRouterProviderRouting:
    if isinstance(value, str):
        normalized = value.strip()
        try:
            return OpenRouterProviderRouting(normalized)
        except ValueError:
            pass
    return OpenRouterProviderRouting.DEFAULT


def _parse_openrouter_credential_source(
    value: object,
    *,
    fallback: OpenRouterCredentialSource = OpenRouterCredentialSource.NONE,
) -> OpenRouterCredentialSource:
    if isinstance(value, str):
        normalized = value.strip()
        try:
            return OpenRouterCredentialSource(normalized)
        except ValueError:
            pass
    return fallback


def _parse_openrouter_selection_alias_profile(value: object):
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return get_openrouter_llm_profile(normalized)
    return None


def _derive_openrouter_selection_alias(
    llm_model: OpenRouterLLMModel,
    selected_source: OpenRouterCredentialSource,
    models: tuple[str, ...] = (),
) -> OpenRouterSelectionAlias:
    alias = get_openrouter_selection_alias_for_model_and_source(
        llm_model.value,
        selected_source.value,
    )
    if models:
        alias = openrouter_alias_for_fields(
            model=llm_model.value,
            source=selected_source.value,
            models=models,
        )
    if alias is None:
        alias = (
            OpenRouterSelectionAlias.GEMMA4_MANAGED.value
            if selected_source == OpenRouterCredentialSource.MANAGED
            else OpenRouterSelectionAlias.GEMMA4_BYOK.value
        )
    return OpenRouterSelectionAlias(alias)


def _parse_openrouter_selection_alias(
    value: object,
    *,
    llm_model: OpenRouterLLMModel,
    selected_source: OpenRouterCredentialSource,
) -> OpenRouterSelectionAlias:
    profile = _parse_openrouter_selection_alias_profile(value)
    if profile is not None and profile.openrouter_model is not None:
        canonical_alias = openrouter_alias_for_fields(
            model=profile.openrouter_model,
            source=profile.openrouter_source,
            models=profile.openrouter_models,
        )
        if canonical_alias is not None:
            return OpenRouterSelectionAlias(canonical_alias)
    return _derive_openrouter_selection_alias(llm_model, selected_source)


def _parse_openrouter_fallback_selection_alias(value: object) -> OpenRouterFallbackSelectionAlias:
    if isinstance(value, str):
        normalized = normalize_openrouter_fallback_selection_alias(value)
        if normalized is not None:
            try:
                return OpenRouterFallbackSelectionAlias(normalized)
            except ValueError:
                pass
    return OpenRouterFallbackSelectionAlias.NONE


def _resolve_openrouter_runtime_main_selection(
    *,
    selection_alias: object,
    llm_model: object,
    selected_source: object,
) -> tuple[
    OpenRouterLLMModel,
    OpenRouterCredentialSource,
    OpenRouterSelectionAlias | None,
]:
    selection_profile = _parse_openrouter_selection_alias_profile(selection_alias)
    if selection_profile is not None and selection_profile.openrouter_model is not None:
        resolved_llm_model = _parse_openrouter_llm_model(selection_profile.openrouter_model)
        resolved_selected_source = _parse_openrouter_credential_source(
            selection_profile.openrouter_source
        )
        if (
            resolved_selected_source == OpenRouterCredentialSource.NONE
            and _parse_openrouter_credential_source(selected_source)
            != OpenRouterCredentialSource.NONE
        ):
            resolved_selected_source = _parse_openrouter_credential_source(selected_source)
        if resolved_selected_source == OpenRouterCredentialSource.NONE:
            return resolved_llm_model, resolved_selected_source, None
        canonical_selection_alias = _derive_openrouter_selection_alias(
            resolved_llm_model,
            resolved_selected_source,
            selection_profile.openrouter_models,
        )
        canonical_profile = get_openrouter_llm_profile(canonical_selection_alias.value)
        assert canonical_profile is not None and canonical_profile.openrouter_model is not None
        return (
            _parse_openrouter_llm_model(canonical_profile.openrouter_model),
            _parse_openrouter_credential_source(canonical_profile.openrouter_source),
            canonical_selection_alias,
        )

    normalized_llm_model = _parse_openrouter_llm_model(llm_model)
    normalized_selected_source = _parse_openrouter_credential_source(selected_source)
    if normalized_selected_source == OpenRouterCredentialSource.NONE:
        return normalized_llm_model, normalized_selected_source, None
    normalized_selection_alias = _derive_openrouter_selection_alias(
        normalized_llm_model, normalized_selected_source
    )
    normalized_profile = get_openrouter_llm_profile(normalized_selection_alias.value)
    assert normalized_profile is not None and normalized_profile.openrouter_model is not None
    return (
        _parse_openrouter_llm_model(normalized_profile.openrouter_model),
        _parse_openrouter_credential_source(normalized_profile.openrouter_source),
        normalized_selection_alias,
    )


def _parse_openrouter_broker_base_url(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return DEFAULT_OPENROUTER_BROKER_BASE_URL


def _parse_local_llm_backend(value: object) -> LocalLLMBackend:
    if isinstance(value, str):
        try:
            return LocalLLMBackend(value.strip())
        except ValueError:
            pass
    return LocalLLMBackend.OLLAMA


def _normalize_local_llm_base_url(value: str) -> str:
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


def _parse_local_llm_base_url(value: object) -> str:
    if isinstance(value, str):
        try:
            return _normalize_local_llm_base_url(value)
        except ValueError:
            pass
    return "http://127.0.0.1:11434/v1"


def _normalize_local_llm_model(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid local llm model")
    normalized = value.strip()
    if not normalized:
        raise ValueError("invalid local llm model")
    return normalized


def _parse_local_llm_model(value: object) -> str:
    if isinstance(value, str):
        try:
            return _normalize_local_llm_model(value)
        except ValueError:
            pass
    return "llama3.1:8b"


def _parse_local_llm_extra_body(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return _default_local_llm_extra_body()
    normalized = {key: val for key, val in value.items() if isinstance(key, str)}
    lowered = {key.lower() for key in normalized}
    if LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS.intersection(lowered):
        return _default_local_llm_extra_body()
    if LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS.intersection(lowered):
        return _default_local_llm_extra_body()
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError):
        return _default_local_llm_extra_body()
    return copy.deepcopy(normalized)


def _normalize_local_llm_data(data: dict[str, Any]) -> bool:
    raw_local_llm = data.get("local_llm")
    local_llm_data = raw_local_llm if isinstance(raw_local_llm, dict) else {}
    normalized = {
        "backend": _parse_local_llm_backend(local_llm_data.get("backend")).value,
        "base_url": _parse_local_llm_base_url(local_llm_data.get("base_url")),
        "model": _parse_local_llm_model(local_llm_data.get("model")),
        "extra_body": _parse_local_llm_extra_body(local_llm_data.get("extra_body")),
    }
    if raw_local_llm != normalized:
        data["local_llm"] = normalized
        return True
    return False


def _loaded_llm_provider(settings_data: dict[str, Any]) -> LLMProviderName:
    provider_data = settings_data.get("provider")
    provider_llm_value = (
        provider_data.get("llm", LLMProviderName.GEMINI.value)
        if isinstance(provider_data, dict)
        else LLMProviderName.GEMINI.value
    )
    return _parse_llm_provider(provider_llm_value)


def _default_openrouter_credential_source_value(data: dict[str, Any]) -> OpenRouterCredentialSource:
    if _loaded_llm_provider(data) == LLMProviderName.OPENROUTER:
        return OpenRouterCredentialSource.BYOK
    return OpenRouterCredentialSource.NONE


def _get_raw_openrouter_selected_source(openrouter_data: dict[str, Any]) -> object:
    if "selected_source" in openrouter_data:
        return openrouter_data["selected_source"]
    if "credential_source" in openrouter_data:
        return openrouter_data["credential_source"]
    if "selected_credential_source" in openrouter_data:
        return openrouter_data["selected_credential_source"]
    return None


def _resolve_openrouter_main_selection(
    openrouter_data: dict[str, Any],
    settings_data: dict[str, Any],
) -> tuple[
    OpenRouterLLMModel,
    OpenRouterCredentialSource,
    OpenRouterSelectionAlias | None,
]:
    raw_selected_source = _parse_openrouter_credential_source(
        _get_raw_openrouter_selected_source(openrouter_data),
        fallback=_default_openrouter_credential_source_value(settings_data),
    )
    if (
        _loaded_llm_provider(settings_data) == LLMProviderName.OPENROUTER
        and raw_selected_source == OpenRouterCredentialSource.NONE
    ):
        raw_selected_source = _default_openrouter_credential_source_value(settings_data)
    selection_profile = _parse_openrouter_selection_alias_profile(
        openrouter_data.get("selection_alias")
    )
    if raw_selected_source == OpenRouterCredentialSource.NONE:
        llm_default = (
            selection_profile.openrouter_model
            if selection_profile is not None and selection_profile.openrouter_model is not None
            else OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
        )
        llm_model = _parse_openrouter_llm_model(openrouter_data.get("llm_model", llm_default))
        return llm_model, raw_selected_source, None

    if selection_profile is not None and selection_profile.openrouter_model is not None:
        llm_model = _parse_openrouter_llm_model(selection_profile.openrouter_model)
        selected_source = _parse_openrouter_credential_source(
            selection_profile.openrouter_source,
            fallback=_default_openrouter_credential_source_value(settings_data),
        )
        if (
            selected_source == OpenRouterCredentialSource.NONE
            and raw_selected_source != OpenRouterCredentialSource.NONE
        ):
            selected_source = raw_selected_source
        if selected_source == OpenRouterCredentialSource.NONE:
            return llm_model, selected_source, None
        selection_alias = _derive_openrouter_selection_alias(
            llm_model,
            selected_source,
            selection_profile.openrouter_models,
        )
        return llm_model, selected_source, selection_alias

    llm_model = _parse_openrouter_llm_model(openrouter_data.get("llm_model"))
    selected_source = raw_selected_source
    selection_alias = _derive_openrouter_selection_alias(llm_model, selected_source)
    return llm_model, selected_source, selection_alias


def _derive_translation_settings_from_runtime_values(
    *,
    provider_llm: LLMProviderName,
    openrouter_model: OpenRouterLLMModel,
    openrouter_selected_source: OpenRouterCredentialSource,
    openrouter_provider_routing: OpenRouterProviderRouting,
    gemini_model: GeminiLLMModel,
    qwen_model: QwenLLMModel,
    cerebras_model: CerebrasLLMModel,
    history: object = None,
) -> TranslationSettings:
    normalized_history = _parse_translation_connection_history(history)

    if provider_llm == LLMProviderName.OPENROUTER:
        if openrouter_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT:
            translation_model = (
                TranslationModel.GEMMA4_26B_31B
                if openrouter_provider_routing == OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY
                else TranslationModel.GEMMA4
            )
            return _normalize_translation_settings(
                model=translation_model,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_selected_source,
                    model=translation_model,
                    provider_routing=openrouter_provider_routing,
                ),
                history=normalized_history,
            )
        if openrouter_model == OpenRouterLLMModel.GEMMA_4_31B_IT:
            return _normalize_translation_settings(
                model=TranslationModel.GEMMA4_31B,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_selected_source,
                    model=TranslationModel.GEMMA4_31B,
                    provider_routing=openrouter_provider_routing,
                ),
                history=normalized_history,
            )
        if openrouter_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH:
            return _normalize_translation_settings(
                model=TranslationModel.DEEPSEEK_V4_FLASH,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_selected_source,
                    model=TranslationModel.DEEPSEEK_V4_FLASH,
                    provider_routing=openrouter_provider_routing,
                ),
                history=normalized_history,
            )
        if openrouter_model == OpenRouterLLMModel.QWEN_35_FLASH_02_23:
            return _normalize_translation_settings(
                model=TranslationModel.DEEPSEEK_V4_FLASH,
                connection=_history_connection_or_default(
                    TranslationModel.DEEPSEEK_V4_FLASH,
                    normalized_history,
                ),
                history=normalized_history,
            )
        if openrouter_model == OpenRouterLLMModel.GEMINI_37_FLASH:
            return _normalize_translation_settings(
                model=TranslationModel.GEMINI_37_FLASH,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_selected_source,
                    model=TranslationModel.GEMINI_37_FLASH,
                    provider_routing=openrouter_provider_routing,
                ),
                history=normalized_history,
            )
        if openrouter_model == OpenRouterLLMModel.GEMINI_31_FLASH_LITE:
            return _normalize_translation_settings(
                model=TranslationModel.GEMINI_31_FLASH_LITE,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_selected_source,
                    model=TranslationModel.GEMINI_31_FLASH_LITE,
                    provider_routing=openrouter_provider_routing,
                ),
                history=normalized_history,
            )

    if provider_llm == LLMProviderName.LOCAL_LLM:
        return _normalize_translation_settings(
            model=TranslationModel.LOCAL_LLM,
            connection=TranslationConnection.OLLAMA,
            history=normalized_history,
        )

    if provider_llm == LLMProviderName.CEREBRAS:
        return _normalize_translation_settings(
            model=TranslationModel.GEMMA4_31B,
            connection=TranslationConnection.CEREBRAS,
            history=normalized_history,
        )

    if provider_llm == LLMProviderName.DEEPSEEK:
        return _normalize_translation_settings(
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OFFICIAL_BYOK,
            history=normalized_history,
        )

    if provider_llm == LLMProviderName.QWEN:
        if qwen_model == QwenLLMModel.QWEN_35_FLASH:
            return _normalize_translation_settings(
                model=TranslationModel.DEEPSEEK_V4_FLASH,
                connection=_history_connection_or_default(
                    TranslationModel.DEEPSEEK_V4_FLASH,
                    normalized_history,
                ),
                history=normalized_history,
            )
        return _normalize_translation_settings(
            model=TranslationModel.QWEN_35_PLUS,
            connection=TranslationConnection.OFFICIAL_BYOK,
            history=normalized_history,
        )

    if gemini_model == GeminiLLMModel.GEMINI_37_FLASH:
        return _normalize_translation_settings(
            model=TranslationModel.GEMINI_37_FLASH,
            connection=TranslationConnection.OFFICIAL_BYOK,
            history=normalized_history,
        )
    return _normalize_translation_settings(
        model=TranslationModel.GEMINI_31_FLASH_LITE,
        connection=TranslationConnection.OFFICIAL_BYOK,
        history=normalized_history,
    )


def _derive_translation_settings_from_runtime(
    settings: AppSettings,
    history: object = None,
) -> TranslationSettings:
    return _derive_translation_settings_from_runtime_values(
        provider_llm=settings.provider.llm,
        openrouter_model=settings.openrouter.llm_model,
        openrouter_selected_source=settings.openrouter.selected_source,
        openrouter_provider_routing=settings.openrouter.provider_routing,
        gemini_model=settings.gemini.llm_model,
        qwen_model=settings.qwen.llm_model,
        cerebras_model=settings.cerebras.llm_model,
        history=history,
    )


def materialize_translation_settings(settings: AppSettings) -> AppSettings:
    settings.translation = _normalize_translation_settings(
        model=_parse_translation_model(settings.translation.model),
        connection=_parse_translation_connection(settings.translation.connection),
        fallback=settings.translation.fallback,
        history=settings.translation.connection_history,
        http_extension_id=settings.translation.http_extension_id,
        previous_llm_model=settings.translation.previous_llm_model,
    )
    model = settings.translation.model
    connection = settings.translation.connection

    if model == TranslationModel.CUSTOM_HTTP:
        return settings

    if model == TranslationModel.GEMMA4_26B_31B:
        settings.provider.llm = LLMProviderName.OPENROUTER
        settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
        settings.openrouter.provider_routing = OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY
        settings.openrouter.selected_source = (
            OpenRouterCredentialSource.MANAGED
            if connection == TranslationConnection.MANAGED
            else OpenRouterCredentialSource.BYOK
        )
        settings.openrouter.selection_alias = OpenRouterSelectionAlias(
            (
                OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED
                if connection == TranslationConnection.MANAGED
                else OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK
            )
        )
        return settings

    if model == TranslationModel.GEMMA4_31B:
        if connection == TranslationConnection.CEREBRAS:
            settings.provider.llm = LLMProviderName.CEREBRAS
            settings.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT
            settings.cerebras.llm_model = CerebrasLLMModel.GEMMA_4_31B
            return settings
        settings.provider.llm = LLMProviderName.OPENROUTER
        settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_31B_IT
        settings.openrouter.provider_routing = OpenRouterProviderRouting.GEMMA4_31B_LATENCY
        settings.openrouter.selected_source = (
            OpenRouterCredentialSource.MANAGED
            if connection == TranslationConnection.MANAGED
            else OpenRouterCredentialSource.BYOK
        )
        settings.openrouter.selection_alias = OpenRouterSelectionAlias(
            (
                OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED
                if connection == TranslationConnection.MANAGED
                else OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK
            )
        )
        return settings

    if model == TranslationModel.GEMMA4:
        settings.provider.llm = LLMProviderName.OPENROUTER
        settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
        settings.openrouter.provider_routing = OpenRouterProviderRouting.GEMMA4_26B_LATENCY
        settings.openrouter.selected_source = (
            OpenRouterCredentialSource.MANAGED
            if connection == TranslationConnection.MANAGED
            else OpenRouterCredentialSource.BYOK
        )
        settings.openrouter.selection_alias = _derive_openrouter_selection_alias(
            settings.openrouter.llm_model,
            settings.openrouter.selected_source,
        )
        return settings

    if model == TranslationModel.DEEPSEEK_V4_FLASH:
        if connection == TranslationConnection.OFFICIAL_BYOK:
            settings.provider.llm = LLMProviderName.DEEPSEEK
            settings.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT
            settings.deepseek.llm_model = DeepSeekLLMModel.DEEPSEEK_V4_FLASH
            return settings
        settings.provider.llm = LLMProviderName.OPENROUTER
        settings.openrouter.llm_model = OpenRouterLLMModel.DEEPSEEK_V4_FLASH
        settings.openrouter.provider_routing = (
            OpenRouterProviderRouting.DEEPSEEK_ONLY
            if connection == TranslationConnection.MANAGED_CHINA
            else OpenRouterProviderRouting.DEFAULT
        )
        settings.openrouter.selected_source = (
            OpenRouterCredentialSource.MANAGED
            if connection in (TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA)
            else OpenRouterCredentialSource.BYOK
        )
        settings.openrouter.selection_alias = _derive_openrouter_selection_alias(
            settings.openrouter.llm_model,
            settings.openrouter.selected_source,
        )
        return settings

    if model == TranslationModel.GEMINI_37_FLASH:
        if connection == TranslationConnection.OPENROUTER:
            settings.provider.llm = LLMProviderName.OPENROUTER
            settings.openrouter.llm_model = OpenRouterLLMModel.GEMINI_37_FLASH
            settings.openrouter.provider_routing = OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY
            settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
            settings.openrouter.selection_alias = _derive_openrouter_selection_alias(
                settings.openrouter.llm_model,
                settings.openrouter.selected_source,
            )
            return settings
        settings.provider.llm = LLMProviderName.GEMINI
        settings.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT
        settings.gemini.llm_model = GeminiLLMModel.GEMINI_37_FLASH
        return settings

    if model == TranslationModel.GEMINI_31_FLASH_LITE:
        if connection == TranslationConnection.OPENROUTER:
            settings.provider.llm = LLMProviderName.OPENROUTER
            settings.openrouter.llm_model = OpenRouterLLMModel.GEMINI_31_FLASH_LITE
            settings.openrouter.provider_routing = OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY
            settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
            settings.openrouter.selection_alias = _derive_openrouter_selection_alias(
                settings.openrouter.llm_model,
                settings.openrouter.selected_source,
            )
            return settings
        settings.provider.llm = LLMProviderName.GEMINI
        settings.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT
        settings.gemini.llm_model = GeminiLLMModel.GEMINI_31_FLASH_LITE
        return settings

    if model == TranslationModel.LOCAL_LLM:
        settings.provider.llm = LLMProviderName.LOCAL_LLM
        settings.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT
        return settings

    settings.provider.llm = LLMProviderName.QWEN
    settings.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_PLUS
    return settings


def _ensure_mapping_block(data: dict[str, Any], key: str) -> tuple[dict[str, Any], bool]:
    block = data.get(key)
    if isinstance(block, dict):
        return block, False
    block = {}
    data[key] = block
    return block, True


def _set_mapping_value(mapping: dict[str, Any], key: str, value: object) -> bool:
    if mapping.get(key) == value:
        return False
    mapping[key] = value
    return True


def _apply_materialized_translation_to_data(
    data: dict[str, Any],
    translation: TranslationSettings,
) -> bool:
    provider_data, changed = _ensure_mapping_block(data, "provider")
    openrouter_data, block_changed = _ensure_mapping_block(data, "openrouter")
    changed = changed or block_changed
    gemini_data, block_changed = _ensure_mapping_block(data, "gemini")
    changed = changed or block_changed
    qwen_data, block_changed = _ensure_mapping_block(data, "qwen")
    changed = changed or block_changed
    deepseek_data, block_changed = _ensure_mapping_block(data, "deepseek")
    changed = changed or block_changed
    cerebras_data, block_changed = _ensure_mapping_block(data, "cerebras")
    changed = changed or block_changed

    translation = _normalize_translation_settings(
        model=_parse_translation_model(translation.model),
        connection=_parse_translation_connection(translation.connection),
        history=translation.connection_history,
        http_extension_id=translation.http_extension_id,
        previous_llm_model=translation.previous_llm_model,
    )

    if translation.model == TranslationModel.CUSTOM_HTTP:
        return changed

    if translation.model in (TranslationModel.GEMMA4_26B_31B, TranslationModel.GEMMA4_31B):
        if (
            translation.model == TranslationModel.GEMMA4_31B
            and translation.connection == TranslationConnection.CEREBRAS
        ):
            changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.CEREBRAS.value)
            changed |= _set_mapping_value(
                openrouter_data,
                "provider_routing",
                OpenRouterProviderRouting.DEFAULT.value,
            )
            changed |= _set_mapping_value(
                cerebras_data,
                "llm_model",
                CerebrasLLMModel.GEMMA_4_31B.value,
            )
            return changed
        selected_source = (
            OpenRouterCredentialSource.MANAGED
            if translation.connection == TranslationConnection.MANAGED
            else OpenRouterCredentialSource.BYOK
        )
        if translation.model == TranslationModel.GEMMA4_26B_31B:
            openrouter_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
            provider_routing = OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY
            selection_alias = (
                OpenRouterSelectionAlias.GEMMA4_26B_31B_MANAGED
                if selected_source == OpenRouterCredentialSource.MANAGED
                else OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK
            )
        else:
            openrouter_model = OpenRouterLLMModel.GEMMA_4_31B_IT
            provider_routing = OpenRouterProviderRouting.GEMMA4_31B_LATENCY
            selection_alias = (
                OpenRouterSelectionAlias.GEMMA4_31B_MANAGED
                if selected_source == OpenRouterCredentialSource.MANAGED
                else OpenRouterSelectionAlias.GEMMA4_31B_BYOK
            )
        changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.OPENROUTER.value)
        changed |= _set_mapping_value(openrouter_data, "llm_model", openrouter_model.value)
        changed |= _set_mapping_value(openrouter_data, "provider_routing", provider_routing.value)
        changed |= _set_mapping_value(openrouter_data, "selected_source", selected_source.value)
        changed |= _set_mapping_value(openrouter_data, "selection_alias", selection_alias.value)
        return changed

    if translation.model == TranslationModel.GEMMA4:
        selected_source = (
            OpenRouterCredentialSource.MANAGED
            if translation.connection == TranslationConnection.MANAGED
            else OpenRouterCredentialSource.BYOK
        )
        selection_alias = _derive_openrouter_selection_alias(
            OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            selected_source,
        )
        changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.OPENROUTER.value)
        changed |= _set_mapping_value(
            openrouter_data,
            "llm_model",
            OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
        )
        changed |= _set_mapping_value(
            openrouter_data,
            "provider_routing",
            OpenRouterProviderRouting.GEMMA4_26B_LATENCY.value,
        )
        changed |= _set_mapping_value(openrouter_data, "selected_source", selected_source.value)
        changed |= _set_mapping_value(openrouter_data, "selection_alias", selection_alias.value)
        return changed

    if translation.model == TranslationModel.DEEPSEEK_V4_FLASH:
        if translation.connection == TranslationConnection.OFFICIAL_BYOK:
            changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.DEEPSEEK.value)
            changed |= _set_mapping_value(
                openrouter_data,
                "provider_routing",
                OpenRouterProviderRouting.DEFAULT.value,
            )
            changed |= _set_mapping_value(
                deepseek_data,
                "llm_model",
                DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value,
            )
            return changed
        selected_source = (
            OpenRouterCredentialSource.MANAGED
            if translation.connection
            in (TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA)
            else OpenRouterCredentialSource.BYOK
        )
        provider_routing = (
            OpenRouterProviderRouting.DEEPSEEK_ONLY
            if translation.connection == TranslationConnection.MANAGED_CHINA
            else OpenRouterProviderRouting.DEFAULT
        )
        selection_alias = _derive_openrouter_selection_alias(
            OpenRouterLLMModel.DEEPSEEK_V4_FLASH,
            selected_source,
        )
        changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.OPENROUTER.value)
        changed |= _set_mapping_value(
            openrouter_data,
            "llm_model",
            OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value,
        )
        changed |= _set_mapping_value(
            openrouter_data,
            "provider_routing",
            provider_routing.value,
        )
        changed |= _set_mapping_value(openrouter_data, "selected_source", selected_source.value)
        changed |= _set_mapping_value(openrouter_data, "selection_alias", selection_alias.value)
        return changed

    if translation.model == TranslationModel.GEMINI_37_FLASH:
        if translation.connection == TranslationConnection.OPENROUTER:
            selection_alias = _derive_openrouter_selection_alias(
                OpenRouterLLMModel.GEMINI_37_FLASH,
                OpenRouterCredentialSource.BYOK,
            )
            changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.OPENROUTER.value)
            changed |= _set_mapping_value(
                openrouter_data,
                "llm_model",
                OpenRouterLLMModel.GEMINI_37_FLASH.value,
            )
            changed |= _set_mapping_value(
                openrouter_data,
                "provider_routing",
                OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY.value,
            )
            changed |= _set_mapping_value(
                openrouter_data,
                "selected_source",
                OpenRouterCredentialSource.BYOK.value,
            )
            changed |= _set_mapping_value(openrouter_data, "selection_alias", selection_alias.value)
            return changed
        changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.GEMINI.value)
        changed |= _set_mapping_value(
            openrouter_data,
            "provider_routing",
            OpenRouterProviderRouting.DEFAULT.value,
        )
        changed |= _set_mapping_value(
            gemini_data,
            "llm_model",
            GeminiLLMModel.GEMINI_37_FLASH.value,
        )
        return changed

    if translation.model == TranslationModel.GEMINI_31_FLASH_LITE:
        if translation.connection == TranslationConnection.OPENROUTER:
            selection_alias = _derive_openrouter_selection_alias(
                OpenRouterLLMModel.GEMINI_31_FLASH_LITE,
                OpenRouterCredentialSource.BYOK,
            )
            changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.OPENROUTER.value)
            changed |= _set_mapping_value(
                openrouter_data,
                "llm_model",
                OpenRouterLLMModel.GEMINI_31_FLASH_LITE.value,
            )
            changed |= _set_mapping_value(
                openrouter_data,
                "provider_routing",
                OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY.value,
            )
            changed |= _set_mapping_value(
                openrouter_data,
                "selected_source",
                OpenRouterCredentialSource.BYOK.value,
            )
            changed |= _set_mapping_value(openrouter_data, "selection_alias", selection_alias.value)
            return changed
        changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.GEMINI.value)
        changed |= _set_mapping_value(
            openrouter_data,
            "provider_routing",
            OpenRouterProviderRouting.DEFAULT.value,
        )
        changed |= _set_mapping_value(
            gemini_data,
            "llm_model",
            GeminiLLMModel.GEMINI_31_FLASH_LITE.value,
        )
        return changed

    if translation.model == TranslationModel.LOCAL_LLM:
        changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.LOCAL_LLM.value)
        return changed

    changed |= _set_mapping_value(provider_data, "llm", LLMProviderName.QWEN.value)
    changed |= _set_mapping_value(
        openrouter_data,
        "provider_routing",
        OpenRouterProviderRouting.DEFAULT.value,
    )
    changed |= _set_mapping_value(qwen_data, "llm_model", QwenLLMModel.QWEN_35_PLUS.value)
    return changed


def _infer_qwen_region_from_legacy_asr_endpoint(value: object) -> QwenRegion | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if "dashscope-intl.aliyuncs.com" in normalized:
        return QwenRegion.SINGAPORE
    if "dashscope.aliyuncs.com" in normalized:
        return QwenRegion.BEIJING
    return None


def _parse_qwen_region(value: object, *, legacy_asr_endpoint: object = None) -> QwenRegion:
    if isinstance(value, str):
        normalized = value.strip()
        try:
            return QwenRegion(normalized)
        except ValueError:
            pass
    inferred = _infer_qwen_region_from_legacy_asr_endpoint(legacy_asr_endpoint)
    if inferred is not None:
        return inferred
    return QwenRegion.BEIJING


def _shared_default_prompt() -> str:
    from puripuly_heart.config.prompts import load_prompt_for_provider

    return load_prompt_for_provider(LLMProviderName.GEMINI.value)


def ensure_prompt_defaults(settings: AppSettings) -> AppSettings:
    system_prompt_empty = not settings.system_prompt.strip()
    if system_prompt_empty:
        prompt = _shared_default_prompt()
        settings.system_prompt = prompt
    settings.system_prompts = {}
    return settings


def detect_system_locale() -> str | None:
    try:
        return locale.getlocale()[0]
    except (ValueError, locale.Error):
        return None


def _normalize_first_run_locale(system_locale: str | None) -> str:
    if system_locale is None:
        return ""
    normalized = system_locale.strip()
    if not normalized:
        return ""
    normalized = normalized.split(".", maxsplit=1)[0]
    normalized = normalized.split("@", maxsplit=1)[0]
    return normalized.replace("_", "-").casefold()


def resolve_first_run_ui_locale(system_locale: str | None) -> str:
    normalized = _normalize_first_run_locale(system_locale)
    if normalized == "ko" or normalized.startswith("ko-") or normalized.startswith("korean"):
        return "ko"
    if normalized == "ja" or normalized.startswith("ja-") or normalized.startswith("japanese"):
        return "ja"
    if normalized == "zh" or normalized.startswith("zh-") or normalized.startswith("chinese"):
        return "zh-CN"
    if normalized == "ru" or normalized.startswith("ru-") or normalized.startswith("russian"):
        return "ru"
    return "en"


def _is_china_first_run_locale(system_locale: str | None) -> bool:
    return resolve_first_run_ui_locale(system_locale) == "zh-CN"


def _apply_china_managed_first_run_defaults(settings: AppSettings) -> None:
    settings.openrouter = replace(
        settings.openrouter,
        selection_alias=OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED,
        provider_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY,
        fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
    )
    settings.translation = _derive_translation_settings_from_runtime_values(
        provider_llm=settings.provider.llm,
        openrouter_model=settings.openrouter.llm_model,
        openrouter_selected_source=settings.openrouter.selected_source,
        openrouter_provider_routing=settings.openrouter.provider_routing,
        gemini_model=settings.gemini.llm_model,
        qwen_model=settings.qwen.llm_model,
        cerebras_model=settings.cerebras.llm_model,
        history=settings.translation.connection_history,
    )
    settings.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.GEMMA4_26B_31B,
        connection=TranslationConnection.OPENROUTER,
    )


def new_settings_for_first_run(system_locale: str | None = None) -> AppSettings:
    if system_locale is None:
        system_locale = detect_system_locale()
    settings = AppSettings()
    settings.ui.locale = resolve_first_run_ui_locale(system_locale)
    if _is_china_first_run_locale(system_locale):
        _apply_china_managed_first_run_defaults(settings)
    else:
        settings.translation.fallback = TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.GEMMA4_26B_31B,
            connection=TranslationConnection.OPENROUTER,
        )
    ensure_prompt_defaults(settings)
    settings = with_telemetry_consent(settings, "allow")
    settings.validate()
    return settings


def _parse_custom_terms(value: object) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("custom_terms must be a dict[str, list[str]]")

    out: dict[str, list[str]] = {}
    for language, terms in value.items():
        if not isinstance(language, str):
            raise ValueError("custom_terms keys must be strings")
        if not isinstance(terms, list):
            raise ValueError("custom_terms values must be lists of strings")

        normalized_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in terms:
            if not isinstance(term, str):
                raise ValueError("custom_terms values must be lists of strings")
            normalized_term = term.strip()
            if not normalized_term or normalized_term in seen_terms:
                continue
            if len(normalized_terms) >= MAX_CUSTOM_VOCAB_TERMS:
                break
            seen_terms.add(normalized_term)
            normalized_terms.append(normalized_term)

        out[language] = normalized_terms
    return out


def _coerce_int(value: object, fallback: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _normalize_internal_sample_rate_hz(value: object) -> int:
    normalized = _coerce_int(value, STT_INTERNAL_SAMPLE_RATE_HZ)
    if normalized == 8000:
        return STT_INTERNAL_SAMPLE_RATE_HZ
    return normalized


def _parse_optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _parse_bool(value: object, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return fallback


def _parse_non_negative_int(value: object, fallback: int = 0) -> int:
    if type(value) is not int:
        return fallback
    if value < 0:
        return fallback
    return value


def _parse_utc_iso8601_timestamp(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    parse_value = f"{normalized[:-1]}+00:00" if normalized.endswith("Z") else normalized
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return normalized


def _parse_optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize_peer_block(data: dict[str, Any], key: str, default_block: dict[str, Any]) -> bool:
    if isinstance(data.get(key), dict):
        return False
    data[key] = copy.deepcopy(default_block)
    return True


def _migrate_settings_dict(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    data: dict[str, Any] = copy.deepcopy(raw)
    changed = False
    peer_block_defaults: dict[str, dict[str, Any]] = {
        "peer_qwen_asr_stt": {"model": None, "region": None},
        "peer_soniox_stt": {
            "model": None,
            "endpoint": None,
            "keepalive_interval_s": None,
            "trailing_silence_ms": None,
        },
    }

    version = _coerce_int(data.get("settings_version"), 1)
    if version < 1:
        version = 1

    if version < 2:
        llm_data = data.get("llm")
        if not isinstance(llm_data, dict):
            llm_data = {}
            data["llm"] = llm_data
            changed = True

        concurrency_limit = _coerce_int(llm_data.get("concurrency_limit"), 1)
        # Preserve explicit custom limits (>1), migrate legacy default 1 to new default 2.
        if concurrency_limit <= 1:
            llm_data["concurrency_limit"] = 2
            changed = True

        version = 2

    if version < 3:
        desktop_audio_data = data.get("desktop_audio")
        if not isinstance(desktop_audio_data, dict):
            desktop_audio_data = {}
            data["desktop_audio"] = desktop_audio_data
            changed = True
        if desktop_audio_data.get("vad_speech_threshold") != 0.6:
            desktop_audio_data["vad_speech_threshold"] = 0.6
            changed = True
        version = 3

    if version < 4:
        raw_provider_data = data.get("provider")
        if raw_provider_data is None:
            provider_data = {}
            data["provider"] = provider_data
            changed = True
        elif isinstance(raw_provider_data, dict):
            provider_data = raw_provider_data
        else:
            provider_data = {
                "stt": STTProviderName.DEEPGRAM.value,
                "llm": LLMProviderName.GEMINI.value,
            }
            data["provider"] = provider_data
            changed = True

        if "peer_stt" not in provider_data:
            provider_data["peer_stt"] = STTProviderName.DEEPGRAM.value
            changed = True

        for key, default_block in peer_block_defaults.items():
            if _normalize_peer_block(data, key, default_block):
                changed = True

        version = 4

    if version < 5:
        openrouter_data = data.get("openrouter")
        if not isinstance(openrouter_data, dict):
            data["openrouter"] = {
                "llm_model": OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
            }
            changed = True

        api_key_verified = data.get("api_key_verified")
        if not isinstance(api_key_verified, dict):
            api_key_verified = {}
            data["api_key_verified"] = api_key_verified
            changed = True
        if "openrouter" not in api_key_verified:
            api_key_verified["openrouter"] = False
            changed = True

        version = 5

    if version < 6:
        llm_data = data.get("llm")
        if not isinstance(llm_data, dict):
            llm_data = {}
            data["llm"] = llm_data
            changed = True

        concurrency_limit = _coerce_int(llm_data.get("concurrency_limit"), 2)
        # Migrate previous default-sized limits up to the faster default while preserving
        # explicit higher custom values.
        if concurrency_limit <= 2:
            llm_data["concurrency_limit"] = 5
            changed = True

        version = 6

    if version < 7:
        desktop_audio_data = data.get("desktop_audio")
        if (
            isinstance(desktop_audio_data, dict)
            and desktop_audio_data.get("vad_hangover_ms") == 900
        ):
            desktop_audio_data["vad_hangover_ms"] = 700
            changed = True

        version = 7

    if version < 8:
        desktop_audio_data = data.get("desktop_audio")
        if (
            isinstance(desktop_audio_data, dict)
            and desktop_audio_data.get("vad_hangover_ms") == 700
        ):
            desktop_audio_data["vad_hangover_ms"] = 600
            changed = True

        version = 8

    if version < 9:
        managed_identity_data = data.get("managed_identity")
        if not isinstance(managed_identity_data, dict):
            managed_identity_data = {}
            data["managed_identity"] = managed_identity_data
            changed = True

        if "installation_id" not in managed_identity_data:
            managed_identity_data["installation_id"] = ""
            changed = True
        if "release_token" not in managed_identity_data:
            managed_identity_data["release_token"] = None
            changed = True
        if "release_token_expires_at" not in managed_identity_data:
            managed_identity_data["release_token_expires_at"] = None
            changed = True

        version = 9

    if version < 10:
        openrouter_data = data.get("openrouter")
        if not isinstance(openrouter_data, dict):
            openrouter_data = {}
            data["openrouter"] = openrouter_data
            changed = True

        raw_selected_source = _get_raw_openrouter_selected_source(openrouter_data)
        normalized_selected_source = _parse_openrouter_credential_source(
            raw_selected_source,
            fallback=_default_openrouter_credential_source_value(data),
        )
        if openrouter_data.get("selected_source") != normalized_selected_source.value:
            openrouter_data["selected_source"] = normalized_selected_source.value
            changed = True
        if "credential_source" in openrouter_data:
            del openrouter_data["credential_source"]
            changed = True
        if "selected_credential_source" in openrouter_data:
            del openrouter_data["selected_credential_source"]
            changed = True

        version = 10

    if version < 11:
        openrouter_data = data.get("openrouter")
        if not isinstance(openrouter_data, dict):
            openrouter_data = {}
            data["openrouter"] = openrouter_data
            changed = True

        normalized_selected_source = _parse_openrouter_credential_source(
            _get_raw_openrouter_selected_source(openrouter_data),
            fallback=_default_openrouter_credential_source_value(data),
        )
        if (
            _default_openrouter_credential_source_value(data) == OpenRouterCredentialSource.BYOK
            and normalized_selected_source == OpenRouterCredentialSource.NONE
        ):
            normalized_selected_source = OpenRouterCredentialSource.BYOK
        if openrouter_data.get("selected_source") != normalized_selected_source.value:
            openrouter_data["selected_source"] = normalized_selected_source.value
            changed = True
        if "credential_source" in openrouter_data:
            del openrouter_data["credential_source"]
            changed = True
        if "selected_credential_source" in openrouter_data:
            del openrouter_data["selected_credential_source"]
            changed = True

        version = 11

    if version < 12:
        openrouter_data = data.get("openrouter")
        if not isinstance(openrouter_data, dict):
            openrouter_data = {}
            data["openrouter"] = openrouter_data
            changed = True

        normalized_broker_base_url = _parse_openrouter_broker_base_url(
            openrouter_data.get("broker_base_url")
        )
        if openrouter_data.get("broker_base_url") != normalized_broker_base_url:
            openrouter_data["broker_base_url"] = normalized_broker_base_url
            changed = True

        version = 12

    if version < 13:
        managed_identity_data = data.get("managed_identity")
        if not isinstance(managed_identity_data, dict):
            managed_identity_data = {}
            data["managed_identity"] = managed_identity_data
            changed = True

        if "verified_hardware_hash" not in managed_identity_data:
            managed_identity_data["verified_hardware_hash"] = None
            changed = True
        if "verified_hardware_hash_salt_version" not in managed_identity_data:
            managed_identity_data["verified_hardware_hash_salt_version"] = None
            changed = True

        version = 13

    if version < 14:
        audio_data = data.get("audio")
        if isinstance(audio_data, dict):
            raw_internal_sample_rate_hz = audio_data.get(
                "internal_sample_rate_hz", STT_INTERNAL_SAMPLE_RATE_HZ
            )
            normalized_internal_sample_rate_hz = _normalize_internal_sample_rate_hz(
                raw_internal_sample_rate_hz
            )
            if raw_internal_sample_rate_hz != normalized_internal_sample_rate_hz:
                audio_data["internal_sample_rate_hz"] = normalized_internal_sample_rate_hz
                changed = True

        version = 14

    if version < 15:
        openrouter_data = data.get("openrouter")
        if not isinstance(openrouter_data, dict):
            openrouter_data = {}
            data["openrouter"] = openrouter_data
            changed = True

        (
            normalized_openrouter_model,
            normalized_openrouter_selected_source,
            normalized_selection_alias,
        ) = _resolve_openrouter_main_selection(openrouter_data, data)
        normalized_selection_alias_value = (
            normalized_selection_alias.value if normalized_selection_alias is not None else None
        )
        if openrouter_data.get("llm_model") != normalized_openrouter_model.value:
            openrouter_data["llm_model"] = normalized_openrouter_model.value
            changed = True
        if openrouter_data.get("selected_source") != normalized_openrouter_selected_source.value:
            openrouter_data["selected_source"] = normalized_openrouter_selected_source.value
            changed = True
        if openrouter_data.get("selection_alias") != normalized_selection_alias_value:
            openrouter_data["selection_alias"] = normalized_selection_alias_value
            changed = True

        normalized_fallback_selection_alias = _parse_openrouter_fallback_selection_alias(
            openrouter_data.get("fallback_selection_alias")
        )
        if (
            openrouter_data.get("fallback_selection_alias")
            != normalized_fallback_selection_alias.value
        ):
            openrouter_data["fallback_selection_alias"] = normalized_fallback_selection_alias.value
            changed = True

        version = 15

    if version < 16:
        managed_identity_data = data.get("managed_identity")
        if not isinstance(managed_identity_data, dict):
            managed_identity_data = {}
            data["managed_identity"] = managed_identity_data
            changed = True

        for key in (
            "active_managed_credential_ref",
            "active_managed_expires_at",
            "founder_letter_seen_credential_ref",
        ):
            if key not in managed_identity_data:
                managed_identity_data[key] = None
                changed = True

        version = 16

    if version < 17:
        audio_data = data.get("audio")
        if isinstance(audio_data, dict):
            input_host_api = audio_data.get("input_host_api")
            if (
                isinstance(input_host_api, str)
                and input_host_api.strip() == WINDOWS_DIRECTSOUND_HOST_API
                and input_host_api != WINDOWS_DIRECTSOUND_HOST_API
            ):
                audio_data["input_host_api"] = WINDOWS_DIRECTSOUND_HOST_API
                changed = True

        version = 17

    if version < 18:
        osc_data = data.get("osc")
        if isinstance(osc_data, dict):
            if "cooldown_s" in osc_data:
                osc_data.pop("cooldown_s")
                changed = True
            if "ttl_s" in osc_data:
                osc_data.pop("ttl_s")
                changed = True

        version = 18

    if version < 19:
        prompt = _shared_default_prompt()
        data["system_prompt"] = prompt
        changed = True
        version = 19

    if version < 20:
        changed = True
        version = 20

    if version < 21:
        desktop_audio_data = data.get("desktop_audio")
        if not isinstance(desktop_audio_data, dict):
            desktop_audio_data = {}
            data["desktop_audio"] = desktop_audio_data
            changed = True
        if desktop_audio_data.get("vad_hangover_ms") != DEFAULT_DESKTOP_AUDIO_VAD_HANGOVER_MS:
            desktop_audio_data["vad_hangover_ms"] = DEFAULT_DESKTOP_AUDIO_VAD_HANGOVER_MS
            changed = True
        version = 21

    if version < 22:
        if _normalize_local_llm_data(data):
            changed = True
        version = 22

    if version < 23:
        managed_identity_data = data.get("managed_identity")
        if not isinstance(managed_identity_data, dict):
            managed_identity_data = {}
            data["managed_identity"] = managed_identity_data
            changed = True

        raw_referral_id = managed_identity_data.get("referral_id")
        normalized_referral_id = normalize_owned_referral_id(raw_referral_id)
        if "referral_id" not in managed_identity_data or raw_referral_id != normalized_referral_id:
            managed_identity_data["referral_id"] = normalized_referral_id
            changed = True

        version = 23

    if version < 24:
        changed = True
        version = 24

    if _normalize_local_llm_data(data):
        changed = True

    stt_data = data.get("stt")
    if not isinstance(stt_data, dict):
        stt_data = {}
        data["stt"] = stt_data
        changed = True

    audio_data = data.get("audio")
    if isinstance(audio_data, dict):
        raw_internal_sample_rate_hz = audio_data.get(
            "internal_sample_rate_hz", STT_INTERNAL_SAMPLE_RATE_HZ
        )
        normalized_internal_sample_rate_hz = _normalize_internal_sample_rate_hz(
            raw_internal_sample_rate_hz
        )
        if raw_internal_sample_rate_hz != normalized_internal_sample_rate_hz:
            audio_data["internal_sample_rate_hz"] = normalized_internal_sample_rate_hz
            changed = True

    if "custom_terms" not in stt_data:
        stt_data["custom_terms"] = _default_custom_terms()
        changed = True

    if "custom_vocabulary_enabled" not in stt_data:
        normalized_custom_terms = _parse_custom_terms(stt_data.get("custom_terms"))
        stt_data["custom_vocabulary_enabled"] = any(
            bool(terms) for terms in normalized_custom_terms.values()
        )
        changed = True

    raw_provider_data = data.get("provider")
    provider_data: dict[str, Any] | None
    if raw_provider_data is None:
        provider_data = {}
        data["provider"] = provider_data
        changed = True
    elif not isinstance(raw_provider_data, dict):
        provider_data = {
            "stt": STTProviderName.DEEPGRAM.value,
            "llm": LLMProviderName.GEMINI.value,
        }
        data["provider"] = provider_data
        changed = True
    else:
        provider_data = raw_provider_data

    if isinstance(provider_data, dict) and "stt" in provider_data:
        raw_stt_provider = provider_data.get("stt")
        normalized_stt_provider = _parse_stt_provider(str(raw_stt_provider)).value
        if raw_stt_provider != normalized_stt_provider:
            provider_data["stt"] = normalized_stt_provider
            changed = True
    if isinstance(provider_data, dict) and "peer_stt" not in provider_data:
        provider_data["peer_stt"] = STTProviderName.DEEPGRAM.value
        changed = True
    if isinstance(provider_data, dict) and "peer_stt" in provider_data:
        raw_peer_provider = provider_data.get("peer_stt")
        normalized_peer_provider = _parse_peer_stt_provider(str(raw_peer_provider)).value
        if raw_peer_provider != normalized_peer_provider:
            provider_data["peer_stt"] = normalized_peer_provider
            changed = True

    if "peer_deepgram_stt" in data:
        del data["peer_deepgram_stt"]
        changed = True

    for key, default_block in peer_block_defaults.items():
        if _normalize_peer_block(data, key, default_block):
            changed = True

    # Keep schema at v2 but backfill Soniox legacy default model upgrade.
    soniox_data = data.get("soniox_stt")
    if isinstance(soniox_data, dict):
        model = soniox_data.get("model")
        if isinstance(model, str):
            normalized = model.strip()
            if normalized in ("stt-rt-v3", "stt-rt-v4"):
                soniox_data["model"] = "stt-rt-v5"
                changed = True

    gemini_data = data.get("gemini")
    if not isinstance(gemini_data, dict):
        gemini_data = {}
        data["gemini"] = gemini_data
        changed = True

    raw_gemini_model = gemini_data.get("llm_model")
    normalized_gemini_model = _parse_gemini_llm_model(raw_gemini_model).value
    if raw_gemini_model != normalized_gemini_model:
        gemini_data["llm_model"] = normalized_gemini_model
        changed = True

    openrouter_data = data.get("openrouter")
    if not isinstance(openrouter_data, dict):
        openrouter_data = {}
        data["openrouter"] = openrouter_data
        changed = True

    (
        normalized_openrouter_model,
        normalized_openrouter_selected_source,
        normalized_selection_alias,
    ) = _resolve_openrouter_main_selection(openrouter_data, data)
    normalized_selection_alias_value = (
        normalized_selection_alias.value if normalized_selection_alias is not None else None
    )
    if openrouter_data.get("llm_model") != normalized_openrouter_model.value:
        openrouter_data["llm_model"] = normalized_openrouter_model.value
        changed = True

    raw_openrouter_routing_mode = openrouter_data.get("routing_mode")
    normalized_openrouter_routing_mode = _parse_openrouter_routing_mode(
        raw_openrouter_routing_mode
    ).value
    if raw_openrouter_routing_mode != normalized_openrouter_routing_mode:
        openrouter_data["routing_mode"] = normalized_openrouter_routing_mode
        changed = True

    raw_openrouter_provider_routing = openrouter_data.get("provider_routing")
    normalized_openrouter_provider_routing = _parse_openrouter_provider_routing(
        raw_openrouter_provider_routing
    ).value
    if raw_openrouter_provider_routing != normalized_openrouter_provider_routing:
        openrouter_data["provider_routing"] = normalized_openrouter_provider_routing
        changed = True

    if openrouter_data.get("selected_source") != normalized_openrouter_selected_source.value:
        openrouter_data["selected_source"] = normalized_openrouter_selected_source.value
        changed = True
    if openrouter_data.get("selection_alias") != normalized_selection_alias_value:
        openrouter_data["selection_alias"] = normalized_selection_alias_value
        changed = True
    if "credential_source" in openrouter_data:
        del openrouter_data["credential_source"]
        changed = True
    if "selected_credential_source" in openrouter_data:
        del openrouter_data["selected_credential_source"]
        changed = True

    raw_fallback_selection_alias = openrouter_data.get("fallback_selection_alias")
    normalized_fallback_selection_alias = _parse_openrouter_fallback_selection_alias(
        raw_fallback_selection_alias
    )
    if raw_fallback_selection_alias != normalized_fallback_selection_alias.value:
        openrouter_data["fallback_selection_alias"] = normalized_fallback_selection_alias.value
        changed = True

    raw_openrouter_broker_base_url = openrouter_data.get("broker_base_url")
    normalized_openrouter_broker_base_url = _parse_openrouter_broker_base_url(
        raw_openrouter_broker_base_url
    )
    if raw_openrouter_broker_base_url != normalized_openrouter_broker_base_url:
        openrouter_data["broker_base_url"] = normalized_openrouter_broker_base_url
        changed = True

    qwen_data = data.get("qwen")
    if not isinstance(qwen_data, dict):
        qwen_data = {}
        data["qwen"] = qwen_data
        changed = True

    qwen_asr_data = data.get("qwen_asr_stt")
    qwen_asr_endpoint = qwen_asr_data.get("endpoint") if isinstance(qwen_asr_data, dict) else None

    raw_qwen_region = qwen_data.get("region")
    normalized_qwen_region = _parse_qwen_region(
        raw_qwen_region,
        legacy_asr_endpoint=qwen_asr_endpoint,
    ).value
    if raw_qwen_region != normalized_qwen_region:
        qwen_data["region"] = normalized_qwen_region
        changed = True

    raw_qwen_model = qwen_data.get("llm_model")
    normalized_qwen_model = _parse_qwen_llm_model(raw_qwen_model).value
    if raw_qwen_model != normalized_qwen_model:
        qwen_data["llm_model"] = normalized_qwen_model
        changed = True

    deepseek_data = data.get("deepseek")
    if not isinstance(deepseek_data, dict):
        deepseek_data = {}
        data["deepseek"] = deepseek_data
        changed = True

    raw_deepseek_model = deepseek_data.get("llm_model")
    normalized_deepseek_model = _parse_deepseek_llm_model(raw_deepseek_model).value
    if raw_deepseek_model != normalized_deepseek_model:
        deepseek_data["llm_model"] = normalized_deepseek_model
        changed = True

    cerebras_data = data.get("cerebras")
    if not isinstance(cerebras_data, dict):
        cerebras_data = {}
        data["cerebras"] = cerebras_data
        changed = True

    raw_cerebras_model = cerebras_data.get("llm_model")
    normalized_cerebras_model = _parse_cerebras_llm_model(raw_cerebras_model).value
    if raw_cerebras_model != normalized_cerebras_model:
        cerebras_data["llm_model"] = normalized_cerebras_model
        changed = True

    if stt_data.get("low_latency_vad_hangover_ms") == LEGACY_LOW_LATENCY_VAD_HANGOVER_MS:
        stt_data["low_latency_vad_hangover_ms"] = DEFAULT_LOW_LATENCY_VAD_HANGOVER_MS
        changed = True

    translation_data = data.get("translation") if isinstance(data.get("translation"), dict) else {}
    translation_history = _parse_translation_connection_history(
        translation_data.get("connection_history") if isinstance(translation_data, dict) else None
    )
    if translation_data.get("previous_llm_model") == "gemma4_31b_cerebras":
        translation_history[TranslationModel.GEMMA4_31B.value] = TranslationConnection.CEREBRAS
    if _translation_data_has_valid_model(translation_data):
        translation_model, translation_connection = _parse_translation_selection(
            translation_data.get("model"),
            translation_data.get("connection"),
        )
        normalized_translation_settings = _normalize_translation_settings(
            model=translation_model,
            connection=translation_connection,
            fallback=translation_data.get("fallback"),
            history=translation_history,
            http_extension_id=translation_data.get("http_extension_id"),
            previous_llm_model=translation_data.get("previous_llm_model"),
        )
    else:
        normalized_translation_settings = _derive_translation_settings_from_runtime_values(
            provider_llm=_parse_llm_provider(
                provider_data.get("llm", LLMProviderName.GEMINI.value)
            ),
            openrouter_model=_parse_openrouter_llm_model(openrouter_data.get("llm_model")),
            openrouter_selected_source=_parse_openrouter_credential_source(
                openrouter_data.get("selected_source"),
                fallback=_default_openrouter_credential_source_value(data),
            ),
            openrouter_provider_routing=_parse_openrouter_provider_routing(
                openrouter_data.get("provider_routing")
            ),
            gemini_model=_parse_gemini_llm_model(gemini_data.get("llm_model")),
            qwen_model=_parse_qwen_llm_model(qwen_data.get("llm_model")),
            cerebras_model=_parse_cerebras_llm_model(cerebras_data.get("llm_model")),
            history=translation_history,
        )
    normalized_translation_data = _translation_settings_to_dict(normalized_translation_settings)
    if data.get("translation") != normalized_translation_data:
        data["translation"] = normalized_translation_data
        changed = True
    if _apply_materialized_translation_to_data(data, normalized_translation_settings):
        changed = True

    api_key_verified_data = data.get("api_key_verified")
    if not isinstance(api_key_verified_data, dict):
        api_key_verified_data = {}
        data["api_key_verified"] = api_key_verified_data
        changed = True
    if "deepseek" not in api_key_verified_data:
        api_key_verified_data["deepseek"] = False
        changed = True
    if "cerebras" not in api_key_verified_data:
        api_key_verified_data["cerebras"] = False
        changed = True

    overlay_data = data.get("overlay")
    if not isinstance(overlay_data, dict):
        overlay_data = {}
        data["overlay"] = overlay_data
        changed = True

    overlay_calibration_data = overlay_data.get("calibration")
    if not isinstance(overlay_calibration_data, dict):
        overlay_calibration_data = {}

    legacy_overlay_calibration_data = data.get("overlay_calibration")
    if not isinstance(legacy_overlay_calibration_data, dict):
        legacy_overlay_calibration_data = {}

    normalized_overlay_calibration = OverlayCalibration().to_dict()
    normalized_overlay_calibration.update(legacy_overlay_calibration_data)
    normalized_overlay_calibration.update(overlay_calibration_data)
    if overlay_data.get("calibration") != normalized_overlay_calibration:
        overlay_data["calibration"] = normalized_overlay_calibration
        changed = True

    normalized_overlay_target = _parse_overlay_target(overlay_data.get("target"))
    if overlay_data.get("target") != normalized_overlay_target:
        overlay_data["target"] = normalized_overlay_target
        changed = True

    normalized_desktop_flet = _desktop_flet_settings_to_dict(
        _parse_desktop_flet_settings(overlay_data.get("desktop_flet"))
    )
    if overlay_data.get("desktop_flet") != normalized_desktop_flet:
        overlay_data["desktop_flet"] = normalized_desktop_flet
        changed = True

    ui_data = data.get("ui")
    if not isinstance(ui_data, dict):
        ui_data = {}
        data["ui"] = ui_data
        changed = True

    normalized_show_translation = bool(
        overlay_data.get("show_translation", ui_data.get("show_overlay_translation", True))
    )
    if overlay_data.get("show_translation") != normalized_show_translation:
        overlay_data["show_translation"] = normalized_show_translation
        changed = True

    normalized_show_peer_original = bool(
        overlay_data.get("show_peer_original", ui_data.get("show_overlay_peer_original", True))
    )
    if overlay_data.get("show_peer_original") != normalized_show_peer_original:
        overlay_data["show_peer_original"] = normalized_show_peer_original
        changed = True

    if "show_overlay_translation" in ui_data:
        del ui_data["show_overlay_translation"]
        changed = True

    if "show_overlay_peer_original" in ui_data:
        del ui_data["show_overlay_peer_original"]
        changed = True

    if "overlay_enabled" in ui_data:
        del ui_data["overlay_enabled"]
        changed = True

    if "peer_translation_enabled" in ui_data:
        del ui_data["peer_translation_enabled"]
        changed = True

    raw_github_star_prompt_clicked = ui_data.get("github_star_prompt_clicked")
    normalized_github_star_prompt_clicked = _parse_bool(raw_github_star_prompt_clicked)
    if (
        "github_star_prompt_clicked" not in ui_data
        or raw_github_star_prompt_clicked != normalized_github_star_prompt_clicked
    ):
        ui_data["github_star_prompt_clicked"] = normalized_github_star_prompt_clicked
        changed = True

    raw_github_star_prompt_last_shown_at = ui_data.get("github_star_prompt_last_shown_at")
    normalized_github_star_prompt_last_shown_at = _parse_utc_iso8601_timestamp(
        raw_github_star_prompt_last_shown_at
    )
    if (
        "github_star_prompt_last_shown_at" not in ui_data
        or raw_github_star_prompt_last_shown_at != normalized_github_star_prompt_last_shown_at
    ):
        ui_data["github_star_prompt_last_shown_at"] = normalized_github_star_prompt_last_shown_at
        changed = True

    raw_github_star_prompt_show_count = ui_data.get("github_star_prompt_show_count")
    normalized_github_star_prompt_show_count = _parse_non_negative_int(
        raw_github_star_prompt_show_count
    )
    if (
        "github_star_prompt_show_count" not in ui_data
        or raw_github_star_prompt_show_count != normalized_github_star_prompt_show_count
        or type(raw_github_star_prompt_show_count)
        is not type(normalized_github_star_prompt_show_count)
    ):
        ui_data["github_star_prompt_show_count"] = normalized_github_star_prompt_show_count
        changed = True

    raw_github_star_prompt_translation_success_observed = ui_data.get(
        "github_star_prompt_translation_success_observed"
    )
    normalized_github_star_prompt_translation_success_observed = _parse_bool(
        raw_github_star_prompt_translation_success_observed
    )
    if (
        "github_star_prompt_translation_success_observed" not in ui_data
        or raw_github_star_prompt_translation_success_observed
        != normalized_github_star_prompt_translation_success_observed
    ):
        ui_data["github_star_prompt_translation_success_observed"] = (
            normalized_github_star_prompt_translation_success_observed
        )
        changed = True

    raw_github_star_prompt_eligible_launch_count = ui_data.get(
        "github_star_prompt_eligible_launch_count"
    )
    normalized_github_star_prompt_eligible_launch_count = _parse_non_negative_int(
        raw_github_star_prompt_eligible_launch_count
    )
    if (
        "github_star_prompt_eligible_launch_count" not in ui_data
        or raw_github_star_prompt_eligible_launch_count
        != normalized_github_star_prompt_eligible_launch_count
        or type(raw_github_star_prompt_eligible_launch_count)
        is not type(normalized_github_star_prompt_eligible_launch_count)
    ):
        ui_data["github_star_prompt_eligible_launch_count"] = (
            normalized_github_star_prompt_eligible_launch_count
        )
        changed = True

    if "overlay_calibration" in data:
        del data["overlay_calibration"]
        changed = True

    managed_identity_data = data.get("managed_identity")
    if not isinstance(managed_identity_data, dict):
        managed_identity_data = {}
        data["managed_identity"] = managed_identity_data
        changed = True

    raw_installation_id = managed_identity_data.get("installation_id")
    normalized_installation_id = (
        raw_installation_id.strip() if isinstance(raw_installation_id, str) else ""
    )
    if raw_installation_id != normalized_installation_id:
        managed_identity_data["installation_id"] = normalized_installation_id
        changed = True

    raw_release_token = managed_identity_data.get("release_token")
    normalized_release_token = _parse_optional_str(raw_release_token)
    if raw_release_token != normalized_release_token:
        managed_identity_data["release_token"] = normalized_release_token
        changed = True

    raw_release_token_expires_at = managed_identity_data.get("release_token_expires_at")
    normalized_release_token_expires_at = _parse_optional_str(raw_release_token_expires_at)
    if raw_release_token_expires_at != normalized_release_token_expires_at:
        managed_identity_data["release_token_expires_at"] = normalized_release_token_expires_at
        changed = True

    raw_verified_hardware_hash = managed_identity_data.get("verified_hardware_hash")
    normalized_verified_hardware_hash = _parse_optional_str(raw_verified_hardware_hash)
    if (
        "verified_hardware_hash" not in managed_identity_data
        or raw_verified_hardware_hash != normalized_verified_hardware_hash
    ):
        managed_identity_data["verified_hardware_hash"] = normalized_verified_hardware_hash
        changed = True

    raw_verified_hardware_hash_salt_version = managed_identity_data.get(
        "verified_hardware_hash_salt_version"
    )
    normalized_verified_hardware_hash_salt_version = _parse_optional_int(
        raw_verified_hardware_hash_salt_version
    )
    if (
        "verified_hardware_hash_salt_version" not in managed_identity_data
        or raw_verified_hardware_hash_salt_version != normalized_verified_hardware_hash_salt_version
    ):
        managed_identity_data["verified_hardware_hash_salt_version"] = (
            normalized_verified_hardware_hash_salt_version
        )
        changed = True

    raw_active_managed_credential_ref = managed_identity_data.get("active_managed_credential_ref")
    normalized_active_managed_credential_ref = _parse_optional_str(
        raw_active_managed_credential_ref
    )
    if (
        "active_managed_credential_ref" not in managed_identity_data
        or raw_active_managed_credential_ref != normalized_active_managed_credential_ref
    ):
        managed_identity_data["active_managed_credential_ref"] = (
            normalized_active_managed_credential_ref
        )
        changed = True

    raw_active_managed_expires_at = managed_identity_data.get("active_managed_expires_at")
    normalized_active_managed_expires_at = _parse_optional_str(raw_active_managed_expires_at)
    if (
        "active_managed_expires_at" not in managed_identity_data
        or raw_active_managed_expires_at != normalized_active_managed_expires_at
    ):
        managed_identity_data["active_managed_expires_at"] = normalized_active_managed_expires_at
        changed = True

    raw_founder_letter_seen_credential_ref = managed_identity_data.get(
        "founder_letter_seen_credential_ref"
    )
    normalized_founder_letter_seen_credential_ref = _parse_optional_str(
        raw_founder_letter_seen_credential_ref
    )
    if (
        "founder_letter_seen_credential_ref" not in managed_identity_data
        or raw_founder_letter_seen_credential_ref != normalized_founder_letter_seen_credential_ref
    ):
        managed_identity_data["founder_letter_seen_credential_ref"] = (
            normalized_founder_letter_seen_credential_ref
        )
        changed = True

    raw_referral_id = managed_identity_data.get("referral_id")
    normalized_referral_id = normalize_owned_referral_id(raw_referral_id)
    if "referral_id" not in managed_identity_data or raw_referral_id != normalized_referral_id:
        managed_identity_data["referral_id"] = normalized_referral_id
        changed = True

    raw_local_managed_claim_sources = managed_identity_data.get("local_managed_claim_sources")
    normalized_local_managed_claim_sources = list(
        normalize_managed_claim_sources(raw_local_managed_claim_sources)
    )
    if (
        "local_managed_claim_sources" not in managed_identity_data
        or raw_local_managed_claim_sources != normalized_local_managed_claim_sources
    ):
        managed_identity_data["local_managed_claim_sources"] = (
            normalized_local_managed_claim_sources
        )
        changed = True

    if "system_prompts" in data:
        data.pop("system_prompts", None)
        changed = True

    if data.get("settings_version") != version:
        data["settings_version"] = version
        changed = True

    return data, changed


def from_dict(data: dict[str, Any]) -> AppSettings:
    from puripuly_heart.config.settings_vnext import migration as vnext_migration

    if vnext_migration.is_vnext_settings_dict(data):
        return from_dict(vnext_migration.to_legacy_dict(vnext_migration.from_dict(data)))

    audio_data = data.get("audio") or {}
    desktop_audio_data = data.get("desktop_audio") or {}
    overlay_data = data.get("overlay") if isinstance(data.get("overlay"), dict) else {}
    legacy_overlay_calibration_data = (
        data.get("overlay_calibration") if isinstance(data.get("overlay_calibration"), dict) else {}
    )
    overlay_calibration_data = (
        overlay_data.get("calibration") if isinstance(overlay_data.get("calibration"), dict) else {}
    )
    merged_overlay_calibration_data = OverlayCalibration().to_dict()
    merged_overlay_calibration_data.update(legacy_overlay_calibration_data)
    merged_overlay_calibration_data.update(overlay_calibration_data)
    stt_data = data.get("stt") or {}
    ui_data = data.get("ui") or {}
    osc_data = data.get("osc") if isinstance(data.get("osc"), dict) else {}
    managed_identity_data = (
        data.get("managed_identity") if isinstance(data.get("managed_identity"), dict) else {}
    )
    peer_qwen_raw = (
        data.get("peer_qwen_asr_stt") if isinstance(data.get("peer_qwen_asr_stt"), dict) else {}
    )
    peer_soniox_data = (
        data.get("peer_soniox_stt") if isinstance(data.get("peer_soniox_stt"), dict) else {}
    )
    raw_provider_data = data.get("provider")
    provider_data = raw_provider_data if isinstance(raw_provider_data, dict) else {}
    if raw_provider_data is None:
        stt_provider_value = STTProviderName.LOCAL_CPU_AUTO.value
    elif isinstance(raw_provider_data, dict):
        stt_provider_value = provider_data.get("stt", STTProviderName.LOCAL_CPU_AUTO.value)
    else:
        stt_provider_value = STTProviderName.DEEPGRAM.value
    raw_peer_provider = (
        provider_data.get("peer_stt", STTProviderName.DEEPGRAM.value)
        if isinstance(raw_provider_data, dict)
        else STTProviderName.DEEPGRAM.value
    )

    input_host_api_raw = (
        audio_data["input_host_api"]
        if "input_host_api" in audio_data
        else WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    )
    input_device_raw = audio_data.get("input_device")
    vad_threshold_raw = stt_data.get("vad_speech_threshold")
    legacy_system_prompt = str(data.get("system_prompt", ""))
    settings_version = _coerce_int(data.get("settings_version"), SETTINGS_SCHEMA_VERSION)
    parsed_custom_terms = _parse_custom_terms(stt_data.get("custom_terms", _default_custom_terms()))
    if "custom_vocabulary_enabled" in stt_data:
        custom_vocabulary_enabled = bool(stt_data.get("custom_vocabulary_enabled"))
    else:
        custom_vocabulary_enabled = any(bool(terms) for terms in parsed_custom_terms.values())

    qwen_raw = data.get("qwen") if isinstance(data.get("qwen"), dict) else {}
    deepseek_raw = data.get("deepseek") if isinstance(data.get("deepseek"), dict) else {}
    cerebras_raw = data.get("cerebras") if isinstance(data.get("cerebras"), dict) else {}
    local_llm_raw = data.get("local_llm") if isinstance(data.get("local_llm"), dict) else {}
    qwen_asr_raw = data.get("qwen_asr_stt") if isinstance(data.get("qwen_asr_stt"), dict) else {}
    openrouter_raw = data.get("openrouter") if isinstance(data.get("openrouter"), dict) else {}
    telemetry_raw = data.get("telemetry") if isinstance(data.get("telemetry"), dict) else {}
    telemetry_state_raw = (
        data.get("telemetry_state") if isinstance(data.get("telemetry_state"), dict) else {}
    )
    openrouter_model, openrouter_selected_source, openrouter_selection_alias = (
        _resolve_openrouter_main_selection(openrouter_raw, data)
    )
    qwen_settings = QwenSettings(
        region=_parse_qwen_region(
            qwen_raw.get("region"),
            legacy_asr_endpoint=qwen_asr_raw.get("endpoint"),
        ),
        llm_model=_parse_qwen_llm_model(qwen_raw.get("llm_model", QwenLLMModel.QWEN_35_PLUS.value)),
    )

    settings = AppSettings(
        settings_version=settings_version,
        provider=ProviderSettings(
            stt=_parse_stt_provider(str(stt_provider_value)),
            peer_stt=_parse_peer_stt_provider(str(raw_peer_provider)),
            llm=_parse_llm_provider(provider_data.get("llm", LLMProviderName.GEMINI.value)),
        ),
        languages=LanguageSettings(
            source_language=data.get("languages", {}).get("source_language", "ko"),
            target_language=data.get("languages", {}).get("target_language", "en"),
            peer_source_language=str(data.get("languages", {}).get("peer_source_language", "")),
            peer_target_language=str(data.get("languages", {}).get("peer_target_language", "")),
            peer_source_mode=(
                "auto"
                if str(data.get("languages", {}).get("peer_source_mode", "manual")) == "soniox_auto"
                else (
                    str(data.get("languages", {}).get("peer_source_mode", "manual"))
                    if str(data.get("languages", {}).get("peer_source_mode", "manual"))
                    in {"manual", "auto"}
                    else "manual"
                )
            ),
            peer_expected_languages=(
                list(
                    dict.fromkeys(
                        str(language).strip()
                        for language in data.get("languages", {}).get("peer_expected_languages", [])
                        if str(language).strip()
                    )
                )
                if isinstance(data.get("languages", {}).get("peer_expected_languages"), list)
                else []
            ),
            recent_source_languages=list(
                dict.fromkeys(
                    list(data.get("languages", {}).get("recent_source_languages") or [])
                    + ["ko", "en", "zh-CN", "ja", "es", "fr"]
                )
            )[:6],
            recent_target_languages=list(
                dict.fromkeys(
                    list(data.get("languages", {}).get("recent_target_languages") or [])
                    + ["ko", "en", "zh-CN", "ja", "es", "fr"]
                )
            )[:6],
        ),
        audio=AudioSettings(
            internal_sample_rate_hz=_normalize_internal_sample_rate_hz(
                audio_data.get("internal_sample_rate_hz", STT_INTERNAL_SAMPLE_RATE_HZ)
            ),
            internal_channels=int(audio_data.get("internal_channels", 1)),
            ring_buffer_ms=int(audio_data.get("ring_buffer_ms", 500)),
            input_host_api=str(input_host_api_raw) if input_host_api_raw is not None else "",
            input_device=str(input_device_raw) if input_device_raw is not None else "",
        ),
        desktop_audio=DesktopAudioSettings(
            output_device=(
                str(desktop_audio_data.get("output_device"))
                if desktop_audio_data.get("output_device") is not None
                else ""
            ),
            vad_speech_threshold=float(desktop_audio_data.get("vad_speech_threshold", 0.5)),
            vad_hangover_ms=int(
                desktop_audio_data.get("vad_hangover_ms", DEFAULT_DESKTOP_AUDIO_VAD_HANGOVER_MS)
            ),
            vad_pre_roll_ms=int(desktop_audio_data.get("vad_pre_roll_ms", 500)),
        ),
        overlay=OverlaySettings(
            target=_parse_overlay_target(overlay_data.get("target")),
            show_translation=bool(
                overlay_data.get("show_translation", ui_data.get("show_overlay_translation", True))
            ),
            show_peer_original=bool(
                overlay_data.get(
                    "show_peer_original", ui_data.get("show_overlay_peer_original", True)
                )
            ),
            calibration=OverlayCalibration(
                anchor=str(
                    merged_overlay_calibration_data.get(
                        "anchor",
                        OverlayCalibration().anchor,
                    )
                ),
                offset_x=float(
                    merged_overlay_calibration_data.get(
                        "offset_x",
                        OverlayCalibration().offset_x,
                    )
                ),
                offset_y=float(
                    merged_overlay_calibration_data.get(
                        "offset_y",
                        OverlayCalibration().offset_y,
                    )
                ),
                distance=float(
                    merged_overlay_calibration_data.get(
                        "distance",
                        OverlayCalibration().distance,
                    )
                ),
                text_scale=float(
                    merged_overlay_calibration_data.get(
                        "text_scale",
                        OverlayCalibration().text_scale,
                    )
                ),
                background_alpha=float(
                    merged_overlay_calibration_data.get(
                        "background_alpha",
                        OverlayCalibration().background_alpha,
                    )
                ),
            ),
            desktop_flet=_parse_desktop_flet_settings(overlay_data.get("desktop_flet")),
        ),
        stt=STTSettings(
            drain_timeout_s=float(stt_data.get("drain_timeout_s", 2.0)),
            vad_speech_threshold=(
                float(vad_threshold_raw) if vad_threshold_raw is not None else 0.4
            ),
            low_latency_mode=bool(stt_data.get("low_latency_mode", False)),
            low_latency_vad_hangover_ms=int(
                stt_data.get(
                    "low_latency_vad_hangover_ms",
                    DEFAULT_LOW_LATENCY_VAD_HANGOVER_MS,
                )
            ),
            low_latency_merge_gap_ms=int(stt_data.get("low_latency_merge_gap_ms", 600)),
            low_latency_spec_retry_max=int(stt_data.get("low_latency_spec_retry_max", 10)),
            custom_vocabulary_enabled=custom_vocabulary_enabled,
            custom_terms=parsed_custom_terms,
            gpu_device_id=str(stt_data.get("gpu_device_id", "auto")).strip() or "auto",
        ),
        deepgram_stt=DeepgramSTTSettings(
            model=str(data.get("deepgram_stt", {}).get("model", "nova-3")),
        ),
        qwen_asr_stt=QwenASRSTTSettings(
            model=str(data.get("qwen_asr_stt", {}).get("model", "qwen3-asr-flash-realtime")),
            endpoint=qwen_settings.get_asr_endpoint(),
        ),
        soniox_stt=SonioxSTTSettings(
            model=str(data.get("soniox_stt", {}).get("model", "stt-rt-v5")),
            endpoint=str(
                data.get("soniox_stt", {}).get(
                    "endpoint", "wss://stt-rt.soniox.com/transcribe-websocket"
                )
            ),
            keepalive_interval_s=float(
                data.get("soniox_stt", {}).get("keepalive_interval_s", 10.0)
            ),
            trailing_silence_ms=int(data.get("soniox_stt", {}).get("trailing_silence_ms", 100)),
        ),
        custom_stt=_parse_custom_stt_settings(data.get("custom_stt")),
        peer_qwen_asr_stt=PeerQwenASRSTTSettings(
            model=_parse_optional_str(peer_qwen_raw.get("model")),
            region=(
                QwenRegion(peer_qwen_raw["region"])
                if peer_qwen_raw.get("region") in {region.value for region in QwenRegion}
                else None
            ),
        ),
        peer_soniox_stt=PeerSonioxSTTSettings(
            model=_parse_optional_str(peer_soniox_data.get("model")),
            endpoint=_parse_optional_str(peer_soniox_data.get("endpoint")),
            keepalive_interval_s=_parse_optional_float(
                peer_soniox_data.get("keepalive_interval_s")
            ),
            trailing_silence_ms=_parse_optional_int(peer_soniox_data.get("trailing_silence_ms")),
        ),
        gemini=GeminiSettings(
            llm_model=_parse_gemini_llm_model(
                data.get("gemini", {}).get("llm_model", GeminiLLMModel.GEMINI_31_FLASH_LITE.value)
            ),
        ),
        openrouter=OpenRouterSettings(
            llm_model=openrouter_model,
            routing_mode=_parse_openrouter_routing_mode(
                openrouter_raw.get(
                    "routing_mode",
                    OpenRouterRoutingMode.LATENCY.value,
                )
            ),
            provider_routing=_parse_openrouter_provider_routing(
                openrouter_raw.get("provider_routing")
            ),
            selected_source=openrouter_selected_source,
            selection_alias=openrouter_selection_alias,
            fallback_selection_alias=_parse_openrouter_fallback_selection_alias(
                openrouter_raw.get("fallback_selection_alias")
            ),
            broker_base_url=_parse_openrouter_broker_base_url(
                openrouter_raw.get("broker_base_url")
            ),
        ),
        qwen=qwen_settings,
        deepseek=DeepSeekSettings(
            llm_model=_parse_deepseek_llm_model(
                deepseek_raw.get("llm_model", DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value)
            ),
        ),
        cerebras=CerebrasSettings(
            llm_model=_parse_cerebras_llm_model(
                cerebras_raw.get("llm_model", CerebrasLLMModel.GEMMA_4_31B.value)
            ),
        ),
        local_llm=LocalLLMSettings(
            backend=_parse_local_llm_backend(local_llm_raw.get("backend")),
            base_url=_parse_local_llm_base_url(local_llm_raw.get("base_url")),
            model=_parse_local_llm_model(local_llm_raw.get("model")),
            extra_body=_parse_local_llm_extra_body(local_llm_raw.get("extra_body")),
        ),
        llm=LLMSettings(concurrency_limit=int(data.get("llm", {}).get("concurrency_limit", 5))),
        osc=OSCSettings(
            host=str(data.get("osc", {}).get("host", "127.0.0.1")),
            port=int(data.get("osc", {}).get("port", data.get("osc", {}).get("send_port", 9000))),
            send_port=(
                int(data["osc"]["send_port"])
                if isinstance(data.get("osc"), dict) and "send_port" in data["osc"]
                else None
            ),
            receive_port=int(data.get("osc", {}).get("receive_port", 9001)),
            connection_mode=str(osc_data.get("connection_mode", "automatic")),
            chatbox_address=str(data.get("osc", {}).get("chatbox_address", "/chatbox/input")),
            chatbox_send=bool(data.get("osc", {}).get("chatbox_send", True)),
            chatbox_clear=bool(data.get("osc", {}).get("chatbox_clear", False)),
            chatbox_max_chars=int(data.get("osc", {}).get("chatbox_max_chars", 144)),
            vrc_mic_intercept=bool(data.get("osc", {}).get("vrc_mic_intercept", False)),
            chatbox_include_source=bool(data.get("osc", {}).get("chatbox_include_source", False)),
        ),
        secrets=SecretsSettings(
            backend=SecretsBackend(
                data.get("secrets", {}).get("backend", SecretsBackend.KEYRING.value)
            ),
            encrypted_file_path=data.get("secrets", {}).get("encrypted_file_path", "secrets.json"),
        ),
        ui=UiSettings(
            locale=str(ui_data.get("locale", "en")),
            overlay_enabled=False,
            peer_translation_enabled=False,
            peer_translation_eula_accepted=bool(
                ui_data.get("peer_translation_eula_accepted", False)
            ),
            integrated_context_enabled=bool(ui_data.get("integrated_context_enabled", True)),
            integrated_context_bootstrapped=bool(
                ui_data.get("integrated_context_bootstrapped", False)
            ),
            clipboard_auto_translate_enabled=bool(
                ui_data.get("clipboard_auto_translate_enabled", False)
            ),
            github_star_prompt_clicked=_parse_bool(ui_data.get("github_star_prompt_clicked")),
            github_star_prompt_last_shown_at=_parse_utc_iso8601_timestamp(
                ui_data.get("github_star_prompt_last_shown_at")
            ),
            github_star_prompt_show_count=_parse_non_negative_int(
                ui_data.get("github_star_prompt_show_count")
            ),
            github_star_prompt_translation_success_observed=_parse_bool(
                ui_data.get("github_star_prompt_translation_success_observed")
            ),
            github_star_prompt_eligible_launch_count=_parse_non_negative_int(
                ui_data.get("github_star_prompt_eligible_launch_count")
            ),
        ),
        api_key_verified=ApiKeyVerificationSettings(
            deepgram=bool(data.get("api_key_verified", {}).get("deepgram", False)),
            soniox=bool(data.get("api_key_verified", {}).get("soniox", False)),
            google=bool(data.get("api_key_verified", {}).get("google", False)),
            openrouter=bool(data.get("api_key_verified", {}).get("openrouter", False)),
            deepseek=bool(data.get("api_key_verified", {}).get("deepseek", False)),
            alibaba_beijing=bool(data.get("api_key_verified", {}).get("alibaba_beijing", False)),
            alibaba_singapore=bool(
                data.get("api_key_verified", {}).get("alibaba_singapore", False)
            ),
            cerebras=bool(data.get("api_key_verified", {}).get("cerebras", False)),
        ),
        managed_identity=ManagedIdentitySettings(
            installation_id=_parse_optional_str(managed_identity_data.get("installation_id")) or "",
            release_token=_parse_optional_str(managed_identity_data.get("release_token")),
            release_token_expires_at=_parse_optional_str(
                managed_identity_data.get("release_token_expires_at")
            ),
            verified_hardware_hash=_parse_optional_str(
                managed_identity_data.get("verified_hardware_hash")
            ),
            verified_hardware_hash_salt_version=_parse_optional_int(
                managed_identity_data.get("verified_hardware_hash_salt_version")
            ),
            active_managed_credential_ref=_parse_optional_str(
                managed_identity_data.get("active_managed_credential_ref")
            ),
            active_managed_expires_at=_parse_optional_str(
                managed_identity_data.get("active_managed_expires_at")
            ),
            founder_letter_seen_credential_ref=_parse_optional_str(
                managed_identity_data.get("founder_letter_seen_credential_ref")
            ),
            referral_id=normalize_owned_referral_id(managed_identity_data.get("referral_id")),
            local_managed_claim_sources=normalize_managed_claim_sources(
                managed_identity_data.get("local_managed_claim_sources")
            ),
            pending_delivery_ack_source=_parse_optional_str(
                managed_identity_data.get("pending_delivery_ack_source")
            ),
            pending_delivery_ack_delivery_id=_parse_optional_str(
                managed_identity_data.get("pending_delivery_ack_delivery_id")
            ),
            pending_delivery_ack_managed_credential_ref=_parse_optional_str(
                managed_identity_data.get("pending_delivery_ack_managed_credential_ref")
            ),
            pending_delivery_ack_expires_at=_parse_optional_str(
                managed_identity_data.get("pending_delivery_ack_expires_at")
            ),
        ),
        telemetry=TelemetrySettings(
            consent=_parse_telemetry_consent(telemetry_raw.get("consent")),
        ),
        telemetry_state=TelemetryStateSettings(
            anonymous_id=_normalize_telemetry_identifier(telemetry_state_raw.get("anonymous_id")),
            sent_translation_success_dates_utc=_normalize_telemetry_sent_dates(
                telemetry_state_raw.get("sent_translation_success_dates_utc")
            ),
        ),
        system_prompt=legacy_system_prompt,
        system_prompts={},
    )

    translation_data = data.get("translation") if isinstance(data.get("translation"), dict) else {}
    translation_history = _parse_translation_connection_history(
        translation_data.get("connection_history") if isinstance(translation_data, dict) else None
    )
    if translation_data.get("previous_llm_model") == "gemma4_31b_cerebras":
        translation_history[TranslationModel.GEMMA4_31B.value] = TranslationConnection.CEREBRAS
    if _translation_data_has_valid_model(translation_data):
        translation_model, translation_connection = _parse_translation_selection(
            translation_data.get("model"),
            translation_data.get("connection"),
        )
        settings.translation = _normalize_translation_settings(
            model=translation_model,
            connection=translation_connection,
            fallback=translation_data.get("fallback"),
            history=translation_history,
            http_extension_id=translation_data.get("http_extension_id"),
            previous_llm_model=translation_data.get("previous_llm_model"),
        )
    else:
        settings.translation = _derive_translation_settings_from_runtime(
            settings,
            history=translation_history,
        )
    materialize_translation_settings(settings)

    ensure_prompt_defaults(settings)
    settings = ensure_telemetry_default_allow(settings)
    settings.validate()
    return settings


from puripuly_heart.config.settings_vnext.facade import (  # noqa: E402,F401
    FacadeSettingsLoadResult,
    load_settings,
    load_settings_with_result,
    load_vnext_settings,
    save_settings,
    save_settings_with_result,
    save_vnext_settings,
)

_FacadeSettingsLoadResult = FacadeSettingsLoadResult
