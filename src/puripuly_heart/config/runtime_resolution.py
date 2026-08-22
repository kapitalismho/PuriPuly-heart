from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, cast

from puripuly_heart.config.llm_profiles import (
    LEGACY_OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
    OPENROUTER_CREDENTIAL_SOURCE_BYOK,
    OPENROUTER_CREDENTIAL_SOURCE_MANAGED,
    OPENROUTER_CREDENTIAL_SOURCE_NONE,
    OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
    OPENROUTER_MODEL_GEMINI_31_FLASH_LITE,
    OPENROUTER_MODEL_GEMINI_37_FLASH,
    OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
    OPENROUTER_MODEL_GEMMA_4_31B_IT,
    OPENROUTER_MODEL_QWEN_35_FLASH_02_23,
    get_openrouter_llm_profile,
    openrouter_alias_for_fields,
)
from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_MANAGED,
    CREDENTIAL_SOURCE_NONE,
    CREDENTIAL_SOURCE_SECRET_STORE,
    OVERLAY_TARGET_DESKTOP,
    OVERLAY_TARGET_STEAMVR,
    RUNTIME_CHANNEL_PEER,
    RUNTIME_CHANNEL_SELF,
    ResolvedCredentialRequirement,
    ResolvedLLMAttemptPlan,
    ResolvedLLMConfig,
    ResolvedLLMFallbackPlan,
    ResolvedLLMTarget,
    ResolvedOptionValue,
    ResolvedOverlayConfig,
    ResolvedSTTConfig,
)

TRANSLATION_MODEL_GEMMA4: Final = "gemma4"
TRANSLATION_MODEL_GEMMA4_26B_31B: Final = "gemma4_26b_31b"
TRANSLATION_MODEL_GEMMA4_31B: Final = "gemma4_31b"
TRANSLATION_MODEL_DEEPSEEK_V4_FLASH: Final = "deepseek_v4_flash"
TRANSLATION_MODEL_GEMINI_37_FLASH: Final = "gemini37_flash"
TRANSLATION_MODEL_GEMINI_31_FLASH_LITE: Final = "gemini31_flash_lite"
TRANSLATION_MODEL_QWEN_35_PLUS: Final = "qwen35_plus"
TRANSLATION_MODEL_OPENROUTER_QWEN_35_FLASH: Final = "openrouter_qwen35_flash"
TRANSLATION_MODEL_MANAGED_GEMMA: Final = "managed_gemma"
TRANSLATION_MODEL_MANAGED_GEMMA_12B: Final = "managed_gemma_12b"
TRANSLATION_MODEL_LOCAL_LLM: Final = "local_llm"
TRANSLATION_MODEL_CUSTOM_HTTP: Final = "custom_http"

_FIRST_HEDGE_DELAY_MS: Final = 1300
_EMERGENCY_HEDGE_DELAY_MS: Final = 4500
_LOSER_GRACE_MS: Final = 50

TranslationModelName: TypeAlias = Literal[
    "gemma4_26b_31b",
    "gemma4_31b",
    "gemma4",
    "deepseek_v4_flash",
    "gemini37_flash",
    "gemini31_flash_lite",
    "qwen35_plus",
    "openrouter_qwen35_flash",
    "managed_gemma",
    "managed_gemma_12b",
    "local_llm",
    "custom_http",
]
TRANSLATION_MODELS: Final[tuple[TranslationModelName, ...]] = (
    TRANSLATION_MODEL_GEMMA4_26B_31B,
    TRANSLATION_MODEL_GEMMA4_31B,
    TRANSLATION_MODEL_GEMMA4,
    TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
    TRANSLATION_MODEL_GEMINI_37_FLASH,
    TRANSLATION_MODEL_GEMINI_31_FLASH_LITE,
    TRANSLATION_MODEL_QWEN_35_PLUS,
    TRANSLATION_MODEL_OPENROUTER_QWEN_35_FLASH,
    TRANSLATION_MODEL_MANAGED_GEMMA,
    TRANSLATION_MODEL_MANAGED_GEMMA_12B,
    TRANSLATION_MODEL_LOCAL_LLM,
    TRANSLATION_MODEL_CUSTOM_HTTP,
)

TRANSLATION_CONNECTION_MANAGED: Final = "managed"
TRANSLATION_CONNECTION_MANAGED_CHINA: Final = "managed_china"
TRANSLATION_CONNECTION_OPENROUTER: Final = "openrouter"
TRANSLATION_CONNECTION_CEREBRAS: Final = "cerebras"
TRANSLATION_CONNECTION_OFFICIAL_BYOK: Final = "official_byok"
TRANSLATION_CONNECTION_OLLAMA: Final = "ollama"
TRANSLATION_CONNECTION_CPU: Final = "cpu"
TRANSLATION_CONNECTION_GPU: Final = "gpu"
TRANSLATION_CONNECTION_CUSTOM_HTTP: Final = "custom_http"

TranslationConnectionName: TypeAlias = Literal[
    "managed",
    "managed_china",
    "openrouter",
    "cerebras",
    "official_byok",
    "ollama",
    "cpu",
    "gpu",
    "custom_http",
]
TRANSLATION_CONNECTIONS: Final[tuple[TranslationConnectionName, ...]] = (
    TRANSLATION_CONNECTION_MANAGED,
    TRANSLATION_CONNECTION_MANAGED_CHINA,
    TRANSLATION_CONNECTION_OPENROUTER,
    TRANSLATION_CONNECTION_CEREBRAS,
    TRANSLATION_CONNECTION_OFFICIAL_BYOK,
    TRANSLATION_CONNECTION_OLLAMA,
    TRANSLATION_CONNECTION_CPU,
    TRANSLATION_CONNECTION_GPU,
    TRANSLATION_CONNECTION_CUSTOM_HTTP,
)
TRANSLATION_CONNECTIONS_BY_MODEL: Final[
    Mapping[TranslationModelName, tuple[TranslationConnectionName, ...]]
] = MappingProxyType(
    {
        TRANSLATION_MODEL_GEMMA4_26B_31B: (
            TRANSLATION_CONNECTION_MANAGED,
            TRANSLATION_CONNECTION_OPENROUTER,
        ),
        TRANSLATION_MODEL_GEMMA4_31B: (
            TRANSLATION_CONNECTION_MANAGED,
            TRANSLATION_CONNECTION_OPENROUTER,
            TRANSLATION_CONNECTION_CEREBRAS,
        ),
        TRANSLATION_MODEL_GEMMA4: (
            TRANSLATION_CONNECTION_MANAGED,
            TRANSLATION_CONNECTION_OPENROUTER,
        ),
        TRANSLATION_MODEL_DEEPSEEK_V4_FLASH: (
            TRANSLATION_CONNECTION_MANAGED,
            TRANSLATION_CONNECTION_MANAGED_CHINA,
            TRANSLATION_CONNECTION_OPENROUTER,
            TRANSLATION_CONNECTION_OFFICIAL_BYOK,
        ),
        TRANSLATION_MODEL_GEMINI_37_FLASH: (
            TRANSLATION_CONNECTION_OFFICIAL_BYOK,
            TRANSLATION_CONNECTION_OPENROUTER,
        ),
        TRANSLATION_MODEL_GEMINI_31_FLASH_LITE: (
            TRANSLATION_CONNECTION_OFFICIAL_BYOK,
            TRANSLATION_CONNECTION_OPENROUTER,
        ),
        TRANSLATION_MODEL_QWEN_35_PLUS: (TRANSLATION_CONNECTION_OFFICIAL_BYOK,),
        TRANSLATION_MODEL_OPENROUTER_QWEN_35_FLASH: (
            TRANSLATION_CONNECTION_MANAGED,
            TRANSLATION_CONNECTION_OPENROUTER,
        ),
        TRANSLATION_MODEL_MANAGED_GEMMA: (
            TRANSLATION_CONNECTION_CPU,
            TRANSLATION_CONNECTION_GPU,
        ),
        TRANSLATION_MODEL_MANAGED_GEMMA_12B: (TRANSLATION_CONNECTION_GPU,),
        TRANSLATION_MODEL_LOCAL_LLM: (TRANSLATION_CONNECTION_OLLAMA,),
        TRANSLATION_MODEL_CUSTOM_HTTP: (TRANSLATION_CONNECTION_CUSTOM_HTTP,),
    }
)
TRANSLATION_CONNECTION_PRIORITY: Final[tuple[TranslationConnectionName, ...]] = (
    TRANSLATION_CONNECTION_MANAGED,
    TRANSLATION_CONNECTION_OPENROUTER,
    TRANSLATION_CONNECTION_OFFICIAL_BYOK,
)

OPENROUTER_SOURCE_NONE: Final = OPENROUTER_CREDENTIAL_SOURCE_NONE
OPENROUTER_SOURCE_MANAGED: Final = OPENROUTER_CREDENTIAL_SOURCE_MANAGED
OPENROUTER_SOURCE_BYOK: Final = OPENROUTER_CREDENTIAL_SOURCE_BYOK
OpenRouterSource: TypeAlias = Literal["none", "managed", "byok"]
OPENROUTER_SOURCES: Final[tuple[OpenRouterSource, ...]] = (
    OPENROUTER_SOURCE_NONE,
    OPENROUTER_SOURCE_MANAGED,
    OPENROUTER_SOURCE_BYOK,
)
OPENROUTER_MANAGED_CREDENTIAL_STANDARD: Final = "standard"
OPENROUTER_MANAGED_CREDENTIAL_QQ: Final = "qq"
OpenRouterManagedCredentialKind: TypeAlias = Literal["standard", "qq"]
OPENROUTER_MANAGED_CREDENTIAL_KINDS: Final[tuple[OpenRouterManagedCredentialKind, ...]] = (
    OPENROUTER_MANAGED_CREDENTIAL_STANDARD,
    OPENROUTER_MANAGED_CREDENTIAL_QQ,
)

PROVIDER_OPENROUTER: Final = "openrouter"
PROVIDER_DEEPSEEK: Final = "deepseek"
PROVIDER_GEMINI: Final = "gemini"
PROVIDER_QWEN: Final = "qwen"
PROVIDER_MANAGED_GEMMA: Final = "managed_gemma"
PROVIDER_LOCAL_LLM: Final = "local_llm"
PROVIDER_CEREBRAS: Final = "cerebras"
PROVIDER_CUSTOM_HTTP: Final = "custom_http"
LLM_PROVIDERS: Final[tuple[str, ...]] = (
    PROVIDER_GEMINI,
    PROVIDER_OPENROUTER,
    PROVIDER_QWEN,
    PROVIDER_MANAGED_GEMMA,
    PROVIDER_DEEPSEEK,
    PROVIDER_LOCAL_LLM,
    PROVIDER_CEREBRAS,
)

GEMINI_MODEL_37_FLASH: Final = "gemini-3.7-flash"
GEMINI_MODEL_31_FLASH_LITE: Final = "gemini-3.1-flash-lite"
DEEPSEEK_MODEL_V4_FLASH: Final = "deepseek-v4-flash"
QWEN_MODEL_35_FLASH: Final = "qwen3.5-flash"
QWEN_MODEL_35_PLUS: Final = "qwen3.5-plus"
LOCAL_LLM_BACKEND_OLLAMA: Final = "ollama"
LOCAL_LLM_DEFAULT_BASE_URL: Final = "http://127.0.0.1:11434/v1"
LOCAL_LLM_DEFAULT_MODEL: Final = "llama3.1:8b"
MANAGED_GEMMA_MODEL: Final = "puripuly-gemma-4-e4b-q4"
MANAGED_GEMMA_12B_MODEL: Final = "puripuly-gemma-4-12b-q4"
CEREBRAS_MODEL_GEMMA_4_31B: Final = "gemma-4-31b"
QWEN_REGION_BEIJING: Final = "beijing"
QWEN_REGION_SINGAPORE: Final = "singapore"

CREDENTIAL_REF_OPENROUTER_BYOK: Final = "openrouter:byok"
CREDENTIAL_REF_OPENROUTER_MANAGED: Final = "openrouter:managed"
CREDENTIAL_REF_OPENROUTER_MANAGED_QQ: Final = "openrouter:managed_qq"
CREDENTIAL_REF_GEMINI_BYOK: Final = "gemini:byok"
CREDENTIAL_REF_DEEPSEEK_BYOK: Final = "deepseek:byok"
CREDENTIAL_REF_CEREBRAS_BYOK: Final = "cerebras:byok"
CREDENTIAL_REF_QWEN_BEIJING: Final = "qwen:beijing"
CREDENTIAL_REF_QWEN_SINGAPORE: Final = "qwen:singapore"
CREDENTIAL_REF_DEEPGRAM_STT: Final = "deepgram:stt"
CREDENTIAL_REF_SONIOX_STT: Final = "soniox:stt"
CREDENTIAL_REF_CUSTOM_STT: Final = "custom:stt"

STT_PROVIDER_LOCAL_CPU_AUTO: Final = "local_cpu_auto"
STT_PROVIDER_LOCAL_PARAKEET_V3: Final = "local_parakeet_v3"
STT_PROVIDER_LOCAL_PARAKEET_JAPANESE: Final = "local_parakeet_ja"
STT_PROVIDER_LOCAL_QWEN: Final = "local_qwen"
STT_PROVIDER_LOCAL_QWEN_GPU: Final = "local_qwen_gpu"
STT_PROVIDER_DEEPGRAM: Final = "deepgram"
STT_PROVIDER_QWEN_ASR: Final = "qwen_asr"
STT_PROVIDER_SONIOX: Final = "soniox"
STT_PROVIDER_CUSTOM: Final = "custom"
STT_PROVIDER_CUSTOM_OFFLINE: Final = "custom_offline"
STT_PROVIDER_CUSTOM_REALTIME: Final = "custom_realtime"
STT_CUSTOM_PROVIDERS: Final[tuple[str, ...]] = (
    STT_PROVIDER_CUSTOM,
    STT_PROVIDER_CUSTOM_OFFLINE,
    STT_PROVIDER_CUSTOM_REALTIME,
)
STT_PROVIDERS: Final[tuple[str, ...]] = (
    STT_PROVIDER_LOCAL_CPU_AUTO,
    STT_PROVIDER_LOCAL_PARAKEET_V3,
    STT_PROVIDER_LOCAL_PARAKEET_JAPANESE,
    STT_PROVIDER_LOCAL_QWEN,
    STT_PROVIDER_LOCAL_QWEN_GPU,
    STT_PROVIDER_DEEPGRAM,
    STT_PROVIDER_QWEN_ASR,
    STT_PROVIDER_SONIOX,
    STT_PROVIDER_CUSTOM,
    STT_PROVIDER_CUSTOM_OFFLINE,
    STT_PROVIDER_CUSTOM_REALTIME,
)
PEER_AUTO_DETECTION_STT_PROVIDERS: Final[tuple[str, ...]] = (
    STT_PROVIDER_LOCAL_QWEN_GPU,
    STT_PROVIDER_SONIOX,
)
STT_DEFAULT_SOURCE_LANGUAGE: Final = "ko"
STT_DEFAULT_PEER_SOURCE_LANGUAGE: Final = "en"
STT_DEFAULT_SAMPLE_RATE_HZ: Final = 16000
STT_DEFAULT_CHANNELS: Final = 1
STT_DEFAULT_RING_BUFFER_MS: Final = 500
STT_DEFAULT_DRAIN_TIMEOUT_S: Final = 2.0
STT_DEFAULT_VAD_SPEECH_THRESHOLD: Final = 0.4
STT_DEFAULT_VAD_HANGOVER_MS: Final = 500
STT_DEFAULT_VAD_PRE_ROLL_MS: Final = 500
PEER_STT_DEFAULT_VAD_SPEECH_THRESHOLD: Final = 0.5
PEER_STT_DEFAULT_VAD_HANGOVER_MS: Final = 500
PEER_STT_DEFAULT_VAD_PRE_ROLL_MS: Final = 500
STT_DEFAULT_LOW_LATENCY_ENABLED: Final = True
STT_DEFAULT_LOW_LATENCY_MERGE_GAP_MS: Final = 600
STT_DEFAULT_LOW_LATENCY_SPEC_RETRY_MAX: Final = 10
DEEPGRAM_STT_MODEL_NOVA_3: Final = "nova-3"
QWEN_ASR_STT_MODEL_REALTIME: Final = "qwen3-asr-flash-realtime"
SONIOX_STT_MODEL_RT_V5: Final = "stt-rt-v5"
SONIOX_STT_DEFAULT_ENDPOINT: Final = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_STT_DEFAULT_KEEPALIVE_INTERVAL_S: Final = 10.0
SONIOX_STT_DEFAULT_TRAILING_SILENCE_MS: Final = 100

_OPENROUTER_MODELS: Final[tuple[str, ...]] = (
    OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
    OPENROUTER_MODEL_GEMMA_4_31B_IT,
    OPENROUTER_MODEL_QWEN_35_FLASH_02_23,
    OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
    OPENROUTER_MODEL_GEMINI_37_FLASH,
    OPENROUTER_MODEL_GEMINI_31_FLASH_LITE,
)
_OPENROUTER_ROUTING_MODES: Final[tuple[str, ...]] = ("latency",)
_OPENROUTER_PROVIDER_ROUTINGS: Final[tuple[str, ...]] = (
    "default",
    "deepseek_only",
    "google_gemini_latency",
    "gemma4_26b_31b_latency",
    "gemma4_31b_latency",
    "gemma4_26b_latency",
    "deepseek_v4_flash_latency",
    "gemma4_31b_cerebras_only",
)


def _empty_options() -> Mapping[str, ResolvedOptionValue]:
    return MappingProxyType({})


def _empty_custom_terms() -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({})


def _freeze_option_mapping(values: Mapping[str, object]) -> Mapping[str, ResolvedOptionValue]:
    frozen: dict[str, ResolvedOptionValue] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError("runtime intent option keys must be strings")
        frozen[key] = _freeze_option_value(value)
    return MappingProxyType(frozen)


def _freeze_option_value(value: object) -> ResolvedOptionValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return _freeze_option_mapping(cast(Mapping[str, object], value))
    if isinstance(value, tuple | list):
        return tuple(_freeze_option_value(item) for item in value)
    raise TypeError("runtime intent option values must be scalars, mappings, lists, or tuples")


def _freeze_custom_terms(values: Mapping[str, object]) -> Mapping[str, tuple[str, ...]]:
    frozen: dict[str, tuple[str, ...]] = {}
    for language, terms in values.items():
        if not isinstance(language, str):
            raise ValueError("custom_terms keys must be strings")
        if isinstance(terms, str) or not isinstance(terms, tuple | list):
            raise ValueError("custom_terms values must be lists or tuples of strings")
        if not all(isinstance(term, str) for term in terms):
            raise ValueError("custom_terms values must contain only strings")
        frozen[language] = tuple(terms)
    return MappingProxyType(frozen)


def _normalize_string(value: object, *, default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return default


def _normalize_allowed(value: object, *, allowed: tuple[str, ...], default: str) -> str:
    normalized = _normalize_string(value, default=default)
    if normalized in allowed:
        return normalized
    return default


def _require_allowed(value: str, allowed: tuple[str, ...], *, field_name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {', '.join(allowed)}")


def _default_translation_connection(model: TranslationModelName) -> TranslationConnectionName:
    supported = TRANSLATION_CONNECTIONS_BY_MODEL[model]
    for connection in TRANSLATION_CONNECTION_PRIORITY:
        if connection in supported:
            return connection
    return supported[0]


def _normalize_translation_model(value: object) -> TranslationModelName:
    if isinstance(value, str) and value.strip() == "gemini3_flash":
        value = TRANSLATION_MODEL_GEMINI_37_FLASH
    return cast(
        TranslationModelName,
        _normalize_allowed(
            value,
            allowed=TRANSLATION_MODELS,
            default=TRANSLATION_MODEL_GEMMA4_26B_31B,
        ),
    )


def _normalize_translation_connection(
    value: object,
    *,
    model: TranslationModelName,
) -> TranslationConnectionName:
    connection = cast(
        TranslationConnectionName,
        _normalize_allowed(
            value,
            allowed=TRANSLATION_CONNECTIONS,
            default=_default_translation_connection(model),
        ),
    )
    if connection in TRANSLATION_CONNECTIONS_BY_MODEL[model]:
        return connection
    return _default_translation_connection(model)


def _normalize_positive_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value > 0:
        return value
    return default


def _normalize_openrouter_source(value: object, *, default: OpenRouterSource) -> OpenRouterSource:
    return cast(
        OpenRouterSource,
        _normalize_allowed(
            value,
            allowed=OPENROUTER_SOURCES,
            default=default,
        ),
    )


def _default_openrouter_source_for_provider(provider_llm: object) -> OpenRouterSource:
    if provider_llm == PROVIDER_OPENROUTER:
        return OPENROUTER_SOURCE_BYOK
    if isinstance(provider_llm, str) and provider_llm.strip() == PROVIDER_OPENROUTER:
        return OPENROUTER_SOURCE_BYOK
    if isinstance(provider_llm, str) and provider_llm.strip() in LLM_PROVIDERS:
        return OPENROUTER_SOURCE_NONE
    return OPENROUTER_SOURCE_MANAGED


def _explicit_openrouter_source(value: object) -> OpenRouterSource | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in OPENROUTER_SOURCES:
            return cast(OpenRouterSource, normalized)
    return None


def _normalize_openrouter_model(value: object) -> str:
    if isinstance(value, str) and value.strip() == LEGACY_OPENROUTER_MODEL_DEEPSEEK_V4_FLASH:
        value = OPENROUTER_MODEL_DEEPSEEK_V4_FLASH
    if isinstance(value, str) and value.strip() == "google/gemini-3-flash-preview":
        value = OPENROUTER_MODEL_GEMINI_37_FLASH
    return _normalize_allowed(
        value,
        allowed=_OPENROUTER_MODELS,
        default=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
    )


def _normalize_gemini_model(value: object) -> str:
    if isinstance(value, str) and value.strip() in {"gemini-3-flash", "gemini-3-flash-preview"}:
        return GEMINI_MODEL_37_FLASH
    return _normalize_allowed(
        value,
        allowed=(GEMINI_MODEL_37_FLASH, GEMINI_MODEL_31_FLASH_LITE),
        default=GEMINI_MODEL_31_FLASH_LITE,
    )


def _normalize_openrouter_managed_credential_kind(
    value: object,
    *,
    default: OpenRouterManagedCredentialKind = OPENROUTER_MANAGED_CREDENTIAL_STANDARD,
) -> OpenRouterManagedCredentialKind:
    return cast(
        OpenRouterManagedCredentialKind,
        _normalize_allowed(
            value,
            allowed=OPENROUTER_MANAGED_CREDENTIAL_KINDS,
            default=default,
        ),
    )


def _canonical_openrouter_alias(
    model: str,
    source: OpenRouterSource,
    models: tuple[str, ...] = (),
) -> str | None:
    if source == OPENROUTER_SOURCE_NONE:
        return None
    return openrouter_alias_for_fields(model=model, source=source, models=models)


def _normalize_openrouter_broker_base_url(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return None


_CUSTOM_STT_COMPATIBILITIES_BY_MODE: Final[Mapping[str, tuple[str, ...]]] = {
    "offline": ("openai_transcription",),
    "realtime": ("openai_realtime",),
}


def _normalize_custom_stt_mode(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"streaming", "realtime", "real-time"}:
            return "realtime"
        if normalized == "offline":
            return "offline"
    return "offline"


def _normalize_custom_stt_compatibility(value: object, *, mode: str) -> str:
    allowed = _CUSTOM_STT_COMPATIBILITIES_BY_MODE[mode]
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in allowed:
            return normalized
        if normalized:
            raise ValueError(
                f"Custom STT compatibility {normalized} is not supported in {mode} mode"
            )
    return allowed[0]


def _no_credential() -> ResolvedCredentialRequirement:
    return ResolvedCredentialRequirement(
        source=CREDENTIAL_SOURCE_NONE,
        required=False,
        reference=None,
    )


def _required_credential(source: str, reference: str) -> ResolvedCredentialRequirement:
    return ResolvedCredentialRequirement(source=source, required=True, reference=reference)


def _openrouter_credential(
    source: OpenRouterSource,
    *,
    managed_credential_kind: OpenRouterManagedCredentialKind = OPENROUTER_MANAGED_CREDENTIAL_STANDARD,
) -> ResolvedCredentialRequirement:
    if source == OPENROUTER_SOURCE_MANAGED:
        reference = (
            CREDENTIAL_REF_OPENROUTER_MANAGED_QQ
            if managed_credential_kind == OPENROUTER_MANAGED_CREDENTIAL_QQ
            else CREDENTIAL_REF_OPENROUTER_MANAGED
        )
        return _required_credential(CREDENTIAL_SOURCE_MANAGED, reference)
    if source == OPENROUTER_SOURCE_BYOK:
        return _required_credential(
            CREDENTIAL_SOURCE_SECRET_STORE,
            CREDENTIAL_REF_OPENROUTER_BYOK,
        )
    return _no_credential()


def _qwen_credential_reference(region: str) -> str:
    if region == QWEN_REGION_SINGAPORE:
        return CREDENTIAL_REF_QWEN_SINGAPORE
    return CREDENTIAL_REF_QWEN_BEIJING


def _qwen_service_endpoint(region: str) -> str:
    if region == QWEN_REGION_SINGAPORE:
        return "https://dashscope-intl.aliyuncs.com/api/v1"
    return "https://dashscope.aliyuncs.com/api/v1"


def _qwen_asr_endpoint(region: str) -> str:
    if region == QWEN_REGION_SINGAPORE:
        return "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
    return "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"


def _translation_connection_from_openrouter_source(
    selected_source: OpenRouterSource,
    *,
    model: TranslationModelName,
    provider_routing: str,
) -> TranslationConnectionName:
    if selected_source == OPENROUTER_SOURCE_MANAGED:
        if model == TRANSLATION_MODEL_DEEPSEEK_V4_FLASH and provider_routing == "deepseek_only":
            return TRANSLATION_CONNECTION_MANAGED_CHINA
        return TRANSLATION_CONNECTION_MANAGED
    if selected_source == OPENROUTER_SOURCE_BYOK:
        return TRANSLATION_CONNECTION_OPENROUTER
    return _default_translation_connection(model)


@dataclass(frozen=True, slots=True)
class TranslationRuntimeIntent:
    model: TranslationModelName = TRANSLATION_MODEL_GEMMA4_26B_31B
    connection: TranslationConnectionName = TRANSLATION_CONNECTION_MANAGED
    concurrency_limit: int = 5

    def __post_init__(self) -> None:
        model = _normalize_translation_model(self.model)
        connection = _normalize_translation_connection(self.connection, model=model)
        concurrency_limit = _normalize_positive_int(self.concurrency_limit, default=5)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "connection", connection)
        object.__setattr__(self, "concurrency_limit", concurrency_limit)
        _require_allowed(model, TRANSLATION_MODELS, field_name="model")
        if connection not in TRANSLATION_CONNECTIONS_BY_MODEL[model]:
            raise ValueError("translation connection is not supported for model")


@dataclass(frozen=True, slots=True)
class TranslationFallbackRuntimeIntent:
    enabled: bool = False
    model: TranslationModelName = TRANSLATION_MODEL_DEEPSEEK_V4_FLASH
    connection: TranslationConnectionName = TRANSLATION_CONNECTION_OFFICIAL_BYOK

    def __post_init__(self) -> None:
        model = _normalize_translation_model(self.model)
        connection = _normalize_translation_connection(self.connection, model=model)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "connection", connection)
        _require_allowed(model, TRANSLATION_MODELS, field_name="fallback model")
        if model == TRANSLATION_MODEL_CUSTOM_HTTP:
            raise ValueError("custom HTTP translation cannot be used as fallback")
        if model in (TRANSLATION_MODEL_MANAGED_GEMMA, TRANSLATION_MODEL_MANAGED_GEMMA_12B):
            raise ValueError("managed local Gemma cannot be used as provider fallback")
        if connection not in TRANSLATION_CONNECTIONS_BY_MODEL[model]:
            raise ValueError("translation fallback connection is not supported for model")


@dataclass(frozen=True, slots=True)
class OpenRouterRuntimeIntent:
    model: str = OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT
    selected_source: OpenRouterSource = OPENROUTER_SOURCE_MANAGED
    selection_alias: str | None = None
    routing_mode: str = "latency"
    provider_routing: str = "default"
    managed_credential_kind: OpenRouterManagedCredentialKind = (
        OPENROUTER_MANAGED_CREDENTIAL_STANDARD
    )
    broker_base_url: str | None = None

    def __post_init__(self) -> None:
        model = _normalize_openrouter_model(self.model)
        source = _normalize_openrouter_source(
            self.selected_source,
            default=OPENROUTER_SOURCE_MANAGED,
        )
        selection_profile = (
            get_openrouter_llm_profile(self.selection_alias)
            if isinstance(self.selection_alias, str)
            else None
        )
        selection_alias = _canonical_openrouter_alias(
            model,
            source,
            selection_profile.openrouter_models if selection_profile is not None else (),
        )
        routing_mode = _normalize_allowed(
            self.routing_mode,
            allowed=_OPENROUTER_ROUTING_MODES,
            default="latency",
        )
        provider_routing = _normalize_allowed(
            self.provider_routing,
            allowed=_OPENROUTER_PROVIDER_ROUTINGS,
            default="default",
        )
        managed_credential_kind = _normalize_openrouter_managed_credential_kind(
            self.managed_credential_kind
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "selected_source", source)
        object.__setattr__(self, "selection_alias", selection_alias)
        object.__setattr__(self, "routing_mode", routing_mode)
        object.__setattr__(self, "provider_routing", provider_routing)
        object.__setattr__(self, "managed_credential_kind", managed_credential_kind)
        object.__setattr__(
            self,
            "broker_base_url",
            _normalize_openrouter_broker_base_url(self.broker_base_url),
        )


@dataclass(frozen=True, slots=True)
class DirectProviderRuntimeIntent:
    gemini_37_flash_model: str = GEMINI_MODEL_37_FLASH
    gemini_31_flash_lite_model: str = GEMINI_MODEL_31_FLASH_LITE
    deepseek_v4_flash_model: str = DEEPSEEK_MODEL_V4_FLASH
    qwen_35_plus_model: str = QWEN_MODEL_35_PLUS
    qwen_region: str = QWEN_REGION_BEIJING
    local_llm_backend: str = LOCAL_LLM_BACKEND_OLLAMA
    local_llm_base_url: str = LOCAL_LLM_DEFAULT_BASE_URL
    local_llm_model: str = LOCAL_LLM_DEFAULT_MODEL
    cerebras_model: str = CEREBRAS_MODEL_GEMMA_4_31B
    local_llm_extra_body: Mapping[str, ResolvedOptionValue] = field(default_factory=_empty_options)

    def __post_init__(self) -> None:
        qwen_region = _normalize_allowed(
            self.qwen_region,
            allowed=(QWEN_REGION_BEIJING, QWEN_REGION_SINGAPORE),
            default=QWEN_REGION_BEIJING,
        )
        object.__setattr__(self, "qwen_region", qwen_region)
        object.__setattr__(
            self,
            "local_llm_extra_body",
            _freeze_option_mapping(cast(Mapping[str, object], self.local_llm_extra_body)),
        )


@dataclass(frozen=True, slots=True)
class STTRuntimeIntent:
    channel: str = RUNTIME_CHANNEL_SELF
    provider: str = STT_PROVIDER_LOCAL_CPU_AUTO
    source_language: str = STT_DEFAULT_SOURCE_LANGUAGE
    source_mode: str = "manual"
    input_host_api: str | None = None
    input_device: str | None = None
    output_device: str | None = None
    sample_rate_hz: int = STT_DEFAULT_SAMPLE_RATE_HZ
    channels: int = STT_DEFAULT_CHANNELS
    ring_buffer_ms: int = STT_DEFAULT_RING_BUFFER_MS
    drain_timeout_s: float = STT_DEFAULT_DRAIN_TIMEOUT_S
    vad_speech_threshold: float = STT_DEFAULT_VAD_SPEECH_THRESHOLD
    vad_hangover_ms: int = STT_DEFAULT_VAD_HANGOVER_MS
    vad_pre_roll_ms: int = STT_DEFAULT_VAD_PRE_ROLL_MS
    low_latency_enabled: bool = STT_DEFAULT_LOW_LATENCY_ENABLED
    low_latency_merge_gap_ms: int = STT_DEFAULT_LOW_LATENCY_MERGE_GAP_MS
    low_latency_spec_retry_max: int = STT_DEFAULT_LOW_LATENCY_SPEC_RETRY_MAX
    custom_vocabulary_enabled: bool = False
    custom_terms: Mapping[str, tuple[str, ...]] = field(default_factory=_empty_custom_terms)
    deepgram_model: str = DEEPGRAM_STT_MODEL_NOVA_3
    qwen_asr_model: str = QWEN_ASR_STT_MODEL_REALTIME
    qwen_region: str = QWEN_REGION_BEIJING
    soniox_model: str = SONIOX_STT_MODEL_RT_V5
    soniox_endpoint: str = SONIOX_STT_DEFAULT_ENDPOINT
    soniox_keepalive_interval_s: float = SONIOX_STT_DEFAULT_KEEPALIVE_INTERVAL_S
    soniox_trailing_silence_ms: int = SONIOX_STT_DEFAULT_TRAILING_SILENCE_MS
    soniox_enable_language_identification: bool = False
    soniox_language_hints: tuple[str, ...] | None = None
    soniox_language_hints_strict: bool = False
    custom_stt_mode: str = "offline"
    custom_stt_compatibility: str = "openai_transcription"
    custom_stt_endpoint: str = ""
    custom_stt_model: str = ""
    custom_stt_extra: Mapping[str, ResolvedOptionValue] = field(default_factory=_empty_options)

    def __post_init__(self) -> None:
        channel = _normalize_allowed(
            self.channel,
            allowed=(RUNTIME_CHANNEL_SELF, RUNTIME_CHANNEL_PEER),
            default=RUNTIME_CHANNEL_SELF,
        )
        provider = _normalize_allowed(
            self.provider,
            allowed=STT_PROVIDERS,
            default=STT_PROVIDER_LOCAL_CPU_AUTO,
        )
        source_language = _normalize_string(
            self.source_language,
            default=(
                STT_DEFAULT_PEER_SOURCE_LANGUAGE
                if channel == RUNTIME_CHANNEL_PEER
                else STT_DEFAULT_SOURCE_LANGUAGE
            ),
        )
        source_mode = _normalize_allowed(
            self.source_mode,
            allowed=("manual", "auto"),
            default="manual",
        )
        if channel != RUNTIME_CHANNEL_PEER or provider not in PEER_AUTO_DETECTION_STT_PROVIDERS:
            source_mode = "manual"
        qwen_region = _normalize_allowed(
            self.qwen_region,
            allowed=(QWEN_REGION_BEIJING, QWEN_REGION_SINGAPORE),
            default=QWEN_REGION_BEIJING,
        )
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "source_language", source_language)
        object.__setattr__(self, "source_mode", source_mode)
        object.__setattr__(
            self,
            "sample_rate_hz",
            _normalize_positive_int(self.sample_rate_hz, default=STT_DEFAULT_SAMPLE_RATE_HZ),
        )
        object.__setattr__(
            self,
            "channels",
            _normalize_positive_int(self.channels, default=STT_DEFAULT_CHANNELS),
        )
        object.__setattr__(
            self,
            "ring_buffer_ms",
            _normalize_positive_int(self.ring_buffer_ms, default=STT_DEFAULT_RING_BUFFER_MS),
        )
        object.__setattr__(
            self,
            "drain_timeout_s",
            self.drain_timeout_s if self.drain_timeout_s > 0 else STT_DEFAULT_DRAIN_TIMEOUT_S,
        )
        object.__setattr__(
            self,
            "vad_speech_threshold",
            (
                self.vad_speech_threshold
                if 0.0 <= self.vad_speech_threshold <= 1.0
                else STT_DEFAULT_VAD_SPEECH_THRESHOLD
            ),
        )
        object.__setattr__(self, "vad_hangover_ms", max(0, int(self.vad_hangover_ms)))
        object.__setattr__(self, "vad_pre_roll_ms", max(0, int(self.vad_pre_roll_ms)))
        object.__setattr__(
            self,
            "low_latency_merge_gap_ms",
            max(0, int(self.low_latency_merge_gap_ms)),
        )
        object.__setattr__(
            self,
            "low_latency_spec_retry_max",
            max(0, int(self.low_latency_spec_retry_max)),
        )
        object.__setattr__(self, "qwen_region", qwen_region)
        object.__setattr__(
            self,
            "soniox_language_hints",
            (
                tuple(
                    str(language).strip()
                    for language in self.soniox_language_hints
                    if str(language).strip()
                )
                if self.soniox_language_hints is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "custom_terms",
            _freeze_custom_terms(cast(Mapping[str, object], self.custom_terms)),
        )


def _default_peer_stt_runtime_intent() -> STTRuntimeIntent:
    return STTRuntimeIntent(
        channel=RUNTIME_CHANNEL_PEER,
        source_language=STT_DEFAULT_PEER_SOURCE_LANGUAGE,
        input_host_api=None,
        input_device=None,
        vad_speech_threshold=PEER_STT_DEFAULT_VAD_SPEECH_THRESHOLD,
        vad_hangover_ms=PEER_STT_DEFAULT_VAD_HANGOVER_MS,
        vad_pre_roll_ms=PEER_STT_DEFAULT_VAD_PRE_ROLL_MS,
    )


@dataclass(frozen=True, slots=True)
class OverlayRuntimeIntent:
    enabled: bool = False
    target: str = OVERLAY_TARGET_STEAMVR
    show_translation: bool = True
    show_peer_original: bool = True
    calibration: Mapping[str, ResolvedOptionValue] = field(default_factory=_empty_options)
    desktop_overlay_options: Mapping[str, ResolvedOptionValue] = field(
        default_factory=_empty_options
    )

    def __post_init__(self) -> None:
        target = _normalize_allowed(
            self.target,
            allowed=(OVERLAY_TARGET_STEAMVR, OVERLAY_TARGET_DESKTOP),
            default=OVERLAY_TARGET_STEAMVR,
        )
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "show_translation", bool(self.show_translation))
        object.__setattr__(self, "show_peer_original", bool(self.show_peer_original))
        object.__setattr__(
            self,
            "calibration",
            _freeze_option_mapping(cast(Mapping[str, object], self.calibration)),
        )
        object.__setattr__(
            self,
            "desktop_overlay_options",
            _freeze_option_mapping(cast(Mapping[str, object], self.desktop_overlay_options)),
        )


@dataclass(frozen=True, slots=True)
class RuntimeResolutionInput:
    translation: TranslationRuntimeIntent = field(default_factory=TranslationRuntimeIntent)
    translation_fallback: TranslationFallbackRuntimeIntent = field(
        default_factory=TranslationFallbackRuntimeIntent
    )
    openrouter: OpenRouterRuntimeIntent = field(default_factory=OpenRouterRuntimeIntent)
    direct: DirectProviderRuntimeIntent = field(default_factory=DirectProviderRuntimeIntent)
    self_stt: STTRuntimeIntent = field(default_factory=STTRuntimeIntent)
    peer_stt: STTRuntimeIntent = field(default_factory=_default_peer_stt_runtime_intent)
    overlay: OverlayRuntimeIntent = field(default_factory=OverlayRuntimeIntent)


def normalize_translation_runtime_intent(
    *,
    model: object,
    connection: object,
    concurrency_limit: object = None,
) -> TranslationRuntimeIntent:
    normalized_model = _normalize_translation_model(model)
    return TranslationRuntimeIntent(
        model=normalized_model,
        connection=_normalize_translation_connection(connection, model=normalized_model),
        concurrency_limit=_normalize_positive_int(concurrency_limit, default=5),
    )


def normalize_openrouter_runtime_intent(
    *,
    provider_llm: object = None,
    model: object = None,
    selected_source: object = None,
    selection_alias: object = None,
    fallback_selection_alias: object = None,
    routing_mode: object = None,
    provider_routing: object = None,
    managed_credential_kind: object = None,
    broker_base_url: object = None,
) -> OpenRouterRuntimeIntent:
    _ = fallback_selection_alias
    selection_profile = None
    if isinstance(selection_alias, str):
        normalized_alias = selection_alias.strip()
        if normalized_alias:
            selection_profile = get_openrouter_llm_profile(normalized_alias)

    explicit_source = _explicit_openrouter_source(selected_source)
    source = explicit_source or _default_openrouter_source_for_provider(provider_llm)
    if selection_profile is not None and selection_profile.openrouter_model is not None:
        resolved_model = _normalize_openrouter_model(selection_profile.openrouter_model)
        resolved_source = _normalize_openrouter_source(
            selection_profile.openrouter_source,
            default=source,
        )
        if (
            resolved_source == OPENROUTER_SOURCE_NONE
            and explicit_source is not None
            and explicit_source != OPENROUTER_SOURCE_NONE
        ):
            resolved_source = explicit_source
    else:
        resolved_model = _normalize_openrouter_model(model)
        resolved_source = source

    return OpenRouterRuntimeIntent(
        model=resolved_model,
        selected_source=resolved_source,
        selection_alias=_canonical_openrouter_alias(
            resolved_model,
            resolved_source,
            selection_profile.openrouter_models if selection_profile is not None else (),
        ),
        routing_mode=_normalize_allowed(
            routing_mode,
            allowed=_OPENROUTER_ROUTING_MODES,
            default="latency",
        ),
        provider_routing=_normalize_allowed(
            provider_routing,
            allowed=_OPENROUTER_PROVIDER_ROUTINGS,
            default="default",
        ),
        managed_credential_kind=_normalize_openrouter_managed_credential_kind(
            managed_credential_kind
        ),
        broker_base_url=_normalize_openrouter_broker_base_url(broker_base_url),
    )


def derive_translation_runtime_intent_from_compatibility(
    *,
    provider_llm: object,
    openrouter_model: object = None,
    openrouter_selected_source: object = None,
    openrouter_provider_routing: object = None,
    gemini_model: object = None,
    qwen_model: object = None,
    cerebras_model: object = None,
    concurrency_limit: object = None,
) -> TranslationRuntimeIntent:
    provider = _normalize_allowed(
        provider_llm,
        allowed=LLM_PROVIDERS,
        default=PROVIDER_GEMINI,
    )
    openrouter_model_value = _normalize_openrouter_model(openrouter_model)
    openrouter_source = _normalize_openrouter_source(
        openrouter_selected_source,
        default=_default_openrouter_source_for_provider(provider),
    )
    provider_routing = _normalize_allowed(
        openrouter_provider_routing,
        allowed=_OPENROUTER_PROVIDER_ROUTINGS,
        default="default",
    )
    concurrency = _normalize_positive_int(concurrency_limit, default=5)

    if provider == PROVIDER_OPENROUTER:
        if provider_routing == "gemma4_26b_31b_latency":
            return TranslationRuntimeIntent(
                model=TRANSLATION_MODEL_GEMMA4_26B_31B,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_source,
                    model=TRANSLATION_MODEL_GEMMA4_26B_31B,
                    provider_routing=provider_routing,
                ),
                concurrency_limit=concurrency,
            )
        if (
            provider_routing == "gemma4_31b_latency"
            or openrouter_model_value == OPENROUTER_MODEL_GEMMA_4_31B_IT
        ):
            return TranslationRuntimeIntent(
                model=TRANSLATION_MODEL_GEMMA4_31B,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_source,
                    model=TRANSLATION_MODEL_GEMMA4_31B,
                    provider_routing=provider_routing,
                ),
                concurrency_limit=concurrency,
            )
        if openrouter_model_value == OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT:
            return TranslationRuntimeIntent(
                model=TRANSLATION_MODEL_GEMMA4,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_source,
                    model=TRANSLATION_MODEL_GEMMA4,
                    provider_routing=provider_routing,
                ),
                concurrency_limit=concurrency,
            )
        if openrouter_model_value == OPENROUTER_MODEL_DEEPSEEK_V4_FLASH:
            return TranslationRuntimeIntent(
                model=TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_source,
                    model=TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
                    provider_routing=provider_routing,
                ),
                concurrency_limit=concurrency,
            )
        if openrouter_model_value == OPENROUTER_MODEL_QWEN_35_FLASH_02_23:
            return TranslationRuntimeIntent(
                model=TRANSLATION_MODEL_OPENROUTER_QWEN_35_FLASH,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_source,
                    model=TRANSLATION_MODEL_OPENROUTER_QWEN_35_FLASH,
                    provider_routing=provider_routing,
                ),
                concurrency_limit=concurrency,
            )
        if openrouter_model_value == OPENROUTER_MODEL_GEMINI_37_FLASH:
            return TranslationRuntimeIntent(
                model=TRANSLATION_MODEL_GEMINI_37_FLASH,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_source,
                    model=TRANSLATION_MODEL_GEMINI_37_FLASH,
                    provider_routing=provider_routing,
                ),
                concurrency_limit=concurrency,
            )
        if openrouter_model_value == OPENROUTER_MODEL_GEMINI_31_FLASH_LITE:
            return TranslationRuntimeIntent(
                model=TRANSLATION_MODEL_GEMINI_31_FLASH_LITE,
                connection=_translation_connection_from_openrouter_source(
                    openrouter_source,
                    model=TRANSLATION_MODEL_GEMINI_31_FLASH_LITE,
                    provider_routing=provider_routing,
                ),
                concurrency_limit=concurrency,
            )
        return TranslationRuntimeIntent(
            model=TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
            connection=_default_translation_connection(TRANSLATION_MODEL_DEEPSEEK_V4_FLASH),
            concurrency_limit=concurrency,
        )

    if provider == PROVIDER_MANAGED_GEMMA:
        return TranslationRuntimeIntent(
            model=TRANSLATION_MODEL_MANAGED_GEMMA,
            connection=TRANSLATION_CONNECTION_CPU,
            concurrency_limit=concurrency,
        )

    if provider == PROVIDER_LOCAL_LLM:
        return TranslationRuntimeIntent(
            model=TRANSLATION_MODEL_LOCAL_LLM,
            connection=TRANSLATION_CONNECTION_OLLAMA,
            concurrency_limit=concurrency,
        )

    if provider == PROVIDER_DEEPSEEK:
        return TranslationRuntimeIntent(
            model=TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
            connection=TRANSLATION_CONNECTION_OFFICIAL_BYOK,
            concurrency_limit=concurrency,
        )

    if provider == PROVIDER_CEREBRAS:
        return TranslationRuntimeIntent(
            model=TRANSLATION_MODEL_GEMMA4_31B,
            connection=TRANSLATION_CONNECTION_CEREBRAS,
            concurrency_limit=concurrency,
        )

    if provider == PROVIDER_QWEN:
        if (
            _normalize_allowed(
                qwen_model,
                allowed=(QWEN_MODEL_35_FLASH, QWEN_MODEL_35_PLUS),
                default=QWEN_MODEL_35_PLUS,
            )
            == QWEN_MODEL_35_FLASH
        ):
            return TranslationRuntimeIntent(
                model=TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
                connection=_default_translation_connection(TRANSLATION_MODEL_DEEPSEEK_V4_FLASH),
                concurrency_limit=concurrency,
            )
        return TranslationRuntimeIntent(
            model=TRANSLATION_MODEL_QWEN_35_PLUS,
            connection=TRANSLATION_CONNECTION_OFFICIAL_BYOK,
            concurrency_limit=concurrency,
        )

    if _normalize_gemini_model(gemini_model) == GEMINI_MODEL_37_FLASH:
        return TranslationRuntimeIntent(
            model=TRANSLATION_MODEL_GEMINI_37_FLASH,
            connection=TRANSLATION_CONNECTION_OFFICIAL_BYOK,
            concurrency_limit=concurrency,
        )
    return TranslationRuntimeIntent(
        model=TRANSLATION_MODEL_GEMINI_31_FLASH_LITE,
        connection=TRANSLATION_CONNECTION_OFFICIAL_BYOK,
        concurrency_limit=concurrency,
    )


def _openrouter_source_for_translation(
    connection: TranslationConnectionName,
    openrouter: OpenRouterRuntimeIntent,
) -> OpenRouterSource:
    _ = openrouter
    if connection in (TRANSLATION_CONNECTION_MANAGED, TRANSLATION_CONNECTION_MANAGED_CHINA):
        return OPENROUTER_SOURCE_MANAGED
    return OPENROUTER_SOURCE_BYOK


def _openrouter_managed_credential_kind_for_translation(
    connection: TranslationConnectionName,
    openrouter: OpenRouterRuntimeIntent,
) -> OpenRouterManagedCredentialKind:
    _ = openrouter
    if connection == TRANSLATION_CONNECTION_MANAGED_CHINA:
        return OPENROUTER_MANAGED_CREDENTIAL_QQ
    return OPENROUTER_MANAGED_CREDENTIAL_STANDARD


def _resolved_openrouter_target(
    *,
    model: str,
    models: tuple[str, ...] = (),
    source: OpenRouterSource,
    openrouter: OpenRouterRuntimeIntent,
    provider_routing: str,
    managed_credential_kind: OpenRouterManagedCredentialKind,
) -> ResolvedLLMTarget:
    return ResolvedLLMTarget(
        provider=PROVIDER_OPENROUTER,
        model=model,
        models=models,
        credential=_openrouter_credential(
            source,
            managed_credential_kind=managed_credential_kind,
        ),
        service_endpoint=openrouter.broker_base_url,
        routing_mode=openrouter.routing_mode,
        provider_routing=provider_routing,
    )


def _resolved_direct_provider_target(
    *,
    provider: str,
    model: str,
    credential: ResolvedCredentialRequirement,
    base_url: str | None = None,
    service_endpoint: str | None = None,
    region: str | None = None,
    provider_options: Mapping[str, object] | None = None,
) -> ResolvedLLMTarget:
    return ResolvedLLMTarget(
        provider=provider,
        model=model,
        credential=credential,
        base_url=base_url,
        service_endpoint=service_endpoint,
        region=region,
        provider_options={} if provider_options is None else provider_options,
    )


def resolve_stt_config(intent: STTRuntimeIntent) -> ResolvedSTTConfig:
    provider = intent.provider
    model: str | None = None
    endpoint: str | None = None
    region: str | None = None
    credential = _no_credential()
    provider_options: Mapping[str, object] = {}

    if provider == STT_PROVIDER_DEEPGRAM:
        model = intent.deepgram_model
        credential = _required_credential(
            CREDENTIAL_SOURCE_SECRET_STORE, CREDENTIAL_REF_DEEPGRAM_STT
        )
    elif provider == STT_PROVIDER_QWEN_ASR:
        model = intent.qwen_asr_model
        region = intent.qwen_region
        endpoint = _qwen_asr_endpoint(intent.qwen_region)
        credential = _required_credential(
            CREDENTIAL_SOURCE_SECRET_STORE,
            _qwen_credential_reference(intent.qwen_region),
        )
    elif provider == STT_PROVIDER_SONIOX:
        model = intent.soniox_model
        endpoint = intent.soniox_endpoint
        credential = _required_credential(CREDENTIAL_SOURCE_SECRET_STORE, CREDENTIAL_REF_SONIOX_STT)
        provider_options = {
            "keepalive_interval_s": intent.soniox_keepalive_interval_s,
            "trailing_silence_ms": intent.soniox_trailing_silence_ms,
        }
        if intent.soniox_enable_language_identification:
            provider_options = {
                **provider_options,
                "enable_language_identification": True,
            }
        if intent.soniox_language_hints is not None:
            provider_options = {
                **provider_options,
                "language_hints": intent.soniox_language_hints,
            }
        if intent.soniox_language_hints_strict:
            provider_options = {
                **provider_options,
                "language_hints_strict": True,
            }
    elif provider in STT_CUSTOM_PROVIDERS:
        if provider == STT_PROVIDER_CUSTOM_REALTIME:
            mode = "realtime"
            compatibility = "openai_realtime"
        elif provider == STT_PROVIDER_CUSTOM_OFFLINE:
            mode = "offline"
            compatibility = "openai_transcription"
        else:
            mode = _normalize_custom_stt_mode(intent.custom_stt_mode)
            compatibility = _normalize_custom_stt_compatibility(
                intent.custom_stt_compatibility,
                mode=mode,
            )
        model = str(intent.custom_stt_model or "").strip() or None
        endpoint = str(intent.custom_stt_endpoint or "").strip()
        credential = ResolvedCredentialRequirement(
            source=CREDENTIAL_SOURCE_SECRET_STORE,
            required=False,
            reference=CREDENTIAL_REF_CUSTOM_STT,
        )
        provider_options = {
            "mode": mode,
            "compatibility": compatibility,
            "extra": _freeze_option_mapping(cast(Mapping[str, object], intent.custom_stt_extra)),
        }

    return ResolvedSTTConfig(
        channel=cast(str, intent.channel),
        source_language=intent.source_language,
        source_mode=cast(Literal["manual", "auto"], intent.source_mode),
        provider=provider,
        model=model,
        endpoint=endpoint,
        region=region,
        credential=credential,
        input_host_api=intent.input_host_api,
        input_device=intent.input_device,
        output_device=intent.output_device,
        sample_rate_hz=intent.sample_rate_hz,
        channels=intent.channels,
        ring_buffer_ms=intent.ring_buffer_ms,
        drain_timeout_s=intent.drain_timeout_s,
        vad_speech_threshold=intent.vad_speech_threshold,
        vad_hangover_ms=intent.vad_hangover_ms,
        vad_pre_roll_ms=intent.vad_pre_roll_ms,
        low_latency_enabled=intent.low_latency_enabled,
        low_latency_merge_gap_ms=intent.low_latency_merge_gap_ms,
        low_latency_spec_retry_max=intent.low_latency_spec_retry_max,
        custom_vocabulary_enabled=intent.custom_vocabulary_enabled,
        custom_terms=intent.custom_terms if intent.custom_vocabulary_enabled else {},
        provider_options=provider_options,
    )


def resolve_overlay_config(intent: OverlayRuntimeIntent) -> ResolvedOverlayConfig:
    return ResolvedOverlayConfig(
        enabled=intent.enabled,
        target=cast(str, intent.target),
        show_translation=intent.show_translation,
        show_peer_original=intent.show_peer_original,
        calibration=intent.calibration,
        desktop_overlay_options=intent.desktop_overlay_options,
    )


def _resolve_translation_target(
    translation: TranslationRuntimeIntent,
    *,
    openrouter: OpenRouterRuntimeIntent,
    direct: DirectProviderRuntimeIntent,
    is_fallback: bool = False,
) -> ResolvedLLMTarget:
    if translation.model == TRANSLATION_MODEL_CUSTOM_HTTP:
        return _resolved_direct_provider_target(
            provider=PROVIDER_CUSTOM_HTTP,
            model=TRANSLATION_MODEL_CUSTOM_HTTP,
            credential=_no_credential(),
        )

    if translation.model == TRANSLATION_MODEL_MANAGED_GEMMA:
        return _resolved_direct_provider_target(
            provider=PROVIDER_MANAGED_GEMMA,
            model=MANAGED_GEMMA_MODEL,
            credential=_no_credential(),
            provider_options={"backend": translation.connection},
        )

    if translation.model == TRANSLATION_MODEL_MANAGED_GEMMA_12B:
        return _resolved_direct_provider_target(
            provider=PROVIDER_MANAGED_GEMMA,
            model=MANAGED_GEMMA_12B_MODEL,
            credential=_no_credential(),
            provider_options={"backend": TRANSLATION_CONNECTION_GPU},
        )

    if translation.model == TRANSLATION_MODEL_GEMMA4_26B_31B:
        return _resolved_openrouter_target(
            model=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
            models=(
                OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
                OPENROUTER_MODEL_GEMMA_4_31B_IT,
            ),
            source=_openrouter_source_for_translation(translation.connection, openrouter),
            openrouter=openrouter,
            provider_routing="gemma4_26b_31b_latency",
            managed_credential_kind=_openrouter_managed_credential_kind_for_translation(
                translation.connection,
                openrouter,
            ),
        )

    if translation.model == TRANSLATION_MODEL_GEMMA4_31B:
        if translation.connection == TRANSLATION_CONNECTION_CEREBRAS:
            return _resolved_direct_provider_target(
                provider=PROVIDER_CEREBRAS,
                model=direct.cerebras_model,
                credential=_required_credential(
                    CREDENTIAL_SOURCE_SECRET_STORE,
                    CREDENTIAL_REF_CEREBRAS_BYOK,
                ),
            )
        return _resolved_openrouter_target(
            model=OPENROUTER_MODEL_GEMMA_4_31B_IT,
            source=_openrouter_source_for_translation(translation.connection, openrouter),
            openrouter=openrouter,
            provider_routing="gemma4_31b_latency",
            managed_credential_kind=_openrouter_managed_credential_kind_for_translation(
                translation.connection,
                openrouter,
            ),
        )

    if translation.model == TRANSLATION_MODEL_GEMMA4:
        return _resolved_openrouter_target(
            model=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
            source=_openrouter_source_for_translation(translation.connection, openrouter),
            openrouter=openrouter,
            provider_routing="gemma4_26b_latency",
            managed_credential_kind=_openrouter_managed_credential_kind_for_translation(
                translation.connection,
                openrouter,
            ),
        )

    if translation.model == TRANSLATION_MODEL_DEEPSEEK_V4_FLASH:
        if translation.connection == TRANSLATION_CONNECTION_OFFICIAL_BYOK:
            return _resolved_direct_provider_target(
                provider=PROVIDER_DEEPSEEK,
                model=direct.deepseek_v4_flash_model,
                credential=_required_credential(
                    CREDENTIAL_SOURCE_SECRET_STORE,
                    CREDENTIAL_REF_DEEPSEEK_BYOK,
                ),
            )
        provider_routing = (
            "deepseek_only"
            if translation.connection == TRANSLATION_CONNECTION_MANAGED_CHINA
            else (
                "deepseek_v4_flash_latency"
                if is_fallback
                else (
                    openrouter.provider_routing
                    if translation.connection == TRANSLATION_CONNECTION_OPENROUTER
                    else "default"
                )
            )
        )
        return _resolved_openrouter_target(
            model=OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
            source=_openrouter_source_for_translation(translation.connection, openrouter),
            openrouter=openrouter,
            provider_routing=provider_routing,
            managed_credential_kind=_openrouter_managed_credential_kind_for_translation(
                translation.connection,
                openrouter,
            ),
        )

    if translation.model == TRANSLATION_MODEL_GEMINI_37_FLASH:
        if translation.connection == TRANSLATION_CONNECTION_OPENROUTER:
            return _resolved_openrouter_target(
                model=OPENROUTER_MODEL_GEMINI_37_FLASH,
                source=_openrouter_source_for_translation(translation.connection, openrouter),
                openrouter=openrouter,
                provider_routing="google_gemini_latency",
                managed_credential_kind=_openrouter_managed_credential_kind_for_translation(
                    translation.connection,
                    openrouter,
                ),
            )
        return _resolved_direct_provider_target(
            provider=PROVIDER_GEMINI,
            model=direct.gemini_37_flash_model,
            credential=_required_credential(
                CREDENTIAL_SOURCE_SECRET_STORE,
                CREDENTIAL_REF_GEMINI_BYOK,
            ),
        )

    if translation.model == TRANSLATION_MODEL_GEMINI_31_FLASH_LITE:
        if translation.connection == TRANSLATION_CONNECTION_OPENROUTER:
            return _resolved_openrouter_target(
                model=OPENROUTER_MODEL_GEMINI_31_FLASH_LITE,
                source=_openrouter_source_for_translation(translation.connection, openrouter),
                openrouter=openrouter,
                provider_routing="google_gemini_latency",
                managed_credential_kind=_openrouter_managed_credential_kind_for_translation(
                    translation.connection,
                    openrouter,
                ),
            )
        return _resolved_direct_provider_target(
            provider=PROVIDER_GEMINI,
            model=direct.gemini_31_flash_lite_model,
            credential=_required_credential(
                CREDENTIAL_SOURCE_SECRET_STORE,
                CREDENTIAL_REF_GEMINI_BYOK,
            ),
        )

    if translation.model == TRANSLATION_MODEL_QWEN_35_PLUS:
        return _resolved_direct_provider_target(
            provider=PROVIDER_QWEN,
            model=direct.qwen_35_plus_model,
            credential=_required_credential(
                CREDENTIAL_SOURCE_SECRET_STORE,
                _qwen_credential_reference(direct.qwen_region),
            ),
            service_endpoint=_qwen_service_endpoint(direct.qwen_region),
            region=direct.qwen_region,
        )

    if translation.model == TRANSLATION_MODEL_OPENROUTER_QWEN_35_FLASH:
        return _resolved_openrouter_target(
            model=OPENROUTER_MODEL_QWEN_35_FLASH_02_23,
            source=_openrouter_source_for_translation(translation.connection, openrouter),
            openrouter=openrouter,
            provider_routing=openrouter.provider_routing,
            managed_credential_kind=_openrouter_managed_credential_kind_for_translation(
                translation.connection,
                openrouter,
            ),
        )

    return _resolved_direct_provider_target(
        provider=PROVIDER_LOCAL_LLM,
        model=direct.local_llm_model,
        credential=_no_credential(),
        base_url=direct.local_llm_base_url,
        provider_options={
            "backend": direct.local_llm_backend,
            "extra_body": direct.local_llm_extra_body,
        },
    )


def _fallback_plan_for_target(
    target: ResolvedLLMTarget,
) -> ResolvedLLMFallbackPlan:
    return ResolvedLLMFallbackPlan(
        target=target,
        timeout_ms=_FIRST_HEDGE_DELAY_MS,
        force_managed_wrapper=(
            target.provider == PROVIDER_OPENROUTER
            and target.credential.source == CREDENTIAL_SOURCE_MANAGED
        ),
        start_on_primary_error=True,
    )


def _emergency_plan_for_primary(
    translation: TranslationRuntimeIntent,
    *,
    primary: ResolvedLLMTarget,
    openrouter: OpenRouterRuntimeIntent,
    direct: DirectProviderRuntimeIntent,
) -> ResolvedLLMAttemptPlan | None:
    if primary.provider != PROVIDER_OPENROUTER:
        return None
    emergency_translation = TranslationRuntimeIntent(
        model=TRANSLATION_MODEL_GEMMA4_31B,
        connection=translation.connection,
        concurrency_limit=translation.concurrency_limit,
    )
    emergency_target = _resolve_translation_target(
        emergency_translation,
        openrouter=openrouter,
        direct=direct,
    )
    return ResolvedLLMAttemptPlan(
        target=ResolvedLLMTarget(
            provider=emergency_target.provider,
            model=emergency_target.model,
            models=emergency_target.models,
            credential=emergency_target.credential,
            base_url=emergency_target.base_url,
            service_endpoint=emergency_target.service_endpoint,
            region=emergency_target.region,
            routing_mode=emergency_target.routing_mode,
            provider_routing="gemma4_31b_cerebras_only",
            provider_options=emergency_target.provider_options,
        ),
        start_after_ms=_EMERGENCY_HEDGE_DELAY_MS,
        start_on_primary_error=False,
    )


def resolve_llm_config(runtime_input: RuntimeResolutionInput) -> ResolvedLLMConfig:
    translation = runtime_input.translation
    openrouter = runtime_input.openrouter
    direct = runtime_input.direct
    primary = _resolve_translation_target(translation, openrouter=openrouter, direct=direct)
    fallback_plan: ResolvedLLMFallbackPlan | None = None
    attempts: list[ResolvedLLMAttemptPlan] = [
        ResolvedLLMAttemptPlan(target=primary),
    ]

    if runtime_input.translation_fallback.enabled and translation.model not in (
        TRANSLATION_MODEL_CUSTOM_HTTP,
        TRANSLATION_MODEL_MANAGED_GEMMA,
        TRANSLATION_MODEL_MANAGED_GEMMA_12B,
        TRANSLATION_MODEL_LOCAL_LLM,
    ):
        fallback_translation = TranslationRuntimeIntent(
            model=runtime_input.translation_fallback.model,
            connection=runtime_input.translation_fallback.connection,
            concurrency_limit=translation.concurrency_limit,
        )
        fallback_target = _resolve_translation_target(
            fallback_translation,
            openrouter=openrouter,
            direct=direct,
            is_fallback=True,
        )
        fallback_plan = _fallback_plan_for_target(fallback_target)
        attempts.append(
            ResolvedLLMAttemptPlan(
                target=fallback_target,
                start_after_ms=fallback_plan.start_after_ms,
                start_on_primary_error=fallback_plan.start_on_primary_error,
            )
        )
        emergency_plan = _emergency_plan_for_primary(
            translation,
            primary=primary,
            openrouter=openrouter,
            direct=direct,
        )
        if emergency_plan is not None:
            attempts.append(emergency_plan)

    return ResolvedLLMConfig(
        primary=primary,
        fallback=fallback_plan,
        attempts=tuple(attempts),
        loser_grace_ms=_LOSER_GRACE_MS,
        concurrency_limit=translation.concurrency_limit,
    )


__all__ = [
    "CREDENTIAL_REF_DEEPSEEK_BYOK",
    "CREDENTIAL_REF_CEREBRAS_BYOK",
    "CREDENTIAL_REF_GEMINI_BYOK",
    "CREDENTIAL_REF_OPENROUTER_BYOK",
    "CREDENTIAL_REF_OPENROUTER_MANAGED",
    "CREDENTIAL_REF_OPENROUTER_MANAGED_QQ",
    "CREDENTIAL_REF_CUSTOM_STT",
    "CREDENTIAL_REF_DEEPGRAM_STT",
    "CREDENTIAL_REF_QWEN_BEIJING",
    "CREDENTIAL_REF_QWEN_SINGAPORE",
    "CREDENTIAL_REF_SONIOX_STT",
    "DEEPSEEK_MODEL_V4_FLASH",
    "DEEPGRAM_STT_MODEL_NOVA_3",
    "DirectProviderRuntimeIntent",
    "CEREBRAS_MODEL_GEMMA_4_31B",
    "GEMINI_MODEL_37_FLASH",
    "GEMINI_MODEL_31_FLASH_LITE",
    "LOCAL_LLM_BACKEND_OLLAMA",
    "LOCAL_LLM_DEFAULT_BASE_URL",
    "LOCAL_LLM_DEFAULT_MODEL",
    "MANAGED_GEMMA_12B_MODEL",
    "MANAGED_GEMMA_MODEL",
    "LLM_PROVIDERS",
    "OPENROUTER_SOURCE_BYOK",
    "OPENROUTER_MANAGED_CREDENTIAL_KINDS",
    "OPENROUTER_MANAGED_CREDENTIAL_QQ",
    "OPENROUTER_MANAGED_CREDENTIAL_STANDARD",
    "OPENROUTER_SOURCE_MANAGED",
    "OPENROUTER_SOURCE_NONE",
    "OPENROUTER_SOURCES",
    "OverlayRuntimeIntent",
    "OpenRouterRuntimeIntent",
    "OpenRouterSource",
    "PROVIDER_DEEPSEEK",
    "PROVIDER_CEREBRAS",
    "PROVIDER_CUSTOM_HTTP",
    "PROVIDER_GEMINI",
    "PROVIDER_MANAGED_GEMMA",
    "PROVIDER_LOCAL_LLM",
    "PROVIDER_OPENROUTER",
    "PROVIDER_QWEN",
    "QWEN_MODEL_35_PLUS",
    "QWEN_MODEL_35_FLASH",
    "QWEN_ASR_STT_MODEL_REALTIME",
    "QWEN_REGION_BEIJING",
    "QWEN_REGION_SINGAPORE",
    "RuntimeResolutionInput",
    "SONIOX_STT_DEFAULT_ENDPOINT",
    "SONIOX_STT_DEFAULT_KEEPALIVE_INTERVAL_S",
    "SONIOX_STT_DEFAULT_TRAILING_SILENCE_MS",
    "SONIOX_STT_MODEL_RT_V5",
    "STT_DEFAULT_CHANNELS",
    "STT_DEFAULT_DRAIN_TIMEOUT_S",
    "STT_DEFAULT_LOW_LATENCY_ENABLED",
    "STT_DEFAULT_LOW_LATENCY_MERGE_GAP_MS",
    "STT_DEFAULT_LOW_LATENCY_SPEC_RETRY_MAX",
    "STT_DEFAULT_PEER_SOURCE_LANGUAGE",
    "STT_DEFAULT_RING_BUFFER_MS",
    "STT_DEFAULT_SAMPLE_RATE_HZ",
    "STT_DEFAULT_SOURCE_LANGUAGE",
    "STT_DEFAULT_VAD_HANGOVER_MS",
    "STT_DEFAULT_VAD_PRE_ROLL_MS",
    "STT_DEFAULT_VAD_SPEECH_THRESHOLD",
    "STT_PROVIDER_DEEPGRAM",
    "STT_PROVIDER_LOCAL_CPU_AUTO",
    "STT_PROVIDER_LOCAL_PARAKEET_JAPANESE",
    "STT_PROVIDER_LOCAL_PARAKEET_V3",
    "STT_PROVIDER_LOCAL_QWEN",
    "STT_PROVIDER_LOCAL_QWEN_GPU",
    "PEER_AUTO_DETECTION_STT_PROVIDERS",
    "STT_PROVIDER_QWEN_ASR",
    "STT_PROVIDER_SONIOX",
    "STT_PROVIDER_CUSTOM",
    "STT_PROVIDERS",
    "STTRuntimeIntent",
    "TRANSLATION_CONNECTION_CEREBRAS",
    "TRANSLATION_CONNECTION_MANAGED",
    "TRANSLATION_CONNECTION_MANAGED_CHINA",
    "TRANSLATION_CONNECTION_CPU",
    "TRANSLATION_CONNECTION_GPU",
    "TRANSLATION_CONNECTION_OFFICIAL_BYOK",
    "TRANSLATION_CONNECTION_OLLAMA",
    "TRANSLATION_CONNECTION_CUSTOM_HTTP",
    "TRANSLATION_CONNECTION_OPENROUTER",
    "TRANSLATION_CONNECTIONS",
    "TRANSLATION_CONNECTIONS_BY_MODEL",
    "TRANSLATION_MODEL_DEEPSEEK_V4_FLASH",
    "TRANSLATION_MODEL_GEMINI_37_FLASH",
    "TRANSLATION_MODEL_GEMINI_31_FLASH_LITE",
    "TRANSLATION_MODEL_GEMMA4",
    "TRANSLATION_MODEL_GEMMA4_26B_31B",
    "TRANSLATION_MODEL_GEMMA4_31B",
    "TRANSLATION_MODEL_CUSTOM_HTTP",
    "TRANSLATION_MODEL_LOCAL_LLM",
    "TRANSLATION_MODEL_MANAGED_GEMMA",
    "TRANSLATION_MODEL_MANAGED_GEMMA_12B",
    "TRANSLATION_MODEL_OPENROUTER_QWEN_35_FLASH",
    "TRANSLATION_MODEL_QWEN_35_PLUS",
    "TRANSLATION_MODELS",
    "TranslationConnectionName",
    "TranslationFallbackRuntimeIntent",
    "TranslationModelName",
    "TranslationRuntimeIntent",
    "derive_translation_runtime_intent_from_compatibility",
    "normalize_openrouter_runtime_intent",
    "normalize_translation_runtime_intent",
    "resolve_overlay_config",
    "resolve_llm_config",
    "resolve_stt_config",
]
