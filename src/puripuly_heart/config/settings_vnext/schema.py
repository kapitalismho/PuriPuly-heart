from __future__ import annotations

import json
import ntpath
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Final, Literal

from puripuly_heart.config.audio_host_api import WINDOWS_WASAPI_COMPATIBILITY_HOST_API
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.translation_policy import (
    FIXED_TRANSLATION_POLICY,
    TranslationRuntimePolicy,
)

VNEXT_SETTINGS_SCHEMA_VERSION: Final = 36
OSC_DEFAULT_HOST: Final = "127.0.0.1"
OSC_DEFAULT_SEND_PORT: Final = 9000
OSC_DEFAULT_RECEIVE_PORT: Final = 9001
OSC_CONNECTION_MODES: Final = ("automatic", "manual", "off")

DEFAULT_OPENROUTER_BROKER_BASE_URL: Final = "https://puripuly-heart-broker.kapitalismho.workers.dev"
DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS: Final = "openrouter_gemma4_26b_31b"
DEFAULT_CUSTOM_VOCAB_TERMS: Final[Mapping[str, tuple[str, ...]]] = {}
MANAGED_AUTH_CLAIM_SOURCE_DISCORD: Final = "discord"
MANAGED_AUTH_CLAIM_SOURCE_QQ: Final = "qq"
MANAGED_AUTH_CLAIM_SOURCES: Final = (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    MANAGED_AUTH_CLAIM_SOURCE_QQ,
)
MANAGED_KEY_DELIVERY_ACK_SOURCES: Final = ("discord", "qq")

RUNTIME_ONLY_LEGACY_SETTINGS_PATHS: Final = frozenset(
    {"ui.overlay_enabled", "ui.peer_translation_enabled"}
)

_LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS: Final = frozenset(
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
_LOCAL_LLM_SECRET_BEARING_EXTRA_BODY_KEYS: Final = frozenset(
    {
        "api_key",
        "authorization",
        "headers",
        "token",
        "secret",
        "password",
    }
)
_PROVIDER_VERIFICATION_STATUSES: Final = frozenset({"unknown", "verified", "failed", "skipped"})
TELEMETRY_CONSENT_VALUES: Final = frozenset({"unknown", "allow", "decline"})
CANONICAL_TRANSLATION_FALLBACK_ALIASES: Final = frozenset(
    {
        "none",
        "deepseek_v4_flash_official",
        "openrouter_deepseek_v4_flash",
        "openrouter_gemma4_26b_a4b",
        DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS,
        "openrouter_gemma4_31b",
        "managed_gemma4_26b_31b",
        "managed_gemma4_31b",
        "cerebras_gemma4_31b",
    }
)
COMPAT_TRANSLATION_FALLBACK_ALIASES: Final = frozenset({"deepseek_v4_flash_china"})
_FALLBACK_ALIAS_FIELDS: Final = {
    "none": (False, "deepseek_v4_flash", "official_byok"),
    "deepseek_v4_flash_official": (True, "deepseek_v4_flash", "official_byok"),
    "openrouter_deepseek_v4_flash": (True, "deepseek_v4_flash", "openrouter"),
    "openrouter_gemma4_26b_a4b": (True, "gemma4", "openrouter"),
    DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS: (True, "gemma4_26b_31b", "openrouter"),
    "openrouter_gemma4_31b": (True, "gemma4_31b", "openrouter"),
    "managed_gemma4_26b_31b": (True, "gemma4_26b_31b", "managed"),
    "managed_gemma4_31b": (True, "gemma4_31b", "managed"),
    "cerebras_gemma4_31b": (True, "gemma4_31b", "cerebras"),
    "deepseek_v4_flash_china": (True, "deepseek_v4_flash", "managed_china"),
}
_FALLBACK_FIELDS_ALIAS: Final = {fields: alias for alias, fields in _FALLBACK_ALIAS_FIELDS.items()}
_PROVIDER_VERIFICATION_SECRET_BEARING_KEY_FRAGMENTS: Final = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "body",
    "client_secret",
    "credential_value",
    "password",
    "payload",
    "private_key",
    "provider_payload",
    "raw",
    "raw_payload",
    "refresh_token",
    "request_body",
    "response_body",
    "secret",
    "token",
)


def _default_translation_connection_history() -> dict[str, str]:
    return {"gemma4_26b_31b": "managed"}


def _default_local_llm_extra_body() -> dict[str, object]:
    return {"reasoning_effort": "none"}


def _default_custom_terms() -> dict[str, list[str]]:
    return {language: list(terms) for language, terms in DEFAULT_CUSTOM_VOCAB_TERMS.items()}


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


def _normalize_extra_body_key(key: str) -> str:
    normalized = key.strip()
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", normalized)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    return re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()


def _is_secret_bearing_extra_body_key(key: str) -> bool:
    return key in _LOCAL_LLM_SECRET_BEARING_EXTRA_BODY_KEYS


def _copy_local_llm_extra_body_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return _copy_local_llm_extra_body(value)
    if isinstance(value, list | tuple):
        return [_copy_local_llm_extra_body_value(item) for item in value]
    raise TypeError("local LLM extra_body values must be JSON-like scalars, mappings, or lists")


def _copy_local_llm_extra_body(values: Mapping[object, object]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for raw_key, raw_value in values.items():
        if not isinstance(raw_key, str):
            continue
        key = _normalize_extra_body_key(raw_key)
        if key in _LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS:
            return _default_local_llm_extra_body()
        if _is_secret_bearing_extra_body_key(key):
            return _default_local_llm_extra_body()
        copied[raw_key] = _copy_local_llm_extra_body_value(raw_value)
    try:
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError):
        return _default_local_llm_extra_body()
    return copied


def _is_secret_bearing_provider_verification_metadata_key(key: str) -> bool:
    return any(fragment in key for fragment in _PROVIDER_VERIFICATION_SECRET_BEARING_KEY_FRAGMENTS)


def is_safe_compatibility_extension_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    return not _is_secret_bearing_provider_verification_metadata_key(_normalize_extra_body_key(key))


def _copy_provider_verification_metadata_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError("provider verification metadata values must be JSON-like scalars")


def _copy_provider_verification_metadata(values: Mapping[object, object]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for raw_key, raw_value in values.items():
        if not isinstance(raw_key, str):
            raise ValueError("provider verification metadata keys must be strings")
        key = _normalize_extra_body_key(raw_key)
        if _is_secret_bearing_provider_verification_metadata_key(key):
            raise ValueError(
                f"secret-bearing provider verification metadata key is not allowed: {raw_key}"
            )
        copied[raw_key] = _copy_provider_verification_metadata_value(raw_value)
    try:
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("provider verification metadata must be JSON serializable") from exc
    return copied


def _copy_compatibility_extension_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return _copy_compatibility_extensions(value)
    if isinstance(value, list | tuple):
        return [_copy_compatibility_extension_value(item) for item in value]
    raise TypeError("compatibility extension values must be JSON-compatible")


def _copy_compatibility_extensions(values: Mapping[object, object]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for raw_key, raw_value in values.items():
        if not isinstance(raw_key, str):
            raise ValueError("compatibility extension keys must be strings")
        key = _normalize_extra_body_key(raw_key)
        if _is_secret_bearing_provider_verification_metadata_key(key):
            raise ValueError(f"secret-bearing compatibility key is not allowed: {raw_key}")
        copied[raw_key] = _copy_compatibility_extension_value(raw_value)
    try:
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("compatibility extensions must be finite JSON values") from exc
    return copied


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"provider verification {field_name} must be a string or null")
    return value


def _required_provider_verification_string(value: object, *, field_name: str) -> str:
    raw = _optional_string(value, field_name=field_name)
    if raw is None:
        raise ValueError(f"provider verification evidence requires non-empty {field_name}")
    normalized = raw.strip()
    if not normalized:
        raise ValueError(f"provider verification evidence requires non-empty {field_name}")
    return normalized


def _optional_provider_verification_string(value: object, *, field_name: str) -> str | None:
    raw = _optional_string(value, field_name=field_name)
    if raw is None:
        return None
    normalized = raw.strip()
    return normalized or None


def _normalize_translation_fallback_alias(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized in CANONICAL_TRANSLATION_FALLBACK_ALIASES | COMPAT_TRANSLATION_FALLBACK_ALIASES:
        return normalized
    return None


def _infer_translation_fallback_alias(
    *,
    enabled: object,
    model: object,
    connection: object,
) -> str:
    fields_key = (bool(enabled), str(model), str(connection))
    return _FALLBACK_FIELDS_ALIAS.get(fields_key, "none")


def _normalize_telemetry_sent_dates(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates: tuple[object, ...] = (value,)
    elif isinstance(value, list | tuple | set | frozenset):
        candidates = tuple(value)
    else:
        candidates = ()
    normalized: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        raw = candidate.strip()
        try:
            parsed = date.fromisoformat(raw)
        except ValueError:
            continue
        normalized.add(parsed.isoformat())
    return tuple(sorted(normalized))


def _normalize_telemetry_identifier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_optional_state_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def new_anonymous_telemetry_identifier() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class QwenTranslationIntent:
    region: str = "beijing"
    llm_model: str = "qwen3.5-plus"


@dataclass(frozen=True, slots=True)
class GeminiTranslationIntent:
    llm_model: str = "gemini-3.1-flash-lite"


@dataclass(frozen=True, slots=True)
class DeepSeekTranslationIntent:
    llm_model: str = "deepseek-v4-flash"


@dataclass(frozen=True, slots=True)
class CerebrasTranslationIntent:
    llm_model: str = "gemma-4-31b"


@dataclass(frozen=True, slots=True)
class TranslationFallbackIntent:
    enabled: bool = False
    model: str = "deepseek_v4_flash"
    connection: str = "official_byok"
    selection_alias: str = "none"

    def __post_init__(self) -> None:
        alias = _normalize_translation_fallback_alias(self.selection_alias)
        if alias is None:
            alias = "none"
        enabled, model, connection = _FALLBACK_ALIAS_FIELDS[alias]
        object.__setattr__(self, "selection_alias", alias)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "connection", connection)


def _default_translation_fallback_intent() -> TranslationFallbackIntent:
    return TranslationFallbackIntent(selection_alias=DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS)


@dataclass(frozen=True, slots=True)
class TranslationIntent:
    model: str = "gemma4_26b_31b"
    connection: str = "managed"
    http_extension_id: str | None = None
    previous_llm_model: str | None = None
    connection_history: dict[str, str] = field(
        default_factory=_default_translation_connection_history
    )
    concurrency_limit: int = 5
    fallback: TranslationFallbackIntent = field(
        default_factory=_default_translation_fallback_intent
    )
    openrouter_broker_base_url: str = DEFAULT_OPENROUTER_BROKER_BASE_URL
    openrouter_routing_mode: str = "latency"
    openrouter_model: str = "google/gemma-4-26b-a4b-it"
    openrouter_selected_source: str = "managed"
    openrouter_selection_alias: str | None = "gemma4_26b_31b_managed"
    openrouter_provider_routing: str = "gemma4_26b_31b_latency"
    gemini: GeminiTranslationIntent = field(default_factory=GeminiTranslationIntent)
    deepseek: DeepSeekTranslationIntent = field(default_factory=DeepSeekTranslationIntent)
    qwen: QwenTranslationIntent = field(default_factory=QwenTranslationIntent)
    cerebras: CerebrasTranslationIntent = field(default_factory=CerebrasTranslationIntent)
    gpu_device_id: str = "auto"

    def __post_init__(self) -> None:
        device_id = self.gpu_device_id if isinstance(self.gpu_device_id, str) else "auto"
        object.__setattr__(self, "gpu_device_id", device_id.strip() or "auto")


@dataclass(frozen=True, slots=True)
class LocalLLMIntent:
    backend: str = "ollama"
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "llama3.1:8b"
    extra_body: dict[str, object] = field(default_factory=_default_local_llm_extra_body)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_body", _copy_local_llm_extra_body(self.extra_body))


@dataclass(frozen=True, slots=True)
class DeepgramSTTIntent:
    model: str = "nova-3"


@dataclass(frozen=True, slots=True)
class QwenASRSTTIntent:
    model: str = "qwen3-asr-flash-realtime"


@dataclass(frozen=True, slots=True)
class SonioxSTTIntent:
    model: str = "stt-rt-v5"
    endpoint: str = "wss://stt-rt.soniox.com/transcribe-websocket"
    keepalive_interval_s: float = 10.0
    trailing_silence_ms: int = 100


@dataclass(frozen=True, slots=True)
class CustomSTTIntent:
    mode: str = "offline"
    compatibility: str = "openai_transcription"
    endpoint: str = ""
    model: str = ""
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class STTIntent:
    provider: str = "local_cpu_auto"
    drain_timeout_s: float = 2.0
    vad_speech_threshold: float = 0.4
    low_latency_mode: bool = True
    low_latency_vad_hangover_ms: int = 500
    low_latency_merge_gap_ms: int = 600
    low_latency_spec_retry_max: int = 10
    custom_vocabulary_enabled: bool = True
    custom_terms: dict[str, list[str]] = field(default_factory=_default_custom_terms)
    gpu_device_id: str = "auto"
    deepgram: DeepgramSTTIntent = field(default_factory=DeepgramSTTIntent)
    qwen_asr: QwenASRSTTIntent = field(default_factory=QwenASRSTTIntent)
    soniox: SonioxSTTIntent = field(default_factory=SonioxSTTIntent)
    custom: CustomSTTIntent = field(default_factory=CustomSTTIntent)

    def __post_init__(self) -> None:
        device_id = self.gpu_device_id if isinstance(self.gpu_device_id, str) else "auto"
        object.__setattr__(self, "gpu_device_id", device_id.strip() or "auto")


@dataclass(frozen=True, slots=True)
class PeerSTTIntent:
    provider: str = "local_cpu_auto"


@dataclass(frozen=True, slots=True)
class LanguageIntent:
    source_language: str = "ko"
    target_language: str = "en"
    peer_source_language: str = "en"
    peer_target_language: str = "ko"
    peer_source_mode: str = "manual"
    peer_expected_languages: list[str] = field(default_factory=list)
    recent_source_languages: list[str] = field(default_factory=lambda: ["en", "zh-CN", "ja"])
    recent_target_languages: list[str] = field(default_factory=lambda: ["en", "zh-CN", "ja"])

    def __post_init__(self) -> None:
        mode = self.peer_source_mode if isinstance(self.peer_source_mode, str) else "manual"
        mode = mode.strip()
        object.__setattr__(
            self,
            "peer_source_mode",
            mode if mode in {"manual", "auto"} else "manual",
        )
        languages = self.peer_expected_languages
        if not isinstance(languages, list):
            languages = []
        object.__setattr__(
            self,
            "peer_expected_languages",
            list(
                dict.fromkeys(
                    language.strip()
                    for language in languages
                    if isinstance(language, str) and language.strip()
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class AudioIntent:
    ring_buffer_ms: int = 500
    input_host_api: str = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    input_device: str = ""


@dataclass(frozen=True, slots=True)
class ProcessCaptureTargetIntent:
    kind: Literal["generic_executable", "vrchat", "discord"]
    executable_identity: str | None = None
    discord_channel: str | None = None
    executable_basename: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "generic_executable":
            identity = _normalize_executable_identity(self.executable_identity)
            if self.discord_channel is not None or self.executable_basename is not None:
                raise ValueError("generic executable targets cannot include Discord identity")
            if ntpath.basename(identity).casefold() in _DISCORD_BASENAME_BY_CHANNEL_CASEFOLDED:
                raise ValueError("Discord targets must use a Discord channel identity")
            object.__setattr__(self, "executable_identity", identity)
            return
        if self.kind == "vrchat":
            identity = _normalize_executable_identity(self.executable_identity)
            if ntpath.basename(identity).casefold() != "vrchat.exe":
                raise ValueError("VRChat targets must identify VRChat.exe")
            if self.discord_channel is not None or self.executable_basename is not None:
                raise ValueError("VRChat targets cannot include Discord identity")
            object.__setattr__(self, "executable_identity", identity)
            return
        if self.kind == "discord":
            channel = _normalize_discord_channel(self.discord_channel)
            basename = _DISCORD_BASENAME_BY_CHANNEL[channel]
            if self.executable_identity is not None:
                raise ValueError("Discord targets cannot persist an installation path")
            if self.executable_basename not in (None, basename):
                raise ValueError("Discord target basename does not match its channel")
            object.__setattr__(self, "discord_channel", channel)
            object.__setattr__(self, "executable_basename", basename)
            return
        raise ValueError(f"unsupported process capture target kind: {self.kind}")

    @classmethod
    def generic_executable(cls, executable_identity: str) -> ProcessCaptureTargetIntent:
        return cls(kind="generic_executable", executable_identity=executable_identity)

    @classmethod
    def vrchat(cls, executable_identity: str) -> ProcessCaptureTargetIntent:
        return cls(kind="vrchat", executable_identity=executable_identity)

    @classmethod
    def discord(cls, channel: str) -> ProcessCaptureTargetIntent:
        return cls(kind="discord", discord_channel=channel)


_DISCORD_BASENAME_BY_CHANNEL: Final[dict[str, str]] = {
    "stable": "Discord.exe",
    "ptb": "DiscordPTB.exe",
    "canary": "DiscordCanary.exe",
}
_DISCORD_BASENAME_BY_CHANNEL_CASEFOLDED: Final[frozenset[str]] = frozenset(
    basename.casefold() for basename in _DISCORD_BASENAME_BY_CHANNEL.values()
)


def _normalize_executable_identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("process executable identity must be non-empty")
    normalized = ntpath.normcase(ntpath.normpath(value.strip().replace("/", "\\")))
    if (
        not normalized
        or normalized in (".", "\\")
        or not _is_drive_qualified_absolute_windows_path(normalized)
        or not ntpath.basename(normalized).casefold().endswith(".exe")
    ):
        raise ValueError("process executable identity must name an executable")
    return normalized


def _is_drive_qualified_absolute_windows_path(value: str) -> bool:
    if value.startswith(("\\\\.\\", "\\\\?\\", "\\??\\", "\\Device\\")):
        return False
    drive, tail = ntpath.splitdrive(value)
    is_drive_qualified = (
        len(drive) == 2 and drive[0].isalpha() and drive[1] == ":" and tail.startswith("\\")
    )
    unc_root = drive[2:].split("\\") if drive.startswith("\\\\") else ()
    is_fully_qualified_unc = len(unc_root) == 2 and all(unc_root) and tail.startswith("\\")
    return is_drive_qualified or is_fully_qualified_unc


def _normalize_discord_channel(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Discord target channel must be a string")
    channel = value.strip().casefold()
    if channel not in _DISCORD_BASENAME_BY_CHANNEL:
        raise ValueError("unsupported Discord target channel")
    return channel


@dataclass(frozen=True, slots=True)
class CaptureTargetIntent:
    kind: Literal["default_output_device", "named_output_device", "process"] = (
        "default_output_device"
    )
    device_name: str | None = None
    process: ProcessCaptureTargetIntent | None = None

    def __post_init__(self) -> None:
        process = self.process
        if isinstance(process, Mapping):
            process = ProcessCaptureTargetIntent(**dict(process))
            object.__setattr__(self, "process", process)
        if self.kind == "default_output_device":
            if self.device_name is not None or process is not None:
                raise ValueError("default output target cannot include device or process data")
            return
        if self.kind == "named_output_device":
            if not isinstance(self.device_name, str) or not self.device_name.strip():
                raise ValueError("named output target requires a non-empty device name")
            if process is not None:
                raise ValueError("named output target cannot include process data")
            return
        if self.kind == "process":
            if self.device_name is not None or not isinstance(process, ProcessCaptureTargetIntent):
                raise ValueError("process target requires only a process identity")
            return
        raise ValueError(f"unsupported capture target kind: {self.kind}")

    @classmethod
    def default_output_device(cls) -> CaptureTargetIntent:
        return cls()

    @classmethod
    def named_output_device(cls, device_name: str) -> CaptureTargetIntent:
        return cls(kind="named_output_device", device_name=device_name)

    @classmethod
    def process_target(cls, process: ProcessCaptureTargetIntent) -> CaptureTargetIntent:
        return cls(kind="process", process=process)


@dataclass(frozen=True, slots=True)
class DesktopAudioIntent:
    output_device: str = ""
    capture_target: CaptureTargetIntent = field(default_factory=CaptureTargetIntent)
    vad_speech_threshold: float = 0.5
    vad_hangover_ms: int = 500
    vad_pre_roll_ms: int = 500


@dataclass(frozen=True, slots=True)
class DesktopFletOverlayPositionIntent:
    x: int | float | None = None
    y: int | float | None = None


@dataclass(frozen=True, slots=True)
class DesktopFletOverlayVisualIntent:
    background_alpha: float = 0.6


@dataclass(frozen=True, slots=True)
class DesktopFletOverlayIntent:
    size_preset: str = "medium"
    position: DesktopFletOverlayPositionIntent = field(
        default_factory=DesktopFletOverlayPositionIntent
    )
    swap_caption_languages: bool = False
    visual: DesktopFletOverlayVisualIntent = field(default_factory=DesktopFletOverlayVisualIntent)


@dataclass(frozen=True, slots=True)
class OverlayIntent:
    target: str = "steamvr"
    show_translation: bool = True
    show_peer_original: bool = True
    calibration: OverlayCalibration = field(default_factory=OverlayCalibration)
    desktop_flet: DesktopFletOverlayIntent = field(default_factory=DesktopFletOverlayIntent)


@dataclass(frozen=True, slots=True)
class OscIntent:
    connection_mode: Literal["automatic", "manual", "off"] = "automatic"
    host: str = OSC_DEFAULT_HOST
    port: int = OSC_DEFAULT_SEND_PORT
    send_port: int = OSC_DEFAULT_SEND_PORT
    receive_port: int = OSC_DEFAULT_RECEIVE_PORT
    chatbox_address: str = "/chatbox/input"
    chatbox_send: bool = True
    chatbox_clear: bool = False
    chatbox_max_chars: int = 144
    vrc_mic_intercept: bool = False
    chatbox_include_source: bool = False

    def __post_init__(self) -> None:
        mode = self.connection_mode if self.connection_mode in OSC_CONNECTION_MODES else "automatic"
        if self.port != OSC_DEFAULT_SEND_PORT and self.send_port == OSC_DEFAULT_SEND_PORT:
            send_port = self.port
        elif self.port == OSC_DEFAULT_SEND_PORT:
            send_port = self.send_port
        elif self.port != self.send_port:
            send_port = self.port
        else:
            send_port = self.send_port
        try:
            send_port = int(send_port)
            receive_port = int(self.receive_port)
        except (TypeError, ValueError) as exc:
            raise ValueError("OSC ports must be integers") from exc
        if not (1 <= send_port <= 65535):
            raise ValueError("OSC send_port must be in 1..65535")
        if not (1 <= receive_port <= 65535):
            raise ValueError("OSC receive_port must be in 1..65535")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("OSC host must be non-empty")
        if not isinstance(self.chatbox_address, str) or not self.chatbox_address.startswith("/"):
            raise ValueError("OSC chatbox_address must start with '/'")
        if self.chatbox_max_chars <= 0:
            raise ValueError("OSC chatbox_max_chars must be > 0")
        object.__setattr__(self, "connection_mode", mode)
        object.__setattr__(self, "port", send_port)
        object.__setattr__(self, "send_port", send_port)
        object.__setattr__(self, "receive_port", receive_port)


@dataclass(frozen=True, slots=True)
class SecretsIntent:
    backend: str = "keyring"
    encrypted_file_path: str = "secrets.json"


@dataclass(frozen=True, slots=True)
class UiIntent:
    locale: str = "en"


@dataclass(frozen=True, slots=True)
class ClipboardIntent:
    auto_translate_enabled: bool = False


@dataclass(frozen=True, slots=True)
class IntegratedContextIntent:
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TelemetryConsentIntent:
    consent: str = "unknown"

    def __post_init__(self) -> None:
        consent = self.consent if isinstance(self.consent, str) else "unknown"
        consent = consent.strip()
        if consent not in TELEMETRY_CONSENT_VALUES:
            consent = "unknown"
        object.__setattr__(self, "consent", consent)


@dataclass(frozen=True, slots=True)
class PromptIntent:
    system_prompt: str = ""


@dataclass(frozen=True, slots=True)
class UserIntentSettings:
    translation: TranslationIntent = field(default_factory=TranslationIntent)
    local_llm: LocalLLMIntent = field(default_factory=LocalLLMIntent)
    stt: STTIntent = field(default_factory=STTIntent)
    peer_stt: PeerSTTIntent = field(default_factory=PeerSTTIntent)
    languages: LanguageIntent = field(default_factory=LanguageIntent)
    audio: AudioIntent = field(default_factory=AudioIntent)
    desktop_audio: DesktopAudioIntent = field(default_factory=DesktopAudioIntent)
    overlay: OverlayIntent = field(default_factory=OverlayIntent)
    osc: OscIntent = field(default_factory=OscIntent)
    secrets: SecretsIntent = field(default_factory=SecretsIntent)
    ui: UiIntent = field(default_factory=UiIntent)
    clipboard: ClipboardIntent = field(default_factory=ClipboardIntent)
    integrated_context: IntegratedContextIntent = field(default_factory=IntegratedContextIntent)
    telemetry: TelemetryConsentIntent = field(default_factory=TelemetryConsentIntent)
    prompts: PromptIntent = field(default_factory=PromptIntent)


@dataclass(frozen=True, slots=True)
class ProviderVerificationEntry:
    status: str = "unknown"
    provider: str | None = None
    secret_key: str | None = None
    secret_revision: str | None = None
    secret_fingerprint: str | None = None
    verifier_context: dict[str, object] = field(default_factory=dict)
    verifier_evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in _PROVIDER_VERIFICATION_STATUSES:
            raise ValueError(f"unsupported provider verification status: {self.status}")
        object.__setattr__(self, "status", status)
        if status == "unknown":
            object.__setattr__(self, "provider", None)
            object.__setattr__(self, "secret_key", None)
            object.__setattr__(self, "secret_revision", None)
            object.__setattr__(self, "secret_fingerprint", None)
            object.__setattr__(self, "verifier_context", {})
            object.__setattr__(self, "verifier_evidence", {})
            return

        provider = _required_provider_verification_string(self.provider, field_name="provider")
        secret_key = _required_provider_verification_string(
            self.secret_key,
            field_name="secret_key",
        )
        secret_revision = _optional_provider_verification_string(
            self.secret_revision,
            field_name="secret_revision",
        )
        secret_fingerprint = _optional_provider_verification_string(
            self.secret_fingerprint,
            field_name="secret_fingerprint",
        )
        verifier_context = _copy_provider_verification_metadata(self.verifier_context)
        if secret_revision is None and secret_fingerprint is None:
            raise ValueError(
                "provider verification evidence requires non-empty secret_revision or "
                "secret_fingerprint"
            )
        if not verifier_context:
            raise ValueError("provider verification evidence requires non-empty verifier_context")

        object.__setattr__(self, "provider", provider)
        object.__setattr__(
            self,
            "secret_key",
            secret_key,
        )
        object.__setattr__(
            self,
            "secret_revision",
            secret_revision,
        )
        object.__setattr__(
            self,
            "secret_fingerprint",
            secret_fingerprint,
        )
        object.__setattr__(
            self,
            "verifier_context",
            verifier_context,
        )
        object.__setattr__(
            self,
            "verifier_evidence",
            _copy_provider_verification_metadata(self.verifier_evidence),
        )


@dataclass(frozen=True, slots=True)
class ProviderVerificationState:
    deepgram: ProviderVerificationEntry = field(default_factory=ProviderVerificationEntry)
    soniox: ProviderVerificationEntry = field(default_factory=ProviderVerificationEntry)
    google: ProviderVerificationEntry = field(default_factory=ProviderVerificationEntry)
    openrouter: ProviderVerificationEntry = field(default_factory=ProviderVerificationEntry)
    deepseek: ProviderVerificationEntry = field(default_factory=ProviderVerificationEntry)
    cerebras: ProviderVerificationEntry = field(default_factory=ProviderVerificationEntry)
    alibaba_beijing: ProviderVerificationEntry = field(default_factory=ProviderVerificationEntry)
    alibaba_singapore: ProviderVerificationEntry = field(default_factory=ProviderVerificationEntry)


@dataclass(frozen=True, slots=True)
class ManagedConnectionState:
    installation_id: str = ""
    release_token: str | None = None
    release_token_expires_at: str | None = None
    verified_hardware_hash: str | None = None
    verified_hardware_hash_salt_version: int | None = None
    active_managed_credential_ref: str | None = None
    active_managed_expires_at: str | None = None
    founder_letter_seen_credential_ref: str | None = None
    referral_id: str | None = None
    local_managed_claim_sources: tuple[str, ...] = ()
    pending_delivery_ack_source: str | None = None
    pending_delivery_ack_delivery_id: str | None = None
    pending_delivery_ack_managed_credential_ref: str | None = None
    pending_delivery_ack_expires_at: str | None = None

    def __post_init__(self) -> None:
        source = _normalize_optional_state_text(self.pending_delivery_ack_source)
        if source not in MANAGED_KEY_DELIVERY_ACK_SOURCES:
            source = None
        delivery_id = _normalize_optional_state_text(self.pending_delivery_ack_delivery_id)
        managed_credential_ref = _normalize_optional_state_text(
            self.pending_delivery_ack_managed_credential_ref
        )
        expires_at = _normalize_optional_state_text(self.pending_delivery_ack_expires_at)
        if source is None or delivery_id is None or managed_credential_ref is None:
            source = None
            delivery_id = None
            managed_credential_ref = None
            expires_at = None
        object.__setattr__(
            self,
            "local_managed_claim_sources",
            normalize_managed_claim_sources(self.local_managed_claim_sources),
        )
        object.__setattr__(self, "pending_delivery_ack_source", source)
        object.__setattr__(self, "pending_delivery_ack_delivery_id", delivery_id)
        object.__setattr__(
            self,
            "pending_delivery_ack_managed_credential_ref",
            managed_credential_ref,
        )
        object.__setattr__(self, "pending_delivery_ack_expires_at", expires_at)


@dataclass(frozen=True, slots=True)
class GithubStarPromptState:
    clicked: bool = False
    last_shown_at: str | None = None
    show_count: int = 0
    translation_success_observed: bool = False
    eligible_launch_count: int = 0


@dataclass(frozen=True, slots=True)
class PeerTranslationState:
    eula_accepted: bool = False


@dataclass(frozen=True, slots=True)
class IntegratedContextState:
    bootstrapped: bool = False


@dataclass(frozen=True, slots=True)
class TelemetryOperationalState:
    anonymous_id: str | None = None
    sent_translation_success_dates_utc: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "anonymous_id", _normalize_telemetry_identifier(self.anonymous_id))
        object.__setattr__(
            self,
            "sent_translation_success_dates_utc",
            _normalize_telemetry_sent_dates(self.sent_translation_success_dates_utc),
        )


@dataclass(frozen=True, slots=True)
class PersistedOperationalState:
    provider_verification: ProviderVerificationState = field(
        default_factory=ProviderVerificationState
    )
    managed_connection: ManagedConnectionState = field(default_factory=ManagedConnectionState)
    github_star_prompt: GithubStarPromptState = field(default_factory=GithubStarPromptState)
    peer_translation: PeerTranslationState = field(default_factory=PeerTranslationState)
    integrated_context: IntegratedContextState = field(default_factory=IntegratedContextState)
    telemetry: TelemetryOperationalState = field(default_factory=TelemetryOperationalState)


@dataclass(frozen=True, slots=True)
class AppSettingsVNext:
    settings_version: int = VNEXT_SETTINGS_SCHEMA_VERSION
    intent: UserIntentSettings = field(default_factory=UserIntentSettings)
    state: PersistedOperationalState = field(default_factory=PersistedOperationalState)
    compatibility_extensions: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compatibility_extensions",
            _copy_compatibility_extensions(self.compatibility_extensions),
        )


def with_translation_runtime_policy(
    settings: AppSettingsVNext,
    policy: TranslationRuntimePolicy = FIXED_TRANSLATION_POLICY,
) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            stt=replace(
                settings.intent.stt,
                low_latency_mode=policy.fast_translation_enabled,
            ),
            integrated_context=replace(
                settings.intent.integrated_context,
                enabled=policy.context_policy == "integrated_preferred",
            ),
        ),
    )


def with_capture_target(
    settings: AppSettingsVNext,
    capture_target: CaptureTargetIntent,
) -> AppSettingsVNext:
    if not isinstance(capture_target, CaptureTargetIntent):
        raise TypeError("capture target mutation requires CaptureTargetIntent")
    desktop_audio = settings.intent.desktop_audio
    output_device = desktop_audio.output_device
    if capture_target.kind == "default_output_device":
        output_device = ""
    elif capture_target.kind == "named_output_device":
        output_device = capture_target.device_name or ""
    return replace(
        settings,
        intent=replace(
            settings.intent,
            desktop_audio=replace(
                desktop_audio,
                output_device=output_device,
                capture_target=capture_target,
            ),
        ),
    )


def with_telemetry_consent(
    settings: AppSettingsVNext,
    consent: str,
    *,
    identifier_factory: object = new_anonymous_telemetry_identifier,
) -> AppSettingsVNext:
    normalized_consent = TelemetryConsentIntent(consent).consent
    current_state = settings.state.telemetry
    if normalized_consent == "decline":
        next_state = TelemetryOperationalState()
    elif normalized_consent == "allow":
        factory = (
            identifier_factory
            if callable(identifier_factory)
            else new_anonymous_telemetry_identifier
        )
        next_state = TelemetryOperationalState(
            anonymous_id=current_state.anonymous_id or str(factory()),
            sent_translation_success_dates_utc=current_state.sent_translation_success_dates_utc,
        )
    else:
        next_state = current_state
    return replace(
        settings,
        intent=replace(settings.intent, telemetry=TelemetryConsentIntent(normalized_consent)),
        state=replace(settings.state, telemetry=next_state),
    )


def ensure_telemetry_default_allow(
    settings: AppSettingsVNext,
    *,
    identifier_factory: object = new_anonymous_telemetry_identifier,
) -> AppSettingsVNext:
    consent = TelemetryConsentIntent(settings.intent.telemetry.consent).consent
    if consent == "decline":
        return settings
    if consent == "allow" and settings.state.telemetry.anonymous_id:
        return settings
    return with_telemetry_consent(settings, "allow", identifier_factory=identifier_factory)


__all__ = [
    "AppSettingsVNext",
    "AudioIntent",
    "CaptureTargetIntent",
    "ClipboardIntent",
    "CerebrasTranslationIntent",
    "CustomSTTIntent",
    "DeepgramSTTIntent",
    "DeepSeekTranslationIntent",
    "DesktopAudioIntent",
    "DesktopFletOverlayIntent",
    "DesktopFletOverlayPositionIntent",
    "DesktopFletOverlayVisualIntent",
    "GithubStarPromptState",
    "GeminiTranslationIntent",
    "IntegratedContextIntent",
    "IntegratedContextState",
    "is_safe_compatibility_extension_key",
    "CANONICAL_TRANSLATION_FALLBACK_ALIASES",
    "COMPAT_TRANSLATION_FALLBACK_ALIASES",
    "DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS",
    "LanguageIntent",
    "LocalLLMIntent",
    "ManagedConnectionState",
    "MANAGED_AUTH_CLAIM_SOURCE_DISCORD",
    "MANAGED_AUTH_CLAIM_SOURCE_QQ",
    "MANAGED_AUTH_CLAIM_SOURCES",
    "MANAGED_KEY_DELIVERY_ACK_SOURCES",
    "OscIntent",
    "OverlayIntent",
    "PeerSTTIntent",
    "PeerTranslationState",
    "PersistedOperationalState",
    "ProcessCaptureTargetIntent",
    "PromptIntent",
    "ProviderVerificationEntry",
    "ProviderVerificationState",
    "QwenASRSTTIntent",
    "QwenTranslationIntent",
    "RUNTIME_ONLY_LEGACY_SETTINGS_PATHS",
    "STTIntent",
    "SecretsIntent",
    "SonioxSTTIntent",
    "TranslationIntent",
    "TranslationFallbackIntent",
    "TelemetryConsentIntent",
    "TelemetryOperationalState",
    "TELEMETRY_CONSENT_VALUES",
    "UiIntent",
    "UserIntentSettings",
    "VNEXT_SETTINGS_SCHEMA_VERSION",
    "OSC_CONNECTION_MODES",
    "OSC_DEFAULT_HOST",
    "OSC_DEFAULT_RECEIVE_PORT",
    "OSC_DEFAULT_SEND_PORT",
    "with_capture_target",
    "with_telemetry_consent",
    "with_translation_runtime_policy",
    "ensure_telemetry_default_allow",
]
