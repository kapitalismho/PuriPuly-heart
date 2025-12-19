from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class STTProviderName(str, Enum):
    ALIBABA = "alibaba"
    DEEPGRAM = "deepgram"


class LLMProviderName(str, Enum):
    GEMINI = "gemini"
    QWEN = "qwen"


class SecretsBackend(str, Enum):
    KEYRING = "keyring"
    ENCRYPTED_FILE = "encrypted_file"


@dataclass(slots=True)
class LanguageSettings:
    source_language: str = "ko-KR"
    target_language: str = "en-US"

    def validate(self) -> None:
        if not self.source_language:
            raise ValueError("source_language must be non-empty")
        if not self.target_language:
            raise ValueError("target_language must be non-empty")


@dataclass(slots=True)
class AudioSettings:
    internal_sample_rate_hz: int = 16000
    internal_channels: int = 1
    ring_buffer_ms: int = 500
    input_host_api: str = ""
    input_device: str = ""

    def validate(self) -> None:
        if self.internal_sample_rate_hz not in (8000, 16000):
            raise ValueError("internal_sample_rate_hz must be 8000 or 16000")
        if self.internal_channels != 1:
            raise ValueError("internal_channels must be 1 (mono)")
        if self.ring_buffer_ms <= 0:
            raise ValueError("ring_buffer_ms must be > 0")
        if self.input_host_api is None:
            raise ValueError("input_host_api must be a string")
        if self.input_device is None:
            raise ValueError("input_device must be a string")


@dataclass(slots=True)
class STTSettings:
    reset_deadline_s: float = 90.0
    drain_timeout_s: float = 2.0
    vad_speech_threshold: float = 0.5

    def validate(self) -> None:
        if self.reset_deadline_s <= 0:
            raise ValueError("reset_deadline_s must be > 0")
        if self.drain_timeout_s <= 0:
            raise ValueError("drain_timeout_s must be > 0")
        if not (0.0 <= self.vad_speech_threshold <= 1.0):
            raise ValueError("vad_speech_threshold must be in 0.0..1.0")




@dataclass(slots=True)
class AlibabaSTTSettings:
    model: str = "paraformer-realtime-v2"
    endpoint: str = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if not self.endpoint:
            raise ValueError("endpoint must be non-empty")


@dataclass(slots=True)
class DeepgramSTTSettings:
    model: str = "nova-3"

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")


@dataclass(slots=True)
class LLMSettings:
    concurrency_limit: int = 1

    def validate(self) -> None:
        if self.concurrency_limit <= 0:
            raise ValueError("concurrency_limit must be > 0")


@dataclass(slots=True)
class OSCSettings:
    host: str = "127.0.0.1"
    port: int = 9000
    chatbox_address: str = "/chatbox/input"
    chatbox_send: bool = True
    chatbox_clear: bool = False
    chatbox_max_chars: int = 144
    cooldown_s: float = 1.5
    ttl_s: float = 7.0

    def validate(self) -> None:
        if not self.host:
            raise ValueError("host must be non-empty")
        if not (0 < self.port <= 65535):
            raise ValueError("port must be in 1..65535")
        if not self.chatbox_address or not self.chatbox_address.startswith("/"):
            raise ValueError("chatbox_address must start with '/'")
        if self.chatbox_max_chars <= 0:
            raise ValueError("chatbox_max_chars must be > 0")
        if self.cooldown_s <= 0:
            raise ValueError("cooldown_s must be > 0")
        if self.ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")


@dataclass(slots=True)
class ProviderSettings:
    stt: STTProviderName = STTProviderName.DEEPGRAM
    llm: LLMProviderName = LLMProviderName.GEMINI

    def validate(self) -> None:
        if not isinstance(self.stt, STTProviderName):
            raise ValueError("invalid stt provider")
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
class AppSettings:
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    languages: LanguageSettings = field(default_factory=LanguageSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    stt: STTSettings = field(default_factory=STTSettings)
    alibaba_stt: AlibabaSTTSettings = field(default_factory=AlibabaSTTSettings)
    deepgram_stt: DeepgramSTTSettings = field(default_factory=DeepgramSTTSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    osc: OSCSettings = field(default_factory=OSCSettings)
    secrets: SecretsSettings = field(default_factory=SecretsSettings)
    system_prompt: str = ""

    def validate(self) -> None:
        self.provider.validate()
        self.languages.validate()
        self.audio.validate()
        self.stt.validate()
        self.alibaba_stt.validate()
        self.deepgram_stt.validate()
        self.llm.validate()
        self.osc.validate()
        self.secrets.validate()


def _enum_to_value(obj: object) -> object:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _enum_to_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_enum_to_value(v) for v in obj]
    return obj


def to_dict(settings: AppSettings) -> dict[str, Any]:
    data: dict[str, Any] = {
        "provider": {"stt": settings.provider.stt.value, "llm": settings.provider.llm.value},
        "languages": {
            "source_language": settings.languages.source_language,
            "target_language": settings.languages.target_language,
        },
        "audio": {
            "internal_sample_rate_hz": settings.audio.internal_sample_rate_hz,
            "internal_channels": settings.audio.internal_channels,
            "ring_buffer_ms": settings.audio.ring_buffer_ms,
            "input_host_api": settings.audio.input_host_api,
            "input_device": settings.audio.input_device,
        },
        "stt": {
            "reset_deadline_s": settings.stt.reset_deadline_s,
            "drain_timeout_s": settings.stt.drain_timeout_s,
            "vad_speech_threshold": settings.stt.vad_speech_threshold,
        },
        "alibaba_stt": {
            "model": settings.alibaba_stt.model,
            "endpoint": settings.alibaba_stt.endpoint,
        },
        "deepgram_stt": {
            "model": settings.deepgram_stt.model,
        },
        "llm": {"concurrency_limit": settings.llm.concurrency_limit},
        "osc": {
            "host": settings.osc.host,
            "port": settings.osc.port,
            "chatbox_address": settings.osc.chatbox_address,
            "chatbox_send": settings.osc.chatbox_send,
            "chatbox_clear": settings.osc.chatbox_clear,
            "chatbox_max_chars": settings.osc.chatbox_max_chars,
            "cooldown_s": settings.osc.cooldown_s,
            "ttl_s": settings.osc.ttl_s,
        },
        "secrets": {
            "backend": settings.secrets.backend.value,
            "encrypted_file_path": settings.secrets.encrypted_file_path,
        },
        "system_prompt": settings.system_prompt,
    }
    return _enum_to_value(data)  # type: ignore[return-value]


def _parse_stt_provider(value: str) -> STTProviderName:
    """Parse STT provider, falling back to DEEPGRAM for legacy/invalid values."""
    try:
        return STTProviderName(value)
    except ValueError:
        return STTProviderName.DEEPGRAM


def from_dict(data: dict[str, Any]) -> AppSettings:
    audio_data = data.get("audio") or {}
    stt_data = data.get("stt") or {}

    input_host_api_raw = audio_data.get("input_host_api")
    input_device_raw = audio_data.get("input_device")
    vad_threshold_raw = stt_data.get("vad_speech_threshold")

    settings = AppSettings(
        provider=ProviderSettings(
            stt=_parse_stt_provider(data.get("provider", {}).get("stt", STTProviderName.DEEPGRAM.value)),
            llm=LLMProviderName(data.get("provider", {}).get("llm", LLMProviderName.GEMINI.value)),
        ),
        languages=LanguageSettings(
            source_language=data.get("languages", {}).get("source_language", "ko-KR"),
            target_language=data.get("languages", {}).get("target_language", "en-US"),
        ),
        audio=AudioSettings(
            internal_sample_rate_hz=int(audio_data.get("internal_sample_rate_hz", 16000)),
            internal_channels=int(audio_data.get("internal_channels", 1)),
            ring_buffer_ms=int(audio_data.get("ring_buffer_ms", 500)),
            input_host_api=str(input_host_api_raw) if input_host_api_raw is not None else "",
            input_device=str(input_device_raw) if input_device_raw is not None else "",
        ),
        stt=STTSettings(
            reset_deadline_s=float(stt_data.get("reset_deadline_s", 90.0)),
            drain_timeout_s=float(stt_data.get("drain_timeout_s", 2.0)),
            vad_speech_threshold=float(vad_threshold_raw) if vad_threshold_raw is not None else 0.5,
        ),
        alibaba_stt=AlibabaSTTSettings(
            model=str(data.get("alibaba_stt", {}).get("model", "paraformer-realtime-v2")),
            endpoint=str(
                data.get("alibaba_stt", {}).get("endpoint", "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference")
            ),
        ),
        deepgram_stt=DeepgramSTTSettings(
            model=str(data.get("deepgram_stt", {}).get("model", "nova-3")),
        ),
        llm=LLMSettings(concurrency_limit=int(data.get("llm", {}).get("concurrency_limit", 1))),
        osc=OSCSettings(
            host=str(data.get("osc", {}).get("host", "127.0.0.1")),
            port=int(data.get("osc", {}).get("port", 9000)),
            chatbox_address=str(data.get("osc", {}).get("chatbox_address", "/chatbox/input")),
            chatbox_send=bool(data.get("osc", {}).get("chatbox_send", True)),
            chatbox_clear=bool(data.get("osc", {}).get("chatbox_clear", False)),
            chatbox_max_chars=int(data.get("osc", {}).get("chatbox_max_chars", 144)),
            cooldown_s=float(data.get("osc", {}).get("cooldown_s", 1.5)),
            ttl_s=float(data.get("osc", {}).get("ttl_s", 7.0)),
        ),
        secrets=SecretsSettings(
            backend=SecretsBackend(data.get("secrets", {}).get("backend", SecretsBackend.KEYRING.value)),
            encrypted_file_path=data.get("secrets", {}).get("encrypted_file_path", "secrets.json"),
        ),
        system_prompt=str(data.get("system_prompt", "")),
    )
    settings.validate()
    return settings


def load_settings(path: Path) -> AppSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("settings file must contain a JSON object")
    return from_dict(raw)


def save_settings(path: Path, settings: AppSettings) -> None:
    settings.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
