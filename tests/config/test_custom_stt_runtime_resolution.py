from __future__ import annotations

from puripuly_heart.app.wiring.wiring_stt_factory import (
    create_stt_backend_from_resolved_config,
    resolve_self_stt_runtime_config,
)
from puripuly_heart.config.runtime_resolution import (
    CREDENTIAL_REF_CUSTOM_STT,
    STT_PROVIDER_CUSTOM,
    STT_PROVIDER_CUSTOM_OFFLINE,
    STT_PROVIDER_CUSTOM_REALTIME,
    STTRuntimeIntent,
    resolve_stt_config,
)
from puripuly_heart.config.settings import AppSettings, STTProviderName


class _MemorySecrets:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values = dict(values or {})

    def get(self, key: str) -> str | None:
        return self._values.get(key)


def test_custom_stt_runtime_resolution_keeps_mode_and_endpoint() -> None:
    config = resolve_stt_config(
        STTRuntimeIntent(
            provider=STT_PROVIDER_CUSTOM,
            custom_stt_mode="realtime",
            custom_stt_compatibility="openai_realtime",
            custom_stt_endpoint="http://127.0.0.1:8080",
            custom_stt_model="gpt-4o-mini-transcribe",
        )
    )

    assert config.provider == STT_PROVIDER_CUSTOM
    assert config.endpoint == "http://127.0.0.1:8080"
    assert config.model == "gpt-4o-mini-transcribe"
    assert config.credential.required is False
    assert config.credential.reference == CREDENTIAL_REF_CUSTOM_STT
    assert config.provider_options["mode"] == "realtime"
    assert config.provider_options["compatibility"] == "openai_realtime"
    assert config.custom_vocabulary_enabled is False


def test_factory_builds_custom_backend_without_requiring_secret() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.CUSTOM
    settings.custom_stt.mode = "offline"
    settings.custom_stt.compatibility = "openai_transcription"
    settings.custom_stt.endpoint = "http://127.0.0.1:8000/v1"
    settings.custom_stt.model = "whisper-1"

    resolved = resolve_self_stt_runtime_config(settings)
    backend = create_stt_backend_from_resolved_config(resolved, secrets=_MemorySecrets())

    assert backend.mode == "offline"
    assert backend.compatibility == "openai_transcription"
    assert backend.endpoint == "http://127.0.0.1:8000/v1"
    assert backend.api_key == ""


def test_custom_offline_provider_uses_transcription_even_if_stored_mode_differs() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.CUSTOM_OFFLINE
    settings.custom_stt.mode = "realtime"
    settings.custom_stt.compatibility = "openai_realtime"
    settings.custom_stt.endpoint = "http://127.0.0.1:8000/v1"
    settings.custom_stt.model = "whisper-1"

    resolved = resolve_self_stt_runtime_config(settings)

    assert resolved.provider == STT_PROVIDER_CUSTOM_OFFLINE
    assert resolved.provider_options["mode"] == "offline"
    assert resolved.provider_options["compatibility"] == "openai_transcription"


def test_custom_realtime_provider_uses_realtime_even_if_stored_mode_differs() -> None:
    config = resolve_stt_config(
        STTRuntimeIntent(
            provider=STT_PROVIDER_CUSTOM_REALTIME,
            custom_stt_mode="offline",
            custom_stt_compatibility="openai_transcription",
            custom_stt_endpoint="wss://127.0.0.1:8080/v1/realtime",
            custom_stt_model="gpt-4o-realtime",
        )
    )

    assert config.provider == STT_PROVIDER_CUSTOM_REALTIME
    assert config.provider_options["mode"] == "realtime"
    assert config.provider_options["compatibility"] == "openai_realtime"


def test_custom_stt_runtime_extra_flows_to_backend() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.CUSTOM
    settings.custom_stt.mode = "offline"
    settings.custom_stt.compatibility = "openai_transcription"
    settings.custom_stt.endpoint = "http://127.0.0.1:8000/v1"
    settings.custom_stt.model = "whisper-1"
    settings.custom_stt.extra = {"prompt": "hello", "max_tokens": 16}
    settings.validate()

    resolved = resolve_self_stt_runtime_config(settings)
    assert resolved.provider_options["extra"] == {"prompt": "hello", "max_tokens": 16}

    backend = create_stt_backend_from_resolved_config(resolved, secrets=_MemorySecrets())
    assert backend.extra == {"prompt": "hello", "max_tokens": 16}
