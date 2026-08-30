from __future__ import annotations

from dataclasses import replace

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, CustomSTTIntent


def _with_custom_stt(
    *,
    provider: str = "custom",
    peer_provider: str = "custom",
    mode: str = "offline",
    compatibility: str = "openai_transcription",
    endpoint: str = "",
    model: str = "",
    extra: dict[str, object] | None = None,
) -> AppSettingsVNext:
    current = AppSettingsVNext()
    return replace(
        current,
        intent=replace(
            current.intent,
            stt=replace(
                current.intent.stt,
                provider=provider,
                custom=CustomSTTIntent(
                    mode=mode,
                    compatibility=compatibility,
                    endpoint=endpoint,
                    model=model,
                    extra={} if extra is None else extra,
                ),
            ),
            peer_stt=replace(current.intent.peer_stt, provider=peer_provider),
        ),
    )


def test_custom_stt_can_be_selected_for_self_and_peer() -> None:
    settings = _with_custom_stt(
        endpoint="http://127.0.0.1:8000/v1",
        model="whisper-1",
    )
    persisted = serialization.to_dict(settings)
    loaded = serialization.from_dict(persisted)

    assert loaded.intent.stt.provider == STTProviderName.CUSTOM.value
    assert loaded.intent.peer_stt.provider == STTProviderName.CUSTOM.value
    assert loaded.intent.stt.custom.mode == "offline"
    assert loaded.intent.stt.custom.compatibility == "openai_transcription"
    assert loaded.intent.stt.custom.endpoint == "http://127.0.0.1:8000/v1"
    assert loaded.intent.stt.custom.model == "whisper-1"
    assert "api_key" not in persisted["intent"]["stt"]["custom"]
    assert "authorization" not in persisted["intent"]["stt"]["custom"]


def test_custom_stt_settings_are_shared_across_self_and_peer() -> None:
    settings = _with_custom_stt(
        peer_provider=STTProviderName.DEEPGRAM.value,
        endpoint="http://127.0.0.1:9000",
        model="shared-model",
    )
    loaded = serialization.from_dict(serialization.to_dict(settings))
    assert loaded.intent.stt.custom.endpoint == "http://127.0.0.1:9000"
    assert loaded.intent.stt.custom.model == "shared-model"


def test_custom_offline_and_realtime_providers_persist() -> None:
    settings = _with_custom_stt(
        provider=STTProviderName.CUSTOM_OFFLINE.value,
        peer_provider=STTProviderName.CUSTOM_REALTIME.value,
        endpoint="http://127.0.0.1:8000/v1",
        model="whisper-1",
    )
    loaded = serialization.from_dict(serialization.to_dict(settings))
    assert loaded.intent.stt.provider == STTProviderName.CUSTOM_OFFLINE.value
    assert loaded.intent.peer_stt.provider == STTProviderName.CUSTOM_REALTIME.value
    assert loaded.intent.stt.custom.endpoint == "http://127.0.0.1:8000/v1"
    assert loaded.intent.stt.custom.model == "whisper-1"


def test_custom_stt_extra_round_trips() -> None:
    extra = {
        "model": "my-model",
        "max_tokens": 32,
        "nested": {"a": [1, 2]},
    }
    settings = _with_custom_stt(extra=extra)
    loaded = serialization.from_dict(serialization.to_dict(settings))
    assert loaded.intent.stt.custom.extra == extra
