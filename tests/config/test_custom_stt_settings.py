from __future__ import annotations

import pytest

from puripuly_heart.config.settings import (
    AppSettings,
    STTProviderName,
    from_dict,
    to_dict,
)
from puripuly_heart.config.settings_vnext.migration import from_legacy_app_settings
from puripuly_heart.core.stt.custom import CustomSTTConfigurationError


def test_custom_stt_can_be_selected_for_self_and_peer() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.CUSTOM
    settings.provider.peer_stt = STTProviderName.CUSTOM
    settings.custom_stt.mode = "offline"
    settings.custom_stt.compatibility = "openai_transcription"
    settings.custom_stt.endpoint = "http://127.0.0.1:8000/v1"
    settings.custom_stt.model = "whisper-1"
    settings.validate()

    persisted = to_dict(settings)
    loaded = from_dict(persisted)

    assert loaded.provider.stt == STTProviderName.CUSTOM
    assert loaded.provider.peer_stt == STTProviderName.CUSTOM
    assert loaded.custom_stt.mode == "offline"
    assert loaded.custom_stt.compatibility == "openai_transcription"
    assert loaded.custom_stt.endpoint == "http://127.0.0.1:8000/v1"
    assert loaded.custom_stt.model == "whisper-1"
    assert "api_key" not in persisted["custom_stt"]
    assert "authorization" not in persisted["custom_stt"]


def test_custom_stt_settings_are_shared_across_self_and_peer() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.CUSTOM
    settings.provider.peer_stt = STTProviderName.DEEPGRAM
    settings.custom_stt.endpoint = "http://127.0.0.1:9000"
    settings.custom_stt.model = "shared-model"

    loaded = from_dict(to_dict(settings))
    assert loaded.custom_stt.endpoint == "http://127.0.0.1:9000"
    assert loaded.custom_stt.model == "shared-model"

    vnext = from_legacy_app_settings(loaded)
    assert vnext.intent.stt.custom.endpoint == "http://127.0.0.1:9000"
    assert vnext.intent.stt.custom.model == "shared-model"


def test_custom_stt_rejects_incompatible_mode_and_protocol() -> None:
    settings = AppSettings()
    settings.custom_stt.mode = "offline"
    settings.custom_stt.compatibility = "openai_realtime"
    with pytest.raises(CustomSTTConfigurationError):
        settings.validate()


def test_custom_stt_strips_credential_bearing_endpoints() -> None:
    settings = AppSettings()
    settings.custom_stt.endpoint = "https://user:token@example.test/v1?api_key=secret"
    settings.validate()
    persisted = to_dict(settings)
    assert persisted["custom_stt"]["endpoint"] == "https://example.test/v1"
    assert "token" not in persisted["custom_stt"]["endpoint"]
    assert "secret" not in persisted["custom_stt"]["endpoint"]


def test_custom_offline_and_realtime_providers_persist() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.CUSTOM_OFFLINE
    settings.provider.peer_stt = STTProviderName.CUSTOM_REALTIME
    settings.custom_stt.endpoint = "http://127.0.0.1:8000/v1"
    settings.custom_stt.model = "whisper-1"
    settings.validate()

    loaded = from_dict(to_dict(settings))

    assert loaded.provider.stt == STTProviderName.CUSTOM_OFFLINE
    assert loaded.provider.peer_stt == STTProviderName.CUSTOM_REALTIME
    assert loaded.custom_stt.endpoint == "http://127.0.0.1:8000/v1"
    assert loaded.custom_stt.model == "whisper-1"


def test_missing_custom_stt_section_defaults_without_credentials() -> None:
    loaded = from_dict({"provider": {"stt": "custom", "peer_stt": "custom"}})
    assert loaded.provider.stt == STTProviderName.CUSTOM
    assert loaded.custom_stt.mode == "offline"
    assert loaded.custom_stt.compatibility == "openai_transcription"
    assert loaded.custom_stt.endpoint == ""
    assert loaded.custom_stt.model == ""
    assert loaded.custom_stt.extra == {}


def test_custom_stt_extra_round_trips_and_rejects_bad_keys() -> None:
    settings = AppSettings()
    settings.custom_stt.extra = {
        "model": "my-model",
        "max_tokens": 32,
        "nested": {"a": [1, 2]},
    }
    settings.validate()
    loaded = from_dict(to_dict(settings))
    assert loaded.custom_stt.extra == {
        "model": "my-model",
        "max_tokens": 32,
        "nested": {"a": [1, 2]},
    }

    vnext = from_legacy_app_settings(loaded)
    assert vnext.intent.stt.custom.extra == loaded.custom_stt.extra

    rejected = AppSettings()
    rejected.custom_stt.extra = {"api_key": "secret"}
    with pytest.raises(CustomSTTConfigurationError):
        rejected.validate()

    rejected_reserved = AppSettings()
    rejected_reserved.custom_stt.extra = {"file": "x"}
    with pytest.raises(CustomSTTConfigurationError):
        rejected_reserved.validate()

    not_json = AppSettings()
    not_json.custom_stt.extra = {"bad": object()}
    with pytest.raises(CustomSTTConfigurationError):
        not_json.validate()
