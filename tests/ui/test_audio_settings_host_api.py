from __future__ import annotations

import json
import sys
from importlib import resources
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

from puripuly_heart.config import audio_host_api as host_api_config
from puripuly_heart.config.audio_host_api import (
    WINDOWS_DIRECTSOUND_HOST_API,
    WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
    WINDOWS_WASAPI_HOST_API,
)
from puripuly_heart.ui.components.settings.audio_settings import AudioSettings
from puripuly_heart.ui.i18n import available_locales, get_locale, set_locale, t

WINDOWS_MME_HOST_API = "MME"


def _fake_sounddevice(monkeypatch: pytest.MonkeyPatch, *, hostapis, devices=()) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_hostapis=lambda: hostapis,
            query_devices=lambda: devices,
        ),
    )


def test_mme_host_api_profile_is_canonical() -> None:
    assert getattr(host_api_config, "WINDOWS_MME_HOST_API", None) == WINDOWS_MME_HOST_API

    profile = host_api_config.normalize_input_host_api(WINDOWS_MME_HOST_API)

    assert profile.saved_value == WINDOWS_MME_HOST_API
    assert profile.actual_host_api == WINDOWS_MME_HOST_API
    assert profile.wasapi_auto_convert is False
    assert profile.wasapi_exclusive is False


def test_host_api_options_include_wasapi_compatibility_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_locale = get_locale()
    try:
        set_locale("ko")
        _fake_sounddevice(
            monkeypatch,
            hostapis=[
                {"name": "MME"},
                {"name": WINDOWS_WASAPI_HOST_API},
                {"name": WINDOWS_DIRECTSOUND_HOST_API},
            ],
        )

        settings = AudioSettings()

        options = settings._get_host_api_options()

        assert [option.value for option in options] == [
            "",
            WINDOWS_MME_HOST_API,
            WINDOWS_WASAPI_HOST_API,
            WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
            WINDOWS_DIRECTSOUND_HOST_API,
        ]
        compatibility_label = options[3].label
        assert compatibility_label == t(
            "settings.audio_host_api.option.windows_wasapi_compatibility"
        )
        assert compatibility_label != WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    finally:
        set_locale(old_locale)


def test_host_api_options_include_mme_only_when_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_locale = get_locale()
    try:
        set_locale("en")
        _fake_sounddevice(
            monkeypatch,
            hostapis=[
                {"name": WINDOWS_MME_HOST_API},
                {"name": WINDOWS_WASAPI_HOST_API},
                {"name": WINDOWS_DIRECTSOUND_HOST_API},
            ],
        )

        settings = AudioSettings()
        options = settings._get_host_api_options()

        mme_options = [option for option in options if option.value == WINDOWS_MME_HOST_API]
        assert len(mme_options) == 1
        assert mme_options[0].label == t("settings.audio_host_api.option.windows_mme")

        _fake_sounddevice(
            monkeypatch,
            hostapis=[
                {"name": WINDOWS_WASAPI_HOST_API},
                {"name": WINDOWS_DIRECTSOUND_HOST_API},
            ],
        )
        options_without_mme = settings._get_host_api_options()

        assert WINDOWS_MME_HOST_API not in [option.value for option in options_without_mme]
    finally:
        set_locale(old_locale)


def test_mme_selection_persists_canonical_value_and_uses_i18n_label() -> None:
    old_locale = get_locale()
    try:
        set_locale("ko")
        settings = AudioSettings()

        settings.host_api = WINDOWS_MME_HOST_API

        assert settings.host_api == WINDOWS_MME_HOST_API
        assert settings.host_api_display_label == t("settings.audio_host_api.option.windows_mme")
    finally:
        set_locale(old_locale)


def test_compatibility_mode_enumerates_wasapi_microphones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_sounddevice(
        monkeypatch,
        hostapis=[
            {"name": WINDOWS_WASAPI_HOST_API},
            {"name": WINDOWS_DIRECTSOUND_HOST_API},
        ],
        devices=[
            {"name": "WASAPI Mic", "hostapi": 0, "max_input_channels": 1},
            {"name": "DirectSound Mic", "hostapi": 1, "max_input_channels": 1},
            {"name": "WASAPI Output", "hostapi": 0, "max_input_channels": 0},
        ],
    )
    settings = AudioSettings()
    settings.host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API

    options = settings._get_microphone_options()

    assert [option.value for option in options] == ["", "WASAPI Mic"]


def test_mme_mode_enumerates_mme_microphones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_sounddevice(
        monkeypatch,
        hostapis=[
            {"name": WINDOWS_MME_HOST_API},
            {"name": WINDOWS_WASAPI_HOST_API},
        ],
        devices=[
            {"name": "MME Mic", "hostapi": 0, "max_input_channels": 1},
            {"name": "WASAPI Mic", "hostapi": 1, "max_input_channels": 1},
        ],
    )
    settings = AudioSettings()
    settings.host_api = WINDOWS_MME_HOST_API

    options = settings._get_microphone_options()

    assert [option.value for option in options] == ["", "MME Mic"]


def test_mme_host_api_label_exists_in_supported_locale_bundles() -> None:
    key = "settings.audio_host_api.option.windows_mme"

    for locale in available_locales():
        bundle_path = resources.files("puripuly_heart").joinpath(f"data/i18n/{locale}.json")
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert bundle.get(key) == WINDOWS_MME_HOST_API


def test_host_api_selection_resets_selected_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.ui.views import settings as settings_view_module

    view = settings_view_module.SettingsView.__new__(settings_view_module.SettingsView)
    view._audio_settings = AudioSettings()
    view._audio_settings.host_api = WINDOWS_WASAPI_HOST_API
    view._audio_settings.microphone = "Previous Mic"
    view._sync_general_audio_card_texts = lambda: None
    view._on_audio_change = lambda: None

    view._on_mic_host_api_selected(WINDOWS_WASAPI_COMPATIBILITY_HOST_API)

    assert view._audio_settings.host_api == WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    assert view._audio_settings.microphone == ""
