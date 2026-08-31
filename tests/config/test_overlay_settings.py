from __future__ import annotations

from dataclasses import replace

from puripuly_heart.config.desktop_overlay_values import DESKTOP_FLET_SIZE_PRESETS
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
)


def test_overlay_settings_desktop_flet_defaults_serialize_canonical_shape() -> None:
    settings = AppSettingsVNext()
    data = serialization.to_dict(settings)

    assert settings.settings_version == VNEXT_SETTINGS_SCHEMA_VERSION
    assert settings.intent.overlay.target == "steamvr"
    assert settings.intent.overlay.desktop_flet.size_preset == "medium"
    assert settings.intent.overlay.desktop_flet.position.x is None
    assert settings.intent.overlay.desktop_flet.position.y is None
    assert not hasattr(settings.intent.overlay.desktop_flet, "locked")
    assert settings.intent.overlay.desktop_flet.swap_caption_languages is False
    assert settings.intent.overlay.desktop_flet.visual.background_alpha == 0.6
    assert data["intent"]["overlay"]["target"] == "steamvr"
    assert data["intent"]["overlay"]["desktop_flet"] == {
        "size_preset": "medium",
        "position": {"x": None, "y": None},
        "swap_caption_languages": False,
        "visual": {"background_alpha": 0.6},
    }
    assert "locked" not in data["intent"]["overlay"]["desktop_flet"]


def test_overlay_settings_desktop_flet_size_presets_match_c_light_caption_layout() -> None:
    assert DESKTOP_FLET_SIZE_PRESETS == {
        "tiny": (640, 160),
        "xsmall": (960, 240),
        "small": (1152, 288),
        "medium": (1344, 336),
        "large": (1600, 400),
        "xlarge": (1792, 448),
    }


def test_overlay_settings_desktop_flet_tiny_preset_round_trips() -> None:
    current = AppSettingsVNext()
    settings = replace(
        current,
        intent=replace(
            current.intent,
            overlay=replace(
                current.intent.overlay,
                desktop_flet=replace(current.intent.overlay.desktop_flet, size_preset="tiny"),
            ),
        ),
    )
    data = serialization.to_dict(settings)
    round_tripped = serialization.from_dict(data)

    assert data["intent"]["overlay"]["desktop_flet"]["size_preset"] == "tiny"
    assert round_tripped.intent.overlay.desktop_flet.size_preset == "tiny"


def test_overlay_settings_desktop_flet_swap_caption_languages_round_trips() -> None:
    current = AppSettingsVNext()
    settings = replace(
        current,
        intent=replace(
            current.intent,
            overlay=replace(
                current.intent.overlay,
                desktop_flet=replace(
                    current.intent.overlay.desktop_flet,
                    swap_caption_languages=True,
                ),
            ),
        ),
    )
    data = serialization.to_dict(settings)
    round_tripped = serialization.from_dict(data)

    assert settings.intent.overlay.desktop_flet.swap_caption_languages is True
    assert data["intent"]["overlay"]["desktop_flet"]["swap_caption_languages"] is True
    assert round_tripped.intent.overlay.desktop_flet.swap_caption_languages is True
