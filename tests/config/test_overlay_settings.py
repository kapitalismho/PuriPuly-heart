from __future__ import annotations

from dataclasses import replace

from puripuly_heart.config.desktop_overlay_values import (
    DESKTOP_FLET_DEFAULT_SIZE_PRESET,
    DESKTOP_FLET_SIZE_PRESET_ORDER,
    DESKTOP_FLET_SIZE_PRESETS,
)
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_SIZE_PRESETS as RENDERER_SIZE_PRESETS,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_caption_size_preset_for_dimensions,
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
    assert DESKTOP_FLET_DEFAULT_SIZE_PRESET in DESKTOP_FLET_SIZE_PRESETS
    assert DESKTOP_FLET_DEFAULT_SIZE_PRESET in RENDERER_SIZE_PRESETS
    for preset_id in DESKTOP_FLET_SIZE_PRESET_ORDER:
        renderer_preset = RENDERER_SIZE_PRESETS[preset_id]
        assert (renderer_preset.window_width, renderer_preset.window_height) == (
            DESKTOP_FLET_SIZE_PRESETS[preset_id]
        ), f"desktop caption preset {preset_id} diverged from settings"
        resolved = _desktop_caption_size_preset_for_dimensions(
            *DESKTOP_FLET_SIZE_PRESETS[preset_id]
        )
        assert resolved.id == preset_id


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


def test_overlay_speaker_transition_mode_round_trips_and_invalid_values_fall_back_to_a() -> None:
    current = AppSettingsVNext()
    settings = replace(
        current,
        intent=replace(
            current.intent,
            overlay=replace(current.intent.overlay, speaker_transition_mode="E"),
        ),
    )

    data = serialization.to_dict(settings)
    round_tripped = serialization.from_dict(data)
    invalid = serialization.from_dict(
        {
            **data,
            "intent": {
                **data["intent"],
                "overlay": {
                    **data["intent"]["overlay"],
                    "speaker_transition_mode": "unknown",
                },
            },
        }
    )

    assert data["intent"]["overlay"]["speaker_transition_mode"] == "E"
    assert round_tripped.intent.overlay.speaker_transition_mode == "E"
    assert invalid.intent.overlay.speaker_transition_mode == "A"
