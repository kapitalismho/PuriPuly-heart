from __future__ import annotations

import json

from puripuly_heart.config.settings import from_dict, to_dict
from tests.config.settings_vnext_test_helpers import legacy_projected_settings_file


def test_peer_translation_eula_acceptance_defaults_false() -> None:
    settings = from_dict({})

    assert settings.ui.peer_translation_eula_accepted is False
    assert to_dict(settings)["ui"]["peer_translation_eula_accepted"] is False


def test_peer_translation_eula_acceptance_round_trips() -> None:
    settings = from_dict({"ui": {"peer_translation_eula_accepted": True}})

    assert settings.ui.peer_translation_eula_accepted is True
    assert from_dict(to_dict(settings)).ui.peer_translation_eula_accepted is True


def test_overlay_display_preferences_round_trip_in_shared_overlay_section() -> None:
    settings = from_dict(
        {
            "ui": {
                "overlay_enabled": True,
                "peer_translation_enabled": True,
                "integrated_context_enabled": True,
                "integrated_context_bootstrapped": True,
            },
            "overlay": {
                "show_translation": False,
                "show_peer_original": False,
                "calibration": {
                    "distance": 1.2,
                    "offset_y": -0.2,
                },
            },
            "desktop_audio": {
                "output_device": "Headphones (Loopback)",
                "vad_speech_threshold": 0.7,
                "vad_hangover_ms": 950,
                "vad_pre_roll_ms": 450,
            },
        }
    )

    assert settings.ui.overlay_enabled is False
    assert settings.overlay.show_translation is False
    assert settings.overlay.show_peer_original is False
    assert settings.overlay.calibration.distance == 1.2
    assert settings.overlay.calibration.offset_y == -0.2
    assert settings.ui.peer_translation_enabled is False
    assert settings.ui.integrated_context_enabled is True
    assert settings.desktop_audio.output_device == "Headphones (Loopback)"

    data = to_dict(settings)
    round_tripped = from_dict(data)

    assert "overlay_enabled" not in data["ui"]
    assert "peer_translation_enabled" not in data["ui"]
    assert "show_overlay_translation" not in data["ui"]
    assert "show_overlay_peer_original" not in data["ui"]
    assert data["overlay"]["show_translation"] is False
    assert data["overlay"]["show_peer_original"] is False
    assert data["overlay"]["calibration"]["distance"] == 1.2
    assert data["ui"]["integrated_context_bootstrapped"] is True
    assert data["desktop_audio"]["vad_hangover_ms"] == 950
    assert "overlay_calibration" not in data
    assert round_tripped.ui.overlay_enabled is False
    assert round_tripped.overlay.show_translation is False
    assert round_tripped.overlay.show_peer_original is False
    assert round_tripped.overlay.calibration.distance == 1.2
    assert round_tripped.ui.peer_translation_enabled is False


def test_overlay_enabled_false_stays_separate_from_overlay_display_serialization() -> None:
    settings = from_dict(
        {
            "ui": {"overlay_enabled": False},
            "overlay": {"show_translation": False},
        }
    )

    data = to_dict(settings)
    round_tripped = from_dict(data)

    assert "overlay_enabled" not in data["ui"]
    assert round_tripped.ui.overlay_enabled is False
    assert data["overlay"]["show_translation"] is False
    assert round_tripped.overlay.show_translation is False


def test_overlay_peer_presentation_refresh_burst_is_not_persisted_in_settings() -> None:
    settings = from_dict(
        {
            "overlay": {
                "debug_peer_refresh_burst": True,
                "peer_presentation_refresh_burst": False,
            }
        }
    )
    data = to_dict(settings)

    assert not hasattr(settings.overlay, "debug_peer_refresh_burst")
    assert not hasattr(settings.overlay, "peer_presentation_refresh_burst")
    assert "debug_peer_refresh_burst" not in data["overlay"]
    assert "peer_presentation_refresh_burst" not in data["overlay"]
    round_tripped = to_dict(from_dict(data))["overlay"]
    assert "debug_peer_refresh_burst" not in round_tripped
    assert "peer_presentation_refresh_burst" not in round_tripped


def test_desktop_audio_settings_round_trip_with_defaults() -> None:
    settings = from_dict({})

    assert settings.desktop_audio.output_device == ""
    assert settings.desktop_audio.vad_speech_threshold == 0.5
    assert settings.desktop_audio.vad_hangover_ms == 500
    assert settings.desktop_audio.vad_pre_roll_ms == 500


def test_desktop_audio_output_device_null_defaults_to_empty_string() -> None:
    settings = from_dict({"desktop_audio": {"output_device": None}})

    assert settings.desktop_audio.output_device == ""


def test_overlay_calibration_round_trips_with_defaults() -> None:
    settings = from_dict(
        {
            "overlay": {
                "calibration": {
                    "anchor": "spatial_locked",
                    "offset_x": 0.15,
                    "offset_y": -0.2,
                    "distance": 1.2,
                    "text_scale": 1.1,
                    "background_alpha": 0.4,
                }
            }
        }
    )

    assert settings.overlay.calibration.anchor == "spatial_locked"
    assert settings.overlay.calibration.distance == 1.2

    data = to_dict(settings)

    assert data["overlay"]["calibration"]["offset_x"] == 0.15
    assert data["overlay"]["calibration"]["background_alpha"] == 0.4
    assert from_dict(data).overlay.calibration.anchor == "spatial_locked"


def test_from_dict_accepts_legacy_overlay_display_and_calibration_shape() -> None:
    settings = from_dict(
        {
            "ui": {
                "show_overlay_translation": False,
                "show_overlay_peer_original": False,
            },
            "overlay_calibration": {
                "offset_x": 0.15,
                "distance": 1.2,
            },
        }
    )

    assert settings.overlay.show_translation is False
    assert settings.overlay.show_peer_original is False
    assert settings.overlay.calibration.offset_x == 0.15
    assert settings.overlay.calibration.distance == 1.2

    data = to_dict(settings)

    assert data["overlay"]["show_translation"] is False
    assert data["overlay"]["show_peer_original"] is False
    assert data["overlay"]["calibration"]["offset_x"] == 0.15
    assert data["overlay"]["calibration"]["distance"] == 1.2
    assert "show_overlay_translation" not in data["ui"]
    assert "show_overlay_peer_original" not in data["ui"]
    assert "overlay_calibration" not in data


def test_load_settings_migrates_legacy_desktop_vad_hangover_to_new_default(tmp_path) -> None:
    from puripuly_heart.config.settings import SETTINGS_SCHEMA_VERSION, load_settings

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "settings_version": 6,
                "desktop_audio": {
                    "vad_hangover_ms": 900,
                },
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(path)
    reloaded = legacy_projected_settings_file(path)

    assert settings.settings_version == SETTINGS_SCHEMA_VERSION
    assert settings.desktop_audio.vad_hangover_ms == 500
    assert reloaded["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert reloaded["desktop_audio"]["vad_hangover_ms"] == 500


def test_load_settings_migrates_desktop_vad_hangover_700_to_500(tmp_path) -> None:
    from puripuly_heart.config.settings import SETTINGS_SCHEMA_VERSION, load_settings

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "settings_version": 7,
                "desktop_audio": {
                    "vad_hangover_ms": 700,
                },
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(path)
    reloaded = legacy_projected_settings_file(path)

    assert settings.settings_version == SETTINGS_SCHEMA_VERSION
    assert settings.desktop_audio.vad_hangover_ms == 500
    assert reloaded["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert reloaded["desktop_audio"]["vad_hangover_ms"] == 500
