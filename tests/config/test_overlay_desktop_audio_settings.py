from __future__ import annotations

from dataclasses import replace

from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, CaptureTargetIntent


def test_peer_translation_eula_acceptance_defaults_false() -> None:
    settings = AppSettingsVNext()
    data = serialization.to_dict(settings)

    assert settings.state.peer_translation.eula_accepted is False
    assert data["state"]["peer_translation"]["eula_accepted"] is False


def test_peer_translation_eula_acceptance_round_trips() -> None:
    current = AppSettingsVNext()
    settings = replace(
        current,
        state=replace(
            current.state,
            peer_translation=replace(current.state.peer_translation, eula_accepted=True),
        ),
    )
    restored = serialization.from_dict(serialization.to_dict(settings))

    assert restored.state.peer_translation.eula_accepted is True


def test_overlay_display_preferences_round_trip_in_shared_overlay_section() -> None:
    current = AppSettingsVNext()
    settings = replace(
        current,
        intent=replace(
            current.intent,
            overlay=replace(
                current.intent.overlay,
                show_translation=False,
                show_peer_original=False,
                calibration=replace(
                    current.intent.overlay.calibration,
                    distance=1.2,
                    offset_y=-0.2,
                ),
            ),
            desktop_audio=replace(
                current.intent.desktop_audio,
                output_device="Headphones (Loopback)",
                capture_target=CaptureTargetIntent.named_output_device("Headphones (Loopback)"),
                vad_speech_threshold=0.7,
                vad_hangover_ms=950,
                vad_pre_roll_ms=450,
            ),
            integrated_context=replace(current.intent.integrated_context, enabled=True),
        ),
        state=replace(
            current.state,
            integrated_context=replace(current.state.integrated_context, bootstrapped=True),
        ),
    )

    data = serialization.to_dict(settings)
    round_tripped = serialization.from_dict(data)

    assert "overlay_enabled" not in data["intent"]["ui"]
    assert "peer_translation_enabled" not in data["intent"]["ui"]
    assert data["intent"]["overlay"]["show_translation"] is False
    assert data["intent"]["overlay"]["show_peer_original"] is False
    assert data["intent"]["overlay"]["calibration"]["distance"] == 1.2
    assert data["state"]["integrated_context"]["bootstrapped"] is True
    assert data["intent"]["desktop_audio"]["vad_hangover_ms"] == 950
    assert "overlay_calibration" not in data
    assert round_tripped.intent.overlay.show_translation is False
    assert round_tripped.intent.overlay.show_peer_original is False
    assert round_tripped.intent.overlay.calibration.distance == 1.2
    assert round_tripped.intent.desktop_audio.output_device == "Headphones (Loopback)"


def test_overlay_peer_presentation_refresh_burst_is_not_persisted_in_settings() -> None:
    settings = AppSettingsVNext()
    data = serialization.to_dict(settings)

    assert not hasattr(settings.intent.overlay, "debug_peer_refresh_burst")
    assert not hasattr(settings.intent.overlay, "peer_presentation_refresh_burst")
    assert "debug_peer_refresh_burst" not in data["intent"]["overlay"]
    assert "peer_presentation_refresh_burst" not in data["intent"]["overlay"]


def test_desktop_audio_settings_round_trip_with_defaults() -> None:
    settings = AppSettingsVNext()

    assert settings.intent.desktop_audio.output_device == ""
    assert settings.intent.desktop_audio.vad_speech_threshold == 0.5
    assert settings.intent.desktop_audio.vad_hangover_ms == 500
    assert settings.intent.desktop_audio.vad_pre_roll_ms == 500


def test_overlay_calibration_round_trips_with_defaults() -> None:
    current = AppSettingsVNext()
    settings = replace(
        current,
        intent=replace(
            current.intent,
            overlay=replace(
                current.intent.overlay,
                calibration=replace(
                    current.intent.overlay.calibration,
                    anchor="spatial_locked",
                    offset_x=0.15,
                    offset_y=-0.2,
                    distance=1.2,
                    text_scale=1.1,
                    background_alpha=0.4,
                ),
            ),
        ),
    )

    data = serialization.to_dict(settings)
    restored = serialization.from_dict(data)

    assert restored.intent.overlay.calibration.anchor == "spatial_locked"
    assert restored.intent.overlay.calibration.distance == 1.2
    assert data["intent"]["overlay"]["calibration"]["offset_x"] == 0.15
    assert data["intent"]["overlay"]["calibration"]["background_alpha"] == 0.4
