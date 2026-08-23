from __future__ import annotations

import json
import math

from puripuly_heart.config.settings import (
    DESKTOP_FLET_SIZE_PRESETS,
    SETTINGS_SCHEMA_VERSION,
    from_dict,
    to_dict,
)


def test_overlay_settings_desktop_flet_defaults_serialize_canonical_shape() -> None:
    settings = from_dict({})

    assert SETTINGS_SCHEMA_VERSION == 25
    assert settings.settings_version == 25
    assert settings.overlay.target == "steamvr"
    assert settings.overlay.desktop_flet.size_preset == "medium"
    assert settings.overlay.desktop_flet.position.x is None
    assert settings.overlay.desktop_flet.position.y is None
    assert settings.overlay.desktop_flet.locked is False
    assert settings.overlay.desktop_flet.swap_caption_languages is False
    assert settings.overlay.desktop_flet.visual.background_alpha == 0.6

    data = to_dict(settings)

    assert data["overlay"]["target"] == "steamvr"
    assert data["overlay"]["desktop_flet"] == {
        "size_preset": "medium",
        "position": {"x": None, "y": None},
        "swap_caption_languages": False,
        "visual": {"background_alpha": 0.6},
    }
    assert "locked" not in data["overlay"]["desktop_flet"]


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
    settings = from_dict({"overlay": {"desktop_flet": {"size_preset": "tiny"}}})

    data = to_dict(settings)
    round_tripped = from_dict(data)

    assert data["overlay"]["desktop_flet"]["size_preset"] == "tiny"
    assert round_tripped.overlay.desktop_flet.size_preset == "tiny"
    assert round_tripped.overlay.desktop_flet.bounds.width == 640
    assert round_tripped.overlay.desktop_flet.bounds.height == 160


def test_overlay_settings_desktop_flet_legacy_locked_loads_startup_safe_and_is_not_serialized() -> (
    None
):
    settings = from_dict(
        {
            "overlay": {
                "target": "desktop",
                "desktop_flet": {
                    "size_preset": "large",
                    "position": {"x": 320, "y": 720},
                    "locked": True,
                    "visual": {"background_alpha": 0.45},
                },
            }
        }
    )

    data = to_dict(settings)
    round_tripped = from_dict(data)

    assert data["overlay"]["target"] == "desktop"
    assert data["overlay"]["desktop_flet"] == {
        "size_preset": "large",
        "position": {"x": 320, "y": 720},
        "swap_caption_languages": False,
        "visual": {"background_alpha": 0.45},
    }
    assert "locked" not in data["overlay"]["desktop_flet"]
    assert round_tripped.overlay.target == "desktop"
    assert round_tripped.overlay.desktop_flet.size_preset == "large"
    assert round_tripped.overlay.desktop_flet.position.x == 320
    assert round_tripped.overlay.desktop_flet.position.y == 720
    assert settings.overlay.desktop_flet.locked is False
    assert round_tripped.overlay.desktop_flet.locked is False
    assert round_tripped.overlay.desktop_flet.visual.background_alpha == 0.45


def test_overlay_settings_target_repairs_invalid_values_to_steamvr() -> None:
    settings = from_dict({"overlay": {"target": "sidecar"}})

    assert settings.overlay.target == "steamvr"
    assert to_dict(settings)["overlay"]["target"] == "steamvr"


def test_overlay_settings_desktop_flet_invalid_canonical_values_repair() -> None:
    settings = from_dict(
        {
            "overlay": {
                "desktop_flet": {
                    "size_preset": "huge",
                    "position": {
                        "x": 100,
                        "y": math.inf,
                    },
                    "locked": "yes",
                    "swap_caption_languages": "yes",
                    "visual": {"background_alpha": True},
                }
            }
        }
    )

    assert settings.overlay.desktop_flet.size_preset == "medium"
    assert settings.overlay.desktop_flet.position.x is None
    assert settings.overlay.desktop_flet.position.y is None
    assert settings.overlay.desktop_flet.locked is False
    assert settings.overlay.desktop_flet.swap_caption_languages is False
    assert settings.overlay.desktop_flet.visual.background_alpha == 0.6

    missing_and_non_finite = from_dict(
        {
            "overlay": {
                "desktop_flet": {
                    "size_preset": "xlarge",
                    "position": {"x": False},
                    "visual": {"background_alpha": math.nan},
                }
            }
        }
    )

    assert missing_and_non_finite.overlay.desktop_flet.size_preset == "xlarge"
    assert missing_and_non_finite.overlay.desktop_flet.position.x is None
    assert missing_and_non_finite.overlay.desktop_flet.position.y is None
    assert missing_and_non_finite.overlay.desktop_flet.visual.background_alpha == 0.6


def test_overlay_settings_desktop_flet_background_alpha_clamps() -> None:
    clamped_high = from_dict({"overlay": {"desktop_flet": {"visual": {"background_alpha": 1.5}}}})
    clamped_low = from_dict({"overlay": {"desktop_flet": {"visual": {"background_alpha": -0.25}}}})

    assert clamped_high.overlay.desktop_flet.visual.background_alpha == 1.0
    assert clamped_low.overlay.desktop_flet.visual.background_alpha == 0.0


def test_overlay_settings_desktop_flet_legacy_bounds_and_visual_migrate_to_canonical() -> None:
    settings = from_dict(
        {
            "overlay": {
                "desktop_flet": {
                    "bounds": {"x": 320, "y": 720, "width": 1600, "height": 400},
                    "visual": {
                        "text_scale": 1.25,
                        "background_alpha": 0.45,
                        "outline_width": 2.5,
                    },
                }
            }
        }
    )

    data = to_dict(settings)

    assert settings.overlay.desktop_flet.size_preset == "large"
    assert settings.overlay.desktop_flet.position.x == 320
    assert settings.overlay.desktop_flet.position.y == 720
    assert settings.overlay.desktop_flet.visual.background_alpha == 0.45
    assert data["overlay"]["desktop_flet"] == {
        "size_preset": "large",
        "position": {"x": 320, "y": 720},
        "swap_caption_languages": False,
        "visual": {"background_alpha": 0.45},
    }
    assert "locked" not in data["overlay"]["desktop_flet"]
    assert "bounds" not in data["overlay"]["desktop_flet"]
    assert "text_scale" not in data["overlay"]["desktop_flet"]["visual"]
    assert "outline_width" not in data["overlay"]["desktop_flet"]["visual"]


def test_overlay_settings_desktop_flet_legacy_bounds_repair_and_tie_breaking() -> None:
    invalid_dimensions = from_dict(
        {
            "overlay": {
                "desktop_flet": {
                    "bounds": {"x": 100, "y": math.inf, "width": math.nan, "height": 400}
                }
            }
        }
    )
    tied_with_medium = from_dict(
        {
            "overlay": {
                "desktop_flet": {
                    "bounds": {
                        "width": (2 * 1152 * 1344) / (1152 + 1344),
                        "height": (2 * 288 * 336) / (288 + 336),
                    }
                }
            }
        }
    )
    tied_without_medium = from_dict(
        {
            "overlay": {
                "desktop_flet": {
                    "bounds": {
                        "width": (2 * 1600 * 1792) / (1600 + 1792),
                        "height": (2 * 400 * 448) / (400 + 448),
                    }
                }
            }
        }
    )

    assert invalid_dimensions.overlay.desktop_flet.size_preset == "medium"
    assert invalid_dimensions.overlay.desktop_flet.position.x is None
    assert invalid_dimensions.overlay.desktop_flet.position.y is None
    assert tied_with_medium.overlay.desktop_flet.size_preset == "medium"
    assert tied_without_medium.overlay.desktop_flet.size_preset == "large"


def test_settings_json_allow_nan_false_with_repaired_desktop_flet_overlay_values() -> None:
    settings = from_dict(
        {
            "overlay": {
                "target": "desktop",
                "desktop_flet": {
                    "bounds": {"x": math.nan, "y": 24, "width": math.inf, "height": -1},
                    "visual": {
                        "text_scale": -math.inf,
                        "background_alpha": math.nan,
                        "outline_width": -1,
                    },
                },
            }
        }
    )

    json.dumps(to_dict(settings), allow_nan=False)


def test_overlay_settings_desktop_flet_swap_caption_languages_round_trips() -> None:
    settings = from_dict({"overlay": {"desktop_flet": {"swap_caption_languages": True}}})

    data = to_dict(settings)
    round_tripped = from_dict(data)

    assert settings.overlay.desktop_flet.swap_caption_languages is True
    assert data["overlay"]["desktop_flet"]["swap_caption_languages"] is True
    assert round_tripped.overlay.desktop_flet.swap_caption_languages is True
