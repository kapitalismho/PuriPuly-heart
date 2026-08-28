from __future__ import annotations

import math

from puripuly_heart.app.wiring import create_desktop_overlay_policy
from puripuly_heart.config import settings
from puripuly_heart.config.desktop_overlay_values import (
    DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
    DESKTOP_FLET_DEFAULT_SIZE_PRESET,
    DESKTOP_FLET_DEFAULT_TEXT_SCALE,
    DESKTOP_FLET_MIN_HEIGHT,
    DESKTOP_FLET_MIN_WIDTH,
    DESKTOP_FLET_SIZE_PRESETS,
    DesktopFletOverlayVisualSettings,
)


def test_desktop_overlay_values_remain_legacy_settings_compatible() -> None:
    assert settings.DESKTOP_FLET_SIZE_PRESETS is DESKTOP_FLET_SIZE_PRESETS
    assert settings.DesktopFletOverlayVisualSettings is DesktopFletOverlayVisualSettings


def test_desktop_overlay_visual_values_preserve_validation_behavior() -> None:
    invalid = DesktopFletOverlayVisualSettings(
        text_scale=1.25,
        background_alpha=math.nan,
        outline_width=2.5,
    )
    invalid.validate()
    assert invalid.text_scale == DESKTOP_FLET_DEFAULT_TEXT_SCALE
    assert invalid.background_alpha == DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA
    assert invalid.outline_width is None

    clamped_high = DesktopFletOverlayVisualSettings(background_alpha=1.5)
    clamped_low = DesktopFletOverlayVisualSettings(background_alpha=-0.25)
    clamped_high.validate()
    clamped_low.validate()
    assert clamped_high.background_alpha == 1.0
    assert clamped_low.background_alpha == 0.0

    integer_zero = DesktopFletOverlayVisualSettings(background_alpha=0)
    integer_one = DesktopFletOverlayVisualSettings(background_alpha=1)
    integer_zero.validate()
    integer_one.validate()
    assert integer_zero.background_alpha == 0
    assert type(integer_zero.background_alpha) is int
    assert integer_one.background_alpha == 1
    assert type(integer_one.background_alpha) is int


def test_desktop_overlay_production_policy_uses_canonical_values() -> None:
    policy = create_desktop_overlay_policy()

    assert policy.minimum_width == DESKTOP_FLET_MIN_WIDTH
    assert policy.minimum_height == DESKTOP_FLET_MIN_HEIGHT
    assert policy.default_text_scale == DESKTOP_FLET_DEFAULT_TEXT_SCALE
    assert policy.default_background_alpha == DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA
    assert policy.default_size_preset == DESKTOP_FLET_DEFAULT_SIZE_PRESET
    assert policy.size_presets is DESKTOP_FLET_SIZE_PRESETS
