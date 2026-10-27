from __future__ import annotations

import math

from puripuly_heart.config.desktop_overlay_values import (
    DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
    DESKTOP_FLET_DEFAULT_TEXT_SCALE,
    DesktopFletOverlayVisualSettings,
)


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
