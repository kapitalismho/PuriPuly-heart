from __future__ import annotations

import math
from dataclasses import dataclass

DESKTOP_FLET_MIN_WIDTH = 480
DESKTOP_FLET_MIN_HEIGHT = 160
DESKTOP_FLET_DEFAULT_TEXT_SCALE = 1.0
DESKTOP_FLET_MIN_TEXT_SCALE = 0.75
DESKTOP_FLET_MAX_TEXT_SCALE = 1.5
DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA = 0.6
DESKTOP_FLET_MIN_BACKGROUND_ALPHA = 0.0
DESKTOP_FLET_MAX_BACKGROUND_ALPHA = 1.0
DESKTOP_FLET_MIN_OUTLINE_WIDTH = 0.5
DESKTOP_FLET_MAX_OUTLINE_WIDTH = 8.0
DESKTOP_FLET_SIZE_PRESET_ORDER = ("tiny", "xsmall", "small", "medium", "large", "xlarge")
DESKTOP_FLET_SIZE_PRESET_DISPLAY_ORDER = tuple(reversed(DESKTOP_FLET_SIZE_PRESET_ORDER))
DESKTOP_FLET_DEFAULT_SIZE_PRESET = "medium"
DESKTOP_FLET_SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "tiny": (640, 160),
    "xsmall": (960, 240),
    "small": (1152, 288),
    "medium": (1344, 336),
    "large": (1600, 400),
    "xlarge": (1792, 448),
}
DESKTOP_FLET_DEFAULT_WIDTH = DESKTOP_FLET_SIZE_PRESETS[DESKTOP_FLET_DEFAULT_SIZE_PRESET][0]
DESKTOP_FLET_DEFAULT_HEIGHT = DESKTOP_FLET_SIZE_PRESETS[DESKTOP_FLET_DEFAULT_SIZE_PRESET][1]


def _normalized_range(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        number = float(value)
    except OverflowError:
        return default
    if not math.isfinite(number):
        return default
    return min(max(value, minimum), maximum)


@dataclass(slots=True, init=False)
class DesktopFletOverlayVisualSettings:
    background_alpha: float = DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA

    def __init__(
        self,
        text_scale: object = None,
        background_alpha: object = DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
        outline_width: object = None,
    ) -> None:
        _ = (text_scale, outline_width)
        self.background_alpha = background_alpha

    def validate(self) -> None:
        self.background_alpha = _normalized_range(
            self.background_alpha,
            default=DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
            minimum=DESKTOP_FLET_MIN_BACKGROUND_ALPHA,
            maximum=DESKTOP_FLET_MAX_BACKGROUND_ALPHA,
        )

    @property
    def text_scale(self) -> float:
        return DESKTOP_FLET_DEFAULT_TEXT_SCALE

    @text_scale.setter
    def text_scale(self, _value: object) -> None:
        return

    @property
    def outline_width(self) -> None:
        return None

    @outline_width.setter
    def outline_width(self, _value: object) -> None:
        return
