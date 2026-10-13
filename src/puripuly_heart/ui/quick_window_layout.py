from __future__ import annotations

DEFAULT_QUICK_WIDTH = 420
DEFAULT_QUICK_HEIGHT = 680


def clamp_window_bounds(values: dict, *, work_area=(0, 0, 1920, 1080)) -> dict:
    left, top, right, bottom = work_area
    available_width = max(1, right - left)
    available_height = max(1, bottom - top)
    requested_width = int(values.get("width", DEFAULT_QUICK_WIDTH))
    requested_height = int(values.get("height", DEFAULT_QUICK_HEIGHT))
    if (requested_width, requested_height) == (480, 520):
        requested_width, requested_height = DEFAULT_QUICK_WIDTH, DEFAULT_QUICK_HEIGHT
    width = min(available_width, max(360, requested_width))
    height = min(available_height, max(300, requested_height))
    return dict(
        width=width,
        height=height,
        left=max(left, min(int(values.get("left", left + 40)), right - width)),
        top=max(top, min(int(values.get("top", top + 40)), bottom - height)),
    )
