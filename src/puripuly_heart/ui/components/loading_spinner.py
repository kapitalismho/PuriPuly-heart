from __future__ import annotations

import flet as ft

from puripuly_heart.ui.theme import COLOR_PRIMARY


def create_button_spinner(
    icon_size: int,
    *,
    semantics_label: str | None = None,
    color: str = COLOR_PRIMARY,
) -> ft.ProgressRing:
    return ft.ProgressRing(
        width=icon_size * 0.7,
        height=icon_size * 0.7,
        stroke_width=max(3, icon_size * 0.06),
        color=color,
        visible=False,
        semantics_label=semantics_label,
    )


def create_section_spinner(
    size: int = 32,
    *,
    stroke_width: int = 3,
    color: str | None = None,
) -> ft.ProgressRing:
    kwargs: dict[str, object] = {
        "width": size,
        "height": size,
        "stroke_width": stroke_width,
    }
    if color is not None:
        kwargs["color"] = color
    return ft.ProgressRing(**kwargs)
