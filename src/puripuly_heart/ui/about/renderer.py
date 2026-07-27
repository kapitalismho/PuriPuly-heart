from __future__ import annotations

import flet as ft

from puripuly_heart.ui.about.contract import AboutSurfaceRegions, AboutSurfaceSlots

ABOUT_ROW_SPACING = 16


def compose_about_pair_row(first: ft.Control, second: ft.Control) -> ft.Container:
    return ft.Container(
        content=ft.Row(
            controls=[
                ft.Container(content=first, expand=True),
                ft.Container(content=second, expand=True),
            ],
            spacing=ABOUT_ROW_SPACING,
            expand=True,
        ),
    )


def compose_about_surface(slots: AboutSurfaceSlots) -> AboutSurfaceRegions:
    header_row = compose_about_pair_row(slots.app_name_card, slots.version_card)
    top_row = compose_about_pair_row(slots.credits_card, slots.inspired_by_card)
    return AboutSurfaceRegions(
        controls=(
            header_row,
            top_row,
            slots.special_thanks_card,
            slots.licenses_card,
        ),
        header_row=header_row,
        top_row=top_row,
    )


__all__ = [
    "ABOUT_ROW_SPACING",
    "compose_about_pair_row",
    "compose_about_surface",
]
