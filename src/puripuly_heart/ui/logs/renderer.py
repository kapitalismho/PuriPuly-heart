from __future__ import annotations

import flet as ft

from puripuly_heart.ui.components.glow import GLOW_CARD, create_glow_stack
from puripuly_heart.ui.logs.contract import LogsSurfaceRegions, LogsSurfaceSlots
from puripuly_heart.ui.theme import COLOR_SURFACE, get_card_shadow

LOGS_HEADER_BUTTON_SPACING = 4
LOGS_CARD_BORDER_RADIUS = 16


def compose_logs_surface(slots: LogsSurfaceSlots) -> LogsSurfaceRegions:
    header_button_row = ft.Row(
        controls=[slots.folder_button, slots.mode_button, slots.conversation_button],
        spacing=LOGS_HEADER_BUTTON_SPACING,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    header = ft.Container(
        content=ft.Row(
            controls=[
                slots.title,
                ft.Container(expand=True),
                header_button_row,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding.only(left=16, right=8, top=8, bottom=0),
    )

    log_scroll = ft.Column(
        controls=[
            ft.Container(
                content=slots.log_text,
                padding=ft.Padding.only(left=16, right=16, top=8, bottom=16),
            )
        ],
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )

    card_content = ft.Column(
        controls=[header, log_scroll],
        spacing=0,
        expand=True,
    )

    card = ft.Container(
        content=create_glow_stack(
            ft.Container(content=card_content, expand=True),
            config=GLOW_CARD,
        ),
        bgcolor=COLOR_SURFACE,
        border_radius=LOGS_CARD_BORDER_RADIUS,
        border=ft.Border.all(1, ft.Colors.with_opacity(0.4, ft.Colors.WHITE)),
        expand=True,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        shadow=get_card_shadow(),
    )

    return LogsSurfaceRegions(
        root=card,
        card=card,
        header=header,
        header_button_row=header_button_row,
        log_scroll=log_scroll,
    )


__all__ = [
    "LOGS_CARD_BORDER_RADIUS",
    "LOGS_HEADER_BUTTON_SPACING",
    "compose_logs_surface",
]
