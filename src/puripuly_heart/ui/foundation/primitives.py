from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS
from puripuly_heart.ui.theme import (
    COLOR_NEUTRAL,
    COLOR_ON_BACKGROUND,
    COLOR_ON_PRIMARY_CONTAINER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_CONTAINER,
    text_style,
)


class FoundationCard(SharedCardWrapper):
    def __init__(self, content: ft.Control, *, width: float | int | None = None) -> None:
        super().__init__(
            content,
            expand=False,
            height=None,
            padding=FOUNDATION_DESIGN_TOKENS.spacing.card,
        )
        self.width = width


class FoundationSectionTitle(ft.Text):
    def __init__(self, value: str) -> None:
        super().__init__(
            value,
            size=FOUNDATION_DESIGN_TOKENS.typography.title,
            weight=ft.FontWeight.BOLD,
            color=COLOR_NEUTRAL,
        )


class FoundationActionButton(ft.TextButton):
    def __init__(
        self,
        label: str,
        *,
        on_click: Callable[[object], None] | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            content=label,
            on_click=on_click,
            disabled=disabled,
            style=ft.ButtonStyle(
                color={
                    ft.ControlState.DEFAULT: COLOR_ON_BACKGROUND,
                    ft.ControlState.HOVERED: COLOR_PRIMARY,
                    ft.ControlState.DISABLED: COLOR_NEUTRAL,
                },
                bgcolor=ft.Colors.TRANSPARENT,
                text_style=text_style(
                    size=FOUNDATION_DESIGN_TOKENS.typography.label,
                    weight=ft.FontWeight.W_600,
                ),
                padding=ft.Padding.symmetric(
                    horizontal=FOUNDATION_DESIGN_TOKENS.spacing.inline,
                    vertical=FOUNDATION_DESIGN_TOKENS.spacing.compact,
                ),
                shape=ft.RoundedRectangleBorder(radius=FOUNDATION_DESIGN_TOKENS.radius.control),
                overlay_color=ft.Colors.with_opacity(
                    0.08,
                    COLOR_PRIMARY,
                ),
                animation_duration=0,
            ),
        )


class FoundationStatusPill(ft.Container):
    def __init__(self, label: str) -> None:
        super().__init__(
            content=ft.Text(
                label,
                size=FOUNDATION_DESIGN_TOKENS.typography.label,
                weight=ft.FontWeight.W_600,
                color=COLOR_ON_PRIMARY_CONTAINER,
            ),
            bgcolor=COLOR_PRIMARY_CONTAINER,
            border_radius=FOUNDATION_DESIGN_TOKENS.radius.control,
            padding=ft.Padding.symmetric(
                horizontal=FOUNDATION_DESIGN_TOKENS.spacing.inline,
                vertical=FOUNDATION_DESIGN_TOKENS.spacing.compact,
            ),
        )


__all__ = [
    "FoundationActionButton",
    "FoundationCard",
    "FoundationSectionTitle",
    "FoundationStatusPill",
]
