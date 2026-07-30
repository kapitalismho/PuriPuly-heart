from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS


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
            color=FOUNDATION_DESIGN_TOKENS.palette.neutral,
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
                    ft.ControlState.DEFAULT: FOUNDATION_DESIGN_TOKENS.palette.on_background,
                    ft.ControlState.HOVERED: FOUNDATION_DESIGN_TOKENS.palette.primary,
                    ft.ControlState.DISABLED: FOUNDATION_DESIGN_TOKENS.palette.neutral,
                },
                bgcolor=ft.Colors.TRANSPARENT,
                text_style=ft.TextStyle(
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
                    FOUNDATION_DESIGN_TOKENS.palette.primary,
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
                color=FOUNDATION_DESIGN_TOKENS.palette.on_primary_container,
            ),
            bgcolor=FOUNDATION_DESIGN_TOKENS.palette.primary_container,
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
