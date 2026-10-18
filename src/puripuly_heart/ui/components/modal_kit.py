from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import is_hover_active, update_control_if_mounted
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)

MODAL_WIDTH = 600
MODAL_WIDE_WIDTH = 880
MODAL_HEIGHT = 700
MODAL_PADDING = 32
MODAL_RADIUS = 28
MODAL_COLUMN_SPACING = 32

TITLE_TEXT_SIZE = 24
SECTION_TEXT_SIZE = 18
CARD_LABEL_SIZE = 20
CARD_DETAIL_SIZE = 16

CARD_HEIGHT = 110
CARD_RADIUS = 16
CARD_PADDING = 24
CARD_SPACING = 12

_DISABLED_OPACITY = 0.35
_SELECTED_DETAIL_OPACITY = 0.8


def build_modal_surface(
    title: str,
    body: ft.Control,
    *,
    width: int = MODAL_WIDE_WIDTH,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(
                    title,
                    size=TITLE_TEXT_SIZE,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_SECONDARY,
                ),
                ft.Container(height=16),
                body,
            ],
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
        width=width,
        height=MODAL_HEIGHT,
        padding=ft.Padding.all(MODAL_PADDING),
        bgcolor=COLOR_SURFACE,
        border_radius=MODAL_RADIUS,
    )


def build_modal_dialog(
    content: ft.Control,
    *,
    semantics_label: str | None = None,
    on_dismiss: Callable[[object], None] | None = None,
) -> ft.AlertDialog:
    return ft.AlertDialog(
        modal=False,
        content=content,
        content_padding=0,
        bgcolor=ft.Colors.TRANSPARENT,
        semantics_label=semantics_label,
        on_dismiss=on_dismiss,
    )


def build_section_header(label: str) -> ft.Text:
    return ft.Text(
        label,
        size=SECTION_TEXT_SIZE,
        weight=ft.FontWeight.BOLD,
        color=COLOR_SECONDARY,
        max_lines=1,
        overflow=ft.TextOverflow.ELLIPSIS,
    )


class ModalCard(ft.Container):
    def __init__(
        self,
        label: str,
        *,
        on_click: Callable[[], None],
        detail: str | None = None,
        selected: bool = False,
        enabled: bool = True,
    ) -> None:
        self.label_text = ft.Text(
            label,
            size=CARD_LABEL_SIZE,
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self.detail_text = ft.Text(
            detail or "",
            size=CARD_DETAIL_SIZE,
            text_align=ft.TextAlign.CENTER,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            visible=detail is not None,
        )
        self._on_click = on_click
        self._selected = selected
        self._enabled = enabled
        self._hovered = False
        super().__init__(
            content=ft.Column(
                controls=[self.label_text, self.detail_text],
                spacing=8,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            height=CARD_HEIGHT,
            padding=ft.Padding.symmetric(horizontal=CARD_PADDING, vertical=12),
            border_radius=CARD_RADIUS,
            alignment=ft.Alignment.CENTER,
            on_click=self._handle_click,
            on_hover=self._handle_hover,
            animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
        )
        self._paint()

    @property
    def selected(self) -> bool:
        return self._selected

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_detail(self, detail: str | None) -> None:
        self.detail_text.value = detail or ""
        self.detail_text.visible = detail is not None

    def set_state(self, *, selected: bool | None = None, enabled: bool | None = None) -> None:
        if selected is not None:
            self._selected = selected
        if enabled is not None:
            self._enabled = enabled
        self._paint()

    def _paint(self) -> None:
        self.disabled = not self._enabled
        if self._selected and self._enabled:
            self.bgcolor = COLOR_PRIMARY
            self.label_text.color = ft.Colors.WHITE
            self.detail_text.color = ft.Colors.with_opacity(
                _SELECTED_DETAIL_OPACITY, ft.Colors.WHITE
            )
            return
        self.bgcolor = COLOR_BACKGROUND
        if not self._enabled:
            muted = ft.Colors.with_opacity(_DISABLED_OPACITY, COLOR_NEUTRAL_DARK)
            self.label_text.color = muted
            self.detail_text.color = muted
            return
        color = COLOR_PRIMARY if self._hovered else COLOR_NEUTRAL_DARK
        self.label_text.color = color
        self.detail_text.color = color

    def _handle_click(self, _event: object) -> None:
        if self._enabled:
            self._on_click()

    def _handle_hover(self, event: ft.ControlEvent) -> None:
        hovered = is_hover_active(event)
        if hovered == self._hovered:
            return
        self._hovered = hovered
        self._paint()
        update_control_if_mounted(self)


__all__ = [
    "CARD_DETAIL_SIZE",
    "CARD_HEIGHT",
    "CARD_LABEL_SIZE",
    "CARD_PADDING",
    "CARD_RADIUS",
    "CARD_SPACING",
    "MODAL_COLUMN_SPACING",
    "MODAL_HEIGHT",
    "MODAL_PADDING",
    "MODAL_RADIUS",
    "MODAL_WIDE_WIDTH",
    "MODAL_WIDTH",
    "SECTION_TEXT_SIZE",
    "TITLE_TEXT_SIZE",
    "ModalCard",
    "build_modal_dialog",
    "build_modal_surface",
    "build_section_header",
]
