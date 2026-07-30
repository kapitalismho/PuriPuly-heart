from typing import Callable

import flet as ft

from puripuly_heart.ui.components.glow import create_glow_stack
from puripuly_heart.ui.flet_runtime import is_hover_active, update_control_if_mounted
from puripuly_heart.ui.theme import (
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    get_card_shadow,
)

# CJK (Chinese, Japanese, Korean) characters start at this Unicode point
_CJK_START = 0x3000
_CAPTION_TEXT_SIZE = 16
_ARROW_SIZE_DELTA = 4


def _weighted_len(text: str) -> int:
    """Calculate weighted length for CJK-aware font sizing."""
    return sum(2 if ord(c) >= _CJK_START else 1 for c in text)


def _row_text_size(source: str, target: str) -> int:
    total_len = _weighted_len(source) + _weighted_len(target)
    if total_len < 16:
        return 34
    if total_len < 22:
        return 30
    if total_len < 28:
        return 26
    if total_len < 34:
        return 22
    if total_len < 42:
        return 18
    return 16


class _LanguageRow(ft.Container):
    def __init__(
        self,
        *,
        label: str,
        on_source_click: Callable[[], None],
        on_target_click: Callable[[], None],
        on_swap_click: Callable[[], None] | None = None,
    ):
        self._on_source_click = on_source_click
        self._on_target_click = on_target_click
        self._on_swap_click = on_swap_click

        self._label_text = ft.Text(
            label,
            size=_CAPTION_TEXT_SIZE,
            weight=ft.FontWeight.W_600,
            color=COLOR_SECONDARY,
            no_wrap=True,
        )
        self._source_text = ft.Text(
            "",
            size=34,
            weight=ft.FontWeight.BOLD,
            color=COLOR_NEUTRAL_DARK,
            text_align=ft.TextAlign.CENTER,
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._target_text = ft.Text(
            "",
            size=34,
            weight=ft.FontWeight.BOLD,
            color=COLOR_NEUTRAL_DARK,
            text_align=ft.TextAlign.CENTER,
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._arrow_icon = ft.Icon(
            icon=ft.Icons.ARROW_RIGHT_ALT,
            size=34 + _ARROW_SIZE_DELTA,
            color=COLOR_SECONDARY,
        )

        caption = ft.Container(
            content=self._label_text,
            alignment=ft.Alignment.TOP_LEFT,
            padding=ft.Padding.only(left=12),
        )
        self._arrow = ft.Container(
            content=self._arrow_icon,
            padding=ft.Padding.symmetric(horizontal=6, vertical=8),
            border_radius=14,
            on_click=lambda _: self._on_swap_click() if self._on_swap_click else None,
            on_hover=self._on_arrow_hover,
        )
        self._source_btn = ft.Container(
            content=self._source_text,
            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            border_radius=14,
            bgcolor=ft.Colors.TRANSPARENT,
            on_hover=self._on_source_hover,
            on_click=lambda _: self._on_source_click(),
            expand=True,
            alignment=ft.Alignment.CENTER,
        )
        self._target_btn = ft.Container(
            content=self._target_text,
            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            border_radius=14,
            bgcolor=ft.Colors.TRANSPARENT,
            on_hover=self._on_target_hover,
            on_click=lambda _: self._on_target_click(),
            expand=True,
            alignment=ft.Alignment.CENTER,
        )

        pair = ft.Container(
            content=ft.Row(
                [self._source_btn, self._arrow, self._target_btn],
                spacing=10,
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            alignment=ft.Alignment.CENTER,
        )

        row_content = ft.Column(
            [caption, pair],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        super().__init__(
            content=row_content,
            padding=ft.Padding.symmetric(vertical=4),
        )

    def _on_source_hover(self, e):
        self._source_text.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_NEUTRAL_DARK
        self._source_text.update()

    def _on_target_hover(self, e):
        self._target_text.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_NEUTRAL_DARK
        self._target_text.update()

    def _on_arrow_hover(self, e):
        self._arrow_icon.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_SECONDARY
        self._arrow_icon.update()

    def set_label(self, label: str) -> None:
        self._label_text.value = label
        update_control_if_mounted(self._label_text)

    def set_languages(self, source: str, target: str, *, size: int | None = None) -> None:
        resolved_size = size if size is not None else _row_text_size(source, target)
        self._source_text.size = resolved_size
        self._target_text.size = resolved_size
        self._arrow_icon.size = resolved_size + _ARROW_SIZE_DELTA
        self._source_text.value = source
        self._target_text.value = target

        update_control_if_mounted(self._source_text)
        update_control_if_mounted(self._target_text)
        update_control_if_mounted(self._arrow_icon)


class LanguageCard(ft.Container):
    """Language display card with explicit self and peer rows."""

    def __init__(
        self,
        on_self_source_click: Callable[[], None],
        on_self_target_click: Callable[[], None],
        on_self_swap_click: Callable[[], None] | None = None,
        on_peer_source_click: Callable[[], None] = lambda: None,
        on_peer_target_click: Callable[[], None] = lambda: None,
        on_peer_swap_click: Callable[[], None] | None = None,
    ):
        self._self_row = _LanguageRow(
            label="",
            on_source_click=on_self_source_click,
            on_target_click=on_self_target_click,
            on_swap_click=on_self_swap_click,
        )
        self._peer_row = _LanguageRow(
            label="",
            on_source_click=on_peer_source_click,
            on_target_click=on_peer_target_click,
            on_swap_click=on_peer_swap_click,
        )

        content_with_glow = create_glow_stack(
            ft.Container(
                content=ft.Column(
                    [self._self_row, self._peer_row],
                    spacing=20,
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                expand=True,
                alignment=ft.Alignment.CENTER,
                padding=16,
            )
        )

        super().__init__(
            content=content_with_glow,
            bgcolor=COLOR_SURFACE,
            border_radius=16,
            border=ft.Border.all(1, ft.Colors.with_opacity(0.4, ft.Colors.WHITE)),
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            shadow=get_card_shadow(),
        )

    def set_row_labels(self, self_label: str, peer_label: str) -> None:
        self._self_row.set_label(self_label)
        self._peer_row.set_label(peer_label)

    def set_languages(
        self,
        self_source: str,
        self_target: str,
        peer_source: str,
        peer_target: str,
    ):
        unified_size = min(
            _row_text_size(self_source, self_target),
            _row_text_size(peer_source, peer_target),
        )
        self._self_row.set_languages(self_source, self_target, size=unified_size)
        self._peer_row.set_languages(peer_source, peer_target, size=unified_size)
