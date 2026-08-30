from typing import Callable, Sequence

import flet as ft

from puripuly_heart.ui.flet_runtime import is_hover_active, update_control_if_mounted
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS
from puripuly_heart.ui.theme import (
    COLOR_DISPLAY_SOURCE,
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)

# CJK (Chinese, Japanese, Korean) characters start at this Unicode point
_CJK_START = 0x3000

LANGUAGE_CARD_HORIZONTAL_PADDING = FOUNDATION_DESIGN_TOKENS.spacing.card
LANGUAGE_CARD_VERTICAL_PADDING = FOUNDATION_DESIGN_TOKENS.spacing.inline
LANGUAGE_ROW_GAP = 20
LANGUAGE_HOVER_ANIMATION_MS = 150
LANGUAGE_SLOT_HORIZONTAL_PADDING = FOUNDATION_DESIGN_TOKENS.spacing.inline
LANGUAGE_SLOT_VERTICAL_PADDING = FOUNDATION_DESIGN_TOKENS.spacing.compact
LANGUAGE_SLOT_RADIUS = FOUNDATION_DESIGN_TOKENS.radius.control
LANGUAGE_ROW_ICON_GAP = FOUNDATION_DESIGN_TOKENS.spacing.compact
LANGUAGE_ROW_ICON_SIZE_DELTA = -6
LANGUAGE_TARGET_LINE_GAP = 2
LANGUAGE_ARROW_SIZE_DELTA = 4

LANGUAGE_ROW_WIDTH = 600 - 2 * LANGUAGE_CARD_HORIZONTAL_PADDING
LANGUAGE_ROW_SPACING = 10
LANGUAGE_ARROW_HORIZONTAL_PADDING = 6
LANGUAGE_CARD_HEIGHT = 338
LANGUAGE_LINE_HEIGHT_RATIO = 1.3
LANGUAGE_WEIGHTED_UNIT_WIDTH_RATIO = 0.5
LANGUAGE_SIZE_CANDIDATES: tuple[int, ...] = (40, 36, 32, 28, 24, 20, 16)

SELF_ROW_ICON = ft.Icons.MIC
PEER_ROW_ICON = ft.Icons.RECORD_VOICE_OVER
SECONDARY_TARGET_ADD_ICON = ft.Icons.ADD


def _weighted_len(text: str) -> int:
    """Calculate weighted length for CJK-aware font sizing."""
    return sum(2 if ord(c) >= _CJK_START else 1 for c in text)


def _slot_line_height(size: int) -> int:
    return int(2 * LANGUAGE_SLOT_VERTICAL_PADDING + round(size * LANGUAGE_LINE_HEIGHT_RATIO))


def _row_icon_column_width(size: int) -> float:
    return size + LANGUAGE_ROW_ICON_SIZE_DELTA


def _arrow_column_width(size: int) -> float:
    return 2 * LANGUAGE_ARROW_HORIZONTAL_PADDING + (size + LANGUAGE_ARROW_SIZE_DELTA)


def _slot_width(size: int) -> float:
    remaining = (
        LANGUAGE_ROW_WIDTH
        - _row_icon_column_width(size)
        - _arrow_column_width(size)
        - 3 * LANGUAGE_ROW_SPACING
    )
    return remaining / 2


def _slot_text_width(size: int) -> float:
    return _slot_width(size) - 2 * LANGUAGE_SLOT_HORIZONTAL_PADDING


def _text_fits(text: str, size: int, available_width: float) -> bool:
    return _weighted_len(text) * size * LANGUAGE_WEIGHTED_UNIT_WIDTH_RATIO <= available_width


def _card_height(size: int, *, target_lines: int) -> int:
    line = _slot_line_height(size)
    self_pair = target_lines * line + (target_lines - 1) * LANGUAGE_TARGET_LINE_GAP
    return 2 * LANGUAGE_CARD_VERTICAL_PADDING + self_pair + LANGUAGE_ROW_GAP + line


def _row_text_size(
    *,
    source_texts: Sequence[str],
    target_texts: Sequence[str],
    target_lines: int,
) -> int:
    for size in LANGUAGE_SIZE_CANDIDATES:
        if _card_height(size, target_lines=target_lines) > LANGUAGE_CARD_HEIGHT:
            continue
        slot_width = _slot_text_width(size)
        if all(_text_fits(text, size, slot_width) for text in source_texts) and all(
            _text_fits(text, size, slot_width) for text in target_texts
        ):
            return size
    return LANGUAGE_SIZE_CANDIDATES[-1]


class _LanguageSlot(ft.Container):
    """One tappable language value."""

    def __init__(
        self,
        *,
        on_click: Callable[[], None],
    ) -> None:
        self._placeholder_revealed = False
        self._base_color = COLOR_NEUTRAL_DARK
        self._placeholder_base_color = COLOR_DISPLAY_SOURCE
        self._text = ft.Text(
            "",
            size=LANGUAGE_SIZE_CANDIDATES[0],
            weight=ft.FontWeight.BOLD,
            color=self._base_color,
            text_align=ft.TextAlign.CENTER,
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._placeholder_icon = ft.Icon(
            icon=SECONDARY_TARGET_ADD_ICON,
            size=LANGUAGE_SIZE_CANDIDATES[0] + LANGUAGE_ROW_ICON_SIZE_DELTA,
            color=self._placeholder_base_color,
            visible=False,
            opacity=0.0,
            animate_opacity=ft.Animation(LANGUAGE_HOVER_ANIMATION_MS, ft.AnimationCurve.EASE_OUT),
        )

        super().__init__(
            content=ft.Row(
                [self._text, self._placeholder_icon],
                spacing=LANGUAGE_ROW_ICON_GAP,
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                tight=True,
            ),
            padding=ft.Padding.symmetric(
                horizontal=LANGUAGE_SLOT_HORIZONTAL_PADDING,
                vertical=LANGUAGE_SLOT_VERTICAL_PADDING,
            ),
            border_radius=LANGUAGE_SLOT_RADIUS,
            bgcolor=ft.Colors.TRANSPARENT,
            alignment=ft.Alignment.CENTER,
            height=_slot_line_height(LANGUAGE_SIZE_CANDIDATES[0]),
            on_click=lambda _event: on_click(),
            on_hover=self._on_hover,
        )

    def _on_hover(self, event: ft.ControlEvent) -> None:
        hovered = is_hover_active(event)
        self._text.color = COLOR_PRIMARY if hovered else self._base_color
        self._placeholder_icon.color = COLOR_PRIMARY if hovered else self._placeholder_base_color
        update_control_if_mounted(self._text)
        update_control_if_mounted(self._placeholder_icon)

    def set_value(self, value: str, *, size: int) -> None:
        icon_size = size + LANGUAGE_ROW_ICON_SIZE_DELTA
        self.height = _slot_line_height(size)
        self._text.size = size
        self._text.value = value
        self._text.visible = bool(value)
        self._placeholder_icon.size = icon_size
        self._placeholder_icon.visible = not value
        self._placeholder_icon.opacity = self._placeholder_target_opacity()
        update_control_if_mounted(self._text)
        update_control_if_mounted(self._placeholder_icon)
        update_control_if_mounted(self)

    def _placeholder_target_opacity(self) -> float:
        return 1.0 if (not self._text.value and self._placeholder_revealed) else 0.0

    def reveal_placeholder(self, revealed: bool) -> None:
        self._placeholder_revealed = bool(revealed)
        target = self._placeholder_target_opacity()
        if self._placeholder_icon.opacity == target:
            return
        self._placeholder_icon.opacity = target
        update_control_if_mounted(self._placeholder_icon)


class _LanguageRow(ft.Container):
    def __init__(
        self,
        *,
        icon: str,
        on_source_click: Callable[[], None],
        on_target_click: Callable[[], None],
        on_swap_click: Callable[[], None] | None = None,
        on_secondary_target_click: Callable[[], None] | None = None,
    ):
        self._on_swap_click = on_swap_click

        self._row_icon = ft.Icon(
            icon=icon,
            size=LANGUAGE_SIZE_CANDIDATES[0] + LANGUAGE_ROW_ICON_SIZE_DELTA,
            color=COLOR_DISPLAY_SOURCE,
        )
        self._icon_holder = ft.Container(
            content=self._row_icon,
            height=_slot_line_height(LANGUAGE_SIZE_CANDIDATES[0]),
            alignment=ft.Alignment.CENTER_LEFT,
        )

        self._source_slot = _LanguageSlot(on_click=on_source_click)
        self._target_slot = _LanguageSlot(on_click=on_target_click)
        self._secondary_slot = (
            _LanguageSlot(on_click=on_secondary_target_click)
            if on_secondary_target_click is not None
            else None
        )

        self._arrow_icon = ft.Icon(
            icon=ft.Icons.ARROW_RIGHT_ALT,
            size=LANGUAGE_SIZE_CANDIDATES[0] + LANGUAGE_ARROW_SIZE_DELTA,
            color=COLOR_SECONDARY,
        )
        self._arrow = ft.Container(
            content=self._arrow_icon,
            padding=ft.Padding.symmetric(
                horizontal=LANGUAGE_ARROW_HORIZONTAL_PADDING,
                vertical=LANGUAGE_SLOT_VERTICAL_PADDING,
            ),
            border_radius=LANGUAGE_SLOT_RADIUS,
            height=_slot_line_height(LANGUAGE_SIZE_CANDIDATES[0]),
            alignment=ft.Alignment.CENTER,
            on_click=lambda _: self._on_swap_click() if self._on_swap_click else None,
            on_hover=self._on_arrow_hover,
        )

        target_lines: list[ft.Control] = [self._target_slot]
        if self._secondary_slot is not None:
            target_lines.append(self._secondary_slot)
        self._target_column = ft.Column(
            target_lines,
            spacing=LANGUAGE_TARGET_LINE_GAP,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            tight=True,
            expand=True,
        )
        self._source_holder = ft.Container(content=self._source_slot, expand=True)
        self._target_holder = ft.Container(
            content=self._target_column,
            expand=True,
            on_hover=self._on_target_area_hover,
        )

        super().__init__(
            content=ft.Row(
                [self._icon_holder, self._source_holder, self._arrow, self._target_holder],
                spacing=LANGUAGE_ROW_SPACING,
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
        )

    def _on_target_area_hover(self, event: ft.ControlEvent) -> None:
        if self._secondary_slot is None:
            return
        self._secondary_slot.reveal_placeholder(is_hover_active(event))

    def _on_arrow_hover(self, e):
        self._arrow_icon.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_SECONDARY
        update_control_if_mounted(self._arrow_icon)

    def set_languages(
        self,
        source: str,
        target: str,
        *,
        size: int,
        secondary_target: str = "",
    ) -> None:
        self._source_slot.set_value(source, size=size)
        self._target_slot.set_value(target, size=size)
        if self._secondary_slot is not None:
            self._secondary_slot.set_value(secondary_target, size=size)
        self._row_icon.size = size + LANGUAGE_ROW_ICON_SIZE_DELTA
        self._icon_holder.height = _slot_line_height(size)
        self._arrow_icon.size = size + LANGUAGE_ARROW_SIZE_DELTA
        self._arrow.height = _slot_line_height(size)
        update_control_if_mounted(self._row_icon)
        update_control_if_mounted(self._icon_holder)
        update_control_if_mounted(self._arrow_icon)
        update_control_if_mounted(self._arrow)


class LanguageCard(ft.Container):
    """Language card with a self row that can carry a secondary target."""

    def __init__(
        self,
        on_self_source_click: Callable[[], None],
        on_self_target_click: Callable[[], None],
        on_self_swap_click: Callable[[], None] | None = None,
        on_peer_source_click: Callable[[], None] = lambda: None,
        on_peer_target_click: Callable[[], None] = lambda: None,
        on_peer_swap_click: Callable[[], None] | None = None,
        on_self_secondary_target_click: Callable[[], None] = lambda: None,
    ):
        self._self_row = _LanguageRow(
            icon=SELF_ROW_ICON,
            on_source_click=on_self_source_click,
            on_target_click=on_self_target_click,
            on_swap_click=on_self_swap_click,
            on_secondary_target_click=on_self_secondary_target_click,
        )
        self._peer_row = _LanguageRow(
            icon=PEER_ROW_ICON,
            on_source_click=on_peer_source_click,
            on_target_click=on_peer_target_click,
            on_swap_click=on_peer_swap_click,
        )

        content_container = ft.Container(
            content=ft.Column(
                [self._self_row, self._peer_row],
                spacing=LANGUAGE_ROW_GAP,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            expand=True,
            alignment=ft.Alignment.CENTER,
            padding=ft.Padding.symmetric(
                horizontal=LANGUAGE_CARD_HORIZONTAL_PADDING,
                vertical=LANGUAGE_CARD_VERTICAL_PADDING,
            ),
        )

        super().__init__(
            content=content_container,
            bgcolor=COLOR_SURFACE,
            border_radius=FOUNDATION_DESIGN_TOKENS.radius.card,
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )

    def set_languages(
        self,
        self_source: str,
        self_target: str,
        peer_source: str,
        peer_target: str,
        self_secondary_target: str = "",
    ):
        size = _row_text_size(
            source_texts=(self_source, peer_source),
            target_texts=(self_target, self_secondary_target, peer_target),
            target_lines=2,
        )
        self._self_row.set_languages(
            self_source,
            self_target,
            size=size,
            secondary_target=self_secondary_target,
        )
        self._peer_row.set_languages(peer_source, peer_target, size=size)
