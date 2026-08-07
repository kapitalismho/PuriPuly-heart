from __future__ import annotations

import re
from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import (
    FILL_PARENT_WIDTH,
    is_hover_active,
    update_control_if_mounted,
)
from puripuly_heart.ui.theme import (
    COLOR_DIVIDER,
    COLOR_NEUTRAL_DARK,
    COLOR_ON_PRIMARY_CONTAINER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_CONTAINER,
    COLOR_SECONDARY,
)

_CHIP_TERM_WIDTH = 220
_CHIP_COMPACT_CHAR_LIMIT = 24
_CHIP_RADIUS = 999
_CHIP_TEXT_SIZE = 22
_CHIP_HORIZONTAL_PADDING = 20
_CHIP_VERTICAL_PADDING = 14
_INPUT_FIELD_RADIUS = 12
_TOKEN_SPLIT_RE = re.compile(r"\s+")


def _update_control_if_mounted(control: ft.Control) -> None:
    update_control_if_mounted(control)


class CustomVocabularyTagEditor(ft.Column):
    """Presentation component for editing Speech Recognition Hint token chips."""

    def __init__(
        self,
        *,
        on_add_terms: Callable[[list[str]], None] | None = None,
        on_remove_term: Callable[[str], None] | None = None,
    ) -> None:
        self.on_add_terms = on_add_terms
        self.on_remove_term = on_remove_term
        self._terms: list[str] = []
        self._remove_label_template = ""
        self._add_label = ""

        self._empty_text = ft.Text(
            "",
            size=14,
            color=COLOR_SECONDARY,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
            visible=False,
        )
        self._chips_wrap = ft.Row(
            controls=[],
            spacing=6,
            run_spacing=8,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            visible=False,
        )

        self._input_field = ft.TextField(
            hint_text="",
            multiline=False,
            max_lines=1,
            width=FILL_PARENT_WIDTH,
            border_radius=_INPUT_FIELD_RADIUS,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            text_size=28,
            color=COLOR_NEUTRAL_DARK,
            on_change=self._handle_input_change,
            on_submit=self._handle_input_submit,
            on_blur=self._handle_input_blur,
        )

        super().__init__(
            controls=[self._input_field],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        self.set_terms([])

    def set_terms(self, terms: list[str]) -> None:
        """Re-render hint chips and empty state from the provided terms."""
        self._terms = list(terms)
        chip_controls = [self._build_chip(term) for term in self._terms]
        self._chips_wrap.controls = chip_controls
        self._chips_wrap.visible = bool(chip_controls)
        self.controls = (
            [self._chips_wrap, self._input_field] if chip_controls else [self._input_field]
        )
        self._empty_text.visible = False
        _update_control_if_mounted(self._chips_wrap)
        _update_control_if_mounted(self._empty_text)
        _update_control_if_mounted(self)

    def set_placeholder(self, text: str) -> None:
        """Accept legacy placeholder copy; token input intentionally stays quiet."""
        _ = text
        self._input_field.hint_text = ""
        _update_control_if_mounted(self._input_field)

    def set_empty_text(self, text: str) -> None:
        """Update empty-state copy."""
        self._empty_text.value = text
        _update_control_if_mounted(self._empty_text)

    def set_remove_label_template(self, template: str) -> None:
        """Accept legacy remove copy; chips intentionally render no hover tooltip."""
        self._remove_label_template = template

    def set_add_label(self, text: str) -> None:
        """Accept legacy add-button copy; token input no longer renders a button."""
        self._add_label = text

    def clear_input(self) -> None:
        """Clear unsubmitted add-input text."""
        self._input_field.value = ""
        _update_control_if_mounted(self._input_field)

    def _term_text_width(self, term: str) -> int | None:
        if len(term) <= _CHIP_COMPACT_CHAR_LIMIT:
            return None
        return _CHIP_TERM_WIDTH

    def _build_chip(self, term: str) -> ft.Container:
        term_text = ft.Text(
            term,
            size=_CHIP_TEXT_SIZE,
            color=COLOR_ON_PRIMARY_CONTAINER,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            no_wrap=True,
            width=self._term_text_width(term),
            semantics_label=term,
        )
        return ft.Container(
            data=term,
            bgcolor=COLOR_PRIMARY_CONTAINER,
            border=ft.Border.all(1, COLOR_DIVIDER),
            border_radius=_CHIP_RADIUS,
            padding=ft.Padding.only(
                left=_CHIP_HORIZONTAL_PADDING,
                right=_CHIP_HORIZONTAL_PADDING,
                top=_CHIP_VERTICAL_PADDING,
                bottom=_CHIP_VERTICAL_PADDING,
            ),
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            content=term_text,
            on_click=lambda _event, visible_term=term: self._handle_remove(visible_term),
            on_hover=self._handle_chip_hover,
        )

    def _handle_chip_hover(self, event: ft.ControlEvent) -> None:
        chip = event.control
        term_text = chip.content
        hovered = is_hover_active(event)
        chip.bgcolor = COLOR_PRIMARY if hovered else COLOR_PRIMARY_CONTAINER
        chip.border = ft.Border.all(1, COLOR_PRIMARY if hovered else COLOR_DIVIDER)
        term_text.color = ft.Colors.WHITE if hovered else COLOR_ON_PRIMARY_CONTAINER
        _update_control_if_mounted(chip)

    def _handle_input_change(self, _event) -> None:
        raw_value = self._input_field.value or ""
        if not raw_value or not _TOKEN_SPLIT_RE.search(raw_value):
            return
        self._commit_input_value()

    def _handle_input_submit(self, _event) -> None:
        self._commit_input_value()

    def _handle_input_blur(self, _event) -> None:
        self._commit_input_value()

    def _commit_input_value(self) -> None:
        raw_value = self._input_field.value or ""
        if raw_value == "" or self.on_add_terms is None:
            return

        raw_terms = [part for part in _TOKEN_SPLIT_RE.split(raw_value.strip()) if part]
        self.clear_input()
        if raw_terms:
            self.on_add_terms(raw_terms)

    def _handle_remove(self, term: str) -> None:
        if self.on_remove_term is not None:
            self.on_remove_term(term)
