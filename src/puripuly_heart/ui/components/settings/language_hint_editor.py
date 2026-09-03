from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.core.language import get_all_language_options
from puripuly_heart.ui.components.language_modal import LanguageModal
from puripuly_heart.ui.flet_runtime import (
    control_page,
    is_hover_active,
    update_control_if_mounted,
)
from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.i18n import get_locale, language_name, t
from puripuly_heart.ui.theme import (
    COLOR_DIVIDER,
    COLOR_ON_PRIMARY_CONTAINER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_CONTAINER,
    COLOR_SECONDARY,
)

_CHIP_RADIUS = 999
_CHIP_TEXT_SIZE = 22
_CHIP_HORIZONTAL_PADDING = 20
_CHIP_VERTICAL_PADDING = 14


def _update_control_if_mounted(control: ft.Control) -> None:
    update_control_if_mounted(control)


class LanguageHintEditor(ft.Column):
    """Editor for selecting language detection hints via a language modal.

    Displays selected languages as removable chips and provides a button
    that opens a :class:`LanguageModal` for adding new hints.
    """

    def __init__(
        self,
        *,
        on_add: Callable[[str], None] | None = None,
        on_remove: Callable[[str], None] | None = None,
    ) -> None:
        self._on_add = on_add
        self._on_remove = on_remove
        self._terms: list[str] = []
        self._recent: list[str] = []

        self._chips_wrap = ft.Row(
            controls=[],
            spacing=6,
            run_spacing=8,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            visible=False,
        )

        self._add_button = ft.TextButton(
            content=t("settings.peer_auto_languages.add"),
            on_click=self._open_modal,
            style=ft.ButtonStyle(
                color={
                    ft.ControlState.HOVERED: COLOR_PRIMARY,
                    ft.ControlState.DEFAULT: COLOR_SECONDARY,
                },
                text_style=ft.TextStyle(
                    size=20,
                    font_family=font_for_language(get_locale()),
                ),
                overlay_color=ft.Colors.TRANSPARENT,
                animation_duration=0,
            ),
        )
        self._add_button_row = ft.Row(
            controls=[self._add_button],
            alignment=ft.MainAxisAlignment.END,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        super().__init__(
            controls=[self._chips_wrap, self._add_button_row],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def apply_locale(self) -> None:
        self._add_button.content = t("settings.peer_auto_languages.add")
        self._add_button.style = ft.ButtonStyle(
            color={
                ft.ControlState.HOVERED: COLOR_PRIMARY,
                ft.ControlState.DEFAULT: COLOR_SECONDARY,
            },
            text_style=ft.TextStyle(
                size=20,
                font_family=font_for_language(get_locale()),
            ),
            overlay_color=ft.Colors.TRANSPARENT,
            animation_duration=0,
        )
        self.set_terms(list(self._terms))

    def set_terms(self, terms: list[str]) -> None:
        self._terms = list(terms)
        for code in self._terms:
            if code not in self._recent:
                self._recent.insert(0, code)
        self._recent = self._recent[:6]
        chip_controls = [self._build_chip(code) for code in self._terms]
        self._chips_wrap.controls = chip_controls
        self._chips_wrap.visible = bool(chip_controls)
        _update_control_if_mounted(self._chips_wrap)
        _update_control_if_mounted(self)

    def _build_chip(self, code: str) -> ft.Container:
        label = language_name(code) or code
        term_text = ft.Text(
            label,
            size=_CHIP_TEXT_SIZE,
            color=COLOR_ON_PRIMARY_CONTAINER,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            no_wrap=True,
            semantics_label=label,
        )
        return ft.Container(
            data=code,
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
            on_click=lambda _event, visible_code=code: self._handle_remove(visible_code),
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

    def _open_modal(self, _event: ft.ControlEvent | None = None) -> None:
        page = control_page(self)
        if page is None:
            return
        languages = get_all_language_options()
        lang_codes = dict(languages)
        recent = [code for code in self._recent if code in lang_codes]
        modal = LanguageModal(
            page=page,
            languages=languages,
            on_select=self._handle_select,
        )
        modal.open(current="", recent=recent)

    def _handle_select(self, code: str) -> None:
        if code in self._recent:
            self._recent.remove(code)
        self._recent.insert(0, code)
        self._recent = self._recent[:6]
        if self._on_add is not None:
            self._on_add(code)

    def _handle_remove(self, code: str) -> None:
        if self._on_remove is not None:
            self._on_remove(code)
