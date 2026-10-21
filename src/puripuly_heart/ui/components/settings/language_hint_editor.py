from __future__ import annotations

from collections.abc import Callable, Sequence

import flet as ft

from puripuly_heart.core.language import get_all_language_options
from puripuly_heart.ui.components.language_modal import LanguageModal
from puripuly_heart.ui.flet_runtime import (
    control_page,
    is_hover_active,
    update_control_if_mounted,
)
from puripuly_heart.ui.i18n import language_name, t
from puripuly_heart.ui.theme import (
    COLOR_DIVIDER,
    COLOR_ON_PRIMARY_CONTAINER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_CONTAINER,
    COLOR_SECONDARY,
    text_style,
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
        self._recent_codes: Callable[[], Sequence[str]] | None = None
        self._remember_recent: Callable[[str], None] | None = None

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
                text_style=text_style(size=20),
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
            text_style=text_style(size=20),
            overlay_color=ft.Colors.TRANSPARENT,
            animation_duration=0,
        )
        self.set_terms(list(self._terms))

    def bind_recent_languages(
        self,
        *,
        codes: Callable[[], Sequence[str]],
        remember: Callable[[str], None],
    ) -> None:
        """Use the same recent-language queue as the dashboard language modals."""
        self._recent_codes = codes
        self._remember_recent = remember

    def recent_language_codes(self) -> tuple[str, ...] | None:
        if self._recent_codes is None:
            return None
        return tuple(self._recent_codes())

    def set_terms(self, terms: list[str]) -> None:
        self._terms = list(terms)
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
        modal = LanguageModal(
            page=page,
            languages=languages,
            on_select=self._handle_select,
        )
        modal.open(current="", recent=self._recent_for_modal())

    def _recent_for_modal(self) -> list[str]:
        if self._recent_codes is None:
            return []
        known = {code for code, _name in get_all_language_options()}
        return [code for code in self._recent_codes() if code in known]

    def _handle_select(self, code: str) -> None:
        if self._remember_recent is not None:
            self._remember_recent(code)
        if self._on_add is not None:
            self._on_add(code)

    def _handle_remove(self, code: str) -> None:
        if self._on_remove is not None:
            self._on_remove(code)
