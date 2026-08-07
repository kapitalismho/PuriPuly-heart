from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import is_control_mounted
from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.i18n import get_locale, t
from puripuly_heart.ui.theme import (
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SURFACE,
)

_DIALOG_WIDTH = 450
_DIALOG_HEIGHT = 500
_CONTENT_SIZE = 386
_PERCENT_TEXT_SIZE = 96
_FAILURE_TEXT_SIZE = 24
_HINT_TEXT_SIZE = 28


def _clamp_level(value: float) -> float:
    level = max(0.0, min(1.0, float(value)))
    if level <= 1e-6:
        return 0.0
    return level


def _level_percent(value: float) -> int:
    return int(round(_clamp_level(value) * 100))


class MicrophoneTestDialog:
    """Minimal microphone-test modal with a large live percentage readout."""

    def __init__(
        self,
        page: ft.Page,
        *,
        on_close: Callable[[], None],
    ) -> None:
        self._page = page
        self._on_close = on_close
        self._dialog: ft.AlertDialog | None = None
        self._level_text: ft.Text | None = None
        self._hint_text: ft.Text | None = None
        self._level = 0.0
        self._failed = False
        self._is_open = False
        self._close_notified = False

    @property
    def dialog(self) -> ft.AlertDialog | None:
        return self._dialog

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        if self._is_open:
            return
        self._close_notified = False
        self._dialog = self._build_dialog()
        self._is_open = True
        self._page.show_dialog(self._dialog)

    def close(self, *, notify: bool = False) -> None:
        dialog = self._dialog
        if dialog is None:
            self._close_notified = True
            return
        was_open = self._is_open
        self._is_open = False
        if notify:
            self._notify_close_once()
        else:
            self._close_notified = True
        if was_open:
            pop_dialog = getattr(self._page, "pop_dialog", None)
            if callable(pop_dialog):
                pop_dialog()

    def reset(self) -> None:
        self._level = 0.0
        self._failed = False
        self._sync_text()

    def set_level(self, value: float) -> None:
        self._level = _clamp_level(value)
        self._failed = False
        self._sync_text()

    def show_failure(self) -> None:
        self._level = 0.0
        self._failed = True
        self._sync_text()

    def _build_dialog(self) -> ft.AlertDialog:
        self._level_text = ft.Text(
            self._text_value(),
            size=self._text_size(),
            weight=ft.FontWeight.BOLD,
            color=self._text_color(),
            text_align=ft.TextAlign.CENTER,
            font_family=font_for_language(get_locale()),
            semantics_label=t("settings.microphone_test.level_label"),
        )
        self._hint_text = ft.Text(
            t("settings.microphone_test.host_api_hint"),
            size=_HINT_TEXT_SIZE,
            color=COLOR_NEUTRAL_DARK,
            text_align=ft.TextAlign.CENTER,
            font_family=font_for_language(get_locale()),
        )

        modal_content = ft.Container(
            width=_DIALOG_WIDTH,
            height=_DIALOG_HEIGHT,
            padding=28,
            bgcolor=COLOR_SURFACE,
            border_radius=30,
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                controls=[
                    ft.Container(
                        width=_CONTENT_SIZE,
                        content=self._level_text,
                        alignment=ft.Alignment.CENTER,
                        bgcolor=ft.Colors.TRANSPARENT,
                        expand=True,
                    ),
                    self._hint_text,
                ],
                spacing=12,
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                tight=False,
            ),
        )

        return ft.AlertDialog(
            modal=False,
            content=modal_content,
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
            semantics_label=t("settings.microphone_test"),
            on_dismiss=self._handle_dismiss,
        )

    def _text_value(self) -> str:
        if self._failed:
            return t("settings.microphone_test.start_failed")
        return f"{_level_percent(self._level)}%"

    def _text_size(self) -> int:
        return _FAILURE_TEXT_SIZE if self._failed else _PERCENT_TEXT_SIZE

    def _text_color(self) -> str:
        return COLOR_NEUTRAL_DARK if self._failed else COLOR_PRIMARY

    def _sync_text(self) -> None:
        if self._level_text is None:
            return
        self._level_text.value = self._text_value()
        self._level_text.size = self._text_size()
        self._level_text.color = self._text_color()
        if not is_control_mounted(self._level_text):
            return
        try:
            self._level_text.update()
        except AssertionError as exc:
            if "Control must be added" not in str(exc):
                raise

    def _handle_dismiss(self, _event) -> None:  # noqa: ANN001
        if not self._is_open:
            return
        self._is_open = False
        self._notify_close_once()

    def _notify_close_once(self) -> None:
        if self._close_notified:
            return
        self._close_notified = True
        self._on_close()
