from __future__ import annotations

from typing import Callable

import flet as ft

from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.i18n import get_locale, t
from puripuly_heart.ui.theme import COLOR_NEUTRAL, COLOR_ON_BACKGROUND, COLOR_SURFACE


class TelemetryConsentDialog:
    def __init__(
        self, page: ft.Page, *, on_allow: Callable[[], None], on_decline: Callable[[], None]
    ):
        self._page = page
        self._on_allow = on_allow
        self._on_decline = on_decline
        self._dialog: ft.AlertDialog | None = None

    def open(self) -> None:
        font = font_for_language(get_locale())
        content = ft.Container(
            width=560,
            padding=ft.padding.all(28),
            bgcolor=COLOR_SURFACE,
            border_radius=24,
            content=ft.Column(
                controls=[
                    ft.Text(
                        t("telemetry.consent.title"),
                        size=24,
                        weight=ft.FontWeight.BOLD,
                        color=COLOR_NEUTRAL,
                        font_family=font,
                    ),
                    ft.Text(
                        t("telemetry.consent.body"),
                        size=16,
                        color=COLOR_ON_BACKGROUND,
                        font_family=font,
                    ),
                    ft.Text(
                        t("telemetry.consent.excludes"),
                        size=14,
                        color=COLOR_NEUTRAL,
                        font_family=font,
                    ),
                    ft.Row(
                        controls=[
                            ft.TextButton(
                                text=t("telemetry.consent.decline"),
                                on_click=lambda _e: self._choose(False),
                            ),
                            ft.FilledButton(
                                text=t("telemetry.consent.allow"),
                                on_click=lambda _e: self._choose(True),
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.END,
                    ),
                ],
                spacing=18,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
        )
        self._dialog = ft.AlertDialog(
            modal=True,
            content=content,
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
            surface_tint_color=ft.Colors.TRANSPARENT,
        )
        self._page.open(self._dialog)

    def _choose(self, allow: bool) -> None:
        if self._dialog is not None:
            self._page.close(self._dialog)
        if allow:
            self._on_allow()
        else:
            self._on_decline()
