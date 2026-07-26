from __future__ import annotations

import flet as ft

from puripuly_heart.ui.flet_runtime import update_control_if_mounted
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_DIVIDER,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY_CONTAINER,
    COLOR_SURFACE,
    COLOR_SURFACE_TONAL,
)

_FIELD_HEIGHT = 72
_TRACK_RADIUS = 12
_TEXT_SIZE = 18
_TEXT_HORIZONTAL_PADDING = 16


def _update_control_if_mounted(control: ft.Control) -> None:
    update_control_if_mounted(control)


class ManagedTrialUsageBar(ft.Row):
    def __init__(self, percent: int | None = None) -> None:
        self._percent: int | None = None
        self._fill_segment = ft.Container(
            height=_FIELD_HEIGHT,
            bgcolor=COLOR_PRIMARY_CONTAINER,
            border_radius=_TRACK_RADIUS,
        )
        self._empty_segment = ft.Container(
            height=_FIELD_HEIGHT,
            bgcolor=ft.Colors.TRANSPARENT,
        )
        self._fill_segments = ft.Row(
            controls=[self._empty_segment],
            spacing=0,
            expand=True,
        )
        self._remaining_text = ft.Text(
            "",
            size=_TEXT_SIZE,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_BACKGROUND,
            text_align=ft.TextAlign.RIGHT,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._track = ft.Container(
            expand=True,
            height=_FIELD_HEIGHT,
            bgcolor=COLOR_SURFACE,
            border=ft.Border.all(1, COLOR_DIVIDER),
            border_radius=_TRACK_RADIUS,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            content=ft.Stack(
                controls=[
                    ft.Container(
                        content=self._fill_segments,
                        bgcolor=COLOR_SURFACE_TONAL,
                        left=0,
                        right=0,
                        top=0,
                        bottom=0,
                    ),
                    ft.Container(
                        content=self._remaining_text,
                        alignment=ft.Alignment.CENTER_RIGHT,
                        padding=ft.Padding.symmetric(horizontal=_TEXT_HORIZONTAL_PADDING),
                        left=0,
                        right=0,
                        top=0,
                        bottom=0,
                    ),
                ],
                expand=True,
            ),
        )
        super().__init__(
            controls=[self._track],
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.set_percent(percent)

    @property
    def percent(self) -> int | None:
        return self._percent

    def _sync_fill_segments(self) -> None:
        if self._percent is None or self._percent <= 0:
            self._empty_segment.expand = 1
            self._fill_segments.controls = [self._empty_segment]
            return

        if self._percent >= 100:
            self._fill_segment.expand = 1
            self._fill_segments.controls = [self._fill_segment]
            return

        self._fill_segment.expand = self._percent
        self._empty_segment.expand = 100 - self._percent
        self._fill_segments.controls = [self._fill_segment, self._empty_segment]

    def set_percent(self, percent: int | None) -> None:
        if percent is None:
            self._percent = None
        else:
            self._percent = max(0, min(100, int(percent)))
        self._sync()

    def apply_locale(self) -> None:
        self._sync()

    def repaint_dynamic_controls(self) -> None:
        for control in (
            self._fill_segments,
            self._fill_segment,
            self._empty_segment,
            self._remaining_text,
        ):
            _update_control_if_mounted(control)

    def _sync(self) -> None:
        self._sync_fill_segments()
        if self._percent is None:
            self._remaining_text.value = t("settings.managed_trial_usage.remaining_placeholder")
        else:
            self._remaining_text.value = t(
                "settings.managed_trial_usage.remaining",
                percent=self._percent,
            )
