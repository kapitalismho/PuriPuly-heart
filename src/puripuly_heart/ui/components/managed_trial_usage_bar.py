from __future__ import annotations

import flet as ft

from puripuly_heart.ui.flet_runtime import update_control_if_mounted
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_ON_ACCENT,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TRANS_TONAL,
)

_FIELD_HEIGHT = 72
_TRACK_RADIUS = 12
_TEXT_SIZE = 18
_TEXT_HORIZONTAL_PADDING = 16


def _update_control_if_mounted(control: ft.Control) -> None:
    update_control_if_mounted(control)


def _text_shader(percent: int | None) -> ft.LinearGradient:
    split = 0.0 if percent is None else percent / 100
    return ft.LinearGradient(
        begin=ft.Alignment.CENTER_LEFT,
        end=ft.Alignment.CENTER_RIGHT,
        colors=[COLOR_ON_ACCENT, COLOR_ON_ACCENT, COLOR_SECONDARY, COLOR_SECONDARY],
        stops=[0.0, split, split, 1.0],
    )


class ManagedTrialUsageBar(ft.Row):
    def __init__(self, percent: int | None = None) -> None:
        self._percent: int | None = None
        self._count: tuple[int, int] | None = None
        self._count_unknown = False
        self._fill_segment = ft.Container(
            height=_FIELD_HEIGHT,
            bgcolor=COLOR_PRIMARY,
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
            color=COLOR_ON_ACCENT,
            text_align=ft.TextAlign.RIGHT,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._text_mask = ft.ShaderMask(
            shader=_text_shader(None),
            blend_mode=ft.BlendMode.SRC_IN,
            content=ft.Container(
                content=self._remaining_text,
                alignment=ft.Alignment.CENTER_RIGHT,
                padding=ft.Padding.symmetric(horizontal=_TEXT_HORIZONTAL_PADDING),
            ),
            left=0,
            right=0,
            top=0,
            bottom=0,
        )
        self._track = ft.Container(
            expand=True,
            height=_FIELD_HEIGHT,
            bgcolor=COLOR_TRANS_TONAL,
            border_radius=_TRACK_RADIUS,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            content=ft.Stack(
                controls=[
                    ft.Container(
                        content=self._fill_segments,
                        left=0,
                        right=0,
                        top=0,
                        bottom=0,
                    ),
                    self._text_mask,
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
        self._text_mask.shader = _text_shader(self._percent)
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
        self._count = None
        self._count_unknown = False
        if percent is None:
            self._percent = None
        else:
            self._percent = max(0, min(100, int(percent)))
        self._sync()

    def set_count(self, remaining: int | None, allowance: int | None) -> None:
        if remaining is None or allowance is None or allowance <= 0:
            self._count = None
            self._count_unknown = True
            self._percent = None
        else:
            remaining = max(0, min(allowance, int(remaining)))
            self._count = (remaining, int(allowance))
            self._count_unknown = False
            self._percent = round(remaining * 100 / allowance)
        self._sync()

    def apply_locale(self) -> None:
        self._sync()

    def repaint_dynamic_controls(self) -> None:
        for control in (
            self._fill_segments,
            self._fill_segment,
            self._empty_segment,
            self._text_mask,
            self._remaining_text,
        ):
            _update_control_if_mounted(control)

    def _sync(self) -> None:
        self._sync_fill_segments()
        if self._count is not None:
            self._remaining_text.value = t(
                "settings.photo.remaining.value",
                remaining=self._count[0],
                allowance=self._count[1],
            )
        elif self._count_unknown:
            self._remaining_text.value = "—"
        elif self._percent is None:
            self._remaining_text.value = t("settings.managed_trial_usage.remaining_placeholder")
        else:
            self._remaining_text.value = t(
                "settings.managed_trial_usage.remaining",
                percent=self._percent,
            )
