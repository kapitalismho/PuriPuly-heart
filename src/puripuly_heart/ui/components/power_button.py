from typing import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import update_control_if_mounted
from puripuly_heart.ui.theme import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    COLOR_TRANS_TONAL,
    COLOR_WARNING,
    get_card_shadow,
)


class PowerButton(ft.Container):
    """STT/TRANS toggle button with ON/OFF/Warning states."""

    def __init__(
        self,
        label: str,
        icon: str,
        on_click: Callable[[], None],
        icon_size: int = 80,
        label_size: int = 32,
        color_on: str | None = None,
    ):
        self._label = label
        self._icon = icon
        self._on_click = on_click
        self._is_on = False
        self._needs_key = False
        self._is_starting = False
        self._color_on = color_on if color_on is not None else COLOR_PRIMARY

        self._icon_control = ft.Icon(icon=icon, size=icon_size, color=COLOR_SECONDARY)
        self._progress_control = ft.ProgressRing(
            width=icon_size * 0.7,
            height=icon_size * 0.7,
            stroke_width=max(3, icon_size * 0.06),
            color=COLOR_PRIMARY,
            visible=False,
            semantics_label=label,
        )
        self._icon_slot = ft.Stack(
            controls=[self._icon_control, self._progress_control],
            width=icon_size,
            height=icon_size,
            alignment=ft.Alignment.CENTER,
        )
        self._label_control = ft.Text(
            label,
            size=label_size,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )

        content_container = ft.Container(
            content=ft.Column(
                [
                    self._icon_slot,
                    self._label_control,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
            alignment=ft.Alignment.CENTER,
        )

        super().__init__(
            content=content_container,
            bgcolor=COLOR_TRANS_TONAL,
            border_radius=16,
            expand=True,
            # alignment=ft.alignment.center,  <-- REMOVED: This was crushing the stack
            on_click=lambda _: self._on_click(),
            animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            shadow=get_card_shadow(),
        )

    def set_state(
        self,
        is_on: bool,
        needs_key: bool = False,
        *,
        is_starting: bool = False,
        status_text: str | None = None,
        helper_text: str | None = None,
    ):
        """Update button visual state."""
        _ = (status_text, helper_text)
        self._is_on = is_on
        self._needs_key = needs_key
        self._is_starting = is_starting
        self._icon_control.visible = not is_starting
        self._progress_control.visible = is_starting

        if needs_key:
            self.bgcolor = COLOR_WARNING
            self.border = None
            self._icon_control.color = ft.Colors.WHITE
            self._label_control.color = ft.Colors.WHITE
        elif is_starting:
            self.bgcolor = COLOR_SURFACE
            self.border = None
            self._icon_control.color = COLOR_SECONDARY
            self._label_control.color = COLOR_PRIMARY
        elif is_on:
            self.bgcolor = self._color_on
            self.border = None
            self._icon_control.color = ft.Colors.WHITE
            self._label_control.color = ft.Colors.WHITE
        else:
            self.bgcolor = COLOR_TRANS_TONAL
            self.border = None
            self._icon_control.color = COLOR_SECONDARY
            self._label_control.color = COLOR_SECONDARY

        update_control_if_mounted(self)

    def set_label(self, label: str) -> None:
        self._label = label
        self._label_control.value = label
        update_control_if_mounted(self._label_control)
