from typing import Callable

import flet as ft

from puripuly_heart.ui.components.loading_spinner import create_button_spinner
from puripuly_heart.ui.flet_runtime import is_hover_active, update_control_if_mounted
from puripuly_heart.ui.theme import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    COLOR_TRANS_TONAL,
    COLOR_WARNING,
)

_HOVER_SCALE = 1.02


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
        self._progress_control = create_button_spinner(
            icon_size, semantics_label=label, color=COLOR_PRIMARY
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
            on_hover=self._on_hover,
            animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            animate_scale=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            scale=1.0,
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
            self._icon_control.color = ft.Colors.WHITE
            self._label_control.color = ft.Colors.WHITE
        elif is_starting:
            self.bgcolor = COLOR_SURFACE
            self._icon_control.color = COLOR_SECONDARY
            self._label_control.color = COLOR_PRIMARY
        elif is_on:
            self.bgcolor = self._color_on
            self._icon_control.color = ft.Colors.WHITE
            self._label_control.color = ft.Colors.WHITE
        else:
            self.bgcolor = COLOR_TRANS_TONAL
            self._icon_control.color = COLOR_SECONDARY
            self._label_control.color = COLOR_SECONDARY

        self.border = None

        update_control_if_mounted(self)

    def _on_hover(self, event: ft.ControlEvent) -> None:
        """Lift the button while the pointer is over it."""
        self.scale = _HOVER_SCALE if (is_hover_active(event) and not self._is_starting) else 1.0
        update_control_if_mounted(self)

    def set_label(self, label: str) -> None:
        self._label = label
        self._label_control.value = label
        update_control_if_mounted(self._label_control)
