from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import is_hover_active
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_ERROR,
    COLOR_NEUTRAL_DARK,
    COLOR_ON_ACCENT,
    COLOR_PRIMARY_CONTAINER,
    COLOR_SECONDARY,
    COLOR_WINDOW_BACKGROUND,
)

_WINDOW_BUTTON_SIZE = 40


class TitleBar(ft.Container):
    """Custom draggable title bar with window controls."""

    def __init__(
        self,
        page: ft.Page,
        *,
        on_close: Callable[[], None],
        on_toggle_theme: Callable[[], None] | None = None,
    ):
        self._page = page
        self._on_close = on_close

        minimize_btn = ft.Container(
            content=ft.Icon(ft.Icons.REMOVE, size=18, color=COLOR_SECONDARY),
            width=_WINDOW_BUTTON_SIZE,
            height=_WINDOW_BUTTON_SIZE,
            alignment=ft.Alignment.CENTER,
            on_click=self._minimize,
            on_hover=self._on_btn_hover,
        )

        self._close_btn = ft.Container(
            content=ft.Icon(ft.Icons.CLOSE, size=18, color=COLOR_SECONDARY),
            width=_WINDOW_BUTTON_SIZE,
            height=_WINDOW_BUTTON_SIZE,
            alignment=ft.Alignment.CENTER,
            border_radius=ft.BorderRadius.only(top_right=16),
            on_click=self._close,
            on_hover=self._on_close_hover,
        )

        controls = [minimize_btn, self._close_btn]
        self._theme_btn: ft.IconButton | None = None
        if on_toggle_theme is not None:
            self._theme_btn = ft.IconButton(
                icon_size=18,
                width=_WINDOW_BUTTON_SIZE,
                height=_WINDOW_BUTTON_SIZE,
                on_click=lambda _: on_toggle_theme(),
            )
            self.apply_theme(page.theme_mode)
            controls.insert(0, self._theme_btn)
        window_controls = ft.Row(controls, spacing=0)

        drag_area = ft.WindowDragArea(
            content=ft.Container(expand=True, height=48),
            expand=True,
        )

        super().__init__(
            content=ft.Row(
                [drag_area, window_controls],
                expand=True,
                spacing=0,
            ),
            bgcolor=COLOR_WINDOW_BACKGROUND,
            height=48,
            border_radius=ft.BorderRadius.only(top_left=16, top_right=16),
        )

    def apply_theme(self, mode: ft.ThemeMode) -> None:
        button = self._theme_btn
        if button is None:
            return
        dark = mode == ft.ThemeMode.DARK
        button.icon = ft.Icons.LIGHT_MODE_OUTLINED if dark else ft.Icons.DARK_MODE_OUTLINED
        button.tooltip = t("theme.switch_to_light" if dark else "theme.switch_to_dark")
        button.style = ft.ButtonStyle(
            padding=0,
            shape=ft.RoundedRectangleBorder(radius=FOUNDATION_DESIGN_TOKENS.radius.control),
            color={
                ft.ControlState.DEFAULT: COLOR_SECONDARY,
                ft.ControlState.HOVERED: COLOR_NEUTRAL_DARK,
            },
            bgcolor={
                ft.ControlState.DEFAULT: ft.Colors.TRANSPARENT,
                ft.ControlState.HOVERED: (
                    ft.Colors.SURFACE_CONTAINER_HIGHEST if dark else COLOR_PRIMARY_CONTAINER
                ),
            },
        )

    def _minimize(self, _):
        self._page.window.minimized = True
        self._page.update()

    def _close(self, _):
        self._on_close()

    def _on_btn_hover(self, e):
        container = e.control
        icon = container.content
        if is_hover_active(e):
            icon.color = COLOR_NEUTRAL_DARK
        else:
            icon.color = COLOR_SECONDARY
        icon.update()

    def _on_close_hover(self, e):
        container = e.control
        icon = container.content
        if is_hover_active(e):
            container.bgcolor = COLOR_ERROR
            icon.color = COLOR_ON_ACCENT
        else:
            container.bgcolor = ft.Colors.TRANSPARENT
            icon.color = COLOR_SECONDARY
        container.update()
        icon.update()
