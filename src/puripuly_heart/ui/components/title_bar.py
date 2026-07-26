from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import update_control_if_mounted
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_DIVIDER,
    COLOR_NEUTRAL,
    COLOR_NEUTRAL_DARK,
)


class TitleBar(ft.Container):
    """Custom draggable title bar with window controls."""

    def __init__(self, page: ft.Page, *, on_close: Callable[[], None]):
        self._page = page
        self._on_close = on_close

        self._title_text = ft.Text(
            "PuriPuly Heart",
            size=14,
            weight=ft.FontWeight.W_600,
            color=COLOR_NEUTRAL_DARK,
            font_family="NanumSquare",
        )

        minimize_btn = ft.Container(
            content=ft.Icon(ft.Icons.REMOVE, size=18, color=COLOR_NEUTRAL),
            width=40,
            height=40,
            alignment=ft.Alignment.CENTER,
            on_click=self._minimize,
            on_hover=self._on_btn_hover,
        )

        self._maximize_btn = ft.Container(
            content=ft.Icon(ft.Icons.CROP_SQUARE, size=16, color=COLOR_NEUTRAL),
            width=40,
            height=40,
            alignment=ft.Alignment.CENTER,
            disabled=True,
        )

        self._close_btn = ft.Container(
            content=ft.Icon(ft.Icons.CLOSE, size=18, color=COLOR_NEUTRAL),
            width=40,
            height=40,
            alignment=ft.Alignment.CENTER,
            border_radius=ft.BorderRadius.only(top_right=16),
            on_click=self._close,
            on_hover=self._on_close_hover,
        )

        window_controls = ft.Row(
            [minimize_btn, self._maximize_btn, self._close_btn],
            spacing=0,
        )

        drag_area = ft.WindowDragArea(
            content=ft.Container(
                content=ft.Row(
                    [
                        ft.Container(content=self._title_text, padding=ft.Padding.only(left=16)),
                        ft.Container(expand=True),
                    ],
                    expand=True,
                ),
                expand=True,
            ),
            expand=True,
        )

        super().__init__(
            content=ft.Row(
                [drag_area, window_controls],
                expand=True,
                spacing=0,
            ),
            bgcolor=COLOR_BACKGROUND,
            height=48,
            border_radius=ft.BorderRadius.only(top_left=16, top_right=16),
            border=ft.Border.only(bottom=ft.BorderSide(1, COLOR_DIVIDER)),
        )

    def _minimize(self, _):
        self._page.window.minimized = True
        self._page.update()

    def _close(self, _):
        self._on_close()

    def _on_btn_hover(self, e):
        container = e.control
        icon = container.content
        if e.data == "true":
            icon.color = COLOR_NEUTRAL_DARK
        else:
            icon.color = COLOR_NEUTRAL
        icon.update()

    def _on_close_hover(self, e):
        container = e.control
        icon = container.content
        if e.data == "true":
            container.bgcolor = ft.Colors.RED_400
            icon.color = ft.Colors.WHITE
        else:
            container.bgcolor = ft.Colors.TRANSPARENT
            icon.color = COLOR_NEUTRAL
        container.update()
        icon.update()

    def set_title(self, title: str) -> None:
        self._title_text.value = title
        update_control_if_mounted(self._title_text)
