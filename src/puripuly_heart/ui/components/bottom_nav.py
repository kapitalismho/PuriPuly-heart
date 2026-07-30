from typing import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import is_hover_active
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_DIVIDER,
    COLOR_NEUTRAL,
    COLOR_PRIMARY,
)


class BottomNavBar(ft.Container):
    """Bottom 4-tab navigation bar with icons only."""

    def __init__(self, on_change: Callable[[int], None]):
        self._on_change = on_change
        self._selected = 0

        self._tabs = [
            ft.Icons.GRID_VIEW,
            ft.Icons.SETTINGS,
            ft.Icons.ARTICLE,
            ft.Icons.INFO_OUTLINE,
        ]

        self._icons: list[ft.Icon] = []
        self._tab_containers: list[ft.Container] = []
        self._build_tabs()

        tabs_row = ft.Row(
            controls=self._build_tabs_with_dividers(),
            expand=True,
            spacing=0,
        )

        super().__init__(
            content=tabs_row,
            bgcolor=COLOR_BACKGROUND,
            height=80,
            border=ft.Border.only(top=ft.BorderSide(1, COLOR_DIVIDER)),
        )

    def _build_tabs(self):
        """Build tab containers with icons only."""
        self._tab_containers = []
        self._icons = []
        for i, icon_name in enumerate(self._tabs):
            is_selected = i == self._selected
            icon_color = COLOR_PRIMARY if is_selected else COLOR_NEUTRAL

            icon = ft.Icon(icon=icon_name, size=30, color=icon_color)
            self._icons.append(icon)

            container = ft.Container(
                content=icon,
                expand=True,
                alignment=ft.Alignment.CENTER,
                on_click=lambda _, idx=i: self._on_tab_click(idx),
                on_hover=lambda e, idx=i: self._on_tab_hover(e, idx),
            )
            self._tab_containers.append(container)

    def _build_tabs_with_dividers(self) -> list[ft.Control]:
        """Build tabs with vertical dividers between them."""
        result: list[ft.Control] = []
        for i, tab in enumerate(self._tab_containers):
            result.append(tab)
            if i < len(self._tab_containers) - 1:
                result.append(ft.VerticalDivider(width=1, color=COLOR_DIVIDER, thickness=1))
        return result

    def _on_tab_click(self, index: int):
        """Handle tab click."""
        if self._selected != index:
            self._selected = index
            self._update_visuals()
            self._on_change(index)

    def _on_tab_hover(self, e, index: int):
        """Handle tab hover."""
        if index != self._selected:
            icon = self._icons[index]
            icon.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_NEUTRAL
            icon.update()

    def _update_visuals(self):
        """Update all tab visuals based on selection."""
        for i, icon in enumerate(self._icons):
            icon.color = COLOR_PRIMARY if i == self._selected else COLOR_NEUTRAL
            icon.update()
