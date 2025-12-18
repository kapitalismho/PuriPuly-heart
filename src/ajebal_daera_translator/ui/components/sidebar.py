import flet as ft
from flet.core.icons import Icons as icons
from flet.core.colors import Colors as colors
from typing import Callable
from ajebal_daera_translator.ui.components.bento_card import BentoCard

class AppSidebar(ft.Container):
    def __init__(self, on_change: Callable[[int], None]):
        super().__init__(
            padding=0, 
            width=80,
            bgcolor=colors.TRANSPARENT,
            alignment=ft.alignment.top_center
        )
        self.on_nav_change = on_change
        self.selected_index = 0
        
        # Build layout
        self._build_sidebar()

    def _build_sidebar(self):
        self.nav_items = [
            {"icon": icons.DASHBOARD_OUTLINED, "selected_icon": icons.DASHBOARD_ROUNDED, "label": "Home"},
            {"icon": icons.HISTORY_OUTLINED, "selected_icon": icons.HISTORY_ROUNDED, "label": "History"},
            {"icon": icons.SETTINGS_OUTLINED, "selected_icon": icons.SETTINGS_ROUNDED, "label": "Settings"},
            {"icon": icons.ARTICLE_OUTLINED, "selected_icon": icons.ARTICLE_ROUNDED, "label": "Logs"},
        ]
        
        self.tiles = []
        for i, item in enumerate(self.nav_items):
            self.tiles.append(self._build_nav_tile(i, item))

        self.content = BentoCard(
            content=ft.Column(
                controls=self.tiles,
                spacing=0, # No gaps between tiles inside the card
                expand=True
            ),
            expand=True,
            padding=0 # Remove padding to let tiles feel "full"
        )
        
    def _build_nav_tile(self, index, item):
        is_selected = self.selected_index == index
        icon_name = item["selected_icon"] if is_selected else item["icon"]
        icon_color = colors.WHITE if is_selected else colors.GREY_500
        bg_color = colors.with_opacity(0.1, colors.WHITE) if is_selected else colors.TRANSPARENT
        
        return ft.Container(
            content=ft.Column([
                ft.Icon(name=icon_name, color=icon_color, size=28),
                ft.Text(item["label"], size=10, color=icon_color, weight=ft.FontWeight.BOLD)
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
            alignment=ft.alignment.center,
            expand=True, # Critical: Fill available space (3 equal parts)
            bgcolor=bg_color,
            on_click=lambda e: self._on_tile_click(index),
            animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
        )

    def _on_tile_click(self, index):
        self.selected_index = index
        self.on_nav_change(index)
        
        # Update UI state manually
        self.tiles.clear()
        for i, item in enumerate(self.nav_items):
             self.tiles.append(self._build_nav_tile(i, item))
        
        # We need to update the column content. 
        # Since we are replacing the list, we need to access the column control.
        # Structure: AppSidebar(Container) -> BentoCard(Container) -> Column
        self.content.content.controls = self.tiles
        self.update()
