"""Settings section card component - About page style."""

import flet as ft

from puripuly_heart.ui.flet_runtime import update_control_if_mounted
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_NEUTRAL,
    COLOR_SURFACE,
)


class SettingsSection(ft.Container):
    """Card section with title and content, matching About page style."""

    def __init__(
        self,
        title_key: str,
        content: ft.Control,
        *,
        expand: bool = False,
    ):
        self._title_key = title_key
        self._content = content

        self._title = ft.Text(
            t(title_key),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_NEUTRAL,
        )

        inner_content = ft.Column(
            controls=[
                self._title,
                ft.Container(height=16),
                self._content,
            ],
            spacing=0,
        )

        content_container = ft.Container(
            content=inner_content,
            expand=True,
            padding=24,
        )

        super().__init__(
            content=content_container,
            bgcolor=COLOR_SURFACE,
            border_radius=16,
            expand=expand,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )

    def apply_locale(self) -> None:
        """Update title text when locale changes."""
        self._title.value = t(self._title_key)
        update_control_if_mounted(self._title)
