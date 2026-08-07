from __future__ import annotations

import flet as ft

from puripuly_heart.ui.theme import COLOR_SURFACE


class SharedCardWrapper(ft.Container):
    DEFAULT_HEIGHT = 300

    def __init__(
        self,
        content: ft.Control,
        *,
        expand: bool | None = None,
        height: float | int | None = DEFAULT_HEIGHT,
        padding: float | int = 24,
    ) -> None:
        resolved_expand = height is not None if expand is None else expand
        resolved_height = self.DEFAULT_HEIGHT if resolved_expand and height is None else height
        content_container = ft.Container(
            content=content,
            expand=resolved_expand,
            padding=padding,
        )

        super().__init__(
            content=content_container,
            bgcolor=COLOR_SURFACE,
            border_radius=16,
            expand=resolved_expand,
            height=resolved_height,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
