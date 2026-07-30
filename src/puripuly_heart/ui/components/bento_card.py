import flet as ft

from puripuly_heart.ui.theme import COLOR_DIVIDER, COLOR_SURFACE


class BentoCard(ft.Container):
    def __init__(
        self,
        content: ft.Control,
        expand=False,
        height=None,
        width=None,
        padding=20,
        bgcolor=None,
    ):
        """A reusable Bento Grid Card component for light theme."""
        bg_color = bgcolor if bgcolor else COLOR_SURFACE

        super().__init__(
            content=content,
            bgcolor=bg_color,
            border_radius=16,
            padding=padding,
            expand=expand,
            height=height,
            width=width,
            border=ft.Border.all(1, COLOR_DIVIDER),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=8,
                color=ft.Colors.with_opacity(0.08, ft.Colors.BLACK),
                offset=ft.Offset(0, 2),
            ),
        )
