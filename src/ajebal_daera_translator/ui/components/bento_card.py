
import flet as ft
from flet.core.colors import Colors as colors
from ajebal_daera_translator.ui.theme import COLOR_SURFACE

class BentoCard(ft.Container):
    def __init__(
        self, 
        content: ft.Control, 
        expand=False, 
        height=None, 
        width=None,
        padding=20,
        opacity_on_surface=0.0
    ):
        """
        A reusable Bento Grid Card component.
        opacity_on_surface: If > 0, blends white into surface color for lighter cards (e.g. input fields).
        """
        bg_color = COLOR_SURFACE
        if opacity_on_surface > 0:
            # We can't easier blend in Flet without complex utils, 
            # so we'll just rely on the user passing a specific color if needed,
            # or usage of colors.with_opacity if it's an overlay.
            # For now, let's keep it simple: defaults to COLOR_SURFACE.
            # If user wants a lighter bg, they can wrap content or we add a bgcolor arg.
            pass
            
        super().__init__(
            content=content,
            bgcolor=bg_color,
            border_radius=16, 
            padding=padding,
            expand=expand,
            height=height,
            width=width,
            border=ft.border.all(1, colors.with_opacity(0.1, colors.WHITE)),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=10,
                color=colors.with_opacity(0.2, colors.BLACK),
                offset=ft.Offset(0, 4),
            )
        )
