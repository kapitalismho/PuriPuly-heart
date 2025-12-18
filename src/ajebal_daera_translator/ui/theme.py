import flet as ft

# Matte Dark Scheme
COLOR_BACKGROUND = "#1E1E1E"  # Solid Deep Matte Grey
COLOR_SURFACE = "#2D2D2D"     # Slightly lighter for cards/inputs
COLOR_ON_BACKGROUND = "#FFFFFF"
COLOR_PRIMARY = "#81D4FA"     # Subtle Light Blue for strict accents
COLOR_ERROR = "#EF5350"
COLOR_SUCCESS = "#66BB6A"
COLOR_DIVIDER = "#424242"

def get_app_theme() -> ft.Theme:
    return ft.Theme(
        color_scheme=ft.ColorScheme(
            background=COLOR_BACKGROUND,
            surface=COLOR_SURFACE,
            on_background=COLOR_ON_BACKGROUND,
            primary=COLOR_PRIMARY,
            error=COLOR_ERROR,
            outline=COLOR_DIVIDER,
        ),
        visual_density=ft.VisualDensity.COMPACT,
        page_transitions=ft.PageTransitionsTheme(
            windows=ft.PageTransitionTheme.NONE,
            macos=ft.PageTransitionTheme.NONE,
            linux=ft.PageTransitionTheme.NONE,
        ),
    )
