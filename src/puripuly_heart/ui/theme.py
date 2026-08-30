import flet as ft

from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS

# Light Theme - Material Design 3 (Seed: #FF6B6B)
COLOR_BACKGROUND = FOUNDATION_DESIGN_TOKENS.palette.background
COLOR_SURFACE = FOUNDATION_DESIGN_TOKENS.palette.surface
COLOR_ON_BACKGROUND = FOUNDATION_DESIGN_TOKENS.palette.on_background
COLOR_PRIMARY = FOUNDATION_DESIGN_TOKENS.palette.primary
COLOR_ERROR = FOUNDATION_DESIGN_TOKENS.palette.error
COLOR_SUCCESS = FOUNDATION_DESIGN_TOKENS.palette.success
COLOR_WARNING = FOUNDATION_DESIGN_TOKENS.palette.warning
COLOR_DIVIDER = FOUNDATION_DESIGN_TOKENS.palette.divider
COLOR_PRIMARY_CONTAINER = FOUNDATION_DESIGN_TOKENS.palette.primary_container
COLOR_ON_PRIMARY_CONTAINER = FOUNDATION_DESIGN_TOKENS.palette.on_primary_container
COLOR_ON_SURFACE_VARIANT = FOUNDATION_DESIGN_TOKENS.palette.on_surface_variant
COLOR_SURFACE_DIM = FOUNDATION_DESIGN_TOKENS.palette.surface_dim

# Additional colors for light theme
COLOR_SECONDARY = FOUNDATION_DESIGN_TOKENS.palette.secondary
COLOR_TERTIARY = FOUNDATION_DESIGN_TOKENS.palette.tertiary
COLOR_TRANS_TONAL = FOUNDATION_DESIGN_TOKENS.palette.translation_tonal
COLOR_TRANS_ON = FOUNDATION_DESIGN_TOKENS.palette.translation_on
COLOR_DISPLAY_SOURCE = FOUNDATION_DESIGN_TOKENS.palette.display_source
COLOR_NEUTRAL = FOUNDATION_DESIGN_TOKENS.palette.neutral
COLOR_NEUTRAL_DARK = FOUNDATION_DESIGN_TOKENS.palette.neutral_dark
COLOR_SURFACE_TONAL = FOUNDATION_DESIGN_TOKENS.palette.surface_tonal


TEXT_BUTTON_PADDING = ft.Padding.symmetric(horizontal=8, vertical=8)


def _clickable_button_style() -> ft.ButtonStyle:
    return ft.ButtonStyle(
        padding=TEXT_BUTTON_PADDING,
        mouse_cursor={
            ft.ControlState.DISABLED: ft.MouseCursor.BASIC,
            ft.ControlState.DEFAULT: ft.MouseCursor.CLICK,
        },
    )


def get_app_theme(
    font_family: str | None = None,
    body_letter_spacing: float | None = None,
) -> ft.Theme:
    clickable = _clickable_button_style()
    text_theme = (
        ft.TextTheme(body_medium=ft.TextStyle(letter_spacing=body_letter_spacing))
        if body_letter_spacing is not None
        else None
    )
    return ft.Theme(
        color_scheme=ft.ColorScheme(
            surface=COLOR_SURFACE,
            on_surface=COLOR_ON_BACKGROUND,
            primary=COLOR_PRIMARY,
            error=COLOR_ERROR,
            outline=COLOR_DIVIDER,
            surface_container_lowest=COLOR_BACKGROUND,
            secondary=COLOR_SECONDARY,
            tertiary=COLOR_TERTIARY,
        ),
        font_family=font_family,
        text_theme=text_theme,
        visual_density=ft.VisualDensity.COMPACT,
        button_theme=ft.ButtonTheme(style=clickable),
        text_button_theme=ft.TextButtonTheme(style=clickable),
        outlined_button_theme=ft.OutlinedButtonTheme(style=clickable),
        filled_button_theme=ft.FilledButtonTheme(style=clickable),
        icon_button_theme=ft.IconButtonTheme(style=clickable),
        page_transitions=ft.PageTransitionsTheme(
            windows=ft.PageTransitionTheme.NONE,
            macos=ft.PageTransitionTheme.NONE,
            linux=ft.PageTransitionTheme.NONE,
        ),
    )
