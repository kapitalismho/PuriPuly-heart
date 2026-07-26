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
COLOR_NEUTRAL = FOUNDATION_DESIGN_TOKENS.palette.neutral
COLOR_NEUTRAL_DARK = FOUNDATION_DESIGN_TOKENS.palette.neutral_dark
COLOR_SURFACE_TONAL = FOUNDATION_DESIGN_TOKENS.palette.surface_tonal


def get_app_theme(font_family: str | None = None) -> ft.Theme:
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
        visual_density=ft.VisualDensity.COMPACT,
        page_transitions=ft.PageTransitionsTheme(
            windows=ft.PageTransitionTheme.NONE,
            macos=ft.PageTransitionTheme.NONE,
            linux=ft.PageTransitionTheme.NONE,
        ),
    )


def get_card_shadow() -> ft.BoxShadow:
    """Return the standard card shadow with warm color from theme.

    Uses COLOR_ON_PRIMARY_CONTAINER for a warm, cohesive shadow that
    blends naturally with the pink/coral color scheme.
    """
    return ft.BoxShadow(
        blur_radius=FOUNDATION_DESIGN_TOKENS.shadow.blur_radius,
        color=ft.Colors.with_opacity(
            FOUNDATION_DESIGN_TOKENS.shadow.opacity,
            COLOR_ON_PRIMARY_CONTAINER,
        ),
        offset=ft.Offset(
            FOUNDATION_DESIGN_TOKENS.shadow.offset_x,
            FOUNDATION_DESIGN_TOKENS.shadow.offset_y,
        ),
        spread_radius=FOUNDATION_DESIGN_TOKENS.shadow.spread_radius,
    )
