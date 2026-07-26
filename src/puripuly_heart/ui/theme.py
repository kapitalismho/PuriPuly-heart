import flet as ft

# Light Theme - Material Design 3 (Seed: #FF6B6B)
COLOR_BACKGROUND = "#FFF8F6"  # Surface
COLOR_SURFACE = "#FFF0EE"  # Surface Container
COLOR_ON_BACKGROUND = "#5C4D4C"  # Neutral Dark
COLOR_PRIMARY = "#FF6B6B"  # Primary
COLOR_ERROR = "#FF5449"  # Error
COLOR_SUCCESS = "#66BB6A"
COLOR_WARNING = "#FF8A65"  # Coral orange for warning states
COLOR_DIVIDER = "#E8D4D2"  # Divider
COLOR_PRIMARY_CONTAINER = "#ffdad8"
COLOR_ON_PRIMARY_CONTAINER = "#733332"
COLOR_ON_SURFACE_VARIANT = "#534341"
COLOR_SURFACE_DIM = "#d7c1c0"  # OFF state background

# Additional colors for light theme
COLOR_SECONDARY = "#B78481"  # Secondary text/icons
COLOR_TERTIARY = "#B28A44"  # Arrow icon
COLOR_TRANS_TONAL = "#F5DEDC"  # Toggle button background (OFF state)
COLOR_TRANS_ON = "#D64058"  # TRANS button ON state
COLOR_NEUTRAL = "#998E8D"  # Inactive icons
COLOR_NEUTRAL_DARK = "#5C4D4C"  # Main text
COLOR_SURFACE_TONAL = "#FCEBE9"  # Alternative surface (hover state)


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
        blur_radius=10,
        color=ft.Colors.with_opacity(0.05, COLOR_ON_PRIMARY_CONTAINER),
        offset=ft.Offset(0, 2),
        spread_radius=0,
    )
