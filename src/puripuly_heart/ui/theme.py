import sys

import flet as ft

from puripuly_heart.ui.fonts import TypeFace, typeface_for_language
from puripuly_heart.ui.foundation.tokens import (
    FOUNDATION_DESIGN_TOKENS,
    WINDOWS_DARK_PALETTE,
    FoundationPaletteTokens,
)

WINDOWS_THEME_SUPPORTED = sys.platform == "win32"
_LIGHT = FOUNDATION_DESIGN_TOKENS.palette
COLOR_BACKGROUND = _LIGHT.background


def _window_color(role: ft.Colors, light: str) -> str:
    return role if WINDOWS_THEME_SUPPORTED else light


COLOR_WINDOW_BACKGROUND = _window_color(ft.Colors.SURFACE_CONTAINER_LOWEST, _LIGHT.background)
COLOR_SURFACE = _window_color(ft.Colors.SURFACE, _LIGHT.surface)
COLOR_ON_BACKGROUND = _window_color(ft.Colors.ON_SURFACE, _LIGHT.on_background)
COLOR_PRIMARY = _window_color(ft.Colors.PRIMARY, _LIGHT.primary)
COLOR_ERROR = _window_color(ft.Colors.ERROR, _LIGHT.error)
COLOR_SUCCESS = _window_color(ft.Colors.SECONDARY_FIXED, _LIGHT.success)
COLOR_WARNING = _window_color(ft.Colors.TERTIARY_FIXED, _LIGHT.warning)
COLOR_DIVIDER = _window_color(ft.Colors.OUTLINE, _LIGHT.divider)
COLOR_PRIMARY_CONTAINER = _window_color(ft.Colors.PRIMARY_CONTAINER, _LIGHT.primary_container)
COLOR_ON_PRIMARY_CONTAINER = _window_color(
    ft.Colors.ON_PRIMARY_CONTAINER, _LIGHT.on_primary_container
)
COLOR_ON_SURFACE_VARIANT = _LIGHT.on_surface_variant
COLOR_SURFACE_DIM = _LIGHT.surface_dim
COLOR_SECONDARY = _window_color(ft.Colors.SECONDARY, _LIGHT.secondary)
COLOR_TERTIARY = _LIGHT.tertiary
COLOR_TRANS_TONAL = _window_color(ft.Colors.TERTIARY_FIXED_DIM, _LIGHT.translation_tonal)
COLOR_TRANS_ON = _window_color(ft.Colors.ON_TERTIARY_FIXED, _LIGHT.translation_on)
COLOR_DISPLAY_SOURCE = _window_color(ft.Colors.ON_SECONDARY_FIXED, _LIGHT.display_source)
COLOR_NEUTRAL = _window_color(ft.Colors.ON_SECONDARY_FIXED_VARIANT, _LIGHT.neutral)
COLOR_NEUTRAL_DARK = _window_color(ft.Colors.ON_SURFACE, _LIGHT.neutral_dark)
COLOR_SURFACE_TONAL = _LIGHT.surface_tonal
COLOR_SURFACE_BRIGHT = _window_color(ft.Colors.SURFACE_BRIGHT, ft.Colors.WHITE)
COLOR_ON_ACCENT = _window_color(ft.Colors.ON_PRIMARY, ft.Colors.WHITE)
COLOR_ACTIVE_CONTROL = _window_color(ft.Colors.PRIMARY_FIXED, _LIGHT.primary)
COLOR_ON_ACTIVE_CONTROL = _window_color(ft.Colors.ON_PRIMARY_FIXED, ft.Colors.WHITE)
COLOR_WARNING_CONTROL = _window_color(ft.Colors.ERROR_CONTAINER, _LIGHT.warning)
COLOR_ON_WARNING_CONTROL = _window_color(ft.Colors.ON_ERROR_CONTAINER, ft.Colors.WHITE)


TEXT_BUTTON_PADDING = ft.Padding.symmetric(horizontal=8, vertical=8)


def _clickable_button_style() -> ft.ButtonStyle:
    return ft.ButtonStyle(
        padding=TEXT_BUTTON_PADDING,
        mouse_cursor={
            ft.ControlState.DISABLED: ft.MouseCursor.BASIC,
            ft.ControlState.DEFAULT: ft.MouseCursor.CLICK,
        },
    )


def text_field_outline_border(
    *, border_radius: int | float
) -> dict[ft.ControlState, ft.InputBorder]:
    def outline(color: str, width: int) -> ft.OutlineInputBorder:
        return ft.OutlineInputBorder(
            border_radius=border_radius,
            side=ft.BorderSide(width=width, color=color),
        )

    return {
        ft.ControlState.DEFAULT: outline(COLOR_DIVIDER, 1),
        ft.ControlState.FOCUSED: outline(COLOR_PRIMARY, 2),
        ft.ControlState.ERROR: outline(COLOR_ERROR, 1),
        ft.ControlState.DISABLED: outline(COLOR_DIVIDER, 1),
    }


def text_style(
    face: TypeFace | None = None,
    *,
    size: int | float | None = None,
    weight: ft.FontWeight | None = None,
    color: str | None = None,
    italic: bool | None = None,
) -> ft.TextStyle:
    if face is None:
        from puripuly_heart.ui.i18n import get_locale

        face = typeface_for_language(get_locale())
    fields: dict[str, object] = {}
    if face.font_family is not None:
        fields["font_family"] = face.font_family
    if face.letter_spacing is not None:
        fields["letter_spacing"] = face.letter_spacing
    if size is not None:
        fields["size"] = size
    if weight is not None:
        fields["weight"] = weight
    if color is not None:
        fields["color"] = color
    if italic is not None:
        fields["italic"] = italic
    return ft.TextStyle(**fields)


def get_app_theme(
    font_family: str | None = None,
    body_letter_spacing: float | None = None,
    *,
    face: TypeFace | None = None,
    dark: bool = False,
) -> ft.Theme:
    chosen = face if face is not None else TypeFace(font_family, body_letter_spacing)
    clickable = _clickable_button_style()
    text_theme = (
        ft.TextTheme(
            body_medium=ft.TextStyle(
                font_family=chosen.font_family,
                letter_spacing=chosen.letter_spacing,
            )
        )
        if chosen.letter_spacing is not None
        else None
    )
    dark = WINDOWS_THEME_SUPPORTED and dark
    palette = WINDOWS_DARK_PALETTE if dark else _LIGHT
    return ft.Theme(
        color_scheme_seed=palette.primary if dark else None,
        color_scheme=_app_color_scheme(palette, dark=dark),
        font_family=chosen.font_family,
        text_theme=text_theme,
        tooltip_theme=(
            ft.TooltipTheme(
                decoration=ft.BoxDecoration(bgcolor=palette.surface_tonal, border_radius=4),
                text_style=text_style(chosen, color=palette.on_background),
            )
            if dark
            else None
        ),
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


def _app_color_scheme(palette: FoundationPaletteTokens, *, dark: bool) -> ft.ColorScheme:
    scheme = ft.ColorScheme(
        surface=palette.surface,
        on_surface=palette.on_background,
        primary=palette.primary,
        error=palette.error,
        outline=palette.divider,
        surface_container_lowest=palette.background,
        secondary=palette.secondary,
        tertiary=palette.tertiary,
    )
    if WINDOWS_THEME_SUPPORTED:
        scheme.primary_container = palette.primary_container
        scheme.on_primary_container = palette.on_primary_container
        scheme.secondary_fixed = palette.success
        scheme.tertiary_fixed = palette.warning
        scheme.tertiary_fixed_dim = palette.translation_tonal
        scheme.on_tertiary_fixed = palette.translation_on
        scheme.on_secondary_fixed = palette.display_source
        scheme.on_secondary_fixed_variant = palette.neutral
        scheme.surface_bright = palette.surface_tonal if dark else ft.Colors.WHITE
        scheme.on_primary = "#2A161C" if dark else ft.Colors.WHITE
        scheme.primary_fixed = palette.primary_container if dark else palette.primary
        scheme.on_primary_fixed = palette.on_primary_container if dark else ft.Colors.WHITE
        scheme.error_container = "#4D3E72" if dark else palette.warning
        scheme.on_error_container = "#E8DFFF" if dark else ft.Colors.WHITE
    if dark:
        scheme.on_secondary = scheme.on_primary
        scheme.on_tertiary = scheme.on_primary
        scheme.on_error = scheme.on_primary
        scheme.on_surface_variant = palette.on_surface_variant
        scheme.surface_dim = palette.surface_dim
        scheme.surface_container_low = palette.surface
        scheme.surface_container = palette.surface_tonal
        scheme.surface_container_high = palette.surface_tonal
        scheme.surface_container_highest = "#504960"
        scheme.outline_variant = palette.divider
        scheme.secondary_container = palette.surface_tonal
        scheme.on_secondary_container = palette.on_background
        scheme.inverse_surface = palette.on_background
        scheme.on_inverse_surface = palette.background
    return scheme
