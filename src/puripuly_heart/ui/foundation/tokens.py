from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FoundationPaletteTokens:
    background: str
    surface: str
    on_background: str
    primary: str
    error: str
    success: str
    warning: str
    divider: str
    primary_container: str
    on_primary_container: str
    on_surface_variant: str
    surface_dim: str
    secondary: str
    tertiary: str
    translation_tonal: str
    translation_on: str
    neutral: str
    neutral_dark: str
    surface_tonal: str


@dataclass(frozen=True, slots=True)
class FoundationSpacingTokens:
    page: int
    card: int
    compact: int
    inline: int


@dataclass(frozen=True, slots=True)
class FoundationRadiusTokens:
    card: int
    control: int


@dataclass(frozen=True, slots=True)
class FoundationTypographyTokens:
    title: int
    body: int
    label: int


@dataclass(frozen=True, slots=True)
class FoundationWindowTokens:
    width: int
    height: int
    resizable: bool
    maximizable: bool
    frameless: bool


@dataclass(frozen=True, slots=True)
class FoundationDesignTokens:
    palette: FoundationPaletteTokens
    spacing: FoundationSpacingTokens
    radius: FoundationRadiusTokens
    typography: FoundationTypographyTokens
    window: FoundationWindowTokens
    default_font_family: str
    icon_asset: str
    accepted_production_revision: str


FOUNDATION_DESIGN_TOKENS = FoundationDesignTokens(
    palette=FoundationPaletteTokens(
        background="#FFF8F6",
        surface="#FFF0EE",
        on_background="#5C4D4C",
        primary="#FF6B6B",
        error="#FF5449",
        success="#66BB6A",
        warning="#FF8A65",
        divider="#E8D4D2",
        primary_container="#ffdad8",
        on_primary_container="#733332",
        on_surface_variant="#534341",
        surface_dim="#d7c1c0",
        secondary="#B78481",
        tertiary="#B28A44",
        translation_tonal="#F5DEDC",
        translation_on="#D64058",
        neutral="#998E8D",
        neutral_dark="#5C4D4C",
        surface_tonal="#FCEBE9",
    ),
    spacing=FoundationSpacingTokens(
        page=16,
        card=24,
        compact=8,
        inline=12,
    ),
    radius=FoundationRadiusTokens(
        card=16,
        control=12,
    ),
    typography=FoundationTypographyTokens(
        title=24,
        body=14,
        label=12,
    ),
    window=FoundationWindowTokens(
        width=1136,
        height=850,
        resizable=False,
        maximizable=False,
        frameless=True,
    ),
    default_font_family="NanumSquareRound",
    icon_asset="icons/icon.ico",
    accepted_production_revision="3fb5ce83e4840ef1fd49f2b5480952a09af66527",
)


__all__ = [
    "FOUNDATION_DESIGN_TOKENS",
    "FoundationDesignTokens",
    "FoundationPaletteTokens",
    "FoundationRadiusTokens",
    "FoundationSpacingTokens",
    "FoundationTypographyTokens",
    "FoundationWindowTokens",
]
