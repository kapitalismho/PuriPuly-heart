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
    display_source: str
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
        background="#FDF9F8",
        surface="#FDF1F0",
        on_background="#554847",
        primary="#FF6B6B",
        error="#FF5449",
        success="#66BB6A",
        warning="#FF8A65",
        divider="#EDE2DF",
        primary_container="#FFDAD8",
        on_primary_container="#733332",
        on_surface_variant="#5F5352",
        surface_dim="#D2C5C5",
        secondary="#B9827D",
        tertiary="#8C6E28",
        translation_tonal="#FBE3E1",
        translation_on="#A84045",
        display_source="#855F5B",
        neutral="#746665",
        neutral_dark="#554847",
        surface_tonal="#F8EEED",
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

WINDOWS_DARK_PALETTE = FoundationPaletteTokens(
    background="#121212",
    surface="#1E1E1E",
    on_background="#E6E6E6",
    primary="#FF6B6B",
    error="#FF7C75",
    success="#82C987",
    warning="#FFAD90",
    divider="#393939",
    primary_container="#503031",
    on_primary_container="#FFD1CC",
    on_surface_variant="#B0B0B0",
    surface_dim="#121212",
    secondary="#BCAFAE",
    tertiary="#D9B875",
    translation_tonal="#322627",
    translation_on="#FF9B98",
    display_source="#C5AAA8",
    neutral="#B0B0B0",
    neutral_dark="#E6E6E6",
    surface_tonal="#252525",
)


__all__ = [
    "FOUNDATION_DESIGN_TOKENS",
    "WINDOWS_DARK_PALETTE",
    "FoundationDesignTokens",
    "FoundationPaletteTokens",
    "FoundationRadiusTokens",
    "FoundationSpacingTokens",
    "FoundationTypographyTokens",
    "FoundationWindowTokens",
]
