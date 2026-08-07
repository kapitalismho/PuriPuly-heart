from dataclasses import dataclass

import flet as ft

from puripuly_heart.ui.theme import COLOR_PRIMARY


@dataclass(frozen=True)
class GlowConfig:
    """Configuration for glow orb effect.

    Attributes:
        width: Width of the glow container.
        height: Height of the glow container.
        blur_radius: How far the shadow spreads (larger = softer glow).
        spread_radius: Size increase of the shadow.
        opacity: Opacity of glow color (0.0 to 1.0).
        right: Offset from right edge (negative = outside).
        bottom: Offset from bottom edge (negative = outside).
    """

    width: int = 200
    height: int = 200
    blur_radius: int = 150
    spread_radius: int = 10
    opacity: float = 0.05
    right: int = -50
    bottom: int = -50


# Preset configurations
GLOW_CARD = GlowConfig()
"""Default glow for cards - subtle corner bloom."""

GLOW_BACKGROUND = GlowConfig(
    width=400,
    height=400,
    blur_radius=250,
    spread_radius=20,
    opacity=0.04,
    right=-100,
    bottom=-100,
)
"""Larger glow for page backgrounds - atmospheric ambient light."""


def _create_glow_orb(config: GlowConfig, color: str = COLOR_PRIMARY) -> ft.Container:
    """Create an invisible placeholder orb container.

    The orb layer is kept transparent so glow effects render as nothing.

    Args:
        config: Glow configuration settings.
        color: Base color for the glow (from theme).

    Returns:
        Container with no glow effect.
    """
    return ft.Container(
        width=config.width,
        height=config.height,
        bgcolor=ft.Colors.TRANSPARENT,
        right=config.right,
        bottom=config.bottom,
    )


def create_glow_stack(
    content: ft.Control,
    config: GlowConfig = GLOW_CARD,
    color: str = COLOR_PRIMARY,
) -> ft.Stack:
    """Wrap content in a Stack with an invisible orb layer in the background.

    Args:
        content: The foreground content control (e.g., Column, Row).
        config: Glow configuration (defaults to GLOW_CARD preset).
        color: Base color for the glow (from theme, defaults to COLOR_PRIMARY).

    Returns:
        Stack containing the orb layer and content layer.
    """
    return ft.Stack(
        controls=[
            _create_glow_orb(config, color),
            ft.Container(content=content, expand=True),
        ],
        expand=True,
    )


def create_background_glow_stack(
    content: ft.Control,
    color: str = COLOR_PRIMARY,
) -> ft.Stack:
    """Wrap content with a large background glow effect.

    Convenience function using GLOW_BACKGROUND preset.
    Ideal for wrapping entire page/view content.

    Args:
        content: The main page content.
        color: Base color for the glow (from theme, defaults to COLOR_PRIMARY).

    Returns:
        Stack with atmospheric background glow.
    """
    return create_glow_stack(content, config=GLOW_BACKGROUND, color=color)
