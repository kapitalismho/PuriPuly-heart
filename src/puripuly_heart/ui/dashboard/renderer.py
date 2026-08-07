from __future__ import annotations

import flet as ft

from puripuly_heart.ui.dashboard.contract import (
    DashboardSurfaceRegions,
    DashboardSurfaceSlots,
)
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS

DASHBOARD_LAYOUT_GAP = FOUNDATION_DESIGN_TOKENS.spacing.inline
DASHBOARD_SHELL_SPACING = FOUNDATION_DESIGN_TOKENS.spacing.page
DASHBOARD_CONTROL_REGION_EXPAND = 45
DASHBOARD_INFO_REGION_EXPAND = 55
DASHBOARD_DISPLAY_CARD_EXPAND = 1
DASHBOARD_LANGUAGE_CARD_EXPAND = 1
DASHBOARD_POWER_BUTTON_ICON_SIZE = 80
DASHBOARD_POWER_BUTTON_LABEL_SIZE = 32


def compose_dashboard_surface(slots: DashboardSurfaceSlots) -> DashboardSurfaceRegions:
    top_controls = ft.Row(
        [
            ft.Container(content=slots.self_capture, expand=True),
            ft.Container(content=slots.peer_capture, expand=True),
        ],
        spacing=DASHBOARD_LAYOUT_GAP,
        expand=True,
    )
    bottom_controls = ft.Row(
        [
            ft.Container(content=slots.translation, expand=True),
            ft.Container(content=slots.overlay, expand=True),
        ],
        spacing=DASHBOARD_LAYOUT_GAP,
        expand=True,
    )
    control_grid = ft.Column(
        [top_controls, bottom_controls],
        spacing=DASHBOARD_LAYOUT_GAP,
        expand=True,
    )
    display_card_slot = ft.Container(
        content=slots.display,
        expand=DASHBOARD_DISPLAY_CARD_EXPAND,
    )
    language_card_slot = ft.Container(
        content=slots.language,
        expand=DASHBOARD_LANGUAGE_CARD_EXPAND,
    )
    info_stack = ft.Column(
        [display_card_slot, language_card_slot],
        spacing=DASHBOARD_LAYOUT_GAP,
        expand=True,
    )
    control_region = ft.Container(
        content=control_grid,
        expand=DASHBOARD_CONTROL_REGION_EXPAND,
    )
    info_region = ft.Container(
        content=info_stack,
        expand=DASHBOARD_INFO_REGION_EXPAND,
    )
    main_surface = ft.Row(
        [control_region, info_region],
        spacing=DASHBOARD_LAYOUT_GAP,
        expand=True,
    )
    shell_content = ft.Column(
        [main_surface],
        spacing=DASHBOARD_LAYOUT_GAP,
        expand=True,
    )
    return DashboardSurfaceRegions(
        root=shell_content,
        shell_content=shell_content,
        main_surface=main_surface,
        control_region=control_region,
        info_region=info_region,
        control_grid=control_grid,
        top_controls=top_controls,
        bottom_controls=bottom_controls,
        info_stack=info_stack,
        display_card_slot=display_card_slot,
        language_card_slot=language_card_slot,
    )


__all__ = [
    "DASHBOARD_CONTROL_REGION_EXPAND",
    "DASHBOARD_DISPLAY_CARD_EXPAND",
    "DASHBOARD_INFO_REGION_EXPAND",
    "DASHBOARD_LANGUAGE_CARD_EXPAND",
    "DASHBOARD_LAYOUT_GAP",
    "DASHBOARD_POWER_BUTTON_ICON_SIZE",
    "DASHBOARD_POWER_BUTTON_LABEL_SIZE",
    "DASHBOARD_SHELL_SPACING",
    "compose_dashboard_surface",
]
