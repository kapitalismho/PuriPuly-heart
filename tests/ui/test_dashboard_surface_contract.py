from __future__ import annotations

import flet as ft

from puripuly_heart.ui.dashboard.contract import DashboardSurfaceSlots
from puripuly_heart.ui.dashboard.renderer import (
    DASHBOARD_CONTROL_REGION_EXPAND,
    DASHBOARD_INFO_REGION_EXPAND,
    DASHBOARD_LAYOUT_GAP,
    DASHBOARD_SHELL_SPACING,
    compose_dashboard_surface,
)
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS


class _CaptureSlotProvider:
    def __init__(self) -> None:
        self.self_control = ft.Text("self")
        self.peer_control = ft.Text("peer")
        self.overlay_button = ft.Text("overlay")

    def self_capture_control(self) -> ft.Control:
        return self.self_control

    def peer_capture_control(self) -> ft.Control:
        return self.peer_control

    def overlay_control(self) -> ft.Control:
        return self.overlay_button


def _slots() -> tuple[DashboardSurfaceSlots, _CaptureSlotProvider]:
    provider = _CaptureSlotProvider()
    slots = DashboardSurfaceSlots.from_capture_provider(
        provider,
        translation=ft.Text("translation"),
        display=ft.Text("display"),
        language=ft.Text("language"),
    )
    return slots, provider


def test_dashboard_surface_layout_uses_shared_design_tokens() -> None:
    assert DASHBOARD_LAYOUT_GAP == FOUNDATION_DESIGN_TOKENS.spacing.inline
    assert DASHBOARD_SHELL_SPACING == FOUNDATION_DESIGN_TOKENS.spacing.page

    slots, _ = _slots()
    surface = compose_dashboard_surface(slots)

    for row_or_column in (
        surface.top_controls,
        surface.bottom_controls,
        surface.control_grid,
        surface.info_stack,
        surface.main_surface,
        surface.shell_content,
    ):
        assert row_or_column.spacing == FOUNDATION_DESIGN_TOKENS.spacing.inline


def test_dashboard_surface_preserves_accepted_region_geometry() -> None:
    slots, _ = _slots()
    surface = compose_dashboard_surface(slots)

    assert surface.control_region.expand == DASHBOARD_CONTROL_REGION_EXPAND
    assert surface.info_region.expand == DASHBOARD_INFO_REGION_EXPAND
    assert surface.main_surface.controls == [surface.control_region, surface.info_region]
    assert surface.control_grid.controls == [surface.top_controls, surface.bottom_controls]
    assert surface.info_stack.controls == [
        surface.display_card_slot,
        surface.language_card_slot,
    ]
    assert surface.shell_content.controls == [surface.main_surface]


def test_dashboard_surface_places_every_slot_in_the_accepted_position() -> None:
    slots, provider = _slots()
    surface = compose_dashboard_surface(slots)

    assert [container.content for container in surface.top_controls.controls] == [
        provider.self_control,
        provider.peer_control,
    ]
    assert [container.content for container in surface.bottom_controls.controls] == [
        slots.translation,
        provider.overlay_button,
    ]
    assert surface.display_card_slot.content is slots.display
    assert surface.language_card_slot.content is slots.language


def test_capture_slot_provider_needs_no_private_dashboard_access() -> None:
    provider = _CaptureSlotProvider()
    slots = DashboardSurfaceSlots.from_capture_provider(
        provider,
        translation=ft.Text("translation"),
        display=ft.Text("display"),
        language=ft.Text("language"),
    )

    assert slots.self_capture is provider.self_control
    assert slots.peer_capture is provider.peer_control
    assert slots.overlay is provider.overlay_button
