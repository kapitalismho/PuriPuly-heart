from __future__ import annotations

import flet as ft

from puripuly_heart.ui.about.contract import AboutSurfaceSlots
from puripuly_heart.ui.about.renderer import compose_about_surface


def test_compose_about_surface_keeps_the_accepted_card_order() -> None:
    slots = AboutSurfaceSlots(
        app_name_card=ft.Text("app"),
        version_card=ft.Text("version"),
        credits_card=ft.Text("credits"),
        inspired_by_card=ft.Text("inspired"),
        special_thanks_card=ft.Text("thanks"),
        licenses_card=ft.Text("licenses"),
    )
    regions = compose_about_surface(slots)

    assert regions.controls == (
        regions.header_row,
        regions.top_row,
        slots.special_thanks_card,
        slots.licenses_card,
    )
    assert regions.header_row.content.controls[0].content is slots.app_name_card
    assert regions.header_row.content.controls[1].content is slots.version_card
    assert regions.top_row.content.controls[0].content is slots.credits_card
    assert regions.top_row.content.controls[1].content is slots.inspired_by_card
    for row in (regions.header_row, regions.top_row):
        assert row.content.expand is True
        assert all(item.expand is True for item in row.content.controls)
