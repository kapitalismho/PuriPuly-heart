from __future__ import annotations

import flet as ft
import pytest

from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.about.contract import AboutSurfaceSlots
from puripuly_heart.ui.about.renderer import ABOUT_ROW_SPACING, compose_about_surface
from puripuly_heart.ui.views.about import AboutView

LOCALES = ("en", "ko", "ja", "zh-CN", "ru")


@pytest.fixture(autouse=True)
def _restore_locale():
    previous = i18n_module.get_locale()
    yield
    i18n_module.set_locale(previous)


@pytest.mark.parametrize("locale", LOCALES)
def test_about_surface_structure_is_locale_stable(locale: str) -> None:
    i18n_module.set_locale(locale)

    about_view = AboutView()
    assert len(about_view.controls) == 4
    header_row, top_row = about_view.controls[0], about_view.controls[1]
    assert isinstance(header_row, ft.Container)
    assert isinstance(top_row, ft.Container)
    assert header_row.content.spacing == ABOUT_ROW_SPACING
    assert top_row.content.spacing == ABOUT_ROW_SPACING


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


def test_about_view_reapplies_locale_through_the_renderer() -> None:
    i18n_module.set_locale("en")
    view = AboutView()
    before = len(view.controls)

    i18n_module.set_locale("ko")
    view._build_ui()

    assert len(view.controls) == before
