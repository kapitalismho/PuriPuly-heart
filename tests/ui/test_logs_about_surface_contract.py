from __future__ import annotations

import flet as ft
import pytest

from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.about.contract import AboutSurfaceSlots
from puripuly_heart.ui.about.renderer import ABOUT_ROW_SPACING, compose_about_surface
from puripuly_heart.ui.logs.contract import LogsIntents, LogsSurfaceSlots
from puripuly_heart.ui.logs.renderer import (
    LOGS_CARD_BORDER_RADIUS,
    LOGS_HEADER_BUTTON_SPACING,
    compose_logs_surface,
)
from puripuly_heart.ui.views.about import AboutView
from puripuly_heart.ui.views.logs import LogsView

LOCALES = ("en", "ko", "ja", "zh-CN", "ru")


@pytest.fixture(autouse=True)
def _restore_locale():
    previous = i18n_module.get_locale()
    yield
    i18n_module.set_locale(previous)


def _slot_controls() -> LogsSurfaceSlots:
    return LogsSurfaceSlots(
        title=ft.Text("title"),
        folder_button=ft.TextButton(content="folder"),
        mode_button=ft.TextButton(content="mode"),
        conversation_button=ft.TextButton(content="conversation"),
        log_text=ft.Text("log"),
    )


def test_compose_logs_surface_keeps_the_accepted_geometry_and_slot_order() -> None:
    slots = _slot_controls()
    regions = compose_logs_surface(slots)

    assert regions.header_button_row.spacing == LOGS_HEADER_BUTTON_SPACING
    assert regions.header_button_row.controls == [
        slots.folder_button,
        slots.mode_button,
        slots.conversation_button,
    ]
    assert regions.card.border_radius == LOGS_CARD_BORDER_RADIUS
    assert regions.card.expand is True
    assert regions.card.clip_behavior == ft.ClipBehavior.HARD_EDGE
    assert regions.log_scroll.scroll == ft.ScrollMode.AUTO
    assert regions.log_scroll.expand is True
    assert regions.root is regions.card

    header_row = regions.header.content
    assert isinstance(header_row, ft.Row)
    assert header_row.controls[0] is slots.title
    assert header_row.controls[-1] is regions.header_button_row
    assert header_row.alignment == ft.MainAxisAlignment.SPACE_BETWEEN

    scroll_container = regions.log_scroll.controls[0]
    assert isinstance(scroll_container, ft.Container)
    assert scroll_container.content is slots.log_text


def test_logs_view_production_composition_comes_from_the_renderer() -> None:
    view = LogsView()

    assert len(view.controls) == 1
    card = view.controls[0]
    assert isinstance(card, ft.Container)
    assert card.border_radius == LOGS_CARD_BORDER_RADIUS
    assert view._header_button_row.controls == [
        view._folder_button,
        view._mode_button,
        view._conversation_button,
    ]
    assert view._log_scroll.controls[0].content is view._log_text


def test_logs_view_binds_the_mode_intent_instead_of_an_adhoc_callback() -> None:
    view = LogsView()
    received: list[str] = []
    view.bind_logs_intents(LogsIntents(runtime_logging_mode_change=received.append))

    view._on_mode_button_click(None)

    assert received == ["detailed"]
    assert view.runtime_logging_mode == "detailed"


@pytest.mark.parametrize("locale", LOCALES)
def test_logs_and_about_surface_structure_is_locale_stable(locale: str) -> None:
    i18n_module.set_locale(locale)

    logs_view = LogsView()
    assert len(logs_view.controls) == 1
    assert len(logs_view._header_button_row.controls) == 3

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
