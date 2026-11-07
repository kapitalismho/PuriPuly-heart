from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("flet")

import flet as ft

from puripuly_heart.ui import motion
from puripuly_heart.ui.components.language_modal import LanguageModal
from puripuly_heart.ui.components.power_button import PowerButton
from tests.helpers.flet_page import DummyPage, attach_dummy_page


class LoopPage(DummyPage):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()

    def drain(self) -> None:
        self.loop.run_until_complete(asyncio.sleep(0.2))


@pytest.fixture
def page():
    page = LoopPage()
    yield page
    page.loop.close()


def test_modal_card_grows_with_overshoot_and_shrinks_on_close(page: LoopPage) -> None:
    card = ft.Container()
    dialog = ft.AlertDialog(content=card)

    motion.show_modal(page, dialog)

    assert page.opened == [dialog]
    assert card.scale == motion.MODAL_START_SCALE
    page.drain()
    assert card.scale == 1.0
    assert card.animate_scale.curve == ft.AnimationCurve.EASE_OUT_BACK
    assert card.animate_scale.duration == 320

    motion.close_modal(page, dialog)

    assert page.closed == [dialog]
    assert card.scale == motion.MODAL_START_SCALE
    assert card.animate_scale.duration == 150


def test_modal_without_task_runner_opens_at_full_size() -> None:
    page = DummyPage()
    card = ft.Container()

    motion.show_modal(page, ft.AlertDialog(content=card))

    assert card.scale is None


def test_language_modal_uses_modal_motion_for_open_and_select(page: LoopPage) -> None:
    chosen: list[str] = []
    modal = LanguageModal(
        page=page, languages=[("ko", "Korean"), ("en", "English")], on_select=chosen.append
    )

    modal.open(current="ko", recent=[])
    page.drain()
    card = modal._dialog.content
    assert card.scale == 1.0

    modal._select("en")

    assert chosen == ["en"]
    assert page.closed == [modal._dialog]
    assert card.scale == motion.MODAL_START_SCALE


def test_power_button_press_dips_then_returns_to_hover_scale(
    monkeypatch: pytest.MonkeyPatch, page: LoopPage
) -> None:
    button = PowerButton("TALK", ft.Icons.MIC, lambda: None)
    attach_dummy_page(monkeypatch, button, page)
    monkeypatch.setattr(button, "update", lambda: None)

    button._on_hover(type("Hover", (), {"data": "true"})())
    button._on_tap_down(None)

    assert button.scale == motion.PRESS_SCALE
    page.drain()
    assert button.scale == 1.02

    button._on_hover(type("Hover", (), {"data": "false"})())
    button._on_tap_down(None)
    page.drain()
    assert button.scale == 1.0


def test_disabled_power_button_does_not_dip(
    monkeypatch: pytest.MonkeyPatch, page: LoopPage
) -> None:
    button = PowerButton("TALK", ft.Icons.MIC, lambda: None)
    attach_dummy_page(monkeypatch, button, page)
    button.disabled = True

    button._on_tap_down(None)

    assert button.scale == 1.0
