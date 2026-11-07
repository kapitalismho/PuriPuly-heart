from __future__ import annotations

import pytest

pytest.importorskip("flet")

import flet as ft

from puripuly_heart.ui.theme import (
    COLOR_DIVIDER,
    COLOR_ERROR,
    COLOR_PRIMARY,
    text_field_outline_border,
)
from tests.helpers.flet_page import DummyPage, attach_dummy_page


def test_text_field_outline_border_covers_every_visual_state() -> None:
    borders = text_field_outline_border(border_radius=12)

    assert set(borders) == {
        ft.ControlState.DEFAULT,
        ft.ControlState.FOCUSED,
        ft.ControlState.ERROR,
        ft.ControlState.DISABLED,
    }
    assert borders[ft.ControlState.DEFAULT].side == ft.BorderSide(1, COLOR_DIVIDER)
    assert borders[ft.ControlState.FOCUSED].side == ft.BorderSide(2, COLOR_PRIMARY)
    assert borders[ft.ControlState.ERROR].side == ft.BorderSide(1, COLOR_ERROR)
    assert borders[ft.ControlState.DISABLED].side == ft.BorderSide(1, COLOR_DIVIDER)
    assert all(border.border_radius == 12 for border in borders.values())


class _RaisingPageControl:
    @property
    def page(self):
        raise RuntimeError("not attached")


def test_attach_dummy_page_replaces_unreadable_page_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _RaisingPageControl()

    page = attach_dummy_page(monkeypatch, control)

    assert control.page is page
    assert bool(control.page) is True


def test_attach_dummy_page_only_changes_target_control_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _RaisingPageControl()
    other_control = _RaisingPageControl()

    page = attach_dummy_page(monkeypatch, control)

    assert control.page is page
    with pytest.raises(RuntimeError, match="not attached"):
        _ = other_control.page


def test_attach_dummy_page_uses_explicit_dummy_page(monkeypatch: pytest.MonkeyPatch) -> None:
    control = _RaisingPageControl()
    page = DummyPage()

    returned = attach_dummy_page(monkeypatch, control, page)
    returned.show_dialog("dialog")
    returned.pop_dialog()

    assert returned is page
    assert control.page is page
    assert page.opened == ["dialog"]
    assert page.closed == ["dialog"]
