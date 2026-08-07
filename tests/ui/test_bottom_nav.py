from __future__ import annotations

import pytest

pytest.importorskip("flet")

import flet as ft

from puripuly_heart.ui.components.bottom_nav import BottomNavBar
from puripuly_heart.ui.theme import COLOR_PRIMARY, COLOR_SECONDARY


def test_bottom_nav_builds_tabs_and_dividers() -> None:
    nav = BottomNavBar(on_change=lambda _idx: None)

    assert len(nav._icons) == 4
    controls = nav._build_tabs_with_dividers()
    assert len(controls) == 7  # 4 tabs + 3 dividers


def test_bottom_nav_click_and_hover_update_states(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft.Icon, "update", lambda self: None)
    seen: list[int] = []
    nav = BottomNavBar(on_change=lambda idx: seen.append(idx))

    nav._on_tab_click(0)
    assert seen == []

    nav._on_tab_click(2)
    assert nav._selected == 2
    assert seen == [2]
    assert nav._icons[2].color == COLOR_PRIMARY
    assert nav._icons[0].color == COLOR_SECONDARY

    nav._on_tab_hover(type("E", (), {"data": "true"})(), 1)
    assert nav._icons[1].color == COLOR_PRIMARY
    nav._on_tab_hover(type("E", (), {"data": "false"})(), 1)
    assert nav._icons[1].color == COLOR_SECONDARY

    # Hover on selected tab should not change color.
    nav._on_tab_hover(type("E", (), {"data": "true"})(), 2)
    assert nav._icons[2].color == COLOR_PRIMARY
