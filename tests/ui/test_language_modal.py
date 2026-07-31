from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.helpers.flet_page import DummyPage

pytest.importorskip("flet")

import flet as ft

from puripuly_heart.ui.components import language_modal as modal_module
from puripuly_heart.ui.components.language_modal import LanguageModal


def test_language_modal_open_builds_dialog_and_recent_grid() -> None:
    page = DummyPage()
    selected: list[str] = []
    modal = LanguageModal(
        page=page,
        languages=[("ko", "Korean"), ("en", "English"), ("ja", "Japanese")],
        on_select=lambda code: selected.append(code),
    )

    modal.open(current="ko", recent=["ko", "en", "ja"])

    assert len(page.opened) == 1
    assert modal._dialog is not None

    grid = modal._build_recent_grid(["ko", "en", "ja"], current="ko")
    assert isinstance(grid, ft.Row)
    assert len(grid.controls) == 3
    assert modal._build_recent_grid([], current="ko") is None


def test_language_modal_builds_list_and_handles_hover_callbacks() -> None:
    page = DummyPage()
    modal = LanguageModal(
        page=page,
        languages=[("ko", "Korean"), ("en", "English")],
        on_select=lambda _code: None,
    )

    lang_list = modal._build_language_list(current="ko")
    assert isinstance(lang_list, ft.ListView)
    assert len(lang_list.controls) == 2

    # Non-selected chip hover: color toggles and update is called.
    chip_text = SimpleNamespace(color=ft.Colors.BLACK)
    chip = SimpleNamespace(content=chip_text, update=lambda: setattr(chip, "updated", True))
    modal._on_chip_hover(SimpleNamespace(control=chip, data="true"))
    assert chip_text.color == modal_module.COLOR_PRIMARY
    modal._on_chip_hover(SimpleNamespace(control=chip, data="false"))
    assert chip_text.color == modal_module.COLOR_NEUTRAL_DARK

    # Selected list item hover: should be ignored.
    selected_text = SimpleNamespace(color=ft.Colors.WHITE)
    selected_item = SimpleNamespace(
        content=selected_text, update=lambda: setattr(selected_item, "updated", True)
    )
    modal._on_item_hover(SimpleNamespace(control=selected_item, data="true"))
    assert not hasattr(selected_item, "updated")

    normal_text = SimpleNamespace(color=modal_module.COLOR_NEUTRAL_DARK)
    normal_item = SimpleNamespace(
        content=normal_text, update=lambda: setattr(normal_item, "updated", True)
    )
    modal._on_item_hover(SimpleNamespace(control=normal_item, data="true"))
    assert normal_text.color == modal_module.COLOR_PRIMARY
    modal._on_item_hover(SimpleNamespace(control=normal_item, data="false"))
    assert normal_text.color == modal_module.COLOR_NEUTRAL_DARK


def test_language_modal_select_closes_dialog_and_calls_callback() -> None:
    page = DummyPage()
    selected: list[str] = []
    modal = LanguageModal(
        page=page,
        languages=[("ko", "Korean")],
        on_select=lambda code: selected.append(code),
    )

    modal.open(current="ko", recent=["ko"])
    modal._select("en")

    assert selected == ["en"]
    assert page.closed == [modal._dialog]
