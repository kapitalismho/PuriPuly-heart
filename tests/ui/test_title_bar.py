from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

from puripuly_heart.ui.components.title_bar import TitleBar
from puripuly_heart.ui.theme import COLOR_NEUTRAL_DARK, COLOR_SECONDARY


class DummyWindow:
    def __init__(self) -> None:
        self.minimized = False
        self.maximized = False


class DummyPage:
    def __init__(self) -> None:
        self.window = DummyWindow()
        self.updated = 0

    def update(self) -> None:
        self.updated += 1


def test_title_bar_window_controls_and_hover(monkeypatch: pytest.MonkeyPatch) -> None:
    page = DummyPage()
    close_calls: list[str] = []
    bar = TitleBar(page, on_close=lambda: close_calls.append("close"))
    monkeypatch.setattr(type(bar._title_text), "update", lambda self: None)

    bar._minimize(None)
    assert page.window.minimized is True
    assert not hasattr(bar, "_maximize_btn")
    assert page.window.maximized is False

    bar._close(None)
    assert close_calls == ["close"]
    assert page.updated == 1

    icon = SimpleNamespace(color=COLOR_SECONDARY, update=lambda: None)
    bar._on_btn_hover(SimpleNamespace(control=SimpleNamespace(content=icon), data="true"))
    assert icon.color == COLOR_NEUTRAL_DARK
    bar._on_btn_hover(SimpleNamespace(control=SimpleNamespace(content=icon), data="false"))
    assert icon.color == COLOR_SECONDARY

    close_container = SimpleNamespace(
        content=SimpleNamespace(color=COLOR_SECONDARY, update=lambda: None),
        bgcolor=None,
        update=lambda: None,
    )
    bar._on_close_hover(SimpleNamespace(control=close_container, data="true"))
    assert close_container.content.color != COLOR_SECONDARY
    bar._on_close_hover(SimpleNamespace(control=close_container, data="false"))
    assert close_container.content.color == COLOR_SECONDARY


def test_title_bar_set_title_updates_text(monkeypatch: pytest.MonkeyPatch) -> None:
    bar = TitleBar(DummyPage(), on_close=lambda: None)
    monkeypatch.setattr(type(bar._title_text), "update", lambda self: None)
    bar.set_title("New Title")
    assert bar._title_text.value == "New Title"
