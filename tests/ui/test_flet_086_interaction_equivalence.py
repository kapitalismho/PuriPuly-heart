"""Equivalence contracts for interaction behavior that changed between Flet 0.28 and 0.86.

Verified against both runtimes with mouse-level probes:

* ``Container.on_hover`` payload: ``"true"``/``"false"`` strings in 0.28.3 became
  ``True``/``False`` booleans in 0.86.1.
* A ``TextField`` without ``width``/``expand`` filled its parent in 0.28.3 but is
  constrained to a fixed 300px box in 0.86.1.
* Material buttons showed the click cursor in 0.28.3 but the default arrow in 0.86.1
  unless a mouse cursor is supplied through the style or the theme.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

import flet as ft

from puripuly_heart.ui.components.bottom_nav import BottomNavBar
from puripuly_heart.ui.components.language_card import LanguageCard
from puripuly_heart.ui.components.settings.prompt_editor import PromptEditor
from puripuly_heart.ui.components.subtab_shell import TextSubtab, TextSubtabShell
from puripuly_heart.ui.flet_runtime import FILL_PARENT_WIDTH, is_hover_active
from puripuly_heart.ui.theme import (
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    get_app_theme,
)
from tests.helpers.paths import REPO_ROOT as ROOT

UI_ROOT = ROOT / "src" / "puripuly_heart" / "ui"
HOVER_ENTER_PAYLOADS = [True, "true", "True"]
HOVER_EXIT_PAYLOADS = [False, "false", None]


@pytest.mark.parametrize("payload", HOVER_ENTER_PAYLOADS)
def test_is_hover_active_accepts_enter_payloads_of_both_runtimes(payload: object) -> None:
    assert is_hover_active(SimpleNamespace(data=payload)) is True


@pytest.mark.parametrize("payload", HOVER_EXIT_PAYLOADS)
def test_is_hover_active_accepts_exit_payloads_of_both_runtimes(payload: object) -> None:
    assert is_hover_active(SimpleNamespace(data=payload)) is False


def test_is_hover_active_tolerates_events_without_data() -> None:
    assert is_hover_active(SimpleNamespace()) is False


def test_no_hover_handler_compares_event_data_to_a_string() -> None:
    violations = []
    for path in sorted(UI_ROOT.rglob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if 'data == "true"' in line or "data == 'true'" in line:
                violations.append(f"{path.relative_to(ROOT)}:{line_number}:{line.strip()}")

    assert violations == []


def test_language_card_hover_reacts_to_boolean_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    card = LanguageCard(
        on_self_source_click=lambda: None,
        on_self_target_click=lambda: None,
        on_self_swap_click=lambda: None,
        on_peer_source_click=lambda: None,
        on_peer_target_click=lambda: None,
        on_peer_swap_click=lambda: None,
    )
    monkeypatch.setattr(type(card._self_row._source_text), "update", lambda self: None)

    card._self_row._on_source_hover(SimpleNamespace(data=True))
    assert card._self_row._source_text.color == COLOR_PRIMARY

    card._self_row._on_source_hover(SimpleNamespace(data=False))
    assert card._self_row._source_text.color == COLOR_NEUTRAL_DARK


def test_bottom_nav_hover_reacts_to_boolean_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    nav = BottomNavBar(on_change=lambda _index: None)
    icon = nav._icons[1]
    monkeypatch.setattr(type(icon), "update", lambda self: None)

    nav._on_tab_hover(SimpleNamespace(data=True), 1)
    assert icon.color == COLOR_PRIMARY

    nav._on_tab_hover(SimpleNamespace(data=False), 1)
    assert icon.color == COLOR_SECONDARY


def test_bottom_docked_subtab_hover_reacts_to_boolean_payloads() -> None:
    shell = TextSubtabShell(
        tabs=[
            TextSubtab(key="first", label="First", controls=()),
            TextSubtab(key="second", label="Second", controls=()),
        ],
        subtab_bar_position="bottom",
    )
    inactive_tab = shell.button_by_key["second"]

    inactive_tab._handle_hover(SimpleNamespace(data=True))
    assert inactive_tab.label.color == COLOR_PRIMARY

    inactive_tab._handle_hover(SimpleNamespace(data=False))
    assert inactive_tab.label.color == COLOR_SECONDARY


def test_prompt_editor_field_fills_available_width() -> None:
    editor = PromptEditor()

    assert editor._text_field.width == FILL_PARENT_WIDTH


def test_every_text_field_declares_an_explicit_width_or_expand() -> None:
    unsized = []
    for path in sorted((ROOT / "src" / "puripuly_heart").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name != "TextField":
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            if not keywords & {"width", "expand"}:
                unsized.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert unsized == []


def test_app_theme_restores_the_click_cursor_on_material_buttons() -> None:
    theme = get_app_theme(font_family="Pretendard")

    button_themes = (
        theme.button_theme,
        theme.text_button_theme,
        theme.outlined_button_theme,
        theme.filled_button_theme,
        theme.icon_button_theme,
    )
    for button_theme in button_themes:
        assert button_theme is not None
        assert button_theme.style is not None
        assert button_theme.style.mouse_cursor == {
            ft.ControlState.DISABLED: ft.MouseCursor.BASIC,
            ft.ControlState.DEFAULT: ft.MouseCursor.CLICK,
        }
