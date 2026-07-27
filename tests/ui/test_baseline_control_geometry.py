from __future__ import annotations

import pytest

pytest.importorskip("flet")

import flet as ft

from puripuly_heart.ui.components.settings.prompt_editor import (
    PROMPT_FIELD_CONTENT_PADDING,
    PromptEditor,
)
from puripuly_heart.ui.theme import TEXT_BUTTON_PADDING, get_app_theme
from puripuly_heart.ui.views.logs import LogsView


def _padding_values(padding: object) -> tuple[float, float, float, float]:
    return (
        float(getattr(padding, "left")),
        float(getattr(padding, "top")),
        float(getattr(padding, "right")),
        float(getattr(padding, "bottom")),
    )


def test_text_button_padding_token_matches_baseline_geometry() -> None:
    assert _padding_values(TEXT_BUTTON_PADDING) == (8.0, 8.0, 8.0, 8.0)


def test_app_theme_pins_text_button_padding() -> None:
    theme = get_app_theme("Arial")
    for button_theme in (
        theme.button_theme,
        theme.text_button_theme,
        theme.outlined_button_theme,
        theme.filled_button_theme,
        theme.icon_button_theme,
    ):
        assert button_theme is not None
        assert button_theme.style is not None
        assert _padding_values(button_theme.style.padding) == (8.0, 8.0, 8.0, 8.0)


def test_logs_header_buttons_inherit_the_pinned_padding() -> None:
    view = LogsView.__new__(LogsView)
    style = LogsView._get_button_style(view, "Arial")
    assert style.padding is None


def test_prompt_field_content_padding_matches_baseline_geometry() -> None:
    assert _padding_values(PROMPT_FIELD_CONTENT_PADDING) == (8.0, 16.0, 8.0, 16.0)


def test_prompt_editor_applies_the_pinned_content_padding() -> None:
    editor = PromptEditor()
    field = editor.controls[0]
    assert isinstance(field, ft.TextField)
    assert _padding_values(field.content_padding) == (8.0, 16.0, 8.0, 16.0)
