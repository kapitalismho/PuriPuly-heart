from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

import flet as ft

from puripuly_heart.ui.components import language_card as language_card_module
from puripuly_heart.ui.components.language_card import LanguageCard
from puripuly_heart.ui.theme import COLOR_NEUTRAL_DARK, COLOR_PRIMARY, COLOR_SECONDARY


def _build_language_card() -> LanguageCard:
    return LanguageCard(
        on_self_source_click=lambda: None,
        on_self_target_click=lambda: None,
        on_self_swap_click=lambda: None,
        on_peer_source_click=lambda: None,
        on_peer_target_click=lambda: None,
        on_peer_swap_click=lambda: None,
    )


def _unwrap_container_content(control: ft.Control) -> ft.Control:
    if isinstance(control, ft.Container):
        return control.content
    return control


def test_language_card_weighted_len_counts_cjk_double_width() -> None:
    assert language_card_module._weighted_len("abc") == 3
    assert language_card_module._weighted_len("한a") == 3


@pytest.mark.parametrize(
    (
        "row_attr",
        "expected_label",
    ),
    [("_self_row", "My voice"), ("_peer_row", "Their voice")],
)
def test_language_card_rows_use_top_left_caption_with_centered_pair_layout(
    row_attr: str, expected_label: str
) -> None:
    card = _build_language_card()
    card.set_row_labels("My voice", "Their voice")

    row = getattr(card, row_attr)

    assert isinstance(row.content, ft.Column)
    assert len(row.content.controls) == 2
    assert row.content.horizontal_alignment == ft.CrossAxisAlignment.STRETCH
    assert row.content.spacing == 10

    caption_control = row.content.controls[0]
    assert isinstance(caption_control, ft.Container)
    assert caption_control.alignment == ft.Alignment.TOP_LEFT
    assert _unwrap_container_content(caption_control) is row._label_text
    assert row._label_text.value == expected_label
    assert row._label_text.size == 16

    pair_control = row.content.controls[1]
    assert isinstance(pair_control, ft.Container)
    assert pair_control.alignment == ft.Alignment.CENTER
    pair_row = _unwrap_container_content(pair_control)
    assert isinstance(pair_row, ft.Row)
    assert pair_row.alignment == ft.MainAxisAlignment.CENTER
    assert pair_row.spacing == 10
    assert pair_row.controls == [row._source_btn, row._arrow, row._target_btn]


def test_language_card_uses_more_air_between_self_and_peer_rows() -> None:
    card = _build_language_card()

    assert isinstance(card.content, ft.Stack)
    content_wrapper = card.content.controls[1]
    assert isinstance(content_wrapper, ft.Container)
    inner_container = content_wrapper.content
    assert isinstance(inner_container, ft.Container)
    rows_column = inner_container.content
    assert isinstance(rows_column, ft.Column)
    assert rows_column.spacing == 20


@pytest.mark.parametrize("row_attr", ["_self_row", "_peer_row"])
def test_language_card_rows_have_no_inner_chrome(row_attr: str) -> None:
    card = _build_language_card()

    row = getattr(card, row_attr)

    assert row.bgcolor is None
    assert row.border is None


def test_language_card_hover_and_set_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    card = _build_language_card()
    monkeypatch.setattr(type(card._self_row._source_text), "update", lambda self: None)
    monkeypatch.setattr(type(card._self_row._target_text), "update", lambda self: None)
    monkeypatch.setattr(type(card._self_row._arrow_icon), "update", lambda self: None)
    monkeypatch.setattr(type(card._peer_row._source_text), "update", lambda self: None)
    monkeypatch.setattr(type(card._peer_row._target_text), "update", lambda self: None)
    monkeypatch.setattr(type(card._peer_row._arrow_icon), "update", lambda self: None)

    card._self_row._on_source_hover(SimpleNamespace(data="true"))
    assert card._self_row._source_text.color == COLOR_PRIMARY
    card._self_row._on_source_hover(SimpleNamespace(data="false"))
    assert card._self_row._source_text.color == COLOR_NEUTRAL_DARK

    card._self_row._on_target_hover(SimpleNamespace(data="true"))
    assert card._self_row._target_text.color == COLOR_PRIMARY
    card._self_row._on_target_hover(SimpleNamespace(data="false"))
    assert card._self_row._target_text.color == COLOR_NEUTRAL_DARK

    card._peer_row._on_arrow_hover(SimpleNamespace(data="true"))
    assert card._peer_row._arrow_icon.color == COLOR_PRIMARY
    card._peer_row._on_arrow_hover(SimpleNamespace(data="false"))
    assert card._peer_row._arrow_icon.color == COLOR_SECONDARY

    card.set_row_labels("My voice", "Their voice")
    card.set_languages("Korean", "English", "Japanese", "French")
    expected_short_size = min(
        language_card_module._row_text_size("Korean", "English"),
        language_card_module._row_text_size("Japanese", "French"),
    )

    assert card._self_row._label_text.value == "My voice"
    assert card._peer_row._label_text.value == "Their voice"
    assert card._self_row._source_text.size == expected_short_size
    assert card._self_row._target_text.size == expected_short_size
    assert card._self_row._arrow_icon.size == expected_short_size + 4
    assert card._peer_row._target_text.value == "French"

    card.set_row_labels("我的语音", "对方语音")
    assert card._self_row._label_text.value == "我的语音"
    assert card._peer_row._label_text.value == "对方语音"

    card.set_languages("A" * 24, "B" * 24, "C" * 24, "D" * 24)
    expected_long_size = language_card_module._row_text_size("A" * 24, "B" * 24)
    assert card._self_row._source_text.size == expected_long_size
    assert card._self_row._target_text.size == expected_long_size
    assert card._self_row._arrow_icon.size == expected_long_size + 4
    assert card._peer_row._target_text.value == "D" * 24


def test_language_card_uses_unified_row_size_for_self_and_peer_visual_balance() -> None:
    card = _build_language_card()

    card.set_languages("Ko", "En", "A very long peer source language", "A very long peer target")

    expected_size = min(
        language_card_module._row_text_size("Ko", "En"),
        language_card_module._row_text_size(
            "A very long peer source language",
            "A very long peer target",
        ),
    )
    assert card._self_row._source_text.size == expected_size
    assert card._self_row._target_text.size == expected_size
    assert card._peer_row._source_text.size == expected_size
    assert card._peer_row._target_text.size == expected_size
    assert card._self_row._arrow_icon.size == expected_size + 4
    assert card._peer_row._arrow_icon.size == expected_size + 4


def test_language_card_uses_single_direction_arrow_icon() -> None:
    card = _build_language_card()

    assert card._self_row._arrow_icon.icon == ft.Icons.ARROW_RIGHT_ALT
    assert card._peer_row._arrow_icon.icon == ft.Icons.ARROW_RIGHT_ALT
