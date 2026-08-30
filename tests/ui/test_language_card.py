from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

import flet as ft

from puripuly_heart.ui.components import language_card as language_card_module
from puripuly_heart.ui.components.language_card import LanguageCard
from puripuly_heart.ui.theme import (
    COLOR_DISPLAY_SOURCE,
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
)


def _build_language_card(clicks: dict[str, int] | None = None) -> LanguageCard:
    def record(name: str):
        def handler() -> None:
            if clicks is not None:
                clicks[name] = clicks.get(name, 0) + 1

        return handler

    return LanguageCard(
        on_self_source_click=record("self_source"),
        on_self_target_click=record("self_target"),
        on_self_swap_click=record("self_swap"),
        on_peer_source_click=record("peer_source"),
        on_peer_target_click=record("peer_target"),
        on_peer_swap_click=record("peer_swap"),
        on_self_secondary_target_click=record("self_secondary"),
    )


def _mute(monkeypatch: pytest.MonkeyPatch, card: LanguageCard) -> None:
    for row in (card._self_row, card._peer_row):
        slots = [row._source_slot, row._target_slot]
        if row._secondary_slot is not None:
            slots.append(row._secondary_slot)
        for slot in slots:
            monkeypatch.setattr(type(slot._text), "update", lambda self: None)
            monkeypatch.setattr(type(slot._placeholder_icon), "update", lambda self: None)
        monkeypatch.setattr(type(row._arrow_icon), "update", lambda self: None)


def test_weighted_len_counts_cjk_double_width() -> None:
    assert language_card_module._weighted_len("abc") == 3
    assert language_card_module._weighted_len("한a") == 3


def test_rows_use_the_dashboard_capture_icons() -> None:
    card = _build_language_card()

    assert card._self_row._row_icon.icon == ft.Icons.MIC
    assert card._peer_row._row_icon.icon == ft.Icons.RECORD_VOICE_OVER
    assert card._self_row._row_icon.color == COLOR_DISPLAY_SOURCE


def test_the_row_icon_is_its_own_leading_column_outside_the_source_slot() -> None:
    card = _build_language_card()

    for row in (card._self_row, card._peer_row):
        pair = row.content
        assert pair.controls[0] is row._icon_holder
        assert row._icon_holder.alignment == ft.Alignment.CENTER_LEFT
        assert row._icon_holder.padding is None
        assert row._icon_holder.on_click is None
        assert row._row_icon not in row._source_slot.content.controls


def test_card_padding_is_asymmetric_so_the_leading_icon_keeps_its_inset() -> None:
    from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS

    card = _build_language_card()
    padding = card.content.padding

    assert padding.left == padding.right == FOUNDATION_DESIGN_TOKENS.spacing.card
    assert padding.top == padding.bottom == FOUNDATION_DESIGN_TOKENS.spacing.inline
    assert padding.left > padding.top


def test_the_row_width_follows_the_horizontal_card_padding() -> None:
    assert language_card_module.LANGUAGE_ROW_WIDTH == (
        600 - 2 * language_card_module.LANGUAGE_CARD_HORIZONTAL_PADDING
    )


def test_rows_have_no_caption_line_and_keep_the_swap_arrow() -> None:
    card = _build_language_card()

    for row in (card._self_row, card._peer_row):
        pair = row.content
        assert isinstance(pair, ft.Row)
        assert pair.vertical_alignment == ft.CrossAxisAlignment.START
        assert pair.controls == [
            row._icon_holder,
            row._source_holder,
            row._arrow,
            row._target_holder,
        ]
        assert row._arrow_icon.icon == ft.Icons.ARROW_RIGHT_ALT
        assert row._arrow.on_click is not None


def test_rows_expose_no_tooltip() -> None:
    card = _build_language_card()

    assert not hasattr(card, "set_row_labels")
    assert card._self_row.tooltip is None
    assert card._peer_row.tooltip is None


def test_only_the_self_row_carries_a_secondary_target_slot() -> None:
    card = _build_language_card()

    assert card._self_row._secondary_slot is not None
    assert card._peer_row._secondary_slot is None
    assert card._self_row._target_column.controls == [
        card._self_row._target_slot,
        card._self_row._secondary_slot,
    ]
    assert card._peer_row._target_column.controls == [card._peer_row._target_slot]


def test_secondary_target_is_styled_exactly_like_the_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _build_language_card()
    _mute(monkeypatch, card)

    card.set_languages("한국어", "English", "English", "한국어", "日本語")

    primary = card._self_row._target_slot._text
    secondary = card._self_row._secondary_slot._text

    assert secondary.value == "日本語"
    assert secondary.color == primary.color == COLOR_NEUTRAL_DARK
    assert secondary.weight == primary.weight == ft.FontWeight.BOLD
    assert secondary.size == primary.size


def test_the_add_placeholder_keeps_its_own_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    card = _build_language_card()
    _mute(monkeypatch, card)

    card.set_languages("한국어", "English", "English", "한국어")
    secondary = card._self_row._secondary_slot

    assert secondary._placeholder_icon.color == COLOR_DISPLAY_SOURCE

    secondary._on_hover(SimpleNamespace(data="true"))
    assert secondary._placeholder_icon.color == COLOR_PRIMARY

    secondary._on_hover(SimpleNamespace(data="false"))
    assert secondary._placeholder_icon.color == COLOR_DISPLAY_SOURCE
    assert secondary._text.color == COLOR_NEUTRAL_DARK


def test_an_empty_secondary_target_hides_the_add_placeholder_until_the_target_area_is_hovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _build_language_card()
    _mute(monkeypatch, card)

    card.set_languages("한국어", "English", "English", "한국어")
    secondary = card._self_row._secondary_slot

    assert secondary._text.visible is False
    assert secondary._placeholder_icon.visible is True
    assert secondary._placeholder_icon.opacity == 0.0
    assert secondary._placeholder_icon.icon == language_card_module.SECONDARY_TARGET_ADD_ICON

    card._self_row._on_target_area_hover(SimpleNamespace(data="true"))
    assert secondary._placeholder_icon.opacity == 1.0

    card._self_row._on_target_area_hover(SimpleNamespace(data="false"))
    assert secondary._placeholder_icon.opacity == 0.0


def test_the_placeholder_fades_instead_of_toggling_layout() -> None:
    card = _build_language_card()
    placeholder = card._self_row._secondary_slot._placeholder_icon

    assert placeholder.animate_opacity is not None
    assert placeholder.animate_opacity.duration == language_card_module.LANGUAGE_HOVER_ANIMATION_MS


def test_hovering_the_target_area_never_reveals_a_placeholder_over_a_chosen_secondary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _build_language_card()
    _mute(monkeypatch, card)

    card.set_languages("한국어", "English", "English", "한국어", "日本語")
    secondary = card._self_row._secondary_slot

    card._self_row._on_target_area_hover(SimpleNamespace(data="true"))

    assert secondary._text.visible is True
    assert secondary._placeholder_icon.visible is False
    assert secondary._placeholder_icon.opacity == 0.0


def test_only_the_target_area_carries_the_reveal_hover() -> None:
    card = _build_language_card()
    row = card._self_row

    assert row.on_hover is None
    assert row._target_holder.on_hover is not None
    assert row._source_holder.on_hover is None
    assert row._icon_holder.on_hover is None


def test_the_peer_target_area_hover_has_no_secondary_slot_to_reveal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _build_language_card()
    _mute(monkeypatch, card)
    card.set_languages("한국어", "English", "English", "한국어")

    card._peer_row._on_target_area_hover(SimpleNamespace(data="true"))

    assert card._peer_row._secondary_slot is None


def test_the_reveal_survives_a_language_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    card = _build_language_card()
    _mute(monkeypatch, card)
    card.set_languages("한국어", "English", "English", "한국어")

    card._self_row._on_target_area_hover(SimpleNamespace(data="true"))
    card.set_languages("한국어", "Français", "English", "한국어")

    assert card._self_row._secondary_slot._placeholder_icon.opacity == 1.0


def test_the_blank_secondary_slot_stays_clickable(monkeypatch: pytest.MonkeyPatch) -> None:
    clicks: dict[str, int] = {}
    card = _build_language_card(clicks)
    _mute(monkeypatch, card)
    card.set_languages("한국어", "English", "English", "한국어")

    assert card._self_row._secondary_slot._placeholder_icon.opacity == 0.0
    card._self_row._secondary_slot.on_click(None)

    assert clicks["self_secondary"] == 1


def test_the_row_height_is_identical_with_and_without_a_secondary_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _build_language_card()
    _mute(monkeypatch, card)

    card.set_languages("한국어", "English", "English", "한국어", "日本語")
    with_secondary = (
        card._self_row._target_slot.height,
        card._self_row._secondary_slot.height,
        card._self_row._arrow.height,
    )

    card.set_languages("한국어", "English", "English", "한국어")
    without_secondary = (
        card._self_row._target_slot.height,
        card._self_row._secondary_slot.height,
        card._self_row._arrow.height,
    )

    assert with_secondary == without_secondary


def test_every_slot_opens_its_own_picker(monkeypatch: pytest.MonkeyPatch) -> None:
    clicks: dict[str, int] = {}
    card = _build_language_card(clicks)
    _mute(monkeypatch, card)

    card._self_row._source_slot.on_click(None)
    card._self_row._target_slot.on_click(None)
    card._self_row._secondary_slot.on_click(None)
    card._self_row._arrow.on_click(None)
    card._peer_row._source_slot.on_click(None)
    card._peer_row._target_slot.on_click(None)
    card._peer_row._arrow.on_click(None)

    assert clicks == {
        "self_source": 1,
        "self_target": 1,
        "self_secondary": 1,
        "self_swap": 1,
        "peer_source": 1,
        "peer_target": 1,
        "peer_swap": 1,
    }


def test_hover_highlights_the_slot_and_the_arrow(monkeypatch: pytest.MonkeyPatch) -> None:
    card = _build_language_card()
    _mute(monkeypatch, card)

    card._self_row._source_slot._on_hover(SimpleNamespace(data="true"))
    assert card._self_row._source_slot._text.color == COLOR_PRIMARY
    card._self_row._source_slot._on_hover(SimpleNamespace(data="false"))
    assert card._self_row._source_slot._text.color == COLOR_NEUTRAL_DARK

    card._self_row._secondary_slot._on_hover(SimpleNamespace(data="true"))
    assert card._self_row._secondary_slot._text.color == COLOR_PRIMARY
    card._self_row._secondary_slot._on_hover(SimpleNamespace(data="false"))
    assert card._self_row._secondary_slot._text.color == COLOR_NEUTRAL_DARK

    card._peer_row._on_arrow_hover(SimpleNamespace(data="true"))
    assert card._peer_row._arrow_icon.color == COLOR_PRIMARY
    card._peer_row._on_arrow_hover(SimpleNamespace(data="false"))
    assert card._peer_row._arrow_icon.color == COLOR_SECONDARY


def test_all_slots_share_one_font_size(monkeypatch: pytest.MonkeyPatch) -> None:
    card = _build_language_card()
    _mute(monkeypatch, card)

    card.set_languages("한국어", "English", "English", "한국어", "日本語")
    expected = language_card_module._row_text_size(
        source_texts=("한국어", "English"),
        target_texts=("English", "日本語", "한국어"),
        target_lines=2,
    )

    assert card._self_row._source_slot._text.size == expected
    assert card._self_row._target_slot._text.size == expected
    assert card._self_row._secondary_slot._text.size == expected
    assert card._peer_row._source_slot._text.size == expected
    assert card._peer_row._target_slot._text.size == expected
    assert (
        card._self_row._arrow_icon.size == expected + language_card_module.LANGUAGE_ARROW_SIZE_DELTA
    )


def test_a_long_language_name_shrinks_every_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    card = _build_language_card()
    _mute(monkeypatch, card)

    card.set_languages("한국어", "English", "English", "한국어")
    short = card._self_row._target_slot._text.size

    card.set_languages("한국어", "A" * 60, "English", "한국어")
    long = card._self_row._target_slot._text.size

    assert long < short
    assert card._peer_row._target_slot._text.size == long


def test_the_size_ladder_never_overflows_the_card_height() -> None:
    for size in language_card_module.LANGUAGE_SIZE_CANDIDATES:
        chosen = language_card_module._row_text_size(
            source_texts=("A" * 200,),
            target_texts=("B" * 200,),
            target_lines=2,
        )
        assert chosen == language_card_module.LANGUAGE_SIZE_CANDIDATES[-1]
        assert (
            language_card_module._card_height(chosen, target_lines=2)
            <= language_card_module.LANGUAGE_CARD_HEIGHT
        )
        assert size in language_card_module.LANGUAGE_SIZE_CANDIDATES


def test_the_common_case_uses_the_largest_size() -> None:
    chosen = language_card_module._row_text_size(
        source_texts=("한국어", "English"),
        target_texts=("English", "日本語", "한국어"),
        target_lines=2,
    )

    assert chosen == language_card_module.LANGUAGE_SIZE_CANDIDATES[0]
    assert (
        language_card_module._card_height(chosen, target_lines=2)
        <= language_card_module.LANGUAGE_CARD_HEIGHT
    )


def test_card_uses_foundation_spacing_tokens() -> None:
    from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS

    card = _build_language_card()
    rows_column = card.content.content

    assert rows_column.spacing == language_card_module.LANGUAGE_ROW_GAP
    assert card.border_radius == FOUNDATION_DESIGN_TOKENS.radius.card


def test_no_slot_carries_an_inert_hover_animation() -> None:
    card = _build_language_card()
    row = card._self_row

    for control in (row._source_slot, row._target_slot, row._secondary_slot, row._arrow):
        assert control.animate is None


def test_the_channel_gap_is_wider_than_the_gap_between_the_two_targets() -> None:
    assert language_card_module.LANGUAGE_ROW_GAP > language_card_module.LANGUAGE_TARGET_LINE_GAP * 4


@pytest.mark.parametrize("row_attr", ["_self_row", "_peer_row"])
def test_rows_have_no_inner_chrome(row_attr: str) -> None:
    card = _build_language_card()

    row = getattr(card, row_attr)

    assert row.bgcolor is None
    assert row.border is None
