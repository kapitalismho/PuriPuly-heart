from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

from puripuly_heart.ui.components import display_card as display_card_module
from puripuly_heart.ui.components.display_card import DisplayCard
from tests.helpers.flet_page import attach_dummy_page


class RuntimeLoggingCapture:
    def __init__(self, *, detailed_enabled: bool = True) -> None:
        self.detailed_enabled = detailed_enabled
        self.detailed_calls: list[tuple[int, str]] = []
        self.detailed_messages: list[tuple[int, str]] = []

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool:
        self.detailed_calls.append((level, message))
        if not self.detailed_enabled:
            return False
        self.detailed_messages.append((level, message))
        return True


def _visible_text(card: DisplayCard) -> str:
    return card._display_text.value or ""


def _visible_color(card: DisplayCard) -> str:
    return card._display_text.color


def _visible_size(card: DisplayCard) -> int:
    return card._display_text.size


def _mute_display_updates(monkeypatch: pytest.MonkeyPatch, card: DisplayCard) -> None:
    monkeypatch.setattr(type(card._display_text), "update", lambda self: None)


def _input_row(card: DisplayCard):
    main_content = card.content.content
    _display_region, input_footer = main_content.controls
    return input_footer.controls[1]


def test_weighted_length_counts_cjk_as_two() -> None:
    assert display_card_module._weighted_len("abc") == 3
    assert display_card_module._weighted_len("한a") == 3


@pytest.mark.parametrize(
    ("size", "expected_lines", "expected_capacity"),
    [
        (48, 3, 65),
        (44, 3, 71),
        (40, 4, 105),
        (36, 4, 116),
        (32, 5, 164),
        (28, 5, 187),
        (24, 6, 263),
    ],
)
def test_smaller_sizes_are_allowed_more_lines_and_hold_more_text(
    size: int, expected_lines: int, expected_capacity: int
) -> None:
    assert display_card_module._lines_for_size(size) == expected_lines
    assert display_card_module._capacity_for_size(size) == expected_capacity


def test_capacity_grows_as_the_size_shrinks() -> None:
    capacities = [
        display_card_module._capacity_for_size(size)
        for size in display_card_module.DISPLAY_SIZE_CANDIDATES
    ]
    line_counts = [
        display_card_module._lines_for_size(size)
        for size in display_card_module.DISPLAY_SIZE_CANDIDATES
    ]

    assert capacities == sorted(capacities)
    assert line_counts == sorted(line_counts)


def test_layout_picks_the_largest_size_whose_capacity_holds_the_text() -> None:
    for size in display_card_module.DISPLAY_SIZE_CANDIDATES:
        capacity = display_card_module._capacity_for_size(size)
        assert display_card_module._display_layout_for_length(capacity) == (
            size,
            display_card_module._lines_for_size(size),
        )


def test_layout_falls_back_to_the_smallest_size_beyond_every_capacity() -> None:
    smallest = display_card_module.DISPLAY_SIZE_CANDIDATES[-1]
    overflowing = display_card_module._capacity_for_size(smallest) + 1

    assert display_card_module._display_layout_for_length(overflowing) == (
        smallest,
        display_card_module._lines_for_size(smallest),
    )


def test_display_size_never_grows_with_length() -> None:
    sizes = [display_card_module._display_layout_for_length(length)[0] for length in range(0, 400)]

    assert sizes == sorted(sizes, reverse=True)
    assert max(sizes) == display_card_module.DISPLAY_SIZE_CANDIDATES[0]
    assert min(sizes) == display_card_module.DISPLAY_SIZE_CANDIDATES[-1]


def test_status_label_maps_known_states() -> None:
    assert display_card_module._status_label("connecting") == display_card_module.t(
        "display.connecting"
    )
    assert display_card_module._status_label("connected") == display_card_module.t(
        "display.connected"
    )
    assert display_card_module._status_label("stopping") == display_card_module.t(
        "display.stopping"
    )
    assert display_card_module._status_label("other") == display_card_module.t(
        "display.disconnected"
    )


def test_only_the_source_text_uses_the_dimmed_colour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    assert display_card_module.DISPLAY_SOURCE_COLOR != display_card_module.DISPLAY_MESSAGE_COLOR

    card.set_display("source text", font_family="font-source")
    assert _visible_text(card) == "source text"
    assert _visible_color(card) == display_card_module.DISPLAY_SOURCE_COLOR
    assert card._display_text.weight == display_card_module.DISPLAY_SOURCE_WEIGHT
    assert card._display_text.font_family == "font-source"

    card.set_display_translation("translated text", font_family="font-target")
    assert _visible_text(card) == "translated text"
    assert _visible_color(card) == display_card_module.DISPLAY_MESSAGE_COLOR
    assert card._display_text.weight == display_card_module.DISPLAY_MESSAGE_WEIGHT
    assert card._display_text.font_family == "font-target"


def test_status_and_notice_keep_the_undimmed_message_colour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_status("connected")
    assert _visible_color(card) == display_card_module.DISPLAY_MESSAGE_COLOR
    assert card._display_text.weight == display_card_module.DISPLAY_MESSAGE_WEIGHT

    card.set_notice("STT files are missing", tone="warning")
    assert _visible_color(card) == display_card_module.DISPLAY_MESSAGE_COLOR
    assert card._display_text.weight == display_card_module.DISPLAY_MESSAGE_WEIGHT


def test_a_notice_over_a_live_source_turn_still_uses_the_message_colour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_display("source text")
    assert _visible_color(card) == display_card_module.DISPLAY_SOURCE_COLOR

    card.set_notice("STT files are missing", tone="warning")
    assert _visible_text(card) == "STT files are missing"
    assert _visible_color(card) == display_card_module.DISPLAY_MESSAGE_COLOR

    card.set_notice(None, None)
    assert _visible_text(card) == "source text"
    assert _visible_color(card) == display_card_module.DISPLAY_SOURCE_COLOR


def test_the_default_display_state_is_an_undimmed_status_label() -> None:
    card = DisplayCard(on_submit=lambda _text: None)

    assert card._display_text.color == display_card_module.DISPLAY_MESSAGE_COLOR
    assert card._display_text.weight == display_card_module.DISPLAY_MESSAGE_WEIGHT


def test_translation_replaces_the_source_in_the_same_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_display("source text")
    card.set_display_translation("translated text")

    assert _visible_text(card) == "translated text"
    assert "source text" not in _visible_text(card)


def test_new_source_drops_the_previous_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_display("first source")
    card.set_display_translation("first translation")
    card.set_display("second source")

    assert _visible_text(card) == "second source"
    assert _visible_color(card) == display_card_module.DISPLAY_SOURCE_COLOR


def test_clearing_the_translation_restores_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_display("source text")
    card.set_display_translation("translated text")
    card.set_display_translation(None)

    assert _visible_text(card) == "source text"
    assert _visible_color(card) == display_card_module.DISPLAY_SOURCE_COLOR


def test_status_replaces_any_turn_content(monkeypatch: pytest.MonkeyPatch) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_display("source text")
    card.set_display_translation("translated text")
    card.set_status("connected", font_family="font-status")

    assert _visible_text(card) == display_card_module.t("display.connected")
    assert card._display_text.font_family == "font-status"


def test_display_text_is_selectable_and_wraps_within_the_line_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    assert card._display_text.selectable is True
    assert card._display_text.no_wrap is False
    assert card._display_text.overflow == display_card_module.ft.TextOverflow.ELLIPSIS

    card.set_display("hi")
    biggest_size = _visible_size(card)
    fewest_lines = card._display_text.max_lines
    assert card._display_text.max_lines == display_card_module._lines_for_size(biggest_size)

    card.set_display("x" * 300)
    smallest_size = _visible_size(card)
    assert smallest_size < biggest_size
    assert card._display_text.max_lines > fewest_lines
    assert card._display_text.max_lines == display_card_module._lines_for_size(smallest_size)


def test_only_the_source_colour_is_dimmed_relative_to_messages() -> None:
    assert (
        display_card_module.DISPLAY_SOURCE_COLOR
        == display_card_module.COLOR_DISPLAY_SOURCE
        == "#A6706D"
    )
    assert (
        display_card_module.DISPLAY_MESSAGE_COLOR
        == display_card_module.COLOR_NEUTRAL_DARK
        == "#5C4D4C"
    )
    assert display_card_module.DISPLAY_SOURCE_WEIGHT != display_card_module.DISPLAY_MESSAGE_WEIGHT
    assert display_card_module.DISPLAY_MESSAGE_WEIGHT == display_card_module.ft.FontWeight.BOLD


def test_display_region_anchors_the_first_line_to_the_top() -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    display_region = card.content.content.controls[0]

    assert display_region.alignment == display_card_module.ft.Alignment.TOP_LEFT
    assert (
        display_region.content.vertical_alignment == display_card_module.ft.CrossAxisAlignment.START
    )


def test_the_first_line_anchor_is_independent_of_text_length_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)
    display_region = card.content.content.controls[0]

    def anchor() -> tuple[object, object, object]:
        return (
            display_region.alignment,
            display_region.content.vertical_alignment,
            display_region.padding,
        )

    card.set_display("hi")
    short_anchor = anchor()
    assert _visible_size(card) == 48

    card.set_display("x" * 200)
    assert _visible_size(card) == 24
    assert anchor() == short_anchor

    card.set_display_translation("y" * 200)
    assert anchor() == short_anchor


def test_translation_never_grows_the_text_beyond_its_source_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_display("x" * 300)
    source_size = _visible_size(card)
    source_lines = card._display_text.max_lines
    assert source_size == 24

    card.set_display_translation("hi")
    assert _visible_size(card) == source_size
    assert card._display_text.max_lines == source_lines


def test_translation_may_shrink_the_text_below_its_source_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_display("hi")
    assert _visible_size(card) == 48
    assert card._display_text.max_lines == 3

    card.set_display_translation("x" * 300)
    assert _visible_size(card) == 24
    assert card._display_text.max_lines == 6


def test_each_turn_resets_the_size_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_display("x" * 300)
    card.set_display_translation("hi")
    assert _visible_size(card) == 24

    card.set_display("hi")
    card.set_display_translation("hello")
    assert _visible_size(card) == 48


def test_blocking_notice_does_not_lower_the_turn_size_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_display("hi")
    card.set_notice("x" * 300, tone="warning")
    assert _visible_size(card) == 24

    card.set_notice(None, None)
    card.set_display_translation("hello")
    assert _visible_size(card) == 48


def test_turn_size_ceiling_survives_a_source_hidden_behind_a_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_notice("busy", tone="warning")
    card.set_display("x" * 300)
    card.set_display_translation("hi")
    assert _visible_text(card) == "busy"

    card.set_notice(None, None)

    assert _visible_text(card) == "hi"
    assert _visible_size(card) == 24


def test_turn_size_ceiling_accounts_for_the_debug_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    source = "x" * 60
    source_prefix = "[P 41c6/src]"
    unprefixed_size = display_card_module._display_layout_for_length(
        display_card_module._weighted_len(source)
    )[0]
    prefixed_size = display_card_module._display_layout_for_length(
        display_card_module._weighted_len(
            display_card_module._apply_debug_prefix(source, source_prefix)
        )
    )[0]
    assert prefixed_size < unprefixed_size

    card.set_display(source, debug_prefix=source_prefix)
    assert _visible_size(card) == prefixed_size

    card.set_display_translation("hi", debug_prefix="[P 41c6/3bd7]")

    assert _visible_size(card) == prefixed_size


def test_notice_takes_the_display_slot_and_restores_after_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_status("other", font_family="status-font")
    card.set_notice("STT files are missing", tone="warning")
    assert _visible_text(card) == "STT files are missing"

    card.set_notice(None, None)
    assert _visible_text(card) == display_card_module.t("display.disconnected")
    assert card._display_text.font_family == "status-font"

    card.set_notice("STT 다운로드 중 63%", tone="info")
    assert _visible_text(card) == "STT 다운로드 중 63%"


def test_notice_action_button_is_exposed_only_with_a_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions: list[str] = []
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_notice("needs attention", tone="warning")
    assert card._notice_action_button.visible is False

    card.set_notice(
        "needs attention",
        tone="warning",
        action_label="Fix",
        on_action=lambda: actions.append("fixed"),
    )
    assert card._notice_action_button.visible is True
    assert card._notice_action_button.content == "Fix"

    card._run_notice_action()
    assert actions == ["fixed"]


def test_yielding_notice_defers_to_live_turn_content(monkeypatch: pytest.MonkeyPatch) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    runtime_logging = RuntimeLoggingCapture()

    card.set_display("source text")
    card.set_notice(
        "SteamVR isn't on. Ran the desktop overlay instead.",
        tone="info",
        yields_to_content=True,
    )
    assert _visible_text(card) == "source text"

    attach_dummy_page(monkeypatch, card._display_text)
    _mute_display_updates(monkeypatch, card)

    card.set_display_translation(
        "translated text",
        runtime_log_detailed=runtime_logging.emit_detailed,
        update_id="upd-notice-1",
        origin_wall_clock_ms=1500,
        utterance_id="utt-notice-1",
        channel="self",
        session_scope="session-notice-1",
        source_text_hash="src-hash-notice-1",
        source_text_len=11,
        logical_turn_key="self:utt-notice-1",
    )

    assert _visible_text(card) == "translated text"
    assert len(runtime_logging.detailed_messages) == 1


def test_yielding_notice_returns_once_the_card_is_idle_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)
    notice = "SteamVR isn't on. Ran the desktop overlay instead."

    card.set_status("connected")
    card.set_notice(notice, tone="info", yields_to_content=True)
    assert _visible_text(card) == notice

    card.set_display("source text")
    card.set_display_translation("translated text")
    assert _visible_text(card) == "translated text"

    card.set_status("connected")
    assert _visible_text(card) == notice


def test_non_yielding_notice_suppresses_the_translation_and_its_visual_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    runtime_logging = RuntimeLoggingCapture()

    card.set_display("source text")
    card.set_notice("STT files are missing", tone="warning")
    attach_dummy_page(monkeypatch, card._display_text)
    _mute_display_updates(monkeypatch, card)

    card.set_display_translation(
        "translated text",
        runtime_log_detailed=runtime_logging.emit_detailed,
        update_id="upd-notice-2",
        origin_wall_clock_ms=1500,
        utterance_id="utt-notice-2",
        channel="self",
        session_scope="session-notice-2",
        source_text_hash="src-hash-notice-2",
        source_text_len=11,
        logical_turn_key="self:utt-notice-2",
    )

    assert _visible_text(card) == "STT files are missing"
    assert runtime_logging.detailed_calls == []
    assert runtime_logging.detailed_messages == []


def test_debug_prefix_applies_to_the_visible_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    _mute_display_updates(monkeypatch, card)

    card.set_display("source text", debug_prefix="[P 41c6/src]")
    assert _visible_text(card) == "[P 41c6/src] source text"

    card.set_display_translation("translated text", debug_prefix="[P 41c6/3bd7]")
    assert _visible_text(card) == "[P 41c6/3bd7] translated text"

    card.set_status("connected")
    assert _visible_text(card) == display_card_module.t("display.connected")
    assert "[P 41c6" not in _visible_text(card)


def test_input_row_holds_only_the_input_field() -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    controls = _input_row(card).controls

    assert controls == [card._input_field]
    assert all(getattr(control, "value", None) != "•" for control in controls)


def test_clear_input_empties_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    monkeypatch.setattr(type(card._input_field), "update", lambda self: None)

    card._input_field.value = "hello"
    card.clear_input()

    assert card._input_field.value == ""


def test_clear_input_empties_the_field_while_unmounted() -> None:
    card = DisplayCard(on_submit=lambda _text: None)

    def explode(_self) -> None:
        raise AssertionError("Control must be added to the page first")

    card._input_field.__class__ = type(
        "_UnmountedTextField",
        (type(card._input_field),),
        {"update": explode},
    )
    card._input_field.value = "hello"

    card.clear_input()

    assert card._input_field.value == ""


@pytest.mark.asyncio
async def test_submit_trims_clears_and_refocuses(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[str] = []
    tasks: list[asyncio.Task[object]] = []

    def run_task(func, *args):
        task = asyncio.create_task(func(*args))
        tasks.append(task)
        return task

    async def focus() -> None:
        submitted.append("focused")

    card = DisplayCard(on_submit=lambda text: submitted.append(text))
    monkeypatch.setattr(type(card._input_field), "update", lambda self: None)

    event = SimpleNamespace(
        control=SimpleNamespace(
            value="  hello  ",
            update=lambda: None,
            focus=focus,
            page=SimpleNamespace(run_task=run_task),
        )
    )
    card._handle_submit(event)
    await asyncio.gather(*tasks)

    assert submitted == ["hello", "focused"]
    assert event.control.value == ""


def test_blank_submit_is_ignored() -> None:
    submitted: list[str] = []
    card = DisplayCard(on_submit=submitted.append)

    card._handle_submit(
        SimpleNamespace(
            control=SimpleNamespace(value="   ", update=lambda: None, focus=lambda: None)
        )
    )

    assert submitted == []


@pytest.mark.asyncio
async def test_input_focus_is_tracked_and_can_be_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focus_changes: list[bool] = []
    focus_calls: list[str] = []
    tasks: list[asyncio.Task[object]] = []

    def run_task(func, *args):
        task = asyncio.create_task(func(*args))
        tasks.append(task)
        return task

    async def focus(_self) -> None:
        focus_calls.append("focus")

    card = DisplayCard(
        on_submit=lambda _text: None,
        on_input_focus_change=focus_changes.append,
    )
    monkeypatch.setattr(type(card._input_field), "focus", focus)
    attach_dummy_page(
        monkeypatch,
        card._input_field,
        SimpleNamespace(run_task=run_task),
    )

    card._handle_input_focus(SimpleNamespace(control=card._input_field))
    card._handle_input_blur(SimpleNamespace(control=card._input_field))
    card.focus_input()
    await asyncio.gather(*tasks)

    assert focus_changes == [True, False]
    assert card.input_is_focused is False
    assert focus_calls == ["focus"]


def test_input_activity_is_reported_without_exposing_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activity: list[bool] = []
    submitted: list[str] = []
    card = DisplayCard(on_submit=submitted.append, on_input_activity=activity.append)
    monkeypatch.setattr(type(card._input_field), "update", lambda self: None)

    card._handle_input_change(SimpleNamespace(control=SimpleNamespace(value="  hello  ")))
    card._handle_input_change(SimpleNamespace(control=SimpleNamespace(value="   ")))
    card._handle_input_blur(SimpleNamespace(control=SimpleNamespace(value="ignored")))
    card._handle_submit(
        SimpleNamespace(
            control=SimpleNamespace(value=" secret text ", update=lambda: None, focus=lambda: None)
        )
    )

    assert activity == [True, False, False, False]
    assert submitted == ["secret text"]
    assert all(isinstance(item, bool) for item in activity)


def test_input_font_and_locale_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    monkeypatch.setattr(type(card._input_field), "update", lambda self: None)
    _mute_display_updates(monkeypatch, card)

    card.set_input_font(None)
    assert card._input_field.text_style.font_family == ""

    card._showing_status = True
    card._status = "stopping"
    card.apply_locale(display_font_family="display-font", input_font_family="input-font")

    assert card._input_field.hint_text == display_card_module.t("display.input_hint")
    assert card._input_field.hint_style.font_family == "display-font"
    assert card._input_field.text_style.font_family == "input-font"
    assert _visible_text(card) == display_card_module.t("display.stopping")


def test_input_footer_stays_outside_the_expanding_display_region() -> None:
    card = DisplayCard(on_submit=lambda _text: None)

    padded_content_layer = card.content
    main_content = padded_content_layer.content
    display_region, input_footer = main_content.controls
    divider_container = input_footer.controls[0]

    assert main_content.alignment == display_card_module.ft.MainAxisAlignment.START
    assert display_region.expand is True
    assert display_region.clip_behavior == display_card_module.ft.ClipBehavior.HARD_EDGE
    assert divider_container.padding.bottom == 4
    assert input_footer.expand is None
    assert input_footer.tight is True


def test_translation_visual_commit_log_reports_the_redefined_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    runtime_logging = RuntimeLoggingCapture()
    events: list[str] = []

    card.set_display("source text", font_family="font-source")
    attach_dummy_page(monkeypatch, card._display_text)
    monkeypatch.setattr(type(card._display_text), "update", lambda self: events.append("display"))
    monkeypatch.setattr(display_card_module.time, "time", lambda: 2.0)

    def emit_detailed(message: str, *, level: int = logging.INFO) -> bool:
        events.append("log")
        return runtime_logging.emit_detailed(message, level=level)

    card.set_display_translation(
        "translated text",
        font_family="font-target",
        runtime_log_detailed=emit_detailed,
        update_id="upd-1",
        origin_wall_clock_ms=1500,
        utterance_id="utt-1",
        channel="peer",
        session_scope="session-1",
        source_text_hash="src-hash-1",
        source_text_len=11,
        logical_turn_key="peer:utt-1",
    )

    assert events == ["display", "log"]
    assert len(runtime_logging.detailed_messages) == 1
    level, message = runtime_logging.detailed_messages[0]
    assert level == logging.INFO
    assert "dashboard_translation_visual_commit" in message
    assert "update_id=upd-1" in message
    assert "origin_wall_clock_ms=1500" in message
    assert "utterance_id=utt-1" in message
    assert "channel=peer" in message
    assert "session_scope=session-1" in message
    assert "source_text_hash=src-hash-1" in message
    assert "source_text_len=11" in message
    assert "logical_turn_key=peer:utt-1" in message
    assert "source_display_text_len=11" in message
    assert "translation_text_len=15" in message
    assert "translation_visible=True" in message
    assert "display_update_issued=True" in message
    assert "elapsed_ms=500" in message
    assert "secondary_" not in message
    assert "source text" not in message
    assert "translated text" not in message


def test_translation_visual_commit_log_is_suppressed_in_basic_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    runtime_logging = RuntimeLoggingCapture(detailed_enabled=False)

    card.set_display("source text")
    attach_dummy_page(monkeypatch, card._display_text)
    _mute_display_updates(monkeypatch, card)

    card.set_display_translation(
        "translated text",
        runtime_log_detailed=runtime_logging.emit_detailed,
        update_id="upd-2",
        origin_wall_clock_ms=1500,
        utterance_id="utt-2",
        channel="self",
        session_scope="session-2",
        source_text_hash="src-hash-2",
        source_text_len=11,
        logical_turn_key="self:utt-2",
    )

    assert len(runtime_logging.detailed_calls) == 1
    assert runtime_logging.detailed_messages == []


def test_source_applied_log_reports_the_redefined_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    runtime_logging = RuntimeLoggingCapture()

    attach_dummy_page(monkeypatch, card._display_text)
    _mute_display_updates(monkeypatch, card)
    monkeypatch.setattr(display_card_module.time, "time", lambda: 2.0)

    card.set_display(
        "source text",
        runtime_log_detailed=runtime_logging.emit_detailed,
        origin_wall_clock_ms=1500,
        utterance_id="utt-3",
        channel="self",
        source_text_len=11,
        transcript_kind="final",
        should_log=True,
    )

    assert len(runtime_logging.detailed_messages) == 1
    _level, message = runtime_logging.detailed_messages[0]
    assert "dashboard_source_applied" in message
    assert "utterance_id=utt-3" in message
    assert "channel=self" in message
    assert "transcript_kind=final" in message
    assert "source_text_len=11" in message
    assert "source_display_text_len=11" in message
    assert "translation_visible=False" in message
    assert "display_update_issued=True" in message
    assert "elapsed_ms=500" in message
    assert "primary_" not in message
    assert "source text" not in message


def test_source_applied_log_is_skipped_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    runtime_logging = RuntimeLoggingCapture()

    attach_dummy_page(monkeypatch, card._display_text)
    _mute_display_updates(monkeypatch, card)

    card.set_display(
        "source text",
        runtime_log_detailed=runtime_logging.emit_detailed,
        utterance_id="utt-4",
        channel="self",
        should_log=False,
    )

    assert runtime_logging.detailed_calls == []
