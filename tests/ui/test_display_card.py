from __future__ import annotations

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


def test_display_card_helpers_cover_length_and_status_labels() -> None:
    assert display_card_module._weighted_len("abc") == 3
    assert display_card_module._weighted_len("한a") == 3
    assert display_card_module._display_size_for_length(8) == 48
    assert display_card_module._display_size_for_length(16) == 40
    assert display_card_module._display_size_for_length(28) == 34
    assert display_card_module._display_size_for_length(40) == 28
    assert display_card_module._display_size_for_length(80) == 24
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


def test_display_card_submit_and_state_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[str] = []
    card = DisplayCard(on_submit=lambda text: submitted.append(text))
    monkeypatch.setattr(type(card._input_field), "update", lambda self: None)

    event = SimpleNamespace(
        control=SimpleNamespace(
            value="  hello  ",
            update=lambda: None,
            focus=lambda: submitted.append("focused"),
        )
    )
    card._handle_submit(event)
    assert submitted == ["hello", "focused"]
    assert event.control.value == ""

    card._handle_submit(
        SimpleNamespace(
            control=SimpleNamespace(value="   ", update=lambda: None, focus=lambda: None)
        )
    )
    assert submitted == ["hello", "focused"]

    card.set_display("primary", is_error=False, font_family="font-a")
    assert card._display_primary.value == "primary"
    assert card._display_secondary.visible is False
    assert card._display_primary.font_family == "font-a"

    card.set_display_translation("secondary", font_family="font-b")
    assert card._display_secondary.value == "secondary"
    assert card._display_secondary.visible is True
    assert card._display_secondary.font_family == "font-b"

    card.set_status("connected", font_family="font-c")
    assert card._showing_status is True
    assert card._display_primary.value == display_card_module.t("display.connected")
    assert card._display_primary.font_family == "font-c"


def test_display_card_tracks_input_focus_and_can_refocus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focus_changes: list[bool] = []
    focus_calls: list[str] = []
    card = DisplayCard(
        on_submit=lambda _text: None,
        on_input_focus_change=focus_changes.append,
    )
    monkeypatch.setattr(type(card._input_field), "focus", lambda self: focus_calls.append("focus"))

    card._handle_input_focus(SimpleNamespace(control=card._input_field))
    card._handle_input_blur(SimpleNamespace(control=card._input_field))
    card.focus_input()

    assert focus_changes == [True, False]
    assert card.input_is_focused is False
    assert focus_calls == ["focus"]


def test_display_card_reports_input_activity_without_exposing_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activity: list[bool] = []
    submitted: list[str] = []
    card = DisplayCard(
        on_submit=submitted.append,
        on_input_activity=activity.append,
    )
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


def test_display_card_input_font_locale_and_sync_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    monkeypatch.setattr(type(card._input_field), "update", lambda self: None)
    monkeypatch.setattr(type(card._display_primary), "update", lambda self: None)
    monkeypatch.setattr(type(card._display_secondary), "update", lambda self: None)

    card.set_input_font(None)
    assert card._input_field.text_style.font_family == ""

    card.set_display("x" * 50, font_family="font-long")
    assert card._display_primary.size == 24

    card.set_display_translation(None)
    assert card._display_secondary.visible is False

    card._showing_status = True
    card._status = "stopping"
    card.apply_locale(display_font_family="display-font", input_font_family="input-font")

    assert card._input_field.hint_text == display_card_module.t("display.input_hint")
    assert card._input_field.hint_style.font_family == "display-font"
    assert card._input_field.text_style.font_family == "input-font"
    assert card._display_primary.value == display_card_module.t("display.stopping")

    card.clear_input()
    assert card._input_field.value == ""


def test_display_card_notice_uses_primary_display_slot_and_restores_after_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    monkeypatch.setattr(type(card._display_primary), "update", lambda self: None)
    monkeypatch.setattr(type(card._display_secondary), "update", lambda self: None)

    card.set_status("other", font_family="status-font")
    original_color = card._display_primary.color
    card.set_notice("STT files are missing", tone="warning")

    assert card._display_primary.value == "STT files are missing"
    assert card._display_secondary.visible is False
    assert card._display_primary.color == original_color

    card.set_notice(None, None)

    assert card._display_primary.value == display_card_module.t("display.disconnected")
    assert card._display_primary.font_family == "status-font"
    assert card._display_primary.color == original_color

    card.set_notice("STT 다운로드 중 63%", tone="info")

    assert card._display_primary.value == "STT 다운로드 중 63%"
    assert card._display_primary.color == original_color


def test_display_card_display_text_is_always_selectable() -> None:
    card = DisplayCard(on_submit=lambda _text: None)

    card.set_display("primary")
    card.set_display_translation("secondary")
    card.set_status("connected")
    card.set_notice("warning", tone="warning")

    assert card._display_primary.selectable is True
    assert card._display_secondary.selectable is True


def test_display_card_primary_and_secondary_wrap_to_two_lines() -> None:
    card = DisplayCard(on_submit=lambda _text: None)

    assert card._display_primary._get_attr("nowrap") is False
    assert card._display_primary.max_lines == 2
    assert card._display_primary.overflow == display_card_module.ft.TextOverflow.ELLIPSIS

    assert card._display_secondary._get_attr("nowrap") is False
    assert card._display_secondary.max_lines == 2
    assert card._display_secondary.overflow == display_card_module.ft.TextOverflow.ELLIPSIS


def test_display_card_input_footer_stays_outside_expanding_display_region() -> None:
    card = DisplayCard(on_submit=lambda _text: None)

    glow_stack = card.content
    padded_content_layer = glow_stack.controls[1].content
    main_content = padded_content_layer.content
    display_region, input_footer = main_content.controls
    divider_container = input_footer.controls[0]

    assert main_content.alignment == display_card_module.ft.MainAxisAlignment.START
    assert display_region.expand is True
    assert display_region.clip_behavior == display_card_module.ft.ClipBehavior.HARD_EDGE
    assert divider_container.padding.bottom == 4
    assert input_footer.expand is None
    assert input_footer.tight is True


def test_display_card_debug_prefix_is_rendered_before_visible_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    monkeypatch.setattr(type(card._display_primary), "update", lambda self: None)
    monkeypatch.setattr(type(card._display_secondary), "update", lambda self: None)

    card.set_display("source text", debug_prefix="[P 41c6/src]")

    assert card._display_primary.value == "[P 41c6/src] source text"
    assert card._display_secondary.visible is False

    card.set_display_translation("translated text", debug_prefix="[P 41c6/3bd7]")

    assert card._display_primary.value == "[P 41c6/3bd7] source text"
    assert card._display_secondary.value == "[P 41c6/3bd7] translated text"
    assert card._display_secondary.visible is True

    card.set_status("connected")

    assert card._display_primary.value == display_card_module.t("display.connected")
    assert "[P 41c6" not in card._display_primary.value


def test_display_card_visual_commit_logs_summary_after_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    runtime_logging = RuntimeLoggingCapture()
    events: list[str] = []

    card.set_display("source text", font_family="font-source")
    attach_dummy_page(monkeypatch, card._display_primary)
    attach_dummy_page(monkeypatch, card._display_secondary)
    monkeypatch.setattr(
        type(card._display_primary), "update", lambda self: events.append("primary")
    )
    monkeypatch.setattr(
        type(card._display_secondary), "update", lambda self: events.append("secondary")
    )
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

    assert events == ["primary", "secondary", "log"]
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
    assert "primary_text_len=11" in message
    assert "secondary_text_len=15" in message
    assert "secondary_visible=True" in message
    assert "primary_update_issued=True" in message
    assert "secondary_update_issued=True" in message
    assert "elapsed_ms=500" in message
    assert "source text" not in message
    assert "translated text" not in message


def test_display_card_visual_commit_is_suppressed_in_basic_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    runtime_logging = RuntimeLoggingCapture(detailed_enabled=False)

    card.set_display("source text")
    attach_dummy_page(monkeypatch, card._display_primary)
    attach_dummy_page(monkeypatch, card._display_secondary)
    monkeypatch.setattr(type(card._display_primary), "update", lambda self: None)
    monkeypatch.setattr(type(card._display_secondary), "update", lambda self: None)

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


def test_display_card_notice_yields_to_live_translation_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    runtime_logging = RuntimeLoggingCapture()

    card.set_display("source text")
    card.set_notice(
        "SteamVR isn't on. Ran the desktop overlay instead.", tone="info", yields_to_content=True
    )
    attach_dummy_page(monkeypatch, card._display_primary)
    attach_dummy_page(monkeypatch, card._display_secondary)
    monkeypatch.setattr(type(card._display_primary), "update", lambda self: None)
    monkeypatch.setattr(type(card._display_secondary), "update", lambda self: None)

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

    assert card._display_primary.value == "source text"
    assert card._display_secondary.value == "translated text"
    assert card._display_secondary.visible is True
    assert len(runtime_logging.detailed_messages) == 1


def test_display_card_yielding_notice_shows_while_idle_and_returns_after_status_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    monkeypatch.setattr(type(card._display_primary), "update", lambda self: None)
    monkeypatch.setattr(type(card._display_secondary), "update", lambda self: None)

    card.set_status("connected")
    card.set_notice(
        "SteamVR isn't on. Ran the desktop overlay instead.",
        tone="info",
        yields_to_content=True,
    )
    assert card._display_primary.value == "SteamVR isn't on. Ran the desktop overlay instead."

    card.set_display("source text")
    card.set_display_translation("translated text")
    assert card._display_primary.value == "source text"
    assert card._display_secondary.value == "translated text"

    card.set_status("connected")
    assert card._display_primary.value == "SteamVR isn't on. Ran the desktop overlay instead."
    assert card._display_secondary.visible is False


def test_display_card_non_yielding_notice_still_blocks_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = DisplayCard(on_submit=lambda _text: None)
    runtime_logging = RuntimeLoggingCapture()

    card.set_display("source text")
    card.set_notice("STT files are missing", tone="warning")
    attach_dummy_page(monkeypatch, card._display_primary)
    attach_dummy_page(monkeypatch, card._display_secondary)
    monkeypatch.setattr(type(card._display_primary), "update", lambda self: None)
    monkeypatch.setattr(type(card._display_secondary), "update", lambda self: None)

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

    assert card._display_primary.value == "STT files are missing"
    assert card._display_secondary.value == ""
    assert runtime_logging.detailed_calls == []
    assert runtime_logging.detailed_messages == []
