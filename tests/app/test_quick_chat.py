import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from puripuly_heart.app.adapters.ui_runtime import UiInputRuntimeAdapter
from puripuly_heart.app.services.manual_typing import ManualTypingOwner
from puripuly_heart.app.services.quick_chat import QuickChatHistory, chat_text
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation
from puripuly_heart.ui.quick_translate_window import QuickWindow
from tests.app.test_quick_translate import Page
from tests.test_desktop_quick_toggle_contract import eventually, make_owner


def event(identity, role="self", translated=False, text="original"):
    payload = (
        Translation(
            utterance_id=identity, source_text=text, translated_text="translated", channel=role
        )
        if translated
        else Transcript(identity, text, True, channel=role)
    )
    return UIEvent(
        UIEventType.TRANSLATION_DONE if translated else UIEventType.TRANSCRIPT_FINAL,
        utterance_id=identity,
        payload=payload,
        channel=role,
    )


def test_roles_dedup_late_translation_and_failed_original():
    history = QuickChatHistory()
    first, second = uuid4(), uuid4()
    history.accept(event(first), translation_enabled=True)
    history.accept(event(second, "peer"), translation_enabled=True)
    history.accept(event(first), translation_enabled=True)
    history.accept(event(second, "peer", True), translation_enabled=True)
    history.accept(UIEvent(UIEventType.ERROR, first, "secret-failure"), translation_enabled=True)
    rows = list(history.records.values())
    assert len(rows) == 2 and [row["role"] for row in rows] == ["self", "peer"]
    assert rows[0]["original"] == "original" and rows[0]["state"] == "failed"
    assert rows[1]["translation"] == "translated"
    assert "secret-failure" not in str(rows)


async def test_existing_ui_bridge_fans_out_only_authorized_events():
    from puripuly_heart.ui.event_dispatch import (
        AppDashboardEventDestination,
        AppHistoryEventDestination,
        UIEventBridge,
    )

    observed = []
    bridge = UIEventBridge(
        event_queue=asyncio.Queue(),
        dashboard_destination=AppDashboardEventDestination(None),
        history_destination=AppHistoryEventDestination(None),
        on_conversation_event=observed.append,
    )
    accepted = event(uuid4(), "peer")
    await bridge._handle_event(accepted)
    retired = UIEvent(
        UIEventType.TRANSCRIPT_FINAL,
        uuid4(),
        accepted.payload,
        channel="peer",
        delivery_authority=SimpleNamespace(claim_delivery=lambda: False),
    )
    await bridge._handle_event(retired)
    assert observed == [accepted]


def test_event_sequence_keeps_rows_in_chronological_order_even_if_delivery_is_reordered():
    window = QuickWindow(Page())
    row = dict(role="peer", original="text", translation="", state="original")
    window.accept_chat({**row, "id": "b", "sequence": 2})
    window.accept_chat({**row, "id": "a", "sequence": 1})
    window.accept_chat(
        {**row, "id": "a", "sequence": 1, "translation": "late", "state": "complete"}
    )
    assert window.order == ["a", "b"]


def test_bounded_history_rejects_late_retired_result():
    history = QuickChatHistory(limit=2)
    identities = [uuid4() for _ in range(3)]
    for identity in identities:
        history.accept(event(identity), translation_enabled=True)
    assert history.accept(event(identities[0], translated=True), translation_enabled=True) is None
    assert len(history.records) == 2


def test_all_display_modes_disabled_pending_and_failure():
    row = dict(role="peer", original="原文", translation="translation", state="complete")
    assert chat_text(row, peer_mode="both", translation_enabled=True) == "原文\n→ translation"
    assert chat_text(row, peer_mode="translation", translation_enabled=True) == "translation"
    assert chat_text(row, peer_mode="original", translation_enabled=True) == "原文"
    assert chat_text(row, peer_mode="translation", translation_enabled=False) == "原文"
    for state in ("pending", "failed"):
        assert (
            chat_text(
                {**row, "translation": "", "state": state},
                peer_mode="translation",
                translation_enabled=True,
            )
            == "原文"
        )


async def test_translation_off_uses_same_original_submission_without_provider():
    identity = uuid4()
    provider = SimpleNamespace(provider_available=False)
    config = SimpleNamespace(
        translation_enabled=False,
        peer_translation_enabled=False,
        source_language="zh",
        target_language="ja",
        peer_source_language="",
        peer_target_language="",
    )
    channel = SimpleNamespace(
        translation_requests=provider,
        submit_text=AsyncMock(return_value=identity),
        config_snapshot=lambda: SimpleNamespace(value=config),
    )
    manual = ManualTypingOwner(lambda: None, lambda _: None, lambda _: None, lambda _: None, 1, 1)
    adapter = UiInputRuntimeAdapter(
        SimpleNamespace(self_translation_channel=channel), manual, None, None
    )
    assert await adapter.submit_quick_text("原文") == (str(identity), "")
    channel.submit_text.assert_awaited_once_with("原文", source="You")
    assert adapter.quick_window_context()["target_language"] == "ja"
    config.target_language = "en"
    assert adapter.quick_window_context()["target_language"] == "en"
    config.translation_enabled = True
    import pytest

    with pytest.raises(RuntimeError, match="backend_not_ready"):
        await adapter.submit_quick_text("no provider")
    assert channel.submit_text.await_count == 1


async def test_owner_nonblocking_events_and_visibility_callback():
    owner, processes = make_owner()
    visibility = []
    owner.on_visibility = visibility.append
    identity = uuid4()
    owner.publish_event(event(identity, "peer"))
    await owner.open(True)
    process = processes[0]
    await eventually(lambda: any(m["type"] == "chat" for m in process.messages))
    process.emit(dict(type="visibility", visible=False, visibility_revision=1))
    await eventually(lambda: visibility[-1] is False)
    await owner.open(True)
    owner.publish_event(event(identity, "peer", True))
    await eventually(lambda: len([m for m in process.messages if m["type"] == "chat"]) == 2)
    assert len(owner.history.records) == 1
    await owner.close()
    assert visibility[-1] is False and process.returncode == 0


async def test_window_canonical_result_dedup_draft_modes_and_hide():
    window = QuickWindow(Page())
    messages = []
    window.emit = messages.append
    window.input.value = "draft"
    await window.send()
    request = messages[-1]["id"]
    canonical = str(uuid4())
    row = dict(
        id="self:" + canonical,
        role="self",
        original="draft",
        translation="translated",
        state="complete",
    )
    window.accept_chat(row)
    window.accept_result(dict(id=request, utterance_id=canonical, translation="translated"))
    assert window.order == [row["id"]] and window.input.value == ""
    window.input.value = "new unsent draft"
    window.page.window.left = 0
    window.page.window.top = 0
    window.accept_chat({**row, "id": "peer:x", "role": "peer"})
    window.accept_context(
        dict(
            peer_display_mode="translation",
            peer_translation_enabled=False,
            source_language="ja",
            target_language="zh",
            translation_enabled=False,
        )
    )
    assert window.history["peer:x"][1].value == "draft"
    await window.hide()
    assert window.input.value == "new unsent draft" and len(window.order) == 2
    assert messages[-1]["visible"] is False


async def test_reedited_identical_draft_is_not_cleared_by_old_completion():
    window = QuickWindow(Page())
    messages = []
    window.emit = messages.append
    window.input.value = "same"
    await window.send()
    window.draft_changed(None)
    window.accept_result(dict(id=messages[-1]["id"], translation="result"))
    assert window.input.value == "same"


async def test_opacity_default_clamping_and_preferences():
    window = QuickWindow(Page())
    messages = []
    window.emit = messages.append
    window.page.window.left = window.page.window.top = 0
    assert window.page.window.opacity == 1
    window.opacity.value = 65
    await window.set_opacity(None)
    assert window.page.window.opacity == 0.65
    window.save_preferences()
    assert messages[-1]["values"]["opacity_percent"] == 65
    window.accept_context(dict(opacity_percent=65))
    assert window.opacity.value == 65 and window.page.window.opacity == 0.65
    window.accept_context(dict(opacity_percent=-10))
    assert window.page.window.opacity == 0.4


def test_reading_history_is_not_forced_back_to_latest_message():
    import flet as ft

    window = QuickWindow(Page())
    assert window.results.auto_scroll is True
    window.history_scrolled(
        SimpleNamespace(
            event_type=ft.ScrollType.USER,
            direction=ft.ScrollDirection.IDLE,
            max_scroll_extent=1000,
            pixels=200,
        )
    )
    assert window.results.auto_scroll is False
    window.history_scrolled(
        SimpleNamespace(
            event_type=ft.ScrollType.USER,
            direction=ft.ScrollDirection.IDLE,
            max_scroll_extent=1000,
            pixels=990,
        )
    )
    assert window.results.auto_scroll is True
    window.history_scrolled(
        SimpleNamespace(event_type=ft.ScrollType.UPDATE, max_scroll_extent=1000, pixels=200)
    )
    assert window.results.auto_scroll is True


async def test_sending_and_completion_follow_latest_even_when_reading_old_messages():
    window = QuickWindow(Page())
    messages = []
    window.emit = messages.append
    window.input.value = "send from history"
    window.results.auto_scroll = False
    await window.send()
    assert window.results.auto_scroll is True
    window.results.auto_scroll = False
    window.accept_result(dict(id=messages[-1]["id"], translation="translated"))
    assert window.results.auto_scroll is True


async def test_explicit_latest_scroll_uses_native_end_offset_and_safe_failure(monkeypatch):
    import puripuly_heart.ui.quick_translate_window as module

    window = QuickWindow(Page())
    messages = []
    window.emit = messages.append
    monkeypatch.setattr(module, "is_control_mounted", lambda _: True)

    async def scroll(**kwargs):
        assert window.results.auto_scroll is False
        assert kwargs == {"offset": -1, "duration": 0}

    native_scroll = AsyncMock(side_effect=scroll)
    monkeypatch.setattr(window.results, "scroll_to", native_scroll)
    await window.scroll_latest()
    assert native_scroll.await_count == 1 and window.results.auto_scroll is True
    native_scroll.side_effect = RuntimeError("private-text sk-test-secret")
    await window.scroll_latest()
    assert messages[-1] == {"type": "failure", "code": "scroll_failed"}
    assert "private-text" not in str(messages) and "sk-test-secret" not in str(messages)


async def test_canonical_failed_submission_updates_original_instead_of_duplicate_row():
    window = QuickWindow(Page())
    messages = []
    window.emit = messages.append
    window.input.value = "original"
    await window.send()
    canonical = str(uuid4())
    row = dict(
        id="self:" + canonical, role="self", original="original", translation="", state="pending"
    )
    window.accept_chat(row)
    window.accept_result(
        dict(id=messages[-1]["id"], utterance_id=canonical, code="translation_failed")
    )
    window.accept_chat(row)
    assert window.order == [row["id"]] and window.records[row["id"]]["state"] == "failed"
    assert window.input.value == "original"


def test_completed_peer_history_remains_translated_when_capture_is_off():
    window = QuickWindow(Page())
    window.accept_chat(
        dict(
            id="peer:x",
            role="peer",
            original="original",
            translation="translated",
            state="complete",
        )
    )
    window.accept_context(
        dict(
            translation_enabled=True,
            peer_translation_enabled=False,
            peer_display_mode="translation",
        )
    )
    assert window.history["peer:x"][1].value == "translated"


def test_composer_toggle_is_adjacent_and_bidirectional():
    from puripuly_heart.ui.views.dashboard import DashboardView

    dashboard = DashboardView()
    values = []
    dashboard.bind_quick_window_toggle(values.append)
    toggle = dashboard.display_card.quick_toggle
    toggle.on_change(SimpleNamespace(control=SimpleNamespace(value=True)))
    assert values == [True]
    dashboard.set_quick_window_visible(False)
    assert toggle.value is False


async def test_stalled_child_does_not_block_events_and_queue_is_bounded():
    owner, processes = make_owner()
    await owner.open(True)
    release = asyncio.Event()
    processes[0].drain = release.wait
    for _ in range(150):
        owner.publish_event(event(uuid4(), "peer"))
    assert len(owner.history.records) == 100 and len(owner.outbox) <= 101
    release.set()
    await owner.close()
    assert owner.delivery_task.done()


async def test_failure_logs_survive_window_hide_and_log_view_reopen(tmp_path):
    from puripuly_heart.core import feature_diagnostics
    from puripuly_heart.core.runtime_logging import (
        RealtimeLogHandler,
        SessionRuntimeLoggingService,
        configure_main_logging,
    )
    from puripuly_heart.ui.views.logs import LogsView

    logger = feature_diagnostics.logger
    previous_handlers, previous_level = list(logger.handlers), logger.level
    sinks = configure_main_logging(root_logger=logger, log_dir=tmp_path)
    service = SessionRuntimeLoggingService(
        root_logger=logger,
        session_logger=logging.getLogger(f"test.quick.{uuid4()}"),
        sinks=sinks,
        ui_handler_factory=RealtimeLogHandler,
    )
    view = LogsView()
    service.attach_realtime_sink(view)
    owner, _ = make_owner()
    window = QuickWindow(Page())
    window.emit = lambda _: None
    window.page.window.left = window.page.window.top = 0

    async def fail(_):
        raise RuntimeError("private-user-text sk-test-never-log")

    owner.submit = fail
    owner.process = object()
    owner.send = AsyncMock()
    try:
        await owner.translate("request", "private-user-text", owner.process)
        await window.hide()
        view._flush_logs()
        assert (
            "[QuickTranslate] failure stage=translation code=translation_failed"
            in view._log_text.value
        )
        service.detach_realtime_sink()
        view.will_unmount()
        service.attach_realtime_sink(view)
        view._flush_logs()
        assert "exception_type=RuntimeError" in view._log_text.value
    finally:
        owner.process = None
        await owner.close()
        service.detach_realtime_sink()
        sinks.close()
        for handler in tuple(logger.handlers):
            if handler not in previous_handlers:
                logger.removeHandler(handler)
        logger.setLevel(previous_level)
    persisted = sinks.log_file.read_text(encoding="utf-8")
    assert "[ERROR]" in persisted and "code=translation_failed" in persisted
    assert "private-user-text" not in persisted and "sk-test-never-log" not in persisted
