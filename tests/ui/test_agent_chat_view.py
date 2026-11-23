from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import flet as ft
import pytest

from puripuly_heart.app.ports.agent_chat import (
    AgentChatSnapshot,
    AgentMessage,
    AgentPresentation,
    AgentReportOffer,
    AgentSecureEntryRequest,
    AgentSetupAction,
    AgentToolResult,
    AgentTurnResult,
)
from puripuly_heart.core.conversation_history import ConversationHistory
from puripuly_heart.domain.events import (
    SelfTurnPresentation,
    SelfTurnTargetPresentation,
    UIEvent,
    UIEventType,
)
from puripuly_heart.domain.models import Translation
from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.i18n import get_locale, provider_label, set_locale
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from puripuly_heart.ui.views import agent_chat
from puripuly_heart.ui.views.agent_chat import AgentChatHooks, AgentChatView
from tests.helpers.flet_page import attach_dummy_page


@pytest.fixture(autouse=True)
def korean():
    previous = get_locale()
    set_locale("ko")
    yield
    set_locale(previous)


class Owner:
    def __init__(self, snapshot: AgentChatSnapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[tuple] = []
        self.observers = []

    def observe(self, callback):
        self.observers.append(callback)
        return lambda: self.observers.remove(callback)

    def publish(self, snapshot: AgentChatSnapshot) -> None:
        self.snapshot = snapshot
        for callback in tuple(self.observers):
            callback(snapshot)

    async def confirm(self, card_id, approved):
        self.calls.append(("confirm", card_id, approved))
        self.publish(replace(self.snapshot, confirmation=None))
        if approved:
            return {"status": "applied", "changes": {"stt.provider": "gemini_transcribe"}}
        return {"status": "declined"}

    def activate_setup_action(self, action_id):
        self.calls.append(("activate", action_id))
        return {"status": "opened"}

    def mark_setup_action_displayed(self, action_id):
        self.calls.append(("setup_displayed", action_id))
        return {"status": "displayed"}

    def mark_secure_entry_displayed(self, request_id):
        self.calls.append(("secure_displayed", request_id))
        return {"status": "displayed"}

    def mark_report_displayed(self, offer_id):
        self.calls.append(("report_displayed", offer_id))
        return {"status": "displayed"}

    async def submit_secure_entry(self, request_id, value):
        self.calls.append(("submit", request_id))
        assert value == "synthetic-key"
        return {"status": "failed", "request_id": request_id}

    def cancel_secure_entry(self, request_id):
        self.calls.append(("cancel_secure", request_id))
        return {"status": "cancelled"}

    async def send_report(self, offer_id):
        self.calls.append(("send_report", offer_id))
        return {"status": "sending"}

    def reject_report(self, offer_id):
        self.calls.append(("reject_report", offer_id))
        return {"status": "rejected"}

    async def send(self, text, *, diagnostic_id=None):
        self.calls.append(
            ("send", text) if diagnostic_id is None else ("send", text, diagnostic_id)
        )
        return AgentTurnResult("completed", "")

    def new_conversation(self):
        self.calls.append(("new",))


def snapshot(**fields) -> AgentChatSnapshot:
    base = AgentChatSnapshot(enabled=True, busy=False, phase="idle", messages=(), tool_results=())
    return replace(base, **fields)


def view_for(owner, **hooks):
    tasks = []

    def run_task(factory):
        assert inspect.iscoroutinefunction(factory)
        task = asyncio.get_running_loop().create_task(factory())
        tasks.append(task)
        return task

    view = AgentChatView(owner, AgentChatHooks(run_task=run_task, **hooks))
    view.did_mount()
    return view, tasks


def left(view):
    return view._switcher.content.content


def zone(panel):
    return panel.content.controls[1]


def back(panel):
    return panel.content.controls[0]


def texts(control):
    return [item.value for item in agent_chat._texts(control)]


@pytest.mark.asyncio
async def test_change_card_tap_approves_and_returns_home_with_result_line():
    card = {"id": "card-1", "changes": {"stt.provider": "gemini_transcribe"}}
    owner = Owner(snapshot(confirmation=card, confirmation_previous={"stt.provider": "deepgram"}))
    view, tasks = view_for(owner)
    assert texts(left(view))[1:] == ["음성 인식", "Deepgram", provider_label("gemini_transcribe")]
    zone(left(view)).on_click(None)
    await asyncio.gather(*tasks, return_exceptions=True)
    assert owner.calls == [("confirm", "card-1", True)]
    assert left(view) is view._home
    owner.publish(snapshot(messages=(AgentMessage("assistant", "바꿨어요."),)))
    status = [item for item in view._messages.controls if item.key == "status:0"]
    assert texts(status[0]) == ["음성 인식 바꿈"]


@pytest.mark.asyncio
async def test_successive_cards_require_distinct_clicks_and_reenable_after_completion():
    first_card = {
        "id": "card-1",
        "changes": {"stt.provider": "gemini_transcribe", "overlay.translation.enabled": False},
    }
    next_card = {"id": "card-2", "changes": {"peer_vad.speech_threshold": 0.35}}
    accepted = asyncio.Event()
    applied = asyncio.Event()
    staged = asyncio.Event()
    returned = asyncio.Event()

    class SequentialOwner(Owner):
        async def confirm(self, card_id, approved):
            self.calls.append(("confirm", card_id, approved))
            if card_id == first_card["id"]:
                self.publish(
                    snapshot(
                        busy=True,
                        phase="working",
                        active_tool=("manage_proposal", "settings"),
                    )
                )
                accepted.set()
                await applied.wait()
                self.publish(
                    snapshot(
                        confirmation=next_card,
                        confirmation_previous={"peer_vad.speech_threshold": 0.5},
                    )
                )
                staged.set()
                await returned.wait()
                return {"status": "applied", "changes": first_card["changes"]}
            self.publish(snapshot())
            return {"status": "applied", "changes": next_card["changes"]}

    owner = SequentialOwner(
        snapshot(
            confirmation=first_card,
            confirmation_previous={
                "stt.provider": "deepgram",
                "overlay.translation.enabled": True,
            },
        )
    )
    view, tasks = view_for(owner)
    first_tap = zone(left(view)).on_click
    first_back = back(left(view)).on_click
    assert owner.calls == []
    first_tap(None)
    first_tap(None)
    first_back(None)
    assert left(view) is not view._home
    assert zone(left(view)).on_click is None
    await asyncio.wait_for(accepted.wait(), 1)
    assert owner.calls == [("confirm", "card-1", True)]
    assert left(view) is view._home
    live = [item for item in view._messages.controls if item.key == "status:live"]
    assert texts(live[0]) == [agent_chat.t("agent_chat.status.change_settings.active")]
    applied.set()
    await asyncio.wait_for(staged.wait(), 1)
    assert left(view) is not view._home
    assert zone(left(view)).on_click is None
    returned.set()
    await asyncio.gather(*tasks)
    assert zone(left(view)).on_click is not None
    first_tap(None)
    first_back(None)
    await asyncio.gather(*tasks)
    assert owner.calls == [("confirm", "card-1", True)]
    next_tap = zone(left(view)).on_click
    next_back = back(left(view)).on_click
    next_tap(None)
    next_tap(None)
    next_back(None)
    await asyncio.gather(*tasks)
    assert owner.calls == [("confirm", "card-1", True), ("confirm", "card-2", True)]
    assert left(view) is view._home
    first_tap(None)
    next_tap(None)
    next_back(None)
    await asyncio.gather(*tasks)
    assert owner.calls == [("confirm", "card-1", True), ("confirm", "card-2", True)]


@pytest.mark.asyncio
async def test_change_back_declines_without_applying():
    owner = Owner(snapshot(confirmation={"id": "card-1", "changes": {"stt.provider": "deepgram"}}))
    view, tasks = view_for(owner)
    back(left(view)).on_click(None)
    await asyncio.gather(*tasks, return_exceptions=True)
    assert owner.calls == [("confirm", "card-1", False)]
    assert left(view) is view._home
    owner.publish(snapshot(messages=(AgentMessage("assistant", "알겠어요."),)))
    assert not any(item.key == "status:0" for item in view._messages.controls)


@pytest.mark.asyncio
async def test_api_key_card_opens_link_then_entry_failure_stays_correctable():
    request = AgentSecureEntryRequest(
        "secure-1", "deepgram", "task", 1, message_index=1, available=True, status="ready"
    )
    action = AgentSetupAction(
        "action-1",
        "deepgram.key_console",
        "deepgram",
        "key_console",
        1,
        message_index=1,
        available=True,
        status="available",
        secure_request_id="secure-1",
    )
    owner = Owner(snapshot(secure_entry_requests=(request,), setup_actions=(action,)))
    shown = []
    view, _ = view_for(owner, show_snackbar=lambda message, color: shown.append((message, color)))
    assert ("setup_displayed", "action-1") in owner.calls
    assert ("secure_displayed", "secure-1") in owner.calls
    zone(left(view)).on_click(None)
    assert ("activate", "action-1") in owner.calls
    field = agent_chat._children(zone(left(view)).content)[2]
    assert isinstance(field, ft.TextField) and field.password
    field.value = "synthetic-key"
    await field.on_blur(None)
    assert ("submit", "secure-1") in owner.calls
    assert shown and shown[-1][1] == ft.Colors.RED_400
    retry = agent_chat._children(zone(left(view)).content)[2]
    assert retry.border_color == agent_chat.COLOR_ERROR and not retry.disabled
    back(left(view)).on_click(None)
    assert left(view).content.controls[1].on_click is not None
    back(left(view)).on_click(None)
    assert ("cancel_secure", "secure-1") in owner.calls
    assert left(view) is view._home


@pytest.mark.asyncio
async def test_report_card_states(monkeypatch):
    opened = []
    monkeypatch.setattr(agent_chat.webbrowser, "open", opened.append)
    owner = Owner(snapshot(report_offers=(AgentReportOffer("support-1", 0, "offered"),)))
    view, tasks = view_for(owner)
    assert texts(left(view))[1:] == ["문제", "보고하기"]
    zone(left(view)).on_click(None)
    await asyncio.gather(*tasks, return_exceptions=True)
    assert ("send_report", "support-1") in owner.calls
    owner.publish(snapshot(report_offers=(AgentReportOffer("support-1", 0, "sending"),)))
    assert texts(left(view))[1:] == ["문제", "보내는 중"]
    assert zone(left(view)).on_click is None
    sent = AgentReportOffer("support-1", 0, "sent", ticket="PP-0123456789ABCDEF")
    owner.publish(snapshot(report_offers=(sent,)))
    assert texts(left(view))[1:] == ["PP-0123456789ABCDEF", "X DM으로 보내기"]
    zone(left(view)).on_click(None)
    assert opened == [agent_chat.X_CONTACT_URL]
    owner.publish(snapshot(report_offers=(AgentReportOffer("support-2", 0, "offered"),)))
    back(left(view)).on_click(None)
    assert ("reject_report", "support-2") in owner.calls


def test_card_hover_and_press_leave_the_switched_child_untouched():
    owner = Owner(snapshot(report_offers=(AgentReportOffer("support-1", 0, "offered"),)))
    view, _ = view_for(owner)
    switched = view._switcher.content
    before = (switched.content, switched.key, switched.scale, switched.opacity)
    zone(left(view)).on_hover(SimpleNamespace(data=True))
    assert left(view).scale == 1.02
    assert view._switcher.content is switched
    assert (switched.content, switched.key, switched.scale, switched.opacity) == before


def test_chat_list_pins_to_the_end_without_animation():
    view, _ = view_for(Owner(snapshot()))
    assert view._messages.auto_scroll is True
    assert view._messages.auto_scroll_animation == 0


def test_chat_transcript_is_one_continuous_selection():
    messages = (
        AgentMessage("user", "마이크가 안 돼"),
        AgentMessage("assistant", "입력 장치를 바꿨어. 이제 그대로 써봐."),
    )
    view, _ = view_for(Owner(snapshot(messages=messages)))

    def ancestors(control, target, path=()):
        if control is target:
            return path
        for child in agent_chat._children(control):
            found = ancestors(child, target, (*path, control))
            if found is not None:
                return found
        return None

    path = ancestors(view, view._messages)
    assert path is not None
    assert sum(isinstance(item, ft.SelectionArea) for item in path) == 1
    transcript = agent_chat._texts(view._messages)
    assert {"마이크가 안 돼", "입력 장치를 바꿨어.", "이제 그대로 써봐."} <= {
        item.value for item in transcript
    }
    assert not any(item.selectable for item in transcript)


def test_chat_preserves_history_while_provisional_reply_is_replaced_and_archived():
    prior = AgentMessage("assistant", "First sentence. Second sentence!")
    prompt = AgentMessage("user", "Next question")
    committed = AgentMessage("assistant", "Authoritative reply.")
    owner = Owner(snapshot(messages=(prior, prompt)))
    view, _ = view_for(owner)
    original = view._messages.controls[1]

    streaming = replace(
        owner.snapshot,
        busy=True,
        phase="responding",
        presentation=AgentPresentation(1, 1, ("Provisional reply.",)),
    )
    owner.publish(streaming)
    assert view._messages.controls[1] is original
    assert texts(view._messages.controls[-1]) == ["Provisional reply."]

    owner.publish(snapshot(messages=(prior, prompt, committed)))
    rows = view._messages.controls
    assert rows[1] is original
    assert texts(rows[1]) == ["First sentence.", "Second sentence!"]
    assert texts(rows[-1]) == ["Authoritative reply."]
    assert not any(row.key.startswith("assistant:live:") for row in rows)

    archived = tuple(replace(message, archived=True) for message in owner.snapshot.messages)
    owner.publish(snapshot(messages=(*archived, AgentMessage("notice", "새 대화로 시작했어요"))))
    rows = view._messages.controls
    assert [row.opacity for row in rows[:-1]] == [0.35] * 4
    assert texts(rows[1]) == ["First sentence.", "Second sentence!"]
    assert texts(rows[-2]) == ["Authoritative reply."]
    assert texts(rows[-1]) == [agent_chat.t("agent_chat.reset_line")]


class Dictation:
    def __init__(self) -> None:
        self.callback = None
        self.restored = 0

    def observe(self, callback):
        self.callback = callback
        return lambda: setattr(self, "callback", None)

    def ensure_capture(self):
        return lambda: setattr(self, "restored", self.restored + 1)


@pytest.mark.asyncio
async def test_sending_from_the_input_keeps_it_focused(monkeypatch):
    owner = Owner(snapshot())
    view, tasks = view_for(owner)
    focus_calls = []

    async def focus(_self):
        focus_calls.append("focus")

    def run_task(func, *args):
        task = asyncio.get_running_loop().create_task(func(*args))
        tasks.append(task)
        return task

    monkeypatch.setattr(type(view._input), "focus", focus)
    attach_dummy_page(
        monkeypatch, view._input, SimpleNamespace(run_task=run_task, update=lambda *_: None)
    )
    view._input.value = "상대 목소리가 안 들려요"
    await view._submit()
    await asyncio.gather(*tasks)

    assert ("send", "상대 목소리가 안 들려요") in owner.calls
    assert view._input.value == ""
    assert focus_calls == ["focus"]


@pytest.mark.asyncio
async def test_voice_dictation_fills_input_and_stop_sends():
    owner = Owner(snapshot())
    dictation = Dictation()
    view, tasks = view_for(owner, dictation=dictation)
    view._voice_tile._on_click()
    assert view._voice_tile._label == "STOP"
    dictation.callback("u1", "마이크는 잘 되는데", True)
    dictation.callback("u2", "상대 목소리가", False)
    assert view._input.value == "마이크는 잘 되는데"
    dictation.callback("u2", "상대 목소리가 안 들려요", True)
    dictation.callback("u3", "아직 말하는", False)
    assert view._input.value == "마이크는 잘 되는데 상대 목소리가 안 들려요"
    view._voice_tile._on_click()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert ("send", "마이크는 잘 되는데 상대 목소리가 안 들려요") in owner.calls
    assert dictation.restored == 1 and dictation.callback is None
    assert view._voice_tile._label == "VOICE"


@pytest.mark.asyncio
async def test_voice_limit_ends_dictation_and_keeps_text(monkeypatch):
    monkeypatch.setattr(agent_chat, "DICTATION_LIMIT_S", 0.01)
    owner = Owner(snapshot())
    dictation = Dictation()
    view, tasks = view_for(owner, dictation=dictation)
    view._voice_tile._on_click()
    dictation.callback("u1", "아직 보내지 마", True)
    await asyncio.gather(*tasks, return_exceptions=True)
    assert view._input.value == "아직 보내지 마"
    assert not any(call[0] == "send" for call in owner.calls)
    assert view._voice_tile._label == "VOICE" and dictation.restored == 1


def test_voice_tile_does_not_change_while_puri_works():
    owner = Owner(snapshot(busy=True, phase="working", active_tool=("read_app", "overview")))
    view = AgentChatView(owner)
    assert view._voice_tile._label == "VOICE"
    live = [item for item in view._messages.controls if item.key == "status:live"]
    assert texts(live[0]) == ["앱 상태 확인 중"]


def test_turn_result_line_comes_from_actual_tool_results():
    owner = Owner(snapshot(busy=True, phase="working"))
    view = AgentChatView(owner)
    view.did_mount()
    results = (AgentToolResult("a", "audio", "observed", {}),)
    owner.publish(
        snapshot(
            messages=(AgentMessage("user", "q"), AgentMessage("assistant", "답이에요.")),
            tool_results=results,
        )
    )
    status = [item for item in view._messages.controls if item.key == "status:1"]
    assert texts(status[0]) == ["마이크 상태 확인함"]


@pytest.mark.asyncio
async def test_error_entry_sends_the_error_with_its_diagnostic_id():
    owner = Owner(snapshot())
    view, tasks = view_for(owner)
    view._open_history()
    view.ask_about_error("Deepgram 인증에 실패했어요", "diag-1")
    await asyncio.gather(*tasks)
    assert owner.calls == [("send", "이 오류를 봐줘. Deepgram 인증에 실패했어요", "diag-1")]
    assert left(view) is view._home


def test_without_a_connection_the_screen_stays_but_input_is_closed():
    view = AgentChatView(Owner(snapshot(enabled=False)))
    line = [item for item in view._messages.controls if item.key == "system:unavailable"]
    assert texts(line[0]) == ["Puri에 연결할 수 없어요"]
    assert view._input.disabled and view._voice_tile.disabled and view._new_tile.disabled
    assert not view._history_tile.disabled and not view._folder_tile.disabled


def test_history_card_shows_manual_original_and_translation_as_one_bubble():
    from puripuly_heart.core.conversation_history import ConversationHistory

    logs = ConversationHistory()
    for source, translation, target in (("안녕", None, None), (None, "你好", 0)):
        logs.append_conversation_record(
            source="You",
            channel="self",
            utterance_id="m1",
            source_text=source,
            translated_text=translation,
            source_language="ko",
            target_language="zh-CN" if translation else None,
            target_index=target,
            parent_utterance_id="m1",
        )
    view = AgentChatView(Owner(snapshot()), AgentChatHooks(conversation_history=logs))
    view._open_history()
    (row,) = view._history_content.controls
    assert [text for text in texts(row) if text != row_time(row)] == ["안녕", "你好"]
    source, translation = agent_chat._texts(row)[:2]
    assert source.font_family == font_for_language("ko")
    assert translation.font_family == font_for_language("zh-CN")
    assert source.font_family != translation.font_family


def row_time(row):
    return agent_chat._texts(row)[-1].value


@pytest.mark.asyncio
async def test_lists_return_to_the_end_or_to_the_last_read_position():
    calls = []
    tasks = []

    class Listing:
        reverse = False

        async def scroll_to(self, offset):
            calls.append(offset)

    def run(factory):
        assert inspect.iscoroutinefunction(factory)
        task = asyncio.get_running_loop().create_task(factory())
        tasks.append(task)
        return task

    memory = agent_chat._ScrollMemory(Listing(), run)
    memory.follow()
    memory.restore()
    await asyncio.gather(*tasks)
    assert calls == [-1, -1]
    memory.remember(SimpleNamespace(pixels=120.0, max_scroll_extent=600.0))
    memory.follow()
    memory.restore()
    await asyncio.gather(*tasks)
    assert calls == [-1, -1, 120.0]
    memory.remember(SimpleNamespace(pixels=590.0, max_scroll_extent=600.0))
    memory.follow()
    await asyncio.gather(*tasks)
    assert calls[-1] == -1


def test_history_back_label_follows_current_locale_and_drops_stale_hover():
    set_locale("en")
    view = AgentChatView(Owner(snapshot()))
    set_locale("ko")
    view._open_history()
    back_text = back(left(view)).content
    assert back_text.value == "돌아가기"
    back(left(view)).on_hover(SimpleNamespace(data=True))
    assert back_text.color == agent_chat.COLOR_PRIMARY
    back(left(view)).on_click(None)
    view._open_history()
    assert back(left(view)).content.color == agent_chat.COLOR_SECONDARY


@pytest.mark.asyncio
async def test_open_history_batches_changes_and_catches_up_after_unmount(monkeypatch):
    monkeypatch.setattr(agent_chat, "HISTORY_REFRESH_S", 0)
    history = ConversationHistory()

    def add(identity, text, *, revision=1):
        history.append_conversation_record(
            source="Listen",
            channel="peer",
            utterance_id=identity,
            source_text=text,
            translated_text=None,
            peer_caption_revision=revision,
        )

    add("a", "first")
    view, tasks = view_for(Owner(snapshot()), conversation_history=history)
    view._open_history()
    await asyncio.gather(*tasks)
    (first,) = view._history_content.controls
    tasks.clear()

    add("a", "duplicate")
    assert tasks == []
    add("b", "next")
    add("b", "revised", revision=2)
    assert len(tasks) == 1
    await asyncio.gather(*tasks)
    assert view._history_content.controls[0] is first
    assert texts(view._history_content.controls[1])[0] == "revised"

    add("b", "newer", revision=3)
    view.will_unmount()
    await asyncio.gather(*tasks)
    assert texts(view._history_content.controls[1])[0] == "revised"
    tasks.clear()
    add("c", "arrived offscreen")
    assert tasks == []
    remount_rows = []

    def record_update(control):
        if control is view:
            remount_rows.extend(texts(row)[0] for row in view._history_content.controls)

    monkeypatch.setattr(agent_chat, "update_control_if_mounted", record_update)
    view.did_mount()
    assert remount_rows == [
        "first",
        "newer",
        "arrived offscreen",
    ]
    await asyncio.gather(*tasks)

    view._close_history()
    tasks.clear()
    add("d", "arrived while closed")
    assert tasks == []
    view._open_history()
    assert texts(view._history_content.controls[-1])[0] == "arrived while closed"
    await asyncio.gather(*tasks)
    view.will_unmount()


@pytest.mark.asyncio
async def test_open_history_shows_managed_self_turn_and_its_translation(monkeypatch):
    monkeypatch.setattr(agent_chat, "HISTORY_REFRESH_S", 0)
    history = ConversationHistory()
    view, tasks = view_for(Owner(snapshot()), conversation_history=history)
    adapter = FletUiPresentationAdapter(SimpleNamespace(conversation_history=history))
    bridge = adapter.create_ui_event_bridge(event_queue=asyncio.Queue(), runtime_logging=None)
    utterance_id = uuid4()
    source_text = "最初の文です。"
    translated_text = "This is the first sentence."
    source = SelfTurnPresentation(
        utterance_id=utterance_id,
        turn_generation=1,
        turn_order=1,
        revision=1,
        source_revision=1,
        source_text=source_text,
        source_language="ja",
        source_closed=True,
        status="active",
        targets=(SelfTurnTargetPresentation("en", 1, "pending"),),
        created_at=0.0,
    )
    final = replace(
        source,
        revision=2,
        status="complete",
        targets=(
            replace(
                source.targets[0],
                status="ready",
                translation=Translation(
                    utterance_id,
                    translated_text,
                    source_text=source_text,
                    source_language="ja",
                    target_language="en",
                ),
            ),
        ),
    )

    async def settle():
        for task in tasks:
            await task
        tasks.clear()

    async def publish(turn):
        await bridge._handle_event(
            UIEvent(
                UIEventType.SELF_TURN_UPDATED,
                utterance_id=utterance_id,
                payload=turn,
                source="Mic",
                runtime_log_handled=True,
            )
        )
        await settle()

    def visible_rows():
        return [(row.key, texts(row)[:-1]) for row in view._history_content.controls]

    key = f"self:{utterance_id}"
    try:
        view._open_history()
        await settle()
        assert visible_rows() == []

        await publish(source)
        assert visible_rows() == [(key, [source_text])]

        await publish(final)
        assert visible_rows() == [(key, [source_text, translated_text])]

        await publish(source)
        assert visible_rows() == [(key, [source_text, translated_text])]
    finally:
        view.will_unmount()
        bridge.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("return_by", ["scroll", "layout"])
async def test_small_history_stays_paged_when_live_turns_arrive_while_reading(
    monkeypatch, return_by
):
    monkeypatch.setattr(agent_chat, "HISTORY_REFRESH_S", 0)
    history = ConversationHistory()
    offsets = []

    async def scroll_to(_listing, *, offset):
        offsets.append(offset)

    monkeypatch.setattr(ft.ListView, "scroll_to", scroll_to)

    def add(i, *, correction=None, revision=1):
        history.append_conversation_record(
            source="Listen",
            channel="peer",
            utterance_id=str(i),
            source_text=f"turn {i}",
            translated_text=correction,
            peer_caption_revision=revision,
        )

    def numbers():
        return [texts(row)[0] for row in view._history_content.controls]

    def scroll(pixels, extent, delta):
        view._history_list.on_scroll(
            SimpleNamespace(
                pixels=pixels,
                max_scroll_extent=extent,
                viewport_dimension=600,
                scroll_delta=delta,
            )
        )

    for i in range(20):
        add(i)
    if return_by == "layout":
        add(15, correction="Long correction\n" * 20, revision=2)
    view, tasks = view_for(Owner(snapshot()), conversation_history=history)
    view._open_history()
    view._history_list.on_size_change(SimpleNamespace(height=600))
    await asyncio.gather(*tasks)
    scroll(350, 700, -350)
    reader = view._history_content.controls[6]
    offsets.clear()

    for start, stop in ((20, 25), (25, 250), (250, 1250)):
        tasks.clear()
        for i in range(start, stop):
            add(i)
        await asyncio.gather(*tasks)
        assert numbers() == [f"turn {i}" for i in range(24)]
        assert view._history_content.controls[6] is reader
        assert offsets == []

    tasks.clear()
    view._close_history()
    view._open_history()
    await asyncio.gather(*tasks)
    assert view._history_content.controls[6] is reader
    assert offsets == [350]

    tasks.clear()
    add(1248, correction="corrected under the same turn", revision=2)
    await asyncio.gather(*tasks)
    tasks.clear()
    if return_by == "scroll":
        scroll(930, 950, None)
        assert numbers() == [f"turn {i}" for i in range(24)]
        scroll(930, 950, 580)
        scroll(950, 950, 20)
        scroll(950, 950, None)
    else:
        view._history_content.on_size_change(SimpleNamespace(height=2100))
        scroll(1200, 1500, 850)
        add(15, correction="Short correction", revision=3)
        await asyncio.gather(*tasks)
        assert numbers() == [f"turn {i}" for i in range(24)]
        view._history_content.on_size_change(SimpleNamespace(height=1550))
    await asyncio.gather(*tasks)
    assert numbers() == [f"turn {i}" for i in range(1226, 1250)]
    assert "corrected under the same turn" in texts(view._history_content.controls[-2])
    assert offsets[-1] == 0
    scroll(900, 950, 900)
    assert numbers() == [f"turn {i}" for i in range(1202, 1250)]
    view.will_unmount()


@pytest.mark.asyncio
async def test_history_keeps_older_position_after_paging_and_newer_growth(monkeypatch):
    monkeypatch.setattr(agent_chat, "HISTORY_PAGE_ROWS", 3)
    monkeypatch.setattr(agent_chat, "HISTORY_REFRESH_S", 0)
    history = ConversationHistory()
    viewport = 80
    extent = 0
    offsets = []

    async def scroll_to(_listing, *, offset):
        pixels = extent + offset + 1 if offset < 0 else offset
        offsets.append(pixels)

    monkeypatch.setattr(ft.ListView, "scroll_to", scroll_to)

    def add(i, translation=None):
        history.append_conversation_record(
            source="Listen",
            channel="peer",
            utterance_id=str(i),
            source_text=f"turn {i}" if translation is None else None,
            translated_text=translation,
            peer_caption_revision=1,
        )

    for i in range(9):
        add(i)
    view, tasks = view_for(Owner(snapshot()), conversation_history=history)

    def visible_rows():
        return [texts(row)[0] for row in view._history_content.controls]

    def layout(height):
        nonlocal extent
        extent = height - viewport
        view._history_content.on_size_change(SimpleNamespace(height=height))

    def scroll(pixels, delta):
        view._history_list.on_scroll(
            SimpleNamespace(
                pixels=pixels,
                max_scroll_extent=extent,
                viewport_dimension=viewport,
                scroll_delta=delta,
            )
        )

    view._open_history()
    view._history_list.on_size_change(SimpleNamespace(height=viewport))
    layout(300)
    await asyncio.gather(*tasks)
    assert view._history_list.controls == [view._history_content]
    assert view._history_list.reverse
    assert offsets == [0]
    assert visible_rows() == [f"turn {i}" for i in (6, 7, 8)]

    scroll(160, 160)
    layout(600)
    scroll(450, 290)
    layout(900)
    assert visible_rows() == [f"turn {i}" for i in range(9)]
    older_position = extent - 450
    reader_row = view._history_content.controls[3]

    view._close_history()
    add(8, "A newer bubble received a translation while History was closed. " * 8)
    add(9)
    view._open_history()
    layout(1000)
    await asyncio.gather(*tasks)
    assert view._history_content.controls[3] is reader_row
    assert visible_rows()[-1] == "turn 8"
    assert len(texts(view._history_content.controls[-1])) == 3
    assert extent - offsets[-1] == older_position

    scroll(offsets[-1], offsets[-1])
    tasks.clear()
    add(7, "Another newer bubble received a translation while History stayed open. " * 8)
    await asyncio.gather(*tasks)
    layout(1100)
    await asyncio.gather(*tasks)
    assert view._history_content.controls[3] is reader_row
    assert len(texts(view._history_content.controls[-2])) == 3
    assert extent - offsets[-1] == older_position

    assert view._history_list.scroll_interval == -1
    scroll(60, 60 - offsets[-1])
    assert visible_rows()[-1] == "turn 8"
    scroll(0, -60)
    await asyncio.gather(*tasks)
    assert visible_rows()[-1] == "turn 9"
    assert offsets[-1] == 0

    view._close_history()
    view._open_history()
    assert visible_rows() == [f"turn {i}" for i in (7, 8, 9)]
    await asyncio.gather(*tasks)
    view.will_unmount()


@pytest.mark.asyncio
async def test_history_keeps_surviving_reader_position_when_older_rows_are_pruned(monkeypatch):
    from puripuly_heart.core.conversation_history import ConversationHistory

    monkeypatch.setattr(agent_chat, "HISTORY_PAGE_ROWS", 3)
    history = ConversationHistory(max_entries=4, cleanup_batch=2)
    offsets = []

    async def scroll_to(_listing, *, offset):
        offsets.append(offset)

    monkeypatch.setattr(ft.ListView, "scroll_to", scroll_to)

    def add(i):
        history.append_conversation_record(
            source="Listen",
            channel="peer",
            utterance_id=str(i),
            source_text=f"turn {i}",
            translated_text=None,
        )

    def scroll(pixels, extent, delta):
        view._history_list.on_scroll(
            SimpleNamespace(
                pixels=pixels,
                max_scroll_extent=extent,
                viewport_dimension=80,
                scroll_delta=delta,
            )
        )

    for i in range(5):
        add(i)
    view, tasks = view_for(Owner(snapshot()), conversation_history=history)
    view._open_history()
    await asyncio.gather(*tasks)
    scroll(160, 220, 160)
    scroll(200, 420, 40)
    reader_row = view._history_content.controls[2]
    assert texts(reader_row)[0] == "turn 2"

    view._close_history()
    add(5)
    add(6)
    view._open_history()
    await asyncio.gather(*tasks)
    assert view._history_content.controls[0] is reader_row
    assert [texts(row)[0] for row in view._history_content.controls] == [
        "turn 2",
        "turn 3",
        "turn 4",
    ]
    assert offsets[-1] == 200
    view.will_unmount()


@pytest.mark.asyncio
async def test_pruned_history_backfills_a_scrollable_window_and_can_reach_latest(monkeypatch):
    from puripuly_heart.core.conversation_history import ConversationHistory

    monkeypatch.setattr(agent_chat, "HISTORY_REFRESH_S", 0)
    history = ConversationHistory()

    def add(i):
        history.append_conversation_record(
            source="Listen",
            channel="peer",
            utterance_id=str(i),
            source_text=f"turn {i}",
            translated_text=None,
        )

    def attached():
        return [texts(row)[0] for row in view._history_content.controls]

    for i in range(1, 506):
        add(i)
    view, tasks = view_for(Owner(snapshot()), conversation_history=history)
    view._open_history()
    await asyncio.gather(*tasks)
    assert attached() == [f"turn {i}" for i in range(482, 506)]
    view._history_list.on_scroll(
        SimpleNamespace(pixels=150, max_scroll_extent=900, viewport_dimension=600, scroll_delta=150)
    )

    tasks.clear()
    for i in range(506, 1502):
        add(i)
    await asyncio.gather(*tasks)
    assert len(history.conversation_records) == 1001
    assert attached() == [f"turn {i}" for i in range(501, 525)]

    view._history_list.on_scroll(
        SimpleNamespace(pixels=0, max_scroll_extent=900, viewport_dimension=600, scroll_delta=-150)
    )
    assert attached() == [f"turn {i}" for i in range(1478, 1502)]
    await asyncio.gather(*tasks)
    view.will_unmount()


def test_chat_input_focus_border_is_coral_before_and_after_dictation():
    dictation = Dictation()
    view = AgentChatView(Owner(snapshot()), AgentChatHooks(dictation=dictation))
    assert view._input.focused_border_color == agent_chat.COLOR_PRIMARY
    assert view._input.border_color == agent_chat.COLOR_DIVIDER
    view._voice_tile._on_click()
    view._stop_dictation(send=False)
    assert view._input.focused_border_color == agent_chat.COLOR_PRIMARY
    assert view._input.border_color == agent_chat.COLOR_DIVIDER


def test_app_dictation_observes_self_history_updates_and_unsubscribes(monkeypatch):
    from uuid import uuid4

    from puripuly_heart.core.conversation_history import ConversationHistory
    from puripuly_heart.domain.events import SelfTurnPresentation
    from puripuly_heart.ui import app as app_module

    monkeypatch.setattr(app_module.time, "monotonic", lambda: 10.0)
    history = ConversationHistory()
    source = app_module._AgentDictationSource(SimpleNamespace(conversation_history=history))
    received = []
    unsubscribe = source.observe(lambda *args: received.append(args))
    old = SelfTurnPresentation(
        utterance_id=uuid4(),
        turn_generation=1,
        turn_order=1,
        revision=1,
        source_revision=1,
        source_text="old speech",
        source_language="en",
        source_closed=False,
        status="active",
        targets=(),
        created_at=9.0,
    )
    history.update_self_turn(old, source="Mic")
    current = replace(old, utterance_id=uuid4(), source_text="draft", created_at=11.0)
    history.update_self_turn(current, source="Mic")
    history.update_self_turn(
        replace(current, utterance_id=uuid4(), source_text="manual"), source="You"
    )
    final = replace(
        current,
        revision=2,
        source_revision=2,
        source_text="final",
        source_closed=True,
        status="complete",
    )
    history.update_self_turn(final, source="Mic")
    unsubscribe()
    history.update_self_turn(
        replace(current, utterance_id=uuid4(), source_text="after unsubscribe"), source="Mic"
    )
    assert received == [
        (str(current.utterance_id), "draft", False),
        (str(current.utterance_id), "final", True),
    ]
