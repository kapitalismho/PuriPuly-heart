from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from types import SimpleNamespace

import flet as ft
import pytest

from puripuly_heart.app.ports.agent_chat import (
    AgentChatSnapshot,
    AgentMessage,
    AgentReportOffer,
    AgentSecureEntryRequest,
    AgentSetupAction,
    AgentToolResult,
    AgentTurnResult,
)
from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.i18n import get_locale, provider_label, set_locale
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
                        active_tool=("change_settings", None),
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
    assert texts(left(view))[1:] == ["PP-0123456789ABCDEF", "X로 문의하기"]
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
    from puripuly_heart.ui.views.logs import LogsView

    logs = LogsView()
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
    view = AgentChatView(
        Owner(snapshot()), AgentChatHooks(conversation_records=lambda: logs.conversation_records)
    )
    view._open_history()
    (row,) = view._history_list.controls
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
        on_scroll = None
        scroll_interval = None

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
    memory._scrolled(SimpleNamespace(pixels=120.0, max_scroll_extent=600.0))
    memory.follow()
    memory.restore()
    await asyncio.gather(*tasks)
    assert calls == [-1, -1, 120.0]
    memory._scrolled(SimpleNamespace(pixels=590.0, max_scroll_extent=600.0))
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
async def test_open_history_refreshes_periodically_and_keeps_existing_rows(monkeypatch):
    from puripuly_heart.ui.views.logs import ConversationRecord

    monkeypatch.setattr(agent_chat, "HISTORY_REFRESH_S", 0.01)
    records = [ConversationRecord("21:02", "Mic", "self", "첫 말", "first", utterance_id="a")]
    view, tasks = view_for(Owner(snapshot()), conversation_records=lambda: records)
    view._open_history()
    (first,) = view._history_list.controls
    records.append(ConversationRecord("21:03", "Peer", "peer", "next", "다음", utterance_id="b"))
    await asyncio.sleep(0.05)
    assert len(view._history_list.controls) == 2
    assert view._history_list.controls[0] is first
    view._close_history()
    await asyncio.wait_for(asyncio.gather(*tasks), 1)


def test_chat_input_focus_border_is_coral_before_and_after_dictation():
    dictation = Dictation()
    view = AgentChatView(Owner(snapshot()), AgentChatHooks(dictation=dictation))
    assert view._input.focused_border_color == agent_chat.COLOR_PRIMARY
    assert view._input.border_color == agent_chat.COLOR_DIVIDER
    view._voice_tile._on_click()
    view._stop_dictation(send=False)
    assert view._input.focused_border_color == agent_chat.COLOR_PRIMARY
    assert view._input.border_color == agent_chat.COLOR_DIVIDER
