from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.app.ports.agent_chat import AgentMessage
from puripuly_heart.ui import agent_chat_preview
from puripuly_heart.ui.agent_chat_preview import PREVIEW_TICKET, AgentChatPreview
from puripuly_heart.ui.i18n import get_locale, provider_label, set_locale
from puripuly_heart.ui.views import agent_chat
from tests.ui.test_agent_chat_view import Owner, back, left, snapshot, texts, view_for, zone

CARD_CALLS = {
    "confirm",
    "activate",
    "setup_displayed",
    "secure_displayed",
    "report_displayed",
    "submit",
    "cancel_secure",
    "send_report",
    "reject_report",
}


@pytest.fixture(autouse=True)
def korean(monkeypatch):
    monkeypatch.setattr(agent_chat_preview, "PREVIEW_DELAY_S", 0)
    previous = get_locale()
    set_locale("ko")
    yield
    set_locale(previous)


def previewing():
    live = Owner(snapshot(messages=(AgentMessage("assistant", "실제 대화"),)))
    view, tasks = view_for(live)
    preview = AgentChatPreview(live, on_finish=lambda: view.use_owner(live))
    view.use_owner(preview)
    return live, preview, view, tasks


def card_calls(owner) -> list[tuple]:
    return [call for call in owner.calls if call[0] in CARD_CALLS]


@pytest.mark.asyncio
async def test_report_preview_cycles_every_state_without_uploading():
    live, preview, view, tasks = previewing()
    shown = []
    for _ in range(5):
        preview.cycle("report")
        shown.append(texts(left(view))[1:])
    assert shown == [
        ["문제", "보고하기"],
        ["문제", "보내는 중"],
        ["실패", "다시 보내기"],
        [PREVIEW_TICKET, "X로 문의하기"],
        ["문제", "보고하기"],
    ]
    zone(left(view)).on_click(None)
    await asyncio.gather(*tasks)
    assert texts(left(view))[1:] == ["실패", "다시 보내기"]
    zone(left(view)).on_click(None)
    await asyncio.gather(*tasks)
    assert texts(left(view))[1:] == [PREVIEW_TICKET, "X로 문의하기"]
    assert card_calls(live) == []


@pytest.mark.asyncio
async def test_change_preview_covers_single_stacked_long_and_empty_cards():
    live, preview, view, tasks = previewing()
    preview.cycle("change")
    assert texts(left(view))[1:] == [
        "음성 인식",
        provider_label("deepgram"),
        provider_label("gemini_transcribe"),
    ]
    preview.cycle("change")
    assert len(zone(left(view)).content.controls) == 2
    preview.cycle("change")
    assert "Headset Microphone" in texts(left(view))[-1]
    preview.cycle("change")
    assert texts(left(view))[2] == "—"
    preview.cycle("change")
    assert texts(left(view))[1:] == ["Deepgram API 키", "저장됨", "없음"]
    preview.cycle("change")
    assert texts(left(view))[1:] == ["ChatGPT 계정", "로그인됨", "로그아웃"]
    zone(left(view)).on_click(None)
    await asyncio.gather(*tasks)
    assert left(view) is view._home
    assert card_calls(live) == []


@pytest.mark.asyncio
async def test_key_preview_reaches_entry_verification_and_failure_without_saving():
    live, preview, view, tasks = previewing()
    preview.cycle("key")
    assert texts(left(view))[1:] == [provider_label("deepgram"), "API 키 만들기"]
    zone(left(view)).on_click(None)
    field = agent_chat._children(zone(left(view)).content)[2]
    assert field.password and not field.disabled
    field.value = "synthetic-key"
    await field.on_blur(None)
    failed = agent_chat._children(zone(left(view)).content)[2]
    assert failed.border_color == agent_chat.COLOR_ERROR
    preview.cycle("key")
    preview.cycle("key")
    assert agent_chat._children(zone(left(view)).content)[2].disabled
    back(left(view)).on_click(None)
    assert left(view) is view._home
    assert card_calls(live) == []


def test_account_preview_alternates_login_and_usage_and_finishes_on_tap():
    live, preview, view, _ = previewing()
    preview.cycle("account")
    login = texts(left(view))[1:]
    preview.cycle("account")
    usage = texts(left(view))[1:]
    assert login[0] == usage[0] == provider_label("chatgpt")
    assert login[1] != usage[1]
    zone(left(view)).on_click(None)
    assert left(view) is view._home
    assert card_calls(live) == []


def test_preview_keeps_the_live_conversation_and_hands_back_the_owner():
    live, preview, view, _ = previewing()
    preview.cycle("report")
    live.publish(
        replace(live.snapshot, messages=(*live.snapshot.messages, AgentMessage("user", "새 질문")))
    )
    assert any("새 질문" in value for value in texts(view._messages))
    assert texts(left(view))[1:] == ["문제", "보고하기"]
    back(left(view)).on_click(None)
    assert view._owner is live
    assert left(view) is view._home
