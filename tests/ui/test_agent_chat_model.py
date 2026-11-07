from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.app.ports.agent_chat import (
    AgentChatSnapshot,
    AgentMessage,
    AgentPresentation,
    AgentReportOffer,
    AgentSecureEntryRequest,
    AgentSetupAction,
    AgentToolResult,
)
from puripuly_heart.ui.agent_chat_model import (
    active_status,
    change_blocks,
    chat_items,
    done_status,
    history_rows,
    select_left,
)
from puripuly_heart.ui.i18n import get_locale, language_name, set_locale
from puripuly_heart.ui.views.logs import ConversationRecord


@pytest.fixture(autouse=True)
def korean():
    previous = get_locale()
    set_locale("ko")
    yield
    set_locale(previous)


def snapshot(**fields) -> AgentChatSnapshot:
    base = AgentChatSnapshot(enabled=True, busy=False, phase="idle", messages=(), tool_results=())
    return replace(base, **fields)


def secure(available=True) -> AgentSecureEntryRequest:
    return AgentSecureEntryRequest(
        "secure-1", "deepgram", "task", 1, message_index=1, available=available, status="ready"
    )


def console(available=True) -> AgentSetupAction:
    return AgentSetupAction(
        "action-1",
        "deepgram.key_console",
        "deepgram",
        "key_console",
        1,
        message_index=1,
        available=available,
        status="available",
        secure_request_id="secure-1",
    )


def test_left_area_follows_the_agreed_priority_and_waits_for_the_reply():
    report = AgentReportOffer("support-1", 1, "offered")
    card = {"id": "card-1", "changes": {"stt.provider": "deepgram"}}
    full = snapshot(
        secure_entry_requests=(secure(),),
        setup_actions=(console(),),
        confirmation=card,
        report_offers=(report,),
    )
    assert select_left(full).kind == "key_create"
    assert select_left(full, key_entry_step={"secure-1"}).kind == "key_entry"
    assert select_left(full, dismissed={"secure:secure-1"}).kind == "change"
    assert (
        select_left(replace(full, confirmation=None), dismissed={"secure:secure-1"}).kind
        == "report"
    )
    streaming = replace(full, presentation=AgentPresentation(1, 0, ("first",), completed=False))
    assert select_left(streaming).kind == "change"
    assert select_left(snapshot(), history_open=True).kind == "history"
    assert select_left(snapshot()).kind == "home"


def test_progress_line_is_app_defined_and_never_model_text():
    working = snapshot(busy=True, phase="working", active_tool=("get_app_context", "diagnostics"))
    assert active_status(working).text == "최근 오류 확인 중"
    assert active_status(working).busy is True
    assert active_status(snapshot(busy=True, phase="thinking")).text == "생각 중"
    results = (
        AgentToolResult("a", "overview", "observed", {}),
        AgentToolResult("b", "diagnostics", "observed", {}),
        AgentToolResult(
            "c",
            None,
            "applied",
            {"changes": {"stt.provider": "gemini_transcribe"}},
            "change_settings",
        ),
        AgentToolResult("d", None, "rejected", {}, "diagnose_microphone"),
        AgentToolResult("e", None, "withdrawn", {}, "withdraw_confirmation"),
    )
    assert done_status(results) == "앱 상태 · 최근 오류 확인함 · 음성 인식 바꿈"
    assert active_status(snapshot()) is None


def test_change_blocks_merge_coupled_settings_and_format_values():
    card = {
        "changes": {
            "translation.model": "gemma_4",
            "translation.connection": "openrouter",
            "overlay.translation.enabled": False,
            "peer_vad.speech_threshold": 0.35,
            "peer_expected_languages": ["en", "ja", "ko"],
        },
        "previous": {
            "translation.model": "deepseek_v4_flash",
            "translation.connection": "openrouter",
            "overlay.translation.enabled": True,
            "peer_vad.speech_threshold": 0.5,
            "peer_expected_languages": [],
        },
    }
    blocks = change_blocks(card)
    assert [block.label for block in blocks] == [
        "번역",
        "자막 번역 표시",
        "상대 음성 감지 민감도",
        "상대 예상 언어",
    ]
    assert (blocks[1].before, blocks[1].after) == ("켜짐", "꺼짐")
    assert (blocks[2].before, blocks[2].after) == ("50%", "35%")
    assert (blocks[3].before, blocks[3].after) == ("0개", "3개")
    assert "translation.model" not in repr(blocks)


def test_change_blocks_keep_routes_indivisible_and_each_language_separate():
    blocks = change_blocks(
        {
            "changes": {
                "translation.model": "gemma_4",
                "translation.connection": "openrouter",
                "audio.input_host_api": "Windows WASAPI",
                "audio.input_device": "Microphone 2",
                "languages": {"source": "en", "target": "de"},
            },
            "previous": {
                "translation.model": "deepseek_v4_flash",
                "translation.connection": "openrouter",
                "audio.input_host_api": "MME",
                "audio.input_device": "Microphone 1",
                "languages": {"source": "ko", "target": "ja"},
            },
        }
    )
    assert len(blocks) == 4
    assert blocks[0].label == "번역"
    assert (blocks[1].before, blocks[1].after) == ("Microphone 1", "Microphone 2")
    assert [(block.before, block.after) for block in blocks[2:]] == [
        (language_name("ko"), language_name("en")),
        (language_name("ja"), language_name("de")),
    ]


def test_chat_items_hide_machine_notices_and_restart_with_reset_line():
    messages = (
        AgentMessage("user", "안 들려요", archived=True),
        AgentMessage("assistant", "마이크는 정상이에요. 상대 소리를 볼게요.", archived=True),
        AgentMessage("notice", "[app notice] key_verification provider=openai result=success"),
        AgentMessage("notice", "30분이 지나 새 대화로 시작했어요"),
        AgentMessage("notice", "오늘 쓸 수 있는 대화 한도를 다 썼어요"),
    )
    items = chat_items(snapshot(messages=messages), turn_lines={1: "앱 상태 확인함"})
    kinds = [item.kind for item in items]
    assert kinds == ["greeting", "user", "status", "assistant", "reset", "greeting", "system"]
    assert items[3].lines == ("마이크는 정상이에요.", "상대 소리를 볼게요.")
    assert items[3].archived is True
    assert not any("[app notice]" in line for item in items for line in item.lines)
    assert items[0].archived is True and items[5].archived is False


def test_manual_reset_dims_the_old_greeting_without_repeating_it():
    messages = (
        AgentMessage("user", "안 들려요", archived=True),
        AgentMessage("notice", "새 대화로 시작했어요"),
        AgentMessage("user", "다시 볼게요"),
    )
    items = chat_items(snapshot(messages=messages), turn_lines={})
    assert [(item.kind, item.archived) for item in items] == [
        ("greeting", True),
        ("user", True),
        ("reset", False),
        ("user", False),
    ]


def record(channel, utterance, source, translation, **fields):
    return ConversationRecord(
        "21:02",
        "Mic" if channel == "self" else "Peer",
        channel,
        source,
        translation,
        utterance_id=utterance,
        **fields,
    )


def test_history_groups_one_utterance_and_keeps_separate_utterances_apart():
    rows = history_rows(
        [
            record("self", "u1", "안녕하세요", "Hello", target_index=0),
            record("self", "u1", "안녕하세요", "こんにちは", target_index=1),
            record("peer", "p1", "Where are you from?", None),
            record("peer", "p2", None, "어디에서 오셨어요?"),
            record("self", "u2", "들려요?", None, disposition="failed"),
        ]
    )
    assert [(row.mine, row.source, row.translations, row.failed) for row in rows] == [
        (True, "안녕하세요", ("Hello", "こんにちは"), False),
        (False, "Where are you from?", (), False),
        (False, None, ("어디에서 오셨어요?",), False),
        (True, "들려요?", (), True),
    ]


def test_history_joins_split_original_and_translation_records_of_one_utterance():
    records = [
        record("self", "m1", "안녕", None, parent_utterance_id="m1"),
        record("peer", "p1", "Hi there", "안녕하세요", parent_utterance_id="p1"),
        record("self", "c1", None, "你好", parent_utterance_id="m1", target_index=0),
        record("self", "s1", "first", None, parent_utterance_id="s"),
        record("self", "s1a", None, "en:first", parent_utterance_id="s"),
        record("self", "s1", "second", None, parent_utterance_id="s"),
        record("self", "s1b", None, "en:second", parent_utterance_id="s"),
    ]
    assert [(row.mine, row.source, row.translations) for row in history_rows(records)] == [
        (True, "안녕", ("你好",)),
        (False, "Hi there", ("안녕하세요",)),
        (True, "first second", ("en:first", "en:second")),
    ]
