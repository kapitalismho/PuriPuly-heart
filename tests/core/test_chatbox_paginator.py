from __future__ import annotations

import logging
import uuid

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.domain.models import OSCMessage
from tests.helpers.fakes import FakeSender


class FakeRuntimeLogging:
    def __init__(self, *, detailed_enabled: bool = False) -> None:
        self.detailed_enabled = detailed_enabled
        self.basic: list[tuple[int, str]] = []
        self.detailed: list[tuple[int, str]] = []

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self.basic.append((level, message))

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool:
        if not self.detailed_enabled:
            return False
        self.detailed.append((level, message))
        return True


class FailingSender(FakeSender):
    def send_chatbox(self, text: str) -> None:
        _ = text
        raise OSError("boom")

    def send_typing(self, is_typing: bool) -> None:
        _ = is_typing
        raise OSError("boom")


class SelectivelyFailingSender(FakeSender):
    def __init__(self, *, fail_texts: set[str]) -> None:
        super().__init__()
        self.fail_texts = fail_texts
        self.attempted: list[str] = []

    def send_chatbox(self, text: str) -> None:
        self.attempted.append(text)
        if text in self.fail_texts:
            raise OSError("boom")
        super().send_chatbox(text)


def _message(text: str, clock: FakeClock) -> OSCMessage:
    return OSCMessage(uuid.uuid4(), text=text, created_at=clock.now())


def test_short_message_sends_immediately_without_cooldown() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(sender=sender, clock=clock, page_interval_s=3.0)

    paginator.enqueue(_message("hello", clock))
    clock.advance(0.5)
    paginator.enqueue(_message("world", clock))

    assert sender.sent == ["hello", "world"]


def test_default_limits_send_144_chars_immediately_and_paginate_145_chars_every_3s() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(sender=sender, clock=clock)

    text_144 = "x" * 144
    text_145 = "y" * 145

    paginator.enqueue(_message(text_144, clock))
    paginator.enqueue(_message(text_145, clock))

    assert sender.sent == [text_144, text_145[:144]]

    clock.advance(2.9)
    paginator.process_due()
    assert sender.sent == [text_144, text_145[:144]]

    clock.advance(0.1)
    paginator.process_due()
    assert sender.sent == [text_144, text_145[:144], text_145[144:]]


def test_long_message_sends_first_page_immediately_and_later_pages_each_wait_interval() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=5,
        page_interval_s=3.0,
    )

    paginator.enqueue(_message("abcdefghijklmnop", clock))

    assert sender.sent == ["abcde"]

    clock.advance(2.9)
    paginator.process_due()
    assert sender.sent == ["abcde"]

    clock.advance(0.1)
    paginator.process_due()
    assert sender.sent == ["abcde", "fghij"]

    paginator.process_due()
    assert sender.sent == ["abcde", "fghij"]

    clock.advance(2.9)
    paginator.process_due()
    assert sender.sent == ["abcde", "fghij"]

    clock.advance(0.1)
    paginator.process_due()
    assert sender.sent == ["abcde", "fghij", "klmno"]


def test_messages_arriving_during_pagination_wait_until_pages_finish_then_short_messages_drain() -> (
    None
):
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=4,
        page_interval_s=3.0,
    )

    paginator.enqueue(_message("abcdefghijkl", clock))
    paginator.enqueue(_message("one", clock))
    paginator.enqueue(_message("two", clock))

    assert sender.sent == ["abcd"]

    clock.advance(3.0)
    paginator.process_due()
    assert sender.sent == ["abcd", "efgh"]

    clock.advance(3.0)
    paginator.process_due()
    assert sender.sent == ["abcd", "efgh", "ijkl", "one", "two"]


def test_long_message_arriving_during_pagination_starts_after_active_pages_finish() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=4,
        page_interval_s=3.0,
    )

    paginator.enqueue(_message("abcdefghijkl", clock))
    paginator.enqueue(_message("mnopqrst", clock))

    assert sender.sent == ["abcd"]

    clock.advance(3.0)
    paginator.process_due()
    assert sender.sent == ["abcd", "efgh"]

    clock.advance(3.0)
    paginator.process_due()
    assert sender.sent == ["abcd", "efgh", "ijkl", "mnop"]

    clock.advance(3.0)
    paginator.process_due()
    assert sender.sent == ["abcd", "efgh", "ijkl", "mnop", "qrst"]


def test_queued_messages_do_not_expire_behind_long_pagination() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=2,
        page_interval_s=3.0,
    )

    paginator.enqueue(_message("abcdefghij", clock))
    paginator.enqueue(_message("ok", clock))

    for _ in range(4):
        clock.advance(3.0)
        paginator.process_due()

    assert clock.now() == 12.0
    assert sender.sent == ["ab", "cd", "ef", "gh", "ij", "ok"]


def test_failed_page_is_dropped_without_retrying_or_blocking_later_pages() -> None:
    clock = FakeClock()
    sender = SelectivelyFailingSender(fail_texts={"fghij"})
    runtime_logging = FakeRuntimeLogging()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=5,
        page_interval_s=3.0,
        runtime_logging=runtime_logging,
    )

    paginator.enqueue(_message("abcdefghijklmnop", clock))
    assert sender.attempted == ["abcde"]
    assert sender.sent == ["abcde"]

    clock.advance(3.0)
    paginator.process_due()
    assert sender.attempted == ["abcde", "fghij"]
    assert sender.sent == ["abcde"]

    paginator.process_due()
    assert sender.attempted == ["abcde", "fghij"]

    clock.advance(2.9)
    paginator.process_due()
    assert sender.attempted == ["abcde", "fghij"]

    clock.advance(0.1)
    paginator.process_due()
    assert sender.attempted == ["abcde", "fghij", "klmno"]
    assert sender.sent == ["abcde", "klmno"]

    clock.advance(3.0)
    paginator.process_due()

    assert sender.attempted == ["abcde", "fghij", "klmno", "p"]
    assert sender.sent == ["abcde", "klmno", "p"]
    assert (
        logging.WARNING,
        "[Basic][OSC] send mode=queued status=failed error=boom",
    ) in runtime_logging.basic


def test_send_immediate_failure_returns_false_and_logs_basic_warning() -> None:
    clock = FakeClock()
    sender = FailingSender()
    runtime_logging = FakeRuntimeLogging()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        runtime_logging=runtime_logging,
    )

    sent = paginator.send_immediate("promo")

    assert sent is False
    assert runtime_logging.basic == [
        (logging.WARNING, "[Basic][OSC] send mode=immediate status=failed error=boom")
    ]


def test_typing_indicator_is_forwarded_and_failure_is_basic_log() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(sender=sender, clock=clock)

    paginator.send_typing(True)

    assert sender.typing == [True]

    failing_sender = FailingSender()
    runtime_logging = FakeRuntimeLogging()
    failing_paginator = ChatboxPaginator(
        sender=failing_sender,
        clock=clock,
        runtime_logging=runtime_logging,
    )
    failing_paginator.send_typing(False)

    assert runtime_logging.basic == [
        (logging.WARNING, "[Basic][OSC] typing status=failed error=boom")
    ]


def test_typing_reasons_keep_indicator_visible_until_all_reasons_clear() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(sender=sender, clock=clock)

    paginator.set_typing_reason("manual_input", True)
    paginator.set_typing_reason("manual_submit:1", True)
    paginator.set_typing_reason("manual_input", False)
    paginator.send_typing(False)
    paginator.set_typing_reason("manual_submit:1", False)

    assert sender.typing == [True, False]


def test_legacy_typing_keeps_indicator_visible_until_legacy_state_clears() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(sender=sender, clock=clock)

    paginator.send_typing(True)
    paginator.set_typing_reason("manual_input", True)
    paginator.set_typing_reason("manual_input", False)
    paginator.clear_typing_reasons()
    paginator.send_typing(False)

    assert sender.typing == [True, False]


def test_clear_typing_reasons_clears_active_aggregate_state() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(sender=sender, clock=clock)

    paginator.set_typing_reason("manual_input", True)
    paginator.set_typing_reason("manual_submit:1", True)
    paginator.clear_typing_reasons()

    assert sender.typing == [True, False]


def test_send_immediate_trims_and_does_not_delay_pagination() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=5,
        page_interval_s=3.0,
    )

    paginator.enqueue(_message("abcdefghijkl", clock))
    clock.advance(1.0)
    sent = paginator.send_immediate(" promo ")
    clock.advance(1.9)
    paginator.process_due()
    assert sender.sent == ["abcde", "promo"]

    clock.advance(0.1)
    paginator.process_due()

    assert sent is True
    assert sender.sent == ["abcde", "promo", "fghij"]


def test_send_immediate_sends_long_text_as_one_packet() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=5,
        page_interval_s=3.0,
    )

    sent = paginator.send_immediate("abcdefghij")
    clock.advance(3.0)
    paginator.process_due()

    assert sent is True
    assert sender.sent == ["abcdefghij"]


def test_invalid_limits_raise_value_error() -> None:
    clock = FakeClock()
    sender = FakeSender()

    with pytest.raises(ValueError, match="max_chars"):
        ChatboxPaginator(sender=sender, clock=clock, max_chars=0)

    with pytest.raises(ValueError, match="page_interval_s"):
        ChatboxPaginator(sender=sender, clock=clock, page_interval_s=0)


def test_chatbox_stage_recorder_separates_enqueue_from_udp_page_send() -> None:
    clock = FakeClock()
    sender = FakeSender()
    stages: list[tuple[str, dict[str, object]]] = []

    def record(event: str, **fields: object) -> None:
        stages.append((event, fields))

    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=4,
        page_interval_s=3.0,
        stage_recorder=record,
    )
    first = _message("abcdefgh", clock)
    second = _message("ijkl", clock)

    paginator.enqueue(first)
    paginator.enqueue(second)

    assert [event for event, _fields in stages] == [
        "message_enqueue",
        "page_send",
        "message_enqueue",
    ]
    assert stages[0][1]["started"] is True
    assert stages[1][1]["page_index"] == 0
    assert stages[1][1]["remaining_parts"] == 1
    assert stages[2][1]["started"] is False
    assert stages[2][1]["pending_messages"] == 1
    assert "text" not in stages[1][1]
    assert all("abcdefgh" not in str(fields) for _event, fields in stages)

    clock.advance(3.0)
    paginator.process_due()

    first_pages = [
        fields
        for event, fields in stages
        if event == "page_send" and fields["utterance_id"] == str(first.utterance_id)
    ]
    assert [fields["page_index"] for fields in first_pages] == [0, 1]
    assert first_pages[1]["remaining_parts"] == 0
