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


def _message(
    text: str,
    clock: FakeClock,
    *,
    utterance_id: uuid.UUID | None = None,
    turn_generation: int | None = None,
    turn_order: int | None = None,
    presentation_revision: int = 0,
    target_indexes: tuple[int, ...] = (),
    target_languages: tuple[str, ...] = (),
) -> OSCMessage:
    return OSCMessage(
        utterance_id or uuid.uuid4(),
        text=text,
        created_at=clock.now(),
        turn_generation=turn_generation,
        turn_order=turn_order,
        presentation_revision=presentation_revision,
        target_indexes=target_indexes,
        target_languages=target_languages,
    )


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


def test_newer_active_revision_replaces_all_unsent_pages() -> None:
    clock = FakeClock()
    sender = FakeSender()
    stages: list[tuple[str, dict[str, object]]] = []
    parent_id = uuid.uuid4()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=4,
        page_interval_s=3.0,
        stage_recorder=lambda event, **fields: stages.append((event, fields)),
    )

    paginator.enqueue(
        _message(
            "abcdefgh",
            clock,
            utterance_id=parent_id,
            turn_generation=2,
            turn_order=7,
            presentation_revision=1,
            target_indexes=(1,),
            target_languages=("ja",),
        )
    )
    paginator.enqueue(
        _message(
            "12345678",
            clock,
            utterance_id=parent_id,
            turn_generation=2,
            turn_order=7,
            presentation_revision=2,
            target_indexes=(0, 1),
            target_languages=("zh-CN", "ja"),
        )
    )
    paginator.enqueue(
        _message(
            "abcdefgh",
            clock,
            utterance_id=parent_id,
            turn_generation=2,
            turn_order=7,
            presentation_revision=1,
            target_indexes=(1,),
            target_languages=("ja",),
        )
    )
    clock.advance(3.0)
    paginator.process_due()

    assert sender.sent == ["abcd", "1234", "5678"]
    replacement = next(fields for event, fields in stages if event == "chatbox_revision_replaced")
    assert replacement["previous_revision"] == 1
    assert replacement["presentation_revision"] == 2
    assert replacement["turn_generation"] == 2
    assert replacement["turn_order"] == 7
    assert replacement["target_indexes"] == (0, 1)
    assert replacement["target_languages"] == ("zh-CN", "ja")
    assert all("abcdefgh" not in str(fields) for _event, fields in stages)


def test_newer_pending_revision_replaces_in_place_behind_system_message() -> None:
    clock = FakeClock()
    sender = FakeSender()
    parent_id = uuid.uuid4()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=4,
        page_interval_s=3.0,
    )

    paginator.enqueue(_message("system!!", clock))
    paginator.enqueue(
        _message(
            "old",
            clock,
            utterance_id=parent_id,
            turn_generation=0,
            turn_order=1,
            presentation_revision=1,
        )
    )
    paginator.enqueue(
        _message(
            "new",
            clock,
            utterance_id=parent_id,
            turn_generation=0,
            turn_order=1,
            presentation_revision=2,
        )
    )
    clock.advance(3.0)
    paginator.process_due()

    assert sender.sent == ["syst", "em!!", "new"]


def test_newer_turn_prunes_active_pages_and_preserves_system_disclosure() -> None:
    clock = FakeClock()
    sender = FakeSender()
    stages: list[tuple[str, dict[str, object]]] = []
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=4,
        page_interval_s=3.0,
        stage_recorder=lambda event, **fields: stages.append((event, fields)),
    )

    paginator.enqueue(
        _message(
            "old-turn",
            clock,
            turn_generation=0,
            turn_order=4,
            presentation_revision=1,
        )
    )
    paginator.enqueue(_message("notice", clock))
    paginator.enqueue(
        _message(
            "new",
            clock,
            turn_generation=0,
            turn_order=5,
            presentation_revision=1,
            target_indexes=(0,),
            target_languages=("zh-CN",),
        )
    )
    clock.advance(3.0)
    paginator.process_due()

    assert sender.sent == ["old-", "new", "noti", "ce"]
    prune = next(fields for event, fields in stages if event == "chatbox_older_turn_pruned")
    assert prune["pruned_pages"] == 1
    assert prune["pruned_messages"] == 0
    assert prune["turn_order"] == 5
    assert prune["target_indexes"] == (0,)
    assert prune["target_languages"] == ("zh-CN",)
    assert all("old-turn" not in str(fields) for _event, fields in stages)


def test_newer_turn_replaces_pending_older_turn_without_dropping_system_routes() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=4,
        page_interval_s=3.0,
    )

    paginator.enqueue(_message("system!!", clock))
    paginator.enqueue(
        _message(
            "old",
            clock,
            turn_generation=3,
            turn_order=8,
            presentation_revision=1,
        )
    )
    paginator.enqueue(_message("note", clock))
    paginator.enqueue(
        _message(
            "new",
            clock,
            turn_generation=3,
            turn_order=9,
            presentation_revision=1,
        )
    )
    clock.advance(3.0)
    paginator.process_due()

    assert sender.sent == ["syst", "em!!", "new", "note"]


def test_two_line_translation_preserves_newline_across_pagination() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=5,
        page_interval_s=3.0,
    )
    text = "abcd\nwxyz"

    paginator.enqueue(
        _message(
            text,
            clock,
            turn_generation=1,
            turn_order=0,
            presentation_revision=2,
        )
    )
    clock.advance(3.0)
    paginator.process_due()

    assert sender.sent == ["abcd\n", "wxyz"]
    assert "".join(sender.sent) == text
    assert all(len(page) <= 5 for page in sender.sent)


def test_metadata_free_multiline_message_keeps_legacy_wrapping() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=5,
        page_interval_s=3.0,
    )

    paginator.enqueue(_message("abcd\nefgh", clock))
    clock.advance(3.0)
    paginator.process_due()

    assert sender.sent == ["abcd", "efgh"]


def test_single_target_turn_identity_keeps_legacy_multiline_wrapping() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=5,
        page_interval_s=3.0,
    )

    paginator.enqueue(
        _message(
            "abcd\nefgh",
            clock,
            turn_generation=1,
            turn_order=0,
            presentation_revision=0,
        )
    )
    clock.advance(3.0)
    paginator.process_due()

    assert sender.sent == ["abcd", "efgh"]


@pytest.mark.parametrize(
    ("older_revision", "newer_revision"),
    [(1, 0), (0, 1)],
)
def test_newer_self_turn_prunes_active_pages_across_single_dual_transitions(
    older_revision: int,
    newer_revision: int,
) -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=4,
        page_interval_s=3.0,
    )

    paginator.enqueue(
        _message(
            "abcdefgh",
            clock,
            turn_generation=0,
            turn_order=0,
            presentation_revision=older_revision,
        )
    )
    paginator.enqueue(
        _message(
            "NEW",
            clock,
            turn_generation=1,
            turn_order=0,
            presentation_revision=newer_revision,
        )
    )
    clock.advance(3.0)
    paginator.process_due()

    assert sender.sent == ["abcd", "NEW"]


def test_latest_self_turn_watermark_rejects_stale_messages_after_queue_drains() -> None:
    clock = FakeClock()
    sender = FakeSender()
    paginator = ChatboxPaginator(sender=sender, clock=clock)
    current_id = uuid.uuid4()

    paginator.enqueue(
        _message(
            "current",
            clock,
            utterance_id=current_id,
            turn_generation=4,
            turn_order=3,
            presentation_revision=2,
        )
    )
    paginator.enqueue(
        _message(
            "older revision",
            clock,
            utterance_id=current_id,
            turn_generation=4,
            turn_order=3,
            presentation_revision=1,
        )
    )
    paginator.enqueue(
        _message(
            "older turn",
            clock,
            turn_generation=4,
            turn_order=2,
            presentation_revision=3,
        )
    )

    assert sender.sent == ["current"]


@pytest.mark.parametrize(
    ("fields", "error_type"),
    [
        ({"presentation_revision": True}, TypeError),
        ({"presentation_revision": 1.5}, TypeError),
        ({"presentation_revision": float("nan")}, TypeError),
        ({"turn_generation": True, "turn_order": 0}, TypeError),
        ({"turn_generation": 0, "turn_order": 1.5}, TypeError),
        ({"presentation_revision": 1}, ValueError),
    ],
)
def test_osc_message_rejects_malformed_self_turn_identity(
    fields: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        OSCMessage(
            utterance_id=uuid.uuid4(),
            text="translation",
            created_at=0.0,
            **fields,
        )
