from dataclasses import FrozenInstanceError, replace
from uuid import uuid4

import pytest

from puripuly_heart.core.conversation_history import ConversationHistory
from puripuly_heart.domain.events import SelfTurnPresentation


def turn() -> SelfTurnPresentation:
    return SelfTurnPresentation(
        utterance_id=uuid4(),
        turn_generation=1,
        turn_order=1,
        revision=1,
        source_revision=1,
        source_text="synthetic speech",
        source_language="en",
        source_closed=False,
        status="active",
        targets=(),
        created_at=1.0,
    )


def test_history_snapshots_are_immutable_and_keep_existing_retention_bounds():
    history = ConversationHistory(max_entries=2, cleanup_batch=1)
    for index in range(3):
        history.append_conversation_record(
            source="You",
            channel="self",
            utterance_id=str(index),
            source_text=f"  source {index}  ",
            translated_text=None,
        )
    before = history.conversation_records
    assert isinstance(before, tuple)
    with pytest.raises(FrozenInstanceError):
        before[0].source_text = "changed"
    history.append_conversation_record(
        source="You",
        channel="self",
        utterance_id="3",
        source_text="source 3",
        translated_text=None,
    )
    assert [record.source_text for record in before] == ["source 0", "source 1", "source 2"]
    assert [record.source_text for record in history.conversation_records] == [
        "source 1",
        "source 2",
        "source 3",
    ]
    history.append_conversation_record(
        source="You",
        channel="self",
        utterance_id="blank",
        source_text=" ",
        translated_text="\n",
    )
    assert len(history.conversation_records) == 3


def test_self_turn_observer_can_unsubscribe_during_delivery_without_skipping_other_observers():
    history = ConversationHistory()
    first, second = [], []

    def observe(snapshot, source):
        first.append((snapshot.source_text, source))
        remove_first()

    remove_first = history.observe_self_turns(observe)
    remove_second = history.observe_self_turns(
        lambda snapshot, source: second.append((snapshot.source_text, source))
    )
    initial = turn()
    history.update_self_turn(initial, source="Mic")
    final = replace(initial, revision=2, source_text="final speech", source_closed=True)
    history.update_self_turn(final, source="Mic")
    remove_second()
    history.update_self_turn(replace(final, revision=3, status="complete"), source="Mic")
    assert first == [("synthetic speech", "Mic")]
    assert second == [("synthetic speech", "Mic"), ("final speech", "Mic")]
    assert history.conversation_records[0].source_text == "final speech"
    assert history.conversation_records[0].turn_status == "complete"
