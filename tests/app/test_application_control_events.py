from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from puripuly_heart.app.services.application_control_events import ControlEvents
from puripuly_heart.domain.events import (
    SelfTurnPresentation,
    SelfTurnTargetPresentation,
    UIEvent,
    UIEventType,
)
from puripuly_heart.domain.models import OSCMessage, Transcript, Translation


async def _collect(stream, count):
    async with asyncio.timeout(5):
        return [await anext(stream) for _ in range(count)]


@pytest.mark.parametrize(
    ("event_count", "after", "expected_gap"),
    ((256, 0, None), (257, 0, 1), (257, 1, None)),
)
async def test_replay_window_and_gap_boundary(event_count, after, expected_gap):
    events = ControlEvents()
    for index in range(event_count):
        events.publish({"topic": "state", "index": index})

    stream = events.subscribe(
        topics=[],
        channel=None,
        include_transcripts=False,
        include_translations=False,
        after=after,
    )
    oldest = max(1, event_count - 255)
    expected_sequences = list(range(max(after + 1, oldest), event_count + 1))
    items = await _collect(stream, len(expected_sequences) + int(expected_gap is not None))
    await stream.aclose()

    gaps = [item for item in items if item["topic"] == "gap"]
    assert len(gaps) == (expected_gap is not None)
    if expected_gap is not None:
        assert gaps[0]["sequence"] == expected_gap
        assert gaps[0]["after"] == after
        assert gaps[0]["reason"] == "history_evicted"
        assert gaps[0]["snapshot_required"] is True
    replayed = [item for item in items if item["topic"] != "gap"]
    assert [item["sequence"] for item in replayed] == expected_sequences
    assert [item["index"] for item in replayed] == list(range(max(after, oldest - 1), event_count))


@pytest.mark.asyncio
async def test_slow_subscriber_gets_gap_then_retained_events_in_order():
    events = ControlEvents()
    stream = events.subscribe(
        topics=["capture"],
        channel=None,
        include_transcripts=False,
        include_translations=False,
        after=0,
    )
    first = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    for index in range(300):
        events.publish({"topic": "capture", "index": index})

    gap = await asyncio.wait_for(first, timeout=1)
    replayed = await _collect(stream, 256)
    await stream.aclose()

    assert gap["sequence"] == 44
    assert gap["after"] == 0
    assert gap["reason"] == "history_evicted"
    assert gap["snapshot_required"] is True
    assert [item["sequence"] for item in replayed] == list(range(45, 301))
    assert [item["index"] for item in replayed] == list(range(44, 300))
    assert events.subscribers == set()


@pytest.mark.asyncio
async def test_filters_content_opt_ins_and_osc_privacy_are_independent():
    events = ControlEvents()
    transcript_id = UUID("00000000-0000-0000-0000-000000000001")
    peer_id = UUID("00000000-0000-0000-0000-000000000002")
    translation_id = UUID("00000000-0000-0000-0000-000000000003")
    osc_id = UUID("00000000-0000-0000-0000-000000000004")
    events.publish_ui(
        UIEvent(
            UIEventType.TRANSCRIPT_FINAL,
            payload=Transcript(transcript_id, "transcript private", True, channel="self"),
            source="capture",
        )
    )
    events.publish_ui(
        UIEvent(
            UIEventType.TRANSCRIPT_FINAL,
            payload=Transcript(peer_id, "peer private", True, channel="peer"),
            source="capture",
        )
    )
    events.publish({"topic": "settings", "channel": "self"})
    events.publish_ui(
        UIEvent(
            UIEventType.TRANSLATION_DONE,
            payload=Translation(translation_id, "translation private", channel="self"),
            source="translation",
        )
    )
    events.publish_ui(
        UIEvent(
            UIEventType.OSC_SENT,
            payload=OSCMessage(osc_id, "chatbox private", 1.0),
            source="osc",
        )
    )

    transcript_stream = events.subscribe(
        topics=["transcript"],
        channel="self",
        include_transcripts=True,
        include_translations=False,
        after=0,
    )
    transcript = (await _collect(transcript_stream, 1))[0]
    await transcript_stream.aclose()
    assert transcript["sequence"] == 1
    assert transcript["text"] == "transcript private"

    translation_stream = events.subscribe(
        topics=["translation"],
        channel="self",
        include_transcripts=False,
        include_translations=True,
        after=0,
    )
    translation = (await _collect(translation_stream, 1))[0]
    await translation_stream.aclose()
    assert translation["sequence"] == 4
    assert translation["text"] == "translation private"

    projected_stream = events.subscribe(
        topics=["transcript", "translation", "osc_sent"],
        channel="self",
        include_transcripts=True,
        include_translations=False,
        after=0,
    )
    projected = await _collect(projected_stream, 3)
    await projected_stream.aclose()

    assert [item["sequence"] for item in projected] == [1, 4, 5]
    assert projected[0]["text"] == "transcript private"
    assert "text" not in projected[1]
    assert "text" not in projected[2]
    assert "chatbox private" not in repr(projected)

    private_stream = events.subscribe(
        topics=["transcript", "translation"],
        channel="self",
        include_transcripts=False,
        include_translations=False,
        after=0,
    )
    private = await _collect(private_stream, 2)
    await private_stream.aclose()
    assert [item["sequence"] for item in private] == [1, 4]
    assert all("text" not in item for item in private)


def _self_speech():
    return SelfTurnPresentation(
        utterance_id=uuid4(),
        turn_generation=1,
        turn_order=0,
        revision=1,
        source_revision=1,
        source_text="speech preview",
        source_language="en",
        source_closed=False,
        status="active",
        targets=(),
        created_at=0.0,
    )


@pytest.mark.asyncio
async def test_self_speech_replay_separates_final_source_and_each_current_target():
    events = ControlEvents()
    draft = _self_speech()
    turn = draft.utterance_id
    stale = SelfTurnTargetPresentation("ko", 1, "ready", Translation(turn, "stale"))
    first = replace(stale, source_revision=2, translation=Translation(uuid4(), "first target"))
    second = SelfTurnTargetPresentation("ja", 2, "ready", Translation(uuid4(), "second target"))

    def publish(snapshot):
        events.publish_ui(UIEvent(UIEventType.SELF_TURN_UPDATED, turn, snapshot, source="Mic"))

    publish(replace(draft, targets=(stale,)))
    sealed = replace(
        draft,
        revision=2,
        source_revision=2,
        source_closed=True,
        source_text="final source",
        targets=(stale, replace(second, status="pending", translation=None)),
    )
    publish(sealed)
    assert [(e["topic"], e["text"]) for e in events.history] == [("transcript", "final source")]
    publish(replace(sealed, revision=3, targets=(stale, second)))
    done = replace(sealed, revision=4, status="complete", targets=(first, second))
    publish(done)
    publish(replace(done, revision=5))
    assert [(e["topic"], e.get("target_index")) for e in events.history] == [
        ("transcript", None),
        ("translation", 1),
        ("translation", 0),
    ]
    for transcripts, translations in ((False, False), (True, False), (False, True), (True, True)):
        stream = events.subscribe(
            topics=["transcript", "translation"],
            channel="self",
            include_transcripts=transcripts,
            include_translations=translations,
            after=0,
        )
        rows = await _collect(stream, 3)
        await stream.aclose()
        assert all(row["utterance_id"] == str(turn) and row["channel"] == "self" for row in rows)
        assert [row.get("text") for row in rows] == [
            "final source" if transcripts else None,
            "second target" if translations else None,
            "first target" if translations else None,
        ]


@pytest.mark.parametrize("retirement", ["cancelled", "expired"])
def test_retired_self_speech_cannot_publish_or_suppress_a_fresh_source_only_turn(retirement):
    events = ControlEvents()
    old = replace(_self_speech(), source_closed=True, status="complete")
    current = replace(old, utterance_id=uuid4(), turn_order=1, source_text="fresh")
    event = UIEvent(UIEventType.SELF_TURN_UPDATED, old.utterance_id, old)
    events.publish_ui(replace(event, channel="peer"))
    events.publish_ui(replace(event, delivery_authority=SimpleNamespace(is_current=lambda: False)))
    events.publish_ui(replace(event, payload=replace(old, status=retirement)))
    events.publish_ui(event)
    events.publish_ui(replace(event, utterance_id=current.utterance_id, payload=current))
    assert [(e["topic"], e["utterance_id"], e["text"]) for e in events.history] == [
        ("transcript", str(current.utterance_id), "fresh"),
    ]


@pytest.mark.asyncio
async def test_subscriber_projection_is_copy_isolated_and_unsubscribe_cleans_up():
    events = ControlEvents()
    published = {"topic": "state", "channel": "self", "detail": {"state": "ready"}}
    events.publish(published)
    published["topic"] = "changed"
    published["detail"]["state"] = "changed by publisher"

    first_stream = events.subscribe(
        topics=[], channel=None, include_transcripts=False, include_translations=False, after=0
    )
    first_task = asyncio.create_task(anext(first_stream))
    await asyncio.sleep(0)
    first = await asyncio.wait_for(first_task, timeout=5)
    assert len(events.subscribers) == 1
    first["topic"] = "changed by subscriber"
    first["detail"]["state"] = "changed by subscriber"
    await first_stream.aclose()
    assert events.subscribers == set()

    second_stream = events.subscribe(
        topics=[], channel=None, include_transcripts=False, include_translations=False, after=0
    )
    second = (await _collect(second_stream, 1))[0]
    await second_stream.aclose()

    assert second["sequence"] == 1 and second["topic"] == "state" and second["channel"] == "self"
    assert second["instance_id"] is None
    assert second["detail"] == {"state": "ready"}
    assert events.history[0]["topic"] == "state"
