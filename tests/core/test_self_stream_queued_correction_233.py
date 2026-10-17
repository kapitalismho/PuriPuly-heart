from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.vad.gating import SpeechEnd
from puripuly_heart.domain.events import SelfTurnPresentation, UIEventType
from puripuly_heart.domain.models import OSCMessage
from tests.helpers.self_stream import (
    Provider,
    Sender,
    begin,
    contribution,
    until,
)
from tests.helpers.translation_owners import compose_translation_test_harness


async def next_snapshot(
    harness,
    predicate: Callable[[SelfTurnPresentation], bool],
) -> SelfTurnPresentation:
    async with asyncio.timeout(3):
        while True:
            event = await harness.ui_events.get()
            if event.type is UIEventType.SELF_TURN_UPDATED and predicate(event.payload):
                return event.payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_chars", "first_correction_page"),
    [(12, "en:alpha"), (20, "en:alpha changed")],
)
async def test_queued_changed_correction_sends_before_ready_successor_without_all_pages_barrier(
    max_chars: int,
    first_correction_page: str,
) -> None:
    provider = Provider()
    sender = Sender()
    clock = FakeClock()
    paginator = ChatboxPaginator(sender=sender, clock=clock, max_chars=max_chars)
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=paginator,
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=10,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    try:
        paginator.enqueue(OSCMessage(uuid4(), "x" * (2 * max_chars), created_at=clock.now()))
        identity = await begin(harness, "alpha")
        provider.release("alpha")
        preview = await next_snapshot(
            harness,
            lambda snapshot: snapshot.source_text == "alpha"
            and snapshot.targets[0].status == "ready",
        )
        assert not preview.source_closed
        await contribution(harness, identity, "alpha changed")
        provider.release("alpha changed")
        correction = await next_snapshot(
            harness,
            lambda snapshot: snapshot.source_text == "alpha changed"
            and snapshot.targets[0].status == "ready",
        )
        assert correction.utterance_id == preview.utterance_id
        assert correction.source_revision > preview.source_revision
        assert correction.revision > preview.revision
        assert not correction.source_closed
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
        sealed = await next_snapshot(
            harness,
            lambda snapshot: snapshot.utterance_id == preview.utterance_id
            and snapshot.status == "complete",
        )
        assert sealed.source_closed
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        successor = await begin(harness, "bravo")
        provider.release("bravo")
        successor_preview = await next_snapshot(
            harness,
            lambda snapshot: snapshot.source_text == "bravo"
            and snapshot.targets[0].status == "ready",
        )
        assert successor_preview.utterance_id != sealed.utterance_id
        assert not successor_preview.source_closed
        assert [call["text"] for call in provider.calls] == ["alpha", "alpha changed", "bravo"]
        assert provider.cancelled == []
        assert sender.packets == ["x" * max_chars]
        clock.advance(3)
        paginator.process_due()
        await until(lambda: len(sender.packets) == 4)
        assert sender.packets == [
            "x" * max_chars,
            "x" * max_chars,
            first_correction_page,
            "en:bravo",
        ]
        await harness.self_owner.handle_vad_event(SpeechEnd(successor))
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        clock.advance(3)
        paginator.process_due()
        assert len(sender.packets) == 4
        assert [entry.text for entry in harness.self_runtime.translation_history] == [
            "alpha changed",
            "bravo",
        ]
        assert len(provider.calls) == 3
    finally:
        await harness.stop()
