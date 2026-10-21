from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import SelfTurnPresentation, STTFinalEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.fakes import FakeSender
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass
class ControlledProvider:
    calls: list[dict] = field(default_factory=list)
    releases: dict[tuple[str, str], asyncio.Event] = field(default_factory=dict)

    async def translate(self, *, utterance_id, text, target_language, context="", **kwargs):
        self.calls.append(dict(text=text, target=target_language, context=context))
        await self.releases.setdefault((text, target_language), asyncio.Event()).wait()
        return Translation(utterance_id=utterance_id, text=f"{target_language}:{text}")

    async def close(self):
        return None


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


async def source(harness, text, *, end=True, identity=None):
    identity = identity or uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(
            identity, pre_roll=np.empty(0, dtype=np.float32), chunk=np.ones(512, dtype=np.float32)
        )
    )
    if end:
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
    await harness.dispatch_stt_event(
        STTFinalEvent(identity, Transcript(utterance_id=identity, text=text, is_final=True))
    )
    return identity


@pytest.mark.asyncio
@pytest.mark.parametrize("targets", [("en",), ("en", "ja")])
async def test_preview_before_close_reuses_running_and_identical_final_without_overlay(targets):
    clock = SystemClock()
    sender = FakeSender()
    provider = ControlledProvider()
    paginator = ChatboxPaginator(sender=sender, clock=clock)
    harness = compose_translation_test_harness(
        chatbox_include_source=False,
        llm=provider,
        stt=None,
        osc=paginator,
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=80,
        self_target_languages=targets,
    )
    try:
        await source(harness, "alpha")
        await eventually(lambda: len(provider.calls) == len(targets))
        parent_id = harness.self_owner.merge_buffer.merge_id
        tasks = dict(harness.translation_turns._parents[parent_id].operations)
        provider.releases[("alpha", targets[-1])].set()
        await eventually(lambda: bool(sender.sent))
        assert harness.self_owner.merge_buffer is not None
        preview_packets = list(sender.sent)
        await eventually(lambda: harness.self_owner.merge_buffer is None)
        parent = harness.translation_turns._parents.get(parent_id)
        if parent is not None:
            assert all(parent.operations[index] is task for index, task in tasks.items())
        assert len(provider.calls) == len(targets)
        for release in provider.releases.values():
            release.set()
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert len(provider.calls) == len(targets)
        if len(targets) == 1:
            assert sender.sent == preview_packets
        else:
            assert sender.sent[-1] == "en:alpha\nja:alpha"
        events = []
        while not harness.ui_events.empty():
            events.append(harness.ui_events.get_nowait())
        snapshots = [
            event.payload for event in events if event.type is UIEventType.SELF_TURN_UPDATED
        ]
        assert snapshots and isinstance(snapshots[-1], SelfTurnPresentation)
        assert snapshots[-1].source_closed and snapshots[-1].status == "complete"
        assert snapshots[0].utterance_id == snapshots[-1].utterance_id
        assert not any(
            event.type in {UIEventType.TRANSCRIPT_FINAL, UIEventType.TRANSLATION_DONE}
            for event in events
        )
        assert [entry.text for entry in harness.self_runtime.translation_history] == [
            "alpha"
        ] * len(targets)
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_late_predecessor_correction_gets_first_page_before_successor_preview():
    clock = SystemClock()
    sender = FakeSender()
    provider = ControlledProvider()
    harness = compose_translation_test_harness(
        llm=provider,
        stt=None,
        osc=ChatboxPaginator(sender=sender, clock=clock),
        chatbox_include_source=False,
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=20,
        self_target_languages=("en",),
    )
    try:
        await source(harness, "alpha")
        await eventually(lambda: harness.self_owner.merge_buffer is None)
        await source(harness, "bravo")
        await eventually(lambda: len(provider.calls) == 2)
        provider.releases[("bravo", "en")].set()
        await asyncio.sleep(0.005)
        assert sender.sent == []
        provider.releases[("alpha", "en")].set()
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert sender.sent == ["en:alpha", "en:bravo"]
        assert [entry.text for entry in harness.self_runtime.translation_history] == [
            "alpha",
            "bravo",
        ]
    finally:
        await harness.stop()
