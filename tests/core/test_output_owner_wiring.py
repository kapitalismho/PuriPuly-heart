from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.overlay.sink import OverlayEventUnion
from puripuly_heart.domain.models import OSCMessage
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass(slots=True)
class RecordingChatbox:
    messages: list[OSCMessage] = field(default_factory=list)

    def enqueue(self, message: OSCMessage) -> None:
        self.messages.append(message)

    def send_typing(self, is_typing: bool) -> None:
        _ = is_typing

    def set_typing_reason(self, reason: str, active: bool) -> None:
        _ = (reason, active)

    def process_due(self) -> None:
        return

    def send_immediate(self, text: str) -> bool:
        _ = text
        return True


@dataclass(slots=True)
class RecordingOverlay:
    events: list[OverlayEventUnion] = field(default_factory=list)

    async def emit(self, event: OverlayEventUnion) -> None:
        self.events.append(event)

    def active_self_overlay_metadata(self) -> None:
        return None


class BlockingOverlay(RecordingOverlay):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def emit(self, event: OverlayEventUnion) -> None:
        self.events.append(event)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.mark.asyncio
async def test_translation_fixture_routes_manual_peer_and_system_output_through_one_owner() -> None:
    chatbox = RecordingChatbox()
    overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=chatbox,
        overlay_sink=overlay,
    )

    await harness.start()
    manual_id = await harness.self_owner.submit_text("manual self text", source="You")
    peer_id = await harness.handle_peer_transcript_final_for_test("peer-only text")
    harness.output_projection.publish_system_disclosure("Peer translation is on")

    assert harness.output_runtime.overlay_sink is overlay
    assert [message.utterance_id for message in chatbox.messages[:1]] == [manual_id]
    assert all("peer-only text" not in message.text for message in chatbox.messages)
    assert len(chatbox.messages) == 2
    assert [event.channel for event in overlay.events] == [
        "self",
        "self",
        "peer",
        "peer",
    ]
    assert [event.utterance_id for event in overlay.events[-2:]] == [peer_id, peer_id]
    peer_chatbox_decisions = [
        decision
        for decision in harness.output_runtime.routing_decisions
        if decision.route == "self_chatbox" and decision.publication_kind == "peer_subtitle"
    ]
    assert len(peer_chatbox_decisions) == 1
    assert peer_chatbox_decisions[0].reason == "peer_chatbox_denied"
    assert all(decision.reason != "duplicate_publication" for decision in peer_chatbox_decisions)

    await harness.stop()

    assert harness.output_runtime.state == "closed"
    assert not harness.output_runtime.has_resources


@pytest.mark.asyncio
async def test_translation_fixture_overlay_replacement_updates_only_owner_destination() -> None:
    first = RecordingOverlay()
    second = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingChatbox(),
        overlay_sink=first,
    )
    await harness.start()
    first_id = await harness.self_owner.submit_text("first", source="You")

    assert harness.output_runtime.overlay_sink is first
    assert [event.utterance_id for event in first.events] == [first_id, first_id]
    await harness.output_projection.replace_overlay_sink(second)
    second_id = await harness.self_owner.submit_text("second", source="You")
    second_event = harness.output_projection.overlay_event_adapter.utterance_closed(
        utterance_id=second_id,
        channel="self",
        is_final=True,
    )
    await harness.output_projection.publish_overlay_event(second_event)
    await harness.output_projection.publish_overlay_event(second_event)

    assert harness.output_runtime.overlay_sink is second
    assert len(first.events) == 2
    assert [event.utterance_id for event in second.events[:2]] == [second_id, second_id]
    assert second.events.count(second_event) == 1
    assert harness.output_runtime.routing_decisions[-1].reason == "duplicate_publication"

    await harness.stop()


@pytest.mark.asyncio
async def test_translation_fixture_overlay_replacement_awaits_old_delivery_before_new_routing() -> (
    None
):
    old = BlockingOverlay()
    replacement = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingChatbox(),
        overlay_sink=old,
    )
    old_event = harness.output_projection.overlay_event_adapter.utterance_closed(
        utterance_id=uuid4(),
        channel="peer",
        is_final=True,
    )

    await harness.start()
    old_publication = asyncio.create_task(
        harness.output_projection.publish_overlay_event(old_event)
    )
    await asyncio.wait_for(old.started.wait(), timeout=0.5)
    await harness.output_projection.replace_overlay_sink(replacement)
    await old_publication
    new_id = await harness.self_owner.submit_text("new destination", source="You")

    assert old.cancelled.is_set()
    assert not harness.output_runtime.has_active_overlay_deliveries
    assert harness.output_projection.overlay_sink is replacement
    assert harness.output_runtime.overlay_sink is replacement
    assert old.events == [old_event]
    assert [event.utterance_id for event in replacement.events] == [new_id, new_id]
    replacement_decision = next(
        decision
        for decision in harness.output_runtime.routing_decisions
        if decision.publication_id == old_event.event_id
    )
    assert replacement_decision.reason == "destination_replaced"

    await harness.stop()


@pytest.mark.asyncio
async def test_translation_fixture_restart_constructs_a_fresh_output_owner() -> None:
    first_chatbox = RecordingChatbox()
    first = compose_translation_test_harness(stt=None, llm=None, osc=first_chatbox)

    await first.start(auto_flush_osc=True)
    await first.stop()

    second_chatbox = RecordingChatbox()
    second = compose_translation_test_harness(stt=None, llm=None, osc=second_chatbox)
    await second.start(auto_flush_osc=True)
    publication_id = await second.self_owner.submit_text("restart output", source="You")

    assert first.output_runtime.state == "closed"
    assert first.output_runtime is not second.output_runtime
    assert [message.utterance_id for message in second_chatbox.messages] == [publication_id]

    await second.stop()

    assert second.output_runtime.state == "closed"
    assert not second.output_runtime.has_resources
