from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.overlay.sink import OverlayEventUnion
from puripuly_heart.domain.models import OSCMessage


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


@pytest.mark.asyncio
async def test_client_hub_routes_manual_peer_and_system_output_through_one_owner() -> None:
    chatbox = RecordingChatbox()
    overlay = RecordingOverlay()
    hub = ClientHub(
        stt=None,
        llm=None,
        osc=chatbox,
        overlay_sink=overlay,
    )

    await hub.start()
    manual_id = await hub.submit_text("manual self text", source="You")
    peer_id = await hub.handle_peer_transcript_final_for_test("peer-only text")
    hub.enqueue_peer_translation_disclosure("Peer translation is on")

    assert hub.output_runtime.overlay_sink is overlay
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
        for decision in hub.output_runtime.routing_decisions
        if decision.route == "self_chatbox" and decision.publication_kind == "peer_subtitle"
    ]
    assert len(peer_chatbox_decisions) == 1
    assert peer_chatbox_decisions[0].reason == "peer_chatbox_denied"
    assert all(decision.reason != "duplicate_publication" for decision in peer_chatbox_decisions)

    await hub.stop()

    assert hub.output_runtime.state == "closed"
    assert not hub.output_runtime.has_resources


@pytest.mark.asyncio
async def test_client_hub_overlay_replacement_updates_only_owner_destination() -> None:
    first = RecordingOverlay()
    second = RecordingOverlay()
    hub = ClientHub(
        stt=None,
        llm=None,
        osc=RecordingChatbox(),
        overlay_sink=first,
    )
    await hub.start()
    first_id = await hub.submit_text("first", source="You")

    assert hub.output_runtime.overlay_sink is first
    assert [event.utterance_id for event in first.events] == [first_id, first_id]
    hub.overlay_sink = second
    second_id = await hub.submit_text("second", source="You")
    second_event = hub.overlay_event_adapter.utterance_closed(
        utterance_id=second_id,
        channel="self",
        is_final=True,
    )
    await hub._emit_overlay_event(second_event)
    await hub._emit_overlay_event(second_event)

    assert hub.output_runtime.overlay_sink is second
    assert len(first.events) == 2
    assert [event.utterance_id for event in second.events[:2]] == [second_id, second_id]
    assert second.events.count(second_event) == 1
    assert hub.output_runtime.routing_decisions[-1].reason == "duplicate_publication"

    await hub.stop()


@pytest.mark.asyncio
async def test_client_hub_restart_constructs_a_fresh_output_owner() -> None:
    first_chatbox = RecordingChatbox()
    first = ClientHub(stt=None, llm=None, osc=first_chatbox)

    await first.start(auto_flush_osc=True)
    await first.stop()

    second_chatbox = RecordingChatbox()
    second = ClientHub(stt=None, llm=None, osc=second_chatbox)
    await second.start(auto_flush_osc=True)
    publication_id = await second.submit_text("restart output", source="You")

    assert first.output_runtime.state == "closed"
    assert first.output_runtime is not second.output_runtime
    assert [message.utterance_id for message in second_chatbox.messages] == [publication_id]

    await second.stop()

    assert second.output_runtime.state == "closed"
    assert not second.output_runtime.has_resources
