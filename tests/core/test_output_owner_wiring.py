from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventUnion
from puripuly_heart.domain.events import UIEventType
from puripuly_heart.domain.models import OSCMessage, Translation
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


class StalledBridgeConnection:
    def __init__(self) -> None:
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()
        self.sent_payloads: list[str] = []

    async def send(self, payload: str) -> None:
        self.send_started.set()
        await self.release_send.wait()
        self.sent_payloads.append(payload)

    async def close(self) -> None:
        return None


class SelfOnlyTranslationProvider:
    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        _ = (system_prompt, context, scene_participant_count)
        return Translation(
            utterance_id=utterance_id,
            text=f"translated {text}",
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            channel="self",
        )

    async def close(self) -> None:
        return None


class ControlledTranslationProvider:
    def __init__(self, response_text: str, *, blocked: bool = False) -> None:
        self.response_text = response_text
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.calls = 0

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        _ = (system_prompt, context, scene_participant_count)
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return Translation(
            utterance_id=utterance_id,
            text=self.response_text,
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            channel="self",
        )

    async def close(self) -> None:
        return None



class IndependentPassthroughProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.translation_is_independent = False

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        _ = (system_prompt, context, scene_participant_count)
        self.calls += 1
        translated = text.encode().decode()
        self.translation_is_independent = translated == text and translated is not text
        return Translation(
            utterance_id=utterance_id,
            text=translated,
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            channel="self",
        )

    async def close(self) -> None:
        return None


class FailingOverlay(RecordingOverlay):
    async def emit(self, event: OverlayEventUnion) -> None:
        self.events.append(event)
        raise RuntimeError("overlay unavailable")


@pytest.mark.asyncio
async def test_production_projection_counts_aliased_source_once_for_legal_large_parent() -> (
    None
):
    source_text = "s" * (400 * 1024)
    translation_text = "t" * (400 * 1024)
    provider = ControlledTranslationProvider(translation_text)
    chatbox = RecordingChatbox()
    overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=chatbox,
        overlay_sink=overlay,
    )

    await harness.start()
    parent_id = await harness.self_owner.submit_text(source_text, source="You")
    await harness.translation_turns.wait_for_parent(parent_id)

    assert provider.calls == 1
    assert len(chatbox.messages) == 1
    assert any(event.utterance_id == parent_id for event in overlay.events)
    assert len(chatbox.messages[0].text) == len(source_text) + len(translation_text) + 3
    assert chatbox.messages[0].text[:1] == "s"
    assert chatbox.messages[0].text[-2:] == "t)"
    assert harness.output_runtime.overlay_admission_snapshot()["reserved_bytes"] == 0
    await harness.stop()


@pytest.mark.asyncio
async def test_production_projection_rejects_independent_equal_payload_copies_above_bound() -> (
    None
):
    source_text = "x" * (600 * 1024)
    provider = IndependentPassthroughProvider()
    chatbox = RecordingChatbox()
    overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=chatbox,
        overlay_sink=overlay,
    )

    await harness.start()
    parent_id = await harness.self_owner.submit_text(source_text, source="You")
    await harness.translation_turns.wait_for_parent(parent_id)

    assert provider.calls == 1
    assert provider.translation_is_independent
    assert chatbox.messages == []
    assert all(event.type != "translation_final" for event in overlay.events)
    assert harness.output_runtime.overlay_admission_snapshot()["reserved_bytes"] == 0
    await harness.stop()


@pytest.mark.asyncio
async def test_production_projection_accepts_exact_one_mib_parent_and_close() -> None:
    metadata = {"ko", "en", "You", "translation_unavailable"}
    source_text = "x" * (
        (1024 * 1024) - sum(len(value.encode("utf-8")) for value in metadata)
    )
    chatbox = RecordingChatbox()
    overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=chatbox,
        overlay_sink=overlay,
    )

    await harness.start()
    parent_id = await harness.self_owner.submit_text(source_text, source="You")
    await harness.translation_turns.wait_for_parent(parent_id)

    assert len(chatbox.messages) == 1
    assert chatbox.messages[0].text == source_text
    assert [event.utterance_id for event in overlay.events] == [parent_id, parent_id]
    assert harness.output_runtime.overlay_admission_snapshot()["reserved_bytes"] == 0
    await harness.stop()


@pytest.mark.asyncio
async def test_caption_off_after_parent_admission_preserves_ui_chatbox_and_history() -> None:
    provider = ControlledTranslationProvider("translated after caption off", blocked=True)
    chatbox = RecordingChatbox()
    overlay = RecordingOverlay()
    replacement_overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=chatbox,
        overlay_sink=overlay,
    )

    await harness.start()
    parent_id = await harness.self_owner.submit_text("caption boundary", source="You")
    await provider.started.wait()
    overlay_count_before_caption_off = len(overlay.events)
    await harness.output_projection.replace_overlay_sink(None)
    await harness.output_projection.replace_overlay_sink(replacement_overlay)
    provider.release.set()
    await harness.translation_turns.wait_for_parent(parent_id)

    assert provider.calls == 1
    assert len(chatbox.messages) == 1
    assert chatbox.messages[0].text == "caption boundary (translated after caption off)"
    ui_events = [
        harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())
    ]
    assert UIEventType.TRANSLATION_DONE in {event.type for event in ui_events}
    assert {item.text for item in harness.self_runtime.translation_history} == {
        "caption boundary"
    }
    assert len(overlay.events) == overlay_count_before_caption_off
    assert replacement_overlay.events == []
    await harness.stop()




@pytest.mark.asyncio
async def test_talk_reset_preserves_inflight_manual_and_peer_state() -> None:
    provider = ControlledTranslationProvider("translated manual", blocked=True)
    chatbox = RecordingChatbox()
    overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=chatbox,
        overlay_sink=overlay,
    )

    await harness.start()
    harness.peer_runtime.remember_context("peer retained", timestamp=0)
    manual_parent = await harness.self_owner.submit_text("manual", source="You")
    await asyncio.wait_for(provider.started.wait(), timeout=1)

    await asyncio.wait_for(harness.self_owner.reset_provider_channel("self"), timeout=1)
    assert not harness.translation_turns.is_parent_closed(manual_parent)
    assert {item.text for item in harness.peer_runtime.translation_history} == {
        "peer retained"
    }

    provider.release.set()
    await asyncio.wait_for(
        harness.translation_turns.wait_for_parent(manual_parent), timeout=1
    )

    assert [message.text for message in chatbox.messages] == [
        "manual (translated manual)"
    ]
    assert {item.text for item in harness.self_runtime.translation_history} == {
        "manual"
    }
    assert any(event.utterance_id == manual_parent for event in overlay.events)
    assert harness.output_runtime.overlay_admission_snapshot()["reserved_bytes"] == 0
    await harness.stop()


@pytest.mark.asyncio
async def test_listen_reset_clears_peer_state_without_retiring_self_parent() -> None:
    provider = ControlledTranslationProvider("translated self", blocked=True)
    chatbox = RecordingChatbox()
    overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=chatbox,
        overlay_sink=overlay,
    )

    await harness.start()
    harness.peer_runtime.remember_context("peer stale", timestamp=0)
    self_parent = await harness.self_owner.submit_text("self retained", source="You")
    await asyncio.wait_for(provider.started.wait(), timeout=1)

    await asyncio.wait_for(harness.peer_owner.reset_provider_channel("peer"), timeout=1)
    assert harness.peer_runtime.translation_history == []
    assert not harness.translation_turns.is_parent_closed(self_parent)

    provider.release.set()
    await asyncio.wait_for(
        harness.translation_turns.wait_for_parent(self_parent), timeout=1
    )
    assert [message.text for message in chatbox.messages] == [
        "self retained (translated self)"
    ]
    assert any(event.utterance_id == self_parent for event in overlay.events)
    assert harness.output_runtime.overlay_admission_snapshot()["reserved_bytes"] == 0
    await harness.stop()


@pytest.mark.asyncio
async def test_overlay_failure_does_not_rerun_translation_or_block_other_destinations() -> None:
    provider = ControlledTranslationProvider("survives overlay failure")
    chatbox = RecordingChatbox()
    overlay = FailingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=chatbox,
        overlay_sink=overlay,
    )

    await harness.start()
    parent_id = await harness.self_owner.submit_text("failure isolation", source="You")
    await harness.translation_turns.wait_for_parent(parent_id)

    assert provider.calls == 1
    assert len(chatbox.messages) == 1
    assert chatbox.messages[0].text == "failure isolation (survives overlay failure)"
    ui_events = [
        harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())
    ]
    assert UIEventType.TRANSLATION_DONE in {event.type for event in ui_events}
    assert {item.text for item in harness.self_runtime.translation_history} == {
        "failure isolation"
    }
    assert harness.output_runtime.overlay_admission_snapshot()["reserved_bytes"] == 0
    await harness.stop()


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
async def test_twelve_turns_reach_presenter_and_bridge_once_in_order() -> None:
    bridge = OverlayBridge(session_token="acceptance-token")
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        bridge=bridge,
        visible_window_target_blocks=12,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingChatbox(),
        overlay_sink=presenter,
    )
    expected_texts = [f"turn-{index:02d}" for index in range(12)]

    await harness.start()
    try:
        for index, text in enumerate(expected_texts):
            if index % 2 == 0:
                await harness.self_owner.submit_text(text, source="You")
            else:
                await harness.handle_peer_transcript_final_for_test(text)

        snapshot = bridge.snapshot()
        rendered_texts = [
            next(text for text in (block.primary_text, block.secondary_text) if text)
            for block in snapshot.blocks
        ]

        assert rendered_texts == expected_texts
        assert len({block.id for block in snapshot.blocks}) == 12
        assert [block.appearance_seq for block in snapshot.blocks] == sorted(
            block.appearance_seq for block in snapshot.blocks
        )
        admitted_revisions = [
            receipt.scene_revision
            for receipt in bridge.delivery_receipts
            if receipt.outcome == "admitted"
        ]
        assert admitted_revisions == list(range(1, snapshot.revision + 1))
    finally:
        await harness.stop()
        await presenter.close()
        await bridge.stop()


@pytest.mark.asyncio
async def test_actual_owner_chain_completes_twelve_parents_while_bridge_socket_is_stalled() -> None:
    bridge = OverlayBridge(session_token="acceptance-token")
    connection = StalledBridgeConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        bridge=bridge,
        visible_window_target_blocks=12,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    chatbox = RecordingChatbox()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SelfOnlyTranslationProvider(),
        osc=chatbox,
        overlay_sink=presenter,
        peer_translation_enabled=False,
    )
    parent_ids = []

    await harness.start()
    try:
        for index in range(12):
            if index % 2 == 0:
                parent_ids.append(
                    await harness.self_owner.submit_text(f"manual-{index}", source="You")
                )
            else:
                parent_ids.append(
                    await harness.handle_peer_transcript_final_for_test(f"peer-{index}")
                )
        await connection.send_started.wait()
        await harness.translation_turns.wait_for_idle()

        snapshot = bridge.snapshot()
        assert len(set(parent_ids)) == 12
        assert len(snapshot.blocks) == 12
        assert len({block.id for block in snapshot.blocks}) == 12
        assert harness.ui_events.qsize() == 24
        assert len(chatbox.messages) == 6
        assert harness.output_runtime.overlay_admission_snapshot() == {
            "active": 0,
            "unsent": 0,
            "batches": 0,
            "reserved_bytes": 0,
            "scopes": {},
        }
        assert connection.sent_payloads == []
    finally:
        connection.release_send.set()
        await harness.stop()
        await presenter.close()
        await bridge.stop()


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
