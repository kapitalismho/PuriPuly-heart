from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from puripuly_heart.core.orchestrator.translation_output_projection import TranslationUiMessageQueue
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.fakes import RecordingOscQueue
from tests.ui.test_event_bridge import DummyApp, make_bridge


def transcript_event(text: str, *, source: str = "Mic", channel: str = "self") -> UIEvent:
    identity = uuid4()
    return UIEvent(
        UIEventType.TRANSCRIPT_FINAL,
        identity,
        Transcript(identity, text, is_final=True, channel=channel),
        source=source,
    )


@pytest.mark.asyncio
async def test_self_ui_stall_bounds_required_events_and_records_overload_and_close() -> None:
    destination = asyncio.Queue(maxsize=1)
    destination.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output)
    try:
        accepted = await owner.publish(transcript_event("first"))
        await asyncio.sleep(0)
        for index in range(32):
            result = await owner.publish(UIEvent(UIEventType.ERROR, payload=f"error-{index}"))
            assert result.decision.reason == "accepted_handoff"
        refused = await owner.publish(transcript_event("overload", source="You"))
        assert refused.decision.reason == "output_overload"
        assert refused.decision.metadata["ui_queue_submitted"] is False
        assert accepted.decision.metadata["accepted_handoff"] is True
        assert len(owner._self_events) == 32
        assert owner._active_self_event is not None
        assert destination.qsize() == 1
        await output.close()
        assert not owner.has_resources
        assert (
            len([d for d in output.routing_decisions if d.reason == "output_runtime_closing"]) == 33
        )
        late = await owner.publish(transcript_event("late"))
        assert late.decision.reason == "output_runtime_closing"
        assert destination.qsize() == 1
    finally:
        await output.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "exception"])
async def test_ui_write_failure_is_destination_local_and_observable(failure: str) -> None:
    class FailingQueue(asyncio.Queue):
        async def put(self, item):
            if failure == "exception":
                raise RuntimeError("destination failed")
            await super().put(item)

    destination = FailingQueue(maxsize=1)
    destination.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output, write_timeout_s=0.01)
    try:
        result = await owner.publish(UIEvent(UIEventType.ERROR, payload="required error"))
        assert result.decision.reason == "accepted_handoff"
        await asyncio.wait_for(owner.wait_for_idle(), 1.0)
        expected = (
            "destination_write_timeout" if failure == "timeout" else "destination_publish_failed"
        )
        receipt = output.routing_decisions[-1]
        assert receipt.reason == expected
        assert receipt.metadata["ui_queue_submitted"] is False
        assert not owner.has_resources
    finally:
        await output.close()


@pytest.mark.asyncio
async def test_shared_destination_authority_keeps_history_without_old_visible_overwrite() -> None:
    destination = asyncio.Queue()
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output)
    output.activate_peer_generation(1)
    app = DummyApp()
    handled_errors = []
    bridge = make_bridge(
        app,
        event_queue=destination,
        error_destination=SimpleNamespace(
            publish_error=lambda text, **_kwargs: handled_errors.append(text) is None,
        ),
    )
    try:
        await owner.publish(transcript_event("old self"))
        await owner.publish(
            UIEvent(
                UIEventType.TRANSLATION_DONE,
                payload=Translation(uuid4(), "old translation"),
                source="Mic",
            )
        )
        parent = uuid4()
        await owner.publish(
            transcript_event("peer", source="Peer", channel="peer"),
            parent_utterance_id=parent,
            publication_generation=1,
            source_order=1,
        )
        await owner.publish(transcript_event("manual", source="You"))
        await owner.wait_for_idle()
        events = [destination.get_nowait() for _ in range(4)]
        newest = next(event for event in events if event.source == "You")
        await bridge._handle_event(newest)
        for event in events:
            if event is not newest:
                await bridge._handle_event(event)
        await bridge._handle_event(newest)
        assert [call[0] for call in app.view_dashboard.display_calls] == ["manual"]
        assert app.view_dashboard.translation_calls == []
        assert sorted(entry[1] for entry in app.history) == [
            "manual",
            "old self",
            "old translation",
            "peer",
        ]
        error = UIEvent(UIEventType.ERROR, payload="old error")
        await owner.publish(error)
        await owner.publish(transcript_event("newer", source="You"))
        older_error, newer = destination.get_nowait(), destination.get_nowait()
        await bridge._handle_event(newer)
        await bridge._handle_event(older_error)
        assert app.view_dashboard.display_calls[-1][0] == "newer"
        assert handled_errors == ["old error"]
    finally:
        bridge.close()
        await output.close()


@pytest.mark.asyncio
async def test_provider_retirement_preserves_manual_and_replacement_rejects_late_callbacks() -> (
    None
):
    destination = asyncio.Queue(maxsize=1)
    destination.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output)
    app = DummyApp()
    bridge = make_bridge(app, event_queue=destination)
    try:
        await owner.publish(transcript_event("retired speech"))
        await asyncio.sleep(0)
        await owner.publish(transcript_event("manual", source="You"))
        owner.retire_self_speech()
        destination.get_nowait()
        await asyncio.wait_for(owner.wait_for_idle(), 1.0)
        manual = destination.get_nowait()
        await bridge._handle_event(manual)
        assert [call[0] for call in app.view_dashboard.display_calls] == ["manual"]
        assert any(d.reason == "publication_generation_retired" for d in output.routing_decisions)
        await owner.publish(transcript_event("submitted old"))
        old = destination.get_nowait()
        destination.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
        await owner.publish(transcript_event("blocked old"))
        await asyncio.sleep(0)
        replacement = asyncio.Queue(maxsize=1)
        await asyncio.wait_for(owner.replace_destination(replacement), 1.0)
        await bridge._handle_event(old)
        assert [call[0] for call in app.view_dashboard.display_calls] == ["manual"]
        await owner.publish(transcript_event("replacement"))
        current = replacement.get_nowait()
        await bridge._handle_event(current)
        assert app.view_dashboard.display_calls[-1][0] == "replacement"
        assert any(d.reason == "destination_replaced" for d in output.routing_decisions)
        await owner.publish(transcript_event("late close"))
        late = replacement.get_nowait()
        await output.close()
        await bridge._handle_event(late)
        assert app.view_dashboard.display_calls[-1][0] == "replacement"
        assert not owner.has_resources
    finally:
        bridge.close()
        await output.close()


@pytest.mark.asyncio
async def test_peer_retention_budget_is_outstanding_not_lifetime_and_reports_overflow() -> None:
    destination = asyncio.Queue(maxsize=1)
    destination.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output)
    output.activate_peer_generation(1)
    parent = uuid4()

    async def publish(index: int):
        if index == 0:
            event = transcript_event(f"run-{index}", source="Peer", channel="peer")
        elif index == 31:
            event = UIEvent(
                UIEventType.ERROR,
                uuid4(),
                payload="run error",
                source="Peer",
                channel="peer",
            )
        else:
            child_id = uuid4()
            event = UIEvent(
                UIEventType.TRANSLATION_DONE,
                child_id,
                payload=Translation(child_id, f"run-{index}", channel="peer"),
                source="Peer",
            )
        return await owner.publish(
            event,
            parent_utterance_id=parent,
            publication_generation=1,
            source_order=1,
        )

    try:
        for index in range(32):
            result = await publish(index)
            assert result.decision.reason == "accepted_handoff"
        await asyncio.sleep(0)
        refused = await publish(32)
        assert refused.decision.reason == "output_overload"
        assert refused.decision.metadata["ui_queue_submitted"] is False
        assert len(owner._active_peer_batch.events) + int(owner._active_peer_batch.writing) == 32
        destination.get_nowait()
        for _ in range(32):
            await asyncio.wait_for(destination.get(), 1.0)
        await owner.wait_for_idle()
        for index in range(33, 73):
            result = await publish(index)
            assert result.decision.reason == "accepted_handoff"
            event = await asyncio.wait_for(destination.get(), 1.0)
            assert event.payload.text == f"run-{index}"
            await owner.wait_for_idle()
        assert not owner._in_flight_keys
        assert not owner.has_resources
        assert len(owner._completed_keys) == 72
    finally:
        await output.close()


@pytest.mark.asyncio
async def test_replacement_joins_unstarted_writers_for_both_lanes() -> None:
    destination = asyncio.Queue(maxsize=1)
    destination.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output)
    output.activate_peer_generation(1)
    try:
        await owner.publish(transcript_event("self"))
        await owner.publish(
            transcript_event("peer", source="Peer", channel="peer"),
            parent_utterance_id=uuid4(),
            publication_generation=1,
            source_order=1,
        )
        replacement = asyncio.Queue(maxsize=1)
        await asyncio.wait_for(owner.replace_destination(replacement), 1.0)
        assert not owner.has_resources
        assert not owner._in_flight_keys
        assert len([d for d in output.routing_decisions if d.reason == "destination_replaced"]) == 2
        await owner.publish(transcript_event("current", source="You"))
        assert replacement.get_nowait().payload.text == "current"
    finally:
        await output.close()


@pytest.mark.asyncio
async def test_output_generation_and_bridge_replacement_reject_submitted_late_callbacks() -> None:
    destination = asyncio.Queue(maxsize=1)
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output)
    app = DummyApp()
    first = make_bridge(app, event_queue=destination)
    second = make_bridge(app, event_queue=destination)
    try:
        task = output.start_ui_event_bridge(first)
        await output.wait_for_ui_event_bridge_started()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await owner.publish(transcript_event("retired configuration"))
        old_generation = destination.get_nowait()
        destination.task_done()
        output.retire_turn_generation("self", 1)
        await first._handle_event(old_generation)
        assert app.history == []
        await owner.publish(transcript_event("old destination", source="You"))
        old_destination = destination.get_nowait()
        destination.task_done()
        output.start_ui_event_bridge(second)
        await output.wait_for_ui_event_bridge_started()
        await first._handle_event(old_destination)
        await second._handle_event(old_destination)
        assert app.history == []
        await owner.publish(transcript_event("current", source="You"))
        await asyncio.wait_for(destination.join(), 1.0)
        assert [entry[1] for entry in app.history] == ["current"]
    finally:
        first.close()
        second.close()
        await output.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("previous_bridge", ["completed", "failed"])
async def test_production_bridge_replacement_recovers_unstarted_self_writer(
    previous_bridge: str,
) -> None:
    destination = asyncio.Queue(maxsize=1)
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output)
    app = DummyApp()
    first = make_bridge(app, event_queue=destination)
    second = make_bridge(app, event_queue=destination)
    if previous_bridge == "completed":
        first.close()
    else:

        async def fail_bridge() -> None:
            raise RuntimeError("previous bridge failed")

        first.run = fail_bridge
    try:
        previous = output.start_ui_event_bridge(first)
        await asyncio.gather(previous, return_exceptions=True)
        assert previous.done()
        destination.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
        retired = await owner.publish(transcript_event("retired speech"))
        assert retired.decision.reason == "accepted_handoff"
        cancelled_writer = owner._self_worker
        assert cancelled_writer is not None
        output.start_ui_event_bridge(second)
        manual = await owner.publish(transcript_event("current manual", source="You"))
        speech = await owner.publish(transcript_event("current speech"))
        assert manual.decision.reason == speech.decision.reason == "accepted_handoff"

        async def wait_for_consumption() -> None:
            while len(app.history) != 2 or owner.has_resources:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_consumption(), 1.0)
        assert cancelled_writer.cancelled()
        assert [entry[1] for entry in app.history] == ["current manual", "current speech"]
        assert [call[0] for call in app.view_dashboard.display_calls] == [
            "current manual",
            "current speech",
        ]
        assert len([d for d in output.routing_decisions if d.reason == "destination_replaced"]) == 1
        assert len([d for d in output.routing_decisions if d.reason == "ui_queue_submitted"]) == 2
        assert owner._self_worker is None
        assert not owner._self_events
    finally:
        first.close()
        second.close()
        if previous_bridge == "failed":
            with pytest.raises(RuntimeError, match="previous bridge failed"):
                await output.close()
        else:
            await output.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("previous_bridge", ["completed", "failed"])
@pytest.mark.parametrize("writer_started", [False, True])
async def test_production_bridge_replacement_recovers_peer_writer(
    previous_bridge: str, writer_started: bool
) -> None:
    destination = asyncio.Queue(maxsize=1)
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output)
    output.activate_peer_generation(1)
    app = DummyApp()
    first = make_bridge(app, event_queue=destination)
    second = make_bridge(app, event_queue=destination)
    if previous_bridge == "completed":
        first.close()
    else:

        async def fail_bridge() -> None:
            raise RuntimeError("previous bridge failed")

        first.run = fail_bridge
    try:
        await asyncio.gather(output.start_ui_event_bridge(first), return_exceptions=True)
        destination.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
        await owner.publish(
            transcript_event("retired peer", source="Peer", channel="peer"),
            parent_utterance_id=uuid4(),
            publication_generation=1,
            source_order=1,
        )
        if writer_started:
            await asyncio.sleep(0)
        output.start_ui_event_bridge(second)
        transcript = transcript_event("current peer", source="Peer", channel="peer")
        await owner.publish(
            transcript,
            parent_utterance_id=transcript.utterance_id,
            publication_generation=1,
            source_order=2,
        )
        child_id = uuid4()
        await owner.publish(
            UIEvent(
                UIEventType.TRANSLATION_DONE,
                child_id,
                Translation(child_id, "current translation", channel="peer"),
                source="Peer",
            ),
            parent_utterance_id=transcript.utterance_id,
            publication_generation=1,
            source_order=2,
        )

        async def wait_for_consumption() -> None:
            while len(app.history) != 2 or owner.has_resources:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_consumption(), 1.0)
        assert [(entry[1], entry[2]) for entry in app.history] == [
            ("current peer", False),
            ("current translation", True),
        ]
        assert [call[0] for call in app.view_dashboard.display_calls] == ["current peer"]
        assert [call[0] for call in app.view_dashboard.translation_calls] == ["current translation"]
        assert [d.reason for d in output.routing_decisions if d.metadata["source_order"] == 1] == [
            "accepted_handoff",
            "destination_replaced",
        ]
        assert not owner._in_flight_keys
    finally:
        first.close()
        second.close()
        if previous_bridge == "failed":
            with pytest.raises(RuntimeError, match="previous bridge failed"):
                await output.close()
        else:
            await output.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_only", [False, True])
async def test_ui_idle_wait_drains_peer_publication_after_prestart_cancellation(
    peer_only: bool,
) -> None:
    destination = asyncio.Queue()
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output)
    output.activate_peer_generation(1)
    try:
        await owner.publish(
            transcript_event("retired peer", source="Peer", channel="peer"),
            parent_utterance_id=uuid4(),
            publication_generation=1,
            source_order=1,
        )
        owner.retire_destination()
        await owner.publish(
            transcript_event("current peer", source="Peer", channel="peer"),
            parent_utterance_id=uuid4(),
            publication_generation=1,
            source_order=2,
        )
        idle = owner.wait_for_peer_idle() if peer_only else owner.wait_for_idle()
        await asyncio.wait_for(idle, 1.0)
        assert destination.get_nowait().payload.text == "current peer"
        assert destination.empty()
        assert not owner.has_resources
        assert not owner._in_flight_keys
    finally:
        await output.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("writer_started", [False, True])
async def test_bridge_replacement_close_does_not_restart_peer_delivery(
    writer_started: bool,
) -> None:
    destination = asyncio.Queue(maxsize=1)
    output = OutputRuntime(chatbox=RecordingOscQueue())
    owner = TranslationUiMessageQueue(destination, output)
    output.activate_peer_generation(1)
    app = DummyApp()
    first = make_bridge(app, event_queue=destination)
    second = make_bridge(app, event_queue=destination)
    first.close()
    try:
        await output.start_ui_event_bridge(first)
        destination.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
        await owner.publish(
            transcript_event("retired peer", source="Peer", channel="peer"),
            parent_utterance_id=uuid4(),
            publication_generation=1,
            source_order=1,
        )
        if writer_started:
            await asyncio.sleep(0)
        output.start_ui_event_bridge(second)
        await owner.publish(
            transcript_event("current peer", source="Peer", channel="peer"),
            parent_utterance_id=uuid4(),
            publication_generation=1,
            source_order=2,
        )
        await output.close()
        await asyncio.sleep(0)
        assert app.history == []
        assert app.view_dashboard.display_calls == []
        assert not owner.has_resources
        assert not owner._in_flight_keys
        assert [d.reason for d in output.routing_decisions if d.metadata["source_order"] == 2] == [
            "accepted_handoff",
            "output_runtime_closing",
        ]
    finally:
        first.close()
        second.close()
        await output.close()
