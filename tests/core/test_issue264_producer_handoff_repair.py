from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
from puripuly_heart.core.stt.backend import (
    STTAudioBoundary,
    STTProviderAudioBoundary,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
)
from puripuly_heart.core.stt.batch_speaker import CausalBatchSession
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.core.test_issue264_shared_speaker import changed, frames
from tests.core.test_issue264_streaming_repair import recorded_short_evidence, timed_final
from tests.core.test_source_context_contract import (
    ControlledProvider,
    close,
    finish,
    harness_for,
    source_fixture,
)
from tests.core.test_streaming_speaker import Provider, next_text, send, span
from tests.core.test_stt_scoped_engine import ControlledScopedSession, wait_until


@pytest.mark.asyncio
async def test_streaming_native_terminal_in_producer_queue_keeps_its_future_part_during_prefix_admission():
    clock = FakeClock(10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    translator = ControlledProvider()
    harness, trace = harness_for(translator, overlay_sink=presenter)
    allocator = harness.peer_owner._speaker_identities
    view = allocator.evidence
    native = Provider()
    wrapper = StreamingSpeakerSession(native, evidence_view=view)
    snapshot, identity = source_fixture()
    request = STTProviderTurnRequest(identity, snapshot.settings)
    native_update = timed_final(
        request,
        "No. He washed",
        [(0, 3, 2.8799999, 3.28), (4, 6, 3.28, 3.52), (7, 13, 3.52, 3.76)],
        result_start=2.8799999,
        result_end=3.825,
    )
    native_evidence = recorded_short_evidence()
    admission_lock = asyncio.Lock()
    await admission_lock.acquire()
    harness.translation_turns._channel_admission_locks["peer"] = admission_lock
    engine = ScopedRecognitionEngine(None, evidence_view=view)
    events = wrapper.turn_events()

    async def deliver(event):
        if isinstance(event, STTProviderTurnUpdate):
            await harness.peer_owner.handle_provider_turn_update(snapshot, event)
        elif isinstance(event, STTProviderTurnTerminal):
            await harness.peer_owner.handle_provider_turn_terminal(
                AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 10), event
            )

    engine.bind_event_sink(deliver)
    try:
        await wrapper.begin_turn(request)
        await send(wrapper, request, 1728, 9728, 1, context=True)
        await send(wrapper, request, 9728, 84992, 2)
        native.emit(
            replace(native_update, source_release=replace(native_update.source_release, text_end=4))
        )
        prefix = await next_text(events)
        assert prefix.source_release.text_end == 4
        assert len(prefix.source_audio[0].speaker_parts) == 2
        future_token = prefix.source_audio[0].speaker_parts[1].speaker_proof_id
        assert future_token is not None
        await engine._emit(prefix)
        await wait_until(lambda: bool(harness.translation_turns._parents))
        native.emit(
            STTProviderTurnTerminal(
                identity,
                "final",
                native_update.text,
                text_authority="authoritative",
                provenance=(native_update.provenance,),
                included_contributions=(native_update.contribution,),
            )
        )
        await native.close()
        await asyncio.wait_for(asyncio.shield(wrapper._consumer), 1)
        queued = next(
            event for event in wrapper._events._events if isinstance(event, STTProviderTurnTerminal)
        )
        assert queued.source_audio[0].speaker_parts[1].speaker_proof_id == future_token
        admission_lock.release()
        await engine.wait_for_event_ingress_drain()
        await finish(harness)
        assert [item.source_child.raw_text for item in trace.items] == ["No. "]
        assert view.has_proof(
            future_token
        ), "annotated future part lost before producer queue drain"
        terminal = await anext(events)
        while not isinstance(terminal, STTProviderTurnTerminal):
            terminal = await anext(events)
        await engine._emit(terminal)
        await engine.wait_for_event_ingress_drain()
        await events.aclose()
        await finish(harness)
        assert [item.source_child.raw_text for item in trace.items] == ["No. ", "He washed"]
        assert [call[0] for call in translator.calls] == ["No. ", "He washed"]
        assert len(harness.peer_runtime.translation_history) == 2
        clock.advance(0.25)
        harness.peer_owner.clock.advance(0.25)
        view.observe(native_evidence)
        await changed(presenter, lambda: allocator.resolved_within_500ms == 2)
        sources = [item.source_child for item in trace.items]
        assert [
            allocator.source_snapshot(source.source_id).assignment.attribution.key.local_speaker_id
            for source in sources
        ] == ["1", "0"]
        assert not view._proofs
    finally:
        if admission_lock.locked():
            admission_lock.release()
        await events.aclose()
        await wrapper.close()
        await engine.close()
        await close(harness)
        await presenter.close()


async def native_terminal_in_producer(kind, view, *, pressure=False):
    native = Provider() if kind == "streaming" else ControlledScopedSession()
    wrapper = (
        StreamingSpeakerSession(native, evidence_view=view)
        if kind == "streaming"
        else CausalBatchSession(native, evidence_view=view)
    )
    snapshot, identity = source_fixture()
    request = STTProviderTurnRequest(identity, snapshot.settings)
    if pressure:
        wrapper._events = STTProviderEventBuffer(
            max_events=1, on_discard=wrapper._discard_buffered_event
        )
        wrapper._put_event(
            STTProviderAudioBoundary(identity, STTAudioBoundary("prefill", 7, 0, 10, "speaker"))
        )
    await wrapper.begin_turn(request)
    await send(wrapper, request, 0, 3200, 1)
    if kind == "streaming":
        update = timed_final(
            request, "Hello.", [(0, 6, 0.0, 0.1)], result_start=0.0, result_end=0.1
        )
        native.emit(
            STTProviderTurnTerminal(
                identity,
                "final",
                update.text,
                text_authority="authoritative",
                provenance=(update.provenance,),
                included_contributions=(update.contribution,),
            )
        )
    else:
        await wrapper.seal_turn(
            identity,
            sealed_content_ranges=(span(0, 3200),),
            seal_reason="silence",
            observed_trailing_silence_ms=600,
        )
        native.emit(
            STTProviderTurnTerminal(
                native.requests[0].identity, "final", "Hello.", text_authority="authoritative"
            )
        )
    await wait_until(
        lambda: any(isinstance(event, STTProviderTurnTerminal) for event in wrapper._events._events)
    )
    return wrapper, snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["streaming", "batch"])
async def test_closed_producer_queue_and_yield_transfer_pin_to_real_admission(kind):
    clock = FakeClock(10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    translator = ControlledProvider()
    harness, trace = harness_for(translator, overlay_sink=presenter)
    allocator = harness.peer_owner._speaker_identities
    view = allocator.evidence
    wrapper, snapshot = await native_terminal_in_producer(kind, view)
    queued = next(
        event for event in wrapper._events._events if isinstance(event, STTProviderTurnTerminal)
    )
    token = queued.source_audio[0].speaker_proof_id
    assert token is not None and view._proofs[token].handoffs == 1
    view.release_unadopted(token, published=True)
    assert view.has_proof(token)
    await wrapper.close()
    assert view.has_proof(token)
    events = wrapper.turn_events()
    normalizer = STTScopedTurnNormalizer(queued.identity)

    async def deliver(event):
        event = normalizer.apply_terminal(event)
        await harness.peer_owner.handle_provider_turn_terminal(
            AudioSegmentTerminalReceipt(event.identity.segment, "final", snapshot, 10), event
        )

    engine = ScopedRecognitionEngine(None, evidence_view=view)
    engine.bind_event_sink(deliver)
    try:
        event = await anext(events)
        assert event.text == "Hello." and view.has_proof(token)
        await engine._emit(event)
        await events.aclose()
        await engine.wait_for_event_ingress_drain()
        await finish(harness)
        assert [item.source_child.raw_text for item in trace.items] == ["Hello."]
        assert [call[0] for call in translator.calls] == ["Hello."]
        assert len(harness.peer_runtime.translation_history) == 1
        assert view.has_proof(token) and view._proofs[token].handoffs == 0
        assert not wrapper._speaker_handoffs and not engine._speaker_handoffs
        clock.advance(0.25)
        harness.peer_owner.clock.advance(0.25)
        view.observe(frames(0, 3200, 0))
        await changed(presenter, lambda: allocator.resolved_within_500ms == 1)
        assert not view._proofs
    finally:
        await events.aclose()
        await wrapper.close()
        await engine.close()
        await close(harness)
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["streaming", "batch"])
@pytest.mark.parametrize("adopted", [False, True])
async def test_producer_yield_cancel_after_close_releases_only_unused_claims(kind, adopted):
    view = SpeakerEvidenceView()
    wrapper, _ = await native_terminal_in_producer(kind, view)
    entered, blocked = asyncio.Event(), asyncio.Event()
    tokens = []

    async def consume():
        events = wrapper.turn_events()
        try:
            event = await anext(events)
            tokens.extend(event.source_audio[0].speaker_proof_ids)
            entered.set()
            await blocked.wait()
        finally:
            await events.aclose()

    consumer = asyncio.create_task(consume())
    try:
        await entered.wait()
        token = tokens[0]
        assert view._proofs[token].handoffs == 1
        view.release_unadopted(token, published=True)
        if adopted:
            view.adopt(token)
        await wrapper.close()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert not wrapper._speaker_handoffs
        assert view.has_proof(token) is adopted
        view.release(token)
        assert not view._proofs
    finally:
        if not consumer.done():
            consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await wrapper.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["streaming", "batch"])
async def test_native_producer_rejection_releases_request_pins_without_queue_policy_change(kind):
    view = SpeakerEvidenceView()
    wrapper, _ = await native_terminal_in_producer(kind, view, pressure=True)
    events = wrapper.turn_events()
    try:
        assert not view._proofs and not wrapper._speaker_handoffs
        event = await anext(events)
        assert isinstance(event, STTProviderTurnTerminal)
        assert (
            event.outcome == "failed" and event.failure_reason == "provider_event_buffer_overflow"
        )
        assert event.source_audio == ()
    finally:
        await wrapper.close()
        await events.aclose()
