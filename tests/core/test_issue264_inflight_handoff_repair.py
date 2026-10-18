from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
from puripuly_heart.core.stt.backend import (
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSourceAudio,
    STTSpeakerAudioPart,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.domain.models import FinalSpeakerRun, SpeakerAttribution
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.core.test_issue264_shared_speaker import changed, frames
from tests.core.test_source_context_contract import (
    ControlledProvider,
    close,
    finish,
    harness_for,
    source_fixture,
)
from tests.core.test_streaming_speaker import span, update
from tests.core.test_stt_scoped_engine import wait_until


def handoff_event(view, *, order=1):
    snapshot, identity = source_fixture(order)
    request = STTProviderTurnRequest(identity, snapshot.settings)
    parent, first, second = (
        view.pin((span(start, end),)) for start, end in ((0, 3200), (0, 1600), (1600, 3200))
    )
    unknown = SpeakerAttribution("unavailable")
    audio = STTSourceAudio(
        "phrase-1",
        (span(0, 3200),),
        speaker_proof_id=parent,
        speaker_parts=(
            STTSpeakerAudioPart(0, 4, (span(0, 1600),), (), unknown, first),
            STTSpeakerAudioPart(4, 8, (span(1600, 3200),), (), unknown, second),
        ),
    )
    event = replace(
        update(request, "One two.", release=True),
        source_audio=(audio,),
        final_speaker_runs=(
            FinalSpeakerRun(
                "One two.", None, "", source="nemotron", attribution_state="unavailable"
            ),
        ),
    )
    return snapshot, event, (parent, first, second)


@pytest.mark.asyncio
@pytest.mark.parametrize("already_adopted", [False, True])
async def test_admission_lock_handoff_survives_rejected_extension_and_resolves_in_place(
    already_adopted,
):
    clock = FakeClock(10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    translator = ControlledProvider()
    harness, trace = harness_for(translator, overlay_sink=presenter)
    view = harness.peer_owner._speaker_identities.evidence
    snapshot, full, tokens = handoff_event(view)
    prefix = replace(full, source_release=replace(full.source_release, text_end=4))
    prefix = STTScopedTurnNormalizer(full.identity).apply_update(prefix)
    if already_adopted:
        view.adopt(tokens[1])
    admission_lock = asyncio.Lock()
    await admission_lock.acquire()
    harness.translation_turns._channel_admission_locks["peer"] = admission_lock
    engine = ScopedRecognitionEngine(None, evidence_view=view)
    engine._event_buffer = STTProviderEventBuffer(
        max_events=1, on_discard=engine._discard_buffered_event
    )

    async def deliver(event):
        if isinstance(event, STTProviderTurnUpdate):
            await harness.peer_owner.handle_provider_turn_update(snapshot, event)

    engine.bind_event_sink(deliver)
    try:
        await engine._emit(prefix)
        await wait_until(lambda: bool(harness.translation_turns._parents))
        assert engine._event_dispatching and engine._event_buffer.depth == 0
        assert all(view.has_proof(token) for token in tokens)
        assert not trace.items and not translator.calls
        _, padding_identity = source_fixture(2)
        await engine._emit(STTProviderTurnTerminal(padding_identity, "final", "padding"))
        await engine._emit(replace(full, sequence=2))
        assert all(view.has_proof(token) for token in tokens)
        admission_lock.release()
        await engine.wait_for_event_ingress_drain()
        await finish(harness)
        assert [item.source_child.raw_text for item in trace.items] == ["One "]
        assert [call[0] for call in translator.calls] == ["One "]
        assert len(harness.peer_runtime.translation_history) == 1
        assert not engine._speaker_handoffs
        assert [view.has_proof(token) for token in tokens] == [False, True, False]
        first = presenter.snapshot().blocks[0]
        assert first.speaker_style == "gray"
        entry = next(iter(presenter._entries.values()))
        expiration = presenter._entry_expiration_deadline(entry)
        clock.advance(0.25)
        harness.peer_owner.clock.advance(0.25)
        view.observe(frames(0, 1600, 0))
        await changed(presenter, lambda: presenter.snapshot().blocks[0].speaker_style == "gold")
        final = presenter.snapshot().blocks[0]
        assert len(presenter.snapshot().blocks) == 1
        assert (final.id, final.occupant_key, final.appearance_seq) == (
            first.id,
            first.occupant_key,
            first.appearance_seq,
        )
        assert (final.primary_text, final.secondary_text) == (
            first.primary_text,
            first.secondary_text,
        )
        assert presenter._entry_expiration_deadline(entry) == expiration
        assert final.card_revision > first.card_revision
        assert harness.peer_owner._speaker_identities.resolved_within_500ms == 1
        assert not view._proofs
    finally:
        if admission_lock.locked():
            admission_lock.release()
        await engine.close()
        await close(harness)
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("same_event", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
@pytest.mark.parametrize("adopted", [False, True])
async def test_concurrent_direct_handoff_alias_outlives_other_delivery(
    outcome, same_event, adopted
):
    view = SpeakerEvidenceView()
    _, event, tokens = handoff_event(view)
    gates = (asyncio.Event(), asyncio.Event())
    calls = []

    async def sink(incoming):
        index = len(calls)
        calls.append(incoming)
        await gates[index].wait()
        if index == 0 and outcome == "failure":
            raise RuntimeError("controlled delivery failure")

    engine = ScopedRecognitionEngine(None, evidence_view=view, event_sink=sink)
    tasks = []
    try:
        tasks.append(asyncio.create_task(engine._emit(event)))
        await wait_until(lambda: len(calls) == 1)
        tasks.append(
            asyncio.create_task(engine._emit(event if same_event else replace(event, sequence=2)))
        )
        await wait_until(lambda: len(calls) == 2)
        if adopted:
            view.adopt(tokens[1])
        _, rejected, unused = handoff_event(view, order=2)
        engine._discard_buffered_event(rejected, ())
        assert not any(view.has_proof(token) for token in unused)
        if outcome == "cancel":
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
        else:
            gates[0].set()
            if outcome == "failure":
                with pytest.raises(RuntimeError, match="controlled delivery failure"):
                    await tasks[0]
            else:
                await tasks[0]
        assert all(view.has_proof(token) for token in tokens)
        gates[1].set()
        await tasks[1]
        assert not engine._speaker_handoffs
        assert [view.has_proof(token) for token in tokens] == [False, adopted, False]
        view.release(tokens[1])
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await engine.close()
        assert not engine._speaker_handoffs and not view._proofs


@pytest.mark.asyncio
@pytest.mark.parametrize("same_event", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_queued_alias_survives_active_dispatch_completion(outcome, same_event):
    view = SpeakerEvidenceView()
    _, event, tokens = handoff_event(view)
    entered, release = asyncio.Event(), asyncio.Event()

    async def sink(incoming):
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise RuntimeError("controlled delivery failure")

    engine = ScopedRecognitionEngine(None, evidence_view=view)
    engine.bind_event_sink(sink)
    dispatch = engine._event_dispatch_task
    try:
        await engine._emit(event)
        await entered.wait()
        await engine._emit(event if same_event else replace(event, sequence=2))
        if outcome == "cancel":
            dispatch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await dispatch
        elif outcome == "failure":
            release.set()
            with pytest.raises(RuntimeError, match="controlled delivery failure"):
                await dispatch
        else:
            second_entered = asyncio.Event()
            second_release = asyncio.Event()

            async def second_sink(incoming):
                second_entered.set()
                await second_release.wait()

            engine._deferred_event_sink = second_sink
            release.set()
            await second_entered.wait()
            assert all(view.has_proof(token) for token in tokens)
            second_release.set()
            await engine.wait_for_event_ingress_drain()
        if outcome != "success":
            assert all(view.has_proof(token) for token in tokens)
            engine.bind_event_sink(lambda incoming: None)
            await engine.wait_for_event_ingress_drain()
        assert not engine._speaker_handoffs and not view._proofs
    finally:
        release.set()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("adopted", [False, True])
async def test_close_reclaims_abandoned_queued_handoffs_without_releasing_active_source(adopted):
    view = SpeakerEvidenceView()
    _, event, tokens = handoff_event(view)
    _, queued, queued_tokens = handoff_event(view, order=2)
    entered = asyncio.Event()

    async def sink(incoming):
        entered.set()
        await asyncio.Event().wait()

    engine = ScopedRecognitionEngine(None, evidence_view=view, event_drain_timeout_s=0.01)
    engine.bind_event_sink(sink)
    dispatch = engine._event_dispatch_task
    try:
        await engine._emit(event)
        await entered.wait()
        await engine._emit(queued)
        if adopted:
            view.adopt(tokens[1])
        await engine.close()
        await asyncio.gather(dispatch, return_exceptions=True)
        assert not engine._speaker_handoffs
        assert not any(view.has_proof(token) for token in queued_tokens)
        assert [view.has_proof(token) for token in tokens] == [False, adopted, False]
        view.release(tokens[1])
    finally:
        await engine.close()
        assert not view._proofs
