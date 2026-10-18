from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal, STTProviderTurnUpdate
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.core.test_issue264_inflight_handoff_repair import handoff_event
from tests.core.test_issue264_shared_speaker import changed, frames
from tests.core.test_source_context_contract import (
    ControlledProvider,
    close,
    finish,
    harness_for,
    source_fixture,
)
from tests.core.test_streaming_speaker import span
from tests.core.test_stt_scoped_engine import wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted_extension", [False, True])
async def test_real_peer_cleanup_preserves_only_live_accepted_source_opportunities(
    accepted_extension,
):
    clock = FakeClock(10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    translator = ControlledProvider()
    harness, trace = harness_for(translator, overlay_sink=presenter)
    allocator = harness.peer_owner._speaker_identities
    view = allocator.evidence
    snapshot, full, tokens = handoff_event(view)
    normalizer = STTScopedTurnNormalizer(full.identity)
    prefix = normalizer.apply_update(
        replace(full, source_release=replace(full.source_release, text_end=4))
    )
    extension = normalizer.apply_update(
        replace(
            full,
            sequence=2,
            assembly="replace",
            source_release=replace(full.source_release, release_id="full"),
            provenance=replace(full.provenance, native_event_id="second"),
        )
    )
    admission_lock = asyncio.Lock()
    await admission_lock.acquire()
    harness.translation_turns._channel_admission_locks["peer"] = admission_lock
    engine = ScopedRecognitionEngine(None, evidence_view=view)
    if not accepted_extension:
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
        assert all(view._proofs[token].handoffs == 1 for token in tokens)
        if not accepted_extension:
            _, padding_identity = source_fixture(2)
            await engine._emit(STTProviderTurnTerminal(padding_identity, "final", "padding"))
        await engine._emit(extension)
        expected_holds = 2 if accepted_extension else 1
        assert all(view._proofs[token].handoffs == expected_holds for token in tokens)
        admission_lock.release()
        await engine.wait_for_event_ingress_drain()
        await finish(harness)
        sources = [item.source_child for item in trace.items]
        expected_text = ["One ", "two."] if accepted_extension else ["One "]
        assert [source.raw_text for source in sources] == expected_text
        assert [call[0] for call in translator.calls] == expected_text
        assert len(harness.peer_runtime.translation_history) == len(expected_text)
        assert not engine._speaker_handoffs
        assert [view.has_proof(token) for token in tokens] == [False, True, accepted_extension]
        assert all(proof.handoffs == 0 for proof in view._proofs.values())
        first = presenter.snapshot().blocks
        assert first and all(block.speaker_style == "gray" for block in first)
        first_texts = [(block.primary_text, block.secondary_text) for block in first]
        first_positions = [(block.id, block.occupant_key, block.appearance_seq) for block in first]
        deadlines = {
            key: presenter._entry_expiration_deadline(entry)
            for key, entry in presenter._entries.items()
        }
        clock.advance(0.25)
        harness.peer_owner.clock.advance(0.25)
        view.observe(frames(0, 1600, 0) + frames(1600, 3200, 1))
        expected_keys = ["0", "1"] if accepted_extension else ["0"]

        def assigned_keys():
            keys = (
                allocator.source_snapshot(source.source_id).assignment.attribution.key
                for source in sources
            )
            return [None if key is None else key.local_speaker_id for key in keys]

        await changed(presenter, lambda: allocator.resolved_within_500ms == len(sources))
        assert assigned_keys() == expected_keys
        final = presenter.snapshot().blocks
        assert len(first) == len(final) == len(sources)
        assert all(block.speaker_style != "gray" for block in final)
        assert len({block.speaker_style for block in final}) == len(sources)
        assert [(block.primary_text, block.secondary_text) for block in final] == first_texts
        assert [
            (block.id, block.occupant_key, block.appearance_seq) for block in final
        ] == first_positions
        assert {
            key: presenter._entry_expiration_deadline(entry)
            for key, entry in presenter._entries.items()
        } == deadlines
        assert not view._proofs
    finally:
        if admission_lock.locked():
            admission_lock.release()
        await engine.close()
        await close(harness)
        await presenter.close()


@pytest.mark.parametrize("stage", ["request", "contribution"])
@pytest.mark.parametrize("cleanup", ["release", "unadopted", "published_unadopted"])
def test_shared_pin_owner_defers_ordinary_cleanup_until_last_live_dependency(stage, cleanup):
    view = SpeakerEvidenceView()
    token = view.pin((span(0, 1600),))
    if stage == "contribution":
        view.publish(token)
    view.retain_handoff(token)
    view.retain_handoff(token)
    if cleanup == "release":
        view.release(token)
    else:
        view.release_unadopted(token, published=cleanup == "published_unadopted")
    eligible = cleanup != "unadopted" or stage == "request"
    assert view.has_proof(token)
    assert view._proofs[token].release_pending is eligible
    view.release_handoff(token)
    assert view.has_proof(token) and view._proofs[token].handoffs == 1
    view.release_handoff(token)
    assert view.has_proof(token) is not eligible
    view.release(token)
    assert not view._proofs


def test_adoption_cancels_pending_unadopted_cleanup_but_explicit_source_retirement_remains_final():
    view = SpeakerEvidenceView()
    token = view.pin((span(0, 1600),))
    view.retain_handoff(token)
    view.publish(token)
    view.release_unadopted(token, published=True)
    view.adopt(token)
    assert not view._proofs[token].release_pending
    view.release_handoff(token)
    assert view.has_proof(token) and view._proofs[token].stage == "source"
    view.retain_handoff(token)
    view.release(token)
    assert not view.has_proof(token)
    view.release_handoff(token)
    assert not view._proofs


@pytest.mark.parametrize("invalidation", ["reset", "release_claims", "epoch", "pressure"])
def test_intentional_invalidation_drops_held_claim_without_resurrection(invalidation):
    view = SpeakerEvidenceView()
    view.observe(frames(0, 160, 0))
    token = view.pin((span(0, 160),))
    view.retain_handoff(token)
    view.release(token)
    assert view.has_proof(token)
    if invalidation == "epoch":
        view.observe(
            tuple(
                replace(item, speaker_epoch=item.speaker_epoch + 1) for item in frames(160, 320, 0)
            )
        )
    elif invalidation == "pressure":
        view.MAX_FRAMES = 2
        view.observe(frames(160, 640, 0))
    else:
        getattr(view, invalidation)()
    assert not view.has_proof(token)
    replacement = view.pin((span(0, 160),))
    assert replacement != token
    view.release_handoff(token)
    assert view.has_proof(replacement)
    view.release(replacement)
    assert not view._proofs


def test_handoff_holds_share_existing_pin_pressure_bound_and_reclaim_on_last_release():
    view = SpeakerEvidenceView()
    tokens = [view.pin((span(index, index + 1),)) for index in range(view.MAX_PROOFS)]
    for token in tokens:
        view.retain_handoff(token)
        view.release_unadopted(token)
    assert len(view._proofs) == view.MAX_PROOFS
    assert view.pin((span(200, 201),)) is None
    for token in tokens:
        view.release_handoff(token)
    assert not view._proofs
    token = view.pin((span(200, 201),))
    assert token is not None
    view.release(token)


@pytest.mark.asyncio
async def test_reset_during_queued_and_active_handoffs_cannot_recreate_or_release_new_claims():
    view = SpeakerEvidenceView()
    _, event, tokens = handoff_event(view)
    entered, release = asyncio.Event(), asyncio.Event()

    async def sink(incoming):
        entered.set()
        await release.wait()

    engine = ScopedRecognitionEngine(None, evidence_view=view)
    engine.bind_event_sink(sink)
    try:
        await engine._emit(event)
        await entered.wait()
        await engine._emit(replace(event, sequence=2))
        assert all(view._proofs[token].handoffs == 2 for token in tokens)
        view.reset()
        replacement = view.pin((span(0, 1600),))
        release.set()
        await engine.wait_for_event_ingress_drain()
        assert not engine._speaker_handoffs
        assert not any(view.has_proof(token) for token in tokens)
        assert view.has_proof(replacement)
        assert view._proofs[replacement].handoffs == 0
        view.release(replacement)
        assert not view._proofs
    finally:
        release.set()
        await engine.close()
