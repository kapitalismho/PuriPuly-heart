from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
from puripuly_heart.core.stt.backend import (
    STTAudioBoundary,
    STTProviderAudioBoundary,
    STTProviderTurnTerminal,
    STTSourceAudio,
    STTSourceRelease,
    STTSpeakerAudioPart,
    discard_source_audio_proofs,
)
from puripuly_heart.core.stt.batch_speaker import CausalBatchSession
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from puripuly_heart.domain.models import SpeakerAttribution
from puripuly_heart.providers.stt.deepgram import _DeepgramSDKSession
from tests.core.test_issue264_shared_speaker import frames
from tests.core.test_issue264_streaming_repair import (
    probabilities,
    recorded_short_evidence,
    timed_final,
)
from tests.core.test_source_context_contract import (
    ControlledProvider,
    close,
    finish,
    harness_for,
    source_fixture,
)
from tests.core.test_streaming_speaker import Provider, next_text, request, send, span, update
from tests.core.test_stt_scoped_engine import ControlledScopedSession, wait_until
from tests.helpers.scoped_provider_sessions import _deepgram_result, _deepgram_session


def capture_bounds(ranges):
    return [(item.normalized_start_sample, item.normalized_end_sample) for item in ranges]


def native_words_result():
    from deepgram.extensions.types.sockets.listen_v1_results_event import ListenV1Word

    result = _deepgram_result("One two.", duration=0.6000003)
    words = [
        ListenV1Word(word=text, punctuated_word=text, start=start, end=end, confidence=1.0)
        for text, start, end in (("One", 0.1, 0.2), ("two.", 0.4, 0.5))
    ]
    return result.model_copy(
        update={
            "channel": result.channel.model_copy(
                update={
                    "alternatives": [
                        result.channel.alternatives[0].model_copy(update={"words": words})
                    ]
                }
            )
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule", ["processed", "queued", "inflight"])
async def test_installed_adapter_authoritative_frontier_precedes_result_scheduling(
    schedule, monkeypatch
):
    req = request()
    native = _deepgram_session(epoch=req.identity.provider_epoch_id)
    entered, release = asyncio.Event(), asyncio.Event()
    writes = 0

    async def write(_session, _payload):
        nonlocal writes
        writes += 1
        if writes == 2 and schedule == "inflight":
            entered.set()
            await release.wait()

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    wrapper = StreamingSpeakerSession(native)
    publications = []
    req = replace(
        req,
        on_native_event=lambda event: publications.append(
            (type(event).__name__, wrapper.sent_until, native._event_projection.scoped_event_depth)
        ),
    )
    pending = None
    try:
        await wrapper.begin_turn(req)
        await send(wrapper, req, 0, 9600, 1)
        if schedule == "inflight":
            pending = asyncio.create_task(send(wrapper, req, 9600, 11200, 2))
            await asyncio.wait_for(entered.wait(), 1)
        native._build_transcript_event(native_words_result())
        assert publications == [("STTProviderTurnUpdate", 9600, 0)]
        turn = wrapper._turns[req.identity]
        assert {fence.sent_until for fence in turn.timing_frontiers.values()} == {9600}
        if schedule == "processed":
            output = await next_text(wrapper.turn_events())
            await send(wrapper, req, 9600, 11200, 2)
        else:
            if pending is not None:
                release.set()
                await pending
            else:
                await send(wrapper, req, 9600, 11200, 2)
            output = await next_text(wrapper.turn_events())
        assert wrapper.sent_until == 11200
        assert output.text == "One two."
        assert capture_bounds(output.source_audio[0].source_ranges) == [(1600, 8000)]
        assert output.source_audio[0].speaker_proof_id is not None
        assert {fence.sent_until for fence in turn.timing_frontiers.values()} == {9600}
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await wrapper.close()


@pytest.mark.asyncio
async def test_unreleased_candidate_keeps_progress_without_a_publication_fence():
    native, req = Provider(), request()
    wrapper = StreamingSpeakerSession(native)
    candidate = replace(
        timed_final(
            req,
            "One two.",
            [(0, 3, 0.1, 0.2), (4, 8, 0.4, 0.5)],
            result_start=0,
            result_end=0.6000003,
        ),
        source_release=None,
    )
    try:
        await wrapper.begin_turn(req)
        await send(wrapper, req, 0, 9600, 1)
        native.emit(candidate)
        first = await next_text(wrapper.turn_events())
        assert first.source_audio[0].speaker_proof_id is not None
        assert capture_bounds(first.source_audio[0].source_ranges) == [(1600, 8000)]
        assert wrapper._turns[req.identity].timing_frontiers == {}
        await send(wrapper, req, 9600, 11200, 2)
        native.emit(
            replace(
                candidate,
                sequence=2,
                assembly="replace",
                provenance=replace(candidate.provenance, native_event_id="candidate-more-input"),
            )
        )
        second = await next_text(wrapper.turn_events())
        assert second.source_release is None
        assert capture_bounds(second.source_audio[0].source_ranges) == [(0, 9600), (9600, 9601)]
        assert second.source_audio[0].speaker_proof_id is not None
        assert wrapper._turns[req.identity].timing_frontiers == {}
    finally:
        await wrapper.close()


@pytest.mark.asyncio
async def test_missing_native_publication_is_not_repaired_at_consumption():
    native, req = Provider(), request()
    wrapper = StreamingSpeakerSession(native)
    event = timed_final(
        req,
        "One two.",
        [(0, 3, 0.1, 0.2), (4, 8, 0.4, 0.5)],
        result_start=0,
        result_end=0.6000003,
    )
    try:
        await wrapper.begin_turn(req)
        await send(wrapper, req, 0, 9600, 1)
        native.events.put(event)
        await send(wrapper, req, 9600, 11200, 2)
        output = await next_text(wrapper.turn_events())
        assert output.text == event.text
        assert output.source_audio[0].speaker_proof_id is None
        assert all(
            fence.sent_until is None
            for fence in wrapper._turns[req.identity].timing_frontiers.values()
        )
        assert all(run.speaker_id is None for run in output.final_speaker_runs)
    finally:
        await wrapper.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_global_samples", [0, 3200])
@pytest.mark.parametrize("complete_words", [False, True])
async def test_actual_task_sentence_coverage_never_heals_an_older_contribution(
    prior_global_samples, complete_words
):
    from tests.providers.test_qwen_audio import (
        control,
        native,
        open_session,
    )
    from tests.providers.test_qwen_audio import request as task_request

    inner, sockets = await open_session("peer")
    inner.backend.endpoint = "wss://controlled.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
    wrapper = StreamingSpeakerSession(inner)
    req = task_request()
    req = replace(
        req, identity=replace(req.identity, segment=replace(req.identity.segment, capture_epoch=7))
    )
    try:
        if prior_global_samples:
            wrapper._synthetic_sent(prior_global_samples)
        await wrapper.begin_turn(req)
        await send(wrapper, req, 50000, 59600, 1)
        task = inner._tasks[req.identity]
        first_message = native(
            task.task_id,
            1,
            "One two.",
            begin_time=0,
            end_time=600.0003,
            words=[
                {"text": "One", "begin_time": 100, "end_time": 200},
                {"text": " two", "punctuation": ".", "begin_time": 400, "end_time": 500},
            ],
        )
        await inner._sentence(task, first_message)
        first_origin = task.origins[0]
        first_provenance = task.provenance[0]
        if complete_words:
            first_provenance = replace(
                first_provenance,
                text_timing=tuple(
                    (
                        replace(timing, semantics="word_interval", complete_boundary=True)
                        if timing.semantics == "word_interval"
                        else timing
                    )
                    for timing in first_provenance.text_timing
                ),
            )
            task.provenance[0] = first_provenance
        await send(wrapper, req, 59600, 61200, 2)
        await inner._sentence(task, native(task.task_id, 2, "More.", begin_time=600, end_time=700))
        assert task.origins[0] is first_origin
        assert task.origins[0].native.audio_coverage == first_origin.native.audio_coverage
        await wrapper.seal_turn(
            req.identity,
            sealed_content_ranges=(span(50000, 61200),),
            seal_reason="silence",
            observed_trailing_silence_ms=600,
        )
        await sockets.sockets[0].push(control(task.task_id, "task-finished"))
        while True:
            event = await asyncio.wait_for(anext(wrapper.turn_events()), 1)
            if isinstance(event, STTProviderTurnTerminal):
                break
        assert event.text == "One two. More."
        first_audio = event.source_audio[0]
        if complete_words:
            assert capture_bounds(first_audio.source_ranges) == [(51600, 58000)]
            assert first_audio.speaker_proof_id is not None
        else:
            assert first_audio.speaker_proof_id is None
        assert all(end != 59601 for _, end in capture_bounds(first_audio.source_ranges))
        assert (
            event.recognition_origins[0].native.audio_coverage == first_origin.native.audio_coverage
        )
        second_audio = event.source_audio[1]
        assert capture_bounds(second_audio.source_ranges) == [(59600, 61200)]
        assert second_audio.speaker_proof_id is not None
    finally:
        await wrapper.close()
    assert all(socket.closed for socket in sockets.sockets)


@pytest.mark.asyncio
async def test_capture_input_ownership_cannot_be_restored_by_old_producer_or_speaker_reset():
    view = SpeakerEvidenceView()
    first, second, fresh = object(), object(), object()
    view.record_input_owner((1, 7), first)
    assert view.input_history_complete((1, 7), first)
    view.record_input_owner((1, 7), second)
    assert not view.input_history_complete((1, 7), first)
    assert not view.input_history_complete((1, 7), second)
    assert not view.input_history_complete((1, 7), None)
    view.reset()
    view.record_input_owner((1, 7), first)
    assert not view.input_history_complete((1, 7), first)
    view.record_input_owner((2, 1), fresh)
    assert view.input_history_complete((2, 1), fresh)
    view.record_input_owner((1, 99), first)
    assert not view.input_history_complete((1, 99), first)
    assert view.input_history_complete((2, 1), fresh)


class IndependentProvider(Provider):
    accepts_stream_input = True
    independent_recognition_units = True

    async def begin_stream(self, stream):
        self.stream_identity = stream

    async def send_stream_audio(self, pcm16le, *, source_ranges):
        self.sent.append((self.stream_identity, pcm16le, source_ranges))

    async def end_stream(self, *, reason):
        self.end_reason = reason


class UncertainWriteProvider(Provider):
    async def send_turn_audio(self, identity, pcm16le, **kwargs):
        raise ConnectionError("uncommitted input write")


async def seed_capture_input(kind, view):
    req = request()
    if kind in ("streaming", "independent", "failed"):
        native = (
            IndependentProvider()
            if kind == "independent"
            else UncertainWriteProvider() if kind == "failed" else Provider()
        )
        wrapper = StreamingSpeakerSession(native, evidence_view=view)
    else:
        native = ControlledScopedSession()
        native.terminal_on_seal = ("empty", "")
        wrapper = CausalBatchSession(native, evidence_view=view)
    try:
        if kind == "independent":
            from puripuly_heart.domain.recognition import RecognitionStreamIdentity

            await wrapper.begin_stream(RecognitionStreamIdentity("peer", 1, 7, "independent", ()))
            await wrapper.send_stream_audio(bytes(9600), source_ranges=(span(0, 4800),))
            await wrapper.end_stream(reason="retire")
        else:
            await wrapper.begin_turn(req)
            if kind == "failed":
                with pytest.raises(ConnectionError, match="uncommitted input"):
                    await send(wrapper, req, 0, 4800, 1)
            elif kind != "empty":
                await send(wrapper, req, 0, 4800, 1)
        if kind == "batch":
            await wrapper.seal_turn(
                req.identity,
                sealed_content_ranges=(span(0, 4800),),
                seal_reason="silence",
                observed_trailing_silence_ms=600,
            )
            await wait_until(lambda: not wrapper._slices)
            events = wrapper.turn_events()
            try:
                while wrapper._events.depth:
                    await anext(events)
            finally:
                await events.aclose()
    finally:
        await wrapper.close()
    assert not view._proofs


class TextBatchSession(ControlledScopedSession):
    async def seal_turn(self, identity, **kwargs):
        await super().seal_turn(identity, **kwargs)
        self.emit(
            STTProviderTurnTerminal(
                identity,
                "final",
                "A. " if len(self.requests) == 1 else "No.",
                text_authority="authoritative",
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("before", ["streaming", "batch", "independent", "failed"])
@pytest.mark.parametrize("after", ["streaming", "batch"])
@pytest.mark.parametrize("source_retry", [False, True])
async def test_replaced_producer_history_is_unknown_but_adjacent_no_still_progresses(
    before, after, source_retry
):
    translator = ControlledProvider()
    harness, trace = harness_for(translator)
    view = harness.peer_owner._speaker_identities.evidence
    await seed_capture_input(before, view)
    req = request()
    base, _ = source_fixture()
    main_start = 0 if source_retry else 4800
    snapshot = replace(
        base,
        identity=req.identity.segment,
        settings=req.settings,
        content_ranges=(span(main_start, 7040),),
        context_ranges=() if source_retry else (span(0, 4800),),
        content_sample_count=7040 - main_start,
        context_sample_count=0 if source_retry else 4800,
        display_group_id=req.display_group_id,
    )
    if after == "streaming":
        native = Provider()
        wrapper = StreamingSpeakerSession(native, evidence_view=view)
    else:
        native = TextBatchSession()
        wrapper = CausalBatchSession(native, evidence_view=view)
    engine = ScopedRecognitionEngine(None, evidence_view=view)
    event_stream = wrapper.turn_events()
    try:
        await wrapper.begin_turn(req)
        proof = frames(0, 5120, 0) + (probabilities(5120, 5440, 0.6, 0.01),) + frames(5440, 7040, 1)
        view.observe(proof)
        await wrapper.observe_speaker_evidence(proof)
        if not source_retry:
            await send(wrapper, req, 0, 4800, 1, context=True)
        await send(wrapper, req, main_start, 7040, 1 if source_retry else 2)
        if after == "streaming":
            native.emit(
                timed_final(
                    req,
                    "A. No.",
                    [(0, 2, 0, 0.34), (3, 6, 0.34, 0.44)],
                    result_start=0,
                    result_end=0.44,
                )
            )
            events = [await next_text(event_stream)]
        else:
            await wrapper.seal_turn(
                req.identity,
                sealed_content_ranges=snapshot.content_ranges,
                seal_reason="silence",
                observed_trailing_silence_ms=600,
            )
            await wait_until(lambda: not wrapper._parents)
            events = None
        normalizer = STTScopedTurnNormalizer(req.identity)

        async def deliver(event):
            if not hasattr(event, "source_audio"):
                return
            event = (
                normalizer.apply_terminal(event)
                if isinstance(event, STTProviderTurnTerminal)
                else normalizer.apply_update(event)
            )
            if event is None:
                return
            if isinstance(event, STTProviderTurnTerminal):
                from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt

                await harness.peer_owner.handle_provider_turn_terminal(
                    AudioSegmentTerminalReceipt(req.identity.segment, "final", snapshot, 10), event
                )
            elif event.source_release is not None:
                await harness.peer_owner.handle_provider_turn_update(snapshot, event)

        engine.event_sink = deliver
        if events is not None:
            for event in events:
                await engine._emit(event)
        else:
            while wrapper._events.depth:
                await engine._emit(await anext(event_stream))
        await event_stream.aclose()
        await finish(harness)
        children = [item.source_child for item in trace.items]
        assert [child.raw_text for child in children] == ["A. ", "No."]
        assert [child.speaker.speaker_id for child in children] == [
            "0" if source_retry else None,
            "1",
        ]
        assert children[0].speaker.attribution_state == (
            "identified" if source_retry else "unavailable"
        )
        assert (children[0].origins[0].speaker_proof_id is not None) == source_retry
        assert len(translator.calls) == len(harness.peer_runtime.translation_history) == 2
        await wrapper.close()
        for child in children:
            harness.peer_owner._speaker_identities.retire_source(child.source_id)
        assert not view._proofs
    finally:
        await event_stream.aclose()
        await wrapper.close()
        await engine.close()
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("pressure", [False, True])
async def test_real_parent_and_parts_are_disposed_on_rejection_but_healthy_close_drains(pressure):
    from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt
    from puripuly_heart.core.clock import FakeClock
    from puripuly_heart.core.overlay.presenter import OverlayPresenter
    from puripuly_heart.ui.overlay_calibration import OverlayCalibration

    presenter = OverlayPresenter(OverlayCalibration(), clock=FakeClock(10))
    harness, trace = harness_for(ControlledProvider(), overlay_sink=presenter)
    view = harness.peer_owner._speaker_identities.evidence
    native = Provider()
    wrapper = StreamingSpeakerSession(native, evidence_view=view)
    snapshot, identity = source_fixture()
    from puripuly_heart.core.stt.backend import STTProviderTurnRequest

    req = STTProviderTurnRequest(identity, snapshot.settings)
    update_event = timed_final(
        req,
        "No. He washed",
        [(0, 3, 2.8799999, 3.28), (4, 6, 3.28, 3.52), (7, 13, 3.52, 3.76)],
        result_start=2.8799999,
        result_end=3.825,
    )
    events = wrapper.turn_events()

    async def deliver(event):
        await harness.peer_owner.handle_provider_turn_terminal(
            AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 10), event
        )

    engine = ScopedRecognitionEngine(None, evidence_view=view, event_sink=deliver)
    try:
        await wrapper.begin_turn(req)
        await send(wrapper, req, 1728, 9728, 1, context=True)
        await send(wrapper, req, 9728, 84992, 2)
        proof = recorded_short_evidence(gray=True)
        view.observe(proof)
        await wrapper.observe_speaker_evidence(proof)
        native.emit(replace(update_event, source_release=None))
        candidate = await next_text(events)
        assert candidate.source_release is None
        while wrapper._events.depth:
            await anext(events)
        await events.aclose()
        wrapper._events = STTProviderEventBuffer(
            max_events=1 if pressure else 2, on_discard=wrapper._discard_buffered_event
        )
        wrapper._put_event(
            STTProviderAudioBoundary(
                req.identity, STTAudioBoundary("prefill", 7, 5440, 3.0, "speaker")
            )
        )
        native.emit(
            STTProviderTurnTerminal(
                req.identity,
                "final",
                update_event.text,
                text_authority="authoritative",
                provenance=(update_event.provenance,),
                included_contributions=(update_event.contribution,),
            )
        )
        await native.close()
        await asyncio.wait_for(asyncio.shield(wrapper._consumer), 1)
        events = wrapper.turn_events()
        if pressure:
            event = await anext(events)
            assert event.outcome == "failed"
            assert event.failure_reason == "provider_event_buffer_overflow"
            assert event.source_audio == ()
            assert not view._proofs
        else:
            assert wrapper._events.depth == 2
            assert len(view._proofs) == 3
            await wrapper.close()
            assert len(view._proofs) == 3
            assert isinstance(await anext(events), STTProviderAudioBoundary)
            event = await anext(events)
            await engine._emit(event)
            await events.aclose()
            await finish(harness)
            assert [item.source_child.raw_text for item in trace.items] == ["No. ", "He washed"]
            assert len(view._proofs) == 2
            for item in trace.items:
                harness.peer_owner._speaker_identities.retire_source(item.source_child.source_id)
        await wrapper.close()
        assert not view._proofs
    finally:
        await events.aclose()
        await wrapper.close()
        await engine.close()
        await close(harness)
        await presenter.close()


def proof_event(req, view):
    tokens = tuple(view.pin((span(left, left + 160),)) for left in (0, 160, 320))
    unknown = SpeakerAttribution("unavailable")
    audio = STTSourceAudio(
        "phrase-1",
        (span(0, 480),),
        speaker_proof_id=tokens[0],
        speaker_parts=(
            STTSpeakerAudioPart(0, 4, (span(0, 160),), (), unknown, tokens[1]),
            STTSpeakerAudioPart(4, 8, (span(160, 480),), (), unknown, tokens[2]),
        ),
    )
    return replace(update(req, "One two.", release=True), source_audio=(audio,)), tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize("kind", ["streaming", "batch"])
async def test_wrapper_discard_keeps_claims_shared_with_accepted_incoming(coalesced, kind):
    view, req = SpeakerEvidenceView(), request()
    wrapper = (
        StreamingSpeakerSession(Provider(), evidence_view=view)
        if kind == "streaming"
        else CausalBatchSession(ControlledScopedSession(), evidence_view=view)
    )
    event, tokens = proof_event(req, view)
    provisional = replace(event, stability="provisional", assembly="replace", source_release=None)
    wrapper._events = STTProviderEventBuffer(
        max_events=1, on_discard=wrapper._discard_buffered_event
    )
    events = wrapper.turn_events()
    try:
        assert wrapper._put_event(provisional)
        incoming = replace(provisional if coalesced else event, sequence=2)
        assert wrapper._put_event(incoming)
        assert all(view.has_proof(token) for token in tokens)
        assert await anext(events) is incoming
        await events.aclose()
        assert not view._proofs
    finally:
        await events.aclose()
        await wrapper.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["streaming", "batch"])
async def test_wrapper_rejection_releases_requests_not_contribution_or_source_handoffs(kind):
    view, req = SpeakerEvidenceView(), request()
    wrapper = (
        StreamingSpeakerSession(Provider(), evidence_view=view)
        if kind == "streaming"
        else CausalBatchSession(ControlledScopedSession(), evidence_view=view)
    )
    event, tokens = proof_event(req, view)
    view.publish(tokens[1])
    view.adopt(tokens[2])
    wrapper._events.close()
    try:
        from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBufferClosed

        with pytest.raises(STTProviderEventBufferClosed):
            wrapper._put_event(event)
        assert not view.has_proof(tokens[0])
        assert view.has_proof(tokens[1]) and view.has_proof(tokens[2])
        discard_source_audio_proofs(event, view, published=True)
        assert not view.has_proof(tokens[1])
        assert view.has_proof(tokens[2])
        view.release(tokens[2])
    finally:
        await wrapper.close()


@pytest.mark.asyncio
async def test_engine_buffer_rejection_disposes_parts_without_retiring_healthy_output():
    view, req = SpeakerEvidenceView(), request()
    engine = ScopedRecognitionEngine(None, evidence_view=view)
    engine._event_buffer = STTProviderEventBuffer(
        max_events=1, on_discard=engine._discard_buffered_event
    )
    healthy, healthy_tokens = proof_event(req, view)
    healthy = STTProviderTurnTerminal(
        req.identity, "final", "healthy", source_audio=healthy.source_audio
    )
    incoming_req = replace(req, identity=replace(req.identity, provider_epoch_id="other"))
    rejected, rejected_tokens = proof_event(incoming_req, view)
    try:
        await engine._emit(healthy)
        assert all(view._proofs[token].stage == "contribution" for token in healthy_tokens)
        await engine._emit(rejected)
        assert all(not view.has_proof(token) for token in rejected_tokens)
        assert all(view.has_proof(token) for token in healthy_tokens)
        delivered = []
        engine.bind_event_sink(delivered.append)
        await engine.wait_for_event_ingress_drain()
        assert delivered[0] is healthy
        assert not view._proofs
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "before,generation,epoch",
    [("streaming", 1, 8), ("batch", 1, 8), ("streaming", 2, 1), ("batch", 2, 1), ("empty", 1, 7)],
)
async def test_fresh_capture_or_no_prior_input_preserves_genuine_context_onset(
    before, generation, epoch
):
    view = SpeakerEvidenceView()
    await seed_capture_input(before, view)
    req = request()
    req = replace(
        req,
        identity=replace(
            req.identity,
            segment=replace(
                req.identity.segment, activation_generation=generation, capture_epoch=epoch
            ),
        ),
    )
    native = Provider()
    wrapper = StreamingSpeakerSession(native, evidence_view=view)
    output = None
    try:
        await wrapper.begin_turn(req)
        for sequence, (start, end, context) in enumerate(((0, 4800, True), (4800, 7040, False)), 1):
            await wrapper.send_turn_audio(
                req.identity,
                bytes((end - start) * 2),
                payload_sequence=sequence,
                source_ranges=(replace(span(start, end), capture_epoch=epoch),),
                context_only=context,
            )
        raw = frames(0, 5120, 0) + (probabilities(5120, 5440, 0.6, 0.01),) + frames(5440, 7040, 1)
        proof = tuple(
            replace(
                item,
                source_map=tuple(
                    replace(piece, capture=replace(piece.capture, capture_epoch=epoch))
                    for piece in item.source_map
                ),
            )
            for item in raw
        )
        view.observe(proof)
        await wrapper.observe_speaker_evidence(proof)
        native.emit(
            timed_final(
                req,
                "A. No.",
                [(0, 2, 0, 0.34), (3, 6, 0.34, 0.44)],
                result_start=0,
                result_end=0.44,
            )
        )
        output = await next_text(wrapper.turn_events())
        assert [run.text for run in output.final_speaker_runs] == ["A. ", "No."]
        assert [run.speaker_id for run in output.final_speaker_runs] == ["0", "1"]
        first = output.source_audio[0].speaker_parts[0]
        claim = view._proofs[first.speaker_proof_id]
        assert claim.replayed == ()
        assert view.resolve(first.speaker_proof_id).speaker_id == "0"
    finally:
        if output is not None:
            discard_source_audio_proofs(output, view)
        await wrapper.close()
        assert not view._proofs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "complete_words,binding",
    [(False, "exact"), (True, "exact"), (True, "other_task"), (True, "other_contribution")],
)
async def test_earlier_published_task_sentence_keeps_its_own_coverage_and_binding(
    complete_words, binding
):
    from puripuly_heart.core.stt.backend import STTTextContribution, STTTextTiming
    from puripuly_heart.domain.recognition import NativeAudioCoverage
    from tests.core.test_source_progressive_integration import origin

    native, req = Provider(), request()
    wrapper = StreamingSpeakerSession(native)
    wrapper._synthetic_sent(3200)
    first = timed_final(
        req, "One two.", [(0, 3, 0.1, 0.2), (4, 8, 0.4, 0.5)], result_start=0, result_end=0.6000003
    )
    first_origin = origin(first.text)
    first_origin = replace(
        first_origin,
        contribution_id="other" if binding == "other_contribution" else "phrase-1",
        identity=replace(
            first_origin.identity, stream=replace(first_origin.identity.stream, channel="peer")
        ),
        native=replace(
            first_origin.native,
            native_task_id="other-task" if binding == "other_task" else "task",
            timestamp_basis="task_relative_estimated",
            input_sample_rate_hz=16000,
            task_input_samples=9600,
            audio_coverage=(
                NativeAudioCoverage(
                    0,
                    9600,
                    7,
                    16000,
                    50000,
                    59600,
                    16000,
                    50000,
                    59600,
                    50000 / 16000,
                    59600 / 16000,
                    False,
                ),
            ),
        ),
    )
    first = replace(
        first,
        recognition_origins=(first_origin,),
        provenance=replace(
            first.provenance,
            audio_clocks=(),
            text_timing=tuple(
                replace(
                    item,
                    clock_id="task",
                    complete_boundary=(item.semantics == "result_span" or complete_words),
                )
                for item in first.provenance.text_timing
            ),
        ),
    )
    second_origin = replace(
        first_origin,
        text_start=8,
        text_end=14,
        contribution_id="phrase-2",
        identity=replace(first_origin.identity, unit_id=uuid4(), receipt_sequence=2),
        native=replace(
            first_origin.native,
            native_task_id="task",
            task_input_samples=11200,
            audio_coverage=(
                NativeAudioCoverage(
                    9600,
                    11200,
                    7,
                    16000,
                    59600,
                    61200,
                    16000,
                    59600,
                    61200,
                    59600 / 16000,
                    61200 / 16000,
                    False,
                ),
            ),
        ),
    )
    second = replace(
        update(req, " More.", release=True),
        sequence=2,
        contribution=STTTextContribution("phrase-2", 8, 14),
        recognition_origins=(second_origin,),
        source_release=STTSourceRelease("second", 8, 14, "end"),
        provenance=replace(
            first.provenance,
            native_event_id="second",
            text_timing=(
                STTTextTiming(
                    "phrase-2",
                    8,
                    14,
                    "result_span",
                    0.6,
                    0.7,
                    clock_id="task",
                    complete_boundary=True,
                ),
            ),
        ),
    )
    try:
        await wrapper.begin_turn(req)
        await send(wrapper, req, 50000, 59600, 1)
        native.emit(first)
        before = await next_text(wrapper.turn_events())
        assert {
            fence.sent_until for fence in wrapper._turns[req.identity].timing_frontiers.values()
        } == {12800}
        await send(wrapper, req, 59600, 61200, 2)
        native.emit(second)
        after = await next_text(wrapper.turn_events())
        assert after.text == "One two. More."
        for event in (before, after):
            audio = event.source_audio[0]
            if complete_words and binding == "exact":
                assert capture_bounds(audio.source_ranges) == [(51600, 58000)]
                assert audio.speaker_proof_id is not None
            else:
                assert audio.speaker_proof_id is None
            assert all(end != 59601 for _, end in capture_bounds(audio.source_ranges))
        old_keys = [
            fence
            for key, fence in wrapper._turns[req.identity].timing_frontiers.items()
            if key[0] == "phrase-1"
        ]
        assert all(fence.sent_until == 12800 for fence in old_keys)
        assert first_origin.native.audio_coverage[0].input_end_sample == 9600
    finally:
        await wrapper.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("new_input_first", [False, True])
async def test_independent_forwarding_uses_send_owned_scope_and_never_claims_mapped_history(
    new_input_first,
):
    from puripuly_heart.domain.recognition import RecognitionStreamIdentity

    entered, release = asyncio.Event(), asyncio.Event()

    class GatedIndependent(IndependentProvider):
        async def send_stream_audio(self, pcm16le, *, source_ranges):
            owned = self.stream_identity
            if not entered.is_set():
                entered.set()
                await release.wait()
            self.sent.append((owned, pcm16le, source_ranges))

    view = SpeakerEvidenceView()
    native = GatedIndependent()
    wrapper = StreamingSpeakerSession(native, evidence_view=view)
    first = RecognitionStreamIdentity("peer", 1, 7, "first", ())
    second = replace(first, activation_generation=2, capture_epoch=1, provider_epoch_id="second")
    payload = bytes(9600)
    capture = (span(0, 4800),)
    pending = None
    try:
        await wrapper.begin_stream(first)
        pending = asyncio.create_task(wrapper.send_stream_audio(payload, source_ranges=capture))
        await entered.wait()
        assert view._input_scope is None
        await wrapper.begin_stream(second)
        if new_input_first:
            await wrapper.send_stream_audio(
                payload, source_ranges=(replace(capture[0], capture_epoch=1),)
            )
        release.set()
        await pending
        assert (first, payload, capture) in native.sent
        assert view._input_scope == ((2, 1) if new_input_first else (1, 7))
        assert not view.input_history_complete(view._input_scope, wrapper._input_owner)
        assert not view.input_history_complete(view._input_scope, None)
        assert not wrapper._maps
        assert not view._proofs
        await wrapper.end_stream(reason="retire")
        assert native.end_reason == "retire"
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await wrapper.close()
