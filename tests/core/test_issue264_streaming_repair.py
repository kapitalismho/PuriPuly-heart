from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.stt.backend import (
    STTAudioClock,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTSourceAudio,
    STTSourceRelease,
    STTTextTiming,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from puripuly_heart.domain.models import FinalSpeakerRun, SpeakerAttribution
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.core.test_causal_batch_speaker import evidence, span
from tests.core.test_issue264_shared_speaker import changed, frames
from tests.core.test_source_context_contract import (
    ControlledProvider,
    close,
    finish,
    harness_for,
    source_fixture,
)
from tests.core.test_streaming_speaker import Provider, next_text, send, update


def probabilities(start, end, first, second):
    item = evidence(start, end, ())
    return replace(
        item,
        frame=replace(item.frame, probabilities=(first, second) + (0.00001,) * 6),
    )


def recorded_short_evidence(*, gray=False):
    edges = {
        53568: (0.04228902980685234, 0.7107461094856262),
        53728: (0.052157796919345856, 0.43016675114631653),
        53888: (0.054736386984586716, 0.20898893475532532),
        54048: (0.056080400943756104, 0.09394676238298416),
        54208: (0.16171850264072418, 0.04052429646253586),
        54368: (0.3135562241077423, 0.030187929049134254),
        54528: (0.5104352235794067, 0.026246581226587296),
        54688: (0.7436028122901917, 0.02074155956506729),
        54848: (0.9030384421348572, 0.010001761838793755),
        55008: (0.9660727977752686, 0.0052249315194785595),
    }
    result = []
    for left in range(1728, 84992, 160):
        if left in edges:
            first, second = edges[left]
        elif left < 41888:
            first, second = 0.9, 0.01
        elif left < 47648:
            first, second = 0.01, 0.01
        elif left < 47968:
            first, second = 0.01, 0.7
        elif left < 53568:
            first, second = 0.01, 0.9 if not gray or left >= 53248 else 0.6
        else:
            first, second = (0.6 if gray else 0.9), 0.01
        result.append(probabilities(left, min(left + 160, 84992), first, second))
    return tuple(result)


def timed_final(req, text, words, *, result_start, result_end, qualified=True):
    event = update(req, text, release=True)
    return replace(
        event,
        provenance=replace(
            event.provenance,
            text_timing=(
                STTTextTiming(
                    "phrase-1",
                    0,
                    len(text),
                    "result_span",
                    result_start,
                    result_end,
                    clock_id="clock",
                    complete_boundary=True,
                    estimated=True,
                ),
                *(
                    STTTextTiming(
                        "phrase-1",
                        start,
                        end,
                        "word_interval",
                        raw_start,
                        raw_end,
                        clock_id="clock",
                        complete_boundary=True,
                        estimated=True,
                    )
                    for start, end, raw_start, raw_end in words
                ),
            ),
            audio_clocks=(STTAudioClock("clock", 16000, 0, qualified),),
        ),
    )


async def admit(harness, session, provider, req, snapshot, event):
    events = session.turn_events()
    engine = ScopedRecognitionEngine(
        None,
        evidence_view=harness.peer_owner._speaker_identities.evidence,
        event_sink=lambda outgoing: harness.peer_owner.handle_provider_turn_update(
            snapshot, outgoing
        ),
    )
    try:
        provider.emit(event)
        outgoing = await next_text(events)
        outgoing = STTScopedTurnNormalizer(req.identity).apply_update(outgoing)
        await engine._emit(outgoing)
    finally:
        await events.aclose()
        await engine.close()
    await finish(harness)
    return outgoing


@pytest.mark.asyncio
async def test_recorded_returning_prefix_gets_complete_word_envelope_and_late_claim():
    clock = FakeClock(_now=10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    translator = ControlledProvider()
    harness, trace = harness_for(translator, overlay_sink=presenter)
    harness.peer_owner.set_clock(clock)
    view = harness.peer_owner._speaker_identities.evidence
    provider = Provider()
    session = StreamingSpeakerSession(provider, evidence_view=view)
    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )
    text = "He washed the detritus from his"
    words = [
        (0, 2, 4.515, 4.915),
        (3, 9, 4.915, 5.3149996),
        (10, 13, 5.3149996, 5.475),
        (14, 22, 5.475, 6.035),
        (23, 27, 6.035, 6.1949997),
        (28, 31, 6.1949997, 6.3549995),
    ]
    try:
        await session.begin_turn(req)
        await send(session, req, 1728, 9728, 1, context=True)
        await send(session, req, 9728, 105472, 2)
        event = timed_final(req, text, words, result_start=4.515, result_end=6.4840003)
        outgoing = await admit(harness, session, provider, req, snapshot, event)
        (audio,) = outgoing.source_audio
        assert audio.speaker_proof_id is not None
        assert (
            audio.source_ranges[0].normalized_start_sample,
            audio.source_ranges[-1].normalized_end_sample,
        ) == (73968, 103408)
        source = trace.items[0].source_child
        token = source.origins[0].speaker_proof_id
        assert token == audio.speaker_proof_id
        assert source.speaker.speaker_id is None
        historical_origins = source.origins
        first = presenter.snapshot().blocks[0]
        assert first.speaker_style == "gray"
        entry = next(iter(presenter._entries.values()))
        ttl = presenter._entry_expiration_deadline(entry)
        await session.close()
        assert view.has_proof(token)
        clock.advance(0.2)
        view.observe(frames(73968, 103408, 0))
        await changed(presenter, lambda: presenter.snapshot().blocks[0].speaker_style == "gold")
        final = presenter.snapshot().blocks[0]
        assert (final.id, final.primary_text, final.secondary_text, final.appearance_seq) == (
            first.id,
            first.primary_text,
            first.secondary_text,
            first.appearance_seq,
        )
        assert presenter._entry_expiration_deadline(entry) == ttl
        assert len(translator.calls) == len(harness.peer_runtime.translation_history) == 1
        assert not view.has_proof(token)
        assert source.speaker.speaker_id is None
        assert source.origins == historical_origins
        assert outgoing.provenance.text_timing[0].raw_end == 6.4840003
    finally:
        await session.close()
        await close(harness)
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["missing_word", "bad_word", "duplicate", "mixed_clock", "unqualified", "synthetic_gap"],
)
async def test_unusable_coarse_span_does_not_repair_incomplete_or_unqualified_words(failure):
    provider = Provider()
    session = StreamingSpeakerSession(provider)
    from tests.core.test_streaming_speaker import request

    req = request()
    event = timed_final(
        req,
        "One two.",
        [(0, 3, 0.1, 0.2), (4, 8, 0.4, 0.5)],
        result_start=0,
        result_end=0.6000003,
        qualified=failure != "unqualified",
    )
    timings = list(event.provenance.text_timing)
    if failure == "missing_word":
        timings.pop(1)
    elif failure == "bad_word":
        timings[1] = replace(timings[1], raw_start=0.3, raw_end=0.2)
    elif failure == "duplicate":
        timings.append(replace(timings[1], text_start=1))
    elif failure == "mixed_clock":
        timings[-1] = replace(timings[-1], clock_id="other")
        event = replace(
            event,
            provenance=replace(
                event.provenance,
                audio_clocks=event.provenance.audio_clocks
                + (STTAudioClock("other", 16000, 0, True),),
            ),
        )
    event = replace(event, provenance=replace(event.provenance, text_timing=tuple(timings)))
    try:
        await session.begin_turn(req)
        if failure == "synthetic_gap":
            await send(session, req, 0, 3200, 1)
            session._synthetic_sent(1600)
            await send(session, req, 4800, 9600, 2)
        else:
            await send(session, req, 0, 9600, 1)
        provider.emit(event)
        result = await next_text(session.turn_events())
        assert result.text == "One two."
        assert result.source_audio[0].speaker_proof_id is None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_word_envelope_retains_foreign_interword_audio_and_valid_coarse_mapping():
    from tests.core.test_streaming_speaker import request

    for coarse_end, expected_bounds in [(0.6000003, (1600, 8000)), (0.6, (0, 9600))]:
        provider, req = Provider(), request()
        session = StreamingSpeakerSession(provider)
        try:
            await session.begin_turn(req)
            await send(session, req, 0, 9600, 1)
            await session.observe_speaker_evidence(
                frames(0, 3200, 0) + frames(3200, 4800, 1) + frames(4800, 9600, 0)
            )
            event = timed_final(
                req,
                "One two.",
                [(0, 3, 0.1, 0.2), (4, 8, 0.4, 0.5)],
                result_start=0,
                result_end=coarse_end,
            )
            provider.emit(event)
            result = await next_text(session.turn_events())
            audio = result.source_audio[0]
            assert (
                audio.source_ranges[0].normalized_start_sample,
                audio.source_ranges[-1].normalized_end_sample,
            ) == expected_bounds
            assert audio.speaker_attribution.state == "mixed"
            assert result.final_speaker_runs[0].attribution_state == "mixed"
            assert session._timeline.resolve(audio.speaker_proof_id).attribution_state == "mixed"
        finally:
            await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("gray", [False, True])
async def test_recorded_no_he_junction_reaches_distinct_children_and_narrow_claims(gray):
    presenter = OverlayPresenter(OverlayCalibration(), clock=FakeClock(_now=10))
    translator = ControlledProvider()
    harness, trace = harness_for(translator, overlay_sink=presenter)
    view = harness.peer_owner._speaker_identities.evidence
    provider = Provider()
    session = StreamingSpeakerSession(provider, evidence_view=view)
    snapshot, identity = source_fixture()
    from puripuly_heart.core.stt.backend import STTProviderTurnRequest

    req = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )
    proof = recorded_short_evidence(gray=gray)
    event = timed_final(
        req,
        "No. He washed",
        [(0, 3, 2.8799999, 3.28), (4, 6, 3.28, 3.52), (7, 13, 3.52, 3.76)],
        result_start=2.8799999,
        result_end=3.825,
    )
    try:
        await session.begin_turn(req)
        await send(session, req, 1728, 9728, 1, context=True)
        await send(session, req, 9728, 84992, 2)
        view.observe(proof)
        await session.observe_speaker_evidence(proof)
        outgoing = await admit(harness, session, provider, req, snapshot, event)
        sources = [item.source_child for item in trace.items]
        assert [source.raw_text for source in sources] == ["No. ", "He washed"]
        assert "".join(source.raw_text for source in sources) == event.text
        assert len(translator.calls) == len(harness.peer_runtime.translation_history) == 2
        assert outgoing.final_speaker_runs[1].source_boundary_before
        parent_token = outgoing.source_audio[0].speaker_proof_id
        assert not view.has_proof(parent_token)
        assert [source.origins[0].contribution_range for source in sources] == [(0, 4), (4, 13)]
        ranges = [source.origins[0].source_ranges for source in sources]
        assert [
            (items[0].normalized_start_sample, items[-1].normalized_end_sample) for items in ranges
        ] == [(47807, 54208), (54208, 61888)]
        tokens = [source.origins[0].speaker_proof_id for source in sources]
        assert len(set(tokens)) == 2 and None not in tokens
        if gray:
            assert [source.speaker.attribution_state for source in sources] == [
                "uncertain",
                "uncertain",
            ]
            assert [block.speaker_style for block in presenter.snapshot().blocks] == [
                "gray",
                "gray",
            ]
            assert all(view.has_proof(token) for token in tokens)
        else:
            assert [source.speaker.speaker_id for source in sources] == ["1", "0"]
        await harness.peer_owner.handle_provider_turn_update(snapshot, outgoing)
        await finish(harness)
        assert len(trace.items) == 2
        await session.close()
        if gray:
            assert all(view.has_proof(token) for token in tokens)
        for source in sources:
            harness.peer_owner._speaker_identities.retire_source(source.source_id)
        assert not view._proofs
    finally:
        await session.close()
        await close(harness)
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "missing_band",
        "overlap",
        "third_slot",
        "unqualified",
        "incomplete_text",
        "multiple_junctions",
        "no_confirm",
    ],
)
async def test_uncertainty_junction_rejects_unsupported_correspondence(failure):
    from tests.core.test_streaming_speaker import request

    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    proof = list(recorded_short_evidence(gray=True))
    event = timed_final(
        req,
        "No. He washed",
        [(0, 3, 2.8799999, 3.28), (4, 6, 3.28, 3.52), (7, 13, 3.52, 3.76)],
        result_start=2.8799999,
        result_end=3.825,
        qualified=failure != "unqualified",
    )
    if failure == "missing_band":
        proof = [
            item for item in proof if item.source_map[0].capture.normalized_start_sample != 54048
        ]
    elif failure in ("overlap", "third_slot"):
        for index, item in enumerate(proof):
            if item.source_map[0].capture.normalized_start_sample == 54048:
                values = (
                    (0.7, 0.7) + (0.01,) * 6
                    if failure == "overlap"
                    else (0.01, 0.01, 0.9) + (0.01,) * 5
                )
                proof[index] = replace(item, frame=replace(item.frame, probabilities=values))
    elif failure == "incomplete_text":
        event = replace(
            event,
            provenance=replace(
                event.provenance,
                text_timing=tuple(
                    item
                    for item in event.provenance.text_timing
                    if item.semantics == "result_span" or item.text_start != 4
                ),
            ),
        )
    elif failure == "multiple_junctions":
        event = timed_final(
            req,
            "No. Ah. He",
            [(0, 3, 2.8799999, 3.26), (4, 7, 3.26, 3.30), (8, 10, 3.30, 3.52)],
            result_start=2.8799999,
            result_end=3.825,
        )
    elif failure == "no_confirm":
        proof = [
            (
                probabilities(55008, 55168, 0.6, 0.01)
                if item.source_map[0].capture.normalized_start_sample == 55008
                else item
            )
            for item in proof
        ]
    try:
        await session.begin_turn(req)
        await send(session, req, 1728, 9728, 1, context=True)
        await send(session, req, 9728, 84992, 2)
        await session.observe_speaker_evidence(proof)
        provider.emit(event)
        result = await next_text(session.turn_events())
        assert len(result.final_speaker_runs) == 1
        assert result.final_speaker_runs[0].text == event.text
        assert not result.final_speaker_runs[0].source_boundary_before
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", ["cancel", "reset", "retire", "expiry"])
async def test_partial_release_repin_terminal_replay_and_adopted_part_disposition(disposition):
    clock = FakeClock(_now=10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    translator = ControlledProvider()
    harness, trace = harness_for(translator, overlay_sink=presenter)
    harness.peer_owner.set_clock(clock)
    allocator = harness.peer_owner._speaker_identities
    view = allocator.evidence
    provider = Provider()
    session = StreamingSpeakerSession(provider, evidence_view=view)
    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )
    event = timed_final(
        req,
        "No. He washed",
        [(0, 3, 2.8799999, 3.28), (4, 6, 3.28, 3.52), (7, 13, 3.52, 3.76)],
        result_start=2.8799999,
        result_end=3.825,
    )
    event = replace(event, source_release=STTSourceRelease("prefix", 0, 4, "boundary"))
    try:
        await session.begin_turn(req)
        await send(session, req, 1728, 9728, 1, context=True)
        await send(session, req, 9728, 84992, 2)
        proof = recorded_short_evidence(gray=True)
        view.observe(proof)
        await session.observe_speaker_evidence(proof)
        first = await admit(harness, session, provider, req, snapshot, event)
        first_source = trace.items[0].source_child
        first_token = first_source.origins[0].speaker_proof_id
        future_token = first.source_audio[0].speaker_parts[1].speaker_proof_id
        assert view.has_proof(first_token)
        assert not view.has_proof(first.source_audio[0].speaker_proof_id)
        assert not view.has_proof(future_token)
        full_event = replace(
            event,
            sequence=2,
            assembly="replace",
            provenance=replace(event.provenance, native_event_id="full"),
            source_release=STTSourceRelease("full", 0, 13, "end"),
        )
        full = await admit(harness, session, provider, req, snapshot, full_event)
        second_source = trace.items[1].source_child
        second_token = second_source.origins[0].speaker_proof_id
        assert second_token is not None and second_token != future_token
        assert view.has_proof(first_token) and view.has_proof(second_token)
        assert [item.source_child.raw_text for item in trace.items] == ["No. ", "He washed"]
        await harness.peer_owner.handle_provider_turn_update(snapshot, full)
        provider.emit(
            STTProviderTurnTerminal(
                req.identity,
                "final",
                event.text,
                text_authority="authoritative",
                provenance=(full_event.provenance,),
                included_contributions=(event.contribution,),
            )
        )
        await asyncio.sleep(0)
        events = session.turn_events()
        try:
            while session._events.depth:
                terminal = await anext(events)
                if isinstance(terminal, STTProviderTurnTerminal):
                    from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt

                    await harness.peer_owner.handle_provider_turn_terminal(
                        AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 10),
                        terminal,
                    )
        finally:
            await events.aclose()
        await finish(harness)
        assert len(trace.items) == len(translator.calls) == 2
        await session.close()
        assert view.has_proof(first_token) and view.has_proof(second_token)
        if disposition == "cancel":
            for item in trace.items:
                allocator.cancel_attempt(item.child_utterance_id)
        elif disposition == "reset":
            view.reset()
        elif disposition == "retire":
            allocator.retire_source(first_source.source_id)
            allocator.retire_source(second_source.source_id)
        else:
            clock.advance(1.1)
            allocator.refresh()
        assert not view._proofs
    finally:
        await session.close()
        await close(harness)
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("exception_type", [RuntimeError, asyncio.CancelledError])
async def test_admission_rejection_or_cancellation_releases_unadopted_parts(
    monkeypatch, exception_type
):
    translator = ControlledProvider()
    harness, _ = harness_for(translator)
    view = harness.peer_owner._speaker_identities.evidence
    provider = Provider()
    session = StreamingSpeakerSession(provider, evidence_view=view)
    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )

    async def reject(_self, _event):
        raise exception_type("controlled admission rejection")

    monkeypatch.setattr(type(harness.peer_owner), "handle_stt_event", reject)
    try:
        await session.begin_turn(req)
        await send(session, req, 1728, 9728, 1, context=True)
        await send(session, req, 9728, 84992, 2)
        proof = recorded_short_evidence(gray=True)
        view.observe(proof)
        await session.observe_speaker_evidence(proof)
        event = timed_final(
            req,
            "No. He washed",
            [(0, 3, 2.8799999, 3.28), (4, 6, 3.28, 3.52), (7, 13, 3.52, 3.76)],
            result_start=2.8799999,
            result_end=3.825,
        )
        with pytest.raises(exception_type):
            await admit(harness, session, provider, req, snapshot, event)
        await session.close()
        assert not view._proofs
        assert translator.calls == []
    finally:
        await session.close()
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed", ["overlap", "outside", "pressure", "empty", "noninteger", "role"]
)
async def test_malformed_or_over_limit_parts_preserve_text_and_release_claims(malformed):
    from puripuly_heart.core.stt.backend import MAX_SPEAKER_AUDIO_PARTS, STTSpeakerAudioPart

    translator = ControlledProvider()
    harness, trace = harness_for(translator)
    view = harness.peer_owner._speaker_identities.evidence
    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(identity, snapshot.settings)
    event = update(req, "one two", release=True)
    parent = view.pin((span(0, 16000),))
    first, second = view.pin((span(0, 8000),)), view.pin((span(8000, 16000),))
    unknown = SpeakerAttribution("unavailable")
    parts = (
        STTSpeakerAudioPart(0, 4, (span(0, 8000),), (), unknown, first),
        STTSpeakerAudioPart(
            3 if malformed == "overlap" else 4,
            7,
            (span(8000, 17000 if malformed == "outside" else 16000),),
            (),
            unknown,
            second,
        ),
    )
    if malformed == "pressure":
        parts = (parts[0],) * MAX_SPEAKER_AUDIO_PARTS + (parts[1],)
    elif malformed == "empty":
        parts = (parts[0], replace(parts[1], source_ranges=()))
    elif malformed == "noninteger":
        parts = (parts[0], replace(parts[1], text_end=None))
    elif malformed == "role":
        parts = (
            parts[0],
            replace(parts[1], source_ranges=(), context_ranges=parts[1].source_ranges),
        )
    event = replace(
        event,
        source_audio=(
            STTSourceAudio(
                "phrase-1",
                (span(0, 16000),),
                speaker_attribution=unknown,
                speaker_proof_id=parent,
                speaker_parts=parts,
            ),
        ),
        final_speaker_runs=(FinalSpeakerRun("one two", None, "11", source="nemotron"),),
        included_contributions=(event.contribution,),
    )
    try:
        engine = ScopedRecognitionEngine(None, evidence_view=view)
        engine._publish_speaker_proofs(event)
        await harness.peer_owner.handle_provider_turn_update(snapshot, event)
        await finish(harness)
        assert [item.source_child.raw_text for item in trace.items] == ["one two"]
        assert trace.items[0].source_child.origins[0].speaker_proof_id is None
        assert len(translator.calls) == 1
        assert not view._proofs
    finally:
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,expected",
    [
        ("genuine", "identified"),
        ("replayed", "uncertain"),
        ("partial", "identified"),
        ("new_capture", "identified"),
        ("history_pressure", "unavailable"),
        ("missing_history", "unavailable"),
        ("source_retry", "identified"),
        ("late_context", "unavailable"),
    ],
)
async def test_successful_capture_replay_is_not_new_support_but_genuine_onset_is(
    mode, expected, monkeypatch
):
    from tests.core.test_streaming_speaker import request

    if mode == "history_pressure":
        monkeypatch.setattr(StreamingSpeakerSession, "MAX_MAP_PIECES", 2)
    provider = Provider()
    session = StreamingSpeakerSession(provider)
    first, req = request(), request()
    try:
        if mode not in ("genuine", "late_context"):
            await session.begin_turn(first)
            prior_end = 2400 if mode == "partial" else 4800
            if mode == "new_capture":
                prior_span = replace(span(0, prior_end), capture_epoch=6)
                await session.send_turn_audio(
                    first.identity,
                    bytes(prior_end * 2),
                    payload_sequence=1,
                    source_ranges=(prior_span,),
                    context_only=False,
                )
            elif mode == "missing_history":
                await session.send_turn_audio(
                    first.identity,
                    bytes(prior_end * 2),
                    payload_sequence=1,
                    source_ranges=(),
                    context_only=False,
                )
            elif mode == "history_pressure":
                await send(session, first, 0, 2400, 1)
                await send(session, first, 2400, prior_end, 2)
            else:
                await send(session, first, 0, prior_end, 1)
            await session.seal_turn(
                first.identity,
                sealed_content_ranges=(),
                seal_reason="delivery_deadline",
                observed_trailing_silence_ms=0,
            )
            provider.emit(STTProviderTurnTerminal(first.identity, "empty"))
            for _ in range(10):
                await asyncio.sleep(0)
            while session._events.depth:
                await session._events.get()
        await session.begin_turn(req)
        if mode == "late_context":
            await send(session, req, 0, 4800, 1)
            await send(session, req, 0, 4800, 2, context=True)
            await send(session, req, 4800, 7040, 3)
            offset = 0.3
        else:
            offset = session.sent_until / 16000
            if mode == "source_retry":
                await send(session, req, 0, 7040, 1)
            else:
                await send(session, req, 0, 4800, 1, context=True)
                await send(session, req, 4800, 7040, 2)
        uncertain = probabilities(5120, 5440, 0.6, 0.01)
        await session.observe_speaker_evidence(
            frames(0, 5120, 0) + (uncertain,) + frames(5440, 7040, 1)
        )
        event = timed_final(
            req,
            "A. No.",
            [(0, 2, offset, offset + 0.34), (3, 6, offset + 0.34, offset + 0.44)],
            result_start=offset,
            result_end=offset + 0.44,
        )
        provider.emit(event)
        outgoing = await next_text(session.turn_events())
        assert "".join(run.text for run in outgoing.final_speaker_runs) == "A. No."
        if mode == "late_context":
            assert len(outgoing.final_speaker_runs) == 1
            assert outgoing.final_speaker_runs[0].attribution_state == expected
            assert outgoing.source_audio[0].speaker_proof_id is None
            return
        assert [run.text for run in outgoing.final_speaker_runs] == ["A. ", "No."]
        assert outgoing.final_speaker_runs[0].attribution_state == expected
        first_part, second_part = outgoing.source_audio[0].speaker_parts
        assert second_part.speaker_attribution.key.local_speaker_id == "1"
        if mode in ("history_pressure", "missing_history"):
            assert first_part.speaker_proof_id is None
            return
        token = first_part.speaker_proof_id
        assert session._timeline.resolve(token).attribution_state == expected
        claim = session._timeline._proofs[token]
        if mode in ("replayed", "partial"):
            assert [
                (s.normalized_start_sample, s.normalized_end_sample) for s in claim.replayed
            ] == [(0, 2400 if mode == "partial" else 4800)]
            assert claim.ranges[0].normalized_start_sample == (2400 if mode == "partial" else 4800)
        else:
            assert claim.replayed == ()
            assert claim.ranges[0].normalized_start_sample == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_confidence_changes_do_not_create_source_translation_boundaries():
    translator = ControlledProvider()
    harness, trace = harness_for(translator)
    provider = Provider()
    session = StreamingSpeakerSession(
        provider, evidence_view=harness.peer_owner._speaker_identities.evidence
    )
    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(identity, snapshot.settings)
    event = timed_final(
        req,
        "One short phrase.",
        [(0, 3, 0, 0.3), (4, 9, 0.3, 0.6), (10, 17, 0.6, 1)],
        result_start=0,
        result_end=1,
    )
    proof = tuple(
        probabilities(left, left + 160, 0.6 if left < 3200 else 0.9, 0.01)
        for left in range(0, 16000, 160)
    )
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 16000, 1)
        await session.observe_speaker_evidence(proof)
        outgoing = await admit(harness, session, provider, req, snapshot, event)
        assert [item.source_child.raw_text for item in trace.items] == [event.text]
        assert len(translator.calls) == len(harness.peer_runtime.translation_history) == 1
        assert not any(run.source_boundary_before for run in outgoing.final_speaker_runs)
        assert outgoing.source_audio[0].speaker_parts == ()
    finally:
        await session.close()
        await close(harness)


@pytest.mark.asyncio
async def test_later_successful_input_does_not_repair_own_final_overhang():
    from tests.core.test_streaming_speaker import request

    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    event = timed_final(
        req,
        "One two.",
        [(0, 3, 0.1, 0.2), (4, 8, 0.4, 0.5)],
        result_start=0,
        result_end=0.6000003,
    )
    event = replace(event, provenance=replace(event.provenance, sent_until_sample=9600))
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 9600, 1)
        provider.emit(event)
        first = await next_text(session.turn_events())
        await send(session, req, 9600, 11200, 2)
        provider.emit(
            STTProviderTurnTerminal(
                req.identity,
                "final",
                event.text,
                text_authority="authoritative",
                provenance=(event.provenance,),
                included_contributions=(event.contribution,),
            )
        )
        while True:
            terminal = await asyncio.wait_for(anext(session.turn_events()), 1)
            if isinstance(terminal, STTProviderTurnTerminal):
                break
        for output in (first, terminal):
            audio = output.source_audio[0]
            assert (
                audio.source_ranges[0].normalized_start_sample,
                audio.source_ranges[-1].normalized_end_sample,
            ) == (1600, 8000)
        assert first.source_audio[0].speaker_proof_id == terminal.source_audio[0].speaker_proof_id
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True])
async def test_engine_delivery_cancellation_disposes_all_unadopted_part_claims(deferred):
    from puripuly_heart.core.stt.backend import STTSpeakerAudioPart
    from tests.core.test_streaming_speaker import request

    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    view = session._timeline
    tokens = tuple(view.pin((span(left, left + 160),)) for left in (0, 160, 320))
    event = replace(
        update(req, "One two.", release=True),
        source_audio=(
            STTSourceAudio(
                "phrase-1",
                (span(0, 480),),
                (),
                speaker_proof_id=tokens[0],
                speaker_parts=(
                    STTSpeakerAudioPart(
                        0, 4, (span(0, 160),), (), SpeakerAttribution("unavailable"), tokens[1]
                    ),
                    STTSpeakerAudioPart(
                        4, 8, (span(160, 480),), (), SpeakerAttribution("unavailable"), tokens[2]
                    ),
                ),
            ),
        ),
    )

    async def cancelled_sink(_event):
        raise asyncio.CancelledError

    engine = ScopedRecognitionEngine(None, evidence_view=view)
    try:
        if deferred:
            engine.bind_event_sink(cancelled_sink)
            await engine._emit(event)
            with pytest.raises(asyncio.CancelledError):
                await engine._event_dispatch_task
        else:
            engine.event_sink = cancelled_sink
            with pytest.raises(asyncio.CancelledError):
                await engine._emit(event)
        assert all(not view.has_proof(token) for token in tokens)
    finally:
        await engine.close()
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("basis", ["stream_clock", "native_task"])
@pytest.mark.parametrize(
    "case",
    ["same_speaker", "transition", "foreign_gap", "incomplete", "unqualified", "invalid_interval"],
)
async def test_coarse_absent_subword_enclosure_remains_whole_and_requires_complete_mapping(
    basis, case
):
    from puripuly_heart.domain.recognition import NativeAudioCoverage
    from tests.core.test_source_progressive_integration import origin

    translator = ControlledProvider()
    harness, trace = harness_for(translator)
    view = harness.peer_owner._speaker_identities.evidence
    provider = Provider()
    session = StreamingSpeakerSession(provider, evidence_view=view)
    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(identity, snapshot.settings)
    clock_id = "task" if basis == "native_task" else "clock"
    intervals = (
        STTTextTiming("phrase-1", 0, 3, "subword_interval", 0, 100, "ms", clock_id),
        STTTextTiming(
            "phrase-1",
            4 if case == "incomplete" else 3,
            6,
            "subword_interval",
            200,
            float("nan") if case == "invalid_interval" else 300,
            "ms",
            clock_id,
        ),
    )
    event = update(req, "abcdef", release=True)
    event = replace(
        event,
        provenance=replace(
            event.provenance,
            text_timing=intervals,
            audio_clocks=(
                ()
                if basis == "native_task"
                else (STTAudioClock("clock", 16000, 0, case != "unqualified"),)
            ),
        ),
    )
    if basis == "native_task":
        native_origin = origin(event.text)
        native_origin = replace(
            native_origin,
            identity=replace(
                native_origin.identity,
                stream=replace(native_origin.identity.stream, channel="peer"),
            ),
            native=replace(
                native_origin.native,
                timestamp_basis=None if case == "unqualified" else "task_relative_estimated",
                input_sample_rate_hz=16000,
                task_input_samples=4800,
                audio_coverage=(
                    NativeAudioCoverage(0, 4800, 7, 16000, 0, 4800, 16000, 0, 4800, 0, 0.3, False),
                ),
            ),
        )
        event = replace(event, recognition_origins=(native_origin,))
    proof = (
        frames(0, 4800, 0)
        if case == "same_speaker"
        else frames(0, 1600, 0) + (evidence(1600, 2400, ()),) + frames(2400, 4800, 1)
    )
    if case == "foreign_gap":
        proof = frames(0, 1600, 0) + frames(1600, 3200, 1) + frames(3200, 4800, 0)
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 4800, 1)
        view.observe(proof)
        await session.observe_speaker_evidence(proof)
        outgoing = await admit(harness, session, provider, req, snapshot, event)
        assert [item.source_child.raw_text for item in trace.items] == [event.text]
        assert len(translator.calls) == len(harness.peer_runtime.translation_history) == 1
        assert not any(run.source_boundary_before for run in outgoing.final_speaker_runs)
        audio = outgoing.source_audio[0]
        assert audio.speaker_parts == ()
        if case in ("incomplete", "unqualified", "invalid_interval"):
            assert audio.speaker_proof_id is None
            assert trace.items[0].source_child.speaker.speaker_id is None
        else:
            assert audio.speaker_proof_id is not None
            assert (
                audio.source_ranges[0].normalized_start_sample,
                audio.source_ranges[-1].normalized_end_sample,
            ) == (0, 4800)
            assert outgoing.final_speaker_runs[0].attribution_state == (
                "identified" if case == "same_speaker" else "mixed"
            )
    finally:
        await session.close()
        await close(harness)
