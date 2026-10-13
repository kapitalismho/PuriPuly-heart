from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.speaker.contracts import ProbabilityFrame, SourceMapPiece, SpeakerEvidence
from puripuly_heart.core.stt.backend import (
    STTAudioClock,
    STTContributionConsumptionLedger,
    STTNativeProvenance,
    STTProviderAudioBoundary,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSourceCapabilities,
    STTSourceRelease,
    STTTextContribution,
    STTTextTiming,
)
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from puripuly_heart.domain.models import FinalSpeakerRun


def span(start, end):
    return AudioCaptureSpan(
        7,
        start,
        16000,
        start,
        end,
        start / 16000,
        end / 16000,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=start,
        normalized_end_sample=end,
    )


def request():
    settings = AudioSegmentSettingsSnapshot(
        "deepgram", (), (), "manual", "en", ("en",), 16000, 0.5, 200, 500
    )
    identity = STTProviderTurnIdentity(
        AudioSegmentIdentity(1, 1, uuid4(), 7), "connection", "request"
    )
    return STTProviderTurnRequest(identity, settings, display_group_id=uuid4())


def evidence(start, end, slot, *, epoch=11, ready=3.0):
    frame = ProbabilityFrame(
        start, end, tuple(0.9 if index == slot else 0.01 for index in range(8))
    )
    return SpeakerEvidence(
        "capture",
        epoch,
        ready,
        (SourceMapPiece(start, end, span(start, end), "original"),),
        end,
        end,
        frame,
    )


class Provider:
    def __init__(self):
        self.events = STTProviderEventBuffer()
        self.sent = []
        self.sealed = []
        self.observer = None

    def set_source_mapping_observer(self, observer):
        self.observer = observer

    async def begin_turn(self, request):
        self.request = request

    async def send_turn_audio(self, identity, pcm16le, **kwargs):
        self.sent.append((identity, pcm16le, kwargs))

    async def seal_turn(self, identity, **kwargs):
        self.sealed.append((identity, kwargs))

    async def abort_turn(self, identity, **kwargs):
        pass

    async def turn_events(self):
        async for event in self.events.events():
            yield event

    async def stop(self):
        self.events.close()

    async def close(self):
        self.events.close()


async def send(session, req, start, end, seq, *, context=False):
    await session.send_turn_audio(
        req.identity,
        bytes((end - start) * 2),
        payload_sequence=seq,
        source_ranges=(span(start, end),),
        context_only=context,
    )


def update(req, text, *, qualified=True, policy="timed", release=False, semantics="word_interval"):
    provenance = STTNativeProvenance(
        native_event_id="receipt-1",
        capabilities=STTSourceCapabilities("actual-deepgram", "nova-3", policy),
        text_timing=(
            STTTextTiming("phrase-1", 0, len(text), semantics, 0, 0.1, "s", "clock", True),
        ),
        audio_clocks=(STTAudioClock("clock", 16000, 0, qualified),),
        native_audio_progress=(("audio_processed_ms", 100.0),),
        sent_until_sample=3200,
    )
    return STTProviderTurnUpdate(
        req.identity,
        1,
        "stable",
        "append",
        text,
        final_speaker_runs=(FinalSpeakerRun(text, "native-wrong", "native"),),
        provenance=provenance,
        contribution=STTTextContribution("phrase-1", 0, len(text)),
        source_release=STTSourceRelease("provider-end", 0, len(text), "end") if release else None,
    )


async def next_text(events):
    while True:
        event = await asyncio.wait_for(anext(events), 1)
        if isinstance(event, STTProviderTurnUpdate):
            return event


@pytest.mark.asyncio
async def test_supported_prefix_releases_before_end_and_successor_progresses():
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 3200, 1)
        await session.observe_speaker_evidence((evidence(0, 1600, 0), evidence(1600, 3200, 1)))
        assert session.audio_boundaries[0][1].normalized_sample == 1600
        assert not provider.sealed
        provider.events.put(update(req, "短い。 "))
        events = session.turn_events()
        stable = await next_text(events)
        early = await next_text(events)
        assert stable.source_release is None
        assert early.source_release.reason == "boundary"
        assert early.text == "短い。 "
        assert early.final_speaker_runs[0].speaker_id == "0"
        assert early.final_speaker_runs[0].source == "nemotron"
        assert not provider.sealed
        ledger = STTContributionConsumptionLedger()
        assert ledger.consume(early) == "短い。 "
        await send(session, req, 3200, 4800, 2)
        assert session.sent_until == 4800
        assert len(provider.sent) == 2
        provider.events.put(
            STTProviderTurnTerminal(
                req.identity, "final", "短い。 残り", text_authority="authoritative"
            )
        )
        terminal = await asyncio.wait_for(anext(events), 1)
        assert ledger.consume(terminal) == "残り"
        assert terminal.provenance[0].native_audio_progress == (("audio_processed_ms", 100.0),)
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("assembly", ["append", "replace"])
async def test_consumed_text_preserves_later_audio_boundary_and_pending_successor(assembly):
    from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
    )

    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )
    provider = Provider()
    session = StreamingSpeakerSession(provider)
    harness, trace = harness_for(ControlledProvider())
    normalizer = STTScopedTurnNormalizer(identity)
    events = session.turn_events()
    first_text, successor_text = "Alpha. ", "Beta. "
    text = first_text + successor_text
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 3200, 1)
        await session.observe_speaker_evidence((evidence(0, 1600, 0), evidence(1600, 3200, 1)))
        provider.events.put(update(req, first_text))
        normalizer.apply_update(await next_text(events))
        first_release = normalizer.apply_update(await next_text(events))
        await harness.peer_owner.handle_provider_turn_update(snapshot, first_release)
        await finish(harness)
        first = trace.items[0].source_child
        assert first.raw_text == first_text
        assert first.speaker.speaker_id == "0"

        await session.observe_speaker_evidence((evidence(3200, 4800, 2, ready=4.0),))
        boundary = await asyncio.wait_for(anext(events), 1)
        assert isinstance(boundary, STTProviderAudioBoundary)
        assert boundary.boundary.normalized_sample == 3200
        assert boundary.boundary.observed_at_monotonic_s == 4.0

        event = update(req, successor_text if assembly == "append" else text)
        provider.events.put(
            replace(
                event,
                sequence=2,
                assembly=assembly,
                contribution=STTTextContribution("phrase-2", len(first_text), len(text)),
                provenance=replace(
                    event.provenance,
                    native_event_id="receipt-2",
                    text_timing=(
                        STTTextTiming(
                            "phrase-2",
                            len(first_text),
                            len(text),
                            "word_interval",
                            0.2,
                            0.3,
                            clock_id="clock",
                            complete_boundary=True,
                        ),
                    ),
                ),
            )
        )
        pending = normalizer.apply_update(await next_text(events))
        assert pending.text == text
        assert pending.source_release is None
        assert pending.final_speaker_runs[-1].attribution_state == "unavailable"
        await send(session, req, 3200, 4000, 2)
        await session.observe_speaker_evidence((evidence(4800, 6400, 3, ready=5.0),))
        await send(session, req, 4000, 4800, 3)
        boundary = await asyncio.wait_for(anext(events), 1)
        assert isinstance(boundary, STTProviderAudioBoundary)
        assert boundary.boundary.normalized_sample == 4800
        assert boundary.boundary.observed_at_monotonic_s == 5.0
        successor = normalizer.apply_update(await next_text(events))
        assert (successor.source_release.text_start, successor.source_release.text_end) == (
            len(first_text),
            len(text),
        )
        assert successor.final_speaker_runs[-1].speaker_id == "2"
        assert successor.source_audio[-1].source_ranges == (span(3200, 4000), span(4000, 4800))
        await harness.peer_owner.handle_provider_turn_update(snapshot, successor)
        await harness.peer_owner.handle_provider_turn_update(snapshot, successor)
        await finish(harness)
        assert [item.source_child.raw_text for item in trace.items] == [
            first_text,
            successor_text,
        ]
        assert [item.source_child.speaker.speaker_id for item in trace.items] == ["0", "2"]

        provider.events.put(
            STTProviderTurnTerminal(
                identity, "final", text + "Tail!", text_authority="authoritative"
            )
        )
        terminal = normalizer.apply_terminal(await asyncio.wait_for(anext(events), 1))
        assert "".join(run.text for run in terminal.final_speaker_runs) == text + "Tail!"
        receipt = AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 6.0)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await finish(harness)
        assert trace.items[0].source_child is first
        assert [item.source_child.raw_text for item in trace.items] == [
            first_text,
            successor_text,
            "Tail!",
        ]
        assert len(harness.peer_runtime.translation_history) == 3
        assert snapshot.state == "open"
        assert not provider.sealed
    finally:
        await session.close()
        await events.aclose()
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("semantics,qualified", [("word_interval", False), ("token_point", True)])
async def test_unknown_clock_or_point_never_becomes_interval(semantics, qualified):
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 3200, 1)
        await session.observe_speaker_evidence((evidence(0, 1600, 0), evidence(1600, 3200, 1)))
        provider.events.put(
            update(req, "No.", qualified=qualified, semantics=semantics, release=True)
        )
        result = await next_text(session.turn_events())
        assert result.source_release.reason == "end"
        assert result.final_speaker_runs[0].speaker_id is None
        assert result.final_speaker_runs[0].source == "nemotron"
        assert STTContributionConsumptionLedger().consume(result) == "No."
        assert not provider.sealed
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("context_slot,expected", [(0, "identified"), (1, "mixed")])
async def test_untimed_considers_all_replayed_context(context_slot, expected):
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 800, 1, context=True)
        await send(session, req, 800, 1600, 2)
        await session.observe_speaker_evidence(
            (evidence(0, 800, context_slot), evidence(800, 1600, 0))
        )
        provider.events.put(update(req, "No.", policy="whole_span", release=True))
        result = await next_text(session.turn_events())
        assert result.source_audio[0].context_ranges == (span(0, 800),)
        assert result.source_audio[0].speaker_attribution.state == expected
        assert result.final_speaker_runs[0].attribution_state == expected
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unknown_actual_backend_does_not_inherit_request_purity():
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 1600, 1)
        await session.observe_speaker_evidence((evidence(0, 1600, 0),))
        provider.events.put(update(req, "No.", policy="unknown", release=True))
        result = await next_text(session.turn_events())
        assert result.source_audio[0].speaker_attribution.state == "unavailable"
        assert result.final_speaker_runs[0].speaker_id is None
    finally:
        await session.close()


def test_raw_text_ranges_distinct_repeats_and_terminal_correction_conserve():
    req = request()
    normalizer = STTScopedTurnNormalizer(req.identity)
    ledger = STTContributionConsumptionLedger()
    first = normalizer.apply_update(update(req, " no ", release=True))
    assert ledger.consume(first) == " no "
    second = normalizer.apply_update(
        STTProviderTurnUpdate(
            req.identity,
            2,
            "stable",
            "append",
            "no ",
            contribution=STTTextContribution("second", 4, 7),
            source_release=STTSourceRelease("second", 4, 7, "end"),
        )
    )
    assert second.included_contributions[0].contribution_id == "phrase-1"
    assert ledger.consume(second) == "no "
    terminal = normalizer.apply_terminal(
        STTProviderTurnTerminal(
            req.identity, "final", " no no corrected", text_authority="authoritative"
        )
    )
    assert ledger.consume(terminal) == "corrected"


def test_boundary_pressure_stays_bounded_without_retiring_source():
    req = request()
    buffer = STTProviderEventBuffer(max_events=1)
    from puripuly_heart.core.stt.backend import STTAudioBoundary

    boundary = STTProviderAudioBoundary(req.identity, STTAudioBoundary("b", 7, 100, 3.0, "speaker"))
    assert buffer.put(boundary)
    assert not buffer.put(boundary)
    assert buffer.depth == 1


@pytest.mark.asyncio
async def test_native_ids_cannot_change_actual_admitted_source_units():
    from puripuly_heart.domain.recognition import NativeTranscriptionEvidence

    req = request()

    async def admitted(native_id):
        provider = Provider()
        session = StreamingSpeakerSession(provider)
        try:
            await session.begin_turn(req)
            await send(session, req, 0, 3200, 1)
            await session.observe_speaker_evidence((evidence(0, 1600, 0), evidence(1600, 3200, 1)))
            event = update(req, "No.", release=True)
            event = replace(
                event,
                final_speaker_runs=(FinalSpeakerRun("No.", native_id, "native"),),
                provenance=replace(
                    event.provenance,
                    transcription=NativeTranscriptionEvidence(speaker_label=native_id),
                ),
            )
            provider.events.put(event)
            result = await next_text(session.turn_events())
            pieces = STTContributionConsumptionLedger().consume_parts(result)
            return (
                tuple(
                    (piece.contribution_id, piece.text_start, piece.text_end, piece.text)
                    for piece in pieces
                ),
                result.final_speaker_runs,
                tuple(
                    (audio.source_ranges, audio.context_ranges, audio.speaker_attribution)
                    for audio in result.source_audio
                ),
            )
        finally:
            await session.close()

    assert await admitted("native-A") == await admitted("contradictory-native-B")


@pytest.mark.asyncio
async def test_explicit_clock_reset_and_synthetic_input_do_not_guess_offsets():
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        provider.observer(800)
        await send(session, req, 0, 3200, 1)
        await session.observe_speaker_evidence((evidence(0, 1600, 0), evidence(1600, 3200, 1)))
        event = update(req, "No.", release=True)
        event = replace(
            event,
            provenance=replace(
                event.provenance,
                audio_clocks=(STTAudioClock("task-reset", 16000, 800, True),),
                text_timing=(replace(event.provenance.text_timing[0], clock_id="task-reset"),),
            ),
        )
        provider.events.put(event)
        result = await next_text(session.turn_events())
        assert session.sent_until == 4000
        assert result.final_speaker_runs[0].speaker_id == "0"
        assert result.source_audio[0].source_ranges == (span(0, 1600),)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_rejected_audio_never_advances_actual_sent_cursor():
    class RejectingProvider(Provider):
        async def send_turn_audio(self, *args, **kwargs):
            raise RuntimeError("write rejected")

    provider, req = RejectingProvider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        with pytest.raises(RuntimeError, match="write rejected"):
            await send(session, req, 0, 1600, 1)
        assert session.sent_until == 0
        assert tuple(session._maps) == ()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_independent_stream_protocol_origin_and_request_reclamation():
    from puripuly_heart.core.stt.backend import (
        STTIndependentRecognitionSession,
        STTProviderInputTerminal,
        STTRecognitionUnit,
    )
    from puripuly_heart.core.stt.recognition_units import recognition_transcript
    from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity

    class IndependentProvider(Provider):
        accepts_stream_input = True
        independent_recognition_units = True

        async def begin_stream(self, stream):
            self.stream = stream

        async def send_stream_audio(self, pcm16le, *, source_ranges):
            self.stream_payload = (pcm16le, source_ranges)

        async def end_stream(self, *, reason):
            self.stream_end = reason

        def recognition_source_covers(self, ranges):
            return ranges == (span(0, 1600),)

    provider, req = IndependentProvider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        assert isinstance(session, STTIndependentRecognitionSession)
        assert session.accepts_stream_input and session.independent_recognition_units
        stream = RecognitionStreamIdentity("peer", 1, 7, "connection", ())
        await session.begin_stream(stream)
        await session.send_stream_audio(bytes(3200), source_ranges=(span(0, 1600),))
        assert session.recognition_source_covers((span(0, 1600),))
        await session.begin_turn(req)
        provider.events.put(STTProviderInputTerminal(req.identity, "submitted", "peer"))
        unit = STTRecognitionUnit(
            RecognitionUnitIdentity(stream, uuid4(), 1),
            "Independent",
            STTNativeProvenance(
                capabilities=STTSourceCapabilities("gemini", "actual-model", "unknown")
            ),
        )
        provider.events.put(unit)
        events = session.turn_events()
        assert isinstance(await anext(events), STTProviderInputTerminal)
        result = await anext(events)
        assert result.identity == unit.identity
        assert session._turns == {}
        transcript = recognition_transcript(result, created_at=4.0, publication_order=1)
        assert transcript.recognition_origins[0].native_provenance == (unit.provenance,)
        assert transcript.recognition_origins[0].source_ranges == ()
        assert transcript.final_speaker_runs == ()
        await session.end_stream(reason="source_eof")
        assert provider.stream_end == "source_eof"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_audio_boundary_callback_records_before_text_and_fences_stale_scope():
    from types import SimpleNamespace

    from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
        TranslationChannelOwnerCallbacks,
    )
    from puripuly_heart.core.stt.backend import STTAudioBoundary
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        harness_for,
        source_fixture,
    )

    snapshot, identity = source_fixture()
    harness, trace = harness_for(ControlledProvider())
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_peer(harness.peer_owner)
    callbacks.bind_peer_capture(
        SimpleNamespace(
            provider_source_snapshot=lambda value: snapshot if value == identity else None
        )
    )
    current = True
    harness.peer_owner.local_asr_runtime = SimpleNamespace(
        is_current_recognition_stream=lambda channel, stream: current
    )
    try:
        boundary = STTAudioBoundary("before-text", 7, 8000, 0.6, "speaker", "11")
        await callbacks.peer_event_handler(STTProviderAudioBoundary(identity, boundary))
        assert harness.peer_owner._stt_consumption.boundaries(identity) == (boundary,)
        assert trace.items == harness.peer_runtime.translation_history == []
        current = False
        await callbacks.peer_event_handler(
            STTProviderAudioBoundary(identity, replace(boundary, boundary_id="stale"))
        )
        assert harness.peer_owner._stt_consumption.boundaries(identity) == (boundary,)
    finally:
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("middle_slot,expected_boundary", [(None, 3200), (2, 1600)])
async def test_audio_boundary_survives_mapped_quiet_without_text(middle_slot, expected_boundary):
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 4800, 1)
        await session.observe_speaker_evidence(
            (
                evidence(0, 1600, 0),
                evidence(1600, 3200, middle_slot),
                evidence(3200, 4800, 1),
            )
        )
        assert session.audio_boundaries[0][1].normalized_sample == expected_boundary
        assert session.audio_boundaries[0][1].observed_at_monotonic_s == 3.0
        assert not provider.sealed
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_same_speaker_word_pieces_create_one_complete_source_not_word_jobs():
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
    )

    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )
    provider = Provider()
    session = StreamingSpeakerSession(provider)
    harness, trace = harness_for(ControlledProvider())
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 3200, 1)
        await session.observe_speaker_evidence((evidence(0, 1600, 0), evidence(1600, 3200, 1)))
        event = update(req, "No. yes.", release=True)
        event = replace(
            event,
            provenance=replace(
                event.provenance,
                text_timing=(
                    STTTextTiming("phrase-1", 0, 3, "word_interval", 0, 0.04, clock_id="clock"),
                    STTTextTiming(
                        "phrase-1",
                        4,
                        8,
                        "word_interval",
                        0.05,
                        0.09,
                        clock_id="clock",
                        complete_boundary=True,
                    ),
                ),
            ),
        )
        provider.events.put(event)
        result = await next_text(session.turn_events())
        result = STTScopedTurnNormalizer(identity).apply_update(result)
        await harness.peer_owner.handle_provider_turn_update(snapshot, result)
        await finish(harness)
        assert [item.source_child.raw_text for item in trace.items] == ["No. yes."]
        assert trace.items[0].source_child.speaker.speaker_id == "0"
        assert trace.items[0].source_child.speaker.source == "nemotron"
        assert trace.items[0].source_child.origins[0].native_provenance == (event.provenance,)
        assert len(harness.peer_runtime.translation_history) == 1
    finally:
        await session.close()
        await close(harness)


async def replay_captured_fragmentation(*, complete_words=True):
    from puripuly_heart.domain.recognition import NativeTranscriptionEvidence
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
    )

    fixture = json.loads(
        (Path(__file__).with_name("fixtures") / "issue216_streaming_fragmentation.json").read_text()
    )
    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )
    provider = Provider()
    session = StreamingSpeakerSession(provider)
    harness, trace = harness_for(ControlledProvider())
    events = session.turn_events()
    normalizer = STTScopedTurnNormalizer(identity)
    evidence_cursor = sent = sequence = 0
    initial = fixture["source_capture_offset"]
    try:
        await session.begin_turn(req)
        sequence += 1
        await send(
            session, req, initial, initial + fixture["context_samples"], sequence, context=True
        )
        sent = fixture["context_samples"]
        for captured in fixture["updates"]:
            sequence += 1
            await send(session, req, initial + sent, initial + captured["sent_until"], sequence)
            sent = captured["sent_until"]
            observed = []
            while evidence_cursor < len(fixture["evidence"]):
                item = fixture["evidence"][evidence_cursor]
                if item["delivered_at"] > captured["at"]:
                    break
                data = item["frame"]
                frame = (
                    ProbabilityFrame(
                        data["start_sample"],
                        data["end_sample"],
                        tuple(data["probabilities"]),
                        tuple(data["active_slots"]),
                    )
                    if data is not None
                    else None
                )
                maps = tuple(
                    SourceMapPiece(
                        piece["compact_start"],
                        piece["compact_end"],
                        span(*piece["range"]) if piece["range"] is not None else None,
                        piece["kind"],
                    )
                    for piece in item["maps"]
                )
                observed.append(
                    SpeakerEvidence(
                        "sanitized-actual-capture",
                        item["epoch"],
                        item["ready"],
                        maps,
                        item["processed"],
                        item["immutable"],
                        frame,
                        item["state"],
                        item["reason"],
                    )
                )
                evidence_cursor += 1
            await session.observe_speaker_evidence(tuple(observed))
            transcription = captured["transcription"]
            provenance = STTNativeProvenance(
                native_event_id=f"captured-{captured['sequence']}",
                barrier=captured["barrier"],
                from_finalize=captured["from_finalize"],
                transcription=NativeTranscriptionEvidence(
                    transcription["finished"],
                    transcription["language_code"],
                    transcription["speaker_label"],
                    tuple(tuple(word) for word in transcription["words"]),
                ),
                capabilities=STTSourceCapabilities(**captured["capabilities"]),
                text_timing=tuple(
                    replace(
                        STTTextTiming(**timing),
                        complete_boundary=(
                            timing["complete_boundary"]
                            or (
                                complete_words
                                and timing["semantics"] == "word_interval"
                                and any(
                                    captured["text"][
                                        timing["text_start"]
                                        - captured["contribution"]["text_start"] : timing[
                                            "text_end"
                                        ]
                                        - captured["contribution"]["text_start"]
                                    ]
                                    == word[0]
                                    and timing["raw_start"] == float(word[1])
                                    and timing["raw_end"] == float(word[2])
                                    for word in transcription["words"]
                                )
                            )
                        ),
                    )
                    for timing in captured["timings"]
                ),
                audio_clocks=tuple(STTAudioClock(**clock) for clock in captured["clocks"]),
                sent_until_sample=sent,
                native_audio_progress=tuple(
                    tuple(value) for value in captured["native_audio_progress"]
                ),
                control_id=captured["control_id"],
                control_kind=captured["control_kind"],
                control_sent_sample=captured["control_sent_sample"],
                native_model_metadata=tuple(tuple(value) for value in captured["model_metadata"]),
            )
            provider.events.put(
                STTProviderTurnUpdate(
                    identity,
                    captured["sequence"],
                    "stable",
                    "append",
                    captured["text"],
                    provenance=provenance,
                    contribution=STTTextContribution(**captured["contribution"]),
                    source_release=STTSourceRelease(**captured["release"]),
                )
            )
            outgoing = normalizer.apply_update(await next_text(events))
            await harness.peer_owner.handle_provider_turn_update(snapshot, outgoing)
            await finish(harness)
        sources = [item.source_child for item in trace.items]
        assert "".join(source.raw_text for source in sources) == "".join(
            item["text"] for item in fixture["updates"]
        )
        assert len(sources) == len(harness.peer_runtime.translation_history)
        if complete_words:
            assert [source.raw_text for source in sources[:2]] == [
                "Washed the detritus from his hair. ",
                "Gentlemen of the press.",
            ]
            assert sources[1].speaker.speaker_id == "1"
            assert [source.origins[0].contribution_range for source in sources[:2]] == [
                (0, 35),
                (35, 58),
            ]
            assert [source.raw_text for source in sources[2:]] == [
                "He washed the detritus from his",
            ]
            assert sources[2].speaker.attribution_state == "unavailable"
        else:
            assert [source.raw_text for source in sources] == [
                item["text"] for item in fixture["updates"]
            ]
            assert len(sources) == 2
            assert sources[0].speaker.attribution_state == "mixed"
            assert all(source.speaker.speaker_id is None for source in sources)
        assert not provider.sealed
        boundary = next(
            item
            for _, item in session.audio_boundaries
            if item.transition_start_sample == 42048 and item.transition_end_sample == 49888
        )
        assert boundary.normalized_sample == 49888
        assert boundary.observed_at_monotonic_s == 82922.5787792
        await send(session, req, initial + sent, initial + sent + 512, sequence + 1)
        return {
            "raw_sources": [source.raw_text for source in sources],
            "attribution": [source.speaker.attribution_state for source in sources],
            "speaker_ids": [source.speaker.speaker_id for source in sources],
            "source_children": len(sources),
            "context_entries": len(harness.peer_runtime.translation_history),
            "transition_interval": [
                boundary.transition_start_sample,
                boundary.transition_end_sample,
            ],
            "safe_successor_position": boundary.normalized_sample,
            "source_before_enclosing_end": snapshot.state == "open",
            "positive_rich_preceding_source": complete_words,
            "successor_progress_samples": 512,
            "optional_controls": len(provider.sealed),
            "external_connections": 0,
        }
    finally:
        await session.close()
        await events.aclose()
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_words", [False, True])
async def test_actual_provider_and_vulkan_trace_preserves_boundary_owned_source_groups(
    complete_words,
):
    await replay_captured_fragmentation(complete_words=complete_words)


@pytest.mark.asyncio
@pytest.mark.parametrize("separation", ["loss", "epoch", "capture", "discontinuity"])
async def test_established_transition_never_bridges_invalid_coverage(separation):
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 4800, 1)
        first = evidence(0, 1600, 0)
        gray = replace(
            evidence(1600, 3200, None),
            frame=ProbabilityFrame(
                1600,
                3200,
                (0.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
            ),
        )
        successor = evidence(3200, 4800, 1)
        if separation == "loss":
            gray = replace(gray, frame=None, reason="lost_speech")
        elif separation == "epoch":
            successor = replace(successor, speaker_epoch=12)
        elif separation == "capture":
            piece = successor.source_map[0]
            successor = replace(
                successor,
                source_map=(replace(piece, capture=replace(piece.capture, capture_epoch=8)),),
            )
        else:
            piece = successor.source_map[0]
            successor = replace(
                successor,
                source_map=(
                    replace(
                        piece, capture=replace(piece.capture, discontinuity_before="capture_loss")
                    ),
                ),
            )
        await session.observe_speaker_evidence((first, gray, successor))
        assert not session.audio_boundaries
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_uncertain_transition_is_retained_but_cannot_release_identified_source():
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 4800, 1)
        gray = replace(
            evidence(1600, 3200, None),
            frame=ProbabilityFrame(
                1600,
                3200,
                (0.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
            ),
        )
        await session.observe_speaker_evidence(
            (evidence(0, 1600, 0), gray, evidence(3200, 4800, 1))
        )
        boundary = session.audio_boundaries[0][1]
        assert (boundary.transition_start_sample, boundary.transition_end_sample) == (1600, 3200)
        event = update(req, "A complete result.")
        event = replace(
            event,
            provenance=replace(
                event.provenance,
                text_timing=(replace(event.provenance.text_timing[0], raw_end=0.15),),
            ),
        )
        provider.events.put(event)
        result = await next_text(session.turn_events())
        assert result.source_release is None
        assert result.final_speaker_runs[0].speaker_id is None
        assert len(session._timeline.frames) <= session._timeline.MAX_FRAMES
    finally:
        await session.close()


def test_boundary_transition_interval_is_paired_and_matches_safe_position():
    from puripuly_heart.core.stt.backend import STTAudioBoundary

    for start, end, position in [(1, None, 2), (None, 2, 2), (3, 2, 2), (-1, 2, 2), (1, 2, 3)]:
        with pytest.raises(ValueError):
            STTAudioBoundary("invalid", 7, position, 3.0, "speaker", "11", start, end)


@pytest.mark.asyncio
async def test_bounded_transition_history_cannot_bridge_evicted_predecessor():
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        count = session._timeline.MAX_FRAMES + 32
        await send(session, req, 0, (count + 2) * 16, 1)
        await session.observe_speaker_evidence(
            tuple(
                [evidence(0, 16, 0)]
                + [evidence(index * 16, (index + 1) * 16, None) for index in range(1, count + 1)]
                + [evidence((count + 1) * 16, (count + 2) * 16, 1)]
            )
        )
        assert not session.audio_boundaries
        assert len(session._timeline.frames) == session._timeline.MAX_FRAMES
        assert not provider.sealed
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("supported_text_cut", [False, True])
async def test_internal_split_requires_both_established_audio_and_complete_text_boundary(
    supported_text_cut,
):
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
    )

    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )
    provider = Provider()
    session = StreamingSpeakerSession(provider)
    harness, trace = harness_for(ControlledProvider())
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 4800, 1)
        await session.observe_speaker_evidence(
            (
                evidence(0, 1600, 0),
                evidence(1600, 2400, None),
                evidence(2400, 4800, 1),
            )
        )
        event = update(req, "No. Yes.", release=True)
        event = replace(
            event,
            provenance=replace(
                event.provenance,
                text_timing=(
                    STTTextTiming(
                        "phrase-1",
                        0,
                        8,
                        "result_span",
                        0,
                        0.2,
                        clock_id="clock",
                        complete_boundary=True,
                    ),
                    STTTextTiming(
                        "phrase-1",
                        0,
                        3,
                        "word_interval",
                        0,
                        0.05,
                        clock_id="clock",
                        complete_boundary=supported_text_cut,
                    ),
                    STTTextTiming("phrase-1", 4, 8, "word_interval", 0.15, 0.2, clock_id="clock"),
                ),
            ),
        )
        provider.events.put(event)
        outgoing = STTScopedTurnNormalizer(identity).apply_update(
            await next_text(session.turn_events())
        )
        await harness.peer_owner.handle_provider_turn_update(snapshot, outgoing)
        await finish(harness)
        sources = [item.source_child for item in trace.items]
        assert "".join(source.raw_text for source in sources) == "No. Yes."
        if supported_text_cut:
            assert [source.raw_text for source in sources] == ["No. ", "Yes."]
            assert [source.speaker.speaker_id for source in sources] == ["0", "1"]
        else:
            assert [source.raw_text for source in sources] == ["No. Yes."]
            assert sources[0].speaker.attribution_state == "mixed"
    finally:
        await session.close()
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("coarse_state", ["mixed", "uncertain"])
async def test_fine_complete_word_prefix_preserves_crossing_group_and_terminal_residual(
    coarse_state,
):
    from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
    )

    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )
    provider = Provider()
    session = StreamingSpeakerSession(provider)
    harness, trace = harness_for(ControlledProvider())
    normalizer = STTScopedTurnNormalizer(identity)
    events = session.turn_events()
    text = "  Alpha, again!\t crossing!  Beta, later.  "
    words = [
        ("Alpha,", 0.0, 0.04),
        ("again!", 0.04, 0.08),
        ("crossing!", 0.09, 0.17),
        ("Beta,", 0.20, 0.24),
        ("later.", 0.24, 0.28),
    ]
    prefix_end = text.index("crossing!")
    successor_start = text.index("Beta,")
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 4800, 1)
        gray = replace(
            evidence(1600, 2400, None),
            frame=ProbabilityFrame(1600, 2400, (0.4,) + (0.01,) * 7),
        )
        await session.observe_speaker_evidence(
            (evidence(0, 1600, 0), gray, evidence(2400, 4800, 1))
        )
        event = update(req, text)
        event = replace(
            event,
            provenance=replace(
                event.provenance,
                text_timing=(
                    STTTextTiming(
                        "phrase-1",
                        0,
                        len(text),
                        "result_span",
                        0,
                        0.30 if coarse_state == "mixed" else 0.15,
                        clock_id="clock",
                        complete_boundary=True,
                    ),
                    *(
                        STTTextTiming(
                            "phrase-1",
                            text.index(word),
                            text.index(word) + len(word),
                            "word_interval",
                            start,
                            end,
                            clock_id="clock",
                            complete_boundary=True,
                        )
                        for word, start, end in words
                    ),
                ),
            ),
        )
        provider.events.put(event)
        available = normalizer.apply_update(await next_text(events))
        assert available.source_release is None
        early = normalizer.apply_update(await next_text(events))
        assert early.source_release.text_end == prefix_end
        await harness.peer_owner.handle_provider_turn_update(snapshot, early)
        await finish(harness)
        first = trace.items[0].source_child
        assert first.raw_text == text[:prefix_end]
        assert first.speaker.speaker_id == "0"
        assert first.origins[0].contribution_range == (0, prefix_end)
        await send(session, req, session.sent_until, session.sent_until + 512, 3)
        provider.events.put(
            replace(
                event,
                sequence=2,
                assembly="replace",
                provenance=replace(event.provenance, native_event_id="receipt-2"),
                source_release=STTSourceRelease("native-final", 0, len(text), "end"),
            )
        )
        remainder = normalizer.apply_update(await next_text(events))
        await harness.peer_owner.handle_provider_turn_update(snapshot, remainder)
        await finish(harness)
        sources = [item.source_child for item in trace.items]
        assert [source.raw_text for source in sources] == [
            text[:prefix_end],
            text[prefix_end:successor_start],
            text[successor_start:],
        ]
        assert [source.speaker.speaker_id for source in sources] == ["0", None, "1"]
        assert sources[1].speaker.attribution_state in {"mixed", "uncertain"}
        assert [source.origins[0].contribution_range for source in sources] == [
            (0, prefix_end),
            (prefix_end, successor_start),
            (successor_start, len(text)),
        ]
        provider.events.put(
            STTProviderTurnTerminal(
                identity, "final", text + "Tail!", text_authority="authoritative"
            )
        )
        terminal = normalizer.apply_terminal(await asyncio.wait_for(anext(events), 1))
        receipt = AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 11)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await finish(harness)
        assert trace.items[0].source_child == first
        assert trace.items[-1].source_child.raw_text == "Tail!"
        assert "".join(item.source_child.raw_text for item in trace.items) == text + "Tail!"
        assert len(trace.items) == len(harness.peer_runtime.translation_history)
        assert snapshot.state == "open"
        assert not provider.sealed
    finally:
        await session.close()
        await events.aclose()
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("qualified,transition", [(True, False), (False, True)])
async def test_lexical_cuts_without_qualified_audio_partition_do_not_create_word_jobs(
    qualified,
    transition,
):
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 4800, 1)
        gray = replace(
            evidence(1600, 2400, None),
            frame=ProbabilityFrame(1600, 2400, (0.4,) + (0.01,) * 7),
        )
        await session.observe_speaker_evidence(
            (evidence(0, 1600, 0), gray, evidence(2400, 4800, 1 if transition else 0))
        )
        event = update(req, "One two three.", qualified=qualified, release=True)
        event = replace(
            event,
            provenance=replace(
                event.provenance,
                text_timing=(
                    STTTextTiming(
                        "phrase-1",
                        0,
                        3,
                        "word_interval",
                        0,
                        0.1,
                        clock_id="clock",
                        complete_boundary=True,
                    ),
                    STTTextTiming(
                        "phrase-1",
                        4,
                        7,
                        "word_interval",
                        0.1,
                        0.15,
                        clock_id="clock",
                        complete_boundary=True,
                    ),
                    STTTextTiming(
                        "phrase-1",
                        8,
                        14,
                        "word_interval",
                        0.15,
                        0.3,
                        clock_id="clock",
                        complete_boundary=True,
                    ),
                ),
            ),
        )
        provider.events.put(event)
        result = await next_text(session.turn_events())
        assert len(result.final_speaker_runs) == 1
        assert result.final_speaker_runs[0].text == event.text
        assert result.final_speaker_runs[0].speaker_id == ("0" if qualified else None)
        assert STTContributionConsumptionLedger().consume(result) == event.text
        assert not provider.sealed
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,uncertain_edge,qualified,expected",
    [
        ("Gentlemen of the press.", "leading", True, "1"),
        ("Washed the detritus from his hair.", "trailing", True, "0"),
        ("Gentlemen of the press.", "long_leading", True, None),
        ("Gentlemen of the press.", "leading", False, None),
    ],
)
async def test_final_phrase_confidence_edges_do_not_fragment_source_or_translation(
    text, uncertain_edge, qualified, expected
):
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
    )

    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(identity, snapshot.settings, display_group_id=snapshot.display_group_id)
    provider = Provider()
    session = StreamingSpeakerSession(provider)
    translator = ControlledProvider()
    harness, trace = harness_for(translator)
    slot = 0 if uncertain_edge == "trailing" else 1
    edge = 4000 if uncertain_edge == "long_leading" else 800
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 32000, 1)
        gray_start, gray_end = (32000 - edge, 32000) if uncertain_edge == "trailing" else (0, edge)
        gray = replace(
            evidence(gray_start, gray_end, None),
            frame=ProbabilityFrame(gray_start, gray_end, (0.4,) + (0.01,) * 7),
        )
        clear = evidence(0, 32000 - edge, slot) if uncertain_edge == "trailing" else evidence(edge, 32000, slot)
        await session.observe_speaker_evidence((clear, gray) if uncertain_edge == "trailing" else (gray, clear))
        words = text.split()
        positions = []
        cursor = 0
        for index, word in enumerate(words):
            start = text.index(word, cursor)
            end = start + len(word)
            positions.append(
                STTTextTiming(
                    "phrase-1", start, end, "word_interval",
                    2 * index / len(words), 2 * (index + 1) / len(words),
                    clock_id="clock", complete_boundary=True, estimated=True,
                )
            )
            cursor = end
        event = update(req, text, qualified=qualified, release=True)
        event = replace(event, provenance=replace(event.provenance, text_timing=tuple(positions)))
        provider.events.put(event)
        result = STTScopedTurnNormalizer(identity).apply_update(await next_text(session.turn_events()))
        await harness.peer_owner.handle_provider_turn_update(snapshot, result)
        await harness.peer_owner.handle_provider_turn_update(snapshot, result)
        await finish(harness)
        sources = [item.source_child for item in trace.items]
        assert [source.raw_text for source in sources] == [text]
        assert sources[0].speaker.speaker_id == expected
        assert sources[0].origins[0].contribution_range == (0, len(text))
        assert sources[0].origins[0].native_provenance == (event.provenance,)
        assert len(harness.peer_runtime.translation_history) == 1
        assert len(translator.calls) == 1
        assert not provider.sealed
    finally:
        await session.close()
        await close(harness)


@pytest.mark.asyncio
async def test_final_phrase_preserves_very_short_clear_no_between_returning_speaker():
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
    )

    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(identity, snapshot.settings, display_group_id=snapshot.display_group_id)
    provider = Provider()
    session = StreamingSpeakerSession(provider)
    translator = ControlledProvider()
    harness, trace = harness_for(translator)
    text = "Use two. No. Use three."
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 32800, 1)
        await session.observe_speaker_evidence(
            (evidence(0, 16000, 0), evidence(16000, 16800, 1), evidence(16800, 32800, 0))
        )
        event = update(req, text, release=True)
        word_ranges = [
            ("Use", 0, 0.0, 0.4), ("two.", 4, 0.4, 1.0),
            ("No.", 9, 1.0, 1.05), ("Use", 13, 1.05, 1.45), ("three.", 17, 1.45, 2.05),
        ]
        event = replace(
            event,
            provenance=replace(
                event.provenance,
                text_timing=tuple(
                    STTTextTiming(
                        "phrase-1", start, start + len(word), "word_interval", a, b,
                        clock_id="clock", complete_boundary=True,
                    )
                    for word, start, a, b in word_ranges
                ),
            ),
        )
        provider.events.put(event)
        result = STTScopedTurnNormalizer(identity).apply_update(await next_text(session.turn_events()))
        await harness.peer_owner.handle_provider_turn_update(snapshot, result)
        await finish(harness)
        sources = [item.source_child for item in trace.items]
        assert [source.raw_text for source in sources] == ["Use two. ", "No. ", "Use three."]
        assert [source.speaker.speaker_id for source in sources] == ["0", "1", "0"]
        assert "".join(source.raw_text for source in sources) == text
        assert [source.origins[0].contribution_range for source in sources] == [(0, 9), (9, 13), (13, 23)]
        assert len(harness.peer_runtime.translation_history) == len(translator.calls) == 3
        assert not provider.sealed
    finally:
        await session.close()
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("region_count", [1, 3])
@pytest.mark.parametrize("qualified", [False, True])
async def test_high_retention_final_preserves_owned_groups_and_qualified_clock(
    region_count, qualified
):
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
    )

    snapshot, identity = source_fixture()
    req = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )
    provider = Provider()
    session = StreamingSpeakerSession(provider)
    translator = ControlledProvider()
    harness, trace = harness_for(translator)
    words = [f"word{index:02}" for index in range(60)]
    text = " ".join(words)
    region_words = 60 // region_count
    contributions = tuple(
        STTTextContribution(
            f"phrase-{index}", index * region_words * 7,
            min((index + 1) * region_words * 7, len(text)),
        )
        for index in range(region_count)
    )
    timings = tuple(
        STTTextTiming(
            f"phrase-{index // region_words}", index * 7, index * 7 + len(word),
            "word_interval", index * 100, (index + 1) * 100, "ms", "clock", True,
        )
        for index, word in enumerate(words)
    ) + tuple(
        STTTextTiming(
            contribution.contribution_id, contribution.text_start, contribution.text_end,
            "result_span", index * region_words * 100, (index + 1) * region_words * 100,
            "ms", "clock", True,
        )
        for index, contribution in enumerate(contributions)
    )
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 96000, 1)
        observations = []
        for index in range(60):
            count = 35 if index < 8 else 34
            for piece in range(count):
                start = index * 1600 + piece * 1600 // count
                end = index * 1600 + (piece + 1) * 1600 // count
                observations.append(evidence(start, end, index % 2))
        await session.observe_speaker_evidence(tuple(observations))
        event = update(req, text, qualified=qualified, release=True)
        event = replace(
            event,
            contribution=None,
            included_contributions=contributions,
            provenance=replace(
                event.provenance, text_timing=timings, sent_until_sample=96000,
            ),
        )
        provider.events.put(event)
        outgoing = await next_text(session.turn_events())
        assert outgoing.identity == identity
        assert outgoing.source_release.text_start == 0
        assert outgoing.source_release.text_end == len(text)
        assert len(outgoing.final_speaker_runs) == (60 if qualified else region_count)
        assert "".join(run.text for run in outgoing.final_speaker_runs) == text
        assert [run.speaker_id for run in outgoing.final_speaker_runs] == (
            [str(index % 2) for index in range(60)]
            if qualified else [None] * region_count
        )
        result = STTScopedTurnNormalizer(identity).apply_update(outgoing)
        await harness.peer_owner.handle_provider_turn_update(snapshot, result)
        await finish(harness)
        sources = [item.source_child for item in trace.items]
        expected_texts = (
            [word + (" " if index < 59 else "") for index, word in enumerate(words)]
            if qualified else [text]
        )
        assert [source.raw_text for source in sources] == expected_texts
        assert "".join(source.raw_text for source in sources) == text
        assert [source.speaker.speaker_id for source in sources] == (
            [str(index % 2) for index in range(60)]
            if qualified else [None]
        )
        assert [source.text_range for source in sources] == (
            [(index * 7, min((index + 1) * 7, len(text))) for index in range(60)]
            if qualified else [(0, len(text))]
        )
        assert all(
            source.origins[0].native_provenance[0].audio_clocks[0].qualified == qualified
            for source in sources
        )
        assert len(translator.calls) == len(sources)
        history = harness.peer_runtime.translation_history
        assert [entry.text for entry in history] == expected_texts[-len(history):]
        assert len({source.source_id for source in sources}) == len(sources)
        assert not provider.sealed
    finally:
        await session.close()
        await close(harness)
