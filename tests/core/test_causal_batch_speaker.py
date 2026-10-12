from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    AudioSegmentSnapshot,
)
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.speaker.contracts import ProbabilityFrame, SourceMapPiece, SpeakerEvidence
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.batch_speaker import CausalBatchSession
from puripuly_heart.core.stt.recognition_units import contribution_transcript
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.providers.stt.local_parakeet_sherpa import (
    LocalParakeetJapaneseSherpaSTTBackend,
    LocalParakeetV3SherpaSTTBackend,
)
from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend


def span(start, end, epoch=7):
    return AudioCaptureSpan(
        epoch,
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


def request(order=1, language="en"):
    settings = AudioSegmentSettingsSnapshot(
        "local_qwen", (), (), "manual", language, (language,), 16000, 0.5, 200, 500
    )
    identity = STTProviderTurnIdentity(
        AudioSegmentIdentity(1, order, uuid4(), 7), "provider-1", f"request-{order}"
    )
    return STTProviderTurnRequest(identity, settings, display_group_id=uuid4())


def evidence(start, end, slots, *, ready=3.0, epoch=11, immutable=None, state="closed"):
    frame = ProbabilityFrame(
        start, end, tuple(0.9 if i in slots else 0.01 for i in range(8)), slots
    )
    return SpeakerEvidence(
        "capture",
        epoch,
        ready,
        (SourceMapPiece(start, end, span(start, end), "original"),),
        end,
        end if immutable is None else immutable,
        frame,
        state,
    )


class Recognizer:
    def __init__(self, failures=()):
        self.calls = []
        self.failures = set(failures)

    def create_stream(self):
        class Stream:
            def __init__(self):
                self.options = {}
                self.samples = None
                self.result = SimpleNamespace(text="")

            def set_option(self, key, value):
                self.options[key] = value

            def accept_waveform(self, rate, samples):
                self.samples = samples.copy()

        return Stream()

    def decode_stream(self, stream):
        self.calls.append(stream)
        if len(self.calls) in self.failures:
            raise RuntimeError("fixture failure")
        stream.result.text = "no"


async def opened(backend_type=LocalQwenSherpaSTTBackend, *, failures=(), wrapped=True):
    backend = backend_type(Path("/fixture"), language_hint="English")
    recognizer = Recognizer(failures)
    backend._recognizer = recognizer
    inner = await backend.open_session(projection=STTSessionProjection("scoped", "provider-1"))
    return backend, recognizer, CausalBatchSession(inner, clock=lambda: 3.1) if wrapped else inner


async def payload(session, req, start, end, seq, *, context=False):
    pcm = np.arange(start, end, dtype=np.int16).tobytes()
    await session.send_turn_audio(
        req.identity,
        pcm,
        payload_sequence=seq,
        source_ranges=(span(start, end),),
        context_only=context,
    )


async def seal(session, req, reason="silence"):
    await session.seal_turn(
        req.identity, sealed_content_ranges=(), seal_reason=reason, observed_trailing_silence_ms=0
    )


async def terminal_events(session, count=1):
    result = []
    async for event in session.turn_events():
        result.append(event)
        if sum(isinstance(item, STTProviderTurnTerminal) for item in result) == count:
            return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_type",
    [
        LocalQwenSherpaSTTBackend,
        LocalParakeetV3SherpaSTTBackend,
        LocalParakeetJapaneseSherpaSTTBackend,
    ],
)
async def test_actual_adapters_early_disjoint_short_a_b_a_and_shared_origins(backend_type):
    backend, recognizer, session = await opened(backend_type)
    req = request()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 1600, 1)
        await session.observe_speaker_evidence((evidence(0, 800, (0,)), evidence(800, 1600, (1,))))
        events = session.turn_events()
        early = await asyncio.wait_for(anext(events), 2)
        assert early.source_release is not None
        assert len(recognizer.calls) == 1
        assert recognizer.calls[0].samples.size == 800
        assert early.final_speaker_runs[0].speaker_id == "0"
        await payload(session, req, 1600, 2400, 2)
        await session.observe_speaker_evidence((evidence(1600, 2400, (0,)),))
        await seal(session, req)
        received = [early]
        async for event in events:
            received.append(event)
            if isinstance(event, STTProviderTurnTerminal):
                break
        normalizer = STTScopedTurnNormalizer(req.identity)
        ledger = STTContributionConsumptionLedger()
        snapshot = AudioSegmentSnapshot(
            req.identity.segment,
            req.settings,
            (span(0, 2400),),
            (),
            (),
            2400,
            0,
            0,
            0,
            0,
            0,
            True,
            "sealed",
            0,
            4,
            "silence",
            req.display_group_id,
        )
        transcripts = []
        for event in received:
            normalized = (
                normalizer.apply_terminal(event)
                if isinstance(event, STTProviderTurnTerminal)
                else normalizer.apply_update(event)
            )
            pieces = ledger.consume_parts(normalized)
            if pieces:
                transcripts.append(
                    contribution_transcript(
                        snapshot,
                        normalized,
                        pieces,
                        created_at=4,
                        publication_order=len(transcripts) + 1,
                    )
                )
        assert [t.text.strip() for t in transcripts] == ["no", "no", "no"]
        assert [t.final_speaker_runs[0].speaker_id for t in transcripts] == ["0", "1", "0"]
        assert [
            tuple(
                (s.normalized_start_sample, s.normalized_end_sample)
                for s in t.recognition_origins[0].source_ranges
            )
            for t in transcripts
        ] == [((0, 800),), ((800, 1600),), ((1600, 2400),)]
        assert all(
            t.recognition_origins[0].display_group_id == req.display_group_id for t in transcripts
        )
        assert all(not t.recognition_origins[0].context_ranges for t in transcripts)
        expected_padding = (
            [0, 32000, 32000] if backend_type is LocalParakeetV3SherpaSTTBackend else [0, 0, 0]
        )
        assert [
            t.recognition_origins[0].synthetic_right_context_samples for t in transcripts
        ] == expected_padding
        assert [t.recognition_origins[0].synthetic_context_sample_rate_hz for t in transcripts] == [
            16000 if count else None for count in expected_padding
        ]
        assert [call.samples.size for call in recognizer.calls] == [
            800 + count for count in expected_padding
        ]
        assert all(
            not count or np.count_nonzero(call.samples[-count:]) == 0
            for call, count in zip(recognizer.calls, expected_padding)
        )
        assert np.array_equal(
            np.concatenate([s.samples[:800] for s in recognizer.calls]),
            np.arange(2400, dtype=np.int16).astype(np.float32) / 32768,
        )
        assert sum(isinstance(event, STTProviderTurnTerminal) for event in received) == 1
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_queued_native_language_options_are_request_owned(wrapped):
    backend, recognizer, session = await opened(wrapped=wrapped)
    first, second = request(), request(2, "ja")
    await backend._decode_lock.acquire()
    try:
        await session.begin_turn(first)
        await payload(session, first, 0, 160, 1)
        await seal(session, first)
        await backend.reconfigure_session_options(
            LocalASRSessionOptions("ja", language_hint="Japanese")
        )
        await session.begin_turn(second)
        await payload(session, second, 160, 320, 1)
        await seal(session, second)
        backend._decode_lock.release()
        events = await asyncio.wait_for(terminal_events(session, 2), 2)
        terminals = [item for item in events if isinstance(item, STTProviderTurnTerminal)]
        assert [call.options["language"] for call in recognizer.calls] == ["English", "Japanese"]
        assert [item.text for item in terminals] == ["no", "no"]
        assert terminals[0].identity != terminals[1].identity
    finally:
        if backend._decode_lock.locked():
            backend._decode_lock.release()
        await session.close()
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["silence", "hardcut", "source_discontinuity"])
async def test_duplicate_stale_and_revised_evidence_never_reclaims_submitted_content(reason):
    backend, recognizer, session = await opened()
    req = request()
    await backend._decode_lock.acquire()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 3200, 1)
        original = (evidence(0, 1600, (0,)), evidence(1600, 3200, (1,)))
        await session.observe_speaker_evidence(original)
        await session.observe_speaker_evidence(original)
        await session.observe_speaker_evidence(
            (evidence(0, 800, (4,), epoch=10), evidence(800, 1600, (5,), epoch=10))
        )
        await session.observe_speaker_evidence((evidence(0, 1600, (3,)),))
        await payload(session, req, 3200, 4000, 2)
        await seal(session, req, reason)
        await session.observe_speaker_evidence(original)
        backend._decode_lock.release()
        await asyncio.wait_for(terminal_events(session), 2)
        assert [call.samples.size for call in recognizer.calls] == [1600, 2400]
        assert np.concatenate([call.samples for call in recognizer.calls]).size == 4000
    finally:
        if backend._decode_lock.locked():
            backend._decode_lock.release()
        await session.close()
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("slots", [(0, 1), ()])
async def test_overlap_and_unknown_keep_all_short_audio_without_forced_cut(slots):
    backend, recognizer, session = await opened()
    req = request()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 1600, 1)
        await session.observe_speaker_evidence((evidence(0, 800, (0,)), evidence(800, 1600, slots)))
        assert not recognizer.calls
        await seal(session, req)
        events = await asyncio.wait_for(terminal_events(session), 2)
        result = events[-1]
        assert result.text == "no"
        assert len(recognizer.calls) == 1
        assert recognizer.calls[0].samples.size == 1600
        assert result.final_speaker_runs[0].attribution_state != "identified" if slots else True
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_declared_rollover_context_prevents_false_whole_result_b_attribution():
    backend, recognizer, session = await opened()
    req = request()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 4800, 1, context=True)
        await payload(session, req, 4800, 6400, 2)
        await session.observe_speaker_evidence(
            (evidence(0, 4800, (0,)), evidence(4800, 6400, (1,)))
        )
        await seal(session, req, "hardcut")
        events = await asyncio.wait_for(terminal_events(session), 2)
        result = events[-1]
        assert result.final_speaker_runs[0].attribution_state == "mixed"
        assert result.source_audio[0].context_ranges == (span(0, 4800),)
        assert result.source_audio[0].source_ranges == (span(4800, 6400),)
        assert recognizer.calls[0].samples.size == 6400
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_failed_prefix_does_not_discard_successful_successor_or_redecode_parent():
    backend, recognizer, session = await opened(failures=(1,))
    req = request()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 1600, 1)
        await session.observe_speaker_evidence((evidence(0, 800, (0,)), evidence(800, 1600, (1,))))
        await seal(session, req)
        events = await asyncio.wait_for(terminal_events(session), 2)
        assert events[-1].outcome == "degraded"
        assert events[-1].text == "no"
        assert [call.samples.size for call in recognizer.calls] == [800, 800]
        assert events[-1].source_audio[0].source_ranges == (
            span(0, 1600).slice_normalized(800, 1600),
        )
        normalizer = STTScopedTurnNormalizer(req.identity)
        final = normalizer.apply_terminal(events[-1])
        assert final.text_authority == "degraded"
        consumption = STTContributionConsumptionLedger()
        consumption.seal_audio(req.identity)
        assert [part.text for part in consumption.consume_parts(final)] == ["no"]
        assert consumption.consume_parts(final) == ()
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_stop_fences_slow_decode_and_retained_tail():
    backend, recognizer, session = await opened()
    req = request()
    await backend._decode_lock.acquire()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 1600, 1)
        await session.observe_speaker_evidence((evidence(0, 800, (0,)), evidence(800, 1600, (1,))))
        await session.abort_turn(req.identity, reason="stop")
        events = await asyncio.wait_for(terminal_events(session), 2)
        assert events[-1].outcome == "cancelled"
        backend._decode_lock.release()
        await asyncio.sleep(0)
        assert not any(getattr(item, "source_release", None) for item in events)
    finally:
        if backend._decode_lock.locked():
            backend._decode_lock.release()
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_open_evidence_revisions_commit_only_at_immutable_frontier():
    backend, recognizer, session = await opened()
    req = request()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 2400, 1)
        await session.observe_speaker_evidence(
            (
                evidence(0, 800, (0,), immutable=0, state="open"),
                evidence(800, 1600, (1,), immutable=0, state="open"),
            )
        )
        assert not recognizer.calls
        await session.observe_speaker_evidence(
            (
                evidence(800, 1600, (0,), immutable=0, state="open"),
                evidence(1600, 2400, (1,), immutable=0, state="open"),
            )
        )
        await session.observe_speaker_evidence(
            (SpeakerEvidence("capture", 11, 3.0, (), 2400, 2400),)
        )
        await seal(session, req)
        await asyncio.wait_for(terminal_events(session), 2)
        assert [call.samples.size for call in recognizer.calls] == [1600, 800]
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_optional_cut_pressure_coalesces_tail_without_historical_burst_or_loss():
    backend, recognizer, session = await opened()
    req = request()
    await backend._decode_lock.acquire()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 20 * 160, 1)
        frames = tuple(
            evidence(index * 160, (index + 1) * 160, (index % 2,)) for index in range(20)
        )
        await session.observe_speaker_evidence(frames)
        await session.observe_speaker_evidence(frames)
        await seal(session, req)
        backend._decode_lock.release()
        events = await asyncio.wait_for(terminal_events(session), 2)
        assert len(recognizer.calls) == session.MAX_PENDING_SLICES + 1
        assert sum(call.samples.size for call in recognizer.calls) == 3200
        assert events[-1].final_speaker_runs[-1].attribution_state == "mixed"
        assert sum(len(item.source_audio) for item in events[-1:]) == 9
    finally:
        if backend._decode_lock.locked():
            backend._decode_lock.release()
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_discontinuity_invalidates_evidence_without_losing_owned_audio():
    backend, recognizer, session = await opened()
    req = request()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 1600, 1)
        await session.observe_speaker_evidence(
            (
                evidence(0, 800, (0,), immutable=0, state="open"),
                SpeakerEvidence("capture", 11, 3, (), 800, 0, reason="source_discontinuity"),
                evidence(800, 1600, (1,), epoch=12),
            )
        )
        await seal(session, req)
        events = await asyncio.wait_for(terminal_events(session), 2)
        assert len(recognizer.calls) == 1
        assert recognizer.calls[0].samples.size == 1600
        assert events[-1].final_speaker_runs[0].attribution_state == "unavailable"
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_actual_engine_preserves_onset_group_and_sourcechild_context_before_endpoint():
    from puripuly_heart.core.audio.ownership import (
        AudioSegmentTerminalReceipt,
        PeerAudioSegmentLedger,
    )
    from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
    from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        until,
    )

    backend = LocalQwenSherpaSTTBackend(Path("/fixture"), language_hint="English")
    backend._recognizer = Recognizer()
    req = request()
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=req.settings)
    received = []

    async def factory(settings, epoch):
        inner = await backend.open_session(projection=STTSessionProjection("scoped", epoch))
        return CausalBatchSession(inner)

    engine = ScopedRecognitionEngine(factory, event_sink=received.append)
    provider = ControlledProvider()
    provider.gates["no"] = asyncio.Event()
    harness, trace = harness_for(provider)
    segment_id = uuid4()
    try:
        await engine.observe_speaker_evidence(
            (
                evidence(0, 8000, (0,)),
                evidence(8000, 16000, (0,)),
                evidence(16000, 24000, (1,)),
            )
        )
        owned = ledger.observe_vad_event(
            SpeechStart(
                segment_id,
                np.zeros(8000, dtype=np.float32),
                np.ones(16000, dtype=np.float32),
                pre_roll_capture=(span(0, 8000),),
                chunk_capture=(span(8000, 24000),),
            ),
            now_monotonic_s=0,
        )
        await engine.handle_owned_vad_event(owned)
        await until(lambda: len(received) == 1)
        assert ledger.snapshots[0].state == "open"
        first = received[0]
        await harness.peer_owner.handle_provider_turn_update(ledger.snapshots[0], first)
        await until(lambda: len(provider.calls) == 1)
        assert trace.items == []
        end = ledger.observe_vad_event(SpeechEnd(segment_id), now_monotonic_s=4)
        await engine.handle_owned_vad_event(end)
        await until(lambda: any(isinstance(event, STTProviderTurnTerminal) for event in received))
        for event in received[1:]:
            if isinstance(event, STTProviderTurnTerminal):
                receipt = AudioSegmentTerminalReceipt(
                    end.segment.identity, event.outcome, end.segment, 4
                )
                await harness.peer_owner.handle_provider_turn_terminal(receipt, event)
            else:
                await harness.peer_owner.handle_provider_turn_update(end.segment, event)
        assert [item.text for item in harness.peer_runtime.translation_history] == ["no", "no"]
        provider.gates["no"].set()
        await finish(harness)
        sources = [item.source_child for item in trace.items]
        assert [source.raw_text for source in sources] == ["no", "no"]
        assert all(source.effective_text == source.raw_text for source in sources)
        assert [source.speaker.speaker_id for source in sources] == ["0", "1"]
        assert all(source.display_group_id == segment_id for source in sources)
        assert sources[0].origins[0].context_ranges == (span(0, 8000),)
        assert not sources[1].origins[0].context_ranges
        assert sources[0].origins[0].speaker_attribution.state == "identified"
        assert sources[0].origins[0].source_ranges == (
            span(8000, 24000).slice_normalized(8000, 16000),
        )
        assert sources[1].origins[0].source_ranges == (
            span(8000, 24000).slice_normalized(16000, 24000),
        )
        assert '"no"' in provider.calls[1][2]
        assert sum(isinstance(event, STTProviderTurnTerminal) for event in received) == 1
    finally:
        provider.gates["no"].set()
        await engine.close()
        await backend.close()
        await close(harness)


@pytest.mark.asyncio
async def test_real_probability_contract_cuts_without_adapter_active_slots_and_preserves_gap():
    backend, recognizer, session = await opened()
    req = request()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 2400, 1)
        raw = (
            evidence(0, 800, (0,)),
            evidence(800, 1200, ()),
            evidence(1200, 1600, (1,)),
            evidence(1600, 2400, (1,)),
        )
        raw = tuple(replace(item, frame=replace(item.frame, active_slots=())) for item in raw)
        await session.observe_speaker_evidence(raw)
        early = await asyncio.wait_for(anext(session.turn_events()), 2)
        assert early.source_audio[0].source_ranges == (span(0, 2400).slice_normalized(0, 1200),)
        assert early.final_speaker_runs[0].speaker_id == "0"
        await seal(session, req)
        await asyncio.wait_for(terminal_events(session), 2)
        assert [call.samples.size for call in recognizer.calls] == [1200, 1200]
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_competing_raw_probabilities_block_pure_transfer_even_with_single_active_hint():
    backend, recognizer, session = await opened()
    req = request()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 2400, 1)
        ambiguous = evidence(800, 1600, (0,))
        ambiguous = replace(
            ambiguous,
            frame=replace(
                ambiguous.frame, probabilities=(0.9, 0.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
            ),
        )
        await session.observe_speaker_evidence(
            (evidence(0, 800, (0,)), ambiguous, evidence(1600, 2400, (1,)))
        )
        assert not recognizer.calls
        await seal(session, req)
        events = await asyncio.wait_for(terminal_events(session), 2)
        assert events[-1].final_speaker_runs[0].attribution_state == "mixed"
        assert len(recognizer.calls) == 1
        assert recognizer.calls[0].samples.size == 2400
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_original_audio_observer_drains_to_actual_peer_sink_once_and_fences_stale_capture():
    from puripuly_heart.app.adapters.peer_capture.peer_capture_vad_sink import (
        PeerCaptureVadSinkAdapter,
    )
    from puripuly_heart.core.audio.ownership import CaptureStreamInput
    from puripuly_heart.core.runtime.peer_channel import (
        _CaptureGeneration,
        _GenerationGuardedVadSink,
    )

    delivered = []
    originals = []
    pending = [evidence(0, 160, (0,))]
    current = True

    class Observer:
        def observe_input(self, capture_session_id, event):
            originals.append((capture_session_id, event))

        def drain_evidence(self, maximum):
            result = tuple(pending[:maximum])
            del pending[:maximum]
            return result

    class Receiver:
        async def observe_speaker_evidence(self, evidence):
            delivered.extend(evidence)

    receiver = Receiver()
    adapter = PeerCaptureVadSinkAdapter(lambda: receiver)
    runtime = SimpleNamespace(
        _speaker_observer=Observer(),
        segment_ledger=None,
        is_current_generation=lambda generation: current,
    )
    guard = _GenerationGuardedVadSink(
        sink=adapter,
        runtime=runtime,
        capture_generation=_CaptureGeneration(1, "capture"),
        provider_ingress_ready=asyncio.Event(),
    )
    original = CaptureStreamInput(np.ones(160, dtype=np.float32), (span(0, 160),))
    try:
        await guard.handle_stream_input(original)
        async with asyncio.timeout(1):
            while not delivered:
                await asyncio.sleep(0)
        assert originals == [("capture", original)]
        assert delivered == [evidence(0, 160, (0,))]
        current = False
        await guard.handle_stream_input(original)
        assert len(originals) == 1
    finally:
        await guard.abort()
    assert guard._speaker_task is None


def test_explicit_audio_ownership_is_validated_before_contribution_consumption_commits():
    from puripuly_heart.core.stt.backend import (
        STTProviderTurnUpdate,
        STTSourceAudio,
        STTSourceRelease,
        STTTextContribution,
    )

    req = request()
    ledger = STTContributionConsumptionLedger()
    contribution = STTTextContribution("selected", 0, 2)
    update = STTProviderTurnUpdate(
        req.identity,
        1,
        "stable",
        "replace",
        "no",
        source_release=STTSourceRelease("release", 0, 2, "boundary"),
        included_contributions=(contribution,),
        source_audio=(STTSourceAudio("wrong", (span(0, 160),)),),
    )
    with pytest.raises(ValueError, match="unresolved audio ownership"):
        ledger.consume(update)
    valid = replace(update, source_audio=(STTSourceAudio("selected", (span(0, 160),)),))
    assert ledger.consume(valid) == "no"
    assert ledger.consume(valid) == ""


@pytest.mark.asyncio
async def test_failed_slice_resolves_before_successor_is_queued_and_capture_continues():
    backend, recognizer, session = await opened(failures=(1,))
    req = request()
    attempts = []
    backend.attempt_log_sink = lambda message, level: attempts.append(message)
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 1600, 1)
        await session.observe_speaker_evidence((evidence(0, 800, (0,)), evidence(800, 1600, (1,))))
        async with asyncio.timeout(2):
            while not attempts:
                await asyncio.sleep(0)
        assert "failure" in attempts[0]
        await payload(session, req, 1600, 2400, 2)
        await seal(session, req)
        events = await asyncio.wait_for(terminal_events(session), 2)
        assert events[-1].outcome == "degraded"
        assert events[-1].text == "no"
        assert [call.samples.size for call in recognizer.calls] == [800, 1600]
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        "local_qwen",
        "local_parakeet_v3",
        "local_parakeet_ja",
        "local_cpu_auto",
        "local_qwen_gpu",
    ],
)
async def test_production_factory_selects_causal_owner_for_every_peer_local_route_only(
    monkeypatch, route
):
    from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest

    from puripuly_heart.app.wiring import wiring_local_asr_provider_runtime as wiring
    from puripuly_heart.core.clock import FakeClock
    from tests.core.runtime.test_local_asr_provider_runtime import _resolved_config

    backend = LocalQwenSherpaSTTBackend(Path("/fixture"))
    backend._recognizer = Recognizer()
    monkeypatch.setattr(
        wiring, "create_stt_backend_from_resolved_config", lambda *args, **kwargs: backend
    )
    factory = wiring.SharedSTTProviderFactory(object(), FakeClock(), 300, Path("/fixture/gpu"))
    engines, sessions = [], []
    try:
        for channel in ("peer", "self"):
            config = _resolved_config(channel, provider=route)
            build = ProviderRuntimeBuildRequest(
                config=config,
                model_id="fixture",
                session_options=LocalASRSessionOptions("en"),
                provider_signature=(route,),
                runtime_signature=(route,),
                recognition_projection="scoped",
            )
            engine = await factory.create(build, gpu_runtime=object())
            engines.append(engine)
            session = await engine.session_factory(request().settings, "provider-1")
            sessions.append(session)
            assert isinstance(session, CausalBatchSession) == (channel == "peer")
    finally:
        for session in sessions:
            await session.close()
        for engine in engines:
            await engine.close()
        await backend.close()


@pytest.mark.asyncio
async def test_cpu_auto_uses_frozen_actual_delegate_not_later_language_label():
    from puripuly_heart.providers.stt.local_cpu import LocalCPUAutoSTTBackend

    delegate = LocalQwenSherpaSTTBackend(Path("/fixture"), language_hint="English")
    recognizer = Recognizer()
    delegate._recognizer = recognizer
    backend = LocalCPUAutoSTTBackend("en")
    backend._delegate = delegate
    backend._resolved_model_id = delegate.model_id
    inner = await backend.open_session(projection=STTSessionProjection("scoped", "provider-1"))
    session = CausalBatchSession(inner)
    req = request()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 1600, 1)
        await backend.reconfigure_session_options(
            LocalASRSessionOptions("ja", language_hint="Japanese")
        )
        await seal(session, req)
        await asyncio.wait_for(terminal_events(session), 2)
        assert recognizer.calls[0].options["language"] == "English"
        assert backend.resolved_model_id == delegate.model_id
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_late_sealed_abort_cannot_truncate_newer_capture_request():
    backend, recognizer, session = await opened()
    first, second = request(), request(2)
    await backend._decode_lock.acquire()
    try:
        await session.begin_turn(first)
        await payload(session, first, 0, 1600, 1)
        await seal(session, first)
        await session.begin_turn(second)
        await payload(session, second, 1600, 2400, 1)
        await session.abort_turn(first.identity, reason="replaced")
        await session.abort_turn(first.identity, reason="duplicate")
        await payload(session, second, 2400, 3200, 2)
        await seal(session, second)
        backend._decode_lock.release()
        events = await asyncio.wait_for(terminal_events(session, 2), 2)
        terminals = [item for item in events if isinstance(item, STTProviderTurnTerminal)]
        assert [item.outcome for item in terminals] == ["cancelled", "final"]
        assert terminals[1].source_audio[0].source_ranges == (span(1600, 2400), span(2400, 3200))
        assert terminals[1].text == "no"
        assert [call.samples.size for call in recognizer.calls] == [1600, 1600]
    finally:
        if backend._decode_lock.locked():
            backend._decode_lock.release()
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_out_of_order_native_results_release_only_source_ordered_prefix():
    from tests.core.test_stt_scoped_engine import ControlledScopedSession

    inner = ControlledScopedSession()
    session = CausalBatchSession(inner)
    req = request()
    pending = None
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 2400, 1)
        await session.observe_speaker_evidence(
            (
                evidence(0, 800, (0,)),
                evidence(800, 1600, (1,)),
                evidence(1600, 2400, (0,)),
            )
        )
        await seal(session, req)
        assert len(inner.requests) == 3
        identities = [item.identity for item in inner.requests]
        stream = session.turn_events()
        pending = asyncio.create_task(anext(stream))
        inner.emit(
            STTProviderTurnTerminal(identities[1], "final", "B", text_authority="authoritative")
        )
        await asyncio.sleep(0)
        assert not pending.done()
        inner.emit(
            STTProviderTurnTerminal(identities[0], "final", "A", text_authority="authoritative")
        )
        first = await asyncio.wait_for(pending, 2)
        second = await asyncio.wait_for(anext(stream), 2)
        assert (first.text, second.text) == ("A", "B")
        inner.emit(
            STTProviderTurnTerminal(identities[2], "final", "A", text_authority="authoritative")
        )
        terminal = await asyncio.wait_for(anext(stream), 2)
        assert terminal.text == "ABA"
        assert [run.speaker_id for run in terminal.final_speaker_runs] == ["0", "1", "0"]
        assert [
            tuple(
                (span.normalized_start_sample, span.normalized_end_sample)
                for span in audio.source_ranges
            )
            for audio in terminal.source_audio
        ] == [
            ((0, 800),),
            ((800, 1600),),
            ((1600, 2400),),
        ]
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await session.close()


@pytest.mark.asyncio
async def test_late_observation_selects_retained_prefix_and_records_distinct_causal_clocks(caplog):
    backend, recognizer, session = await opened()
    req = request()
    try:
        await session.begin_turn(req)
        await session.send_turn_audio(
            req.identity,
            bytes(48000 * 2),
            payload_sequence=1,
            source_ranges=(span(0, 48000),),
            context_only=False,
        )
        assert not recognizer.calls
        with caplog.at_level("INFO", logger="puripuly_heart.core.stt.batch_speaker"):
            await session.observe_speaker_evidence(
                (evidence(0, 38400, (0,)), evidence(38400, 48000, (1,)))
            )
        (selection,) = [record.getMessage() for record in caplog.records]
        assert "reason=speaker source_boundary=38400 evidence_ready=3.0" in selection
        assert "submitted_at=3.1 submitted_cursor=38400 context_samples=0" in selection
        await seal(session, req)
        events = await asyncio.wait_for(terminal_events(session), 1)
        assert [call.samples.size for call in recognizer.calls] == [38400, 9600]
        assert [
            (
                audio.source_ranges[0].normalized_start_sample,
                audio.source_ranges[-1].normalized_end_sample,
            )
            for audio in events[-1].source_audio
        ] == [(0, 38400), (38400, 48000)]
    finally:
        await session.close()
        await backend.close()


@pytest.mark.asyncio
async def test_parakeet_native_window_is_frozen_synthetic_context_not_capture_or_rollover(
    monkeypatch,
):
    backend, recognizer, session = await opened(LocalParakeetV3SherpaSTTBackend)
    inner_type = type(session.inner)
    native_requests, native_payloads = [], []
    begin = inner_type.begin_turn
    send = inner_type.send_turn_audio

    async def recorded_begin(inner, native_request):
        native_requests.append(native_request)
        await begin(inner, native_request)

    async def recorded_send(inner, identity, pcm, **kwargs):
        native_payloads.append((identity, pcm, kwargs))
        await send(inner, identity, pcm, **kwargs)

    monkeypatch.setattr(inner_type, "begin_turn", recorded_begin)
    monkeypatch.setattr(inner_type, "send_turn_audio", recorded_send)
    first, rollover = request(), request(2, "ja")
    await backend._decode_lock.acquire()
    try:
        await session.begin_turn(first)
        await payload(session, first, 0, 7200, 1)
        await session.observe_speaker_evidence(
            (
                evidence(0, 2400, (0,)),
                evidence(2400, 4800, (1,)),
                evidence(4800, 7200, (0,)),
            )
        )
        await backend.reconfigure_session_options(
            LocalASRSessionOptions("ja", language_hint="Japanese")
        )
        await seal(session, first, "hardcut")
        await session.begin_turn(rollover)
        await payload(session, rollover, 2400, 7200, 1, context=True)
        await payload(session, rollover, 7200, 8800, 2)
        await seal(session, rollover)
        assert [item.synthetic_right_context_samples for item in native_requests] == [
            0,
            32000,
            32000,
            0,
        ]
        assert [item.synthetic_context_sample_rate_hz for item in native_requests] == [
            None,
            16000,
            16000,
            None,
        ]
        with pytest.raises(FrozenInstanceError):
            native_requests[1].synthetic_right_context_samples = 0
        synthetic_payloads = [
            (pcm, kwargs) for _, pcm, kwargs in native_payloads if not kwargs["source_ranges"]
        ]
        assert len(synthetic_payloads) == 2
        assert all(
            pcm == bytes(64000) and kwargs["context_only"] for pcm, kwargs in synthetic_payloads
        )
        backend._decode_lock.release()
        events = await asyncio.wait_for(terminal_events(session, 2), 2)
        terminals = [event for event in events if isinstance(event, STTProviderTurnTerminal)]
        assert [stream.options["language"] for stream in recognizer.calls] == [
            "English",
            "English",
            "English",
            "Japanese",
        ]
        assert [stream.samples.size for stream in recognizer.calls] == [2400, 34400, 34400, 6400]
        assert [run.speaker_id for run in terminals[0].final_speaker_runs] == ["0", "1", "0"]
        assert [
            (
                audio.source_ranges[0].normalized_start_sample,
                audio.source_ranges[-1].normalized_end_sample,
            )
            for audio in terminals[0].source_audio
        ] == [(0, 2400), (2400, 4800), (4800, 7200)]
        assert all(not audio.context_ranges for audio in terminals[0].source_audio)
        assert all(
            audio.display_group_id == first.display_group_id for audio in terminals[0].source_audio
        )
        assert terminals[1].source_audio[0].context_ranges == (span(2400, 7200),)
        assert terminals[1].source_audio[0].synthetic_right_context_samples == 0
        assert terminals[1].source_audio[0].display_group_id == rollover.display_group_id
    finally:
        if backend._decode_lock.locked():
            backend._decode_lock.release()
        await session.close()
        await backend.close()


@pytest.mark.parametrize("count,rate", [(-1, None), (32000, None), (0, 16000), (32000, 0)])
def test_synthetic_context_metadata_cannot_be_unmapped_or_negative(count, rate):
    from puripuly_heart.core.stt.backend import STTSourceAudio

    with pytest.raises(ValueError, match="synthetic native context"):
        replace(
            request(), synthetic_right_context_samples=count, synthetic_context_sample_rate_hz=rate
        )
    with pytest.raises(ValueError, match="synthetic native context"):
        STTSourceAudio(
            "source",
            (span(0, 160),),
            synthetic_right_context_samples=count,
            synthetic_context_sample_rate_hz=rate,
        )
