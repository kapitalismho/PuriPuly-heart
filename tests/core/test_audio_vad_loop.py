from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from puripuly_heart.core.audio.desktop_pipeline import DesktopPeerPipeline
from puripuly_heart.core.audio.format import (
    AudioCaptureDiscontinuity,
    AudioCaptureSpan,
    AudioFrameF32,
)
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    OwnedVadEvent,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart, VadGating
from puripuly_heart.domain.events import STTSessionState
from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend
from tests.helpers.audio import FakeAudioSource, make_frames
from tests.helpers.fakes import FakeSender, SpeechAwareFakeBackend, SpeechAwareFakeSession
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.helpers.vad import SequenceVadEngine


async def test_audio_vad_loop_pipeline_smoke():
    clock = FakeClock()
    sender = FakeSender()
    osc = ChatboxPaginator(sender=sender, clock=clock)

    stt = ManagedSTTProvider(backend=SpeechAwareFakeBackend(), sample_rate_hz=16000, clock=clock)
    harness = compose_translation_test_harness(
        stt=stt, llm=None, osc=osc, clock=clock, fallback_transcript_only=True
    )
    await harness.start(auto_flush_osc=False)

    probs = [0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0]
    vad = VadGating(
        SequenceVadEngine(probs=probs), sample_rate_hz=16000, ring_buffer_ms=64, hangover_ms=64
    )

    chunks = [
        np.zeros((512,), dtype=np.float32),
        np.zeros((512,), dtype=np.float32),
        np.ones((512,), dtype=np.float32),
        np.ones((512,), dtype=np.float32),
        np.zeros((512,), dtype=np.float32),
        np.zeros((512,), dtype=np.float32),
        np.zeros((512,), dtype=np.float32),
    ]
    audio = np.concatenate(chunks, axis=0)

    # Deliberately split into uneven frames to exercise chunking.
    splits = [1000, 1000, 1000, audio.size - 3000]
    frames = make_frames(audio, sample_rate_hz=16000, splits=splits)
    source = FakeAudioSource(frames)
    await run_audio_vad_loop(
        source=source,
        vad=vad,
        sink=harness.self_owner,
        target_sample_rate_hz=16000,
    )

    for _ in range(50):
        if "FINAL" in sender.sent:
            break
        await asyncio.sleep(0.01)

    assert "FINAL" in sender.sent
    await harness.stop()

async def test_peer_audio_ownership_preserves_resampled_ranges_across_continuous_rollover():
    source_cursor = 0
    frames: list[AudioFrameF32] = []
    for sequence, frame_count in enumerate((17, 29, 23)):
        samples = np.ones((frame_count, 2), dtype=np.float32)
        frames.append(
            AudioFrameF32(
                samples=samples.reshape(-1),
                sample_rate_hz=48000,
                channels=2,
                capture=AudioCaptureSpan(
                    capture_epoch=4,
                    callback_sequence=sequence,
                    source_sample_rate_hz=48000,
                    source_start_sample=source_cursor,
                    source_end_sample=source_cursor + frame_count,
                    source_start_monotonic_s=source_cursor / 48000,
                    source_end_monotonic_s=(source_cursor + frame_count) / 48000,
                ),
            )
        )
        source_cursor += frame_count

    source = DesktopPeerPipeline(
        source=FakeAudioSource(frames),
        target_sample_rate_hz=16000,
    )
    vad = VadGating(
        SequenceVadEngine(probs=[0.9, 0.9, 0.9]),
        sample_rate_hz=16000,
        chunk_samples=8,
        ring_buffer_ms=1,
        hangover_ms=640,
        max_segment_ms=1,
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=7,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="test",
            provider_signature=("test",),
            runtime_signature=("runtime",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.5,
            vad_hangover_ms=640,
            vad_pre_roll_ms=1,
        ),
    )
    owned_events: list[OwnedVadEvent] = []

    class OwnedSink:
        async def handle_owned_vad_event(self, event: OwnedVadEvent) -> None:
            owned_events.append(event)

        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"unowned event reached sink: {event!r}")

    await run_audio_vad_loop(
        source=source,
        vad=vad,
        sink=OwnedSink(),
        target_sample_rate_hz=16000,
        segment_ledger=ledger,
        monotonic_clock=lambda: 2.0,
    )

    snapshots = ledger.snapshots
    assert len(snapshots) == 2
    assert [snapshot.identity.segment_order for snapshot in snapshots] == [1, 2]
    assert [snapshot.identity.capture_epoch for snapshot in snapshots] == [4, 4]
    assert [snapshot.genuine_onset for snapshot in snapshots] == [True, False]
    assert [snapshot.prefix_context_sample_count for snapshot in snapshots] == [0, 0]
    assert [snapshot.content_sample_count for snapshot in snapshots] == [16, 7]
    assert [snapshot.synthetic_context_sample_count for snapshot in snapshots] == [0, 1]
    assert [
        (
            snapshot.content_ranges[0].normalized_start_sample,
            snapshot.content_ranges[-1].normalized_end_sample,
        )
        for snapshot in snapshots
    ] == [(0, 16), (16, 23)]
    assert [snapshot.seal_reason for snapshot in snapshots] == [
        "max_duration",
        "source_eof",
    ]

    starts = [
        event.event
        for event in owned_events
        if event.event.__class__.__name__ == "SpeechStart"
    ]
    assert [len(event.pre_roll) for event in starts] == [0, 0]
    assert [event.genuine_onset for event in starts] == [True, False]

    second_receipt = ledger.terminalize(
        snapshots[1].identity.segment_id,
        outcome="empty",
        now_monotonic_s=3.0,
    )
    duplicate = ledger.terminalize(
        snapshots[1].identity.segment_id,
        outcome="failed",
        now_monotonic_s=4.0,
    )
    assert duplicate is second_receipt
    assert ledger.drain_ready_terminal_receipts() == ()

    ledger.terminalize(
        snapshots[0].identity.segment_id,
        outcome="failed",
        now_monotonic_s=5.0,
    )
    retired = ledger.drain_ready_terminal_receipts()
    assert [receipt.outcome for receipt in retired] == ["failed", "empty"]
    assert [receipt.identity.segment_order for receipt in retired] == [1, 2]
    assert ledger.drain_ready_terminal_receipts() == ()

    next_settings = replace(snapshots[0].settings, provider_id="next")
    ledger.rebind(activation_generation=8, settings=next_settings)
    next_id = uuid4()
    next_capture = AudioCaptureSpan(
        capture_epoch=4,
        callback_sequence=3,
        source_sample_rate_hz=48000,
        source_start_sample=69,
        source_end_sample=93,
        source_start_monotonic_s=69 / 48000,
        source_end_monotonic_s=93 / 48000,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=23,
        normalized_end_sample=31,
    )
    ledger.observe_vad_event(
        SpeechStart(
            next_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            chunk_capture=(next_capture,),
        ),
        now_monotonic_s=6.0,
    )
    ledger.observe_vad_event(
        SpeechEnd(next_id, trailing_silence_ms=0, reason="silence"),
        now_monotonic_s=6.1,
    )
    rebound = ledger.snapshots[0]
    assert snapshots[0].settings.provider_id == "test"
    assert rebound.settings.provider_id == "next"
    assert rebound.identity.activation_generation == 8
    cancelled = ledger.cancel_unfinished(now_monotonic_s=6.2)
    assert [receipt.outcome for receipt in cancelled] == ["cancelled"]
    assert ledger.drain_ready_terminal_receipts() == cancelled

async def test_peer_audio_unknown_gap_fails_open_segment_without_turning_loss_into_silence():
    first = AudioFrameF32(
        samples=np.ones((10,), dtype=np.float32),
        sample_rate_hz=16000,
        capture=AudioCaptureSpan(
            capture_epoch=8,
            callback_sequence=0,
            source_sample_rate_hz=16000,
            source_start_sample=0,
            source_end_sample=10,
            source_start_monotonic_s=0.0,
            source_end_monotonic_s=10 / 16000,
        ),
    )
    successor = AudioFrameF32(
        samples=np.ones((8,), dtype=np.float32),
        sample_rate_hz=16000,
        capture=AudioCaptureSpan(
            capture_epoch=9,
            callback_sequence=1,
            source_sample_rate_hz=16000,
            source_start_sample=0,
            source_end_sample=8,
            source_start_monotonic_s=1.0,
            source_end_monotonic_s=1.0 + 8 / 16000,
            discontinuity_before=AudioCaptureDiscontinuity(
                kind="unknown_loss",
                observed_at_monotonic_s=1.0,
            ),
        ),
    )
    vad = VadGating(
        SequenceVadEngine(probs=[0.9, 0.9]),
        sample_rate_hz=16000,
        chunk_samples=8,
        ring_buffer_ms=1,
        hangover_ms=640,
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=9,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="test",
            provider_signature=("test",),
            runtime_signature=("runtime",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.5,
            vad_hangover_ms=640,
            vad_pre_roll_ms=1,
        ),
    )
    owned_events: list[OwnedVadEvent] = []

    class OwnedSink:
        async def handle_owned_vad_event(self, event: OwnedVadEvent) -> None:
            owned_events.append(event)

        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"unowned event reached sink: {event!r}")

    await run_audio_vad_loop(
        source=FakeAudioSource([first, successor]),
        vad=vad,
        sink=OwnedSink(),
        target_sample_rate_hz=16000,
        segment_ledger=ledger,
        monotonic_clock=lambda: 6.0,
    )

    snapshots = ledger.snapshots
    assert len(snapshots) == 2
    assert [snapshot.identity.capture_epoch for snapshot in snapshots] == [8, 9]
    assert [snapshot.content_sample_count for snapshot in snapshots] == [8, 8]
    assert [snapshot.failed_normalized_sample_count for snapshot in snapshots] == [2, 0]
    assert [snapshot.failed_source_sample_count for snapshot in snapshots] == [2, 0]
    assert [snapshot.seal_reason for snapshot in snapshots] == [
        "source_discontinuity",
        "source_eof",
    ]
    assert ledger.terminal_receipts[0].outcome == "failed"
    assert [event.event.reason for event in owned_events if hasattr(event.event, "reason")] == [
        "source_discontinuity",
        "source_eof",
    ]

    cancelled = ledger.cancel_unfinished(now_monotonic_s=7.0)
    assert [receipt.outcome for receipt in cancelled] == ["cancelled"]
    retired = ledger.drain_ready_terminal_receipts()
    assert [receipt.outcome for receipt in retired] == ["failed", "cancelled"]
    assert ledger.drain_ready_terminal_receipts() == ()

async def test_known_resampler_discontinuity_seals_exact_accepted_source_edge():
    frames = [
        AudioFrameF32(
            samples=np.ones((end - start,), dtype=np.float32),
            sample_rate_hz=48000,
            capture=AudioCaptureSpan(
                capture_epoch=2,
                callback_sequence=sequence,
                source_sample_rate_hz=48000,
                source_start_sample=start,
                source_end_sample=end,
                source_start_monotonic_s=start / 48000,
                source_end_monotonic_s=end / 48000,
                discontinuity_before=discontinuity,
            ),
        )
        for sequence, start, end, discontinuity in (
            (0, 0, 1024, None),
            (1, 1024, 2048, None),
            (2, 2048, 2148, None),
            (
                3,
                3072,
                4096,
                AudioCaptureDiscontinuity(
                    kind="known_loss",
                    observed_at_monotonic_s=3072 / 48000,
                    lost_source_samples=924,
                ),
            ),
        )
    ]
    ledger = PeerAudioSegmentLedger(
        activation_generation=3,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="test",
            provider_signature=("test",),
            runtime_signature=("runtime",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.5,
            vad_hangover_ms=640,
            vad_pre_roll_ms=500,
        ),
    )

    class Sink:
        async def handle_owned_vad_event(self, _event: OwnedVadEvent) -> None:
            return None

        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"unowned event reached sink: {event!r}")

    await run_audio_vad_loop(
        source=DesktopPeerPipeline(FakeAudioSource(frames)),
        vad=VadGating(
            SequenceVadEngine(probs=[0.9]),
            sample_rate_hz=16000,
            chunk_samples=512,
            ring_buffer_ms=500,
            hangover_ms=640,
        ),
        sink=Sink(),
        target_sample_rate_hz=16000,
        segment_ledger=ledger,
        monotonic_clock=lambda: 1.0,
    )

    segment = ledger.snapshots[0]
    assert segment.content_ranges[0].source_start_sample == 0
    assert (
        segment.content_ranges[-1].source_end_sample
        == segment.failed_ranges[0].source_start_sample
    )
    assert segment.failed_ranges[-1].source_end_sample == 2148
    assert all(
        left.source_end_sample == right.source_start_sample
        for left, right in zip(segment.failed_ranges, segment.failed_ranges[1:])
    )
    assert segment.seal_reason == "source_discontinuity"
    assert ledger.terminal_receipts[0].outcome == "failed"

async def test_unexpected_source_end_discards_tail_and_accounts_failed_residue():
    frames = [
        AudioFrameF32(
            samples=np.ones((1024,), dtype=np.float32),
            sample_rate_hz=48000,
            capture=AudioCaptureSpan(
                capture_epoch=5,
                callback_sequence=sequence,
                source_sample_rate_hz=48000,
                source_start_sample=sequence * 1024,
                source_end_sample=(sequence + 1) * 1024,
                source_start_monotonic_s=sequence * 1024 / 48000,
                source_end_monotonic_s=(sequence + 1) * 1024 / 48000,
            ),
        )
        for sequence in range(2)
    ]

    class TerminalSource:
        terminal_reason = "target_exited"

        async def frames(self):
            for frame in frames:
                yield frame

        async def close(self) -> None:
            return None

    ledger = PeerAudioSegmentLedger(
        activation_generation=4,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="test",
            provider_signature=("test",),
            runtime_signature=("runtime",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.5,
            vad_hangover_ms=640,
            vad_pre_roll_ms=500,
        ),
    )

    class Sink:
        async def handle_owned_vad_event(self, _event: OwnedVadEvent) -> None:
            return None

        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"unowned event reached sink: {event!r}")

    await run_audio_vad_loop(
        source=DesktopPeerPipeline(TerminalSource()),
        vad=VadGating(
            SequenceVadEngine(probs=[0.9]),
            sample_rate_hz=16000,
            chunk_samples=512,
            ring_buffer_ms=500,
            hangover_ms=640,
        ),
        sink=Sink(),
        target_sample_rate_hz=16000,
        segment_ledger=ledger,
        monotonic_clock=lambda: 1.0,
    )

    segment = ledger.snapshots[0]
    assert segment.content_sample_count == 512
    assert segment.synthetic_context_sample_count == 0
    assert segment.content_ranges[0].source_start_sample == 0
    assert (
        segment.content_ranges[-1].source_end_sample
        == segment.failed_ranges[0].source_start_sample
    )
    assert segment.failed_ranges[-1].source_end_sample == 2048
    assert segment.seal_reason == "source_discontinuity"
    assert ledger.terminal_receipts[0].outcome == "failed"


async def test_audio_vad_loop_ingests_next_utterance_while_local_decode_is_blocked(
    monkeypatch,
) -> None:
    decode_started = asyncio.Event()
    release_decode = asyncio.Event()
    decode_calls: list[np.ndarray] = []

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        decode_calls.append(samples_f32.copy())
        if len(decode_calls) == 1:
            decode_started.set()
            await release_decode.wait()
        return f"final-{len(decode_calls)}"

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    stt = ManagedSTTProvider(backend=backend, sample_rate_hz=16000)
    vad = VadGating(
        SequenceVadEngine(probs=[0.9, 0.0, 0.9, 0.0]),
        sample_rate_hz=16000,
        ring_buffer_ms=32,
        hangover_ms=0,
    )
    audio = np.concatenate(
        [
            np.ones(512, dtype=np.float32),
            np.zeros(512, dtype=np.float32),
            np.full(512, 0.5, dtype=np.float32),
            np.zeros(512, dtype=np.float32),
        ]
    )
    source = FakeAudioSource(make_frames(audio, sample_rate_hz=16000, splits=[512] * 4))

    loop_task = asyncio.create_task(
        run_audio_vad_loop(
            source=source,
            vad=vad,
            sink=stt,
            target_sample_rate_hz=16000,
        )
    )
    await asyncio.wait_for(decode_started.wait(), timeout=0.5)
    await asyncio.wait_for(asyncio.shield(loop_task), timeout=0.5)
    release_decode.set()
    await stt.close()

    assert len(decode_calls) == 2


async def test_run_audio_vad_loop_applies_audio_gate_before_forwarding_to_sink():
    original = np.arange(8, dtype=np.float32)
    gated = np.full((8,), 9.0, dtype=np.float32)
    sink_events: list[np.ndarray] = []
    gate_inputs: list[np.ndarray] = []
    vad_inputs: list[np.ndarray] = []

    class FakeSource:
        async def frames(self):
            yield AudioFrameF32(samples=original, sample_rate_hz=16000)

        async def close(self) -> None:
            return None

    class FakeVad:
        chunk_samples = 8

        def process_chunk(self, chunk: np.ndarray):
            vad_inputs.append(chunk.copy())
            return [chunk.copy()]

    class FakeSink:
        async def handle_vad_event(self, event: np.ndarray) -> None:
            sink_events.append(event)

    class FakeGate:
        def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
            gate_inputs.append(chunk.copy())
            return gated

    await run_audio_vad_loop(
        source=FakeSource(),
        vad=FakeVad(),
        sink=FakeSink(),
        target_sample_rate_hz=16000,
        audio_gate=FakeGate(),
    )

    assert np.array_equal(gate_inputs[0], original)
    assert np.array_equal(vad_inputs[0], gated)
    assert np.array_equal(sink_events[0], gated)


class _PeerOnlySink:
    def __init__(self, harness: PeerTranslationChannelOwner) -> None:
        self._harness = harness

    async def handle_vad_event(self, event) -> None:  # noqa: ANN001
        await self._harness.peer_owner.handle_peer_vad_event(event)


class _RecordingSpeechBackend:
    def __init__(self) -> None:
        self.open_calls = 0
        self.sessions: list[SpeechAwareFakeSession] = []

    async def open_session(self) -> SpeechAwareFakeSession:
        self.open_calls += 1
        session = SpeechAwareFakeSession()
        self.sessions.append(session)
        return session


async def test_peer_pipeline_drops_short_candidate_before_opening_stt_session():
    clock = FakeClock()
    sender = FakeSender()
    osc = ChatboxPaginator(sender=sender, clock=clock)
    backend = _RecordingSpeechBackend()
    peer_stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        channel="peer",
        clock=clock,
    )
    harness = compose_translation_test_harness(
        stt=None, peer_stt=peer_stt, llm=None, osc=osc, clock=clock
    )
    await harness.start(auto_flush_osc=False)

    probs = [0.0, 0.9, 0.9, 0.0]
    vad = VadGating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        speech_threshold=0.6,
        hangover_ms=64,
        start_debounce_chunks=3,
        start_commit_chunks=3,
    )

    audio = np.concatenate(
        [np.full((512,), float(i), dtype=np.float32) for i in range(len(probs))], axis=0
    )
    frames = make_frames(audio, sample_rate_hz=16000, splits=[1000, audio.size - 1000])
    source = FakeAudioSource(frames)
    await run_audio_vad_loop(
        source=source,
        vad=vad,
        sink=_PeerOnlySink(harness),
        target_sample_rate_hz=16000,
    )

    assert backend.open_calls == 0
    assert peer_stt.state == STTSessionState.DISCONNECTED
    assert harness.peer_runtime.utterances == {}

    await harness.stop()


async def test_peer_pipeline_commits_after_candidate_reaches_minimum_length():
    clock = FakeClock()
    sender = FakeSender()
    osc = ChatboxPaginator(sender=sender, clock=clock)
    backend = _RecordingSpeechBackend()
    peer_stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        channel="peer",
        clock=clock,
    )
    harness = compose_translation_test_harness(
        stt=None, peer_stt=peer_stt, llm=None, osc=osc, clock=clock
    )
    await harness.start(auto_flush_osc=False)

    probs = [0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0]
    vad = VadGating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        speech_threshold=0.6,
        hangover_ms=64,
        start_debounce_chunks=3,
        start_commit_chunks=3,
    )

    audio = np.concatenate(
        [np.full((512,), float(i), dtype=np.float32) for i in range(len(probs))], axis=0
    )
    frames = make_frames(audio, sample_rate_hz=16000, splits=[1000, 1000, audio.size - 2000])
    source = FakeAudioSource(frames)
    await run_audio_vad_loop(
        source=source,
        vad=vad,
        sink=_PeerOnlySink(harness),
        target_sample_rate_hz=16000,
    )

    for _ in range(50):
        if harness.peer_runtime.utterances:
            break
        await asyncio.sleep(0.01)

    assert backend.open_calls == 1
    assert harness.peer_runtime.utterances

    await harness.stop()
