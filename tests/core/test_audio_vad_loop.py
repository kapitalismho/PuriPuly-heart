from __future__ import annotations

import asyncio
from dataclasses import replace
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
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.vad.gating import (
    SpeechEnd,
    SpeechStart,
    VadGating,
    create_peer_vad_gating,
)
from tests.helpers.audio import FakeAudioSource
from tests.helpers.vad import SequenceVadEngine


async def test_peer_audio_ownership_preserves_resampled_ranges_for_continuous_speech():
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
    assert len(snapshots) == 1
    assert [snapshot.identity.segment_order for snapshot in snapshots] == [1]
    assert [snapshot.identity.capture_epoch for snapshot in snapshots] == [4]
    assert [snapshot.genuine_onset for snapshot in snapshots] == [True]
    assert [snapshot.prefix_context_sample_count for snapshot in snapshots] == [0]
    assert [snapshot.content_sample_count for snapshot in snapshots] == [23]
    assert [snapshot.synthetic_context_sample_count for snapshot in snapshots] == [1]
    assert [
        (
            snapshot.content_ranges[0].normalized_start_sample,
            snapshot.content_ranges[-1].normalized_end_sample,
        )
        for snapshot in snapshots
    ] == [(0, 23)]
    assert [snapshot.seal_reason for snapshot in snapshots] == [
        "source_eof",
    ]

    starts = [
        event.event for event in owned_events if event.event.__class__.__name__ == "SpeechStart"
    ]
    assert [len(event.pre_roll) for event in starts] == [0]
    assert [event.genuine_onset for event in starts] == [True]

    empty_receipt = ledger.terminalize(
        snapshots[0].identity.segment_id,
        outcome="empty",
        now_monotonic_s=3.0,
        text_authority="authoritative",
    )
    duplicate = ledger.terminalize(
        snapshots[0].identity.segment_id,
        outcome="failed",
        now_monotonic_s=4.0,
    )
    assert duplicate is empty_receipt
    assert ledger.snapshots == ()
    retired = ledger.terminal_receipts
    assert [receipt.outcome for receipt in retired] == ["empty"]
    assert [receipt.identity.segment_order for receipt in retired] == [1]
    assert retired[0] is empty_receipt

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
    assert ledger.snapshots == ()


async def test_listen_continuous_audio_rolls_at_seven_seconds_without_reused_content():
    frame_count = 440
    frames = [
        AudioFrameF32(
            samples=np.full((512,), float(sequence + 1), dtype=np.float32),
            sample_rate_hz=16000,
            channels=1,
            capture=AudioCaptureSpan(
                capture_epoch=12,
                callback_sequence=sequence,
                source_sample_rate_hz=16000,
                source_start_sample=sequence * 512,
                source_end_sample=(sequence + 1) * 512,
                source_start_monotonic_s=sequence * 512 / 16000,
                source_end_monotonic_s=(sequence + 1) * 512 / 16000,
            ),
        )
        for sequence in range(frame_count)
    ]
    vad = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.9] * frame_count),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        speech_threshold=0.5,
        hangover_ms=500,
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=12,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="blocked-test-provider",
            provider_signature=("blocked",),
            runtime_signature=("blocked",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.5,
            vad_hangover_ms=500,
            vad_pre_roll_ms=500,
        ),
    )
    owned_events: list[OwnedVadEvent] = []

    class NonblockingSink:
        async def handle_owned_vad_event(self, event: OwnedVadEvent) -> None:
            owned_events.append(event)

    await run_audio_vad_loop(
        source=FakeAudioSource(frames),
        vad=vad,
        sink=NonblockingSink(),
        target_sample_rate_hz=16000,
        segment_ledger=ledger,
    )

    snapshots = ledger.snapshots
    assert [snapshot.identity.segment_order for snapshot in snapshots] == [1, 2, 3]
    assert [snapshot.seal_reason for snapshot in snapshots] == [
        "delivery_deadline",
        "delivery_deadline",
        "source_eof",
    ]
    assert [snapshot.genuine_onset for snapshot in snapshots] == [True, False, False]
    assert [snapshot.prefix_context_sample_count for snapshot in snapshots] == [0, 0, 0]
    assert sum(snapshot.content_sample_count for snapshot in snapshots) == frame_count * 512
    content_ranges = [
        capture_range for snapshot in snapshots for capture_range in snapshot.content_ranges
    ]
    assert content_ranges[0].normalized_start_sample == 0
    assert content_ranges[-1].normalized_end_sample == frame_count * 512
    assert all(
        content_ranges[index - 1].normalized_end_sample
        == content_ranges[index].normalized_start_sample
        for index in range(1, len(content_ranges))
    )


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

    snapshots = (
        ledger.terminal_receipts[0].segment,
        *ledger.snapshots,
    )
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
    assert [receipt.outcome for receipt in ledger.terminal_receipts] == [
        "failed",
        "cancelled",
    ]


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

    segment = ledger.terminal_receipts[0].segment
    assert segment.content_ranges[0].source_start_sample == 0
    assert (
        segment.content_ranges[-1].source_end_sample == segment.failed_ranges[0].source_start_sample
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

    segment = ledger.terminal_receipts[0].segment
    assert segment.content_sample_count == 512
    assert segment.synthetic_context_sample_count == 0
    assert segment.content_ranges[0].source_start_sample == 0
    assert (
        segment.content_ranges[-1].source_end_sample == segment.failed_ranges[0].source_start_sample
    )
    assert segment.failed_ranges[-1].source_end_sample == 2048
    assert segment.seal_reason == "source_discontinuity"
    assert ledger.terminal_receipts[0].outcome == "failed"


async def test_peer_off_controller_steps_to_224ms_and_seals_exact_uneven_source_range() -> None:
    sample_count = 125 * 512
    frames: list[AudioFrameF32] = []
    cursor = 0
    sequence = 0
    split_index = 0
    while cursor < sample_count:
        frame_samples = (200, 312)[split_index % 2]
        end = min(sample_count, cursor + frame_samples)
        frames.append(
            AudioFrameF32(
                samples=np.ones((end - cursor,), dtype=np.float32),
                sample_rate_hz=16000,
                capture=AudioCaptureSpan(
                    capture_epoch=12,
                    callback_sequence=sequence,
                    source_sample_rate_hz=16000,
                    source_start_sample=cursor,
                    source_end_sample=end,
                    source_start_monotonic_s=cursor / 16000,
                    source_end_monotonic_s=end / 16000,
                ),
            )
        )
        cursor = end
        sequence += 1
        split_index += 1
    vad = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.9] * 118 + [0.0] * 7),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=900,
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=14,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="test",
            provider_signature=("test",),
            runtime_signature=("runtime",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.5,
            vad_hangover_ms=900,
            vad_pre_roll_ms=500,
        ),
    )
    owned_events: list[OwnedVadEvent] = []

    class Sink:
        async def handle_owned_vad_event(self, event: OwnedVadEvent) -> None:
            owned_events.append(event)

        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"unowned event reached sink: {event!r}")

    await run_audio_vad_loop(
        source=FakeAudioSource(frames),
        vad=vad,
        sink=Sink(),
        target_sample_rate_hz=16000,
        segment_ledger=ledger,
        monotonic_clock=lambda: 0.0,
    )

    assert len(ledger.snapshots) == 1
    segment = ledger.snapshots[0]
    end = next(owned.event for owned in owned_events if isinstance(owned.event, SpeechEnd))
    assert segment.opened_at_monotonic_s == 0.0
    assert segment.sealed_at_monotonic_s == 0.0
    assert segment.seal_reason == "delivery_pause"
    assert segment.content_sample_count == sample_count
    assert segment.content_ranges[0].normalized_start_sample == 0
    assert segment.content_ranges[-1].normalized_end_sample == sample_count
    assert end.trailing_silence_ms == 224
    assert end.reason == "delivery_pause"
    assert segment.settings.delivery_profile_effective == "off"


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


async def test_capture_progress_distinguishes_no_frames_from_frames_without_speech() -> None:
    logs: list[str] = []

    class DelayedSource:
        async def frames(self):
            await asyncio.sleep(0.02)
            yield AudioFrameF32(
                samples=np.zeros((8,), dtype=np.float32),
                sample_rate_hz=16000,
                channels=1,
            )

        async def close(self) -> None:
            return None

    class Sink:
        async def handle_vad_event(self, _event: object) -> None:
            return None

    vad = VadGating(
        SequenceVadEngine(probs=[0.0, 0.0]),
        sample_rate_hz=16000,
        chunk_samples=4,
        ring_buffer_ms=1,
        hangover_ms=640,
    )

    await run_audio_vad_loop(
        source=DelayedSource(),
        vad=vad,
        sink=Sink(),
        target_sample_rate_hz=16000,
        log_basic=logs.append,
        no_frame_timeout_s=0.005,
        progress_interval_audio_ms=0.25,
    )

    assert any("state=no_frames" in message for message in logs)
    assert any("state=frames_resumed" in message for message in logs)
    assert any("state=frames_without_admitted_speech" in message for message in logs)
