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
    SpeechChunk,
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
    source_activity: list[tuple[bool, float]] = []

    class OwnedSink:
        async def handle_owned_vad_event(self, event: OwnedVadEvent) -> None:
            owned_events.append(event)

        async def observe_source_activity(
            self,
            *,
            speech_observed: bool,
            observed_at_monotonic_s: float,
        ) -> None:
            source_activity.append((speech_observed, observed_at_monotonic_s))

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
    assert source_activity == [(True, 2.0), (True, 2.0), (True, 2.0)]

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


async def test_listen_continuous_audio_rolls_at_six_seconds_with_context_only_tail():
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
    assert [snapshot.prefix_context_sample_count for snapshot in snapshots] == [0, 4800, 4800]
    starts = [owned.event for owned in owned_events if isinstance(owned.event, SpeechStart)]
    assert [start.pre_roll.size for start in starts] == [0, 4800, 4800]
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
    assert [
        (
            snapshot.context_ranges[0].normalized_start_sample,
            snapshot.context_ranges[-1].normalized_end_sample,
        )
        for snapshot in snapshots[1:]
    ] == [
        (
            snapshots[index - 1].content_ranges[-1].normalized_end_sample - 4800,
            snapshots[index - 1].content_ranges[-1].normalized_end_sample,
        )
        for index in (1, 2)
    ]


async def test_peer_deadline_waits_for_inflight_frame_before_sealing(monkeypatch):
    from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController

    mid_frame = asyncio.Event()
    sealed = asyncio.Event()
    started = False
    triggered = False
    timer_started = False
    owned_events = []
    stream_events = []

    async def deadline(controller, segment_id, delay_s):
        nonlocal timer_started
        if timer_started:
            await asyncio.Event().wait()
        timer_started = True
        await mid_frame.wait()
        await controller._seal(segment_id, reason="delivery_deadline", rollover=True)
        sealed.set()

    monkeypatch.setattr(ListenDeliveryController, "_run_hard_timer", deadline)
    frames = [
        AudioFrameF32(
            np.full(512, 0.25, dtype=np.float32),
            16000,
            capture=AudioCaptureSpan(
                1,
                index,
                16000,
                index * 512,
                (index + 1) * 512,
                index * 0.032,
                (index + 1) * 0.032,
            ),
        )
        for index in range(8)
    ]

    class Source:
        async def frames(self):
            for frame in frames:
                yield frame
                if triggered:
                    await asyncio.wait_for(sealed.wait(), timeout=1)

    class Sink:
        async def handle_owned_vad_event(self, event):
            nonlocal started
            owned_events.append(event)
            if isinstance(event.event, SpeechStart):
                started = True

        async def handle_stream_input(self, event):
            nonlocal triggered
            stream_events.append(event)
            if started and not triggered:
                triggered = True
                mid_frame.set()
                await asyncio.sleep(0)
                await asyncio.sleep(0)

    vad = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.9] * len(frames)),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        speech_threshold=0.5,
        hangover_ms=500,
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="gemini_transcribe",
            provider_signature=("gemini_transcribe",),
            runtime_signature=("gemini_transcribe",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.5,
            vad_hangover_ms=500,
            vad_pre_roll_ms=500,
        ),
    )
    await run_audio_vad_loop(
        source=Source(),
        vad=vad,
        sink=Sink(),
        target_sample_rate_hz=16000,
        segment_ledger=ledger,
        monotonic_clock=lambda: 0.0,
    )
    assert sealed.is_set()
    snapshots = ledger.snapshots
    assert [item.seal_reason for item in snapshots] == ["delivery_deadline", "source_eof"]
    assert sum(item.content_sample_count for item in snapshots) == 8 * 512
    assert snapshots[0].content_ranges[-1].normalized_end_sample == (
        snapshots[1].content_ranges[0].normalized_start_sample
    )
    assert [
        event.capture[-1].normalized_end_sample for event in stream_events if event.capture
    ] == [(index + 1) * 512 for index in range(8)]
    assert [
        event.event.genuine_onset for event in owned_events if isinstance(event.event, SpeechStart)
    ] == [True, False]


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
    assert [snapshot.prefix_context_sample_count for snapshot in snapshots] == [0, 0]
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


async def test_peer_vad_windows_distinguish_discarded_and_committed_candidates(caplog) -> None:
    vad = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.0, 0.8, 0.8, 0.0, 0.8, 0.8, 0.8, 0.0]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
    )
    frames = [
        AudioFrameF32(
            samples=np.full((vad.chunk_samples,), 0.25, dtype=np.float32),
            sample_rate_hz=16000,
        )
        for _ in range(8)
    ]
    received: list[object] = []

    class Sink:
        async def handle_vad_event(self, event: object) -> None:
            received.append(event)

    with caplog.at_level("INFO", logger="puripuly_heart.core.runtime.audio_vad_loop"):
        await run_audio_vad_loop(
            source=FakeAudioSource(frames),
            vad=vad,
            sink=Sink(),
            channel_label="peer",
            target_sample_rate_hz=16000,
            peer_diagnostic_interval_audio_ms=128,
        )

    windows = [
        dict(item.split("=", 1) for item in record.getMessage().split()[2:])
        for record in caplog.records
        if record.getMessage().startswith("[VAD] window channel=peer")
    ]
    starts = [event for event in received if isinstance(event, SpeechStart)]
    assert len(starts) == 1
    assert len(windows) == 2
    assert windows[0]["threshold_hits"] == "2"
    assert windows[0]["candidate_discarded"] == "1"
    assert float(windows[0]["candidate_max_run_ms"]) == 64
    assert windows[0]["committed"] == "0"
    assert windows[1]["threshold_hits"] == "3"
    assert windows[1]["committed"] == "1"
    assert windows[1]["candidate_discarded"] == "0"
    assert float(windows[1]["input_rms"]) == 0.25
    assert any(
        f"utterance_id={starts[0].utterance_id}" in record.getMessage()
        and "kind=onset" in record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[VAD] committed channel=peer")
    )


async def test_peer_vad_windows_include_unchanged_silence_and_unknown_fake_scores(caplog) -> None:
    class FakeVad:
        chunk_samples = 8

        def process_chunk(self, _chunk: np.ndarray) -> list[object]:
            return []

    class Sink:
        async def handle_vad_event(self, _event: object) -> None:
            raise AssertionError("no speech expected")

    with caplog.at_level("INFO", logger="puripuly_heart.core.runtime.audio_vad_loop"):
        await run_audio_vad_loop(
            source=FakeAudioSource(
                [AudioFrameF32(samples=np.zeros((32,), dtype=np.float32), sample_rate_hz=16000)]
            ),
            vad=FakeVad(),
            sink=Sink(),
            channel_label="peer",
            target_sample_rate_hz=16000,
            peer_diagnostic_interval_audio_ms=1,
        )

    windows = [
        dict(item.split("=", 1) for item in record.getMessage().split()[2:])
        for record in caplog.records
        if record.getMessage().startswith("[VAD] window channel=peer")
    ]
    assert len(windows) == 2
    assert all(window["threshold_hits"] == "0" for window in windows)
    assert all(window["max_probability"] == "unknown" for window in windows)
    assert all(window["input_rms"] == "0.00000" for window in windows)


async def test_peer_vad_window_counts_rollover_as_committed_continuation(caplog) -> None:
    vad = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.9, 0.9, 0.9, 0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
    )
    starts: list[SpeechStart] = []
    buffered_chunks = 0

    class Sink:
        async def handle_vad_event(self, event: object) -> None:
            nonlocal buffered_chunks
            if isinstance(event, SpeechStart):
                starts.append(event)
            elif isinstance(event, SpeechChunk):
                buffered_chunks += 1
                if buffered_chunks == 2:
                    assert vad.seal_active_for_rollover(reason="delivery_deadline") is not None

    with caplog.at_level("INFO", logger="puripuly_heart.core.runtime.audio_vad_loop"):
        await run_audio_vad_loop(
            source=FakeAudioSource(
                [
                    AudioFrameF32(
                        samples=np.full((4 * vad.chunk_samples,), 0.25, dtype=np.float32),
                        sample_rate_hz=16000,
                    )
                ]
            ),
            vad=vad,
            sink=Sink(),
            channel_label="peer",
            target_sample_rate_hz=16000,
            peer_diagnostic_interval_audio_ms=128,
        )

    assert len(starts) == 2
    assert starts[0].genuine_onset is True
    assert starts[1].genuine_onset is False
    windows = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[VAD] window channel=peer")
    ]
    assert any("committed=2" in window and "rollovers=1" in window for window in windows)
    assert any(
        f"utterance_id={starts[1].utterance_id}" in record.getMessage()
        and "kind=rollover" in record.getMessage()
        for record in caplog.records
    )


async def test_peer_vad_flushes_pending_candidate_before_cancellation_reaches_caller(
    caplog,
) -> None:
    class CancelledSource:
        async def frames(self):
            yield AudioFrameF32(
                samples=np.full((1024,), 0.25, dtype=np.float32), sample_rate_hz=16000
            )
            raise asyncio.CancelledError

    class Sink:
        async def handle_vad_event(self, _event: object) -> None:
            raise AssertionError("uncommitted candidate must not reach recognition")

    vad = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.8, 0.8]),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )
    with caplog.at_level("INFO", logger="puripuly_heart.core.runtime.audio_vad_loop"):
        try:
            await run_audio_vad_loop(
                source=CancelledSource(),
                vad=vad,
                sink=Sink(),
                channel_label="peer",
                target_sample_rate_hz=16000,
            )
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation must propagate")

    windows = [
        dict(item.split("=", 1) for item in record.getMessage().split()[2:])
        for record in caplog.records
        if record.getMessage().startswith("[VAD] window channel=peer")
    ]
    assert len(windows) == 1
    assert windows[0]["reason"] == "cancel"
    assert windows[0]["observed_audio_ms"] == "64"
    assert windows[0]["candidate_pending_chunks"] == "2"
    assert windows[0]["candidate_discarded"] == "0"
    assert windows[0]["committed"] == "0"


async def test_stream_input_keeps_real_missed_onset_and_eof_tail_without_vad_admission():
    class NoOnsetVad:
        chunk_samples = 8

        def process_chunk(self, _chunk: np.ndarray) -> list[object]:
            return []

    frames = [
        AudioFrameF32(
            samples=np.arange(start, end, dtype=np.float32) / 16,
            sample_rate_hz=16000,
            capture=AudioCaptureSpan(
                capture_epoch=9,
                callback_sequence=sequence,
                source_sample_rate_hz=16000,
                source_start_sample=start,
                source_end_sample=end,
                source_start_monotonic_s=start / 16000,
                source_end_monotonic_s=end / 16000,
            ),
        )
        for sequence, (start, end) in enumerate(((0, 8), (8, 12)))
    ]
    sent: list[tuple[np.ndarray, object, str | None]] = []

    class Sink:
        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"no local onset expected: {event!r}")

        async def handle_stream_input(self, event) -> None:
            sent.append((event.chunk.copy(), event.capture, event.boundary_reason))

    await run_audio_vad_loop(
        source=FakeAudioSource(frames),
        vad=NoOnsetVad(),
        sink=Sink(),
        target_sample_rate_hz=16000,
    )

    assert [reason for _pcm, _capture, reason in sent] == [None, None, "source_eof"]
    np.testing.assert_array_equal(
        np.concatenate([pcm for pcm, _capture, reason in sent if reason is None]),
        np.arange(12, dtype=np.float32) / 16,
    )
    assert [
        (span.normalized_start_sample, span.normalized_end_sample)
        for _pcm, capture, reason in sent
        if reason is None
        for span in capture
    ] == [(0, 8), (8, 12)]


async def test_capture_epoch_change_fences_before_new_source_pcm():
    class NoOnsetVad:
        chunk_samples = 8

        def process_chunk(self, _chunk: np.ndarray) -> list[object]:
            return []

    frames = [
        AudioFrameF32(
            samples=np.full((8,), float(epoch), dtype=np.float32),
            sample_rate_hz=16000,
            capture=AudioCaptureSpan(
                capture_epoch=epoch,
                callback_sequence=0,
                source_sample_rate_hz=16000,
                source_start_sample=0,
                source_end_sample=8,
                source_start_monotonic_s=0,
                source_end_monotonic_s=8 / 16000,
            ),
        )
        for epoch in (3, 4)
    ]
    received: list[tuple[int | None, str | None]] = []

    class Sink:
        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"no onset expected: {event!r}")

        async def handle_stream_input(self, event) -> None:
            received.append(
                (
                    event.capture[0].capture_epoch if event.capture else None,
                    event.boundary_reason,
                )
            )

    await run_audio_vad_loop(
        source=FakeAudioSource(frames),
        vad=NoOnsetVad(),
        sink=Sink(),
        target_sample_rate_hz=16000,
    )

    assert received == [
        (3, None),
        (None, "source_discontinuity"),
        (4, None),
        (None, "source_eof"),
    ]


async def test_muted_input_fences_stream_without_sending_synthetic_silence():
    class NoOnsetVad:
        chunk_samples = 8

        def process_chunk(self, _chunk: np.ndarray) -> list[object]:
            return []

    class Gate:
        def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
            return np.zeros_like(chunk) if chunk[0] == 0.25 else chunk

    events: list[tuple[float | None, str | None]] = []

    class Sink:
        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"no onset expected: {event!r}")

        async def handle_stream_input(self, event) -> None:
            events.append(
                (
                    float(event.chunk[0]) if event.chunk.size else None,
                    event.boundary_reason,
                )
            )

    await run_audio_vad_loop(
        source=FakeAudioSource(
            [
                AudioFrameF32(
                    samples=np.full((8,), sample, dtype=np.float32),
                    sample_rate_hz=16000,
                )
                for sample in (0.125, 0.25, 0.375)
            ]
        ),
        vad=NoOnsetVad(),
        sink=Sink(),
        target_sample_rate_hz=16000,
        audio_gate=Gate(),
    )

    assert events == [
        (0.125, None),
        (None, "source_discontinuity"),
        (0.375, None),
        (None, "source_eof"),
    ]


async def test_stream_input_carries_vad_speech_fact_with_real_source_timing_before_onset():
    frames = [
        AudioFrameF32(
            samples=np.full(end - start, 0.25, dtype=np.float32),
            sample_rate_hz=16000,
            capture=AudioCaptureSpan(
                capture_epoch=9,
                callback_sequence=sequence,
                source_sample_rate_hz=16000,
                source_start_sample=start,
                source_end_sample=end,
                source_start_monotonic_s=10 + start / 16000,
                source_end_monotonic_s=10 + end / 16000,
            ),
        )
        for sequence, (start, end) in enumerate(((0, 8), (8, 16), (16, 20)))
    ]
    vad = VadGating(
        SequenceVadEngine(probs=[0.9, 0.1, 0.9]),
        sample_rate_hz=16000,
        chunk_samples=8,
        ring_buffer_ms=1,
        hangover_ms=640,
    )
    received: list[tuple[bool, int, float]] = []
    delivery_order: list[str] = []
    activity_times: list[float] = []

    class Sink:
        async def handle_stream_input(self, event) -> None:
            if event.capture:
                received.append(
                    (
                        event.speech_observed,
                        event.chunk.size,
                        event.capture[-1].source_end_monotonic_s,
                    )
                )
                delivery_order.append("stream")

        async def handle_vad_event(self, event) -> None:
            if isinstance(event, SpeechStart):
                delivery_order.append("start")

        async def observe_source_activity(
            self, *, speech_observed: bool, observed_at_monotonic_s: float
        ) -> None:
            activity_times.append(observed_at_monotonic_s)

    await run_audio_vad_loop(
        source=FakeAudioSource(frames),
        vad=vad,
        sink=Sink(),
        target_sample_rate_hz=16000,
        monotonic_clock=lambda: 500.0,
    )

    assert received == [
        (True, 8, 10 + 8 / 16000),
        (False, 8, 10 + 16 / 16000),
        (True, 4, 10 + 20 / 16000),
    ]
    assert delivery_order == ["stream", "start", "stream", "stream"]
    assert activity_times == [500.0, 500.0, 500.0]
