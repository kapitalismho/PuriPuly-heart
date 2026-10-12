from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureDiscontinuity, AudioCaptureSpan
from puripuly_heart.core.audio.ownership import CaptureStreamInput
from puripuly_heart.core.speaker import (
    BackendReady,
    BackendUpdate,
    ProbabilityFrame,
    RuntimeLimits,
    SpeakerRuntime,
)


class DeterministicBackend:
    def __init__(self, *, tail: int = 20, stride: int = 10) -> None:
        self.tail = tail
        self.stride = stride
        self.starts = 0
        self.stops = 0
        self.resets = []
        self.pauses = 0
        self.signatures = []
        self.fed = 0
        self.frames = 0
        self.pending = np.empty(0, dtype=np.float32)
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()
        self.block_start = False
        self.failure: Exception | None = None
        self.max_pending = 0

    async def start(self):
        self.starts += 1
        if self.block_start:
            self.entered.set()
            assert self.gate is not None
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure
        return BackendReady(self.tail, self.stride - 1, (("execution", "deterministic-test-only"),))

    async def reset(self, epoch):
        self.resets.append(epoch)
        self.signatures.clear()
        self.pending = np.empty(0, dtype=np.float32)
        self.fed = self.frames = 0

    async def feed(self, pcm, compact_start_sample):
        assert not pcm.flags.writeable
        assert pcm.flags.c_contiguous
        assert compact_start_sample == self.fed
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure
        self.fed += pcm.size
        values = np.concatenate((self.pending, pcm))
        result = []
        count = values.size // self.stride
        for index in range(count):
            amplitude = round(
                float(np.mean(values[index * self.stride : (index + 1) * self.stride])), 2
            )
            probabilities = [0.03] * 8
            if amplitude > 0:
                if amplitude not in self.signatures:
                    self.signatures.append(amplitude)
                slot = self.signatures.index(amplitude)
                probabilities[slot] = 0.83
                if amplitude == 0.75:
                    probabilities[(slot + 1) % 8] = 0.67
            start = self.frames * self.stride
            result.append(ProbabilityFrame(start, start + self.stride, tuple(probabilities)))
            self.frames += 1
        self.pending = values[count * self.stride :].copy()
        self.max_pending = max(self.max_pending, self.pending.size)
        return BackendUpdate(self.frames * self.stride, self.frames * self.stride, tuple(result))

    async def pause(self):
        self.pauses += 1
        return BackendUpdate(self.frames * self.stride, self.frames * self.stride)

    async def stop(self):
        self.stops += 1


def event(start, count=20, amplitude=0.25, *, speech=True, epoch=1, rate=48_000):
    span = AudioCaptureSpan(
        epoch,
        start // 20,
        rate,
        start * rate // 16_000,
        (start + count) * rate // 16_000,
        10 + start / 16_000,
        10 + (start + count) / 16_000,
        normalized_sample_rate_hz=16_000,
        normalized_start_sample=start,
        normalized_end_sample=start + count,
    )
    return CaptureStreamInput(
        np.full(count, amplitude, np.float32), (span,), speech_observed=speech
    )


async def settled(runtime):
    for _ in range(100):
        await asyncio.sleep(0)
        snapshot = runtime.snapshot()
        if snapshot.pending_items == 0 and snapshot.state != "starting":
            await asyncio.sleep(0)
            return
    pytest.fail("speaker owner did not make bounded forward progress")


def raw(evidence):
    return [item for item in evidence if item.frame is not None]


@pytest.mark.asyncio
async def test_source_mapping_overlap_and_actual_ready_time():
    backend = DeterministicBackend()
    ready_time = [50.0]
    runtime = SpeakerRuntime(backend, clock=lambda: ready_time[0])
    await runtime.start()
    backend.gate = asyncio.Event()
    first = event(0, amplitude=0.75)
    runtime.observe_input("physical-peer", first)
    await backend.entered.wait()
    first.chunk[:] = 0
    ready_time[0] = 52.0
    backend.gate.set()
    await settled(runtime)
    evidence = raw(runtime.drain_evidence(100))
    assert len(evidence) == 2
    assert all(item.observation_ready_monotonic_s == 52.0 for item in evidence)
    assert evidence[0].frame.probabilities[:2] == (0.83, 0.67)
    assert evidence[0].frame.active_slots == ()
    assert evidence[0].source_map[0].capture.source_start_sample == 0
    assert evidence[0].source_map[0].capture.source_end_sample == 30
    assert evidence[1].source_map[0].capture.normalized_start_sample == 10
    assert evidence[1].range_state == "closed"
    assert runtime.snapshot().processed_through_sample == 20
    await runtime.stop()


@pytest.mark.asyncio
async def test_causal_onset_short_gap_tail_pause_and_returning_slot():
    backend = DeterministicBackend()
    runtime = SpeakerRuntime(backend, onset_context_samples=20, short_gap_samples=20)
    await runtime.start()
    runtime.observe_input("physical", event(0, 40, 0, speech=False))
    runtime.observe_input("physical", event(40))
    runtime.observe_input("physical", event(60, 10, 0, speech=False))
    runtime.observe_input("physical", event(70, 20, 0.5))
    await settled(runtime)
    epoch = runtime.snapshot().speaker_epoch
    initial = runtime.drain_evidence(100)
    assert any(item.reason == "omitted_silence" for item in initial)
    onset = raw(initial)[0].source_map[0].capture
    assert onset.normalized_start_sample == 20
    assert backend.pauses == 0
    runtime.observe_input("physical", event(90, 20, 0, speech=False))
    await settled(runtime)
    tail = runtime.drain_evidence(100)
    assert backend.pauses == 1
    assert any(piece.kind == "synthetic_padding" for item in raw(tail) for piece in item.source_map)
    fed_at_pause = backend.fed
    runtime.observe_input("physical", event(110, 40, 0, speech=False))
    await settled(runtime)
    assert backend.fed == fed_at_pause
    runtime.observe_input("physical", event(150, 20, 0.25))
    await settled(runtime)
    returning = raw(runtime.drain_evidence(100))
    speech = [
        item for item in returning if item.source_map[0].capture.normalized_start_sample >= 150
    ]
    assert speech[0].frame.probabilities[0] == 0.83
    assert runtime.snapshot().speaker_epoch == epoch
    assert backend.starts == 1
    assert len(backend.resets) == 1
    await runtime.stop()


@pytest.mark.asyncio
async def test_eof_resolves_causal_tail_without_future_input():
    backend = DeterministicBackend()
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    runtime.observe_input("physical", event(0))
    runtime.observe_input("physical", CaptureStreamInput(np.empty(0, np.float32), (), "source_eof"))
    await settled(runtime)
    assert backend.fed == 40
    assert backend.pauses == 1
    assert backend.stops == 0
    assert any(item.reason == "vad_pause" for item in runtime.drain_evidence(100))
    runtime.observe_input("physical", event(20))
    assert runtime.snapshot().stale_events == 1
    await runtime.stop()


@pytest.mark.asyncio
async def test_reset_new_epoch_and_stale_inflight_result():
    backend = DeterministicBackend()
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    backend.gate = asyncio.Event()
    runtime.observe_input("physical", event(0))
    await backend.entered.wait()
    old_epoch = runtime.snapshot().speaker_epoch
    runtime.reset()
    first_reset_epoch = runtime.snapshot().speaker_epoch
    runtime.reset()
    second_reset_epoch = runtime.snapshot().speaker_epoch
    assert old_epoch < first_reset_epoch < second_reset_epoch
    runtime.observe_input("physical", event(20, amplitude=0.5))
    backend.gate.set()
    await settled(runtime)
    evidence = raw(runtime.drain_evidence(100))
    assert all(item.speaker_epoch == second_reset_epoch for item in evidence)
    assert all(item.source_map[0].capture.normalized_start_sample >= 20 for item in evidence)
    assert evidence[0].frame.probabilities[0] == 0.83
    assert backend.starts == 1
    await runtime.stop()


@pytest.mark.asyncio
async def test_original_once_discontinuity_and_capture_replacement_fences():
    backend = DeterministicBackend()
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    first = event(0)
    runtime.observe_input("capture-A", first)
    await settled(runtime)
    epoch = runtime.snapshot().speaker_epoch
    runtime.observe_input("capture-A", first)
    runtime.observe_input(
        "capture-A", CaptureStreamInput(np.empty(0, np.float32), (), "asr_finalize")
    )
    runtime.observe_input("capture-A", event(20))
    await settled(runtime)
    assert backend.fed == 40
    assert runtime.snapshot().speaker_epoch == epoch
    changed = event(40)
    span = replace(
        changed.capture[0], discontinuity_before=AudioCaptureDiscontinuity("unknown_loss", 12)
    )
    runtime.observe_input("capture-A", replace(changed, capture=(span,)))
    await settled(runtime)
    assert runtime.snapshot().speaker_epoch > epoch
    runtime.end_capture("capture-A", "capture_replaced")
    runtime.observe_input("capture-B", event(0, epoch=2))
    runtime.end_capture("capture-A", "late-stop")
    runtime.observe_input("capture-A", event(60))
    await settled(runtime)
    assert runtime.snapshot().capture_session_id == "capture-B"
    assert runtime.snapshot().stale_events == 3
    await runtime.stop()


@pytest.mark.asyncio
async def test_overload_drops_backlog_and_recovers_live_with_explicit_gap():
    limits = RuntimeLimits(max_input_samples=20, max_pending_samples=40, max_pending_items=4)
    backend = DeterministicBackend(tail=0)
    runtime = SpeakerRuntime(backend, limits=limits, onset_context_samples=20)
    await runtime.start()
    backend.gate = asyncio.Event()
    runtime.observe_input("physical", event(0))
    await backend.entered.wait()
    old_epoch = runtime.snapshot().speaker_epoch
    runtime.observe_input("physical", event(20))
    runtime.observe_input("physical", event(40))
    assert runtime.snapshot().overloads == 1
    assert runtime.snapshot().pending_samples <= 40
    runtime.observe_input("physical", event(60))
    backend.gate.set()
    await settled(runtime)
    evidence = runtime.drain_evidence(100)
    gaps = [item for item in evidence if item.reason == "input_overload" and item.source_map]
    assert gaps
    assert all(item.speaker_epoch > old_epoch for item in raw(evidence))
    assert raw(evidence)[0].source_map[0].capture.normalized_start_sample == 60
    assert backend.fed == 20
    await runtime.stop()


@pytest.mark.asyncio
async def test_long_input_slow_consumer_bounded_progress_and_copying():
    limits = RuntimeLimits(
        max_input_samples=3200,
        max_pending_samples=16_000,
        max_pending_items=10,
        max_map_pieces=20,
        max_evidence_items=16,
        max_frames_per_update=32,
    )
    backend = DeterministicBackend(tail=0, stride=160)
    runtime = SpeakerRuntime(backend, limits=limits, onset_context_samples=3200)
    await runtime.start()
    for index in range(10_000):
        runtime.observe_input("physical", event(index * 3200, count=3200))
        await settled(runtime)
        if index == 0:
            epoch = runtime.snapshot().speaker_epoch
        snapshot = runtime.snapshot()
        assert snapshot.retained_map_pieces <= limits.max_map_pieces
        assert snapshot.pending_samples <= limits.max_pending_samples
        assert snapshot.evidence_items <= 16
        assert snapshot.owned_tasks == 1
    snapshot = runtime.snapshot()
    assert backend.fed == 32_000_000
    assert backend.max_pending <= 159
    assert snapshot.speaker_epoch == epoch
    assert snapshot.overloads == 0
    assert snapshot.evidence_evictions == 209_984
    assert snapshot.max_copy_samples == 3200
    assert snapshot.copied_samples == 32_000_000
    evidence = runtime.drain_evidence(10_000)
    assert len(evidence) == 16
    assert (
        max(
            piece.capture.normalized_end_sample
            for item in evidence
            for piece in item.source_map
            if piece.capture is not None
        )
        == 32_000_000
    )
    assert evidence[-1].immutable_through_sample == 32_000_000
    assert runtime.drain_evidence(10) == ()
    await runtime.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("during_start", [False, True])
async def test_stop_during_owned_work_cancels_and_closes_once(during_start):
    backend = DeterministicBackend()
    backend.gate = asyncio.Event()
    backend.block_start = during_start
    runtime = SpeakerRuntime(backend)
    start_task = asyncio.create_task(runtime.start())
    if during_start:
        await backend.entered.wait()
    else:
        await start_task
        runtime.observe_input("physical", event(0))
        await backend.entered.wait()
    await runtime.stop()
    await asyncio.gather(start_task, return_exceptions=True)
    await runtime.stop()
    runtime.observe_input("physical", event(20))
    assert runtime.snapshot().state == "stopped"
    assert runtime.snapshot().owned_tasks == 0
    assert backend.stops == 1
    assert not raw(runtime.drain_evidence(100))


@pytest.mark.asyncio
@pytest.mark.parametrize("during_work", [False, True])
async def test_backend_unavailable_preserves_explicit_failure_without_fallback(during_work):
    backend = DeterministicBackend()
    runtime = SpeakerRuntime(backend)
    if not during_work:
        backend.failure = RuntimeError("Vulkan device unavailable")
    await runtime.start()
    if during_work:
        backend.failure = RuntimeError("device lost")
        runtime.observe_input("physical", event(0))
        await settled(runtime)
        assert any(item.reason == "backend_work_failed" for item in runtime.drain_evidence(100))
    assert runtime.snapshot().state == "unavailable"
    assert runtime.snapshot().error is not None
    assert backend.starts == 1
    await runtime.stop()


@pytest.mark.asyncio
async def test_piecewise_frame_maps_across_resampling_boundaries():
    backend = DeterministicBackend(tail=0)
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    first = event(0, 5, rate=48_000)
    second = event(5, 5, rate=32_000)
    runtime.observe_input("physical", first)
    runtime.observe_input("physical", second)
    await settled(runtime)
    evidence = raw(runtime.drain_evidence(100))
    assert len(evidence) == 1
    assert len(evidence[0].source_map) == 2
    assert evidence[0].source_map[0].capture.source_sample_rate_hz == 48_000
    assert evidence[0].source_map[1].capture.source_sample_rate_hz == 32_000
    assert evidence[0].processed_through_sample == 10
    await runtime.stop()


@pytest.mark.asyncio
async def test_known_lost_source_gap_and_map_horizon_failure():
    backend = DeterministicBackend(tail=0)
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    runtime.observe_input("physical", event(0))
    await settled(runtime)
    epoch = runtime.snapshot().speaker_epoch
    runtime.observe_input("physical", event(40))
    await settled(runtime)
    assert runtime.snapshot().speaker_epoch > epoch
    assert any(item.reason == "source_coverage_gap" for item in runtime.drain_evidence(100))
    await runtime.stop()

    class StalledBackend(DeterministicBackend):
        async def feed(self, pcm, compact_start_sample):
            return BackendUpdate(0, 0)

    stalled = SpeakerRuntime(StalledBackend(tail=0))
    await stalled.start()
    stalled.observe_input("physical", event(0))
    await settled(stalled)
    assert stalled.snapshot().state == "unavailable"
    assert "mapping horizon" in stalled.snapshot().error
    await stalled.stop()


@pytest.mark.asyncio
async def test_open_revision_and_true_consumed_frontier_not_submission():
    class RevisingBackend(DeterministicBackend):
        async def start(self):
            return BackendReady(0, 100)

        async def feed(self, pcm, compact_start_sample):
            if compact_start_sample == 0:
                return BackendUpdate(10, 0, (ProbabilityFrame(0, 10, (0.3,) * 8),))
            return BackendUpdate(
                40,
                40,
                (
                    ProbabilityFrame(0, 10, (0.7,) * 8),
                    ProbabilityFrame(10, 40, (0.4,) * 8),
                ),
            )

    runtime = SpeakerRuntime(RevisingBackend())
    await runtime.start()
    runtime.observe_input("physical", event(0))
    await settled(runtime)
    first = raw(runtime.drain_evidence(100))
    assert first[0].range_state == "open"
    assert runtime.snapshot().processed_through_sample == 10
    assert runtime.snapshot().immutable_through_sample == 0
    runtime.observe_input("physical", event(20))
    await settled(runtime)
    revision = raw(runtime.drain_evidence(100))
    assert revision[0].source_map == first[0].source_map
    assert revision[0].frame.probabilities == (0.7,) * 8
    assert all(item.range_state == "closed" for item in revision)
    assert runtime.snapshot().retained_map_pieces == 0
    await runtime.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", ["too_many", "nan", "history_replay"])
async def test_unbounded_or_invalid_probability_output_is_explicit_failure(malformed):
    class InvalidBackend(DeterministicBackend):
        async def feed(self, pcm, compact_start_sample):
            update = await super().feed(pcm, compact_start_sample)
            if compact_start_sample == 0:
                return update
            if malformed == "too_many":
                return replace(update, frames=update.frames * 3)
            if malformed == "nan":
                return replace(
                    update,
                    frames=(
                        update.frames[0],
                        replace(update.frames[1], probabilities=(float("nan"),) * 8),
                    ),
                )
            return replace(update, frames=(ProbabilityFrame(0, 10, (0.5,) * 8),))

    runtime = SpeakerRuntime(InvalidBackend(tail=0), limits=RuntimeLimits(max_frames_per_update=4))
    await runtime.start()
    runtime.observe_input("physical", event(0))
    await settled(runtime)
    runtime.drain_evidence(100)
    runtime.observe_input("physical", event(20))
    await settled(runtime)
    assert runtime.snapshot().state == "unavailable"
    evidence = runtime.drain_evidence(100)
    assert not raw(evidence)
    assert any(item.reason == "backend_work_failed" for item in evidence)
    await runtime.stop()


@pytest.mark.asyncio
async def test_long_silence_bounded_onset_without_identity_aging():
    backend = DeterministicBackend()
    runtime = SpeakerRuntime(backend, onset_context_samples=20, short_gap_samples=20)
    await runtime.start()
    runtime.observe_input("physical", event(0))
    runtime.observe_input("physical", event(20, amplitude=0, speech=False))
    await settled(runtime)
    epoch = runtime.snapshot().speaker_epoch
    fed = backend.fed
    for index in range(1000):
        runtime.observe_input("physical", event(40 + index * 20, amplitude=0, speech=False))
        snapshot = runtime.snapshot()
        assert snapshot.onset_samples <= 20
        assert snapshot.pending_samples == 0
        assert snapshot.retained_map_pieces <= snapshot.limits.max_map_pieces
    assert backend.fed == fed
    assert runtime.snapshot().speaker_epoch == epoch
    runtime.observe_input("physical", event(20_040))
    await settled(runtime)
    returned = raw(runtime.drain_evidence(5000))
    assert returned[-1].frame.probabilities[0] == 0.83
    assert backend.starts == 1
    assert len(backend.resets) == 1
    await runtime.stop()


@pytest.mark.asyncio
async def test_internal_discontinuity_cannot_link_slots_across_lost_coverage():
    backend = DeterministicBackend(tail=0)
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    first = event(0)
    second = event(20, amplitude=0.5)
    second_span = replace(
        second.capture[0], discontinuity_before=AudioCaptureDiscontinuity("known_loss", 10, 40)
    )
    combined = CaptureStreamInput(
        np.concatenate((first.chunk, second.chunk)),
        (first.capture[0], second_span),
        speech_observed=True,
    )
    runtime.observe_input("physical", combined)
    await settled(runtime)
    evidence = runtime.drain_evidence(100)
    assert any(item.reason == "source_coverage_gap" and item.source_map for item in evidence)
    results = raw(evidence)
    gap = next(item for item in evidence if item.reason == "source_coverage_gap")
    assert results[0].speaker_epoch > gap.speaker_epoch
    assert results[0].source_map[0].capture.normalized_start_sample == 20
    assert results[0].frame.probabilities[0] == 0.83
    await runtime.stop()


@pytest.mark.asyncio
async def test_session_tombstones_are_bounded_without_accepting_stale_sessions():
    runtime = SpeakerRuntime(
        DeterministicBackend(tail=0), limits=RuntimeLimits(max_capture_sessions=2)
    )
    await runtime.start()
    for session in ("one", "two"):
        runtime.observe_input(session, event(0))
        await settled(runtime)
        runtime.end_capture(session, "replacement")
    runtime.observe_input("one", event(20))
    assert runtime.snapshot().stale_events == 1
    runtime.observe_input("three", event(0))
    assert runtime.snapshot().capture_session_id is None
    assert runtime.snapshot().error == "capture_session_capacity_exhausted"
    await runtime.stop()


@pytest.mark.asyncio
async def test_late_original_capture_epoch_and_discontinuity_flag_do_not_reset():
    backend = DeterministicBackend(tail=0)
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    runtime.observe_input("physical", event(0, epoch=2))
    await settled(runtime)
    epoch = runtime.snapshot().speaker_epoch
    runtime.observe_input(
        "physical", replace(event(0, epoch=1), boundary_reason="source_discontinuity")
    )
    runtime.observe_input(
        "physical", replace(event(0, epoch=2), boundary_reason="source_discontinuity")
    )
    await settled(runtime)
    assert runtime.snapshot().speaker_epoch == epoch
    assert runtime.snapshot().stale_events == 2
    assert backend.fed == 20
    await runtime.stop()


@pytest.mark.asyncio
async def test_stop_does_not_schedule_another_stream_reset():
    backend = DeterministicBackend()
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    runtime.observe_input("physical", event(0))
    await settled(runtime)
    before = tuple(backend.resets)
    await runtime.stop()
    assert tuple(backend.resets) == before
    assert backend.stops == 1


@pytest.mark.asyncio
async def test_returned_mapping_budget_is_independent_of_segment_count():
    limits = RuntimeLimits(max_map_pieces=8, max_evidence_map_pieces=8)
    backend = DeterministicBackend(tail=0)
    runtime = SpeakerRuntime(backend, limits=limits)
    await runtime.start()
    for index in range(20):
        runtime.observe_input("physical", event(index * 20))
        await settled(runtime)
        if index == 0:
            epoch = runtime.snapshot().speaker_epoch
        assert runtime.snapshot().evidence_map_pieces <= 8
    snapshot = runtime.snapshot()
    assert snapshot.evidence_items <= limits.max_evidence_items
    assert snapshot.evidence_evictions > 0
    assert snapshot.speaker_epoch == epoch
    assert backend.fed == 400
    evidence = runtime.drain_evidence(100)
    assert sum(len(item.source_map) for item in evidence) == 8
    assert runtime.snapshot().evidence_map_pieces == 0
    await runtime.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", [False, True])
async def test_immediate_eof_end_drains_owned_native_like_pending_work(blocked):
    backend = DeterministicBackend(tail=80, stride=40)
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    if blocked:
        backend.gate = asyncio.Event()
    runtime.observe_input("physical", event(0))
    if blocked:
        await backend.entered.wait()
    else:
        await settled(runtime)
        assert runtime.snapshot().processed_through_sample == 0
        assert runtime.snapshot().immutable_through_sample == 0
    epoch = runtime.snapshot().speaker_epoch
    runtime.observe_input("physical", CaptureStreamInput(np.empty(0, np.float32), (), "source_eof"))
    runtime.end_capture("physical", "source_eof")
    runtime.end_capture("physical", "source_eof")
    assert runtime.snapshot().speaker_epoch == epoch
    assert runtime.snapshot().capture_session_id == "physical"
    assert runtime.snapshot().owned_tasks == 1
    assert runtime.snapshot().stale_events == 1
    if blocked:
        assert backend.fed == 0
        backend.gate.set()
    await settled(runtime)
    evidence = runtime.drain_evidence(100)
    frames = raw(evidence)
    original = [
        piece.capture for item in frames for piece in item.source_map if piece.capture is not None
    ]
    assert sum(span.normalized_sample_count for span in original) == 20
    assert all(item.speaker_epoch == epoch for item in frames)
    assert any(piece.kind == "synthetic_padding" for item in frames for piece in item.source_map)
    assert any(item.reason == "source_eof" for item in evidence)
    assert not any(item.reason == "source_eof_pending_coverage" for item in evidence)
    assert runtime.snapshot().capture_session_id is None
    assert runtime.snapshot().speaker_epoch == epoch
    assert backend.pauses == 1
    assert len(backend.resets) == 1
    assert backend.starts == 1
    assert backend.stops == 0
    runtime.end_capture("physical", "source_eof")
    assert runtime.snapshot().stale_events == 2
    await runtime.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["replacement", "manual_reset", "stop"])
async def test_pending_graceful_eof_is_immediately_fenced_by_new_authority(fence):
    backend = DeterministicBackend(tail=20)
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    backend.gate = asyncio.Event()
    runtime.observe_input("old-physical", event(0))
    await backend.entered.wait()
    epoch = runtime.snapshot().speaker_epoch
    runtime.observe_input(
        "old-physical", CaptureStreamInput(np.empty(0, np.float32), (), "source_eof")
    )
    runtime.end_capture("old-physical", "source_eof")
    if fence == "stop":
        await runtime.stop()
        assert not raw(runtime.drain_evidence(100))
        assert runtime.snapshot().owned_tasks == 0
        return
    if fence == "manual_reset":
        runtime.reset()
        assert runtime.snapshot().capture_session_id is None
    runtime.observe_input("new-physical", event(0, amplitude=0.5, epoch=2))
    assert runtime.snapshot().speaker_epoch > epoch
    runtime.end_capture("old-physical", "source_eof")
    backend.gate.set()
    await settled(runtime)
    frames = raw(runtime.drain_evidence(100))
    assert frames
    assert all(item.capture_session_id == "new-physical" for item in frames)
    assert all(item.speaker_epoch > epoch for item in frames)
    assert backend.pauses == 0
    assert runtime.snapshot().capture_session_id == "new-physical"
    await runtime.stop()


@pytest.mark.asyncio
async def test_new_capture_after_completed_graceful_eof_starts_fresh_compact_axis():
    backend = DeterministicBackend(tail=20)
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    runtime.observe_input("old-physical", event(0))
    runtime.observe_input(
        "old-physical", CaptureStreamInput(np.empty(0, np.float32), (), "source_eof")
    )
    runtime.end_capture("old-physical", "source_eof")
    await settled(runtime)
    old = raw(runtime.drain_evidence(100))
    assert runtime.snapshot().capture_session_id is None
    assert backend.fed == 40
    epoch = old[0].speaker_epoch
    runtime.observe_input("new-physical", event(0, amplitude=0.5, epoch=2))
    await settled(runtime)
    new = raw(runtime.drain_evidence(100))
    assert new[0].frame.start_sample == 0
    assert new[0].speaker_epoch > epoch
    assert new[0].capture_session_id == "new-physical"
    assert backend.fed == 20
    assert runtime.snapshot().ready
    assert backend.starts == 1
    assert len(backend.resets) == 2
    await runtime.stop()


@pytest.mark.asyncio
async def test_failed_cleanup_retains_product_owner_until_successful_retry():
    class RetriableBackend(DeterministicBackend):
        async def stop(self):
            self.stops += 1
            if self.stops == 1:
                raise RuntimeError("native child cleanup failed")

    backend = RetriableBackend()
    runtime = SpeakerRuntime(backend)
    owner = runtime
    await runtime.start()
    runtime.observe_input("physical", event(0))
    await settled(runtime)
    epoch = runtime.snapshot().speaker_epoch

    async def disable():
        nonlocal owner
        await owner.stop()
        owner = None

    with pytest.raises(RuntimeError, match="native child cleanup failed"):
        await disable()
    assert owner is runtime
    snapshot = runtime.snapshot()
    assert snapshot.state == "cleanup_pending"
    assert not snapshot.ready
    assert snapshot.error == "backend_stop_failed: native child cleanup failed"
    assert snapshot.owned_tasks == 0
    assert snapshot.capture_session_id is None
    cleanup_epoch = snapshot.speaker_epoch
    assert cleanup_epoch > epoch
    runtime.observe_input("new-physical", event(0, epoch=2))
    assert runtime.snapshot().capture_session_id is None
    runtime.reset()
    assert runtime.snapshot().speaker_epoch == cleanup_epoch
    await disable()
    assert owner is None
    assert runtime.snapshot().state == "stopped"
    assert runtime.snapshot().error is None
    assert runtime.snapshot().speaker_epoch == cleanup_epoch
    assert backend.stops == 2
    await runtime.stop()
    assert backend.stops == 2


@pytest.mark.asyncio
async def test_concurrent_stop_callers_share_cleanup_despite_caller_cancellation():
    class GatedBackend(DeterministicBackend):
        def __init__(self):
            super().__init__()
            self.stop_entered = asyncio.Event()
            self.stop_gate = asyncio.Event()

        async def stop(self):
            self.stops += 1
            self.stop_entered.set()
            await self.stop_gate.wait()

    backend = GatedBackend()
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    first = asyncio.create_task(runtime.stop())
    await backend.stop_entered.wait()
    second = asyncio.create_task(runtime.stop())
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert runtime.snapshot().state == "stopping"
    assert runtime.snapshot().owned_tasks == 1
    assert backend.stops == 1
    runtime.observe_input("physical", event(0))
    assert runtime.snapshot().capture_session_id is None
    backend.stop_gate.set()
    await second
    assert runtime.snapshot().state == "stopped"
    assert runtime.snapshot().owned_tasks == 0
    assert backend.stops == 1
    await runtime.stop()
    assert backend.stops == 1


@pytest.mark.asyncio
async def test_cancelled_stop_caller_leaves_failed_background_cleanup_retryable():
    class GatedRetriableBackend(DeterministicBackend):
        def __init__(self):
            super().__init__()
            self.stop_entered = asyncio.Event()
            self.stop_gate = asyncio.Event()

        async def stop(self):
            self.stops += 1
            if self.stops == 1:
                self.stop_entered.set()
                await self.stop_gate.wait()
                raise RuntimeError("delayed cleanup failure")

    backend = GatedRetriableBackend()
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    caller = asyncio.create_task(runtime.stop())
    await backend.stop_entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    backend.stop_gate.set()
    for _ in range(100):
        await asyncio.sleep(0)
        if runtime.snapshot().state == "cleanup_pending":
            break
    assert runtime.snapshot().state == "cleanup_pending"
    assert "delayed cleanup failure" in runtime.snapshot().error
    await runtime.stop()
    assert backend.stops == 2
    assert runtime.snapshot().state == "stopped"
    assert runtime.snapshot().error is None


@pytest.mark.asyncio
async def test_backend_cleanup_cancellation_is_debt_not_stopped_success():
    class CancelledBackend(DeterministicBackend):
        async def stop(self):
            self.stops += 1
            if self.stops == 1:
                raise asyncio.CancelledError()

    backend = CancelledBackend()
    runtime = SpeakerRuntime(backend)
    await runtime.start()
    with pytest.raises(asyncio.CancelledError):
        await runtime.stop()
    assert runtime.snapshot().state == "cleanup_pending"
    assert runtime.snapshot().error == "backend_stop_cancelled"
    await runtime.stop()
    assert runtime.snapshot().state == "stopped"
    assert runtime.snapshot().error is None
    assert backend.stops == 2


@pytest.mark.asyncio
async def test_shared_epoch_allocator_fences_interleaved_runtime_identities():
    first_backend = DeterministicBackend(tail=0)
    second_backend = DeterministicBackend(tail=0)
    first = SpeakerRuntime(first_backend)
    second = SpeakerRuntime(second_backend)
    await first.start()
    await second.start()
    try:
        first.observe_input("same-physical", event(0))
        second.observe_input("same-physical", event(20, amplitude=0.5))
        await settled(first)
        await settled(second)
        old_first = raw(first.drain_evidence(100))[0]
        old_second = raw(second.drain_evidence(100))[0]
        assert old_first.speaker_epoch < old_second.speaker_epoch
        assert old_first.frame.probabilities[0] == old_second.frame.probabilities[0] == 0.83
        first.reset()
        first.observe_input("same-physical", event(20, amplitude=0.5))
        await settled(first)
        new_first = raw(first.drain_evidence(100))[0]
        assert old_second.speaker_epoch < new_first.speaker_epoch
        assert second.snapshot().speaker_epoch == old_second.speaker_epoch
        second.reset()
        second.observe_input("same-physical", event(40))
        await settled(second)
        new_second = raw(second.drain_evidence(100))[0]
        assert new_first.speaker_epoch < new_second.speaker_epoch
        assert (
            len(
                {
                    old_first.speaker_epoch,
                    old_second.speaker_epoch,
                    new_first.speaker_epoch,
                    new_second.speaker_epoch,
                }
            )
            == 4
        )
    finally:
        await first.stop()
        await second.stop()
