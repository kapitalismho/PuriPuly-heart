from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan, AudioFrameF32
from puripuly_heart.core.audio.ownership import CaptureStreamInput
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.runtime.peer_channel import _CaptureGeneration, _GenerationGuardedVadSink
from puripuly_heart.core.speaker import RuntimeLimits, SpeakerRuntime
from puripuly_heart.core.vad.gating import VadGating
from tests.core.runtime.test_peer_capture_session import (
    FakeVadSink,
    make_config,
    make_owner,
    wait_until,
)
from tests.core.speaker.test_runtime import DeterministicBackend, event
from tests.helpers.vad import SequenceVadEngine


class Observer:
    def __init__(self):
        self.inputs = []
        self.ends = []

    def observe_input(self, capture_session_id, event):
        self.inputs.append((capture_session_id, event))

    def end_capture(self, capture_session_id, reason):
        self.ends.append((capture_session_id, reason))


@pytest.mark.asyncio
async def test_original_stream_fanout_precedes_provider_route_and_readiness():
    observer = Observer()
    generation = _CaptureGeneration(1)
    runtime = SimpleNamespace(
        _speaker_observer=observer, is_current_generation=lambda _: True, segment_ledger=None
    )
    sink = _GenerationGuardedVadSink(
        sink=object(),
        runtime=runtime,
        capture_generation=generation,
        provider_ingress_ready=asyncio.Event(),
    )
    pcm = np.ones(512, dtype=np.float32)
    first = CaptureStreamInput(pcm, (), speech_observed=True)
    await sink.handle_stream_input(first)
    generation.value = 2
    await sink.handle_stream_input(CaptureStreamInput(pcm, (), speech_observed=False))
    assert len(observer.inputs) == 2
    assert observer.inputs[0][1] is first
    assert observer.inputs[0][0] == observer.inputs[1][0]
    assert observer.inputs[0][1].speech_observed is True
    assert observer.inputs[1][1].speech_observed is False
    await sink.abort()


@pytest.mark.asyncio
async def test_speaker_failure_does_not_abort_original_capture():
    class BrokenObserver(Observer):
        def observe_input(self, *_):
            raise RuntimeError("unavailable GPU")

    runtime = SimpleNamespace(
        _speaker_observer=BrokenObserver(),
        is_current_generation=lambda _: True,
        segment_ledger=None,
    )
    sink = _GenerationGuardedVadSink(
        sink=object(),
        runtime=runtime,
        capture_generation=_CaptureGeneration(1),
        provider_ingress_ready=asyncio.Event(),
    )
    await sink.handle_stream_input(
        CaptureStreamInput(np.ones(512, dtype=np.float32), (), speech_observed=True)
    )
    await sink.abort()


@pytest.mark.asyncio
async def test_six_second_rollover_has_no_speaker_replay_or_detach():
    observer = Observer()
    eof = asyncio.Event()
    release_source = asyncio.Event()
    total_samples = 16_000 * 7

    class Source:
        terminal_reason = None

        async def frames(self):
            for start in range(0, total_samples, 512):
                end = min(start + 512, total_samples)
                yield AudioFrameF32(
                    samples=np.arange(start, end, dtype=np.float32),
                    sample_rate_hz=16_000,
                    capture=AudioCaptureSpan(
                        capture_epoch=0,
                        callback_sequence=start // 512,
                        source_sample_rate_hz=16_000,
                        source_start_sample=start,
                        source_end_sample=end,
                        source_start_monotonic_s=start / 16_000,
                        source_end_monotonic_s=end / 16_000,
                    ),
                )
            eof.set()
            await release_source.wait()

        async def close(self):
            release_source.set()

    owner, *_ = make_owner(
        source_factory=lambda *_: Source(),
        run_audio_loop=run_audio_vad_loop,
        speaker_observer=observer,
        vad_factory=lambda _: VadGating(
            SequenceVadEngine(probs=[0.9] * 300),
            sample_rate_hz=16000,
            chunk_samples=512,
            ring_buffer_ms=500,
            hangover_ms=900,
            external_delivery_boundaries=True,
            hard_rollover_pre_roll_ms=300,
        ),
    )
    try:
        await owner.apply_intent(make_config(provider_id="local_qwen"), enabled=True)
        await asyncio.wait_for(eof.wait(), 1)
        assert observer.ends == []
        chunks = [event.chunk for _, event in observer.inputs if len(event.chunk)]
        fed = np.concatenate(chunks)
        assert len(fed) == total_samples - total_samples % 512
        assert np.array_equal(fed, np.arange(len(fed), dtype=np.float32))
        assert len({session for session, _ in observer.inputs}) == 1
        assert owner.segment_ledgers[-1].snapshots[0].seal_reason == "delivery_deadline"
        session = observer.inputs[0][0]
        owner._rebind_capture_generation(owner.snapshot.generation)
        assert observer.ends == []
        release_source.set()
        await wait_until(lambda: bool(observer.ends))
        assert observer.ends[0][0] == session
        chunks = [event.chunk for _, event in observer.inputs if len(event.chunk)]
        assert np.array_equal(np.concatenate(chunks), np.arange(total_samples, dtype=np.float32))
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_physical_capture_stop_detaches_and_replacement_gets_new_session():
    observer = Observer()

    async def loop(*, sink, **_):
        await sink.handle_stream_input(
            CaptureStreamInput(np.ones(512, dtype=np.float32), (), speech_observed=True)
        )
        await asyncio.Event().wait()

    owner, *_ = make_owner(run_audio_loop=loop, speaker_observer=observer)
    try:
        await owner.apply_intent(make_config(), enabled=True)
        await wait_until(lambda: len(observer.inputs) == 1)
        first = observer.inputs[0][0]
        owner._rebind_capture_generation(owner.snapshot.generation)
        assert observer.ends == []
        await owner.apply_intent(make_config(), enabled=False)
        assert observer.ends == [(first, "capture_stopped")]
        await owner.apply_intent(make_config(), enabled=True)
        await wait_until(lambda: len(observer.inputs) == 2)
        second = observer.inputs[1][0]
        assert second != first
        assert observer.ends == [(first, "capture_stopped")]
    finally:
        await owner.close()
    assert observer.ends[-1] == (second, "capture_stopped")


@pytest.mark.asyncio
async def test_overloaded_stalled_speaker_does_not_block_ready_live_asr():
    release = asyncio.Event()
    backend = DeterministicBackend()
    backend.gate = release
    feeding = backend.entered
    captured = asyncio.Event()

    class StreamSink(FakeVadSink):
        def __init__(self):
            super().__init__()
            self.inputs = []

        async def handle_stream_input(self, event):
            self.inputs.append(event)

    speaker = SpeakerRuntime(
        backend, limits=RuntimeLimits(max_input_samples=512, max_pending_samples=8192)
    )
    sink = StreamSink()
    events = [event(index * 512, count=512) for index in range(20)]

    async def capture_loop(**kwargs):
        guard = kwargs["sink"]
        await guard.handle_stream_input(events[0])
        await feeding.wait()
        for input_event in events[1:]:
            await guard.handle_stream_input(input_event)
        captured.set()
        await release.wait()

    owner, *_ = make_owner(run_audio_loop=capture_loop, sink=sink, speaker_observer=speaker)
    try:
        await speaker.start()
        await owner.apply_intent(make_config(provider_id="gemini_transcribe"), enabled=True)
        await asyncio.wait_for(captured.wait(), 1)
        await wait_until(lambda: len(sink.inputs) == len(events))
        assert [item.event for item in sink.inputs] == events
        assert speaker.snapshot().overloads > 0
    finally:
        release.set()
        await owner.close()
        await speaker.stop()
