from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.vad.gating import VadGating
from puripuly_heart.domain.events import STTSessionState
from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend
from tests.helpers.audio import FakeAudioSource, make_frames
from tests.helpers.client_hub import compose_client_hub
from tests.helpers.fakes import FakeSender, SpeechAwareFakeBackend, SpeechAwareFakeSession
from tests.helpers.vad import SequenceVadEngine


async def test_audio_vad_loop_pipeline_smoke():
    clock = FakeClock()
    sender = FakeSender()
    osc = ChatboxPaginator(sender=sender, clock=clock)

    stt = ManagedSTTProvider(backend=SpeechAwareFakeBackend(), sample_rate_hz=16000, clock=clock)
    hub = compose_client_hub(stt=stt, llm=None, osc=osc, clock=clock, fallback_transcript_only=True)
    await hub.start(auto_flush_osc=False)

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
    await run_audio_vad_loop(source=source, vad=vad, sink=hub, target_sample_rate_hz=16000)

    for _ in range(50):
        if "FINAL" in sender.sent:
            break
        await asyncio.sleep(0.01)

    assert "FINAL" in sender.sent
    await hub.stop()


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
    def __init__(self, hub: ClientHub) -> None:
        self._hub = hub

    async def handle_vad_event(self, event) -> None:  # noqa: ANN001
        await self._hub.handle_peer_vad_event(event)


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
    hub = compose_client_hub(stt=None, peer_stt=peer_stt, llm=None, osc=osc, clock=clock)
    await hub.start(auto_flush_osc=False)

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
        sink=_PeerOnlySink(hub),
        target_sample_rate_hz=16000,
    )

    assert backend.open_calls == 0
    assert peer_stt.state == STTSessionState.DISCONNECTED
    assert hub.peer_runtime.utterances == {}

    await hub.stop()


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
    hub = compose_client_hub(stt=None, peer_stt=peer_stt, llm=None, osc=osc, clock=clock)
    await hub.start(auto_flush_osc=False)

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
        sink=_PeerOnlySink(hub),
        target_sample_rate_hz=16000,
    )

    for _ in range(50):
        if hub.peer_runtime.utterances:
            break
        await asyncio.sleep(0.01)

    assert backend.open_calls == 1
    assert hub.peer_runtime.utterances

    await hub.stop()
