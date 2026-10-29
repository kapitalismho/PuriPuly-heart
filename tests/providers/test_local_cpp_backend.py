from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from puripuly_heart.core.asr_worker import (
    ASRWorkerActivation,
    ASRWorkerClosedError,
    ASRWorkerDevice,
    ASRWorkerTranscription,
)
from puripuly_heart.core.local_asr.local_stt_assets import ORUKEET_MODEL_ID
from puripuly_heart.core.stt.backend import STTSessionProjection
from puripuly_heart.providers.stt.local_cpp import LocalCppSTTBackend
from tests.providers.test_local_cpu_backends import _scoped_request


class NativeWorker:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.is_closed = False
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.running = False
        self.calls = 0
        self.cancelled = []
        self.activation = None
        self.events = asyncio.Queue()
        self.audio = []

    async def activate(self, **options):
        self.activation = options
        return ASRWorkerActivation(ASRWorkerDevice("cpu", 0, "CPU", "CPU", "cpu", 0, 0), 0.0, 0.0)

    async def transcribe(self, *, request_id, channel, audio_path, language_hint=None):
        assert audio_path.is_file()
        assert channel == self.activation["channel"]
        with wave.open(str(audio_path), "rb") as wav:
            self.audio.append(
                (
                    wav.getparams(),
                    np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy(),
                )
            )
        self.calls += 1
        self.running = True
        self.started.set()
        try:
            await self.finish.wait()
            return ASRWorkerTranscription("completed speech", None, 0.1, 0.2, 2.0)
        finally:
            self.running = False

    async def cancel(self, request_id):
        self.cancelled.append(request_id)

    async def next_event(self):
        value = await self.events.get()
        raise ASRWorkerClosedError("closed") from value

    async def close(self):
        assert not self.running
        self.is_closed = True
        self.events.put_nowait(None)


class WorkerFactory:
    def __init__(self):
        self.workers = []

    async def start(self, **_options):
        worker = NativeWorker(1000 + len(self.workers))
        self.workers.append(worker)
        return worker

    async def close_pending_startups(self):
        return None


def backend(tmp_path, channel, factory, monkeypatch):
    instance = LocalCppSTTBackend(
        model_dir=tmp_path,
        model_id=ORUKEET_MODEL_ID,
        provider_id="local_orukeet",
        stream_label=channel,
        process_factory=factory,
    )
    monkeypatch.setattr(instance, "_validate_runtime_assets", lambda: None)
    return instance


@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_cpu_worker_receives_ordered_mono_16khz_pcm16_signal(
    tmp_path: Path, monkeypatch, channel: str
):
    factory = WorkerFactory()
    provider = backend(tmp_path, channel, factory, monkeypatch)
    signals = [
        np.array([-1.0, 0.25, -0.75, 0.0, 1.0, -0.125, 0.75], dtype=np.float32),
        np.array([0.5, -0.25, 0.125, -0.5, 0.0], dtype=np.float32),
    ]
    expected_pcm = [
        [-32768, 8192, -24576, 0, 32767, -4096, 24576],
        [16384, -8192, 4096, -16384, 0],
    ]
    first = asyncio.create_task(provider.decode_f32(signals[0]))
    while not factory.workers:
        await asyncio.sleep(0)
    worker = factory.workers[0]
    worker.finish.set()
    try:
        assert await first == "completed speech"
        assert await provider.decode_f32(signals[1]) == "completed speech"
    finally:
        await provider.close()
    assert len(worker.audio) == len(expected_pcm)
    for turn, ((params, samples), expected) in enumerate(
        zip(worker.audio, expected_pcm, strict=True)
    ):
        assert (params.nchannels, params.framerate, params.sampwidth, params.comptype) == (
            1,
            16000,
            2,
            "NONE",
        )
        assert params.nframes == len(expected)
        np.testing.assert_allclose(
            samples,
            expected,
            rtol=0,
            atol=1,
            err_msg=f"CPU worker staged WAV signal mismatch: channel={channel}, turn={turn}",
        )


async def test_cpu_channels_load_lazily_and_remain_independent_during_stop(
    tmp_path: Path, monkeypatch
):
    factory = WorkerFactory()
    talk = backend(tmp_path, "self", factory, monkeypatch)
    listen = backend(tmp_path, "peer", factory, monkeypatch)
    assert factory.workers == []
    first = asyncio.create_task(talk.decode_f32(np.ones(160, dtype=np.float32)))
    while not factory.workers:
        await asyncio.sleep(0)
    talk_worker = factory.workers[0]
    await talk_worker.started.wait()
    assert len(factory.workers) == 1
    second = asyncio.create_task(listen.decode_f32(np.ones(160, dtype=np.float32)))
    while len(factory.workers) != 2:
        await asyncio.sleep(0)
    listen_worker = factory.workers[1]
    await listen_worker.started.wait()
    assert talk_worker.running and listen_worker.running
    assert talk_worker.activation["model_path"] == listen_worker.activation["model_path"]
    assert talk_worker.activation["n_threads"] == listen_worker.activation["n_threads"] == 2
    assert talk_worker.activation["backend"] == listen_worker.activation["backend"] == "cpu"
    closing = asyncio.create_task(talk.close())
    await asyncio.sleep(0)
    assert not closing.done()
    assert not talk_worker.is_closed
    listen_worker.finish.set()
    assert await second == "completed speech"
    assert not listen_worker.is_closed
    third = asyncio.create_task(listen.decode_f32(np.ones(160, dtype=np.float32)))
    assert await third == "completed speech"
    talk_worker.finish.set()
    await first
    await closing
    assert talk_worker.is_closed
    assert listen_worker.calls == 2
    assert not listen_worker.is_closed
    await listen.close()
    assert listen_worker.is_closed


async def test_cancelled_started_cpu_decode_drains_before_releasing_model_or_audio(
    tmp_path: Path, monkeypatch
):
    factory = WorkerFactory()
    provider = backend(tmp_path, "self", factory, monkeypatch)
    decode = asyncio.create_task(provider.decode_f32(np.ones(160, dtype=np.float32)))
    while not factory.workers:
        await asyncio.sleep(0)
    worker = factory.workers[0]
    await worker.started.wait()
    decode.cancel()
    closing = asyncio.create_task(provider.close())
    await asyncio.sleep(0.01)
    assert not decode.done()
    assert not closing.done()
    assert not worker.is_closed
    assert len(worker.cancelled) == 1
    worker.finish.set()
    with pytest.raises(asyncio.CancelledError):
        await decode
    await closing
    assert worker.is_closed


async def test_cpu_queued_cancel_does_not_compute_or_publish_the_retired_turn(
    tmp_path: Path, monkeypatch
):
    factory = WorkerFactory()
    provider = backend(tmp_path, "peer", factory, monkeypatch)
    session = await provider.open_session(projection=STTSessionProjection("scoped", "local-epoch"))
    worker = factory.workers[0]
    first, queued = _scoped_request(1), _scoped_request(2)
    for request in (first, queued):
        await session.begin_turn(request)
        await session.send_turn_audio(
            request.identity,
            b"\x00\x40" * 160,
            payload_sequence=1,
            source_ranges=(),
            context_only=False,
        )
        await session.seal_turn(
            request.identity,
            sealed_content_ranges=(),
            seal_reason="silence",
            observed_trailing_silence_ms=800,
        )
    await worker.started.wait()
    await session.abort_turn(queued.identity, reason="revoked")
    events = session.turn_events()
    cancelled = await anext(events)
    assert cancelled.identity == queued.identity
    assert cancelled.outcome == "cancelled"
    worker.finish.set()
    final = await anext(events)
    assert final.identity == first.identity
    assert final.outcome == "final"
    await session.close()
    assert worker.calls == 1
    await provider.close()
