from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from puripuly_heart.app.headless_mic import run_audio_vad_loop
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.osc.smart_queue import SmartOscQueue
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.vad.gating import VadGating


@dataclass(slots=True)
class FakeAudioSource:
    frames_list: list[AudioFrameF32]

    async def frames(self):
        for item in self.frames_list:
            yield item

    async def close(self) -> None:
        return


@dataclass(slots=True)
class SequenceVadEngine:
    probs: list[float]
    idx: int = 0

    def speech_probability(self, _samples: np.ndarray, *, sample_rate_hz: int) -> float:
        _ = sample_rate_hz
        prob = self.probs[self.idx]
        self.idx = min(self.idx + 1, len(self.probs) - 1)
        return prob

    def reset(self) -> None:
        return


@dataclass(slots=True)
class FakeSender:
    sent: list[str]
    typing: list[bool]

    def __init__(self) -> None:
        self.sent = []
        self.typing = []

    def send_chatbox(self, text: str) -> None:
        self.sent.append(text)

    def send_typing(self, is_typing: bool) -> None:
        self.typing.append(is_typing)


@dataclass(slots=True)
class FakeSession:
    audio: list[bytes]
    _queue: asyncio.Queue
    _seen_speech: bool = False

    def __init__(self) -> None:
        self.audio = []
        self._queue = asyncio.Queue()
        self._seen_speech = False

    async def send_audio(self, pcm16le: bytes) -> None:
        self.audio.append(pcm16le)
        is_silence = all(b == 0 for b in pcm16le)
        if not is_silence:
            self._seen_speech = True
            await self._queue.put(STTBackendTranscriptEvent(text="PARTIAL", is_final=False))
        elif self._seen_speech:
            await self._queue.put(STTBackendTranscriptEvent(text="FINAL", is_final=True))
            self._seen_speech = False

    async def on_speech_end(self) -> None:
        if self._seen_speech:
            await self._queue.put(STTBackendTranscriptEvent(text="FINAL", is_final=True))
            self._seen_speech = False

    async def stop(self) -> None:
        await self._queue.put(None)

    async def close(self) -> None:
        await self._queue.put(None)

    async def events(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


@dataclass(slots=True)
class FakeBackend:
    async def open_session(self) -> FakeSession:
        return FakeSession()


def test_headless_mic_pipeline_smoke():
    async def run():
        clock = FakeClock()
        sender = FakeSender()
        osc = SmartOscQueue(sender=sender, clock=clock, ttl_s=100.0)

        stt = ManagedSTTProvider(backend=FakeBackend(), sample_rate_hz=16000, clock=clock)
        hub = ClientHub(stt=stt, llm=None, osc=osc, clock=clock, fallback_transcript_only=True)
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
        frames: list[AudioFrameF32] = []
        offset = 0
        for n in splits:
            frames.append(AudioFrameF32(samples=audio[offset : offset + n], sample_rate_hz=16000))
            offset += n

        source = FakeAudioSource(frames)
        await run_audio_vad_loop(source=source, vad=vad, hub=hub, target_sample_rate_hz=16000)

        for _ in range(50):
            if sender.sent:
                break
            await asyncio.sleep(0.01)

        assert sender.sent == ["FINAL"]
        await hub.stop()

    asyncio.run(run())
