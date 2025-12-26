from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from ajebal_daera_translator.core.clock import FakeClock
from ajebal_daera_translator.core.stt.backend import STTBackendTranscriptEvent
from ajebal_daera_translator.core.stt.controller import ManagedSTTProvider
from ajebal_daera_translator.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from ajebal_daera_translator.domain.events import STTSessionState, STTSessionStateEvent


@dataclass(slots=True)
class FakeSession:
    audio: list[bytes]
    _queue: asyncio.Queue
    _closed: bool = False

    def __init__(self) -> None:
        self.audio = []
        self._queue = asyncio.Queue()

    async def send_audio(self, pcm16le: bytes) -> None:
        self.audio.append(pcm16le)
        if len(self.audio) == 1:
            await self._queue.put(STTBackendTranscriptEvent(text="partial", is_final=False))

    async def stop(self) -> None:
        await self._queue.put(STTBackendTranscriptEvent(text="final", is_final=True))
        await self._queue.put(None)  # sentinel

    async def on_speech_end(self) -> None:
        """Handle end of speech (no-op for fake session)."""
        pass

    async def close(self) -> None:
        self._closed = True

    async def events(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


@dataclass(slots=True)
class FakeBackend:
    sessions: list[FakeSession]

    def __init__(self) -> None:
        self.sessions = []

    async def open_session(self) -> FakeSession:
        s = FakeSession()
        self.sessions.append(s)
        return s


def _samples(value: float, n: int = 512) -> np.ndarray:
    return np.full((n,), value, dtype=np.float32)


async def _next_event(stream, *, timeout_s: float = 0.2):
    return await asyncio.wait_for(stream.__anext__(), timeout=timeout_s)


def test_stt_controller_connects_on_speech_start():
    async def run():
        clock = FakeClock()
        backend = FakeBackend()
        stt = ManagedSTTProvider(
            backend=backend, sample_rate_hz=16000, clock=clock, reset_deadline_s=90.0
        )

        uid = __import__("uuid").uuid4()
        stream = stt.events()
        await stt.handle_vad_event(SpeechStart(uid, pre_roll=_samples(0.0), chunk=_samples(1.0)))
        first = await _next_event(stream)

        assert len(backend.sessions) == 1
        assert isinstance(first, STTSessionStateEvent)
        assert first.state == STTSessionState.STREAMING

    asyncio.run(run())


def test_stt_controller_resets_with_bridging_during_speech():
    async def run():
        clock = FakeClock()
        backend = FakeBackend()
        stt = ManagedSTTProvider(
            backend=backend,
            sample_rate_hz=16000,
            clock=clock,
            reset_deadline_s=1.0,
            drain_timeout_s=0.05,
            bridging_ms=64,
        )

        uid = __import__("uuid").uuid4()
        stream = stt.events()
        await stt.handle_vad_event(SpeechStart(uid, pre_roll=_samples(0.0), chunk=_samples(1.0)))
        _ = await _next_event(stream)

        clock.advance(1.1)
        await stt.handle_vad_event(SpeechChunk(uid, chunk=_samples(2.0)))

        await asyncio.sleep(0.01)
        assert len(backend.sessions) == 2
        assert len(backend.sessions[1].audio) >= 1  # bridging audio

    asyncio.run(run())


def test_stt_controller_resets_on_silence():
    async def run():
        clock = FakeClock()
        backend = FakeBackend()
        stt = ManagedSTTProvider(
            backend=backend,
            sample_rate_hz=16000,
            clock=clock,
            reset_deadline_s=1.0,
            drain_timeout_s=0.05,
        )

        uid = __import__("uuid").uuid4()
        stream = stt.events()
        await stt.handle_vad_event(SpeechStart(uid, pre_roll=_samples(0.0), chunk=_samples(1.0)))
        _ = await _next_event(stream)

        clock.advance(1.1)
        await stt.handle_vad_event(SpeechEnd(uid))

        for _ in range(10):
            ev = await _next_event(stream)
            if isinstance(ev, STTSessionStateEvent) and ev.state == STTSessionState.DISCONNECTED:
                return
        raise AssertionError("Expected DISCONNECTED state event")

    asyncio.run(run())
