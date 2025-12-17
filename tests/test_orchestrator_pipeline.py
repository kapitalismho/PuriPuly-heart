from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from ajebal_daera_translator.core.clock import FakeClock
from ajebal_daera_translator.core.llm.provider import SemaphoreLLMProvider
from ajebal_daera_translator.core.orchestrator.hub import ClientHub
from ajebal_daera_translator.core.osc.smart_queue import SmartOscQueue
from ajebal_daera_translator.core.stt.backend import STTBackendTranscriptEvent
from ajebal_daera_translator.core.stt.controller import ManagedSTTProvider
from ajebal_daera_translator.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from ajebal_daera_translator.domain.models import Translation


@dataclass(slots=True)
class FakeSender:
    sent: list[str]

    def __init__(self) -> None:
        self.sent = []

    def send_chatbox(self, text: str) -> None:
        self.sent.append(text)


@dataclass(slots=True)
class FakeLLM:
    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
    ) -> Translation:
        await asyncio.sleep(0.01)
        return Translation(utterance_id=utterance_id, text="TRANSLATED")


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


def _samples(value: float, n: int = 512) -> np.ndarray:
    return np.full((n,), value, dtype=np.float32)


def test_orchestrator_e2e_headless():
    async def run():
        clock = FakeClock()
        sender = FakeSender()
        osc = SmartOscQueue(sender=sender, clock=clock, ttl_s=100.0)

        stt = ManagedSTTProvider(
            backend=FakeBackend(),
            sample_rate_hz=16000,
            clock=clock,
            reset_deadline_s=90.0,
        )

        llm = SemaphoreLLMProvider(inner=FakeLLM(), semaphore=asyncio.Semaphore(1))
        hub = ClientHub(stt=stt, llm=llm, osc=osc, clock=clock)
        await hub.start(auto_flush_osc=False)

        uid = __import__("uuid").uuid4()
        await hub.handle_vad_event(SpeechStart(uid, pre_roll=_samples(0.0), chunk=_samples(1.0)))
        await hub.handle_vad_event(SpeechChunk(uid, chunk=_samples(0.0)))
        await hub.handle_vad_event(SpeechEnd(uid))

        # Wait for translation and OSC send
        for _ in range(50):
            if sender.sent:
                break
            await asyncio.sleep(0.01)

        assert sender.sent == ["FINAL (TRANSLATED)"]
        await hub.stop()

    asyncio.run(run())
