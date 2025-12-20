from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import numpy as np

from ajebal_daera_translator.core.audio.ring_buffer import RingBufferF32

logger = logging.getLogger(__name__)


class VadEngine(Protocol):
    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float: ...
    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SpeechStart:
    utterance_id: UUID
    pre_roll: np.ndarray
    chunk: np.ndarray


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    utterance_id: UUID
    chunk: np.ndarray


@dataclass(frozen=True, slots=True)
class SpeechEnd:
    utterance_id: UUID


VadEvent = SpeechStart | SpeechChunk | SpeechEnd


def default_chunk_samples(sample_rate_hz: int) -> int:
    if sample_rate_hz == 16000:
        return 512
    if sample_rate_hz == 8000:
        return 256
    raise ValueError("Silero VAD streaming supports only 8000 or 16000 Hz")


@dataclass(slots=True)
class VadGating:
    engine: VadEngine
    sample_rate_hz: int
    speech_threshold: float
    hangover_chunks: int
    chunk_samples: int
    _ring: RingBufferF32
    _in_speech: bool
    _utterance_id: UUID | None
    _silence_run: int

    def __init__(
        self,
        engine: VadEngine,
        *,
        sample_rate_hz: int,
        ring_buffer_ms: int = 500,
        speech_threshold: float = 0.5,
        hangover_ms: int = 1200,
        chunk_samples: int | None = None,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if ring_buffer_ms <= 0:
            raise ValueError("ring_buffer_ms must be > 0")
        if hangover_ms < 0:
            raise ValueError("hangover_ms must be >= 0")

        self.engine = engine
        self.sample_rate_hz = sample_rate_hz
        self.speech_threshold = speech_threshold
        self.chunk_samples = chunk_samples or default_chunk_samples(sample_rate_hz)

        chunk_ms = (self.chunk_samples / self.sample_rate_hz) * 1000.0
        self.hangover_chunks = int(math.ceil(hangover_ms / chunk_ms)) if hangover_ms > 0 else 0

        capacity_samples = int(self.sample_rate_hz * (ring_buffer_ms / 1000.0))
        self._ring = RingBufferF32(capacity_samples=capacity_samples)

        self._in_speech = False
        self._utterance_id = None
        self._silence_run = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def utterance_id(self) -> UUID | None:
        return self._utterance_id

    def reset(self) -> None:
        self.engine.reset()
        self._ring.clear()
        self._in_speech = False
        self._utterance_id = None
        self._silence_run = 0

    def process_chunk(self, chunk: np.ndarray) -> list[VadEvent]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size != self.chunk_samples:
            raise ValueError(f"chunk must have {self.chunk_samples} samples")

        prob = self.engine.speech_probability(chunk, sample_rate_hz=self.sample_rate_hz)

        events: list[VadEvent] = []

        if not self._in_speech:
            if prob >= self.speech_threshold:
                self._in_speech = True
                self._silence_run = 0
                self._utterance_id = uuid.uuid4()

                pre_roll = self._ring.get_last_samples(self._ring.capacity_samples)
                logger.info(f"[VAD] SpeechStart: id={str(self._utterance_id)[:8]}, prob={prob:.2f}")
                events.append(SpeechStart(self._utterance_id, pre_roll=pre_roll, chunk=chunk.copy()))
            self._ring.append(chunk)
            return events

        # in speech
        events.append(SpeechChunk(self._utterance_id, chunk=chunk.copy()))  # type: ignore[arg-type]

        if prob >= self.speech_threshold:
            self._silence_run = 0
            return events

        self._silence_run += 1
        if self._silence_run >= self.hangover_chunks:
            logger.info(f"[VAD] SpeechEnd: id={str(self._utterance_id)[:8]}")
            events.append(SpeechEnd(self._utterance_id))  # type: ignore[arg-type]
            self._in_speech = False
            self._utterance_id = None
            self._silence_run = 0
            self.engine.reset()

        self._ring.append(chunk)
        return events
