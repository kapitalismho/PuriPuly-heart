from __future__ import annotations

import hashlib
from typing import Sequence

from experiments.psem_small_model_probe.adapter.protocol import (
    BindingError,
    StepOut,
    frame_bytes,
    validate_pcm16_chunk,
)


class StubAdapter:
    sample_rate_hz = 16000

    def __init__(
        self,
        frame_ms: int = 20,
        anchor_pattern: Sequence[float] = (0.0, 1.0),
        speech_pattern: Sequence[float] = (0.9,),
        min_bind_ms: int = 1000,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        self.frame_ms = frame_ms
        self.anchor_pattern = tuple(anchor_pattern)
        self.speech_pattern = tuple(speech_pattern)
        self.min_bind_ms = min_bind_ms
        self._reset_called = False
        self._bound = False
        self._step_index = 0
        self._source_time_ms = 0
        self.bind_span_hash: str | None = None

    def reset(self) -> None:
        self._reset_called = True
        self._bound = False
        self._step_index = 0
        self._source_time_ms = 0
        self.bind_span_hash = None

    def bind(self, reference_pcm16: bytes) -> None:
        if not self._reset_called:
            raise RuntimeError("bind() requires reset() first")
        unit = frame_bytes(self.frame_ms, self.sample_rate_hz)
        if len(reference_pcm16) == 0 or len(reference_pcm16) < self.min_bind_ms * 16 * 2:
            raise BindingError(
                f"reference span too short: {len(reference_pcm16)} bytes "
                f"(need >= {self.min_bind_ms} ms mono 16 kHz int16 LE)"
            )
        if len(reference_pcm16) % unit != 0:
            raise ValueError("reference span must be a frame multiple")
        self.bind_span_hash = hashlib.sha256(reference_pcm16).hexdigest()
        self._bound = True

    def step(self, pcm16_chunk: bytes) -> StepOut:
        if not self._bound:
            raise RuntimeError("step() requires bind() first")
        n = validate_pcm16_chunk(
            pcm16_chunk, frame_ms=self.frame_ms, sample_rate_hz=self.sample_rate_hz
        )
        outs = []
        for _ in range(n):
            i = self._step_index
            anchor = self.anchor_pattern[i % len(self.anchor_pattern)]
            speech = self.speech_pattern[i % len(self.speech_pattern)]
            self._source_time_ms += self.frame_ms
            outs.append(StepOut(speech=speech, anchor=anchor,
                                aux={"stub_index": i}, source_time_ms=self._source_time_ms))
            self._step_index += 1
        return outs[0] if n == 1 else outs[-1]
