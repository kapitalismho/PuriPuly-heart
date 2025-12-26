from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ajebal_daera_translator.core.vad.gating import SpeechEnd, SpeechStart, VadGating


@dataclass(slots=True)
class SequenceVadEngine:
    probs: list[float]
    idx: int = 0

    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float:
        prob = self.probs[self.idx]
        self.idx = min(self.idx + 1, len(self.probs) - 1)
        return prob

    def reset(self) -> None:
        return


def _chunk(value: float, *, n: int) -> np.ndarray:
    return np.full((n,), value, dtype=np.float32)


def test_vad_gating_emits_start_and_end_with_hangover():
    # 32ms chunks @16k => 512 samples
    probs = [0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0]
    engine = SequenceVadEngine(probs=probs)
    gating = VadGating(engine, sample_rate_hz=16000, ring_buffer_ms=64, hangover_ms=64)

    events = []
    for i in range(len(probs)):
        events.extend(gating.process_chunk(_chunk(float(i), n=gating.chunk_samples)))

    start = next(e for e in events if isinstance(e, SpeechStart))
    end = next(e for e in events if isinstance(e, SpeechEnd))

    assert start.utterance_id == end.utterance_id
    assert start.pre_roll.shape[0] == 1024  # 64ms @ 16k


def test_vad_gating_pre_roll_contains_previous_audio():
    probs = [0.0, 0.0, 0.9]
    engine = SequenceVadEngine(probs=probs)
    gating = VadGating(engine, sample_rate_hz=16000, ring_buffer_ms=64, hangover_ms=0)

    # append two silent chunks (values 0,1) then speech chunk (value 2)
    gating.process_chunk(_chunk(0.0, n=gating.chunk_samples))
    gating.process_chunk(_chunk(1.0, n=gating.chunk_samples))
    events = gating.process_chunk(_chunk(2.0, n=gating.chunk_samples))

    start = next(e for e in events if isinstance(e, SpeechStart))
    assert start.pre_roll.shape[0] == 1024
    assert np.allclose(start.pre_roll[:512], 0.0)
    assert np.allclose(start.pre_roll[512:], 1.0)
