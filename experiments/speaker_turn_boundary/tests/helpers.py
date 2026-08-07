from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SequenceVadEngine:
    probs: list[float]
    idx: int = 0

    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float:
        _ = samples
        _ = sample_rate_hz
        prob = self.probs[self.idx]
        self.idx = min(self.idx + 1, len(self.probs) - 1)
        return prob

    def reset(self) -> None:
        return


def chunk_samples(value: float, *, n: int = 512) -> np.ndarray:
    return np.full((n,), value, dtype=np.float32)


def write_pcm16_wav(path, samples: np.ndarray, *, sample_rate_hz: int = 16000) -> None:
    import wave

    scaled = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(scaled * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm.tobytes())
