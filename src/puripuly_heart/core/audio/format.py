from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioFrameF32:
    samples: np.ndarray
    sample_rate_hz: int


def mixdown_to_mono_f32(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        mono = samples
    elif samples.ndim == 2:
        mono = samples.mean(axis=1)
    else:
        raise ValueError("samples must be 1D (mono) or 2D (frames, channels)")

    return np.asarray(mono, dtype=np.float32)


def resample_f32_linear(samples: np.ndarray, *, from_rate_hz: int, to_rate_hz: int) -> np.ndarray:
    if from_rate_hz <= 0 or to_rate_hz <= 0:
        raise ValueError("sample rates must be > 0")
    if from_rate_hz == to_rate_hz:
        return np.asarray(samples, dtype=np.float32)

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return samples

    src_len = int(samples.shape[0])
    dst_len = int(math.floor(src_len * (to_rate_hz / from_rate_hz)))
    dst_len = max(dst_len, 1)

    x_old = np.arange(src_len, dtype=np.float32)
    x_new = np.linspace(0.0, src_len - 1, num=dst_len, dtype=np.float32)
    out = np.interp(x_new, x_old, samples).astype(np.float32)
    return out


def normalize_audio_f32(
    raw_samples: np.ndarray,
    *,
    input_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> AudioFrameF32:
    mono = mixdown_to_mono_f32(raw_samples)
    if input_sample_rate_hz != target_sample_rate_hz:
        mono = resample_f32_linear(
            mono, from_rate_hz=input_sample_rate_hz, to_rate_hz=target_sample_rate_hz
        )
    return AudioFrameF32(samples=mono, sample_rate_hz=target_sample_rate_hz)


def float32_to_pcm16le_bytes(samples: np.ndarray) -> bytes:
    samples = np.asarray(samples, dtype=np.float32)
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = np.round(clipped * 32767.0).astype("<i2")
    return int16.tobytes()


def pcm16le_bytes_to_float32(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype="<i2").astype(np.float32)
    return arr / 32768.0
