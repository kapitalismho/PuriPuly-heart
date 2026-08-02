from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LOCAL_ASR_TRAILING_SILENCE_SAFETY_TAIL_MS = 128


@dataclass(frozen=True, slots=True)
class LocalASRTrailingSilenceTrim:
    samples_f32: np.ndarray
    audio_ms_before: float
    reported_trailing_silence_ms: int | None
    trimmed_samples: int
    actual_trimmed_ms: float
    submitted_audio_ms: float


def trim_local_asr_trailing_silence(
    samples_f32: np.ndarray,
    *,
    sample_rate_hz: int,
    trailing_silence_ms: int | None,
) -> LocalASRTrailingSilenceTrim:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    samples = np.asarray(samples_f32, dtype=np.float32).reshape(-1)
    reported_ms = max(int(trailing_silence_ms or 0), 0)
    requested_trim_samples = max(
        int((reported_ms - LOCAL_ASR_TRAILING_SILENCE_SAFETY_TAIL_MS) * sample_rate_hz / 1000),
        0,
    )
    safety_tail_samples = (
        min(
            samples.size,
            max(
                1,
                math.ceil(LOCAL_ASR_TRAILING_SILENCE_SAFETY_TAIL_MS * sample_rate_hz / 1000),
            ),
        )
        if samples.size
        else 0
    )
    trimmed_samples = min(
        requested_trim_samples,
        max(samples.size - safety_tail_samples, 0),
    )
    submitted = samples[: samples.size - trimmed_samples] if trimmed_samples else samples
    duration_per_sample_ms = 1000.0 / float(sample_rate_hz)
    return LocalASRTrailingSilenceTrim(
        samples_f32=submitted,
        audio_ms_before=samples.size * duration_per_sample_ms,
        reported_trailing_silence_ms=trailing_silence_ms,
        trimmed_samples=trimmed_samples,
        actual_trimmed_ms=trimmed_samples * duration_per_sample_ms,
        submitted_audio_ms=submitted.size * duration_per_sample_ms,
    )
