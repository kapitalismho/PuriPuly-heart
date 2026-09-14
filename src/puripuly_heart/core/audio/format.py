from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

CaptureDiscontinuityKind = Literal["known_loss", "unknown_loss"]


@dataclass(frozen=True, slots=True)
class AudioCaptureDiscontinuity:
    kind: CaptureDiscontinuityKind
    observed_at_monotonic_s: float
    lost_source_samples: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "known_loss":
            if self.lost_source_samples is None or self.lost_source_samples <= 0:
                raise ValueError("known loss requires a positive source sample count")
        elif self.lost_source_samples is not None:
            raise ValueError("unknown loss cannot declare a source sample count")


@dataclass(frozen=True, slots=True)
class AudioCaptureSpan:
    capture_epoch: int
    callback_sequence: int
    source_sample_rate_hz: int
    source_start_sample: int
    source_end_sample: int
    source_start_monotonic_s: float
    source_end_monotonic_s: float
    discontinuity_before: AudioCaptureDiscontinuity | None = None
    normalized_sample_rate_hz: int | None = None
    normalized_start_sample: int | None = None
    normalized_end_sample: int | None = None

    def __post_init__(self) -> None:
        if self.capture_epoch < 0 or self.callback_sequence < 0:
            raise ValueError("capture epoch and callback sequence must be non-negative")
        if self.source_sample_rate_hz <= 0:
            raise ValueError("source sample rate must be positive")
        if self.source_start_sample < 0 or self.source_end_sample <= self.source_start_sample:
            raise ValueError("source range must be non-empty and increasing")
        if self.source_end_monotonic_s < self.source_start_monotonic_s:
            raise ValueError("source monotonic range must be increasing")
        normalized = (
            self.normalized_sample_rate_hz,
            self.normalized_start_sample,
            self.normalized_end_sample,
        )
        if any(value is not None for value in normalized):
            if any(value is None for value in normalized):
                raise ValueError("normalized range fields must be provided together")
            if self.normalized_sample_rate_hz is None or self.normalized_sample_rate_hz <= 0:
                raise ValueError("normalized sample rate must be positive")
            if (
                self.normalized_start_sample is None
                or self.normalized_end_sample is None
                or self.normalized_start_sample < 0
                or self.normalized_end_sample <= self.normalized_start_sample
            ):
                raise ValueError("normalized range must be non-empty and increasing")

    @property
    def source_sample_count(self) -> int:
        return self.source_end_sample - self.source_start_sample

    @property
    def normalized_sample_count(self) -> int:
        if self.normalized_start_sample is None or self.normalized_end_sample is None:
            return 0
        return self.normalized_end_sample - self.normalized_start_sample

    def with_normalized_range(
        self,
        *,
        sample_rate_hz: int,
        start_sample: int,
        end_sample: int,
    ) -> "AudioCaptureSpan":
        return replace(
            self,
            normalized_sample_rate_hz=sample_rate_hz,
            normalized_start_sample=start_sample,
            normalized_end_sample=end_sample,
        )

    def slice_normalized(self, start_sample: int, end_sample: int) -> "AudioCaptureSpan":
        normalized_start = self.normalized_start_sample
        normalized_end = self.normalized_end_sample
        if normalized_start is None or normalized_end is None:
            raise ValueError("capture span has no normalized range")
        if (
            start_sample < normalized_start
            or end_sample > normalized_end
            or end_sample <= start_sample
        ):
            raise ValueError("slice must be inside the normalized range")
        normalized_count = normalized_end - normalized_start
        source_count = self.source_sample_count
        left_ratio = (start_sample - normalized_start) / normalized_count
        right_ratio = (end_sample - normalized_start) / normalized_count
        source_start = self.source_start_sample + round(source_count * left_ratio)
        source_end = self.source_start_sample + round(source_count * right_ratio)
        source_start = min(source_start, self.source_end_sample - 1)
        source_end = max(source_start + 1, min(source_end, self.source_end_sample))
        monotonic_duration = self.source_end_monotonic_s - self.source_start_monotonic_s
        return replace(
            self,
            source_start_sample=source_start,
            source_end_sample=source_end,
            source_start_monotonic_s=self.source_start_monotonic_s
            + monotonic_duration * left_ratio,
            source_end_monotonic_s=self.source_start_monotonic_s + monotonic_duration * right_ratio,
            discontinuity_before=(
                self.discontinuity_before if start_sample == normalized_start else None
            ),
            normalized_start_sample=start_sample,
            normalized_end_sample=end_sample,
        )


@dataclass(frozen=True, slots=True)
class AudioFrameF32:
    samples: np.ndarray
    sample_rate_hz: int
    channels: int = 1
    capture: AudioCaptureSpan | None = None
    discarded_capture_before: tuple[AudioCaptureSpan, ...] = ()
    discontinuity_before: AudioCaptureDiscontinuity | None = None


def reshape_audio_samples_f32(samples: np.ndarray, *, channels: int = 1) -> np.ndarray:
    if channels <= 0:
        raise ValueError("channels must be > 0")

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        return samples
    if samples.ndim != 1:
        raise ValueError("samples must be 1D or 2D")
    if channels == 1:
        return samples
    if samples.size % channels != 0:
        raise ValueError("interleaved samples must divide evenly by channels")
    return samples.reshape((-1, channels))


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
    channels: int = 1,
) -> AudioFrameF32:
    mono = mixdown_to_mono_f32(reshape_audio_samples_f32(raw_samples, channels=channels))
    if input_sample_rate_hz != target_sample_rate_hz:
        mono = resample_f32_linear(
            mono, from_rate_hz=input_sample_rate_hz, to_rate_hz=target_sample_rate_hz
        )
    return AudioFrameF32(samples=mono, sample_rate_hz=target_sample_rate_hz)


def normalize_audio_frame_f32(
    frame: AudioFrameF32,
    *,
    target_sample_rate_hz: int,
) -> AudioFrameF32:
    return normalize_audio_f32(
        frame.samples,
        input_sample_rate_hz=frame.sample_rate_hz,
        target_sample_rate_hz=target_sample_rate_hz,
        channels=frame.channels,
    )


def float32_to_pcm16le_bytes(samples: np.ndarray) -> bytes:
    samples = np.asarray(samples, dtype=np.float32)
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = np.round(clipped * 32767.0).astype("<i2")
    return int16.tobytes()


def pcm16le_bytes_to_float32(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype="<i2").astype(np.float32)
    return arr / 32768.0
