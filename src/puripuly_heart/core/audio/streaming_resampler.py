from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from puripuly_heart.core.audio.format import (
    AudioCaptureSpan,
    mixdown_to_mono_f32,
    reshape_audio_samples_f32,
)

NOOP_SAMPLE_RATE_HZ = 16000


def _import_soxr() -> Any:
    return importlib.import_module("soxr")


@dataclass(slots=True)
class MonoFirstStreamingResampler:
    input_sample_rate_hz: int
    output_sample_rate_hz: int = NOOP_SAMPLE_RATE_HZ
    input_channels: int = 1
    _stream: Any | None = field(init=False, default=None, repr=False)
    _flushed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.input_sample_rate_hz <= 0 or self.output_sample_rate_hz <= 0:
            raise ValueError("sample rates must be > 0")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be > 0")
        if self._uses_noop_path:
            return
        self._stream = _import_soxr().ResampleStream(
            self.input_sample_rate_hz,
            self.output_sample_rate_hz,
            1,
            dtype="float32",
            quality="MQ",
        )

    @property
    def _uses_noop_path(self) -> bool:
        return (
            self.input_sample_rate_hz == NOOP_SAMPLE_RATE_HZ
            and self.output_sample_rate_hz == NOOP_SAMPLE_RATE_HZ
        )

    def resample_chunk(self, samples: np.ndarray, *, last: bool = False) -> np.ndarray:
        if self._flushed:
            raise RuntimeError("stream has already been flushed")

        mono = self._prepare_mono_chunk(samples)
        if self._uses_noop_path:
            output = mono
        else:
            if self._stream is None:
                raise RuntimeError("soxr stream is unavailable")
            output = np.asarray(self._stream.resample_chunk(mono, last=last), dtype=np.float32)

        if last:
            self._flushed = True
        return output

    def flush(self) -> np.ndarray:
        return self.resample_chunk(np.empty((0,), dtype=np.float32), last=True)

    def _prepare_mono_chunk(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return np.empty((0,), dtype=np.float32)

        reshaped = reshape_audio_samples_f32(samples, channels=self.input_channels)
        if reshaped.ndim == 2 and reshaped.shape[1] != self.input_channels:
            raise ValueError("2D samples channel count must match input_channels")
        return mixdown_to_mono_f32(reshaped)

@dataclass(slots=True)
class CaptureMappedStreamingResampler:
    input_sample_rate_hz: int
    output_sample_rate_hz: int = NOOP_SAMPLE_RATE_HZ
    input_channels: int = 1
    _resampler: MonoFirstStreamingResampler = field(init=False, repr=False)
    _pending_capture: AudioCaptureSpan | None = field(init=False, default=None, repr=False)
    _capture_epoch: int | None = field(init=False, default=None, repr=False)
    _next_normalized_sample: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self._resampler = self._new_resampler()

    def process(
        self,
        samples: np.ndarray,
        capture: AudioCaptureSpan | None,
    ) -> tuple[np.ndarray, AudioCaptureSpan | None, tuple[AudioCaptureSpan, ...]]:
        discarded: tuple[AudioCaptureSpan, ...] = ()
        if capture is not None:
            epoch_changed = (
                self._capture_epoch is not None
                and capture.capture_epoch != self._capture_epoch
            )
            discontinuous = capture.discontinuity_before is not None or epoch_changed
            if discontinuous:
                discarded = self.discard_pending()
                self._resampler = self._new_resampler()
            if self._capture_epoch is None or epoch_changed:
                self._capture_epoch = capture.capture_epoch
                self._next_normalized_sample = self._capture_start(capture)
            elif capture.discontinuity_before is not None:
                self._next_normalized_sample = self._capture_start(capture)
            self._append_capture(capture)
        output = self._resampler.resample_chunk(samples)
        mapped = self._consume_capture(int(output.size))
        return output, mapped, discarded

    def finish(
        self,
        *,
        orderly: bool,
    ) -> tuple[np.ndarray, AudioCaptureSpan | None, tuple[AudioCaptureSpan, ...]]:
        if not orderly:
            return (
                np.empty((0,), dtype=np.float32),
                None,
                self.discard_pending(),
            )
        output = self._resampler.flush()
        mapped = self._consume_capture(int(output.size))
        pending = self._pending_capture
        if pending is not None and mapped is not None:
            mapped = replace(
                mapped,
                source_end_sample=pending.source_end_sample,
                source_end_monotonic_s=pending.source_end_monotonic_s,
            )
            self._pending_capture = None
        discarded = self.discard_pending()
        return output, mapped, discarded

    def discard_pending(self) -> tuple[AudioCaptureSpan, ...]:
        pending = self._pending_capture
        self._pending_capture = None
        return () if pending is None else (pending,)

    def _append_capture(self, capture: AudioCaptureSpan) -> None:
        start = self._capture_start(capture)
        end = self._capture_end(capture)
        pending = self._pending_capture
        if pending is None:
            if end > start:
                self._pending_capture = replace(
                    capture,
                    normalized_sample_rate_hz=self.output_sample_rate_hz,
                    normalized_start_sample=start,
                    normalized_end_sample=end,
                )
            else:
                self._pending_capture = capture
            return

        pending_start = (
            pending.normalized_start_sample
            if pending.normalized_start_sample is not None
            else self._project_source_sample(pending.source_start_sample)
        )
        projected_end = max(end, pending_start + 1)
        self._pending_capture = replace(
            pending,
            source_end_sample=capture.source_end_sample,
            source_end_monotonic_s=capture.source_end_monotonic_s,
            normalized_sample_rate_hz=self.output_sample_rate_hz,
            normalized_start_sample=pending_start,
            normalized_end_sample=projected_end,
        )

    def _consume_capture(self, sample_count: int) -> AudioCaptureSpan | None:
        if sample_count <= 0:
            return None
        pending = self._pending_capture
        if pending is None:
            return None
        start = pending.normalized_start_sample
        end = pending.normalized_end_sample
        if start is None or end is None:
            start = self._next_normalized_sample
            end = start + sample_count
            pending = replace(
                pending,
                normalized_sample_rate_hz=self.output_sample_rate_hz,
                normalized_start_sample=start,
                normalized_end_sample=end,
            )
        output_end = start + sample_count
        if output_end > end:
            end = output_end
            pending = replace(pending, normalized_end_sample=end)
        mapped = pending.slice_normalized(start, output_end)
        self._next_normalized_sample = output_end
        self._pending_capture = (
            None
            if output_end == end
            else pending.slice_normalized(output_end, end)
        )
        return mapped

    def _capture_start(self, capture: AudioCaptureSpan) -> int:
        if (
            capture.normalized_sample_rate_hz == self.output_sample_rate_hz
            and capture.normalized_start_sample is not None
        ):
            return capture.normalized_start_sample
        return self._project_source_sample(capture.source_start_sample)

    def _capture_end(self, capture: AudioCaptureSpan) -> int:
        if (
            capture.normalized_sample_rate_hz == self.output_sample_rate_hz
            and capture.normalized_end_sample is not None
        ):
            return capture.normalized_end_sample
        return self._project_source_sample(capture.source_end_sample)

    def _project_source_sample(self, source_sample: int) -> int:
        return round(
            source_sample * self.output_sample_rate_hz / self.input_sample_rate_hz
        )

    def _new_resampler(self) -> MonoFirstStreamingResampler:
        return MonoFirstStreamingResampler(
            input_sample_rate_hz=self.input_sample_rate_hz,
            output_sample_rate_hz=self.output_sample_rate_hz,
            input_channels=self.input_channels,
        )


__all__ = [
    "CaptureMappedStreamingResampler",
    "MonoFirstStreamingResampler",
    "NOOP_SAMPLE_RATE_HZ",
]
