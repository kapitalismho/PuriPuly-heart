from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puripuly_heart.config.resolved import vad_exit_threshold
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.streaming_resampler import MonoFirstStreamingResampler
from puripuly_heart.core.vad.gating import VadEngine, VadGating

AUDIO_TEST_VAD_SAMPLE_RATE_HZ = 16000


@dataclass(frozen=True, slots=True)
class AudioTestVadSettings:
    speech_threshold: float
    hangover_ms: int
    ring_buffer_ms: int
    start_debounce_chunks: int = 1
    start_commit_chunks: int = 1


@dataclass(frozen=True, slots=True)
class AudioTestVadSample:
    probability: float
    level: float
    speech: bool
    threshold: float


class AudioTestVadProbe:
    """Runs one channel's VAD gating over captured frames without delivering speech."""

    def __init__(self, *, engine: VadEngine, settings: AudioTestVadSettings) -> None:
        self._settings = settings
        self._gating = VadGating(
            engine=engine,
            sample_rate_hz=AUDIO_TEST_VAD_SAMPLE_RATE_HZ,
            ring_buffer_ms=max(1, settings.ring_buffer_ms),
            speech_threshold=settings.speech_threshold,
            continuation_threshold=vad_exit_threshold(settings.speech_threshold),
            hangover_ms=settings.hangover_ms,
            start_debounce_chunks=settings.start_debounce_chunks,
            start_commit_chunks=settings.start_commit_chunks,
        )
        self._normalizer: MonoFirstStreamingResampler | None = None
        self._source_format: tuple[int, int] | None = None
        self._pending = np.empty((0,), dtype=np.float32)

    @property
    def settings(self) -> AudioTestVadSettings:
        return self._settings

    def reconfigure(self, settings: AudioTestVadSettings) -> None:
        if settings == self._settings:
            return
        self._settings = settings
        self._gating.reconfigure_next_segment(
            speech_threshold=settings.speech_threshold,
            continuation_threshold=vad_exit_threshold(settings.speech_threshold),
            hangover_ms=settings.hangover_ms,
            ring_buffer_ms=max(1, settings.ring_buffer_ms),
        )

    def process(self, frame: AudioFrameF32) -> tuple[AudioTestVadSample, ...]:
        frame_format = (int(frame.sample_rate_hz), int(frame.channels))
        if self._source_format is None:
            self._source_format = frame_format
            self._normalizer = MonoFirstStreamingResampler(
                input_sample_rate_hz=frame_format[0],
                output_sample_rate_hz=AUDIO_TEST_VAD_SAMPLE_RATE_HZ,
                input_channels=frame_format[1],
            )
        elif frame_format != self._source_format:
            raise ValueError("capture_format_changed")
        assert self._normalizer is not None
        samples = np.asarray(frame.samples, dtype=np.float32)
        if not np.all(np.isfinite(samples)):
            raise ValueError("invalid_pcm")
        mono = self._normalizer.resample_chunk(samples.reshape(-1, frame_format[1]))
        buffer = np.concatenate((self._pending, np.asarray(mono, dtype=np.float32).reshape(-1)))
        chunk_samples = self._gating.chunk_samples
        observations: list[AudioTestVadSample] = []
        offset = 0
        while buffer.size - offset >= chunk_samples:
            chunk = buffer[offset : offset + chunk_samples]
            offset += chunk_samples
            self._gating.process_chunk(chunk)
            probability = self._gating.last_probability
            observations.append(
                AudioTestVadSample(
                    probability=float(probability) if probability is not None else 0.0,
                    level=min(1.0, float(np.max(np.abs(chunk)))),
                    speech=self._gating.in_speech,
                    threshold=self._gating.speech_threshold,
                )
            )
        self._pending = buffer[offset:].copy()
        return tuple(observations)


__all__ = [
    "AUDIO_TEST_VAD_SAMPLE_RATE_HZ",
    "AudioTestVadProbe",
    "AudioTestVadSample",
    "AudioTestVadSettings",
]
