from __future__ import annotations

import numpy as np

from puripuly_heart.core.audio.streaming_resampler import MonoFirstStreamingResampler


class TestRecording:
    __test__ = False

    def __init__(self, *, seconds=10, target_rate=16000):
        self.limit_seconds = max(1.0, min(30.0, float(seconds)))
        self.target_rate = target_rate
        self.raw_chunks = []
        self.processed_chunks = []
        self.sample_count = 0
        self.bytes_used = 0
        self.sample_rate = None
        self.channels = None
        self.converter = None
        self.finished = False
        self.device = {}
        self.peak = 0.0

    @property
    def duration(self):
        return self.sample_count / self.sample_rate if self.sample_rate else 0.0

    def append(self, frame):
        if self.finished:
            return False
        rate, channels = int(frame.sample_rate_hz), int(frame.channels)
        if self.sample_rate is None:
            self.sample_rate, self.channels = rate, channels
            self.converter = MonoFirstStreamingResampler(
                input_sample_rate_hz=rate,
                output_sample_rate_hz=self.target_rate,
                input_channels=channels,
            )
        elif (rate, channels) != (self.sample_rate, self.channels):
            raise ValueError("capture_format_changed")
        samples = np.asarray(frame.samples, dtype=np.float32).reshape(-1, channels)
        remaining = max(0, int(self.limit_seconds * rate) - self.sample_count)
        samples = samples[:remaining].copy()
        if not np.all(np.isfinite(samples)):
            raise ValueError("invalid_pcm")
        if samples.nbytes + self.bytes_used > 64 * 1024 * 1024:
            raise MemoryError("recording_memory_limit")
        if samples.size:
            self.peak = max(self.peak, float(np.max(np.abs(samples))))
            processed = self.converter.resample_chunk(samples)
            self.raw_chunks.append(samples)
            self.processed_chunks.append(processed.copy())
            self.sample_count += len(samples)
            self.bytes_used += samples.nbytes + processed.nbytes
        if self.sample_count >= int(self.limit_seconds * rate):
            self.finish()
            return False
        return True

    def finish(self):
        if self.finished:
            return
        if self.converter is not None:
            self.processed_chunks.append(self.converter.flush())
        self.finished = True

    def audio(self, *, processed=False):
        if not self.finished or not self.sample_count:
            raise ValueError("no_recording")
        chunks = self.processed_chunks if processed else self.raw_chunks
        return np.concatenate(chunks), self.target_rate if processed else self.sample_rate

    def clear(self):
        self.raw_chunks.clear()
        self.processed_chunks.clear()
        self.converter = None
        self.sample_count = 0
        self.bytes_used = 0
        self.finished = True
        self.device.clear()

    def snapshot(self):
        return dict(
            recorded_seconds=self.duration,
            recording_complete=self.finished,
            sample_rate=self.sample_rate,
            channels=self.channels,
            actual_device=dict(self.device),
            has_recording=bool(self.sample_count),
            peak=self.peak,
            target_rate=self.target_rate,
        )
