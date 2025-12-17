from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RingBufferF32:
    capacity_samples: int
    _buffer: np.ndarray
    _write_pos: int
    _filled: bool

    def __init__(self, *, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be > 0")
        self.capacity_samples = capacity_samples
        self._buffer = np.zeros((capacity_samples,), dtype=np.float32)
        self._write_pos = 0
        self._filled = False

    def clear(self) -> None:
        self._buffer.fill(0.0)
        self._write_pos = 0
        self._filled = False

    def append(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return

        if samples.size >= self.capacity_samples:
            self._buffer[:] = samples[-self.capacity_samples :]
            self._write_pos = 0
            self._filled = True
            return

        end = self._write_pos + samples.size
        if end <= self.capacity_samples:
            self._buffer[self._write_pos : end] = samples
        else:
            first = self.capacity_samples - self._write_pos
            self._buffer[self._write_pos :] = samples[:first]
            self._buffer[: end - self.capacity_samples] = samples[first:]

        self._write_pos = end % self.capacity_samples
        if self._write_pos == 0:
            self._filled = True

    def get_last_samples(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0,), dtype=np.float32)

        available = self.capacity_samples if self._filled else self._write_pos
        count = min(count, available)

        start = (self._write_pos - count) % self.capacity_samples
        if count == 0:
            return np.zeros((0,), dtype=np.float32)

        if start < self._write_pos or self._filled is False:
            return self._buffer[start : start + count].copy()

        tail = self._buffer[start:].copy()
        head_len = count - tail.size
        head = self._buffer[:head_len].copy()
        return np.concatenate([tail, head])

