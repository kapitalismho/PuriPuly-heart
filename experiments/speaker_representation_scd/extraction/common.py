from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ExtractionBatch:
    model_id: str
    layers: dict[str, np.ndarray]
    valid_lengths: dict[str, np.ndarray]
    observed_source_samples: np.ndarray
    official_embedding: np.ndarray | None = None


def trailing_window(
    waveform: np.ndarray, frontier_sample: int, window_samples: int
) -> np.ndarray:
    if waveform.ndim != 1:
        raise ValueError("waveform must be one-dimensional")
    if frontier_sample < 0 or frontier_sample > waveform.shape[0]:
        raise ValueError("frontier_sample is outside the waveform")
    if window_samples <= 0 or window_samples > frontier_sample:
        raise ValueError("window_samples must fit before the frontier")
    return np.ascontiguousarray(
        waveform[frontier_sample - window_samples : frontier_sample], dtype=np.float32
    )


def mean_pool_valid(values: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError("values must have shape batch,time,dimension")
    if valid_lengths.shape != (values.shape[0],):
        raise ValueError("valid_lengths must have one entry per batch row")
    pooled = np.empty((values.shape[0], values.shape[2]), dtype=np.float32)
    for index, length in enumerate(valid_lengths.tolist()):
        if length <= 0 or length > values.shape[1]:
            raise ValueError("valid length is outside the feature tensor")
        pooled[index] = values[index, :length].mean(axis=0, dtype=np.float64)
    return pooled


def l2_normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError("values must have shape batch,dimension")
    norms = np.linalg.norm(values, axis=1)
    valid = np.isfinite(norms) & (norms > 0)
    result = np.full(values.shape, np.nan, dtype=np.float32)
    result[valid] = values[valid] / norms[valid, None]
    return result, valid
