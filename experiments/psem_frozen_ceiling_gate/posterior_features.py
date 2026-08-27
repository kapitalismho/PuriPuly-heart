from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalContract:
    causal_lags: tuple[int, ...]
    future_lags: tuple[int, ...]


def temporal_features(
    base: np.ndarray,
    episode_ids: np.ndarray,
    contract: TemporalContract,
    *,
    noncausal: bool,
) -> np.ndarray:
    offsets = tuple(-value for value in contract.causal_lags)
    if noncausal:
        offsets += contract.future_lags
    rows = np.arange(base.shape[0])
    blocks = []
    for offset in offsets:
        selected = np.clip(rows + offset, 0, base.shape[0] - 1)
        valid = episode_ids[selected] == episode_ids
        block = base[selected].copy()
        block[~valid] = base[~valid]
        blocks.append(block)
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def scalar_base(
    probabilities: np.ndarray,
    evidence_delay_ms: np.ndarray,
) -> np.ndarray:
    return np.column_stack((probabilities[:, 0], evidence_delay_ms / 1000.0)).astype(np.float32)


def fullslot_base(
    probabilities: np.ndarray,
    alive: np.ndarray,
    evidence_delay_ms: np.ndarray,
    reset: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        (
            probabilities,
            alive.astype(np.float32),
            alive.sum(axis=1) / alive.shape[1],
            evidence_delay_ms / 1000.0,
            reset.astype(np.float32),
        )
    ).astype(np.float32)
