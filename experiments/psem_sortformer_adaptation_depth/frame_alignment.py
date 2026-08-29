from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

FRAME_SAMPLES = 1280
SLOT_COUNT = 4


class FrameAlignmentError(RuntimeError):
    pass


def native_frame_coordinates(frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise FrameAlignmentError("native frame count is invalid")
    starts = np.arange(frame_count, dtype=np.int64) * FRAME_SAMPLES
    return starts, starts + FRAME_SAMPLES


def native_episode_timeline(reference: Any, frame_count: int) -> tuple[str | None, ...]:
    starts, ends = native_frame_coordinates(frame_count)
    centers = starts + (ends - starts) // 2
    result: list[str | None] = [None] * frame_count
    previous_end = -1
    for episode in reference.episodes:
        start = int(episode.anchor_emit_sample)
        end = int(episode.end_emit_sample)
        if start < previous_end or end <= start:
            raise FrameAlignmentError("anchor episodes are overlapping or unordered")
        selected = np.flatnonzero(np.logical_and(centers >= start, centers < end))
        for index in selected:
            result[int(index)] = str(episode.episode_id)
        previous_end = end
    return tuple(result)


def action_sample_indices(
    native_ends: Sequence[int] | np.ndarray,
    action_ends: Sequence[int] | np.ndarray,
) -> np.ndarray:
    native = np.asarray(native_ends, dtype=np.int64)
    action = np.asarray(action_ends, dtype=np.int64)
    if (
        native.ndim != 1
        or action.ndim != 1
        or native.size == 0
        or action.size == 0
        or np.any(np.diff(native) <= 0)
        or np.any(np.diff(action) <= 0)
    ):
        raise FrameAlignmentError("native or action frame ends are invalid")
    indices = np.searchsorted(native, action, side="right") - 1
    if np.any(indices < 0) or np.any(indices >= native.size):
        raise FrameAlignmentError("action grid extends outside completed native evidence")
    return indices.astype(np.int64, copy=False)


def mapping_from_action_probabilities(
    session: Any,
    probabilities: np.ndarray,
    alive: np.ndarray,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    expected = (len(session.starts), SLOT_COUNT)
    if probabilities.shape != expected or alive.shape != expected:
        raise FrameAlignmentError("action-aligned posterior geometry is invalid")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise FrameAlignmentError("action-aligned posterior values are invalid")
    episode_ids = np.asarray(
        ["" if str(value) in {"", "None"} else str(value) for value in session.episode_ids],
        dtype=str,
    )
    valid = np.logical_and(session.valid, np.logical_not(session.masked))
    slots: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for episode in session.reference.episodes:
        support = np.logical_and.reduce(
            (episode_ids == episode.episode_id, valid, session.anchor_present)
        )
        if not np.any(support):
            slots[str(episode.episode_id)] = 0
            rows.append(
                {
                    "anchor_episode_id": str(episode.episode_id),
                    "status": "unmapped",
                    "slot_index": None,
                    "support_scores": [0.0, 0.0, 0.0, 0.0],
                    "support_frame_count": 0,
                }
            )
            continue
        scores = (probabilities[support] * alive[support].astype(np.float32)).mean(axis=0)
        slot = int(np.argmax(scores))
        slots[str(episode.episode_id)] = slot
        rows.append(
            {
                "anchor_episode_id": str(episode.episode_id),
                "status": "mapped",
                "slot_index": slot,
                "support_scores": [float(item) for item in scores],
                "support_frame_count": int(support.sum()),
            }
        )
    return slots, rows


def align_native_predictions(
    session: Any,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise FrameAlignmentError("native prediction rows are empty")
    native_starts = np.asarray([row["source_frame_start_sample"] for row in rows], dtype=np.int64)
    native_ends = np.asarray([row["source_frame_end_sample"] for row in rows], dtype=np.int64)
    expected_starts, expected_ends = native_frame_coordinates(len(rows))
    if not np.array_equal(native_starts, expected_starts) or not np.array_equal(
        native_ends, expected_ends
    ):
        raise FrameAlignmentError("native prediction coordinates are not the frozen 80 ms grid")
    indices = action_sample_indices(native_ends, session.ends)
    logits = np.asarray([row["raw_sortformer_activity_logits"] for row in rows], dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    alive = np.asarray([row["slot_alive"] for row in rows], dtype=np.bool_)
    reset = np.asarray([row["state_reset"] for row in rows], dtype=np.bool_)
    frontiers = np.asarray(
        [row["model_evidence_frontier_source_sample"] for row in rows], dtype=np.int64
    )
    anchor_logits = np.asarray([row["raw_anchor_present_logit"] for row in rows], dtype=np.float64)
    replacement_logits = np.asarray(
        [row["raw_replacement_evidence_logit"] for row in rows], dtype=np.float64
    )
    return {
        "indices": indices,
        "probabilities": probabilities[indices].astype(np.float32),
        "alive": alive[indices],
        "reset": reset[indices],
        "frontiers": frontiers[indices],
        "anchor_logits": anchor_logits[indices],
        "replacement_logits": replacement_logits[indices],
        "native_probabilities": probabilities.astype(np.float32),
        "native_alive": alive,
        "native_reset": reset,
    }
