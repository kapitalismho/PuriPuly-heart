from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .phase5_policy import canonical_json


class Phase5ControlError(RuntimeError):
    pass


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def active_intervals_from_lifecycle(
    events: Sequence[dict[str, Any]], processed_end: int
) -> list[dict[str, Any]]:
    active: dict[str, dict[str, Any]] = {}
    intervals: list[dict[str, Any]] = []
    for event in events:
        kind = str(event["event_kind"])
        utterance_id = str(event["normalized_utterance_id"])
        if kind == "speech_start":
            if utterance_id in active:
                raise Phase5ControlError("duplicate active VAD utterance")
            active[utterance_id] = event
        elif kind == "speech_end":
            start = active.pop(utterance_id, None)
            if start is None:
                raise Phase5ControlError("VAD speech end has no start")
            intervals.append(
                {
                    "utterance_id": utterance_id,
                    "start": int(start["event_source_sample"]),
                    "end": int(event["event_source_sample"]),
                    "start_observed_source_sample": int(start["observed_source_sample_at_emit"]),
                    "end_observed_source_sample": int(event["observed_source_sample_at_emit"]),
                }
            )
        elif kind == "terminal" and event.get("active_state_remained"):
            start = active.pop(utterance_id, None)
            if start is None:
                raise Phase5ControlError("terminal active VAD state has no start")
            intervals.append(
                {
                    "utterance_id": utterance_id,
                    "start": int(start["event_source_sample"]),
                    "end": processed_end,
                    "start_observed_source_sample": int(start["observed_source_sample_at_emit"]),
                    "end_observed_source_sample": int(event["observed_source_sample_at_emit"]),
                }
            )
    if active:
        raise Phase5ControlError("unterminated VAD active interval")
    intervals.sort(key=lambda row: (row["start"], row["end"], row["utterance_id"]))
    if any(left["end"] > right["start"] for left, right in zip(intervals, intervals[1:])):
        raise Phase5ControlError("VAD active intervals overlap")
    return intervals


def log_rms(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise Phase5ControlError("invalid energy window")
    return math.log(max(float(np.sqrt(np.mean(values * values))), 1e-12))


def causal_energy_candidates(
    samples: np.ndarray,
    active_intervals: Sequence[dict[str, Any]],
    *,
    window_samples: int = 4000,
    step_samples: int = 512,
    offset_samples: int = 0,
) -> list[dict[str, Any]]:
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    candidates: list[dict[str, Any]] = []
    for interval in active_intervals:
        start = int(interval["start"])
        end = int(interval["end"])
        first = ((start + window_samples + step_samples - 1) // step_samples) * step_samples
        for boundary in range(first, end - window_samples + 1, step_samples):
            observed = boundary + window_samples
            if observed > waveform.size:
                raise Phase5ControlError("energy candidate exceeds waveform")
            left = log_rms(waveform[boundary - window_samples : boundary])
            right = log_rms(waveform[boundary:observed])
            identity = {
                "boundary_source_sample": boundary + offset_samples,
                "observed_source_sample": observed + offset_samples,
                "window_samples": window_samples,
                "step_samples": step_samples,
            }
            candidates.append(
                {
                    "candidate_id": "energy:" + content_sha256(identity)[:32],
                    "boundary_source_sample": boundary + offset_samples,
                    "observed_source_sample": observed + offset_samples,
                    "change_strength": abs(right - left),
                    "left_log_rms": left,
                    "right_log_rms": right,
                    "window_samples": window_samples,
                }
            )
    candidates.sort(
        key=lambda row: (
            int(row["observed_source_sample"]),
            int(row["boundary_source_sample"]),
            str(row["candidate_id"]),
        )
    )
    return candidates
