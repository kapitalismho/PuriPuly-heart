from __future__ import annotations

import numpy as np
import pytest

from experiments.speaker_turn_boundary.turn_episode.phase5_controls import (
    Phase5ControlError,
    active_intervals_from_lifecycle,
    causal_energy_candidates,
)


def event(
    kind: str,
    utterance_id: str,
    source_sample: int,
    observed: int,
    *,
    active: bool = False,
) -> dict[str, object]:
    return {
        "event_kind": kind,
        "normalized_utterance_id": utterance_id,
        "event_source_sample": source_sample,
        "observed_source_sample_at_emit": observed,
        "active_state_remained": active,
    }


def test_active_intervals_preserve_causal_observation() -> None:
    rows = active_intervals_from_lifecycle(
        [
            event("speech_start", "u0", 1000, 1512),
            event("speech_end", "u0", 5000, 13000),
            event("terminal", "none", 16000, 16000),
        ],
        16000,
    )
    assert rows == [
        {
            "utterance_id": "u0",
            "start": 1000,
            "end": 5000,
            "start_observed_source_sample": 1512,
            "end_observed_source_sample": 13000,
        }
    ]


def test_terminal_closes_active_interval() -> None:
    rows = active_intervals_from_lifecycle(
        [
            event("speech_start", "u0", 1000, 1512),
            event("terminal", "u0", 16000, 16000, active=True),
        ],
        16000,
    )
    assert rows[0]["end"] == 16000


def test_unpaired_end_fails() -> None:
    with pytest.raises(Phase5ControlError, match="no start"):
        active_intervals_from_lifecycle(
            [event("speech_end", "u0", 5000, 13000)],
            16000,
        )


def test_energy_candidates_use_only_observed_right_window() -> None:
    waveform = np.concatenate(
        [np.full(8000, 0.1, dtype=np.float32), np.full(8000, 0.8, dtype=np.float32)]
    )
    candidates = causal_energy_candidates(
        waveform,
        [
            {
                "start": 0,
                "end": 16000,
                "start_observed_source_sample": 512,
                "end_observed_source_sample": 16512,
            }
        ],
        window_samples=4000,
        step_samples=512,
    )
    strongest = max(candidates, key=lambda row: row["change_strength"])
    assert strongest["boundary_source_sample"] == 7680
    assert strongest["observed_source_sample"] == 11680
    assert all(row["boundary_source_sample"] <= row["observed_source_sample"] for row in candidates)
