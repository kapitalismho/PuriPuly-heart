from __future__ import annotations

import numpy as np

from experiments.speaker_representation_scd.r8_sortformer_feasibility import (
    _chunk_metrics,
    _one_to_one,
    decode_change_events,
)


def test_decoder_emits_overlap_onset_and_direct_replacement() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    assert decode_change_events(probabilities, 0.5, duplicate_suppression_ms=0) == [1280, 3840]


def test_decoder_skips_first_speaker_and_same_speaker_gap_return() -> None:
    probabilities = np.asarray(
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    assert decode_change_events(probabilities, 0.5) == []


def test_decoder_emits_different_speaker_short_gap_only() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    assert decode_change_events(probabilities, 0.5, duplicate_suppression_ms=0) == [2560]


def test_duplicate_suppression_preserves_events_more_than_200ms_apart() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    assert decode_change_events(probabilities, 0.5) == [1280, 5120]


def test_matcher_is_one_to_one() -> None:
    matched, false, misses = _one_to_one([100, 120, 900], [110, 910], 30)
    assert matched == [(100, 110), (900, 910)]
    assert false == [120]
    assert misses == []


def test_backlog_accumulates_across_slow_chunks() -> None:
    rows = [
        {"total_us": 500_000, "compression_called": False, "compression_us": 0},
        {"total_us": 510_000, "compression_called": True, "compression_us": 2_000},
        {"total_us": 300_000, "compression_called": False, "compression_us": 0},
    ]
    metrics = _chunk_metrics(rows, 480.0)
    assert metrics["maximum_backlog_ms"] == 50.0
    assert metrics["deadline_miss_count"] == 2
    assert metrics["compression_call_count"] == 1
