from __future__ import annotations

import numpy as np
import pytest

from experiments.psem_relative_occupancy_gate.contracts import (
    ActivityInterval,
    Trace,
    evaluation_cells,
    sample_trace_at_cells,
)
from experiments.psem_relative_occupancy_gate.evaluate import (
    monotonic_boundary_matches,
    weighted_average_precision,
    weighted_binary_pr_curve,
)


def test_sortformer_80ms_support_maps_to_100ms_centers() -> None:
    trace = Trace(
        source_id="fixture",
        family="streaming_sortformer",
        slot_ids=("slot-0",),
        probabilities=np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32),
        frame_start_samples=np.asarray([0, 1280, 2560], dtype=np.int64),
        frame_end_samples=np.asarray([1280, 2560, 3840], dtype=np.int64),
        evidence_frontier_samples=np.asarray([17920, 19200, 20480], dtype=np.int64),
        slot_alive=np.ones((3, 1), dtype=np.bool_),
        state_reset=np.asarray([True, False, False], dtype=np.bool_),
        metadata={},
    )
    cells = evaluation_cells((ActivityInterval(0, 4800, (), False),), 0, 4800)
    sampled = sample_trace_at_cells(trace, cells)
    assert sampled["probabilities"][:, 0].tolist() == pytest.approx([0.1, 0.2, 0.0])
    assert sampled["trace_valid"].tolist() == [True, True, False]
    assert sampled["evidence_frontier_samples"].tolist() == [17920, 19200, -1]


def test_trace_rejects_frontier_before_audio() -> None:
    with pytest.raises(ValueError, match="frontier"):
        Trace(
            source_id="fixture",
            family="family",
            slot_ids=("slot-0",),
            probabilities=np.ones((1, 1), dtype=np.float32),
            frame_start_samples=np.asarray([0], dtype=np.int64),
            frame_end_samples=np.asarray([1600], dtype=np.int64),
            evidence_frontier_samples=np.asarray([1599], dtype=np.int64),
            slot_alive=np.ones((1, 1), dtype=np.bool_),
            state_reset=np.asarray([True], dtype=np.bool_),
            metadata={},
        )


def test_trace_requires_initial_reset_marker() -> None:
    with pytest.raises(ValueError, match="initial model-state reset"):
        Trace(
            source_id="fixture",
            family="family",
            slot_ids=("slot-0",),
            probabilities=np.ones((1, 1), dtype=np.float32),
            frame_start_samples=np.asarray([0], dtype=np.int64),
            frame_end_samples=np.asarray([1600], dtype=np.int64),
            evidence_frontier_samples=np.asarray([1600], dtype=np.int64),
            slot_alive=np.ones((1, 1), dtype=np.bool_),
            state_reset=np.asarray([False], dtype=np.bool_),
            metadata={},
        )


def test_empty_trace_maps_to_invalid_cells() -> None:
    trace = Trace(
        source_id="fixture",
        family="family",
        slot_ids=("slot-0", "slot-1"),
        probabilities=np.empty((0, 2), dtype=np.float32),
        frame_start_samples=np.empty(0, dtype=np.int64),
        frame_end_samples=np.empty(0, dtype=np.int64),
        evidence_frontier_samples=np.empty(0, dtype=np.int64),
        slot_alive=np.empty((0, 2), dtype=np.bool_),
        state_reset=np.empty(0, dtype=np.bool_),
        metadata={},
    )
    cells = evaluation_cells((ActivityInterval(0, 1600, (), False),), 0, 1600)
    sampled = sample_trace_at_cells(trace, cells)
    assert sampled["trace_valid"].tolist() == [False]
    assert sampled["evidence_frontier_samples"].tolist() == [-1]


def test_partial_final_cell_keeps_exact_scored_end() -> None:
    cells = evaluation_cells((ActivityInterval(0, 2500, (), False),), 0, 2500)
    assert [(cell.start_sample, cell.end_sample, cell.center_sample) for cell in cells] == [
        (0, 1600, 800),
        (1600, 2500, 2400),
    ]


def test_weighted_pr_accepts_generators_for_multiple_thresholds() -> None:
    curve = weighted_binary_pr_curve(
        (value for value in [True, False]),
        (value for value in [0.9, 0.1]),
        (value for value in [2.0, 1.0]),
        [0.2, 0.8],
    )
    assert [row["f1"] for row in curve] == [1.0, 1.0]
    assert weighted_average_precision([True, False], [0.9, 0.1], [2.0, 1.0]) == 1.0


def test_boundary_alignment_maximizes_count_then_minimizes_displacement() -> None:
    matches = monotonic_boundary_matches([100, 210, 400], [90, 200, 300], 110)
    assert matches == [(0, 0), (1, 1), (2, 2)]
