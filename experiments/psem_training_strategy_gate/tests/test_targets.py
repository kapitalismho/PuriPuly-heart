from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    generate_labels,
    load_contract,
)
from experiments.psem_training_strategy_gate.targets import (
    ADJACENT_SOLO_FAMILY,
    CELL_COUNT,
    FUTURE_SAMPLES,
    HOP_SAMPLES,
    OVERLAP_BRIDGE_FAMILY,
    PAST_SAMPLES,
    SILENCE_GAP_FAMILY,
    TargetError,
    build_window_targets,
    nearest_grid_sample,
    valid_center_samples,
)


def _labels(*intervals: CanonicalInterval):
    return generate_labels(
        intervals,
        contract=load_contract(version="psem-handoff-v1"),
        scored_start_sample=intervals[0].start_sample,
        scored_end_sample=intervals[-1].end_sample,
    )


def test_fixed_window_geometry_and_unsnapped_handoff_mapping() -> None:
    labels = _labels(
        CanonicalInterval(0, 35137, ("A",)),
        CanonicalInterval(35137, 80000, ("B",)),
    )
    targets = build_window_targets("session", labels, 35200)
    assert targets.window_start_sample == 3200
    assert targets.window_end_sample == 51200
    assert targets.observed_frontier_sample == 51200
    assert len(targets.cell_centers_sample) == CELL_COUNT
    assert targets.cell_centers_sample[0] == targets.window_start_sample + HOP_SAMPLES // 2
    assert targets.handoff_target == 1
    assert targets.handoff_mask is True
    assert targets.handoff_event_samples == (35137,)
    assert any(
        pair.family == ADJACENT_SOLO_FAMILY and pair.target == 1 for pair in targets.relation_pairs
    )


def test_overlap_return_is_negative_and_overlap_takeover_is_positive() -> None:
    returned = _labels(
        CanonicalInterval(0, 32000, ("A",)),
        CanonicalInterval(32000, 35200, ("A", "B")),
        CanonicalInterval(35200, 80000, ("A",)),
    )
    taken_over = _labels(
        CanonicalInterval(0, 32000, ("A",)),
        CanonicalInterval(32000, 35200, ("A", "B")),
        CanonicalInterval(35200, 80000, ("B",)),
    )
    returned_targets = build_window_targets("return", returned, 35200)
    takeover_targets = build_window_targets("takeover", taken_over, 35200)
    assert returned_targets.handoff_target == 0
    assert takeover_targets.handoff_target == 1
    assert returned_targets.state_targets.count(2) == 2
    assert any(
        pair.family == OVERLAP_BRIDGE_FAMILY and pair.target == 0
        for pair in returned_targets.relation_pairs
    )
    assert any(
        pair.family == OVERLAP_BRIDGE_FAMILY and pair.target == 1
        for pair in takeover_targets.relation_pairs
    )


@pytest.mark.parametrize(("speaker", "target"), [("A", 0), ("B", 1)])
def test_overlap_then_short_gap_keeps_relation_supervision(speaker: str, target: int) -> None:
    labels = _labels(
        CanonicalInterval(0, 32000, ("A",)),
        CanonicalInterval(32000, 35200, ("A", "B")),
        CanonicalInterval(35200, 38400, ()),
        CanonicalInterval(38400, 80000, (speaker,)),
    )
    targets = build_window_targets("mixed", labels, 38400)
    assert any(
        pair.family == OVERLAP_BRIDGE_FAMILY and pair.target == target
        for pair in targets.relation_pairs
    )


def test_long_gap_masks_handoff_and_relation_at_resolution() -> None:
    labels = _labels(
        CanonicalInterval(0, 32000, ("A",)),
        CanonicalInterval(32000, 52800, ()),
        CanonicalInterval(52800, 96000, ("B",)),
    )
    targets = build_window_targets("long-gap", labels, 52800)
    assert targets.handoff_target == 0
    assert targets.handoff_mask is False
    assert not any(pair.family == SILENCE_GAP_FAMILY for pair in targets.relation_pairs)


def test_initial_start_is_masked_and_not_a_handoff() -> None:
    labels = _labels(
        CanonicalInterval(0, 32000, ()),
        CanonicalInterval(32000, 80000, ("A",)),
    )
    targets = build_window_targets("initial", labels, 32000)
    assert targets.handoff_target == 0
    assert targets.handoff_mask is True
    assert not any(
        pair.left_cell <= 19 and pair.right_cell >= 20 for pair in targets.relation_pairs
    )


def test_same_speaker_gap_is_negative_with_declared_relation_family() -> None:
    labels = _labels(
        CanonicalInterval(0, 32000, ("A",)),
        CanonicalInterval(32000, 35200, ()),
        CanonicalInterval(35200, 80000, ("A",)),
    )
    targets = build_window_targets("same-gap", labels, 35200)
    assert targets.handoff_target == 0
    assert targets.handoff_mask is True
    assert any(
        pair.family == SILENCE_GAP_FAMILY and pair.target == 0 for pair in targets.relation_pairs
    )


def test_overlap_to_silence_unresolved_is_masked() -> None:
    labels = _labels(
        CanonicalInterval(0, 40000, ("A",)),
        CanonicalInterval(40000, 44800, ("A", "B")),
        CanonicalInterval(44800, 80000, ()),
    )
    targets = build_window_targets("unresolved", labels, 44800)
    assert targets.handoff_target == 0
    assert targets.handoff_mask is False


def test_complex_overlap_and_ambiguous_boundaries_are_masked() -> None:
    complex_labels = _labels(
        CanonicalInterval(0, 32000, ("A",)),
        CanonicalInterval(32000, 35200, ("A", "B", "C")),
        CanonicalInterval(35200, 80000, ("B",)),
    )
    ambiguous_labels = _labels(
        CanonicalInterval(0, 32000, ("A",)),
        CanonicalInterval(32000, 35200, ("A",), ambiguous=True),
        CanonicalInterval(35200, 80000, ("B",)),
    )
    assert build_window_targets("complex", complex_labels, 32000).handoff_mask is False
    assert build_window_targets("ambiguous", ambiguous_labels, 32000).handoff_mask is False


def test_subcell_ambiguous_span_masks_adjacent_relation() -> None:
    labels = _labels(
        CanonicalInterval(0, 32000, ("A",)),
        CanonicalInterval(32000, 32800, ("A",), ambiguous=True),
        CanonicalInterval(32800, 100000, ("B",)),
    )
    targets = build_window_targets("ambiguous-subcell", labels, 35200)
    assert not any(
        pair.left_cell == 17
        and pair.right_cell == 18
        and pair.family == ADJACENT_SOLO_FAMILY
        for pair in targets.relation_pairs
    )


def test_relation_mask_class_masks_handoff_and_relation_cells() -> None:
    labels = _labels(
        CanonicalInterval(0, 32000, ("A",)),
        CanonicalInterval(
            32000,
            35200,
            ("A",),
            handoff_relation_mask_classes=("ambiguous_nonlexical_vocalization",),
            mask_annotation_ids=("mask-1",),
        ),
        CanonicalInterval(35200, 80000, ("B",)),
    )
    targets = build_window_targets("relation-mask", labels, 32000)
    assert targets.handoff_mask is False
    assert targets.state_mask[20]
    assert all(not (pair.left_cell <= 20 <= pair.right_cell) for pair in targets.relation_pairs)


@pytest.mark.parametrize(
    "labels",
    [
        generate_labels(
            (CanonicalInterval(0, 80000, ("A",)),),
            scored_start_sample=0,
            scored_end_sample=80000,
        ),
        replace(
            _labels(CanonicalInterval(0, 80000, ("A",))),
            contract_document_sha256="0" * 64,
        ),
        replace(
            _labels(CanonicalInterval(0, 80000, ("A",))),
            sample_rate_hz=8000,
        ),
    ],
)
def test_target_layer_rejects_unpinned_label_identity(labels) -> None:
    with pytest.raises(TargetError, match="pinned psem-handoff-v1"):
        build_window_targets("wrong-labels", labels, 32000)


def test_grid_tie_break_and_valid_center_boundaries_are_sample_exact() -> None:
    assert nearest_grid_sample(800) == 0
    assert nearest_grid_sample(801) == 1600
    centers = valid_center_samples(17, 80017)
    assert centers.start >= 17 + PAST_SAMPLES
    assert centers.start % HOP_SAMPLES == 0
    assert centers[-1] + FUTURE_SAMPLES <= 80017
    with pytest.raises(TargetError):
        build_window_targets(
            "unaligned",
            _labels(CanonicalInterval(0, 80000, ("A",))),
            32001,
        )
