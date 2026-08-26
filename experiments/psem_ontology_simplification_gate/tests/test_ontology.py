from __future__ import annotations

import pytest

from experiments.psem_ontology_simplification_gate.derive_anchor_overlap import (
    AnchorOverlapState,
    anchor_overlap_proxy,
    derive_gt_anchor_overlap_state,
    derive_model_anchor_overlap_state,
)
from experiments.psem_ontology_simplification_gate.derive_simple_anchor import (
    SimpleAnchorState,
    derive_gt_simple_anchor_state,
)


@pytest.mark.parametrize(
    ("speech", "anchor", "expected"),
    [
        (False, False, SimpleAnchorState.NO_SPEECH),
        (True, True, SimpleAnchorState.ANCHOR_SPEECH),
        (True, False, SimpleAnchorState.NON_ANCHOR_SPEECH),
    ],
)
def test_simple_anchor_gt_states(speech: bool, anchor: bool, expected: SimpleAnchorState) -> None:
    assert derive_gt_simple_anchor_state(speech_present=speech, anchor_present=anchor) is expected


@pytest.mark.parametrize(
    ("speakers", "expected"),
    [
        (set(), AnchorOverlapState.NO_SPEECH),
        ({"A"}, AnchorOverlapState.ANCHOR_ONLY),
        ({"A", "B"}, AnchorOverlapState.ANCHOR_WITH_OVERLAP),
        ({"A", "B", "C"}, AnchorOverlapState.ANCHOR_WITH_OVERLAP),
        ({"B"}, AnchorOverlapState.NON_ANCHOR_SPEECH),
        ({"B", "C"}, AnchorOverlapState.NON_ANCHOR_SPEECH),
    ],
)
def test_anchor_overlap_gt_states(speakers: set[str], expected: AnchorOverlapState) -> None:
    assert (
        derive_gt_anchor_overlap_state(
            speech_present=bool(speakers),
            anchor_present="A" in speakers,
            anchor_overlap_present="A" in speakers and len(speakers) >= 2,
        )
        is expected
    )


def test_anchor_overlap_gt_rejects_inconsistent_state() -> None:
    with pytest.raises(ValueError, match="cannot be present"):
        derive_gt_anchor_overlap_state(
            speech_present=True,
            anchor_present=False,
            anchor_overlap_present=True,
        )


def test_anchor_overlap_proxy_is_deterministic_conjunction() -> None:
    assert anchor_overlap_proxy(0.4, 0.8) == 0.4
    assert anchor_overlap_proxy(0.9, 0.2) == 0.2


def test_primary_inconsistent_mapping_is_fail_closed() -> None:
    state = derive_model_anchor_overlap_state(
        speech_present=True,
        p_anchor=0.4,
        p_nonanchor_max=0.8,
        anchor_threshold=0.5,
        anchor_overlap_threshold=0.35,
    )
    assert state is AnchorOverlapState.ANCHOR_UNCERTAIN


def test_strict_inconsistent_mapping_is_non_anchor_speech() -> None:
    state = derive_model_anchor_overlap_state(
        speech_present=True,
        p_anchor=0.4,
        p_nonanchor_max=0.8,
        anchor_threshold=0.5,
        anchor_overlap_threshold=0.35,
        strict_inconsistent=True,
    )
    assert state is AnchorOverlapState.NON_ANCHOR_SPEECH
