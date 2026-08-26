from __future__ import annotations

import pytest

from experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies import (
    SimplifiedReplacementDecoder,
    _apply_speech_gate,
    _causal_exposure,
    _decode_episode,
    _episode_anchor_records,
    _episode_exposure_ranges,
    _linear_fail_closed_exposure,
    _product_event_metrics,
)
from experiments.psem_relative_occupancy_gate.contracts import (
    ActivityInterval,
    AnchorLifecycle,
    EvaluationCell,
)
from experiments.psem_relative_occupancy_gate.decoder import simulate_gt_session
from experiments.psem_relative_occupancy_gate.model_decode import (
    ModelObservation,
    PosteriorCell,
)
from experiments.psem_relative_occupancy_gate.model_evaluate import (
    gate1_fail_closed_exposure,
    product_event_metrics,
)


def _advance(
    decoder: SimplifiedReplacementDecoder,
    start: int,
    end: int,
    *,
    replacement: bool,
    pause: bool = False,
    lifecycle: AnchorLifecycle = AnchorLifecycle.ANCHORED,
):
    return decoder.advance(
        start_sample=start,
        end_sample=end,
        evidence_frontier_sample=end,
        replacement_evidence=replacement,
        pause=pause,
        lifecycle=lifecycle,
        anchor_id="A" if lifecycle is AnchorLifecycle.ANCHORED else None,
        anchor_episode_id="episode-1" if lifecycle is AnchorLifecycle.ANCHORED else None,
    )


def _diagnostic_cell(
    index: int,
    *,
    trace_valid: bool = True,
    state_reset: bool = False,
    anchor_alive: bool = True,
    masked: bool = False,
) -> PosteriorCell:
    start = index * 1600
    end = start + 1600
    return PosteriorCell(
        cell=EvaluationCell(
            index=index,
            start_sample=start,
            end_sample=end,
            center_sample=start + 800,
            active_speakers=("A",),
            masked=masked,
        ),
        probabilities=(0.9, 0.1),
        slot_alive=(anchor_alive, True),
        evidence_frontier_sample=end,
        state_reset=state_reset,
        trace_valid=trace_valid,
    )


@pytest.mark.parametrize(
    "trigger",
    (
        {"trace_valid": False},
        {"state_reset": True},
        {"anchor_alive": False},
    ),
)
def test_diagnostic_records_remain_fail_closed_after_continuity_loss(
    trigger: dict[str, bool],
) -> None:
    records, invalidated, invalidated_active = _episode_anchor_records(
        source_id="source",
        episode_id="episode-1",
        anchor_speaker="A",
        anchor_slot_index=0,
        episode_start=0,
        episode_end=4800,
        cells=(
            _diagnostic_cell(0),
            _diagnostic_cell(1, **trigger),
            _diagnostic_cell(2),
        ),
    )
    assert [value.start_sample for value in records] == [0]
    assert invalidated == 3200
    assert invalidated_active == 3200


def test_masked_diagnostic_reset_invalidates_before_mask_handling() -> None:
    records, invalidated, invalidated_active = _episode_anchor_records(
        source_id="source",
        episode_id="episode-1",
        anchor_speaker="A",
        anchor_slot_index=0,
        episode_start=0,
        episode_end=4800,
        cells=(
            _diagnostic_cell(0),
            _diagnostic_cell(1, state_reset=True, masked=True),
            _diagnostic_cell(2),
        ),
    )
    assert [value.start_sample for value in records] == [0]
    assert invalidated == 3200
    assert invalidated_active == 1600


def test_valid_diagnostic_mask_pauses_without_invalidating_continuity() -> None:
    records, invalidated, invalidated_active = _episode_anchor_records(
        source_id="source",
        episode_id="episode-1",
        anchor_speaker="A",
        anchor_slot_index=0,
        episode_start=0,
        episode_end=4800,
        cells=(
            _diagnostic_cell(0),
            _diagnostic_cell(1, masked=True),
            _diagnostic_cell(2),
        ),
    )
    assert [value.start_sample for value in records] == [0, 3200]
    assert invalidated == 0
    assert invalidated_active == 0


def test_exact_source_boundary_and_evidence_frontier_control_emission() -> None:
    decoder = SimplifiedReplacementDecoder("source", 1600)
    assert _advance(decoder, 0, 800, replacement=True) is None
    event = _advance(decoder, 800, 1600, replacement=True)
    assert event is not None
    assert event.boundary_source_sample == 0
    assert event.decoder_emit_sample == 1600
    assert event.model_evidence_frontier_sample == 1600


def test_mask_pauses_replacement_evidence() -> None:
    decoder = SimplifiedReplacementDecoder("source", 1600)
    assert _advance(decoder, 0, 800, replacement=True) is None
    assert _advance(decoder, 800, 1600, replacement=False, pause=True) is None
    event = _advance(decoder, 1600, 2400, replacement=True)
    assert event is not None
    assert event.boundary_source_sample == 0
    assert event.decoder_emit_sample == 2400


def test_no_speech_or_anchor_speech_clears_pending_evidence() -> None:
    decoder = SimplifiedReplacementDecoder("source", 1600)
    assert _advance(decoder, 0, 800, replacement=True) is None
    assert _advance(decoder, 800, 1600, replacement=False) is None
    assert _advance(decoder, 1600, 2400, replacement=True) is None


def test_unanchored_and_uncertain_lifecycle_disable_cuts() -> None:
    for lifecycle in (
        AnchorLifecycle.UNANCHORED,
        AnchorLifecycle.ANCHOR_UNCERTAIN,
    ):
        decoder = SimplifiedReplacementDecoder("source", 800)
        assert (
            _advance(
                decoder,
                0,
                1600,
                replacement=True,
                lifecycle=lifecycle,
            )
            is None
        )


def test_decoder_has_no_candidate_identity_or_handover_memory() -> None:
    decoder = SimplifiedReplacementDecoder("source", 1600)
    names = set(vars(decoder))
    assert not any("candidate" in value or "handover" in value for value in names)


def test_mid_episode_reset_remains_fail_closed_for_old_slot() -> None:
    observations = (
        ModelObservation(
            start_sample=0,
            end_sample=800,
            probabilities=(0.9, 0.1),
            slot_alive=(True, True),
            trace_valid=True,
            state_reset=False,
            masked=False,
            speech_present=True,
            evidence_frontier_sample=800,
        ),
        ModelObservation(
            start_sample=800,
            end_sample=1600,
            probabilities=(0.9, 0.1),
            slot_alive=(True, True),
            trace_valid=True,
            state_reset=True,
            masked=False,
            speech_present=True,
            evidence_frontier_sample=1600,
        ),
        ModelObservation(
            start_sample=1600,
            end_sample=3200,
            probabilities=(0.1, 0.9),
            slot_alive=(True, True),
            trace_valid=True,
            state_reset=False,
            masked=False,
            speech_present=True,
            evidence_frontier_sample=3200,
        ),
    )
    assert (
        _decode_episode(
            source_id="source",
            episode_id="episode-1",
            anchor_id="A",
            anchor_slot_index=0,
            episode_start=0,
            episode_end=3200,
            observations=observations,
            candidate="simple_anchor",
            anchor_threshold=0.5,
            overlap_threshold=None,
            strict_inconsistent=False,
            confirmation_samples=800,
        )
        is None
    )


def test_candidate_b_inconsistent_support_is_reported_uncertain() -> None:
    observation = ModelObservation(
        start_sample=0,
        end_sample=1600,
        probabilities=(0.4, 0.5),
        slot_alive=(True, True),
        trace_valid=True,
        state_reset=False,
        masked=False,
        speech_present=True,
        evidence_frontier_sample=1600,
    )
    anchored, uncertain = _episode_exposure_ranges(
        observations=(observation,),
        episode_start=0,
        episode_end=1600,
        candidate="anchor_overlap",
        anchor_slot_index=0,
        anchor_threshold=0.5,
        overlap_threshold=0.35,
        strict_inconsistent=False,
    )
    assert anchored == []
    assert uncertain == [(0, 1600)]


def test_masked_reset_remains_fail_closed_for_old_slot() -> None:
    observations = (
        ModelObservation(
            start_sample=0,
            end_sample=800,
            probabilities=(0.9, 0.1),
            slot_alive=(True, True),
            trace_valid=True,
            state_reset=False,
            masked=False,
            speech_present=True,
            evidence_frontier_sample=800,
        ),
        ModelObservation(
            start_sample=800,
            end_sample=1600,
            probabilities=(0.9, 0.1),
            slot_alive=(True, True),
            trace_valid=True,
            state_reset=True,
            masked=True,
            speech_present=True,
            evidence_frontier_sample=1600,
        ),
        ModelObservation(
            start_sample=1600,
            end_sample=3200,
            probabilities=(0.1, 0.9),
            slot_alive=(True, True),
            trace_valid=True,
            state_reset=False,
            masked=False,
            speech_present=True,
            evidence_frontier_sample=3200,
        ),
    )
    assert (
        _decode_episode(
            source_id="source",
            episode_id="episode-1",
            anchor_id="A",
            anchor_slot_index=0,
            episode_start=0,
            episode_end=3200,
            observations=observations,
            candidate="simple_anchor",
            anchor_threshold=0.5,
            overlap_threshold=None,
            strict_inconsistent=False,
            confirmation_samples=800,
        )
        is None
    )


def test_candidate_uncertainty_reclassifies_only_old_anchored_speech() -> None:
    row = {
        "timeline": [
            {
                "start_sample": 0,
                "end_sample": 1600,
                "masked": False,
                "speech_present": True,
                "lifecycle": "ANCHORED",
            },
            {
                "start_sample": 1600,
                "end_sample": 3200,
                "masked": False,
                "speech_present": True,
                "lifecycle": "UNANCHORED",
            },
            {
                "start_sample": 3200,
                "end_sample": 4800,
                "masked": False,
                "speech_present": True,
                "lifecycle": "ANCHORED",
            },
        ]
    }
    exposure = _causal_exposure(row, [(800, 2400), (4000, 4800)])
    assert exposure["unanchored_active_speech_seconds"] == 0.1
    assert exposure["anchor_uncertain_active_speech_seconds"] == 0.1
    assert exposure["speaker_protection_enabled_fraction"] == pytest.approx(1 / 3)


def test_linear_fail_closed_exposure_matches_predecessor_definition() -> None:
    intervals = (
        ActivityInterval(0, 1600, ("A",), False),
        ActivityInterval(1600, 3200, ("B",), False),
        ActivityInterval(3200, 4800, ("A",), True),
        ActivityInterval(4800, 6400, (), False),
    )
    anchored = [(0, 800), (800, 1600)]
    uncertain = [(1600, 2400), (2400, 3200)]
    expected = gate1_fail_closed_exposure(
        intervals=intervals,
        anchored_ranges=anchored,
        uncertain_ranges=uncertain,
        exact_contamination_seconds=0.25,
    )
    actual = _linear_fail_closed_exposure(
        intervals=intervals,
        anchored_ranges=anchored,
        uncertain_ranges=uncertain,
        exact_contamination_seconds=0.25,
    )
    assert actual == expected


def test_production_speech_gate_splits_exact_source_support() -> None:
    observation = ModelObservation(
        start_sample=0,
        end_sample=1600,
        probabilities=(0.5, 0.1),
        slot_alive=(True, True),
        trace_valid=True,
        state_reset=True,
        masked=False,
        speech_present=False,
        evidence_frontier_sample=2000,
    )
    pieces = _apply_speech_gate((observation,), [{"start_sample": 320, "end_sample": 1280}])
    assert [(value.start_sample, value.end_sample, value.speech_present) for value in pieces] == [
        (0, 320, False),
        (320, 1280, True),
        (1280, 1600, False),
    ]
    assert [value.state_reset for value in pieces] == [True, False, False]


def test_optimized_product_metrics_match_issue97_reference() -> None:
    intervals = (
        ActivityInterval(0, 1600, ("A",), False),
        ActivityInterval(1600, 3200, ("B",), False),
        ActivityInterval(3200, 4800, (), False),
        ActivityInterval(4800, 6400, ("B",), False),
    )
    reference = simulate_gt_session(
        source_id="source",
        intervals=intervals,
        confirmation_samples=800,
        enrollment_samples=800,
        silence_reset_samples=800,
    )
    contamination = [
        (
            value.anchor_speaker,
            value.anchor_emit_sample,
            value.end_emit_sample,
        )
        for value in reference.episodes
    ]
    expected = product_event_metrics(
        predicted_events=reference.events,
        reference=reference,
        intervals=intervals,
        contamination_episodes=contamination,
        tolerance_samples=800,
    )
    actual = _product_event_metrics(
        predicted_events=reference.events,
        reference=reference,
        intervals=intervals,
        contamination_episodes=contamination,
        tolerance_samples=800,
    )
    assert actual == expected
