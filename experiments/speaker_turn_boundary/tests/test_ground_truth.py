from __future__ import annotations

import pytest

from experiments.speaker_turn_boundary.ground_truth import (
    GroundTruthValidationError,
    SpeakerRegion,
    classify_active_speaker_transitions,
    validate_region_sequence,
)


def region(start: int, end: int, speakers: set[str], *, ambiguous: bool = False) -> SpeakerRegion:
    return SpeakerRegion(
        audio_epoch=0,
        start_sample=start,
        end_sample=end,
        speakers=frozenset(speakers),
        ambiguous=ambiguous,
    )


def kinds(transitions) -> list[tuple[str, bool]]:
    return [(transition.kind, transition.positive) for transition in transitions]


def changes(transitions) -> list[tuple[str, int]]:
    return [
        (transition.kind, transition.next_start_sample)
        for transition in transitions
        if transition.positive
    ]


def test_table_initial_start_empty_to_a_is_not_a_change():
    _, transitions = classify_active_speaker_transitions(
        [region(0, 100, set()), region(100, 200, {"A"})]
    )
    assert ("initial_start", False) in kinds(transitions)


def test_table_clean_handoff_a_to_b_is_positive_at_b_onset():
    changes_out, transitions = classify_active_speaker_transitions(
        [region(0, 100, {"A"}), region(100, 200, {"B"})]
    )
    assert ("clean_handoff", True) in kinds(transitions)
    assert [(c.kind, c.change_sample) for c in changes_out] == [("clean_handoff", 100)]
    assert changes(transitions) == [("clean_handoff", 100)]


def test_table_interruption_a_to_ab_is_positive_at_b_onset():
    _, transitions = classify_active_speaker_transitions(
        [region(0, 100, {"A"}), region(100, 200, {"A", "B"})]
    )
    assert ("interruption_onset", True) in kinds(transitions)


def test_table_speaker_left_ab_to_b_is_not_a_change():
    _, transitions = classify_active_speaker_transitions(
        [region(0, 100, {"A", "B"}), region(100, 200, {"B"})]
    )
    assert ("speaker_left", False) in kinds(transitions)
    assert changes(transitions) == []


def test_table_gap_a_to_empty_to_b_is_positive_at_b_onset():
    changes_out, transitions = classify_active_speaker_transitions(
        [
            region(0, 100, {"A"}),
            region(100, 200, set()),
            region(200, 300, {"B"}),
        ]
    )
    assert ("gap_speaker_change", True) in kinds(transitions)
    assert [(c.kind, c.change_sample) for c in changes_out] == [("gap_speaker_change", 200)]


def test_table_gap_a_to_empty_to_a_is_not_a_change():
    _, transitions = classify_active_speaker_transitions(
        [
            region(0, 100, {"A"}),
            region(100, 200, set()),
            region(200, 300, {"A"}),
        ]
    )
    assert ("gap_same_speaker", False) in kinds(transitions)
    assert changes(transitions) == []


def test_gap_persists_across_multiple_silence_regions():
    changes_out, transitions = classify_active_speaker_transitions(
        [
            region(0, 100, {"A"}),
            region(100, 200, set()),
            region(200, 300, set()),
            region(300, 400, {"B"}),
        ]
    )
    assert ("gap_speaker_change", True) in kinds(transitions)
    assert [(c.kind, c.change_sample) for c in changes_out] == [("gap_speaker_change", 300)]


def test_generalized_disjoint_sets_are_clean_handoff():
    _, transitions = classify_active_speaker_transitions(
        [region(0, 100, {"A", "B"}), region(100, 200, {"C"})]
    )
    assert ("clean_handoff", True) in kinds(transitions)


def test_generalized_partial_replacement_is_interruption_onset():
    _, transitions = classify_active_speaker_transitions(
        [region(0, 100, {"A", "B"}), region(100, 200, {"A", "C"})]
    )
    assert ("interruption_onset", True) in kinds(transitions)


def test_same_speaker_split_region_is_not_a_change():
    _, transitions = classify_active_speaker_transitions(
        [region(0, 100, {"A"}), region(100, 200, {"A"})]
    )
    assert ("same_speaker", False) in kinds(transitions)
    assert changes(transitions) == []


def test_ambiguous_region_excludes_positives_and_is_tagged():
    changes_out, transitions = classify_active_speaker_transitions(
        [
            region(0, 100, {"A"}),
            region(100, 200, {"A", "B"}, ambiguous=True),
            region(200, 300, {"B"}),
        ]
    )
    kinds_out = kinds(transitions)
    assert ("ambiguous", False) in kinds_out
    assert ("ambiguous_adjacent", False) in kinds_out
    assert changes_out == []


def test_ambiguous_region_clears_gap_comparison_state():
    changes_out, transitions = classify_active_speaker_transitions(
        [
            region(0, 100, {"A"}),
            region(100, 200, set(), ambiguous=True),
            region(200, 300, {"B"}),
        ]
    )
    assert ("ambiguous", False) in kinds(transitions)
    assert ("ambiguous_adjacent", False) in kinds(transitions)
    assert changes_out == []


def test_silence_start_and_silence_transitions_are_not_positive():
    _, transitions = classify_active_speaker_transitions(
        [
            region(0, 100, {"A"}),
            region(100, 200, set()),
            region(200, 300, set()),
        ]
    )
    assert ("silence_start", False) in kinds(transitions)
    assert ("silence", False) in kinds(transitions)


def test_initial_silence_then_speech_is_initial_start():
    _, transitions = classify_active_speaker_transitions(
        [
            region(0, 100, set()),
            region(100, 200, set()),
            region(200, 300, {"A"}),
        ]
    )
    assert ("initial_start", False) in kinds(transitions)
    assert changes(transitions) == []


def test_multi_step_sequence_handoff_then_leave():
    changes_out, transitions = classify_active_speaker_transitions(
        [
            region(0, 100, {"A"}),
            region(100, 200, {"B"}),
            region(200, 300, {"B", "C"}),
            region(300, 400, {"C"}),
        ]
    )
    assert [(c.kind, c.change_sample) for c in changes_out] == [
        ("clean_handoff", 100),
        ("interruption_onset", 200),
    ]
    assert ("speaker_left", False) in kinds(transitions)


def test_single_region_produces_no_transitions():
    changes_out, transitions = classify_active_speaker_transitions([region(0, 100, {"A"})])
    assert changes_out == []
    assert transitions == []


def test_region_validation_rejects_gap_between_regions():
    with pytest.raises(GroundTruthValidationError):
        classify_active_speaker_transitions([region(0, 100, {"A"}), region(150, 200, {"B"})])


def test_region_validation_rejects_overlap_and_mixed_epochs():
    with pytest.raises(GroundTruthValidationError):
        classify_active_speaker_transitions([region(0, 100, {"A"}), region(50, 200, {"B"})])
    mixed = [
        SpeakerRegion(audio_epoch=0, start_sample=0, end_sample=10, speakers=frozenset()),
        SpeakerRegion(audio_epoch=1, start_sample=10, end_sample=20, speakers=frozenset()),
    ]
    with pytest.raises(GroundTruthValidationError):
        validate_region_sequence(mixed)


def test_region_validation_rejects_bad_samples_and_labels():
    with pytest.raises(GroundTruthValidationError):
        SpeakerRegion(audio_epoch=0, start_sample=-1, end_sample=10, speakers=frozenset())
    with pytest.raises(GroundTruthValidationError):
        SpeakerRegion(audio_epoch=0, start_sample=10, end_sample=10, speakers=frozenset())
    with pytest.raises(GroundTruthValidationError):
        SpeakerRegion(audio_epoch=0, start_sample=0, end_sample=10, speakers=frozenset({""}))


def test_region_to_dict_from_dict_round_trip():
    original = region(0, 100, {"A", "B"})
    restored = SpeakerRegion.from_dict(original.to_dict())
    assert restored == original


def test_gt_classification_is_deterministic_regardless_of_set_iteration():
    first, _ = classify_active_speaker_transitions(
        [region(0, 100, {"A"}), region(100, 200, {"A", "B"})]
    )
    second, _ = classify_active_speaker_transitions(
        [region(0, 100, {"A"}), region(100, 200, {"B", "A"})]
    )
    assert first == second
