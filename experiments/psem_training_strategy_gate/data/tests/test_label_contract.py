from __future__ import annotations

import inspect
import json
from dataclasses import replace

import pytest

from experiments.psem_training_strategy_gate.data import label_contract as module
from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    LabelContractError,
    generate_labels,
    load_contract,
    normalize_intervals,
)

SAMPLE_RATE = 16000


def samples(milliseconds: int) -> int:
    return milliseconds * SAMPLE_RATE // 1000


def timeline(*states: tuple[int, tuple[str, ...], bool]) -> list[CanonicalInterval]:
    cursor = 0
    rows: list[CanonicalInterval] = []
    for milliseconds, speakers, ambiguous in states:
        end = cursor + samples(milliseconds)
        rows.append(
            CanonicalInterval(
                start_sample=cursor,
                end_sample=end,
                active_speakers=tuple(sorted(speakers)),
                ambiguous=ambiguous,
                source_annotation_ids=(f"fixture-{len(rows)}",),
            )
        )
        cursor = end
    return rows


def transition(result, topology: str):
    return next(row for row in result.transitions if row["primary_topology"] == topology)


def episode(result, topology: str):
    return next(row for row in result.topology_episodes if row["primary_topology"] == topology)


def test_contract_is_sample_exact_and_preserves_hard_continuity_value() -> None:
    contract = load_contract()
    assert contract.contract_version == "psem-handoff-v0"
    assert contract.sample_rate_hz == SAMPLE_RATE
    assert contract.coordinate_convention == "zero_based_half_open_unsnapped_source_samples"
    assert contract.grid_mapping == "forbidden_in_dataset_labels"
    assert contract.local_continuity_max_gap_samples == samples(1200)
    assert contract.reliable_solo_min_duration_samples == samples(200)


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_rate_hz": 8000},
        {"sample_rate_hz": 16000.0},
        {"local_continuity_max_gap_ms": 1000},
        {"reliable_solo_min_duration_ms": 100},
        {"contract_version": "psem-handoff-v1"},
        {"status": "frozen"},
        {"coordinate_convention": "rounded_grid_samples"},
        {"grid_mapping": "round_to_100_ms_cells"},
    ],
)
def test_supplied_contracts_enforce_installed_version(changes: dict[str, object]) -> None:
    invalid_contract = replace(load_contract(), **changes)
    with pytest.raises(LabelContractError):
        generate_labels(timeline((400, ("A",), False)), contract=invalid_contract)


def test_contract_json_rejects_non_integral_numbers(tmp_path) -> None:
    raw = json.loads(module.CONTRACT_PATH.read_text(encoding="utf-8"))
    raw["source_coordinate_convention"]["sample_rate_hz"] = 16000.9
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(LabelContractError, match="exact integer"):
        load_contract(path)


def test_contract_identity_binds_all_semantic_json_sections(tmp_path) -> None:
    installed = load_contract()
    raw = json.loads(module.CONTRACT_PATH.read_text(encoding="utf-8"))
    raw["primary_event"]["overlap_onset_is_handoff"] = True
    path = tmp_path / "semantic-change.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    altered = load_contract(path)
    assert altered.document_sha256 != installed.document_sha256
    with pytest.raises(LabelContractError, match="installed contract"):
        generate_labels(timeline((400, ("A",), False)), contract=altered)
    result = generate_labels(timeline((400, ("A",), False)))
    assert result.contract_document_sha256 == installed.document_sha256


def test_installed_contract_document_rejects_same_version_drift(
    tmp_path, monkeypatch
) -> None:
    raw = json.loads(module.CONTRACT_PATH.read_text(encoding="utf-8"))
    raw["constants_ms"]["reliable_solo_min_duration"] = 100
    path = tmp_path / "installed-contract.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(module, "CONTRACT_PATH", path)
    with pytest.raises(LabelContractError, match="pinned version identity"):
        generate_labels(timeline((400, ("A",), False)))


def test_normalization_requires_a_complete_non_overlapping_timeline() -> None:
    with pytest.raises(LabelContractError, match="gap"):
        normalize_intervals(
            [
                CanonicalInterval(0, 100, ("A",)),
                CanonicalInterval(101, 200, ("B",)),
            ]
        )
    with pytest.raises(LabelContractError, match="overlap"):
        normalize_intervals(
            [
                CanonicalInterval(0, 100, ("A",)),
                CanonicalInterval(99, 200, ("B",)),
            ]
        )


def test_normalization_rejects_unordered_canonical_intervals() -> None:
    with pytest.raises(LabelContractError, match="source order"):
        normalize_intervals(
            [
                CanonicalInterval(100, 200, ("B",)),
                CanonicalInterval(0, 100, ("A",)),
            ]
        )


def test_non_integral_source_samples_are_rejected_without_truncation() -> None:
    with pytest.raises(LabelContractError, match="exact integer"):
        normalize_intervals(
            [
                {
                    "start_sample": 0,
                    "end_sample": 6400.9,
                    "active_speakers": ["A"],
                }
            ]
        )
    with pytest.raises(LabelContractError, match="exact integer"):
        normalize_intervals([CanonicalInterval(0.0, 6400, ("A",))])


@pytest.mark.parametrize(
    "updates",
    [
        {"active_speakers": "AB"},
        {"active_speakers": ["A", "A"]},
        {"speaker_identity_known": "false"},
        {"ambiguous": 0},
        {"source_annotation_ids": "annotation-1"},
    ],
)
def test_malformed_interval_metadata_is_rejected(updates: dict[str, object]) -> None:
    interval: dict[str, object] = {
        "start_sample": 0,
        "end_sample": samples(400),
        "active_speakers": ["A"],
    }
    interval.update(updates)
    with pytest.raises(LabelContractError):
        normalize_intervals([interval])


def test_same_speaker_continuation_has_no_handoff() -> None:
    result = generate_labels(timeline((400, ("A",), False), (400, ("A",), False)))
    assert len(result.intervals) == 1
    assert [row["handoff_confirmed"] for row in result.transitions] == [0]
    assert result.exposure["stable_singleton_samples"] == samples(800)


def test_direct_different_speaker_handoff_uses_unsnapped_b_onset() -> None:
    result = generate_labels(timeline((400, ("A",), False), (400, ("B",), False)))
    row = transition(result, "clean_direct_different_speaker_handoff")
    assert row["handoff_confirmed"] == 1
    assert row["handoff_source_sample"] == samples(400)
    assert row["relation_target"] == "different"
    assert row["coverage_gate_eligible"] is True


def test_same_speaker_gap_resume_is_a_coverage_negative() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (300, (), False),
            (400, ("A",), False),
        )
    )
    row = transition(result, "same_speaker_silence_gap_resume")
    assert row["handoff_confirmed"] == 0
    assert row["relation_target"] == "same"
    assert row["gap_samples"] == samples(300)
    assert row["coverage_gate_eligible"] is True


def test_different_speaker_gap_handoff_is_at_b_onset() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (300, (), False),
            (400, ("B",), False),
        )
    )
    row = transition(result, "silence_gap_different_speaker_handoff")
    assert row["handoff_confirmed"] == 1
    assert row["handoff_source_sample"] == samples(700)
    assert row["relation_target"] == "different"


def test_overlap_return_does_not_inherit_overlap_onset_as_an_event() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (200, ("A", "B"), False),
            (400, ("A",), False),
        )
    )
    row = transition(result, "overlap_return")
    assert row["handoff_confirmed"] == 0
    assert row["handoff_source_sample"] is None
    assert row["relation_target"] == "same"
    assert row["coverage_gate_eligible"] is True
    assert all(
        candidate["handoff_source_sample"] != samples(400) for candidate in result.transitions
    )


def test_overlap_takeover_handoff_is_first_b_only_sample() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (200, ("A", "B"), False),
            (400, ("B",), False),
        )
    )
    row = transition(result, "overlap_takeover")
    assert row["handoff_confirmed"] == 1
    assert row["handoff_source_sample"] == samples(600)
    assert row["relation_target"] == "different"


def test_unknown_identity_inside_overlap_masks_takeover_supervision() -> None:
    result = generate_labels(
        [
            CanonicalInterval(0, samples(400), ("A",)),
            CanonicalInterval(
                samples(400),
                samples(600),
                ("A", "B"),
                speaker_identity_known=False,
            ),
            CanonicalInterval(samples(600), samples(1000), ("B",)),
        ]
    )
    row = transition(result, "unknown_speaker_crossing")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"
    assert row["coverage_gate_eligible"] is False


@pytest.mark.parametrize(
    ("final_speaker", "topology", "handoff_target", "source_sample"),
    [
        ("B", "overlap_gap_takeover", 1, samples(900)),
        ("A", "overlap_gap_return", 0, None),
    ],
)
def test_overlap_then_short_gap_resolves_at_reliable_solo(
    final_speaker: str,
    topology: str,
    handoff_target: int,
    source_sample: int | None,
) -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (200, ("A", "B"), False),
            (300, (), False),
            (400, (final_speaker,), False),
        )
    )
    row = transition(result, topology)
    assert row["handoff_confirmed"] == handoff_target
    assert row["handoff_source_sample"] == source_sample
    assert row["relation_target"] == ("different" if handoff_target else "same")
    assert row["coverage_gate_eligible"] is False


def test_overlap_then_long_gap_remains_continuity_unknown() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (200, ("A", "B"), False),
            (1300, (), False),
            (400, ("B",), False),
        )
    )
    row = transition(result, "continuity_unknown")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"


def test_overlap_to_silence_is_unresolved_and_masked() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (200, ("A", "B"), False),
            (400, (), False),
        )
    )
    row = transition(result, "overlap_to_silence_unresolved")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"
    assert row["coverage_gate_eligible"] is False


def test_overlap_to_silence_with_sub_reliable_tail_remains_unresolved() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (200, ("A", "B"), False),
            (200, (), False),
            (100, ("B",), False),
        )
    )
    row = transition(result, "overlap_to_silence_unresolved")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"


def test_short_backchannel_retains_two_handoffs_but_counts_one_episode() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (600, ("B",), False),
            (400, ("A",), False),
        )
    )
    handoffs = [row for row in result.transitions if row["handoff_confirmed"] == 1]
    assert [row["handoff_source_sample"] for row in handoffs] == [
        samples(400),
        samples(1000),
    ]
    short = episode(result, "short_backchannel_return")
    assert len(short["transition_ids"]) == 2
    assert short["coverage_gate_eligible"] is True
    assert all(row["coverage_gate_eligible"] is False for row in handoffs)


def test_overlap_takeover_entry_is_exclusive_short_backchannel_coverage() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (200, ("A", "B"), False),
            (600, ("B",), False),
            (400, ("A",), False),
        )
    )
    short = episode(result, "short_backchannel_return")
    assert short["coverage_gate_eligible"] is True
    assert "entry:overlap_takeover" in short["secondary_tags"]
    assert all(
        row["coverage_gate_eligible"] is False
        for row in result.transitions
        if row["transition_id"] in short["transition_ids"]
    )


def test_initial_silence_to_solo_is_not_a_handoff() -> None:
    result = generate_labels(timeline((300, (), False), (400, ("A",), False)))
    row = transition(result, "initial_start")
    assert row["handoff_confirmed"] == 0
    assert row["handoff_source_sample"] is None
    assert row["relation_target"] is None


def test_micro_gap_can_be_a_target_but_not_official_gap_coverage() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (70, (), False),
            (400, ("B",), False),
        )
    )
    row = transition(result, "micro_gap_different_speaker_handoff")
    assert row["handoff_confirmed"] == 1
    assert row["handoff_source_sample"] == samples(470)
    assert row["coverage_gate_eligible"] is False
    assert "micro_gap" in row["secondary_tags"]


def test_boundary_jitter_gap_is_reconciled_as_direct_coverage() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (50, (), False),
            (400, ("B",), False),
        )
    )
    row = transition(result, "clean_direct_different_speaker_handoff")
    assert row["handoff_source_sample"] == samples(450)
    assert row["coverage_gate_eligible"] is True
    assert "boundary_jitter_gap" in row["secondary_tags"]


def test_same_speaker_jitter_fragments_form_reliable_solo_before_handoff() -> None:
    result = generate_labels(
        timeline(
            (150, ("A",), False),
            (50, (), False),
            (150, ("A",), False),
            (400, ("B",), False),
        )
    )
    assert len(result.intervals) == 2
    assert result.intervals[0].active_speakers == ("A",)
    assert result.intervals[0].duration_samples == samples(350)
    row = transition(result, "clean_direct_different_speaker_handoff")
    assert row["handoff_confirmed"] == 1
    assert row["handoff_source_sample"] == samples(350)


def test_micro_overlap_can_be_a_target_but_not_official_overlap_coverage() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (70, ("A", "B"), False),
            (400, ("B",), False),
        )
    )
    row = transition(result, "micro_overlap_takeover")
    assert row["handoff_confirmed"] == 1
    assert row["handoff_source_sample"] == samples(470)
    assert row["coverage_gate_eligible"] is False
    assert "micro_overlap" in row["secondary_tags"]


def test_long_gap_masks_continuity_and_handoff() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (1300, (), False),
            (400, ("B",), False),
        )
    )
    row = transition(result, "continuity_unknown")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"


def test_complex_overlap_transition_is_masked() -> None:
    result = generate_labels(
        timeline(
            (300, ("A", "B"), False),
            (300, ("B", "C"), False),
        )
    )
    row = transition(result, "complex_overlap_transition")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"
    resolved = generate_labels(
        timeline(
            (300, ("A", "B"), False),
            (300, ("B", "C"), False),
            (400, ("C",), False),
        )
    )
    initial = transition(resolved, "initial_start")
    assert initial["handoff_confirmed"] is None
    assert initial["mask_state"] == "masked"


def test_three_speaker_overlap_return_is_masked_as_complex() -> None:
    result = generate_labels(
        timeline(
            (300, ("A",), False),
            (300, ("A", "B", "C"), False),
            (300, ("A",), False),
        )
    )
    row = transition(result, "complex_overlap_transition")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"


def test_adjacent_short_returns_do_not_double_count_a_transition() -> None:
    result = generate_labels(
        timeline(
            (300, ("A",), False),
            (300, ("B",), False),
            (300, ("A",), False),
            (300, ("B",), False),
        )
    )
    short = [
        row
        for row in result.topology_episodes
        if row["primary_topology"] == "short_backchannel_return"
    ]
    assert len(short) == 1
    transition_ids = [transition_id for row in short for transition_id in row["transition_ids"]]
    assert len(transition_ids) == len(set(transition_ids))


def test_ambiguous_annotation_crossing_masks_handoff_and_relation() -> None:
    result = generate_labels(
        timeline(
            (400, ("A",), False),
            (100, (), True),
            (400, ("B",), False),
        )
    )
    row = transition(result, "ambiguous_annotation_crossing")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"
    assert result.exposure["ambiguous_samples"] == samples(100)


def test_unknown_speaker_identity_is_not_reliable_handoff_supervision() -> None:
    result = generate_labels(
        [
            CanonicalInterval(
                0,
                samples(400),
                ("unknown-1",),
                speaker_identity_known=False,
            ),
            CanonicalInterval(samples(400), samples(800), ("B",)),
        ]
    )
    assert all(row["handoff_confirmed"] is None for row in result.transitions)
    assert result.activity_labels[0]["state"] == "singleton"
    assert result.activity_labels[0]["mask_state"] == "valid"
    row = transition(result, "unknown_speaker_region")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"


def test_unknown_identity_with_known_speaker_count_retains_overlap_activity() -> None:
    result = generate_labels(
        [
            CanonicalInterval(
                0,
                samples(300),
                ("unknown-1", "unknown-2"),
                speaker_identity_known=False,
            )
        ]
    )
    assert result.activity_labels[0]["state"] == "overlap"
    assert result.activity_labels[0]["mask_state"] == "valid"
    assert result.exposure["ongoing_overlap_samples"] == samples(300)
    row = transition(result, "unknown_speaker_region")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None


def test_ambiguous_prefix_masks_the_first_reliable_solo_target() -> None:
    result = generate_labels(
        [
            CanonicalInterval(0, samples(100), (), ambiguous=True),
            CanonicalInterval(samples(100), samples(500), ("B",)),
        ]
    )
    row = transition(result, "initial_start")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"
    assert row["secondary_tags"] == ["unresolved_scored_prefix"]


def test_unknown_active_speech_without_ids_is_not_silence_or_gap_coverage() -> None:
    result = generate_labels(
        [
            CanonicalInterval(0, samples(400), ("A",)),
            CanonicalInterval(
                samples(400),
                samples(500),
                (),
                speaker_identity_known=False,
            ),
            CanonicalInterval(samples(500), samples(900), ("B",)),
        ]
    )
    activity = result.activity_labels[1]
    assert activity["state"] == "unknown_speech"
    assert activity["mask_state"] == "masked"
    row = transition(result, "unknown_speaker_crossing")
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["coverage_gate_eligible"] is False
    assert result.exposure["unknown_identity_samples"] == samples(100)
    assert not any(
        episode_row["primary_topology"]
        in {
            "silence_gap_different_speaker_handoff",
            "same_speaker_silence_gap_resume",
        }
        and episode_row["coverage_gate_eligible"]
        for episode_row in result.topology_episodes
    )


def test_isolated_three_speaker_overlap_has_a_masked_diagnostic_topology() -> None:
    result = generate_labels([CanonicalInterval(0, samples(300), ("A", "B", "C"))])
    row = transition(result, "complex_overlap_region")
    assert result.activity_labels[0]["state"] == "overlap"
    assert result.activity_labels[0]["mask_state"] == "valid"
    assert row["handoff_confirmed"] is None
    assert row["relation_target"] is None
    assert row["mask_state"] == "masked"
    assert row["coverage_gate_eligible"] is False
    assert row["overlap_samples"] == samples(300)
    resolved = generate_labels(
        [
            CanonicalInterval(0, samples(300), ("A", "B", "C")),
            CanonicalInterval(samples(300), samples(700), ("B",)),
        ]
    )
    initial = transition(resolved, "initial_start")
    assert initial["handoff_confirmed"] is None
    assert initial["mask_state"] == "masked"


def test_label_generator_does_not_import_forbidden_old_event_modules() -> None:
    source = inspect.getsource(module)
    assert "r7_relation_verifier" not in source
    assert "r7b_local_segmentation" not in source
    assert "new_speaker_onset" not in source
