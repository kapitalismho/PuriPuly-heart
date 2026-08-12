from __future__ import annotations

import random

from experiments.speaker_turn_boundary.turn_episode.phase5_scoring import (
    jittered_scorable_intervals,
    score_policy_episode,
)
from experiments.speaker_turn_boundary.turn_episode.schemas import ReferenceAction
from experiments.speaker_turn_boundary.turn_episode.scoring import (
    Action,
    _match_weight,
    action_eligible,
    match_episode,
)


def action(
    action_id: str,
    boundary: int,
    observed: int,
    *,
    origin: str = "detector",
    action_kind: str | None = None,
    associated_vad_action_id: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "final_action_id": action_id,
        "action_kind": action_kind
        or ("add_hard_boundary" if origin == "detector" else "retain_vad"),
        "origin": origin,
        "event_id": action_id.removeprefix("final:"),
        "boundary_source_sample": boundary,
        "observed_source_sample_at_emit": observed,
        "source_session_id": "source",
        "audio_epoch": 1,
    }
    if associated_vad_action_id is not None:
        row["associated_vad_action_id"] = associated_vad_action_id
    return row


def b0_action(action_id: str, boundary: int, observed: int) -> dict[str, object]:
    return {
        "action_id": action_id,
        "action_kind": "retain_vad",
        "origin": "vad",
        "boundary_source_sample": boundary,
        "observed_source_sample_at_emit": observed,
        "source_session_id": "source",
        "audio_epoch": 1,
    }


def reference(boundary: int) -> ReferenceAction:
    return ReferenceAction(
        reference_id="source:episode:target",
        audio_epoch=1,
        source_session_id="source",
        action_kind="hard_boundary",
        target_sample=boundary,
        acceptable_interval=(boundary - 1000, boundary + 1000),
        evidence_onset_sample=boundary,
        scorable=True,
        primary_case=True,
        episode_pool_tag="hard_only",
    )


def ambiguous_reference(reference_id: str, target: int) -> ReferenceAction:
    return ReferenceAction(
        reference_id=reference_id,
        audio_epoch=1,
        source_session_id="source",
        action_kind="hard_boundary",
        target_sample=target,
        acceptable_interval=(9000, 12000),
        evidence_onset_sample=target,
        scorable=True,
        primary_case=True,
        episode_pool_tag="hard_only",
    )


def test_contamination_latches_across_a_b_a() -> None:
    result = score_policy_episode(
        [],
        [],
        [],
        [(0, 4000, "A"), (4000, 8000, "B"), (8000, 12000, "A")],
        [],
        [],
        None,
        [],
        scored_start=0,
        scored_end=12000,
        episode_tag="hard_only",
    )
    assert result["clean_gap_contaminated_samples"] == 8000
    assert result["clean_gap_singleton_denominator_samples"] == 12000


def test_matched_action_can_also_be_harmful() -> None:
    result = score_policy_episode(
        [action("a0", 8000, 10000)],
        [],
        [reference(8000)],
        [(0, 16000, "A")],
        [],
        [],
        [(7000, 9000)],
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="hard_only",
    )
    assert result["match_count"] == 1
    assert result["harm_or_structure_counts"]["harmful_active_split"] == 1
    assert result["harm_or_structure_counts"]["lexical_split"] == 1


def test_overlap_episode_is_not_clean_gap_headline() -> None:
    result = score_policy_episode(
        [action("a0", 8000, 10000)],
        [],
        [],
        [(0, 8000, "A"), (12000, 16000, "B")],
        [],
        [(8000, 12000)],
        None,
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="overlap_present",
    )
    assert result["clean_gap_headline_eligible"] is False
    assert result["clean_gap_contaminated_samples"] is None
    assert result["harm_or_structure_counts"]["overlap_hard_action"] == 1


def test_duplicate_reference_actions_are_counted() -> None:
    result = score_policy_episode(
        [action("a0", 7900, 9000), action("a1", 8100, 10000)],
        [],
        [reference(8000)],
        [(0, 16000, "A")],
        [],
        [],
        None,
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="hard_only",
    )
    assert result["harm_or_structure_counts"]["duplicate_hard_boundary"] == 1


def test_b0_acceleration_and_true_recovery_are_distinct() -> None:
    accelerated = score_policy_episode(
        [
            action(
                "final:d0",
                8000,
                9000,
                action_kind="accelerate_or_replace_vad",
                associated_vad_action_id="v0",
            )
        ],
        [b0_action("v0", 8000, 12000)],
        [reference(8000)],
        [(0, 8000, "A"), (8000, 16000, "B")],
        [],
        [],
        None,
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="hard_only",
    )
    recovered = score_policy_episode(
        [action("final:d1", 8000, 9000)],
        [],
        [reference(8000)],
        [(0, 8000, "A"), (8000, 16000, "B")],
        [],
        [],
        None,
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="hard_only",
    )
    assert accelerated["matches"][0]["benefit_attribution"] == "accelerated_b0_success"
    assert recovered["matches"][0]["benefit_attribution"] == "recovered_b0_hard_miss"


def test_ambiguous_candidate_matching_preserves_independent_b0_assignment() -> None:
    result = score_policy_episode(
        [
            action(
                "final:linked",
                9500,
                12000,
                action_kind="accelerate_or_replace_vad",
                associated_vad_action_id="v0",
            ),
            action("final:extra", 10000, 12000),
        ],
        [b0_action("v0", 9000, 12000)],
        [ambiguous_reference("r0", 11000), ambiguous_reference("r1", 10500)],
        [(0, 8000, "A"), (8000, 16000, "B")],
        [],
        [],
        None,
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="hard_only",
    )
    assert [(row["reference_id"], row["action_id"]) for row in result["matches"]] == [
        ("r0", "final:linked")
    ]
    assert result["hard_miss_count"] == 1


def test_structural_and_vad_pause_exclusions_and_unique_extra_turn() -> None:
    structural = score_policy_episode(
        [
            action(
                "s0",
                8000,
                9000,
                origin="vad",
                action_kind="structural_max_duration",
            )
        ],
        [],
        [],
        [(0, 16000, "A")],
        [],
        [],
        None,
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="negative_only",
    )
    retained_vad = score_policy_episode(
        [action("v0", 8000, 9000, origin="vad")],
        [],
        [],
        [(0, 7000, "A"), (9000, 16000, "A")],
        [(7000, 9000)],
        [],
        None,
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="negative_only",
    )
    detector_pause = score_policy_episode(
        [action("d0", 8000, 9000)],
        [],
        [],
        [(0, 7000, "A"), (9000, 16000, "A")],
        [(7000, 9000)],
        [],
        None,
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="negative_only",
    )
    assert structural["harm_or_structure_counts"]["harmful_active_split"] == 0
    assert structural["harm_or_structure_counts"]["structural_split"] == 1
    assert retained_vad["harm_or_structure_counts"]["same_speaker_pause_split"] == 0
    assert detector_pause["harm_or_structure_counts"]["same_speaker_pause_split"] == 1
    assert detector_pause["same_speaker_extra_turn_count"] == 1


def test_structural_action_is_segmentation_only() -> None:
    structural = action(
        "s0",
        8000,
        9000,
        origin="vad",
        action_kind="structural_max_duration",
    )
    baseline_structural = dict(structural)
    baseline_structural["action_id"] = "s0"
    result = score_policy_episode(
        [structural],
        [baseline_structural],
        [reference(8000)],
        [(0, 16000, "A")],
        [],
        [(7000, 9000)],
        [(7000, 9000)],
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="hard_only",
    )
    counts = result["harm_or_structure_counts"]
    assert result["match_count"] == 0
    assert result["b0_match_count"] == 0
    assert counts["duplicate_hard_boundary"] == 0
    assert counts["lexical_split"] == 0
    assert counts["lexical_not_observable"] == 0
    assert counts["overlap_hard_action"] == 0
    assert counts["structural_split"] == 1
    assert result["same_speaker_extra_turn_count"] == 0


def test_observable_empty_word_timing_is_not_missing() -> None:
    inputs = {
        "final_actions": [action("d0", 8000, 9000)],
        "b0_actions": [],
        "references": [],
        "singleton_intervals": [(0, 16000, "A")],
        "pause_intervals": [],
        "overlap_intervals": [],
        "unscored_intervals": [],
        "scored_start": 0,
        "scored_end": 16000,
        "episode_tag": "negative_only",
    }
    observable = score_policy_episode(word_intervals=[], **inputs)
    missing = score_policy_episode(word_intervals=None, **inputs)
    assert observable["harm_or_structure_counts"]["lexical_split"] == 0
    assert observable["harm_or_structure_counts"]["lexical_not_observable"] == 0
    assert missing["harm_or_structure_counts"]["lexical_not_observable"] == 1


def test_unscored_action_only_counts_as_unscored() -> None:
    inputs = {
        "b0_actions": [],
        "references": [reference(8000)],
        "singleton_intervals": [(0, 7500, "A"), (9500, 15000, "B")],
        "pause_intervals": [(7500, 9500)],
        "overlap_intervals": [(7500, 9500)],
        "word_intervals": None,
        "unscored_intervals": [(7500, 9500)],
        "scored_start": 0,
        "scored_end": 15000,
        "episode_tag": "hard_only",
    }
    without_action = score_policy_episode([], **inputs)
    with_action = score_policy_episode([action("d0", 8000, 9000)], **inputs)
    assert (
        with_action["contamination_by_owner_threshold"]
        == without_action["contamination_by_owner_threshold"]
    )
    assert with_action["match_count"] == 0
    assert with_action["detector_created_hard_action_count"] == 0
    assert with_action["same_speaker_extra_turn_count"] == 0
    assert (
        with_action["short_active_fragment_counts"]
        == without_action["short_active_fragment_counts"]
    )
    counts = with_action["harm_or_structure_counts"]
    assert counts["unscored_action"] == 1
    assert sum(value for key, value in counts.items() if key != "unscored_action") == 0
    assert with_action["sampled_singleton_exposure_samples"] == 13000
    assert with_action["sampled_unscored_exposure_samples"] == 2000


def test_overlap_hard_action_suppression_counterfactual() -> None:
    result = score_policy_episode(
        [action("d0", 10000, 12000)],
        [],
        [],
        [(0, 8000, "A"), (12000, 16000, "B")],
        [],
        [(8000, 12000)],
        None,
        [],
        scored_start=0,
        scored_end=16000,
        episode_tag="overlap_present",
    )
    view = result["overlap_hard_action_counterfactual_by_owner_threshold"]["100"]
    assert view["actual_contaminated_samples"] == 0
    assert view["suppressed_contaminated_samples"] == 4000
    assert view["actual_minus_suppressed_samples"] == -4000


def test_overlapping_unscored_intervals_use_union_exposure() -> None:
    result = score_policy_episode(
        [],
        [],
        [],
        [(0, 15000, "A")],
        [],
        [],
        None,
        [(12000, 15000), (14000, 15000)],
        scored_start=0,
        scored_end=15000,
        episode_tag="negative_only",
    )
    assert result["sampled_unscored_exposure_samples"] == 3000
    assert result["sampled_singleton_exposure_samples"] == 12000
    assert result["segment_source_duration_samples"] == [12000]


def test_jitter_precedes_unscored_exclusion() -> None:
    raw = [(0, 2000, "A"), (2000, 4000, "B")]
    jittered = jittered_scorable_intervals(raw, [(800, 1200)], 320)
    assert jittered == [
        (0, 800, "A"),
        (1200, 2320, "A"),
        (2320, 3680, "B"),
    ]


def test_sparse_matcher_matches_dense_objective_with_fixed_pairs() -> None:
    def dense_pairs(
        actions: list[Action],
        references: list[ReferenceAction],
        fixed: dict[str, str] | None = None,
    ) -> tuple[tuple[str, str], ...]:
        refs = sorted(references, key=lambda row: (row.evidence_onset_sample, row.reference_id))
        ordered = sorted(
            actions,
            key=lambda row: (
                row.boundary_source_sample,
                row.observed_source_sample_at_emit,
                row.action_id,
            ),
        )
        required = dict(fixed or {})
        required_by_reference = {
            reference_id: action_id for action_id, reference_id in required.items()
        }
        empty = ((0, 0, 0, 0), ())

        def better(left: object, right: object) -> object:
            if left is None:
                return right
            if right is None:
                return left
            left_state = left
            right_state = right
            if left_state[0] != right_state[0]:
                return left_state if left_state[0] > right_state[0] else right_state
            return left_state if left_state[1] < right_state[1] else right_state

        dp = [[None] * (len(ordered) + 1) for _ in range(len(refs) + 1)]
        for column in range(len(ordered) + 1):
            dp[0][column] = empty
        for row in range(len(refs) + 1):
            dp[row][0] = empty
        for row in range(1, len(refs) + 1):
            for column in range(1, len(ordered) + 1):
                best = better(dp[row - 1][column], dp[row][column - 1])
                previous = dp[row - 1][column - 1]
                ref = refs[row - 1]
                candidate = ordered[column - 1]
                compatible = required.get(candidate.action_id) in (
                    None,
                    ref.reference_id,
                ) and required_by_reference.get(ref.reference_id) in (
                    None,
                    candidate.action_id,
                )
                if compatible and action_eligible(candidate, ref, 0, 50000):
                    weight = _match_weight(ref, candidate)
                    state = (
                        tuple(previous[0][index] + weight[index] for index in range(4)),
                        tuple(sorted(previous[1] + ((ref.reference_id, candidate.action_id),))),
                    )
                    best = better(best, state)
                dp[row][column] = best
        return dp[-1][-1][1]

    for seed in range(100):
        generator = random.Random(seed)
        references = []
        for index in range(8):
            target = 3000 + index * 4500 + generator.randrange(-500, 501)
            references.append(
                ReferenceAction(
                    reference_id=f"r{index}",
                    audio_epoch=1,
                    source_session_id="s",
                    action_kind="hard_boundary",
                    target_sample=target,
                    acceptable_interval=(target - 2500, target + 2500),
                    evidence_onset_sample=target,
                    scorable=True,
                    primary_case=True,
                    episode_pool_tag="hard_only",
                )
            )
        actions = []
        for index in range(12):
            boundary = 1000 + index * 3200 + generator.randrange(-500, 501)
            actions.append(
                Action(
                    f"a{index}",
                    boundary,
                    boundary + generator.randrange(0, 8000),
                    "hard",
                    generator.choice(("b0", "detector")),
                    "s",
                    1,
                )
            )
        expected = dense_pairs(actions, references)
        actual = tuple(
            sorted(
                (row.reference_id, row.action_id)
                for row in match_episode(actions, references, 0, 50000)[0]
            )
        )
        assert actual == expected
        fixed_reference, fixed_action = generator.choice(expected)
        fixed = {fixed_action: fixed_reference}
        fixed_expected = dense_pairs(actions, references, fixed)
        fixed_actual = tuple(
            sorted(
                (row.reference_id, row.action_id)
                for row in match_episode(
                    actions,
                    references,
                    0,
                    50000,
                    fixed_pairs=fixed,
                )[0]
            )
        )
        assert fixed_actual == fixed_expected
