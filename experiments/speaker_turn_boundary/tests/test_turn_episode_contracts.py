"""Scientific contract tests for the turn_episode_v1 Phase 0 schemas and contracts.

Phase 0 exit gate: the action/scoring contract passes its invariant tests. Coverage follows
the approved bundle Section 11.2: synthetic contract tests for PRD Section 28 invariants
2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 27, 28, 29, 30,
31, 32, 34 and gate-only checks for 1, 26, 35.
"""

from __future__ import annotations

import pytest

from experiments.speaker_turn_boundary.turn_episode.contracts import (
    HARD_FINAL_ACTION_KINDS,
    NATURAL_EXPOSURE_POOL_ID,
    bootstrap_resamples_blocks,
    clean_gap_headline_eligible,
    cluster_actions_exclusive,
    cluster_kind_representative_compatible,
    cross_split_overlap_fails_closed,
    frozen_contract_valid,
    gap_boundary_matches_inside_interval,
    gap_credit_requires_evidence_observation,
    group_disjoint,
    hard_and_soft_from_one_cluster_never_both,
    harm_flags_preserved,
    harmful_active_split_valid,
    heldout_complete,
    in_scored_region,
    inside_acceptable_interval,
    intervals_total_without_double_count,
    lexical_split_requires_word_timing,
    matches_one_to_one,
    max_confidence_usable,
    natural_rate_source_allowed,
    no_duplicate_hard_boundary,
    pre_existing_gap_vad_boundary_valid,
    premature_split_receives_no_false_credit,
    proposal_causal_valid,
    proposals_arrive_by_cluster_close,
    reset_plus_warmup_allowed,
    safe_frontier_trace_valid,
    same_speaker_extra_turn_count,
    segment_contamination,
    semantics_comparable,
    stale_epoch_action_rejected,
    turn_owner_requires_threshold,
    units_have_one_block,
    unscored_excluded,
)
from experiments.speaker_turn_boundary.turn_episode.schemas import (
    ACTIONIZATION_SCHEMA_VERSION,
    PROPOSAL_GENERATION_SCHEMA_VERSION,
    FinalAction,
    LogicalBoundaryCluster,
    ProposalEvent,
    ReferenceAction,
    TurnEpisodeSchemaError,
)


def _proposal(
    proposal_id: str,
    *,
    kind: str = "dominant_replacement",
    boundary: int = 16000,
    observed: int = 17600,
    epoch: int = 0,
    semantics: str = "posterior_activation",
) -> ProposalEvent:
    return ProposalEvent(
        proposal_id=proposal_id,
        family="ls_eend",
        checkpoint="ckpt",
        profile_id="profile",
        audio_epoch=epoch,
        source_session_id="session",
        proposal_kind=kind,
        boundary_source_sample=boundary,
        observed_source_sample_at_emit=observed,
        emitted_monotonic_ns=0,
        confidence=0.8,
        confidence_semantics_id=semantics,
        state_provenance="episode_reset+warmup",
        debug_evidence={},
    )


def _reference(
    reference_id: str,
    *,
    kind: str = "hard_boundary",
    target: int = 16000,
    interval: tuple[int, int] | None = None,
    pool: str = "hard_only",
    scorable: bool = True,
    primary: bool = True,
) -> ReferenceAction:
    if interval is None:
        interval = (target - 8000, target + 8000)
    return ReferenceAction(
        reference_id=reference_id,
        audio_epoch=0,
        source_session_id="session",
        action_kind=kind,
        target_sample=target,
        acceptable_interval=interval,
        evidence_onset_sample=target,
        scorable=scorable,
        primary_case=primary,
        episode_pool_tag=pool,
    )


def _action(
    action_id: str,
    *,
    kind: str = "add_hard_boundary",
    boundary: int | None = 16000,
    availability: int = 17600,
    epoch: int = 0,
    cluster_id: str | None = None,
    matched_reference_id: str | None = None,
    flags: tuple[str, ...] = (),
) -> FinalAction:
    if boundary is not None and availability < boundary:
        availability = boundary
    return FinalAction(
        action_id=action_id,
        audio_epoch=epoch,
        source_session_id="session",
        action_kind=kind,
        boundary_source_sample=boundary,
        observed_source_sample_at_emit=availability,
        emitted_monotonic_ns=0,
        availability_source_sample=availability,
        cluster_id=cluster_id,
        matched_reference_id=matched_reference_id,
        harm_or_structure_flags=flags,
    )


def _cluster(
    cluster_id: str,
    members: tuple[str, ...],
    *,
    output_kind: str = "dominant_replacement",
    representative: str | None = None,
    subset: tuple[str, ...] | None = None,
    reason: str = "first",
    close: int = 17600,
) -> LogicalBoundaryCluster:
    if representative is None:
        representative = members[0]
    if subset is None:
        subset = (representative,)
    return LogicalBoundaryCluster(
        cluster_id=cluster_id,
        audio_epoch=0,
        source_session_id="session",
        member_proposal_ids=members,
        output_kind=output_kind,
        compatible_representative_subset=subset,
        representative_proposal_id=representative,
        representative_reason=reason,
        confidence_semantics_id="posterior_activation",
        suppression_reason="none",
        open_frontier_sample=16000,
        close_frontier_sample=close,
        availability_source_sample=close,
        boundary_spread_samples=0,
        confidence_distribution=(0.8,),
    )


# --- Invariant 1: causal observation frontier ---


def test_proposal_causal_valid_rejects_future_boundary() -> None:
    assert proposal_causal_valid(_proposal("p1", boundary=16000, observed=17600))
    with pytest.raises(TurnEpisodeSchemaError):
        _proposal("p2", boundary=17600, observed=16000)


# --- Invariant 2: no cluster member after cluster close ---


def test_no_cluster_member_arrives_after_close() -> None:
    proposals = [
        _proposal("p1", observed=17000),
        _proposal("p2", observed=17500),
        _proposal("p3", observed=18000),
    ]
    cluster = _cluster("c1", ("p1", "p2"), close=17600)
    assert proposals_arrive_by_cluster_close(cluster, proposals)
    late = _cluster("c2", ("p1", "p3"), close=17600)
    assert not proposals_arrive_by_cluster_close(late, proposals)


# --- Invariants 4 and Section 10: cluster output exclusivity ---


def test_cluster_produces_at_most_one_product_action() -> None:
    cluster = _cluster("c1", ("p1",))
    assert cluster_actions_exclusive(cluster, [_action("a1", cluster_id="c1")])
    assert not cluster_actions_exclusive(
        cluster, [_action("a1", cluster_id="c1"), _action("a2", cluster_id="c1")]
    )


def test_hard_and_soft_from_one_cluster_never_both() -> None:
    cluster = _cluster("c1", ("p1",))
    hard = _action("a1", kind="add_hard_boundary", cluster_id="c1")
    soft = _action("a2", kind="emit_soft_marker", cluster_id="c1")
    assert hard_and_soft_from_one_cluster_never_both(cluster, [hard])
    assert hard_and_soft_from_one_cluster_never_both(cluster, [soft])
    assert not hard_and_soft_from_one_cluster_never_both(cluster, [hard, soft])


# --- Invariant 5: no duplicated final hard boundary at one source sample ---


def test_no_duplicate_hard_boundary_at_one_sample() -> None:
    assert no_duplicate_hard_boundary(
        [_action("a1", boundary=16000), _action("a2", boundary=20000)]
    )
    assert not no_duplicate_hard_boundary(
        [_action("a1", boundary=16000), _action("a2", boundary=16000)]
    )
    soft_and_hard_at_same_sample_ok = [
        _action("a1", kind="emit_soft_marker", boundary=None),
        _action("a2", boundary=16000),
    ]
    assert no_duplicate_hard_boundary(soft_and_hard_at_same_sample_ok)


# --- Invariant 6: matching is ordered and one-to-one ---


def test_matching_one_to_one() -> None:
    assert matches_one_to_one(["r1", "r2"], ["a1", "a2"])
    assert not matches_one_to_one(["r1", "r1"], ["a1", "a2"])
    assert not matches_one_to_one(["r1", "r2"], ["a1", "a1"])


# --- Invariants 7 and 9: gap interval-valued matching; pre-existing VAD validity ---


def test_gap_boundary_anywhere_inside_silence_matches() -> None:
    gap = _reference("r-gap", target=20000, interval=(17000, 20000))
    for sample in (17000, 18000, 19500, 20000):
        assert gap_boundary_matches_inside_interval(sample, gap)
    assert not gap_boundary_matches_inside_interval(16999, gap)
    assert not gap_boundary_matches_inside_interval(20001, gap)
    assert inside_acceptable_interval(18000, gap)


def test_pre_existing_vad_gap_boundary_remains_valid() -> None:
    gap = _reference("r-gap", target=20000, interval=(17000, 20000))
    assert pre_existing_gap_vad_boundary_valid(18000, gap)
    assert not pre_existing_gap_vad_boundary_valid(25000, gap)


# --- Invariant 8: no gap speaker-change credit before evidence onset ---


def test_no_gap_credit_before_b_onset() -> None:
    gap = _reference("r-gap", target=20000, interval=(17000, 20000))
    assert not gap_credit_requires_evidence_observation(16000, gap)
    assert gap_credit_requires_evidence_observation(20000, gap)
    assert gap_credit_requires_evidence_observation(21000, gap)


# --- Invariants 10/11: overlap cannot enter the clean/gap headline ---


def test_overlap_episodes_and_soft_references_excluded_from_headline() -> None:
    hard_clean = _reference("r1")
    assert clean_gap_headline_eligible(hard_clean)
    overlap_pool = _reference("r2", pool="overlap_present")
    assert not clean_gap_headline_eligible(overlap_pool)
    soft = _reference("r3", kind="soft_overlap_marker")
    assert not clean_gap_headline_eligible(soft)
    unscored = _reference("r4", scorable=False)
    assert not clean_gap_headline_eligible(unscored)
    non_primary = _reference("r5", primary=False)
    assert not clean_gap_headline_eligible(non_primary)


# --- Invariant 12: warm-up exclusion ---


def test_warmup_actions_excluded_from_scored_counts() -> None:
    scored_start, scored_end = 16000, 320000
    assert in_scored_region(32000, scored_start, scored_end)
    assert not in_scored_region(8000, scored_start, scored_end)


# --- Invariant 13: unscored intervals excluded from numerators ---


def test_unscored_intervals_excluded() -> None:
    assert unscored_excluded(_reference("r1", scorable=False))
    assert unscored_excluded(_reference("r2", kind="unscored"))
    assert not unscored_excluded(_reference("r3"))


# --- Invariants 14/15/16: contamination algorithm and ownership ---


def test_contamination_samples_never_double_counted() -> None:
    intervals = [(0, 100), (50, 150), (200, 300)]
    assert intervals_total_without_double_count(intervals) == 250
    assert sum(end - start for start, end in intervals) == 300


def test_turn_owner_requires_100ms_threshold() -> None:
    assert not turn_owner_requires_threshold(99)
    assert turn_owner_requires_threshold(100)
    assert turn_owner_requires_threshold(100, threshold_ms=50)
    assert not turn_owner_requires_threshold(99, threshold_ms=100)


def test_segment_contamination_charges_subsequent_different_speakers() -> None:
    runs = [
        ("A", 0.0, 50.0),
        ("A", 50.0, 400.0),
        ("B", 400.0, 700.0),
        ("A", 700.0, 1000.0),
    ]
    owner, contaminated_ms = segment_contamination(runs)
    assert owner == "A"
    assert contaminated_ms == pytest.approx(600.0)


def test_premature_split_receives_no_false_credit() -> None:
    assert premature_split_receives_no_false_credit("A", "A")
    assert not premature_split_receives_no_false_credit("B", "A")


# --- Invariant 17: benefit attribution and harm flags are orthogonal ---


def test_matched_action_can_also_be_harmful() -> None:
    matched_harmful = _action(
        "a1",
        matched_reference_id="r1",
        flags=("harmful_active_split", "lexical_split"),
    )
    assert matched_harmful.matched_reference_id == "r1"
    assert "harmful_active_split" in matched_harmful.harm_or_structure_flags
    assert harm_flags_preserved(matched_harmful, ["harmful_active_split", "lexical_split"])
    assert not harm_flags_preserved(
        matched_harmful, ["harmful_active_split", "duplicate_hard_boundary"]
    )
    unmatched_harmful = _action("a2", flags=("harmful_active_split",))
    assert harm_flags_preserved(unmatched_harmful, ["harmful_active_split"])


# --- Invariant 18: harmful active split requires same singleton speaker both sides ---


def test_harmful_active_split_requires_same_speaker_both_guarded_sides() -> None:
    assert harmful_active_split_valid(
        frozenset({"A"}), frozenset({"A"}), 300.0, 300.0, guard_ms=200
    )
    assert not harmful_active_split_valid(
        frozenset({"A"}), frozenset({"B"}), 300.0, 300.0, guard_ms=200
    )
    assert not harmful_active_split_valid(
        frozenset({"A", "B"}), frozenset({"A", "B"}), 300.0, 300.0, guard_ms=200
    )
    assert not harmful_active_split_valid(
        frozenset({"A"}), frozenset({"A"}), 150.0, 300.0, guard_ms=200
    )


# --- Invariant 19: missing word timing is not absence of lexical harm ---


def test_lexical_split_requires_word_timing() -> None:
    assert lexical_split_requires_word_timing(True, True)
    assert not lexical_split_requires_word_timing(True, False)
    assert not lexical_split_requires_word_timing(False, True)


# --- Invariant 20: same-speaker pause splits remain counted as extra turns ---


def test_pause_split_counted_as_extra_turn() -> None:
    assert same_speaker_extra_turn_count(("same_speaker_pause_split",)) == 1
    assert same_speaker_extra_turn_count(("harmful_active_split",)) == 0


# --- Invariant 24: cluster output kind and representative compatibility ---


def test_cluster_kind_representative_compatibility() -> None:
    proposals = [
        _proposal("p1", kind="dominant_replacement", observed=17000),
        _proposal("p2", kind="overlap_onset", observed=17500),
    ]
    compatible = _cluster(
        "c1", ("p1", "p2"), output_kind="dominant_replacement", representative="p1"
    )
    assert cluster_kind_representative_compatible(compatible, proposals)
    incompatible = _cluster("c1", ("p1", "p2"), output_kind="overlap_onset", representative="p1")
    assert not cluster_kind_representative_compatible(incompatible, proposals)


# --- Invariant 25: max_confidence only across comparable semantics ---


def test_max_confidence_restricted_to_comparable_semantics() -> None:
    same = _cluster(
        "c1",
        ("p1", "p2"),
        output_kind="dominant_replacement",
        representative="p1",
        subset=("p1", "p2"),
        reason="max_confidence",
    )
    proposals_same = [
        _proposal("p1", observed=17000),
        _proposal("p2", observed=17500),
    ]
    assert max_confidence_usable(same, proposals_same)
    proposals_mixed = [
        _proposal("p1", observed=17000, semantics="posterior_activation"),
        _proposal("p2", observed=17500, semantics="other_activation"),
    ]
    assert not max_confidence_usable(same, proposals_mixed)
    assert not semantics_comparable(["a", "b"])
    assert semantics_comparable(["a", "a"])


# --- Invariant 26: reset-plus-warm-up gate ---


def test_reset_plus_warmup_gate() -> None:
    assert not reset_plus_warmup_allowed(False)
    assert reset_plus_warmup_allowed(True)


# --- Invariant 27: diagnostic_dev and frontier_dev are group-disjoint ---


def test_diagnostic_frontier_group_disjointness() -> None:
    assert group_disjoint(["g1", "g2"], ["g3", "g4"])
    assert not group_disjoint(["g1", "g2"], ["g2", "g4"])


# --- Invariant 28: bootstrap resamples blocks, not transitions ---


def test_bootstrap_resamples_whole_blocks() -> None:
    unit_block = {"e1": "b1", "e2": "b1", "e3": "b2", "e4": "b2"}
    assert units_have_one_block(unit_block)
    assert bootstrap_resamples_blocks(unit_block, ["e1", "e2", "e3", "e4"])
    assert not bootstrap_resamples_blocks(unit_block, ["e1", "e3"])
    assert not bootstrap_resamples_blocks(unit_block, ["e1", "e5"])


# --- Invariant 29: cross-split overlap fails closed ---


def test_cross_split_overlap_fails_closed() -> None:
    assert cross_split_overlap_fails_closed(["h1", "h2"], ["h3"])
    assert not cross_split_overlap_fails_closed(["h1", "h2"], ["h2"])


# --- Invariant 30: natural rates only from unbiased natural exposure ---


def test_natural_rate_source_gate() -> None:
    assert natural_rate_source_allowed(NATURAL_EXPOSURE_POOL_ID)
    assert natural_rate_source_allowed("anything", complete_source_coverage=True)
    assert not natural_rate_source_allowed("diagnostic_dev")
    assert not natural_rate_source_allowed("frontier_dev")


# --- Invariant 31: held-out cannot open without a valid frozen self-hash ---


def test_frozen_contract_self_hash_gate() -> None:
    valid = {
        "self_hash": "abc",
        "episode_manifest_hashes": {},
        "split_manifest_hashes": {},
        "source_and_annotation_hashes": {},
        "model_checkpoint_frontend_hashes": {},
        "proposal_profiles": {},
        "clustering_params": {},
        "action_mapping": {},
        "vad_fusion": {},
        "scoring_code_hashes": {},
        "panel_profile_ids": [],
        "bootstrap_seed": 1,
        "block_graph_hash": "g",
        "expected_session_and_episode_counts": {"sessions": 8, "episodes": 100},
    }
    assert frozen_contract_valid(valid)
    missing_self_hash = dict(valid)
    missing_self_hash["self_hash"] = ""
    assert not frozen_contract_valid(missing_self_hash)
    missing_key = {k: v for k, v in valid.items() if k != "panel_profile_ids"}
    assert not frozen_contract_valid(missing_key)


# --- Invariant 32: incomplete held-out sessions cannot produce a decision ---


def test_incomplete_heldout_cannot_decide() -> None:
    assert heldout_complete(["s1", "s2"], ["s1", "s2"])
    assert not heldout_complete(["s1"], ["s1", "s2"])
    assert not heldout_complete([], ["s1"])


# --- Invariant 34: stale epoch actions rejected ---


def test_stale_epoch_action_rejected() -> None:
    assert stale_epoch_action_rejected(0, 1)
    assert not stale_epoch_action_rejected(1, 1)
    assert not stale_epoch_action_rejected(2, 1)


# --- Invariant 35: safe frontier monotonic and conservative within an epoch ---


def test_safe_frontier_trace_monotonic_and_conservative() -> None:
    valid = [
        (0, 512, 0),
        (0, 1024, 512),
        (0, 1536, 1024),
        (1, 512, 0),
        (1, 1024, 512),
    ]
    assert safe_frontier_trace_valid(valid)
    future_safe = [(0, 512, 1024)]
    assert not safe_frontier_trace_valid(future_safe)
    non_monotonic = [(0, 512, 256), (0, 1024, 128)]
    assert not safe_frontier_trace_valid(non_monotonic)
    non_monotonic_observed = [(0, 1024, 0), (0, 512, 0)]
    assert not safe_frontier_trace_valid(non_monotonic_observed)


# --- Schema-version separation and hard action kinds ---


def test_proposal_and_actionization_schema_versions_are_separate() -> None:
    assert PROPOSAL_GENERATION_SCHEMA_VERSION != ACTIONIZATION_SCHEMA_VERSION
    assert HARD_FINAL_ACTION_KINDS == frozenset(
        {"retain_vad", "accelerate_or_replace_vad", "add_hard_boundary"}
    )


def test_soft_marker_requires_no_boundary() -> None:
    action = _action("a1", kind="emit_soft_marker", boundary=None)
    assert action.boundary_source_sample is None
    with pytest.raises(TurnEpisodeSchemaError):
        _action("a2", kind="add_hard_boundary", boundary=None)


def test_reference_requires_evidence_onset_equal_target_for_hard_boundary() -> None:
    with pytest.raises(TurnEpisodeSchemaError):
        ReferenceAction(
            reference_id="r1",
            audio_epoch=0,
            source_session_id="s",
            action_kind="hard_boundary",
            target_sample=16000,
            acceptable_interval=(8000, 16000),
            evidence_onset_sample=8000,
            scorable=True,
            primary_case=True,
            episode_pool_tag="hard_only",
        )
