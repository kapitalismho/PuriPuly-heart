"""turn_episode_v1 frozen tables and scientific contract functions.

Phase 0 deliverable (approved by the Phase 0 pre-execution review). Implements the frozen
tables (PRD Sections 9.1, 9.2.6-9.2.8, 11.3, 12.1, 13.2, 14.1) and pure invariant functions
for the Section 28 scientific contract tests that are implementable without audio, model
execution, clustering replay, or fusion replay (bundle Section 11.2).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from experiments.speaker_turn_boundary.turn_episode.schemas import (
    FinalAction,
    LogicalBoundaryCluster,
    ProposalEvent,
    ReferenceAction,
)

CANONICAL_SAMPLE_RATE_HZ = 16000
CANONICAL_SAMPLES_PER_MS = CANONICAL_SAMPLE_RATE_HZ // 1000

# --- Frozen policy tables (PRD Sections 9.1, 9.2.6, 11.3, 12.1, 13.2, 14.1) ---

LS_KIND_PRIORITY: tuple[str, ...] = (
    "overlap_onset",
    "dominant_replacement",
    "new_track_onset",
    "track_instability",
)
HARD_FINAL_ACTION_KINDS: frozenset[str] = frozenset(
    {"retain_vad", "accelerate_or_replace_vad", "add_hard_boundary"}
)
LOCALIZATION_TOLERANCE_MS_PRIMARY = 500
LOCALIZATION_TOLERANCE_MS_VIEW = 250
AVAILABILITY_DEADLINES_MS: tuple[int, ...] = (250, 500, 1000, 1500, 2000)
TURN_OWNER_THRESHOLD_MS = 100
TURN_OWNER_SENSITIVITY_MS: tuple[int, ...] = (50, 200)
HARMFUL_ACTIVE_SPLIT_GUARD_MS = 200
HARMFUL_ACTIVE_SPLIT_SENSITIVITY_MS: tuple[int, ...] = (100, 300)
CLUSTER_DEBOUNCE_MS: tuple[int, ...] = (0, 100, 250)
CLUSTER_BOUNDARY_RADIUS_MS: tuple[int, ...] = (250, 500)
REFRACTORY_MS: tuple[int, ...] = (0, 250, 500)
VAD_ASSOCIATION_RADIUS_MS: tuple[int, ...] = (250, 500)
SAME_SILENCE_INTERVAL_ASSOCIATION: tuple[bool, ...] = (False, True)

NATURAL_EXPOSURE_POOL_ID = "natural_exposure_validation"

_HARD_BOUNDARY_KINDS = frozenset({"hard_boundary"})


def ms_to_samples(ms: int) -> int:
    return ms * CANONICAL_SAMPLES_PER_MS


def samples_to_ms(samples: int) -> float:
    return samples / CANONICAL_SAMPLES_PER_MS


# --- Invariant 1: causal observation frontier ---


def proposal_causal_valid(proposal: ProposalEvent) -> bool:
    return (
        proposal.observed_source_sample_at_emit >= proposal.boundary_source_sample
        and proposal.audio_epoch >= 0
    )


# --- Invariant 2: no cluster member after cluster close ---


def proposals_arrive_by_cluster_close(
    cluster: LogicalBoundaryCluster,
    proposals: Sequence[ProposalEvent],
) -> bool:
    by_id = {proposal.proposal_id: proposal for proposal in proposals}
    for proposal_id in cluster.member_proposal_ids:
        proposal = by_id.get(proposal_id)
        if proposal is None:
            return False
        if proposal.observed_source_sample_at_emit > cluster.close_frontier_sample:
            return False
        if proposal.audio_epoch != cluster.audio_epoch:
            return False
    return True


# --- Invariant 4 / Section 10: cluster produces at most one product action ---


def cluster_actions_exclusive(
    cluster: LogicalBoundaryCluster, actions: Sequence[FinalAction]
) -> bool:
    derived = [action for action in actions if action.cluster_id == cluster.cluster_id]
    return len(derived) <= 1


def hard_and_soft_from_one_cluster_never_both(
    cluster: LogicalBoundaryCluster, actions: Sequence[FinalAction]
) -> bool:
    derived = [action for action in actions if action.cluster_id == cluster.cluster_id]
    kinds = {action.action_kind for action in derived}
    hard = bool(kinds & HARD_FINAL_ACTION_KINDS)
    soft = "emit_soft_marker" in kinds
    return not (hard and soft)


# --- Invariant 5: no duplicated final hard boundary at one source sample ---


def no_duplicate_hard_boundary(actions: Sequence[FinalAction]) -> bool:
    seen: set[tuple[int, int]] = set()
    for action in actions:
        if action.action_kind not in HARD_FINAL_ACTION_KINDS:
            continue
        if action.boundary_source_sample is None:
            return False
        key = (action.audio_epoch, action.boundary_source_sample)
        if key in seen:
            return False
        seen.add(key)
    return True


# --- Invariant 6: matching is ordered and one-to-one ---


def matches_one_to_one(
    matched_reference_ids: Sequence[str],
    matched_action_ids: Sequence[str],
) -> bool:
    if len(matched_reference_ids) != len(matched_action_ids):
        return False
    return len(set(matched_reference_ids)) == len(matched_reference_ids) and len(
        set(matched_action_ids)
    ) == len(matched_action_ids)


# --- Invariants 7 and 9: gap interval-valued matching and pre-existing VAD validity ---


def inside_acceptable_interval(sample: int, reference: ReferenceAction) -> bool:
    start, end = reference.acceptable_interval
    return start <= sample <= end


def gap_boundary_matches_inside_interval(sample: int, reference: ReferenceAction) -> bool:
    if reference.action_kind not in _HARD_BOUNDARY_KINDS:
        return False
    return inside_acceptable_interval(sample, reference)


def pre_existing_gap_vad_boundary_valid(sample: int, reference: ReferenceAction) -> bool:
    return gap_boundary_matches_inside_interval(sample, reference)


# --- Invariant 8: no gap speaker-change credit before evidence onset ---


def gap_credit_requires_evidence_observation(
    observed_source_sample_at_emit: int,
    reference: ReferenceAction,
) -> bool:
    return observed_source_sample_at_emit >= reference.evidence_onset_sample


# --- Invariants 10/11: overlap cannot enter the clean/gap headline ---


def clean_gap_headline_eligible(reference: ReferenceAction) -> bool:
    return (
        reference.episode_pool_tag == "hard_only"
        and reference.action_kind in _HARD_BOUNDARY_KINDS
        and reference.scorable
        and reference.primary_case
    )


# --- Invariant 12: warm-up exclusion ---


def in_scored_region(sample: int, scored_start_sample: int, scored_end_sample: int) -> bool:
    return scored_start_sample <= sample < scored_end_sample


# --- Invariant 13: unscored intervals excluded from numerators ---


def unscored_excluded(reference: ReferenceAction) -> bool:
    return (not reference.scorable) or reference.action_kind == "unscored"


# --- Invariants 14/15/16: contamination algorithm and ownership threshold ---


def intervals_total_without_double_count(intervals: Sequence[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    total = 0
    merged_start: int | None = None
    merged_end: int | None = None
    for start, end in ordered:
        if merged_start is None:
            merged_start, merged_end = start, end
            continue
        if start <= merged_end:
            merged_end = max(merged_end, end)
            continue
        total += merged_end - merged_start
        merged_start, merged_end = start, end
    if merged_start is not None and merged_end is not None:
        total += merged_end - merged_start
    return total


def turn_owner_requires_threshold(
    singleton_run_ms: int, threshold_ms: int = TURN_OWNER_THRESHOLD_MS
) -> bool:
    return singleton_run_ms >= threshold_ms


def segment_contamination(
    singleton_runs_ms: Sequence[tuple[str, float, float]],
    owner_threshold_ms: int = TURN_OWNER_THRESHOLD_MS,
) -> tuple[str | None, float]:
    owner: str | None = None
    contaminated = False
    contaminated_ms = 0.0
    for speaker, start_ms, end_ms in singleton_runs_ms:
        run_ms = end_ms - start_ms
        qualifying = turn_owner_requires_threshold(run_ms, owner_threshold_ms)
        if owner is None:
            if qualifying:
                owner = speaker
            continue
        if contaminated:
            if qualifying:
                contaminated_ms += run_ms
            continue
        if speaker != owner and qualifying:
            contaminated_ms += run_ms
            contaminated = True
    return owner, contaminated_ms


def premature_split_receives_no_false_credit(
    successor_owner: str | None,
    original_owner: str,
) -> bool:
    """A premature split before the real handoff receives no contamination-reduction
    benefit: when the successor segment is still owned by the original speaker, later
    qualifying different-speaker speech inside it remains contamination (Section 13.3)."""
    return successor_owner == original_owner


# --- Invariant 17: benefit attribution and harm flags are orthogonal ---


def benefit_harm_orthogonal(attribution: str, harm_flags: Sequence[str]) -> bool:
    """Every benefit attribution may coexist with any harm flag; no mutual exclusion
    exists between the benefit axis and the harm axis (Section 12.3, invariant 17)."""
    return True


def harm_flags_preserved(action: FinalAction, observed_harm_conditions: Iterable[str]) -> bool:
    """A matched action must not have its harm flags stripped by benefit attribution."""
    return set(observed_harm_conditions) <= set(action.harm_or_structure_flags)


# --- Invariant 18: harmful active split requires the same singleton speaker both sides ---


def harmful_active_split_valid(
    before_speakers: frozenset[str],
    after_speakers: frozenset[str],
    before_ms: float,
    after_ms: float,
    guard_ms: int = HARMFUL_ACTIVE_SPLIT_GUARD_MS,
) -> bool:
    if len(before_speakers) != 1 or len(after_speakers) != 1:
        return False
    if before_speakers != after_speakers:
        return False
    return before_ms >= guard_ms and after_ms >= guard_ms


# --- Invariant 19: missing word timing is not_observable, not absence of harm ---


def lexical_split_requires_word_timing(
    has_word_timing: bool,
    inside_word_with_20ms_both_sides: bool,
) -> bool:
    if not has_word_timing:
        return False
    return inside_word_with_20ms_both_sides


# --- Invariant 20: same-speaker pause splits remain counted as extra turns ---


def same_speaker_extra_turn_count(harm_flags: Sequence[str]) -> int:
    return int("same_speaker_pause_split" in set(harm_flags))


# --- Invariant 24: cluster output kind and representative are semantically compatible ---


def cluster_kind_representative_compatible(
    cluster: LogicalBoundaryCluster,
    proposals: Sequence[ProposalEvent],
) -> bool:
    by_id = {proposal.proposal_id: proposal for proposal in proposals}
    representative = by_id.get(cluster.representative_proposal_id)
    if representative is None:
        return False
    if representative.proposal_kind != cluster.output_kind:
        return False
    if representative.audio_epoch != cluster.audio_epoch:
        return False
    if representative.proposal_id not in cluster.compatible_representative_subset:
        return False
    return True


# --- Invariant 25: max_confidence only across comparable confidence semantics ---


def semantics_comparable(confidence_semantics_ids: Iterable[str]) -> bool:
    unique = set(confidence_semantics_ids)
    return len(unique) == 1


def max_confidence_usable(
    cluster: LogicalBoundaryCluster,
    proposals: Sequence[ProposalEvent],
) -> bool:
    if cluster.representative_reason != "max_confidence":
        return True
    by_id = {proposal.proposal_id: proposal for proposal in proposals}
    subset_ids = [
        by_id[proposal_id].confidence_semantics_id
        for proposal_id in cluster.compatible_representative_subset
        if proposal_id in by_id
    ]
    if len(subset_ids) != len(cluster.compatible_representative_subset):
        return False
    return semantics_comparable(subset_ids)


# --- Invariant 26: reset-plus-warm-up gate ---


def reset_plus_warmup_allowed(state_equivalence_passed: bool) -> bool:
    return state_equivalence_passed


# --- Invariant 27: diagnostic_dev and frontier_dev are group-disjoint ---


def group_disjoint(diagnostic_group_ids: Iterable[str], frontier_group_ids: Iterable[str]) -> bool:
    return not (set(diagnostic_group_ids) & set(frontier_group_ids))


# --- Invariant 28: bootstrap resamples blocks, not transitions ---


def units_have_one_block(unit_block_map: dict[str, str]) -> bool:
    if not unit_block_map:
        return False
    return len(set(unit_block_map.values())) >= 1


def bootstrap_resamples_blocks(
    unit_block_map: dict[str, str],
    resampled_units: Sequence[str],
) -> bool:
    """True iff the resampled units are exactly the union of whole blocks."""
    if not resampled_units:
        return False
    unknown = [unit for unit in resampled_units if unit not in unit_block_map]
    if unknown:
        return False
    resampled_blocks = {unit_block_map[unit] for unit in resampled_units}
    union = {unit for unit, block in unit_block_map.items() if block in resampled_blocks}
    return set(resampled_units) == union


# --- Invariant 29: cross-split overlap fails closed ---


def cross_split_overlap_fails_closed(
    dev_group_hashes: Iterable[str],
    heldout_group_hashes: Iterable[str],
) -> bool:
    return not (set(dev_group_hashes) & set(heldout_group_hashes))


# --- Invariant 30: natural rates only from unbiased natural exposure ---


def natural_rate_source_allowed(pool_id: str, complete_source_coverage: bool = False) -> bool:
    return pool_id == NATURAL_EXPOSURE_POOL_ID or complete_source_coverage


# --- Invariant 31: held-out cannot open without a valid frozen self-hash ---


def frozen_contract_valid(contract: dict[str, object]) -> bool:
    required_keys = {
        "self_hash",
        "episode_manifest_hashes",
        "split_manifest_hashes",
        "source_and_annotation_hashes",
        "model_checkpoint_frontend_hashes",
        "proposal_profiles",
        "clustering_params",
        "action_mapping",
        "vad_fusion",
        "scoring_code_hashes",
        "panel_profile_ids",
        "bootstrap_seed",
        "block_graph_hash",
        "expected_session_and_episode_counts",
    }
    if not required_keys <= set(contract):
        return False
    if not isinstance(contract.get("self_hash"), str) or not contract["self_hash"]:
        return False
    expected_counts = contract.get("expected_session_and_episode_counts")
    return isinstance(expected_counts, dict) and bool(expected_counts)


# --- Invariant 32: incomplete held-out sessions cannot produce a decision ---


def heldout_complete(
    completed_sessions: Sequence[str],
    expected_sessions: Sequence[str],
) -> bool:
    return set(completed_sessions) == set(expected_sessions) and bool(expected_sessions)


# --- Invariant 34: stale epoch actions cannot mutate the current epoch ---


def stale_epoch_action_rejected(action_epoch: int, current_epoch: int) -> bool:
    return action_epoch < current_epoch


# --- Invariant 35: safe frontier monotonic and conservative within an epoch ---


def safe_frontier_trace_valid(
    trace: Sequence[tuple[int, int, int]],
) -> bool:
    """Trace entries are (audio_epoch, observed_source_sample, safe_boundary_frontier_sample)."""
    last_observed: dict[int, int] = {}
    last_safe: dict[int, int] = {}
    for audio_epoch, observed, safe in trace:
        if safe > observed:
            return False
        prev_observed = last_observed.get(audio_epoch)
        prev_safe = last_safe.get(audio_epoch)
        if prev_observed is not None and observed < prev_observed:
            return False
        if prev_safe is not None and safe < prev_safe:
            return False
        last_observed[audio_epoch] = observed
        last_safe[audio_epoch] = safe
    return True
