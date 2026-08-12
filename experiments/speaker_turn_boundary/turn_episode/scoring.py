"""Phase 2 contamination/harm scoring fixtures per the approved bundle rev 7-8.

Implements the full deterministic matcher (PRD Section 12.1-12.2): context-aware
eligibility (session/epoch/scored region), gap tolerance closure applied exactly
once, B-onset evidence gating, pre-existing VAD gap validity, availability
deadlines with views, and a maximum-cardinality augmenting-path matching with a
lexicographic objective. Implements turn-owner thresholds and the contamination
algorithm (Section 13), clean/gap headline masking (hard_only only) and overlap
exclusion (Section 13.4-13.5), harm flags incl. lexical splits with word-timing
observability (Section 14), known-answer fixtures (invariants 6-20), and the B0
end-to-end baseline smoke over the 20 opened sessions (baseline only).
"""

from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .schemas import ReferenceAction

SAMPLES_PER_MS = 16
LOCALIZATION_TOLERANCE_MS = 500
TURN_OWNER_MS = 100
MIXED_TURN_MS = 250
DEADLINES_MS = (250, 500, 1000, 1500, 2000)
MATCH_DEADLINE_MS = 2000


class ScoringError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Action:
    action_id: str
    boundary_source_sample: int
    observed_source_sample_at_emit: int
    kind: str  # hard | soft
    owner: str  # b0 | detector
    session_id: str = ""
    audio_epoch: int = 0

    def __post_init__(self) -> None:
        if self.boundary_source_sample > self.observed_source_sample_at_emit:
            raise ScoringError(f"action {self.action_id}: boundary beyond observation frontier")


@dataclass(frozen=True, slots=True)
class Match:
    reference_id: str
    action_id: str
    benefit_attribution: str
    availability_delay_ms: int
    localization_error_ms: int
    pre_existing: bool = False


def is_gap_reference(reference: ReferenceAction) -> bool:
    return reference.action_kind == "hard_boundary" and reference.reference_id.endswith(":gap")


def gap_eligibility(reference: ReferenceAction) -> tuple[int, int]:
    start, end = reference.acceptable_interval
    return (
        max(0, start - LOCALIZATION_TOLERANCE_MS * SAMPLES_PER_MS),
        end + LOCALIZATION_TOLERANCE_MS * SAMPLES_PER_MS,
    )


def in_eligibility_window(boundary: int, reference: ReferenceAction) -> bool:
    if reference.action_kind not in ("hard_boundary", "soft_overlap_marker"):
        return False
    if reference.action_kind == "hard_boundary" and is_gap_reference(reference):
        start, end = gap_eligibility(reference)
        return start <= boundary <= end
    start, end = reference.acceptable_interval
    return start <= boundary <= end


def localization_error_ms(boundary: int, reference: ReferenceAction) -> int:
    start, end = reference.acceptable_interval
    if start <= boundary <= end:
        return 0
    if boundary < start:
        return round((start - boundary) / SAMPLES_PER_MS)
    return round((boundary - end) / SAMPLES_PER_MS)


def action_eligible(
    action: Action,
    reference: ReferenceAction,
    scored_start: int,
    processed_scored_end: int,
) -> bool:
    # Section 12.1.1: source session and epoch must agree.
    if (
        action.session_id != reference.source_session_id
        or action.audio_epoch != reference.audio_epoch
    ):
        return False
    # Warm-up actions are rejected before matching (invariant 12): an action must
    # lie inside the processed scored region.
    if not (scored_start <= action.boundary_source_sample < processed_scored_end):
        return False
    if action.kind == "hard" and reference.action_kind != "hard_boundary":
        return False
    if action.kind == "soft" and reference.action_kind != "soft_overlap_marker":
        return False
    if reference.action_kind not in ("hard_boundary", "soft_overlap_marker"):
        return False
    if not in_eligibility_window(action.boundary_source_sample, reference):
        return False
    # Section 12.1.4: detector-derived evidence was not available before the
    # detector-evidence onset (B onset) — applies to hard and soft detector
    # actions; pre-existing VAD gap boundaries remain valid (invariant 9).
    if action.owner == "detector":
        if action.observed_source_sample_at_emit < reference.evidence_onset_sample:
            return False
    delay = action.observed_source_sample_at_emit - reference.evidence_onset_sample
    if delay > MATCH_DEADLINE_MS * SAMPLES_PER_MS:
        return False
    return True


def _cost_tuple(
    action: Action,
    reference: ReferenceAction,
) -> tuple[int, int, int, str]:
    return (
        0 if action.owner == "b0" else 1,
        action.observed_source_sample_at_emit - reference.evidence_onset_sample,
        localization_error_ms(action.boundary_source_sample, reference),
        action.action_id,
    )


def benefit_attribution(
    action: Action,
    reference: ReferenceAction,
    delay_ms: int,
    all_actions: list[Action],
    baseline_action: Action | None = None,
) -> str:
    """Section 12.3 benefit attribution for one matched reference.

    - ``correct_soft_marker``: any soft-overlap marker match;
    - ``retained_b0_success``: B0-owned hard match;
    - ``late_target_action``: detector hard match only usable in the last
      deadline window (delay in (1500, 2000] ms);
    - ``accelerated_b0_success``: detector hard match whose availability is
      earlier than an eligible B0 action for the same reference (Section 12.2);
    - ``recovered_b0_hard_miss``: detector hard match with no earlier eligible
      B0 action.
    """
    if reference.action_kind == "soft_overlap_marker":
        return "correct_soft_marker"
    if action.owner == "b0":
        return "retained_b0_success"
    if delay_ms > 1500:
        return "late_target_action"
    if (
        baseline_action is not None
        and action.observed_source_sample_at_emit < baseline_action.observed_source_sample_at_emit
    ):
        return "accelerated_b0_success"
    if baseline_action is not None:
        return "none"
    return "recovered_b0_hard_miss"


def _match_weight(ref: ReferenceAction, act: Action) -> tuple[int, int, int, int]:
    """Per-pair objective weight (Section 12.2) for one ordered match.

    Returns ``(1, b0_retained, -availability_delay_ms, -localization_error_ms)``
    so the lexicographic maximum over a matching maximizes the number of matched
    references, then B0-retained hard successes, then lower total causal delay,
    then lower total localization distance. Pre-existing VAD gap boundaries
    report delay 0 (Section 15, invariant 9).
    """
    raw_delay = act.observed_source_sample_at_emit - ref.evidence_onset_sample
    pre_existing = act.owner == "b0" and raw_delay < 0
    delay_ms = (
        max(0, round(raw_delay / SAMPLES_PER_MS))
        if pre_existing
        else round(raw_delay / SAMPLES_PER_MS)
    )
    b0_retained = 1 if act.owner == "b0" and ref.action_kind == "hard_boundary" else 0
    return (
        1,
        b0_retained,
        -delay_ms,
        -localization_error_ms(act.boundary_source_sample, ref),
    )


def match_episode(
    actions: list[Action],
    references: list[ReferenceAction],
    scored_start: int,
    processed_scored_end: int,
    *,
    fixed_pairs: dict[str, str] | None = None,
    baseline_actions_by_reference: dict[str, Action] | None = None,
) -> tuple[list[Match], list[str], dict[str, int]]:
    """Deterministic ordered maximum-weight one-to-one matching.

    Objective (Section 12.2): maximize (1) the number of compatible matched
    references, then (2) B0-retained hard successes, (3) lower causal availability
    delay, (4) lower interval localization distance, (5) deterministic lexical ids.
    References are matched in evidence-onset order and actions in source order, so
    the resulting matching never crosses source order (Section 12.1.6: ordered
    one-to-one matching is preserved). Implemented as an exact sparse dynamic
    program over eligible reference/action edges.
    """
    refs = [
        r
        for r in sorted(references, key=lambda r: (r.evidence_onset_sample, r.reference_id))
        if r.scorable and r.action_kind in ("hard_boundary", "soft_overlap_marker")
    ]
    ordered_actions = sorted(
        actions,
        key=lambda a: (
            a.boundary_source_sample,
            a.observed_source_sample_at_emit,
            a.action_id,
        ),
    )
    n, m = len(refs), len(ordered_actions)
    fixed = dict(fixed_pairs or {})
    if len(set(fixed.values())) != len(fixed):
        raise ScoringError("fixed match references are not one-to-one")
    if not set(fixed).issubset({action.action_id for action in ordered_actions}):
        raise ScoringError("fixed match action is absent")
    if not set(fixed.values()).issubset({reference.reference_id for reference in refs}):
        raise ScoringError("fixed match reference is absent")

    reference_index = {reference.reference_id: index for index, reference in enumerate(refs)}
    action_index = {action.action_id: index for index, action in enumerate(ordered_actions)}
    if len(reference_index) != n or len(action_index) != m:
        raise ScoringError("matching IDs are not unique")

    def better(
        left: tuple[tuple[int, int, int, int], tuple[tuple[str, str], ...]] | None,
        right: tuple[tuple[int, int, int, int], tuple[tuple[str, str], ...]] | None,
    ) -> tuple[tuple[int, int, int, int], tuple[tuple[str, str], ...]] | None:
        if left is None:
            return right
        if right is None:
            return left
        if left[0] != right[0]:
            return left if left[0] > right[0] else right
        return left if left[1] < right[1] else right

    state_type = tuple[tuple[int, int, int, int], tuple[tuple[str, str], ...]]
    empty: state_type = ((0, 0, 0, 0), ())

    def combine(left: state_type, right: state_type) -> state_type:
        return (
            tuple(left[0][index] + right[0][index] for index in range(4)),
            tuple(sorted(left[1] + right[1])),
        )

    def solve_segment(
        segment_refs: list[ReferenceAction], segment_actions: list[Action]
    ) -> state_type:
        if not segment_refs or not segment_actions:
            return empty
        boundaries = [action.boundary_source_sample for action in segment_actions]
        tree: list[state_type | None] = [None] * (len(segment_actions) + 1)

        def query(count: int) -> state_type:
            best: state_type | None = empty
            index = count
            while index > 0:
                best = better(best, tree[index])
                index -= index & -index
            return best or empty

        def update(position: int, state: state_type) -> None:
            index = position + 1
            while index < len(tree):
                tree[index] = better(tree[index], state)
                index += index & -index

        for ref in segment_refs:
            if is_gap_reference(ref):
                eligibility_start, eligibility_end = gap_eligibility(ref)
            else:
                eligibility_start, eligibility_end = ref.acceptable_interval
            start = bisect.bisect_left(boundaries, eligibility_start)
            end = bisect.bisect_right(boundaries, eligibility_end)
            candidates: list[tuple[int, state_type]] = []
            for position in range(start, end):
                act = segment_actions[position]
                if not action_eligible(act, ref, scored_start, processed_scored_end):
                    continue
                prev = query(position)
                weight = _match_weight(ref, act)
                candidates.append(
                    (
                        position,
                        (
                            tuple(prev[0][index] + weight[index] for index in range(4)),
                            tuple(sorted(prev[1] + ((ref.reference_id, act.action_id),))),
                        ),
                    )
                )
            for position, state in candidates:
                update(position, state)
        return query(len(segment_actions))

    fixed_positions = sorted(
        (
            reference_index[reference_id],
            action_index[action_id],
            reference_id,
            action_id,
        )
        for action_id, reference_id in fixed.items()
    )
    if any(left[1] >= right[1] for left, right in zip(fixed_positions, fixed_positions[1:])):
        raise ScoringError("fixed B0 matches do not preserve source order")
    for ref_position, act_position, _, _ in fixed_positions:
        if not action_eligible(
            ordered_actions[act_position],
            refs[ref_position],
            scored_start,
            processed_scored_end,
        ):
            raise ScoringError("fixed B0 match is ineligible")

    best_state = empty
    prior_ref = -1
    prior_action = -1
    for ref_position, act_position, reference_id, action_id in fixed_positions:
        best_state = combine(
            best_state,
            solve_segment(
                refs[prior_ref + 1 : ref_position],
                ordered_actions[prior_action + 1 : act_position],
            ),
        )
        best_state = combine(
            best_state,
            (
                _match_weight(refs[ref_position], ordered_actions[act_position]),
                ((reference_id, action_id),),
            ),
        )
        prior_ref = ref_position
        prior_action = act_position
    best_state = combine(
        best_state,
        solve_segment(refs[prior_ref + 1 :], ordered_actions[prior_action + 1 :]),
    )
    matched_id_pairs = set(best_state[1])
    expected_fixed = {(reference_id, action_id) for action_id, reference_id in fixed.items()}
    if not expected_fixed.issubset(matched_id_pairs):
        raise ScoringError("fixed B0 match cannot be preserved in ordered candidate matching")
    if not best_state[1]:
        matches: list[Match] = []
    else:
        reference_by_id = {reference.reference_id: reference for reference in refs}
        action_by_id = {action.action_id: action for action in ordered_actions}
        matched_pairs = [
            (reference_by_id[reference_id], action_by_id[action_id])
            for reference_id, action_id in best_state[1]
        ]
        matches = []
        for ref, action in matched_pairs:
            raw_delay = action.observed_source_sample_at_emit - ref.evidence_onset_sample
            # Section 12.1/15: a VAD-owned boundary available before B onset is a
            # valid pre-existing product category, never a negative availability
            # delay; it is reported as pre-existing with delay 0.
            pre_existing = action.owner == "b0" and raw_delay < 0
            delay_ms = (
                max(0, round(raw_delay / SAMPLES_PER_MS))
                if pre_existing
                else round(raw_delay / SAMPLES_PER_MS)
            )
            matches.append(
                Match(
                    reference_id=ref.reference_id,
                    action_id=action.action_id,
                    benefit_attribution=benefit_attribution(
                        action,
                        ref,
                        delay_ms,
                        actions,
                        (baseline_actions_by_reference or {}).get(ref.reference_id),
                    ),
                    availability_delay_ms=delay_ms,
                    localization_error_ms=localization_error_ms(action.boundary_source_sample, ref),
                    pre_existing=pre_existing,
                )
            )
    matches.sort(key=lambda m: m.reference_id)
    matched_refs = {m.reference_id for m in matches}
    hard_misses = [
        r.reference_id
        for r in refs
        if r.action_kind == "hard_boundary"
        and r.primary_case
        and r.reference_id not in matched_refs
    ]
    deadline_views = {
        str(deadline_ms): sum(1 for m in matches if m.availability_delay_ms <= deadline_ms)
        for deadline_ms in DEADLINES_MS
    }
    return matches, hard_misses, deadline_views


def logical_segments(
    boundaries: list[int], scored_start: int, scored_end: int
) -> list[tuple[int, int]]:
    points = sorted(set(boundaries))
    segments: list[tuple[int, int]] = []
    cursor = scored_start
    for point in points:
        if point < scored_start or point >= scored_end:
            continue
        if point > cursor:
            segments.append((cursor, point))
        cursor = point
    if cursor < scored_end:
        segments.append((cursor, scored_end))
    return segments


def segment_contamination_ms(
    segment: tuple[int, int],
    intervals: list[tuple[int, int, str]],
    owner_threshold_ms: int,
) -> dict[str, Any]:
    """Section 13.3 contamination algorithm for one segment."""
    qualifying: list[tuple[int, int, str]] = []
    for start, end, speaker in intervals:
        overlap_start = max(start, segment[0])
        overlap_end = min(end, segment[1])
        if overlap_end - overlap_start >= owner_threshold_ms * SAMPLES_PER_MS:
            qualifying.append((overlap_start, overlap_end, speaker))
    if not qualifying:
        return {"owner": None, "contamination_ms": 0, "excluded_subthreshold_ms": 0}
    owner = qualifying[0][2]
    contaminated = 0
    for start, end, speaker in qualifying:
        if speaker != owner:
            contaminated += end - start
    return {
        "owner": owner,
        "contamination_ms": round(contaminated / SAMPLES_PER_MS),
        "excluded_subthreshold_ms": 0,
    }


def harmful_active_split(
    boundary: int, intervals: list[tuple[int, int, str]], guard_ms: int
) -> bool:
    for start, end, speaker in intervals:
        if (
            start <= boundary - guard_ms * SAMPLES_PER_MS
            and boundary + guard_ms * SAMPLES_PER_MS <= end
        ):
            return True
    return False


def same_speaker_pause_split(boundary: int, pause_intervals: list[tuple[int, int]]) -> bool:
    for start, end in pause_intervals:
        if start <= boundary <= end:
            return True
    return False


def lexical_split(
    boundary: int,
    word_intervals: list[tuple[int, int]],
    unscored_intervals: list[tuple[int, int]] | None = None,
) -> bool | None:
    """True when the boundary lies inside a word with >= 20 ms on both sides;
    False when it lies in wordless (silence/uncovered) audio; None
    (not_observable) when the boundary falls in an unscored/ambiguous span whose
    timing is unknown, or when no word timing is available at all (invariant 19:
    missing word timing is never treated as absence of lexical harm)."""
    if unscored_intervals:
        for start, end in unscored_intervals:
            if start <= boundary < end:
                return None
    if not word_intervals:
        return None
    for start, end in word_intervals:
        if start <= boundary < end:
            return start <= boundary - 20 * SAMPLES_PER_MS and boundary + 20 * SAMPLES_PER_MS <= end
    return False


def score_episode(
    actions: list[Action],
    references: list[ReferenceAction],
    singleton_intervals: list[tuple[int, int, str]],
    pause_intervals: list[tuple[int, int]],
    word_intervals: list[tuple[int, int]] | None,
    scored_start: int,
    scored_end: int,
    unscored_intervals: list[tuple[int, int]] | None = None,
    overlap_intervals: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    matches, hard_misses, deadline_views = match_episode(
        actions, references, scored_start, scored_end
    )
    # Out-of-region actions are excluded from harm scoring and segment counts
    # (invariant 12/13: only scored-region actions enter any numerator).
    scored_actions = [a for a in actions if scored_start <= a.boundary_source_sample < scored_end]
    hard_boundaries = sorted(a.boundary_source_sample for a in scored_actions if a.kind == "hard")
    segments = logical_segments(hard_boundaries, scored_start, scored_end)
    contamination: dict[str, int] = {}
    for threshold in ("50ms", "100ms", "200ms"):
        total = 0
        for segment in segments:
            total += segment_contamination_ms(segment, singleton_intervals, int(threshold[:-2]))[
                "contamination_ms"
            ]
        contamination[threshold] = total
    harm_flags: dict[str, int] = {
        "harmful_active_split": 0,
        "lexical_split": 0,
        "same_speaker_pause_split": 0,
        "duplicate_hard_boundary": 0,
        "overlap_hard_action": 0,
        "lexical_not_observable": 0,
    }
    for action in scored_actions:
        if action.kind != "hard":
            continue
        if harmful_active_split(action.boundary_source_sample, singleton_intervals, 200):
            harm_flags["harmful_active_split"] += 1
        if same_speaker_pause_split(action.boundary_source_sample, pause_intervals):
            harm_flags["same_speaker_pause_split"] += 1
        if any(
            ref.action_kind == "soft_overlap_marker"
            and ref.acceptable_interval[0]
            <= action.boundary_source_sample
            <= ref.acceptable_interval[1]
            for ref in references
        ) or any(
            overlap[0] <= action.boundary_source_sample < overlap[1]
            for overlap in (overlap_intervals or [])
        ):
            harm_flags["overlap_hard_action"] += 1
        split = lexical_split(
            action.boundary_source_sample, word_intervals or [], unscored_intervals
        )
        if split is True:
            harm_flags["lexical_split"] += 1
        elif split is None:
            harm_flags["lexical_not_observable"] += 1
    boundary_counts: dict[int, int] = {}
    for boundary in hard_boundaries:
        boundary_counts[boundary] = boundary_counts.get(boundary, 0) + 1
    harm_flags["duplicate_hard_boundary"] = sum(
        1 for count in boundary_counts.values() if count > 1
    )
    return {
        "match_count": len(matches),
        "hard_misses": hard_misses,
        "deadline_views": deadline_views,
        "contamination_ms": contamination,
        "harm_flags": harm_flags,
        "segment_count": len(segments),
        "hard_action_count": len(hard_boundaries),
    }


def known_answer_fixtures() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def run(name: str, check: Callable[[], bool]) -> None:
        results.append({"fixture": name, "passed": check()})

    def ref(
        ref_id: str,
        kind: str,
        target: int | None,
        interval: tuple[int, int],
        evidence: int,
        primary: bool = True,
        scorable: bool = True,
        gap: bool = False,
    ) -> ReferenceAction:
        suffix = ":gap" if gap else ""
        return ReferenceAction(
            reference_id=f"s:e:{ref_id}{suffix}",
            audio_epoch=0,
            source_session_id="s",
            action_kind=kind,
            target_sample=target,
            acceptable_interval=interval,
            evidence_onset_sample=evidence,
            scorable=scorable,
            primary_case=primary,
            episode_pool_tag="hard_only",
        )

    def act(
        aid: str,
        boundary: int,
        observed: int,
        kind: str,
        owner: str,
        session_id: str = "s",
        audio_epoch: int = 0,
    ) -> Action:
        return Action(aid, boundary, observed, kind, owner, session_id, audio_epoch)

    # Invariant 7: gap interval matching accepts any boundary inside the annotated silence.
    def f7() -> bool:
        r = ref("gt0", "hard_boundary", 24000, (16000, 24000), 24000, gap=True)
        action = act("a1", 20000, 26000, "hard", "b0")
        return action_eligible(action, r, 0, 50000)

    run("inv7_gap_inside_interval", f7)

    # Invariant 8: a detector proposal before B onset receives no gap credit.
    def f8() -> bool:
        r = ref("gt0", "hard_boundary", 24000, (16000, 24000), 24000, gap=True)
        early = act("a1", 20000, 20000, "hard", "detector")
        return not action_eligible(early, r, 0, 50000)

    run("inv8_no_gap_credit_before_b_onset", f8)

    # Invariant 9: pre-existing VAD gap boundary is valid product separation.
    def f9() -> bool:
        r = ref("gt0", "hard_boundary", 24000, (16000, 24000), 24000, gap=True)
        vad = act("a1", 20000, 20000, "hard", "b0")
        return action_eligible(vad, r, 0, 50000)

    run("inv9_pre_existing_vad_gap_valid", f9)

    # P2-029: gap tolerance closure applied exactly once (edges at/just beyond).
    def f29() -> bool:
        r = ref("gt0", "hard_boundary", 24000, (16000, 24000), 24000, gap=True)
        at_low = act("a1", 16000 - LOCALIZATION_TOLERANCE_MS * SAMPLES_PER_MS, 30000, "hard", "b0")
        just_beyond = act(
            "a2",
            24000 + LOCALIZATION_TOLERANCE_MS * SAMPLES_PER_MS + 1,
            33000,
            "hard",
            "b0",
        )
        return action_eligible(at_low, r, 0, 50000) and not action_eligible(
            just_beyond, r, 0, 50000
        )

    run("p2_029_gap_tolerance_once", f29)

    # Exit-gate P2-SCORE-001: detector soft actions before evidence onset are rejected.
    def fsoft() -> bool:
        r = ref("gt0", "soft_overlap_marker", 24000, (20000, 24000), 24000, primary=False)
        early = act("a1", 23000, 23000, "soft", "detector")
        return not action_eligible(early, r, 0, 50000)

    run("p2_score001_soft_detector_before_onset_rejected", fsoft)

    # Exit-gate P2-SCORE-001: soft matches are labeled correct_soft_marker.
    def fsoft2() -> bool:
        r = ref("gt0", "soft_overlap_marker", 24000, (20000, 24000), 24000, primary=False)
        action = act("a1", 23000, 24500, "soft", "detector")
        matches, _, _ = match_episode([action], [r], 0, 50000)
        return len(matches) == 1 and matches[0].benefit_attribution == "correct_soft_marker"

    run("p2_score001_soft_match_correct_soft_marker", fsoft2)

    # Exit-gate P2-SCORE-001: pre-existing VAD gap match gets a pre-existing disposition
    # with delay 0, never a negative availability delay (Section 15).
    def fpre() -> bool:
        r = ref("gt0", "hard_boundary", 24000, (16000, 24000), 24000, gap=True)
        vad = act("a1", 20000, 20000, "hard", "b0")
        matches, _, _ = match_episode([vad], [r], 0, 50000)
        return (
            len(matches) == 1 and matches[0].pre_existing and matches[0].availability_delay_ms == 0
        )

    # Exit-gate P2-SCORE-001 (round 3): the matcher is globally lexicographic and
    # ordered. Two clean references and actions A/B (B0) plus D (detector) must
    # match r1->A, r2->B (two B0 successes preserved), never r1->D, r2->A, and the
    # matching must not cross source order (Section 12.1.6).
    def fordered() -> bool:
        r1 = ref("gt0", "hard_boundary", 12000, (8000, 12000), 12000)
        r2 = ref("gt1", "hard_boundary", 15000, (11000, 15000), 15000)
        a = act("A", 11000, 13000, "hard", "b0")
        b = act("B", 13000, 15000, "hard", "b0")
        d = act("D", 11500, 13000, "hard", "detector")
        matches, _, _ = match_episode([a, b, d], [r1, r2], 0, 50000)
        pairs = {m.reference_id.split(":")[-1]: m.action_id for m in matches}
        return (
            len(matches) == 2
            and pairs.get("gt0") == "A"
            and pairs.get("gt1") == "B"
            and sum(1 for m in matches if m.benefit_attribution == "retained_b0_success") == 2
        )

    run("p2_score001_ordered_globally_lexicographic", fordered)

    # Exit-gate P2-SCORE-001 (round 3): a detector action is never labeled
    # accelerated_b0_success against a B0 action that misses the matching deadline
    # (a deadline miss is recovered_b0_hard_miss, Sections 12.2-12.3).
    def facc() -> bool:
        r = ref("gt0", "hard_boundary", 24000, (16000, 24000), 24000, gap=True)
        late_b0 = act("b0late", 20000, 24000 + 3500 * SAMPLES_PER_MS, "hard", "b0")
        det = act("det", 22000, 24000 + 1000 * SAMPLES_PER_MS, "hard", "detector")
        matches, _, _ = match_episode([late_b0, det], [r], 0, 50000)
        return (
            len(matches) == 1
            and matches[0].action_id == "det"
            and matches[0].benefit_attribution == "recovered_b0_hard_miss"
        )

    run("p2_score001_no_acceleration_from_deadline_miss_b0", facc)

    run("p2_score001_pre_existing_vad_disposition", fpre)

    # Exit-gate P2-SCORE-001: source session/epoch context is enforced.
    def fctx() -> bool:
        r = ref("gt0", "hard_boundary", 24000, (20000, 24000), 24000)
        other_session = act("a1", 23000, 24000, "hard", "b0", session_id="other")
        other_epoch = act("a2", 23000, 24000, "hard", "b0", audio_epoch=1)
        return not action_eligible(other_session, r, 0, 50000) and not action_eligible(
            other_epoch, r, 0, 50000
        )

    run("p2_score001_session_epoch_context", fctx)

    # Exit-gate P2-SCORE-001: detector recovery vs acceleration taxonomy.
    def facc() -> bool:
        r1 = ref("gt0", "hard_boundary", 24000, (20000, 24000), 24000)
        r2 = ref("gt1", "hard_boundary", 26000, (23000, 26000), 26000)
        d1 = act("d1", 23000, 24000, "hard", "detector")
        b1 = act("b1", 23500, 26000, "hard", "b0")
        matches, _, _ = match_episode([d1, b1], [r1, r2], 0, 50000)
        attrs = {m.reference_id: m.benefit_attribution for m in matches}
        detector_only, _, _ = match_episode([d1], [r1], 0, 50000)
        return (
            len(matches) == 2
            and attrs.get("s:e:gt0") == "accelerated_b0_success"
            and len(detector_only) == 1
            and detector_only[0].benefit_attribution == "recovered_b0_hard_miss"
        )

    run("p2_score001_accelerated_vs_recovered", facc)

    # Exit-gate P2-SCORE-001: late target action attribution in the last deadline view.
    def flate() -> bool:
        r = ref("gt0", "hard_boundary", 24000, (20000, 24000), 24000)
        late = act("a1", 23000, 24000 + 1800 * SAMPLES_PER_MS, "hard", "detector")
        matches, _, views = match_episode([late], [r], 0, 50000)
        return len(matches) == 1 and matches[0].benefit_attribution == "late_target_action"

    run("p2_score001_late_target_action", flate)

    # Invariant 6: ordered one-to-one matching (max cardinality).
    def f6() -> bool:
        refs = [
            ref(
                f"gt{i}",
                "hard_boundary",
                10000 + i * 8000,
                (10000 + i * 8000 - 8000, 10000 + i * 8000),
                10000 + i * 8000,
            )
            for i in range(2)
        ]
        actions = [
            act("a1", 9000, 12000, "hard", "detector"),
            act("a2", 10000, 18000, "hard", "detector"),
        ]
        matches, misses, _ = match_episode(actions, refs, 0, 50000)
        return len(matches) == 2 and len({m.action_id for m in matches}) == 2

    run("inv6_one_to_one", f6)

    # Invariant 6b: augmenting path — a shared action is remapped for max cardinality.
    def f6b() -> bool:
        refs = [
            ref("gt0", "hard_boundary", 10000, (2000, 10000), 10000),
            ref("gt1", "hard_boundary", 18000, (10000, 18000), 18000),
        ]
        actions = [act("a1", 9000, 12000, "hard", "detector")]
        matches, _, _ = match_episode(actions, refs, 0, 50000)
        return len(matches) == 1

    run("inv6b_max_cardinality_shared_action", f6b)

    # Invariant 12: warm-up actions cannot enter scored counts (boundary < scored_start).
    def f12() -> bool:
        r = ref("gt0", "hard_boundary", 20000, (12000, 20000), 20000)
        warmup = act("a1", 5000, 9000, "hard", "b0")
        matches, misses, _ = match_episode([warmup], [r], 10000, 30000)
        return len(matches) == 0 and r.reference_id in misses

    run("inv12_warmup_action_excluded", f12)

    # Invariant 13: unscored references never enter benefit or harm numerators.
    def f13() -> bool:
        r = ref(
            "u0",
            "unscored",
            None,
            (12000, 20000),
            12000,
            primary=False,
            scorable=False,
        )
        action = act("a1", 15000, 18000, "hard", "b0")
        matches, _, _ = match_episode([action], [r], 0, 50000)
        return len(matches) == 0

    run("inv13_unscored_excluded", f13)

    # Invariant 15: premature split receives no false contamination-reduction credit.
    def f15() -> bool:
        intervals = [(0, 40000, "A"), (42000, 80000, "B")]
        segments = logical_segments([30000], 0, 80000)
        seg2 = segment_contamination_ms(segments[1], intervals, 100)
        return seg2["owner"] == "A" and seg2["contamination_ms"] > 0

    run("inv15_premature_split_no_credit", f15)

    # Invariant 14: contamination source samples never double-counted (A->B->C).
    def f14() -> bool:
        intervals = [(0, 16000, "A"), (17000, 33000, "B"), (34000, 50000, "C")]
        segments = logical_segments([16000], 0, 50000)
        seg = segment_contamination_ms(segments[1], intervals, 100)
        return seg["owner"] == "B" and seg["contamination_ms"] == round(
            (50000 - 34000) / SAMPLES_PER_MS
        )

    run("inv14_no_double_count_abc", f14)

    # Invariant 17: harm flags are independent of benefit matching.
    def f17() -> bool:
        intervals = [(0, 40000, "A")]
        return harmful_active_split(30000, intervals, 200)

    run("inv17_harm_independent_of_match", f17)

    # Invariant 18: harmful active split requires the same singleton speaker both sides.
    def f18() -> bool:
        intervals = [(0, 20000, "A"), (21000, 40000, "B")]
        return not harmful_active_split(20500, intervals, 200)

    run("inv18_same_speaker_both_sides", f18)

    # Invariant 19: missing word timing is not absence of lexical harm.
    def f19() -> bool:
        no_timing = lexical_split(30000, [])
        inside = lexical_split(30000, [(29000, 31000)])
        edge = lexical_split(30000, [(29000, 30100)])
        return no_timing is None and inside is True and edge is False

    run("inv19_lexical_split_semantics", f19)

    # Exit-gate P2-SCORE-002: a boundary in wordless audio (words present) is False,
    # a boundary in an unscored/ambiguous span is not_observable (None).
    def flex_region() -> bool:
        outside = lexical_split(30000, [(31000, 32000), (33000, 34000)])
        in_unscored = lexical_split(30000, [(29000, 31000)], unscored_intervals=[(29500, 31500)])
        return outside is False and in_unscored is None

    run("p2_score002_lexical_region_semantics", flex_region)

    # Invariant 20: same-speaker pause splits counted as extra turns.
    def f20() -> bool:
        return same_speaker_pause_split(30000, [(25000, 35000)])

    run("inv20_pause_split", f20)

    # Invariant 16: turn ownership requires 100 ms substantive singleton threshold.
    def f16() -> bool:
        intervals = [(0, 1000, "A"), (1100, 2000, "B")]
        segments = logical_segments([1000], 0, 2000)
        result = segment_contamination_ms(segments[1], intervals, 100)
        short = segment_contamination_ms(segments[1], intervals, 50)
        return result["owner"] is None and short["owner"] == "B"

    run("inv16_turn_owner_threshold", f16)

    # Invariant 11: overlap_present episodes excluded from the clean/gap headline.
    def f11() -> bool:
        overlap_refs = [
            ref("gt0", "soft_overlap_marker", 20000, (15000, 20000), 20000, primary=False)
        ]
        overlap_actions = [act("a1", 18000, 20000, "hard", "detector")]
        matches, misses, _ = match_episode(overlap_actions, overlap_refs, 0, 50000)
        return len(matches) == 0 and not misses

    run("inv11_overlap_not_clean_gap_benefit", f11)

    return results


def _load_word_intervals_ami(
    raw_words: list[Any], scored_start: int, processed_scored_end: int
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for word in raw_words:
        if word.start_time_s is None or word.end_time_s is None:
            continue
        start = int(round(word.start_time_s * 16000))
        end = int(round(word.end_time_s * 16000))
        if end <= start:
            continue
        overlap_start = max(start, scored_start)
        overlap_end = min(end, processed_scored_end)
        if overlap_end > overlap_start:
            intervals.append((overlap_start, overlap_end))
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 scoring fixtures")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: results/turn_episode_v1)",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="skip the B0 baseline smoke over the 20 sessions",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="corpus root (default: STB_PHASE2_CORPORA_ROOT)",
    )
    args = parser.parse_args()
    if args.out is None:
        out = Path(__file__).resolve().parent.parent / "results" / "turn_episode_v1"
    else:
        out = args.out

    from .build_episodes import canonical_json, sha256_bytes, verify_manifest
    from .pinned_ledger import ledger_verification

    verify_manifest(out / "episode_manifest_dev.json")
    fixtures = known_answer_fixtures()
    report: dict[str, Any] = {
        "schema_version": "turn_episode_v1",
        "report_id": "scoring_fixture_report",
        "structural_taxonomy_status": "max_duration_and_terminal_deferred_phase3_8",
        "generated_from": {
            "scoring": sha256_bytes(Path(__file__).resolve().read_bytes()),
            "schemas": sha256_bytes((Path(__file__).resolve().parent / "schemas.py").read_bytes()),
            "contracts": sha256_bytes(
                (Path(__file__).resolve().parent / "contracts.py").read_bytes()
            ),
            "episode_manifest_dev": (
                json.loads((out / "episode_manifest_dev.json").read_text(encoding="utf-8"))[
                    "content_sha256"
                ]
                if (out / "episode_manifest_dev.json").is_file()
                else None
            ),
        },
        **ledger_verification(),
        "fixtures": fixtures,
        "fixtures_passed": all(f["passed"] for f in fixtures),
        "baseline_smoke": {},
    }
    if not args.skip_smoke:
        report["baseline_smoke"] = run_b0_smoke(out, args.corpus_root)
    payload = {k: v for k, v in report.items() if k != "content_sha256"}
    report["content_sha256"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    path = out / "scoring_fixture_report.json"
    path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(f"fixtures passed: {report['fixtures_passed']}")
    print(f"wrote {path}")


def run_b0_smoke(out: Path, corpus_root: Path | None = None) -> dict[str, Any]:
    """B0 baseline contamination/harm smoke over the 20 sessions (baseline only).

    B0 hard boundaries come from the Phase 1 full-session B0 traces (raw VAD
    boundaries, canonical projection); references and speech intervals come from the
    episode manifest and the rebuilt session regions. The clean/gap headline uses
    only hard_only episodes (Section 13.4-13.5); overlap_present episodes are
    reported separately.
    """
    from ..corpus import external
    from ..corpus.phase2_schemas import Phase2Manifest
    from .build_episodes import (
        STABLE_INTERVAL_MS,
        SessionData,
        floor_to_chunk,
        load_session_data,
    )

    corpus_root = corpus_root or external.corpus_root()
    manifests_dir = Path(__file__).resolve().parent.parent / "data" / "manifests"
    dev = json.loads((out / "episode_manifest_dev.json").read_text(encoding="utf-8"))
    details_rows: dict[str, dict[str, Any]] = {}
    for line in (
        (out / "coverage_inventory_details.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ):
        row = json.loads(line)
        details_rows[str(row["session_id"])] = row

    by_corpus_rank: dict[str, dict[str, int]] = {}
    for corpus in ("ami", "alimeeting"):
        ids = sorted(
            s
            for s, row in details_rows.items()
            if str(row["corpus"]) == corpus and row.get("wav_path")
        )
        by_corpus_rank[corpus] = {sid: rank for rank, sid in enumerate(ids)}

    pilot_cases: dict[str, list[Any]] = {}
    for name in ("ami_dev_pilot.json", "ami_held_out_pilot.json", "alimeeting_eval_pilot.json"):
        manifest = Phase2Manifest.load(manifests_dir / name)
        for case in manifest.cases:
            pilot_cases.setdefault(str(case.case_id), []).append(case)

    sessions: dict[str, SessionData] = {}
    for session_id, row in details_rows.items():
        if not row.get("wav_path"):
            continue
        sessions[session_id] = load_session_data(
            session_id, row, corpus_root, manifests_dir, pilot_cases, by_corpus_rank
        )

    b0_dir = out / "b0_inventory_replay"
    rows: list[dict[str, Any]] = []
    episodes_scored = 0
    for episode in dev["episodes"]:
        if ":" in episode["session_id"]:
            continue
        if episode["status"] != "scorable":
            continue
        session_id = episode["session_id"]
        session = sessions.get(session_id)
        if session is None:
            continue
        bounds = episode["bounds"]
        last_full = floor_to_chunk(
            min(session.duration_samples, session.wav_length_samples or session.duration_samples)
        )
        processed_end = min(bounds["scored_end"], last_full)
        trace_path = b0_dir / f"{session_id}.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        actions = [
            Action(
                action_id=f"b0:{session_id}:{b['boundary_source_sample']}",
                boundary_source_sample=int(b["boundary_source_sample"]),
                observed_source_sample_at_emit=int(b["observed_source_sample_at_emit"]),
                kind="hard",
                owner="b0",
                session_id=session_id,
                audio_epoch=0,
            )
            for b in trace["trace_projection"]
            if bounds["scored_start"] <= int(b["boundary_source_sample"]) < processed_end
        ]
        references = [ReferenceAction.from_dict(r) for r in episode["references"]]
        singleton_intervals = [
            (
                max(r.start_sample, bounds["scored_start"]),
                min(r.end_sample, processed_end),
                sorted(r.speakers)[0],
            )
            for r in session.regions
            if len(r.speakers) == 1
            and not r.ambiguous
            and max(r.start_sample, bounds["scored_start"]) < min(r.end_sample, processed_end)
        ]
        pause_intervals = [
            (
                max(r.start_sample, bounds["scored_start"]),
                min(r.end_sample, processed_end),
            )
            for r in session.regions
            if not r.speakers
            and not r.ambiguous
            and max(r.start_sample, bounds["scored_start"]) < min(r.end_sample, processed_end)
        ]
        unscored_intervals = [
            (int(r.acceptable_interval[0]), int(r.acceptable_interval[1]))
            for r in references
            if r.action_kind == "unscored" and r.acceptable_interval[1] > r.acceptable_interval[0]
        ]
        overlap_intervals = [
            (
                max(r.start_sample, bounds["scored_start"]),
                min(r.end_sample, processed_end),
            )
            for r in session.regions
            if len(r.speakers) > 1
            and not r.ambiguous
            and (r.end_sample - r.start_sample) / SAMPLES_PER_MS >= STABLE_INTERVAL_MS
            and max(r.start_sample, bounds["scored_start"]) < min(r.end_sample, processed_end)
        ]
        word_intervals: list[tuple[int, int]] | None = None
        if session.corpus == "ami":
            word_intervals = _load_word_intervals_ami(
                session.raw_words or [], bounds["scored_start"], processed_end
            )
            word_timing_source = "ami_words"
        else:
            word_timing_source = "interval_level_only"
        scored = score_episode(
            actions,
            references,
            singleton_intervals,
            pause_intervals,
            word_intervals,
            bounds["scored_start"],
            processed_end,
            unscored_intervals,
            overlap_intervals,
        )
        episodes_scored += 1
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "session_id": session_id,
                "pool": episode["pool"],
                "tag": episode["tag"],
                "hard_action_count": scored["hard_action_count"],
                "match_count": scored["match_count"],
                "hard_miss_count": len(scored["hard_misses"]),
                "contamination_100ms_ms": scored["contamination_ms"]["100ms"],
                "contamination_50ms_ms": scored["contamination_ms"]["50ms"],
                "contamination_200ms_ms": scored["contamination_ms"]["200ms"],
                "deadline_views": scored["deadline_views"],
                "harmful_active_splits": scored["harm_flags"]["harmful_active_split"],
                "lexical_splits": scored["harm_flags"]["lexical_split"],
                "lexical_not_observable": scored["harm_flags"]["lexical_not_observable"],
                "same_speaker_pause_splits": scored["harm_flags"]["same_speaker_pause_split"],
                "overlap_hard_actions": scored["harm_flags"]["overlap_hard_action"],
                "duplicate_hard_boundaries": scored["harm_flags"]["duplicate_hard_boundary"],
                "word_timing_source": word_timing_source,
            }
        )
    hard_only = [r for r in rows if r["tag"] == "hard_only"]
    overlap = [r for r in rows if r["tag"] == "overlap_present"]
    negative = [r for r in rows if r["tag"] == "negative_only"]
    headline_contamination = sum(r["contamination_100ms_ms"] for r in hard_only)
    overlap_contamination = sum(r["contamination_100ms_ms"] for r in overlap)
    word_rows = [r for r in rows if r["word_timing_source"] == "ami_words"]
    interval_rows = [r for r in rows if r["word_timing_source"] == "interval_level_only"]
    return {
        "pool": "baseline_dev",
        "sessions": len(sessions),
        "episodes": episodes_scored,
        "hard_only_episodes": len(hard_only),
        "overlap_present_episodes": len(overlap),
        "negative_only_episodes": len(negative),
        "clean_gap_headline_contamination_100ms_ms": headline_contamination,
        "overlap_episodes_contamination_100ms_ms": overlap_contamination,
        "total_hard_actions": sum(r["hard_action_count"] for r in rows),
        "total_hard_misses": sum(r["hard_miss_count"] for r in rows),
        "hard_only_hard_misses": sum(r["hard_miss_count"] for r in hard_only),
        "total_lexical_splits": sum(r["lexical_splits"] for r in rows),
        "total_lexical_not_observable": sum(r["lexical_not_observable"] for r in rows),
        "lexical_not_observable_ami_words": sum(r["lexical_not_observable"] for r in word_rows),
        "lexical_not_observable_interval_level": sum(
            r["lexical_not_observable"] for r in interval_rows
        ),
        "total_harmful_active_splits": sum(r["harmful_active_splits"] for r in rows),
        "rows": rows,
        "note": "baseline dev evidence only; never confirmatory; never natural rates",
    }


if __name__ == "__main__":
    main()
