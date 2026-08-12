from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .pcm_oracle import contamination_samples, turn_spans
from .phase5_inputs import merge_intervals
from .schemas import ReferenceAction
from .scoring import (
    Action,
    action_eligible,
    gap_eligibility,
    is_gap_reference,
    match_episode,
)

HARD_ACTION_KINDS = {
    "retain_vad",
    "accelerate_or_replace_vad",
    "add_hard_boundary",
    "structural_max_duration",
}


class Phase5ScoringError(RuntimeError):
    pass


def scoring_action_rows(
    rows: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], Action]]:
    pairs: list[tuple[dict[str, Any], Action]] = []
    for row in rows:
        action_kind = str(row["action_kind"])
        if action_kind in HARD_ACTION_KINDS:
            kind = "hard"
        elif action_kind == "emit_soft_marker":
            kind = "soft"
        else:
            continue
        origin = str(row.get("origin", ""))
        owner = "b0" if origin == "vad" else "detector"
        pairs.append(
            (
                row,
                Action(
                    action_id=str(row.get("final_action_id") or row["action_id"]),
                    boundary_source_sample=int(row["boundary_source_sample"]),
                    observed_source_sample_at_emit=int(row["observed_source_sample_at_emit"]),
                    kind=kind,
                    owner=owner,
                    session_id=str(row["source_session_id"]),
                    audio_epoch=int(row["audio_epoch"]),
                ),
            )
        )
    return pairs


def scoring_actions(rows: Sequence[dict[str, Any]]) -> list[Action]:
    return [action for _, action in scoring_action_rows(rows)]


def interval_intersection_samples(intervals: Sequence[Sequence[Any]], start: int, end: int) -> int:
    return sum(max(0, min(end, int(row[1])) - max(start, int(row[0]))) for row in intervals)


def point_in_intervals(point: int, intervals: Sequence[Sequence[Any]]) -> bool:
    return any(int(row[0]) <= point < int(row[1]) for row in intervals)


def subtract_intervals(
    intervals: Sequence[Sequence[Any]], exclusions: Sequence[Sequence[Any]]
) -> list[tuple[Any, ...]]:
    normalized_exclusions = sorted(
        (int(row[0]), int(row[1])) for row in exclusions if int(row[0]) < int(row[1])
    )
    result: list[tuple[Any, ...]] = []
    for raw in intervals:
        pieces = [(int(raw[0]), int(raw[1]))]
        for excluded_start, excluded_end in normalized_exclusions:
            next_pieces: list[tuple[int, int]] = []
            for start, end in pieces:
                if excluded_end <= start or end <= excluded_start:
                    next_pieces.append((start, end))
                    continue
                if start < excluded_start:
                    next_pieces.append((start, excluded_start))
                if excluded_end < end:
                    next_pieces.append((excluded_end, end))
            pieces = next_pieces
        suffix = tuple(raw[2:])
        result.extend((start, end, *suffix) for start, end in pieces if start < end)
    return result


def span_second_speaker_samples(
    span: dict[str, Any],
    singleton_intervals: Sequence[Sequence[Any]],
    owner_threshold_ms: int,
) -> int:
    threshold = owner_threshold_ms * 16
    qualifying: list[tuple[int, int, str]] = []
    for raw_start, raw_end, raw_speaker in singleton_intervals:
        start = max(int(span["start"]), int(raw_start))
        end = min(int(span["end"]), int(raw_end))
        if end - start >= threshold:
            qualifying.append((start, end, str(raw_speaker)))
    qualifying.sort()
    if not qualifying:
        return 0
    owner = qualifying[0][2]
    changed = False
    total = 0
    for start, end, speaker in qualifying:
        if speaker != owner:
            changed = True
        if changed:
            total += end - start
    return total


def active_samples_in_span(
    span: dict[str, Any], singleton_intervals: Sequence[Sequence[Any]]
) -> int:
    return interval_intersection_samples(singleton_intervals, int(span["start"]), int(span["end"]))


def owner_for_span(
    span: dict[str, Any], singleton_intervals: Sequence[Sequence[Any]]
) -> str | None:
    for raw_start, raw_end, raw_speaker in singleton_intervals:
        start = max(int(span["start"]), int(raw_start))
        end = min(int(span["end"]), int(raw_end))
        if end - start >= 1600:
            return str(raw_speaker)
    return None


def duplicate_reference_actions(
    actions: Sequence[Action],
    references: Sequence[ReferenceAction],
    scored_start: int,
    scored_end: int,
) -> int:
    duplicates = 0
    ordered_actions = sorted(
        (action for action in actions if action.kind == "hard"),
        key=lambda action: (
            action.boundary_source_sample,
            action.observed_source_sample_at_emit,
            action.action_id,
        ),
    )
    boundaries = [action.boundary_source_sample for action in ordered_actions]
    for reference in references:
        if reference.action_kind != "hard_boundary" or not reference.scorable:
            continue
        if is_gap_reference(reference):
            eligibility_start, eligibility_end = gap_eligibility(reference)
        else:
            eligibility_start, eligibility_end = reference.acceptable_interval
        start = bisect.bisect_left(boundaries, eligibility_start)
        end = bisect.bisect_right(boundaries, eligibility_end)
        eligible = [
            action
            for action in ordered_actions[start:end]
            if action_eligible(action, reference, scored_start, scored_end)
        ]
        duplicates += max(0, len(eligible) - 1)
    return duplicates


def intersecting_interval_rows(
    span: dict[str, Any],
    intervals: Sequence[Sequence[Any]],
    starts: Sequence[int],
    ends: Sequence[int],
) -> Sequence[Sequence[Any]]:
    first = bisect.bisect_right(ends, int(span["start"]))
    last = bisect.bisect_left(starts, int(span["end"]))
    return intervals[first:last]


def jittered_intervals(
    intervals: Sequence[Sequence[Any]], delta_samples: int
) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    for index, raw in enumerate(intervals):
        start = int(raw[0])
        end = int(raw[1])
        left = -delta_samples if index % 2 == 0 else delta_samples
        right = delta_samples if index % 2 == 0 else -delta_samples
        shifted_start = max(0, start + left)
        shifted_end = max(shifted_start, end + right)
        rows.append((shifted_start, shifted_end, str(raw[2])))
    return rows


def jittered_scorable_intervals(
    intervals: Sequence[Sequence[Any]],
    exclusions: Sequence[Sequence[Any]],
    delta_samples: int,
) -> list[tuple[Any, ...]]:
    return subtract_intervals(jittered_intervals(intervals, delta_samples), exclusions)


def score_policy_episode(
    final_actions: Sequence[dict[str, Any]],
    b0_actions: Sequence[dict[str, Any]],
    references: Sequence[ReferenceAction],
    singleton_intervals: Sequence[Sequence[Any]],
    pause_intervals: Sequence[Sequence[int]],
    overlap_intervals: Sequence[Sequence[int]],
    word_intervals: Sequence[Sequence[int]] | None,
    unscored_intervals: Sequence[Sequence[int]],
    *,
    scored_start: int,
    scored_end: int,
    episode_tag: str,
) -> dict[str, Any]:
    normalized_unscored = merge_intervals(unscored_intervals)
    all_action_pairs = scoring_action_rows(final_actions)
    all_baseline_pairs = scoring_action_rows(b0_actions)
    unscored_action_count = sum(
        1
        for _, action in all_action_pairs
        if scored_start <= action.boundary_source_sample < scored_end
        and point_in_intervals(action.boundary_source_sample, normalized_unscored)
    )
    action_pairs = [
        pair
        for pair in all_action_pairs
        if not point_in_intervals(pair[1].boundary_source_sample, normalized_unscored)
    ]
    baseline_pairs = [
        pair
        for pair in all_baseline_pairs
        if not point_in_intervals(pair[1].boundary_source_sample, normalized_unscored)
    ]
    matching_action_pairs = [
        pair for pair in action_pairs if str(pair[0]["action_kind"]) != "structural_max_duration"
    ]
    matching_baseline_pairs = [
        pair for pair in baseline_pairs if str(pair[0]["action_kind"]) != "structural_max_duration"
    ]
    actions = [action for _, action in matching_action_pairs]
    baseline_actions = [action for _, action in matching_baseline_pairs]
    clipped_singleton_raw = [
        (max(scored_start, int(row[0])), min(scored_end, int(row[1])), str(row[2]))
        for row in singleton_intervals
        if max(scored_start, int(row[0])) < min(scored_end, int(row[1]))
    ]
    clipped_singleton = [
        (int(row[0]), int(row[1]), str(row[2]))
        for row in subtract_intervals(clipped_singleton_raw, normalized_unscored)
    ]
    baseline_matches, baseline_hard_misses, baseline_deadline_views = match_episode(
        baseline_actions,
        list(references),
        scored_start,
        scored_end,
    )
    baseline_actions_by_id = {action.action_id: action for action in baseline_actions}
    baseline_reference_by_action = {
        match.action_id: match.reference_id for match in baseline_matches
    }
    baseline_actions_by_reference = {
        match.reference_id: baseline_actions_by_id[match.action_id] for match in baseline_matches
    }
    fixed_pairs: dict[str, str] = {}
    reference_by_id = {reference.reference_id: reference for reference in references}
    for row, action in matching_action_pairs:
        baseline_action_id: str | None = None
        if str(row.get("origin", "")) == "vad":
            baseline_action_id = str(row.get("event_id") or row.get("action_id") or "")
        elif row.get("associated_vad_action_id") is not None:
            baseline_action_id = str(row["associated_vad_action_id"])
        reference_id = baseline_reference_by_action.get(baseline_action_id or "")
        if reference_id is None:
            continue
        reference = reference_by_id.get(reference_id)
        if reference is not None and action_eligible(
            action, reference, scored_start, scored_end
        ):
            fixed_pairs[action.action_id] = reference_id
    matches, hard_misses, deadline_views = match_episode(
        actions,
        list(references),
        scored_start,
        scored_end,
        fixed_pairs=fixed_pairs,
        baseline_actions_by_reference=baseline_actions_by_reference,
    )
    hard_boundaries = sorted(
        {
            action.boundary_source_sample
            for _, action in action_pairs
            if action.kind == "hard" and scored_start <= action.boundary_source_sample < scored_end
        }
    )
    spans = turn_spans(scored_start, scored_end, hard_boundaries)
    contamination_views = {
        str(owner_ms): contamination_samples(spans, clipped_singleton, owner_ms)
        for owner_ms in (50, 100, 200)
    }
    jitter_views = {
        str(delta_ms): contamination_samples(
            spans,
            jittered_scorable_intervals(clipped_singleton_raw, normalized_unscored, delta_ms * 16),
            100,
        )
        for delta_ms in (-50, -20, 20, 50)
    }
    harm = Counter(
        {
            "harmful_active_split": 0,
            "lexical_split": 0,
            "lexical_not_observable": 0,
            "same_speaker_pause_split": 0,
            "duplicate_hard_boundary": duplicate_reference_actions(
                actions, references, scored_start, scored_end
            ),
            "overlap_hard_action": 0,
            "structural_split": 0,
            "unscored_action": unscored_action_count,
        }
    )
    harmful_sensitivity = {str(guard): 0 for guard in (100, 200, 300)}
    scored_hard = [
        (row, action)
        for row, action in action_pairs
        if action.kind == "hard"
        if scored_start <= action.boundary_source_sample < scored_end
    ]
    singleton_starts = [int(row[0]) for row in clipped_singleton]
    singleton_ends = [int(row[1]) for row in clipped_singleton]
    normalized_pauses = merge_intervals(pause_intervals)
    pause_starts = [row[0] for row in normalized_pauses]
    normalized_overlap = merge_intervals(overlap_intervals)
    overlap_starts = [row[0] for row in normalized_overlap]
    word_timing_observable = word_intervals is not None
    normalized_words = sorted(
        (int(row[0]), int(row[1])) for row in (word_intervals if word_intervals is not None else ())
    )
    word_starts = [row[0] for row in normalized_words]
    word_prefix_max_ends: list[int] = []
    for _, end in normalized_words:
        word_prefix_max_ends.append(
            max(end, word_prefix_max_ends[-1] if word_prefix_max_ends else end)
        )

    def guarded_singleton(boundary: int, guard_ms: int) -> bool:
        index = bisect.bisect_right(singleton_starts, boundary) - 1
        if index < 0:
            return False
        guard = guard_ms * 16
        return (
            singleton_starts[index] <= boundary - guard
            and boundary + guard <= singleton_ends[index]
        )

    def inside_pause(boundary: int) -> bool:
        index = bisect.bisect_right(pause_starts, boundary) - 1
        return index >= 0 and boundary <= normalized_pauses[index][1]

    def inside_overlap(boundary: int) -> bool:
        index = bisect.bisect_right(overlap_starts, boundary) - 1
        return index >= 0 and boundary < normalized_overlap[index][1]

    def lexical_disposition(boundary: int) -> bool | None:
        if not word_timing_observable:
            return None
        index = bisect.bisect_right(word_starts, boundary - 320) - 1
        return index >= 0 and boundary + 320 <= word_prefix_max_ends[index]

    for row, action in scored_hard:
        boundary = action.boundary_source_sample
        structural = str(row["action_kind"]) == "structural_max_duration"
        if structural:
            harm["structural_split"] += 1
            continue
        detector_created_action = str(row.get("origin", "")) in ("detector", "control") and str(
            row["action_kind"]
        ) in ("add_hard_boundary", "accelerate_or_replace_vad")
        for guard in (100, 200, 300):
            if guarded_singleton(boundary, guard):
                harmful_sensitivity[str(guard)] += 1
        if guarded_singleton(boundary, 200):
            harm["harmful_active_split"] += 1
        if detector_created_action and inside_pause(boundary):
            harm["same_speaker_pause_split"] += 1
        split = lexical_disposition(boundary)
        if split is True:
            harm["lexical_split"] += 1
        elif split is None:
            harm["lexical_not_observable"] += 1
        if inside_overlap(boundary):
            harm["overlap_hard_action"] += 1
    span_singleton_rows = [
        intersecting_interval_rows(span, clipped_singleton, singleton_starts, singleton_ends)
        for span in spans
    ]
    active_durations = [
        active_samples_in_span(span, rows) for span, rows in zip(spans, span_singleton_rows)
    ]
    source_durations = [
        int(span["end"])
        - int(span["start"])
        - interval_intersection_samples(normalized_unscored, int(span["start"]), int(span["end"]))
        for span in spans
    ]
    short_fragments = {
        str(threshold): sum(1 for value in active_durations if 0 < value < threshold * 16)
        for threshold in (250, 500, 1000)
    }
    owners = [owner_for_span(span, rows) for span, rows in zip(spans, span_singleton_rows)]
    structural_boundaries = {
        action.boundary_source_sample
        for row, action in scored_hard
        if str(row["action_kind"]) == "structural_max_duration"
    }
    same_owner_boundaries = {
        int(spans[index]["end"])
        for index, (left, right) in enumerate(zip(owners, owners[1:]))
        if left is not None and left == right
        if int(spans[index]["end"]) not in structural_boundaries
    }
    detector_pause_boundaries = {
        action.boundary_source_sample
        for row, action in scored_hard
        if str(row.get("origin", "")) in ("detector", "control")
        and str(row["action_kind"]) in ("add_hard_boundary", "accelerate_or_replace_vad")
        and inside_pause(action.boundary_source_sample)
    }
    same_owner_adjacencies = len(same_owner_boundaries)
    same_speaker_extra_turn_boundaries = same_owner_boundaries | detector_pause_boundaries
    second_speaker_turns = {
        str(threshold): sum(
            1
            for span, rows in zip(spans, span_singleton_rows)
            if span_second_speaker_samples(span, rows, 100) >= threshold * 16
        )
        for threshold in (100, 250, 500)
    }
    attribution = Counter(match.benefit_attribution for match in matches)
    detector_created = sum(
        1
        for row, action in scored_hard
        if row.get("origin") in ("detector", "control")
        and row.get("action_kind") in ("add_hard_boundary", "accelerate_or_replace_vad")
    )
    headline = episode_tag == "hard_only"
    primary = contamination_views["100"] if headline else None
    overlap_detector_boundaries = {
        action.boundary_source_sample
        for row, action in scored_hard
        if str(row.get("origin", "")) in ("detector", "control")
        and str(row["action_kind"]) in ("add_hard_boundary", "accelerate_or_replace_vad")
        and inside_overlap(action.boundary_source_sample)
    }
    counterfactual_boundaries = [
        boundary for boundary in hard_boundaries if boundary not in overlap_detector_boundaries
    ]
    counterfactual_spans = turn_spans(scored_start, scored_end, counterfactual_boundaries)
    overlap_counterfactual = {}
    for owner_ms in (50, 100, 200):
        actual = contamination_views[str(owner_ms)]
        suppressed = contamination_samples(counterfactual_spans, clipped_singleton, owner_ms)
        overlap_counterfactual[str(owner_ms)] = {
            "actual_contaminated_samples": int(actual["contaminated_samples"]),
            "suppressed_contaminated_samples": int(suppressed["contaminated_samples"]),
            "actual_minus_suppressed_samples": int(actual["contaminated_samples"])
            - int(suppressed["contaminated_samples"]),
            "suppressed_detector_boundary_count": len(overlap_detector_boundaries),
        }
    return {
        "match_count": len(matches),
        "matches": [
            {
                "reference_id": row.reference_id,
                "action_id": row.action_id,
                "benefit_attribution": row.benefit_attribution,
                "availability_delay_ms": row.availability_delay_ms,
                "localization_error_ms": row.localization_error_ms,
                "pre_existing": row.pre_existing,
            }
            for row in matches
        ],
        "hard_miss_count": len(hard_misses),
        "deadline_views": deadline_views,
        "b0_match_count": len(baseline_matches),
        "b0_matches": [
            {
                "reference_id": row.reference_id,
                "action_id": row.action_id,
                "availability_delay_ms": row.availability_delay_ms,
                "localization_error_ms": row.localization_error_ms,
                "pre_existing": row.pre_existing,
            }
            for row in baseline_matches
        ],
        "b0_hard_miss_count": len(baseline_hard_misses),
        "b0_deadline_views": baseline_deadline_views,
        "benefit_attribution_counts": dict(sorted(attribution.items())),
        "contamination_by_owner_threshold": contamination_views,
        "overlap_hard_action_counterfactual_by_owner_threshold": overlap_counterfactual,
        "turn_owner_jitter_sensitivity": jitter_views,
        "clean_gap_headline_eligible": headline,
        "clean_gap_contaminated_samples": (
            int(primary["contaminated_samples"]) if primary is not None else None
        ),
        "clean_gap_singleton_denominator_samples": (
            int(primary["denominator_samples"]) if primary is not None else None
        ),
        "second_speaker_turn_counts": second_speaker_turns,
        "harm_or_structure_counts": dict(sorted(harm.items())),
        "harmful_active_split_sensitivity": harmful_sensitivity,
        "detector_created_hard_action_count": detector_created,
        "same_speaker_extra_turn_count": len(same_speaker_extra_turn_boundaries),
        "short_active_fragment_counts": short_fragments,
        "segment_source_duration_samples": source_durations,
        "segment_active_duration_samples": active_durations,
        "consecutive_same_owner_fragment_adjacencies": same_owner_adjacencies,
        "sampled_singleton_exposure_samples": interval_intersection_samples(
            clipped_singleton, scored_start, scored_end
        ),
        "sampled_overlap_exposure_samples": interval_intersection_samples(
            subtract_intervals(overlap_intervals, normalized_unscored), scored_start, scored_end
        ),
        "sampled_unscored_exposure_samples": interval_intersection_samples(
            normalized_unscored, scored_start, scored_end
        ),
    }
