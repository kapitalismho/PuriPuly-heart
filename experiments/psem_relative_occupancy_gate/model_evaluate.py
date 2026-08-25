from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from experiments.psem_relative_occupancy_gate.contracts import (
    ActivityInterval,
    AnchorEpisode,
    AnchorLifecycle,
    RelativeState,
)
from experiments.psem_relative_occupancy_gate.decoder import (
    GTSessionResult,
    RelativeObservation,
    ReplacementDecoder,
    ReplacementEvent,
    simulate_gt_session,
)
from experiments.psem_relative_occupancy_gate.evaluate import (
    monotonic_boundary_matches,
    state_from_presence,
    timeline_exposure,
    weighted_average_precision,
    weighted_binary_confusion,
    weighted_binary_pr_curve,
    weighted_relative_state_confusion,
)
from experiments.psem_relative_occupancy_gate.io_utils import percentile
from experiments.psem_relative_occupancy_gate.model_decode import (
    CausalAnchorEpisode,
    CausalSessionResult,
    ModelObservation,
    OracleAnchorMapping,
    PosteriorCell,
    oracle_anchor_mapping,
    relative_probabilities,
)


@dataclass(frozen=True, slots=True)
class PrimitiveRecord:
    source_id: str
    anchor_episode_id: str
    center_sample: int
    weight_samples: int
    anchor_label: bool
    other_label: bool
    anchor_score: float
    other_score: float
    evidence_frontier_sample: int


@dataclass(frozen=True, slots=True)
class AnnotatedCausalEpisode:
    episode: CausalAnchorEpisode
    expected_anchor_speaker: str | None
    opportunity_start_sample: int | None
    oracle_slot_index: int | None
    correct_anchor: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.episode.to_dict(),
            "expected_anchor_speaker": self.expected_anchor_speaker,
            "opportunity_start_sample": self.opportunity_start_sample,
            "oracle_slot_index": self.oracle_slot_index,
            "correct_anchor": self.correct_anchor,
        }


def intervals_from_manifest(row: dict[str, Any]) -> tuple[ActivityInterval, ...]:
    return tuple(ActivityInterval.from_dict(value) for value in row["intervals"])


def gt_reference_session(
    row: dict[str, Any],
    *,
    replacement_confirmation_samples: int,
    enrollment_samples: int,
    silence_reset_samples: int,
) -> GTSessionResult:
    return simulate_gt_session(
        source_id=str(row["source_id"]),
        intervals=intervals_from_manifest(row),
        confirmation_samples=replacement_confirmation_samples,
        enrollment_samples=enrollment_samples,
        silence_reset_samples=silence_reset_samples,
    )


def gate1_primitive_records(
    *,
    source_id: str,
    episodes: Sequence[AnchorEpisode],
    cells: Sequence[PosteriorCell],
    slot_ids: Sequence[str],
) -> tuple[tuple[PrimitiveRecord, ...], tuple[OracleAnchorMapping, ...]]:
    records: list[PrimitiveRecord] = []
    mappings: list[OracleAnchorMapping] = []
    for episode in episodes:
        try:
            mapping = oracle_anchor_mapping(episode, cells, slot_ids)
        except ValueError:
            continue
        mappings.append(mapping)
        start_index = bisect_left(
            cells,
            episode.anchor_emit_sample,
            key=lambda value: value.cell.center_sample,
        )
        end_index = bisect_left(
            cells,
            episode.end_emit_sample,
            key=lambda value: value.cell.center_sample,
        )
        for posterior in cells[start_index:end_index]:
            cell = posterior.cell
            if cell.masked:
                continue
            probabilities = relative_probabilities(posterior, mapping.slot_index)
            if probabilities is None:
                continue
            p_anchor, p_other = probabilities
            records.append(
                PrimitiveRecord(
                    source_id=source_id,
                    anchor_episode_id=episode.episode_id,
                    center_sample=cell.center_sample,
                    weight_samples=cell.duration_samples,
                    anchor_label=episode.anchor_speaker in cell.active_speakers,
                    other_label=any(
                        speaker != episode.anchor_speaker for speaker in cell.active_speakers
                    ),
                    anchor_score=p_anchor,
                    other_score=p_other,
                    evidence_frontier_sample=posterior.evidence_frontier_sample,
                )
            )
    return tuple(records), tuple(mappings)


def primitive_metrics(
    records: Sequence[PrimitiveRecord],
    thresholds: Sequence[float],
    selected_thresholds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    if not records and selected_thresholds is None:
        raise ValueError("primitive metrics require valid records")
    anchor_labels = [value.anchor_label for value in records]
    other_labels = [value.other_label for value in records]
    anchor_scores = [value.anchor_score for value in records]
    other_scores = [value.other_score for value in records]
    weights = [float(value.weight_samples) for value in records]
    anchor_array = np.asarray(anchor_labels, dtype=np.bool_)
    other_array = np.asarray(other_labels, dtype=np.bool_)
    anchor_score_array = np.asarray(anchor_scores, dtype=np.float64)
    other_score_array = np.asarray(other_scores, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    truth_indices = np.where(
        anchor_array,
        np.where(other_array, 2, 1),
        np.where(other_array, 3, 0),
    )
    operating_points: list[dict[str, Any]] = []
    for anchor_threshold in thresholds:
        for other_threshold in thresholds:
            predicted_indices = np.where(
                anchor_score_array >= anchor_threshold,
                np.where(other_score_array >= other_threshold, 2, 1),
                np.where(other_score_array >= other_threshold, 3, 0),
            )
            matrix = np.bincount(
                truth_indices * 4 + predicted_indices,
                weights=weight_array,
                minlength=16,
            ).reshape(4, 4)
            f1_values = []
            for index in range(4):
                tp = float(matrix[index, index])
                fp = float(matrix[:, index].sum() - tp)
                fn = float(matrix[index, :].sum() - tp)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1_values.append(
                    2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
                )
            operating_points.append(
                {
                    "anchor_threshold": float(anchor_threshold),
                    "other_threshold": float(other_threshold),
                    "macro_f1": float(np.mean(f1_values)),
                }
            )
    if selected_thresholds is None:
        selected = min(
            operating_points,
            key=lambda value: (
                -float(value["macro_f1"]),
                float(value["anchor_threshold"]),
                float(value["other_threshold"]),
            ),
        )
        selection_rule = "maximum_duration_weighted_four_state_macro_f1_then_lowest_thresholds"
    else:
        anchor_value, other_value = selected_thresholds
        selected = next(
            value
            for value in operating_points
            if value["anchor_threshold"] == anchor_value and value["other_threshold"] == other_value
        )
        selection_rule = "frozen_gate1_dev_operating_point"
    anchor_threshold = float(selected["anchor_threshold"])
    other_threshold = float(selected["other_threshold"])
    selected_state_confusion = weighted_relative_state_confusion(
        anchor_labels,
        other_labels,
        anchor_scores,
        other_scores,
        weights,
        anchor_threshold,
        other_threshold,
    )
    confusion = selected_state_confusion["confusion_weight"]
    state_order = selected_state_confusion["state_order"]
    other_only_index = state_order.index(RelativeState.OTHER_ONLY.value)
    anchor_only_index = state_order.index(RelativeState.ANCHOR_ONLY.value)
    false_other = sum(
        float(confusion[index][other_only_index])
        for index in range(len(state_order))
        if index != other_only_index
    )
    missed_other = sum(
        float(confusion[other_only_index][index])
        for index in range(len(state_order))
        if index != other_only_index
    )
    return {
        "record_count": len(records),
        "episode_count": len({value.anchor_episode_id for value in records}),
        "duration_seconds": sum(weights) / 16000.0,
        "anchor_present": {
            "average_precision": weighted_average_precision(anchor_labels, anchor_scores, weights),
            "pr_curve": weighted_binary_pr_curve(anchor_labels, anchor_scores, weights, thresholds),
            "selected_confusion": weighted_binary_confusion(
                anchor_labels, anchor_scores, weights, anchor_threshold
            ),
        },
        "other_present": {
            "average_precision": weighted_average_precision(other_labels, other_scores, weights),
            "pr_curve": weighted_binary_pr_curve(other_labels, other_scores, weights, thresholds),
            "selected_confusion": weighted_binary_confusion(
                other_labels, other_scores, weights, other_threshold
            ),
        },
        "selected_operating_point": {
            **selected,
            "selection_rule": selection_rule,
            "state_confusion": selected_state_confusion,
            "state_duration_errors_seconds": {
                "false_OTHER_ONLY": false_other / 16000.0,
                "missed_OTHER_ONLY": missed_other / 16000.0,
                "false_OTHER_ONLY_inside_GT_ANCHOR_ONLY": float(
                    confusion[anchor_only_index][other_only_index]
                )
                / 16000.0,
            },
        },
        "operating_points": operating_points,
    }


def transition_timing(
    records: Sequence[PrimitiveRecord],
    anchor_threshold: float,
    other_threshold: float,
) -> dict[str, Any]:
    by_episode: dict[str, list[PrimitiveRecord]] = {}
    for record in records:
        by_episode.setdefault(record.anchor_episode_id, []).append(record)
    tracked = {
        "ANCHOR_ONLY_TO_OTHER_ONLY": [],
        "ANCHOR_PLUS_OTHER_TO_OTHER_ONLY": [],
        "OTHER_ONSET": [],
        "OTHER_OFFSET": [],
    }
    availability_delays = []
    for episode_records in by_episode.values():
        ordered = sorted(episode_records, key=lambda value: value.center_sample)
        truth_states = [
            state_from_presence(value.anchor_label, value.other_label) for value in ordered
        ]
        predicted_states = [
            state_from_presence(
                value.anchor_score >= anchor_threshold,
                value.other_score >= other_threshold,
            )
            for value in ordered
        ]
        for index in range(1, len(ordered)):
            previous = truth_states[index - 1]
            current = truth_states[index]
            kinds: list[str] = []
            if previous is RelativeState.ANCHOR_ONLY and current is RelativeState.OTHER_ONLY:
                kinds.append("ANCHOR_ONLY_TO_OTHER_ONLY")
            if previous is RelativeState.ANCHOR_PLUS_OTHER and current is RelativeState.OTHER_ONLY:
                kinds.append("ANCHOR_PLUS_OTHER_TO_OTHER_ONLY")
            if not ordered[index - 1].other_label and ordered[index].other_label:
                kinds.append("OTHER_ONSET")
            if ordered[index - 1].other_label and not ordered[index].other_label:
                kinds.append("OTHER_OFFSET")
            for kind in kinds:
                availability_delays.append(
                    (ordered[index].evidence_frontier_sample - ordered[index].center_sample)
                    * 1000.0
                    / 16000.0
                )
                target_state = current
                match = next(
                    (
                        later
                        for later in range(index, min(index + 21, len(ordered)))
                        if predicted_states[later] is target_state
                    ),
                    None,
                )
                tracked[kind].append(
                    None
                    if match is None
                    else (ordered[match].center_sample - ordered[index].center_sample)
                    * 1000.0
                    / 16000.0
                )
    result: dict[str, Any] = {}
    for kind, values in tracked.items():
        observed = [float(value) for value in values if value is not None]
        result[kind] = {
            "transition_count": len(values),
            "matched_within_2000ms": len(observed),
            "delay_ms": {
                "p50": percentile(observed, 50),
                "p90": percentile(observed, 90),
            },
        }
    result["MODEL_EVIDENCE_AVAILABILITY"] = {
        "transition_count": len(availability_delays),
        "delay_ms": {
            "p50": percentile(availability_delays, 50),
            "p90": percentile(availability_delays, 90),
        },
    }
    return result


def _state_for_probabilities(
    probabilities: Sequence[float],
    alive: Sequence[bool],
    anchor_slot_index: int,
    anchor_threshold: float,
    other_threshold: float,
) -> RelativeState | None:
    if not alive[anchor_slot_index]:
        return None
    p_anchor = probabilities[anchor_slot_index]
    p_other = max(
        (
            value
            for index, (value, valid) in enumerate(zip(probabilities, alive, strict=True))
            if index != anchor_slot_index and valid
        ),
        default=0.0,
    )
    return state_from_presence(p_anchor >= anchor_threshold, p_other >= other_threshold)


def decode_gate1_episode(
    *,
    source_id: str,
    episode: AnchorEpisode,
    mapping: OracleAnchorMapping,
    observations: Sequence[ModelObservation],
    anchor_threshold: float,
    other_threshold: float,
    replacement_confirmation_samples: int,
) -> ReplacementEvent | None:
    decoder = ReplacementDecoder(source_id, replacement_confirmation_samples)
    continuity_valid = True
    start_index = bisect_right(
        observations,
        episode.anchor_emit_sample,
        key=lambda value: value.end_sample,
    )
    end_index = bisect_left(
        observations,
        episode.end_emit_sample,
        key=lambda value: value.start_sample,
    )
    for observation in observations[start_index:end_index]:
        start = max(observation.start_sample, episode.anchor_emit_sample)
        end = min(observation.end_sample, episode.end_emit_sample)
        if end <= start:
            continue
        if observation.state_reset and observation.start_sample > episode.anchor_emit_sample:
            continuity_valid = False
        valid = (
            continuity_valid
            and observation.trace_valid
            and observation.slot_alive[mapping.slot_index]
        )
        state = None
        lifecycle = AnchorLifecycle.ANCHOR_UNCERTAIN
        masked = observation.masked
        if masked:
            lifecycle = AnchorLifecycle.ANCHORED
        elif valid:
            state = _state_for_probabilities(
                observation.probabilities,
                observation.slot_alive,
                mapping.slot_index,
                anchor_threshold,
                other_threshold,
            )
            lifecycle = AnchorLifecycle.ANCHORED
        event = decoder.advance(
            RelativeObservation(
                start_sample=start,
                end_sample=end,
                state=state,
                masked=masked,
                evidence_frontier_sample=max(observation.evidence_frontier_sample, end),
            ),
            lifecycle=lifecycle,
            anchor_id=episode.anchor_speaker if lifecycle is AnchorLifecycle.ANCHORED else None,
            anchor_episode_id=(
                episode.episode_id if lifecycle is AnchorLifecycle.ANCHORED else None
            ),
        )
        if event is not None:
            return event
    return None


def _duration_where(
    intervals: Sequence[ActivityInterval],
    start_sample: int,
    end_sample: int,
    predicate: Any,
) -> int:
    total = 0
    for interval in intervals:
        start = max(start_sample, interval.start_sample)
        end = min(end_sample, interval.end_sample)
        if end > start and predicate(interval):
            total += end - start
    return total


def active_speech_samples(intervals: Sequence[ActivityInterval]) -> int:
    return sum(
        value.end_sample - value.start_sample for value in intervals if value.active_speakers
    )


def exact_episode_contamination_samples(
    intervals: Sequence[ActivityInterval],
    *,
    anchor_speaker: str,
    start_sample: int,
    end_sample: int,
) -> int:
    return _duration_where(
        intervals,
        start_sample,
        end_sample,
        lambda interval: (
            not interval.masked
            and anchor_speaker not in interval.active_speakers
            and bool(interval.active_speakers)
        ),
    )


def product_event_metrics(
    *,
    predicted_events: Sequence[ReplacementEvent],
    reference: GTSessionResult,
    intervals: Sequence[ActivityInterval],
    contamination_episodes: Sequence[tuple[str, int, int]],
    tolerance_samples: int,
) -> dict[str, Any]:
    predicted = sorted(predicted_events, key=lambda value: value.boundary_source_sample)
    references = sorted(reference.events, key=lambda value: value.boundary_source_sample)
    matches = monotonic_boundary_matches(
        [value.boundary_source_sample for value in predicted],
        [value.boundary_source_sample for value in references],
        tolerance_samples,
    )
    matched_predicted = {left for left, _ in matches}
    matched_references = {right for _, right in matches}
    emit_delays = [
        (predicted[left].decoder_emit_sample - references[right].boundary_source_sample)
        * 1000.0
        / 16000.0
        for left, right in matches
    ]
    evidence_delays = [
        (predicted[left].model_evidence_frontier_sample - references[right].boundary_source_sample)
        * 1000.0
        / 16000.0
        for left, right in matches
    ]
    boundary_errors = [
        (predicted[left].boundary_source_sample - references[right].boundary_source_sample)
        * 1000.0
        / 16000.0
        for left, right in matches
    ]
    predicted_by_reference = {right: predicted[left] for left, right in matches}
    contamination_per_replacement = []
    scored_end_sample = intervals[-1].end_sample
    for index, reference_event in enumerate(references):
        next_boundary = (
            references[index + 1].boundary_source_sample
            if index + 1 < len(references)
            else scored_end_sample
        )
        predicted_event = predicted_by_reference.get(index)
        stop = (
            min(predicted_event.decoder_emit_sample, next_boundary)
            if predicted_event is not None
            else next_boundary
        )
        start = reference_event.boundary_source_sample
        contamination_per_replacement.append(
            exact_episode_contamination_samples(
                intervals,
                anchor_speaker=reference_event.anchor_id,
                start_sample=start,
                end_sample=max(start, stop),
            )
            / 16000.0
        )
    logical_episode_contamination = sum(
        exact_episode_contamination_samples(
            intervals,
            anchor_speaker=anchor,
            start_sample=start,
            end_sample=end,
        )
        for anchor, start, end in contamination_episodes
    )
    active_samples = active_speech_samples(intervals)
    active_hours = active_samples / 16000.0 / 3600.0
    contamination_seconds = sum(contamination_per_replacement)
    return {
        "predicted_cut_count": len(predicted),
        "reference_replacement_count": len(references),
        "matched_replacement_count": len(matches),
        "false_cut_count": len(predicted) - len(matched_predicted),
        "missed_replacement_count": len(references) - len(matched_references),
        "speaker_induced_cut_count_per_active_speech_hour": (
            len(predicted) / active_hours if active_hours else None
        ),
        "active_speech_seconds": active_samples / 16000.0,
        "exclusive_other_contamination_seconds": contamination_seconds,
        "exclusive_other_contamination_seconds_per_active_speech_hour": (
            contamination_seconds / active_hours if active_hours else None
        ),
        "logical_episode_exclusive_other_contamination_seconds": (
            logical_episode_contamination / 16000.0
        ),
        "contamination_seconds_per_true_replacement": {
            "p50": percentile(contamination_per_replacement, 50),
            "p90": percentile(contamination_per_replacement, 90),
        },
        "replacement_emit_delay_ms": {
            "p50": percentile(emit_delays, 50),
            "p90": percentile(emit_delays, 90),
        },
        "model_evidence_delay_ms": {
            "p50": percentile(evidence_delays, 50),
            "p90": percentile(evidence_delays, 90),
        },
        "backdated_boundary_error_ms": {
            "p50": percentile(boundary_errors, 50),
            "p90": percentile(boundary_errors, 90),
        },
        "replacement_emit_delay_values_ms": emit_delays,
        "model_evidence_delay_values_ms": evidence_delays,
        "backdated_boundary_error_values_ms": boundary_errors,
        "contamination_values_seconds_per_true_replacement": contamination_per_replacement,
        "matches": [
            {
                "predicted_index": left,
                "reference_index": right,
                "predicted_boundary_sample": predicted[left].boundary_source_sample,
                "reference_boundary_sample": references[right].boundary_source_sample,
            }
            for left, right in matches
        ],
    }


def _covered_samples(
    start_sample: int, end_sample: int, ranges: Sequence[tuple[int, int]]
) -> int:
    return sum(
        max(0, min(end_sample, right) - max(start_sample, left)) for left, right in ranges
    )


def gate1_fail_closed_exposure(
    *,
    intervals: Sequence[ActivityInterval],
    anchored_ranges: Sequence[tuple[int, int]],
    uncertain_ranges: Sequence[tuple[int, int]],
    exact_contamination_seconds: float,
) -> dict[str, float]:
    masked_samples = 0
    masked_active_samples = 0
    unanchored_active_samples = 0
    uncertain_active_samples = 0
    for interval in intervals:
        duration = interval.end_sample - interval.start_sample
        if interval.masked:
            masked_samples += duration
            if interval.active_speakers:
                masked_active_samples += duration
            continue
        if not interval.active_speakers:
            continue
        anchored = _covered_samples(
            interval.start_sample, interval.end_sample, anchored_ranges
        )
        uncertain = _covered_samples(
            interval.start_sample, interval.end_sample, uncertain_ranges
        )
        uncertain_active_samples += uncertain
        unanchored_active_samples += max(0, duration - anchored - uncertain)
    unknown = masked_active_samples + unanchored_active_samples + uncertain_active_samples
    return {
        "masked_seconds": masked_samples / 16000.0,
        "masked_active_speech_seconds": masked_active_samples / 16000.0,
        "unanchored_active_speech_seconds": unanchored_active_samples / 16000.0,
        "anchor_uncertain_active_speech_seconds": uncertain_active_samples / 16000.0,
        "fail_closed_unknown_active_speech_seconds": unknown / 16000.0,
        "exclusive_other_contamination_upper_bound_seconds": (
            exact_contamination_seconds
            + unanchored_active_samples / 16000.0
            + uncertain_active_samples / 16000.0
        ),
    }


def gate1_product_session(
    *,
    source_id: str,
    reference: GTSessionResult,
    cells: Sequence[PosteriorCell],
    slot_ids: Sequence[str],
    observations: Sequence[ModelObservation],
    intervals: Sequence[ActivityInterval],
    anchor_threshold: float,
    other_threshold: float,
    replacement_confirmation_samples: int,
    tolerance_samples: int,
) -> tuple[dict[str, Any], tuple[OracleAnchorMapping, ...], tuple[ReplacementEvent, ...]]:
    mappings: list[OracleAnchorMapping] = []
    events: list[ReplacementEvent] = []
    contamination_episodes: list[tuple[str, int, int]] = []
    anchored_ranges: list[tuple[int, int]] = []
    uncertain_ranges: list[tuple[int, int]] = []
    for episode in reference.episodes:
        try:
            mapping = oracle_anchor_mapping(episode, cells, slot_ids)
        except ValueError:
            uncertain_ranges.append((episode.anchor_emit_sample, episode.end_emit_sample))
            continue
        mappings.append(mapping)
        event = decode_gate1_episode(
            source_id=source_id,
            episode=episode,
            mapping=mapping,
            observations=observations,
            anchor_threshold=anchor_threshold,
            other_threshold=other_threshold,
            replacement_confirmation_samples=replacement_confirmation_samples,
        )
        if event is not None:
            events.append(event)
        end = min(
            episode.end_emit_sample,
            event.decoder_emit_sample if event is not None else episode.end_emit_sample,
        )
        contamination_episodes.append((episode.anchor_speaker, episode.anchor_emit_sample, end))
        anchored_ranges.append((episode.anchor_emit_sample, end))
    metrics = product_event_metrics(
        predicted_events=events,
        reference=reference,
        intervals=intervals,
        contamination_episodes=contamination_episodes,
        tolerance_samples=tolerance_samples,
    )
    metrics.update(
        gate1_fail_closed_exposure(
            intervals=intervals,
            anchored_ranges=anchored_ranges,
            uncertain_ranges=uncertain_ranges,
            exact_contamination_seconds=metrics["exclusive_other_contamination_seconds"],
        )
    )
    return metrics, tuple(mappings), tuple(events)


def _clip_intervals(
    intervals: Sequence[ActivityInterval], start_sample: int, end_sample: int
) -> tuple[ActivityInterval, ...]:
    values = []
    start_index = bisect_right(
        intervals,
        start_sample,
        key=lambda value: value.end_sample,
    )
    end_index = bisect_left(
        intervals,
        end_sample,
        key=lambda value: value.start_sample,
    )
    for interval in intervals[start_index:end_index]:
        start = max(start_sample, interval.start_sample)
        end = min(end_sample, interval.end_sample)
        if end > start:
            values.append(ActivityInterval(start, end, interval.active_speakers, interval.masked))
    return tuple(values)


def first_gt_singleton_opportunity(
    intervals: Sequence[ActivityInterval],
    *,
    start_sample: int,
    end_sample: int,
    enrollment_samples: int,
    silence_reset_samples: int,
) -> tuple[str, int, int] | None:
    opportunities = gt_singleton_opportunities(
        intervals,
        start_sample=start_sample,
        end_sample=end_sample,
        enrollment_samples=enrollment_samples,
        silence_reset_samples=silence_reset_samples,
    )
    return opportunities[0] if opportunities else None


def gt_singleton_opportunities(
    intervals: Sequence[ActivityInterval],
    *,
    start_sample: int,
    end_sample: int,
    enrollment_samples: int,
    silence_reset_samples: int,
) -> tuple[tuple[str, int, int], ...]:
    clipped = _clip_intervals(intervals, start_sample, end_sample)
    if not clipped:
        return ()
    result = simulate_gt_session(
        source_id="gt_opportunity",
        intervals=clipped,
        confirmation_samples=max(end_sample - start_sample + 1, 1),
        enrollment_samples=enrollment_samples,
        silence_reset_samples=silence_reset_samples,
    )
    return tuple(
        (
            enrollment.anchor_id,
            enrollment.opportunity_start_sample,
            enrollment.anchor_emit_sample,
        )
        for enrollment in result.enrollments
    )


def annotate_causal_episodes(
    *,
    session: CausalSessionResult,
    intervals: Sequence[ActivityInterval],
    cells: Sequence[PosteriorCell],
    slot_ids: Sequence[str],
    scored_start_sample: int,
    gt_enrollment_samples: int,
    silence_reset_samples: int,
    oracle_reference: GTSessionResult | None = None,
    oracle_slots: dict[str, int] | None = None,
) -> tuple[AnnotatedCausalEpisode, ...]:
    result: list[AnnotatedCausalEpisode] = []
    window_start = scored_start_sample
    for episode in session.episodes:
        opportunity = first_gt_singleton_opportunity(
            intervals,
            start_sample=window_start,
            end_sample=max(episode.anchor_emit_sample, window_start + 1),
            enrollment_samples=gt_enrollment_samples,
            silence_reset_samples=silence_reset_samples,
        )
        expected_speaker = opportunity[0] if opportunity is not None else None
        opportunity_start = opportunity[1] if opportunity is not None else None
        oracle_slot: int | None = None
        if expected_speaker is not None:
            reference_episode = (
                next(
                    (
                        value
                        for value in oracle_reference.episodes
                        if value.anchor_speaker == expected_speaker
                        and value.opportunity_start_sample
                        <= (opportunity_start or window_start)
                        < value.end_emit_sample
                    ),
                    None,
                )
                if oracle_reference is not None
                else None
            )
            if reference_episode is not None and oracle_slots is not None:
                oracle_slot = oracle_slots.get(reference_episode.episode_id)
            else:
                evaluation_episode = AnchorEpisode(
                    episode_id=episode.episode_id,
                    source_id=session.source_id,
                    anchor_speaker=expected_speaker,
                    opportunity_start_sample=opportunity_start or window_start,
                    anchor_emit_sample=opportunity_start or window_start,
                    end_emit_sample=max(episode.end_emit_sample, episode.anchor_emit_sample + 1),
                    replacement_boundary_sample=None,
                )
                try:
                    oracle_slot = oracle_anchor_mapping(
                        evaluation_episode, cells, slot_ids
                    ).slot_index
                except ValueError:
                    oracle_slot = None
        result.append(
            AnnotatedCausalEpisode(
                episode=episode,
                expected_anchor_speaker=expected_speaker,
                opportunity_start_sample=opportunity_start,
                oracle_slot_index=oracle_slot,
                correct_anchor=oracle_slot is not None and oracle_slot == episode.anchor_slot_index,
            )
        )
        window_start = episode.end_emit_sample
    return tuple(result)


def annotate_causal_with_gt_reference(
    *,
    session: CausalSessionResult,
    gt_reference: GTSessionResult,
    cells: Sequence[PosteriorCell],
    slot_ids: Sequence[str],
    gt_oracle_slots: dict[str, int] | None = None,
) -> tuple[AnnotatedCausalEpisode, ...]:
    result: list[AnnotatedCausalEpisode] = []
    for episode in session.episodes:
        gt_episode = next(
            (
                value
                for value in gt_reference.episodes
                if value.opportunity_start_sample
                <= episode.candidate_start_sample
                < value.end_emit_sample
                or value.opportunity_start_sample
                <= episode.anchor_emit_sample
                < value.end_emit_sample
            ),
            None,
        )
        expected_speaker = gt_episode.anchor_speaker if gt_episode is not None else None
        opportunity_start = gt_episode.opportunity_start_sample if gt_episode is not None else None
        oracle_slot: int | None = None
        if expected_speaker is not None:
            if gt_episode is not None and gt_oracle_slots is not None:
                oracle_slot = gt_oracle_slots.get(gt_episode.episode_id)
            else:
                evaluation_episode = AnchorEpisode(
                    episode_id=episode.episode_id,
                    source_id=session.source_id,
                    anchor_speaker=expected_speaker,
                    opportunity_start_sample=opportunity_start or episode.candidate_start_sample,
                    anchor_emit_sample=opportunity_start or episode.candidate_start_sample,
                    end_emit_sample=max(episode.end_emit_sample, episode.anchor_emit_sample + 1),
                    replacement_boundary_sample=None,
                )
                try:
                    oracle_slot = oracle_anchor_mapping(
                        evaluation_episode, cells, slot_ids
                    ).slot_index
                except ValueError:
                    oracle_slot = None
        result.append(
            AnnotatedCausalEpisode(
                episode=episode,
                expected_anchor_speaker=expected_speaker,
                opportunity_start_sample=opportunity_start,
                oracle_slot_index=oracle_slot,
                correct_anchor=oracle_slot is not None and oracle_slot == episode.anchor_slot_index,
            )
        )
    return tuple(result)


def count_causal_opportunities(
    *,
    session: CausalSessionResult,
    intervals: Sequence[ActivityInterval],
    scored_start_sample: int,
    scored_end_sample: int,
    enrollment_samples: int,
    silence_reset_samples: int,
) -> int:
    count = 0
    window_start = scored_start_sample
    for episode in session.episodes:
        window_end = min(max(episode.anchor_emit_sample, window_start + 1), scored_end_sample)
        if window_end > window_start and first_gt_singleton_opportunity(
            intervals,
            start_sample=window_start,
            end_sample=window_end,
            enrollment_samples=enrollment_samples,
            silence_reset_samples=silence_reset_samples,
        ) is not None:
            count += 1
        window_start = min(max(episode.end_emit_sample, window_start), scored_end_sample)
    if window_start < scored_end_sample and first_gt_singleton_opportunity(
        intervals,
        start_sample=window_start,
        end_sample=scored_end_sample,
        enrollment_samples=enrollment_samples,
        silence_reset_samples=silence_reset_samples,
    ) is not None:
        count += 1
    return count


def causal_anchor_metrics(
    annotated: Sequence[AnnotatedCausalEpisode],
    expected_opportunity_count: int,
) -> dict[str, Any]:
    delays = [
        (value.episode.anchor_emit_sample - value.opportunity_start_sample) * 1000.0 / 16000.0
        for value in annotated
        if value.opportunity_start_sample is not None
    ]
    wrong_count = sum(not value.correct_anchor for value in annotated)
    total_enrollment_count = len(annotated)
    enrollment_count = sum(value.opportunity_start_sample is not None for value in annotated)
    failure_count = max(expected_opportunity_count - enrollment_count, 0)
    return {
        "expected_opportunity_count": expected_opportunity_count,
        "enrollment_count": enrollment_count,
        "wrong_anchor_count": wrong_count,
        "total_enrollment_count": total_enrollment_count,
        "unmatched_enrollment_count": total_enrollment_count - enrollment_count,
        "wrong_anchor_rate": (
            wrong_count / total_enrollment_count if total_enrollment_count else 1.0
        ),
        "enrollment_failure_count": failure_count,
        "enrollment_failure_rate": (
            failure_count / expected_opportunity_count if expected_opportunity_count else 0.0
        ),
        "no_anchor_timeout_rate": (
            failure_count / expected_opportunity_count if expected_opportunity_count else 0.0
        ),
        "enrollment_delay_ms": {
            "p50": percentile(delays, 50),
            "p90": percentile(delays, 90),
        },
        "enrollment_delay_values_ms": delays,
        "fraction_enrolled_within_1000ms": (
            sum(value <= 1000.0 for value in delays) / expected_opportunity_count
            if expected_opportunity_count
            else 0.0
        ),
        "fraction_enrolled_within_1500ms": (
            sum(value <= 1500.0 for value in delays) / expected_opportunity_count
            if expected_opportunity_count
            else 0.0
        ),
    }


def expected_outer_opportunity_count(
    row: dict[str, Any],
    *,
    enrollment_samples: int,
    silence_reset_samples: int,
) -> int:
    intervals = intervals_from_manifest(row)
    result = simulate_gt_session(
        source_id=str(row["source_id"]),
        intervals=intervals,
        confirmation_samples=int(row["scored_end_sample"]) + 1,
        enrollment_samples=enrollment_samples,
        silence_reset_samples=silence_reset_samples,
    )
    return len(result.enrollments)


def causal_primitive_records(
    *,
    source_id: str,
    annotated: Sequence[AnnotatedCausalEpisode],
    cells: Sequence[PosteriorCell],
) -> tuple[PrimitiveRecord, ...]:
    records: list[PrimitiveRecord] = []
    for annotated_episode in annotated:
        episode = annotated_episode.episode
        anchor_speaker = annotated_episode.expected_anchor_speaker
        if anchor_speaker is None:
            continue
        start_index = bisect_left(
            cells,
            episode.anchor_emit_sample,
            key=lambda value: value.cell.center_sample,
        )
        end_index = bisect_left(
            cells,
            episode.end_emit_sample,
            key=lambda value: value.cell.center_sample,
        )
        for posterior in cells[start_index:end_index]:
            cell = posterior.cell
            if cell.masked:
                continue
            probabilities = relative_probabilities(posterior, episode.anchor_slot_index)
            if probabilities is None:
                continue
            p_anchor, p_other = probabilities
            records.append(
                PrimitiveRecord(
                    source_id=source_id,
                    anchor_episode_id=episode.episode_id,
                    center_sample=cell.center_sample,
                    weight_samples=cell.duration_samples,
                    anchor_label=anchor_speaker in cell.active_speakers,
                    other_label=any(speaker != anchor_speaker for speaker in cell.active_speakers),
                    anchor_score=p_anchor,
                    other_score=p_other,
                    evidence_frontier_sample=posterior.evidence_frontier_sample,
                )
            )
    return tuple(records)


def causal_product_metrics(
    *,
    session: CausalSessionResult,
    annotated: Sequence[AnnotatedCausalEpisode],
    reference: GTSessionResult,
    intervals: Sequence[ActivityInterval],
    tolerance_samples: int,
    expected_opportunity_count: int,
) -> dict[str, Any]:
    contamination_episodes = [
        (
            value.expected_anchor_speaker,
            value.episode.anchor_emit_sample,
            value.episode.end_emit_sample,
        )
        for value in annotated
        if value.expected_anchor_speaker is not None
    ]
    product = product_event_metrics(
        predicted_events=session.replacement_events,
        reference=reference,
        intervals=intervals,
        contamination_episodes=[
            (str(anchor), start, end) for anchor, start, end in contamination_episodes
        ],
        tolerance_samples=tolerance_samples,
    )
    anchor = causal_anchor_metrics(annotated, expected_opportunity_count)
    boundary_ordered_events = sorted(
        session.replacement_events, key=lambda value: value.boundary_source_sample
    )
    matched_episode_ids = {
        boundary_ordered_events[int(value["predicted_index"])].anchor_episode_id
        for value in product["matches"]
    }
    wrong_false_cuts = 0
    cascade_lengths: list[int] = []
    active_cascade = 0
    events_by_episode = {
        value.anchor_episode_id: value for value in session.replacement_events
    }
    for annotation in sorted(annotated, key=lambda value: value.episode.anchor_emit_sample):
        event = events_by_episode.get(annotation.episode.episode_id)
        if (
            event is not None
            and not annotation.correct_anchor
            and event.anchor_episode_id not in matched_episode_ids
        ):
            wrong_false_cuts += 1
            active_cascade += 1
        elif active_cascade:
            cascade_lengths.append(active_cascade)
            active_cascade = 0
    if active_cascade:
        cascade_lengths.append(active_cascade)
    uncertain_samples = sum(
        span.end_sample - span.start_sample
        for span in session.timeline
        if span.lifecycle is AnchorLifecycle.ANCHOR_UNCERTAIN
    )
    scored_samples = sum(span.end_sample - span.start_sample for span in session.timeline)
    exposure = timeline_exposure(session.timeline)
    slot_loss_count = sum(
        value.episode.end_reason == "slot_continuity_invalid" for value in annotated
    )
    return {
        **product,
        **anchor,
        "slot_loss_count": slot_loss_count,
        "slot_loss_rate": slot_loss_count / len(annotated) if annotated else 0.0,
        "anchor_uncertain_seconds": uncertain_samples / 16000.0,
        "scored_seconds": scored_samples / 16000.0,
        "anchor_uncertain_time_fraction": (
            uncertain_samples / scored_samples if scored_samples else 0.0
        ),
        "masked_seconds": exposure["masked_seconds"],
        "masked_active_speech_seconds": exposure["masked_active_speech_seconds"],
        "unanchored_active_speech_seconds": exposure["unanchored_active_speech_seconds"],
        "anchor_uncertain_active_speech_seconds": exposure[
            "anchor_uncertain_active_speech_seconds"
        ],
        "fail_closed_unknown_active_speech_seconds": exposure[
            "fail_closed_unknown_active_speech_seconds"
        ],
        "exclusive_other_contamination_upper_bound_seconds": (
            product["exclusive_other_contamination_seconds"]
            + exposure["unanchored_active_speech_seconds"]
            + exposure["anchor_uncertain_active_speech_seconds"]
        ),
        "false_cuts_after_wrong_anchor": wrong_false_cuts,
        "anchor_error_cascade_length": {
            "maximum": max(cascade_lengths, default=0),
            "p50": percentile(cascade_lengths, 50),
            "p90": percentile(cascade_lengths, 90),
            "distribution": dict(sorted(Counter(cascade_lengths).items())),
        },
    }
