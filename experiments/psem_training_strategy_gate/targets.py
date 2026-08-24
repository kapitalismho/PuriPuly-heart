from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from experiments.psem_training_strategy_gate.data.label_contract import LabelResult

SAMPLE_RATE_HZ = 16000
HOP_SAMPLES = 1600
PAST_SAMPLES = 32000
FUTURE_SAMPLES = 16000
WINDOW_SAMPLES = 48000
CELL_COUNT = 30
STATE_TO_INDEX = {"silence": 0, "singleton": 1, "overlap": 2}
LABEL_CONTRACT_VERSION = "psem-handoff-v1"
LABEL_CONTRACT_SHA256 = "3915ab5d6fe3c8e2eb0933ce619f425eb9a08cf6bc3a46eacf370647956772d2"
ADJACENT_SOLO_FAMILY = "adjacent_reliable_solo_cells"
SILENCE_GAP_FAMILY = "reliable_solo_endpoints_separated_only_by_valid_local_silence"
OVERLAP_BRIDGE_FAMILY = "reliable_solo_before_overlap_to_reliable_solo_after_overlap"


class TargetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RelationPair:
    left_cell: int
    right_cell: int
    target: int
    family: str
    transition_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_cell": self.left_cell,
            "right_cell": self.right_cell,
            "target": self.target,
            "family": self.family,
            "transition_id": self.transition_id,
        }


@dataclass(frozen=True, slots=True)
class WindowTargets:
    source_id: str
    boundary_sample: int
    window_start_sample: int
    window_end_sample: int
    observed_frontier_sample: int
    cell_centers_sample: tuple[int, ...]
    handoff_target: int
    handoff_mask: bool
    handoff_event_samples: tuple[int, ...]
    state_targets: tuple[int, ...]
    state_mask: tuple[bool, ...]
    cell_speakers: tuple[tuple[str, ...], ...]
    relation_pairs: tuple[RelationPair, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "boundary_sample": self.boundary_sample,
            "window_start_sample": self.window_start_sample,
            "window_end_sample": self.window_end_sample,
            "observed_frontier_sample": self.observed_frontier_sample,
            "cell_centers_sample": list(self.cell_centers_sample),
            "handoff_target": self.handoff_target,
            "handoff_mask": self.handoff_mask,
            "handoff_event_samples": list(self.handoff_event_samples),
            "state_targets": list(self.state_targets),
            "state_mask": list(self.state_mask),
            "cell_speakers": [list(value) for value in self.cell_speakers],
            "relation_pairs": [pair.to_dict() for pair in self.relation_pairs],
        }


def nearest_grid_sample(source_sample: int, hop_samples: int = HOP_SAMPLES) -> int:
    if isinstance(source_sample, bool) or not isinstance(source_sample, int):
        raise TargetError("source sample must be an integer")
    if isinstance(hop_samples, bool) or not isinstance(hop_samples, int) or hop_samples <= 0:
        raise TargetError("hop samples must be a positive integer")
    quotient, remainder = divmod(source_sample, hop_samples)
    if remainder <= hop_samples // 2:
        return quotient * hop_samples
    return (quotient + 1) * hop_samples


def valid_center_samples(
    scored_start_sample: int,
    scored_end_sample: int,
) -> range:
    if (
        isinstance(scored_start_sample, bool)
        or not isinstance(scored_start_sample, int)
        or isinstance(scored_end_sample, bool)
        or not isinstance(scored_end_sample, int)
        or scored_start_sample < 0
        or scored_end_sample <= scored_start_sample
    ):
        raise TargetError("scored source range is invalid")
    first = ((scored_start_sample + PAST_SAMPLES + HOP_SAMPLES - 1) // HOP_SAMPLES) * HOP_SAMPLES
    final = scored_end_sample - FUTURE_SAMPLES
    if first > final:
        return range(0)
    return range(first, final + 1, HOP_SAMPLES)


def _interval_indices(labels: LabelResult, samples: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    index = 0
    intervals = labels.intervals
    for sample in samples:
        while index < len(intervals) and sample >= intervals[index].end_sample:
            index += 1
        if (
            index >= len(intervals)
            or sample < intervals[index].start_sample
            or sample >= intervals[index].end_sample
        ):
            raise TargetError("cell center lies outside the canonical source timeline")
        result.append(index)
    return tuple(result)


def _reliable_solo(labels: LabelResult, interval_index: int) -> bool:
    interval = labels.intervals[interval_index]
    return (
        not interval.ambiguous
        and interval.speaker_identity_known
        and not interval.handoff_relation_mask_classes
        and len(interval.active_speakers) == 1
        and interval.duration_samples >= 3200
    )


def _cell_for_interval(
    interval_index: int,
    interval_indices: Sequence[int],
    *,
    first: bool,
) -> int | None:
    matches = [index for index, value in enumerate(interval_indices) if value == interval_index]
    if not matches:
        return None
    return matches[0] if first else matches[-1]


def _relation_pairs(
    labels: LabelResult,
    interval_indices: Sequence[int],
    cell_speakers: Sequence[tuple[str, ...]],
) -> tuple[RelationPair, ...]:
    pairs: dict[tuple[int, int, str], RelationPair] = {}
    for left in range(CELL_COUNT - 1):
        right = left + 1
        left_interval = interval_indices[left]
        right_interval = interval_indices[right]
        if not all(
            _reliable_solo(labels, interval_index)
            for interval_index in range(left_interval, right_interval + 1)
        ):
            continue
        target = int(cell_speakers[left][0] != cell_speakers[right][0])
        key = (left, right, ADJACENT_SOLO_FAMILY)
        pairs[key] = RelationPair(left, right, target, ADJACENT_SOLO_FAMILY)
    for transition in labels.transitions:
        relation = transition.get("relation_target")
        previous_index = transition.get("from_interval_index")
        current_index = transition.get("to_interval_index")
        if (
            transition.get("mask_state") != "valid"
            or relation not in {"same", "different"}
            or not isinstance(previous_index, int)
            or not isinstance(current_index, int)
        ):
            continue
        gap_samples = transition.get("gap_samples")
        overlap_samples = transition.get("overlap_samples")
        if isinstance(gap_samples, int) and gap_samples > 0 and overlap_samples == 0:
            family = SILENCE_GAP_FAMILY
        elif isinstance(overlap_samples, int) and overlap_samples > 0:
            family = OVERLAP_BRIDGE_FAMILY
        else:
            continue
        left = _cell_for_interval(previous_index, interval_indices, first=False)
        right = _cell_for_interval(current_index, interval_indices, first=True)
        if left is None or right is None or left >= right:
            continue
        pair = RelationPair(
            left,
            right,
            int(relation == "different"),
            family,
            str(transition["transition_id"]),
        )
        pairs[(left, right, family)] = pair
    return tuple(pairs[key] for key in sorted(pairs))


def _masked_transition_centers(labels: LabelResult) -> set[int]:
    centers: set[int] = set()
    for transition in labels.transitions:
        if transition.get("mask_state") != "masked":
            continue
        current_index = transition.get("to_interval_index")
        if isinstance(current_index, int):
            centers.add(nearest_grid_sample(labels.intervals[current_index].start_sample))
            continue
        if transition.get("primary_topology") != "overlap_to_silence_unresolved":
            continue
        previous_index = transition.get("from_interval_index")
        if not isinstance(previous_index, int):
            continue
        saw_overlap = False
        for interval in labels.intervals[previous_index + 1 :]:
            if interval.ambiguous:
                continue
            if len(interval.active_speakers) >= 2:
                saw_overlap = True
                continue
            if saw_overlap and not interval.active_speakers:
                centers.add(nearest_grid_sample(interval.start_sample))
                break
    return centers


def _validate_labels(labels: LabelResult) -> None:
    observed = (
        labels.contract_version,
        labels.contract_document_sha256,
        labels.sample_rate_hz,
    )
    expected = (LABEL_CONTRACT_VERSION, LABEL_CONTRACT_SHA256, SAMPLE_RATE_HZ)
    if observed != expected:
        raise TargetError("labels do not match the pinned psem-handoff-v1 contract")
    if not labels.intervals or len(labels.activity_labels) != len(labels.intervals):
        raise TargetError("labels do not contain a complete canonical timeline")


def build_window_targets(
    source_id: str,
    labels: LabelResult,
    boundary_sample: int,
) -> WindowTargets:
    if not source_id:
        raise TargetError("source id is required")
    _validate_labels(labels)
    if (
        isinstance(boundary_sample, bool)
        or not isinstance(boundary_sample, int)
        or boundary_sample % HOP_SAMPLES
    ):
        raise TargetError("boundary must lie on the 100 ms source grid")
    scored_start = labels.intervals[0].start_sample
    scored_end = labels.intervals[-1].end_sample
    if boundary_sample not in valid_center_samples(scored_start, scored_end):
        raise TargetError("boundary does not admit the fixed three-second evidence window")
    window_start = boundary_sample - PAST_SAMPLES
    window_end = boundary_sample + FUTURE_SAMPLES
    cell_centers = tuple(
        window_start + index * HOP_SAMPLES + HOP_SAMPLES // 2 for index in range(CELL_COUNT)
    )
    interval_indices = _interval_indices(labels, cell_centers)
    state_targets: list[int] = []
    state_mask: list[bool] = []
    cell_speakers: list[tuple[str, ...]] = []
    for interval_index in interval_indices:
        activity = labels.activity_labels[interval_index]
        state = activity["state"]
        valid = state in STATE_TO_INDEX and activity["mask_state"] == "valid"
        state_targets.append(STATE_TO_INDEX.get(state, -1))
        state_mask.append(valid)
        cell_speakers.append(tuple(activity["active_speakers"]))
    positive_events = tuple(
        sorted(
            int(transition["handoff_source_sample"])
            for transition in labels.transitions
            if transition.get("mask_state") == "valid"
            and transition.get("handoff_confirmed") == 1
            and isinstance(transition.get("handoff_source_sample"), int)
            and nearest_grid_sample(int(transition["handoff_source_sample"])) == boundary_sample
        )
    )
    masked_centers = _masked_transition_centers(labels)
    boundary_index = _interval_indices(labels, (boundary_sample,))[0]
    boundary_interval = labels.intervals[boundary_index]
    boundary_ambiguous = (
        boundary_interval.ambiguous
        or not boundary_interval.speaker_identity_known
        or bool(boundary_interval.handoff_relation_mask_classes)
        or len(boundary_interval.active_speakers) >= 3
    )
    handoff_mask = boundary_sample not in masked_centers and not boundary_ambiguous
    return WindowTargets(
        source_id=source_id,
        boundary_sample=boundary_sample,
        window_start_sample=window_start,
        window_end_sample=window_end,
        observed_frontier_sample=window_end,
        cell_centers_sample=cell_centers,
        handoff_target=int(bool(positive_events)),
        handoff_mask=handoff_mask,
        handoff_event_samples=positive_events,
        state_targets=tuple(state_targets),
        state_mask=tuple(state_mask),
        cell_speakers=tuple(cell_speakers),
        relation_pairs=_relation_pairs(labels, interval_indices, cell_speakers),
    )
