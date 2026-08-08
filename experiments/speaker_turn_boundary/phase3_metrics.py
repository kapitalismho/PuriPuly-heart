from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2

from experiments.speaker_turn_boundary.events import SpeakerBoundaryEvent
from experiments.speaker_turn_boundary.ground_truth import SpeakerChangeGT
from experiments.speaker_turn_boundary.metrics import RECALL_DEADLINES_MS

SAMPLE_RATE_HZ = 16000
LOCALIZATION_TOLERANCE_MS = 500
ATTRIBUTION_HORIZON_MS = 2000
REFERENCE_FALSE_CUT_RATES_PER_SOURCE_HOUR = (0.5, 1.0, 2.0, 5.0)
REFERENCE_EXTRA_FALSE_CUT_COUNTS = (0, 1, 2, 3, 5, 10, 20, 50)
PRIMARY_DEADLINE_MS = 500
PHASE3_METRIC_SCHEMA = "experiments.speaker_turn_boundary.phase3.metrics.v2"


class Phase3MetricError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ObservedCut:
    event: SpeakerBoundaryEvent
    kind: str
    source_index: int

    @property
    def audio_epoch(self) -> int:
        return self.event.audio_epoch

    @property
    def boundary_sample(self) -> int:
        return self.event.boundary_source_sample

    @property
    def observed_sample(self) -> int:
        return self.event.observed_source_sample_at_emit


@dataclass(frozen=True, slots=True)
class CausalMatch:
    gt_index: int
    cut_index: int
    change: SpeakerChangeGT
    cut: ObservedCut
    observation_delay_samples: int
    localization_error_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "gt_index": self.gt_index,
            "cut_index": self.cut_index,
            "audio_epoch": self.change.audio_epoch,
            "change_sample": self.change.change_sample,
            "boundary_sample": self.cut.boundary_sample,
            "observed_sample": self.cut.observed_sample,
            "observation_delay_ms": self.observation_delay_samples / 16.0,
            "localization_error_ms": self.localization_error_samples / 16.0,
            "cut_kind": self.cut.kind,
            "cut_source": self.cut.event.source,
            "source_index": self.cut.source_index,
        }


def events_as_cuts(
    events: Iterable[SpeakerBoundaryEvent],
    *,
    kind: str,
    source_indices: Iterable[int] | None = None,
) -> list[ObservedCut]:
    event_list = list(events)
    indices = list(source_indices) if source_indices is not None else list(range(len(event_list)))
    if len(indices) != len(event_list):
        raise Phase3MetricError("source index count does not match event count")
    return [
        ObservedCut(event=event, kind=kind, source_index=index)
        for event, index in zip(event_list, indices)
    ]


def _eligible(
    change: SpeakerChangeGT,
    cut: ObservedCut,
    *,
    deadline_samples: int,
    localization_samples: int,
) -> bool:
    if change.audio_epoch != cut.audio_epoch:
        return False
    delay = cut.observed_sample - change.change_sample
    if delay < 0 or delay > deadline_samples:
        return False
    return abs(cut.boundary_sample - change.change_sample) <= localization_samples


def causal_ordered_match(
    gt_changes: list[SpeakerChangeGT],
    cuts: list[ObservedCut],
    *,
    deadline_ms: int,
    localization_tolerance_ms: int = LOCALIZATION_TOLERANCE_MS,
) -> list[CausalMatch]:
    deadline_samples = deadline_ms * 16
    localization_samples = localization_tolerance_ms * 16
    matches: list[CausalMatch] = []
    epochs = sorted(
        {change.audio_epoch for change in gt_changes} | {cut.audio_epoch for cut in cuts}
    )
    for epoch in epochs:
        epoch_changes = sorted(
            [
                (index, change)
                for index, change in enumerate(gt_changes)
                if change.audio_epoch == epoch
            ],
            key=lambda item: (item[1].change_sample, item[0]),
        )
        epoch_cuts = sorted(
            [(index, cut) for index, cut in enumerate(cuts) if cut.audio_epoch == epoch],
            key=lambda item: (
                item[1].boundary_sample,
                item[1].observed_sample,
                item[1].kind,
                item[1].source_index,
                item[0],
            ),
        )
        matches.extend(
            _match_epoch(
                epoch_changes,
                epoch_cuts,
                deadline_samples=deadline_samples,
                localization_samples=localization_samples,
            )
        )
    return sorted(matches, key=lambda match: (match.gt_index, match.cut_index))


def _match_epoch(
    changes: list[tuple[int, SpeakerChangeGT]],
    cuts: list[tuple[int, ObservedCut]],
    *,
    deadline_samples: int,
    localization_samples: int,
) -> list[CausalMatch]:
    change_count = len(changes)
    cut_count = len(cuts)
    scores: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(cut_count + 1)] for _ in range(change_count + 1)
    ]
    actions: list[list[str]] = [["" for _ in range(cut_count + 1)] for _ in range(change_count + 1)]
    for change_pos in range(1, change_count + 1):
        actions[change_pos][0] = "skip_change"
    for cut_pos in range(1, cut_count + 1):
        actions[0][cut_pos] = "skip_cut"
    action_priority = {"skip_change": 0, "skip_cut": 1, "match": 2}
    for change_pos in range(1, change_count + 1):
        _, change = changes[change_pos - 1]
        for cut_pos in range(1, cut_count + 1):
            _, cut = cuts[cut_pos - 1]
            options: list[tuple[tuple[int, int, int, int], str]] = [
                (scores[change_pos - 1][cut_pos], "skip_change"),
                (scores[change_pos][cut_pos - 1], "skip_cut"),
            ]
            if _eligible(
                change,
                cut,
                deadline_samples=deadline_samples,
                localization_samples=localization_samples,
            ):
                previous = scores[change_pos - 1][cut_pos - 1]
                delay = cut.observed_sample - change.change_sample
                localization = abs(cut.boundary_sample - change.change_sample)
                score = (
                    previous[0] + 1,
                    previous[1] - delay,
                    previous[2] - localization,
                    previous[3] - cut_pos,
                )
                options.append((score, "match"))
            best_score, best_action = max(
                options,
                key=lambda item: (item[0], action_priority[item[1]]),
            )
            scores[change_pos][cut_pos] = best_score
            actions[change_pos][cut_pos] = best_action
    change_pos = change_count
    cut_pos = cut_count
    matched: list[CausalMatch] = []
    while change_pos > 0 or cut_pos > 0:
        action = actions[change_pos][cut_pos]
        if action == "match":
            gt_index, change = changes[change_pos - 1]
            cut_index, cut = cuts[cut_pos - 1]
            matched.append(
                CausalMatch(
                    gt_index=gt_index,
                    cut_index=cut_index,
                    change=change,
                    cut=cut,
                    observation_delay_samples=cut.observed_sample - change.change_sample,
                    localization_error_samples=cut.boundary_sample - change.change_sample,
                )
            )
            change_pos -= 1
            cut_pos -= 1
        elif action == "skip_cut":
            cut_pos -= 1
        elif action == "skip_change":
            change_pos -= 1
        else:
            raise Phase3MetricError("matching reconstruction reached an invalid state")
    matched.reverse()
    return matched


def causal_series(
    gt_changes: list[SpeakerChangeGT],
    cuts: list[ObservedCut],
) -> dict[int, list[CausalMatch]]:
    return {
        deadline: causal_ordered_match(
            gt_changes,
            cuts,
            deadline_ms=deadline,
        )
        for deadline in RECALL_DEADLINES_MS
    }


def locked_product_series(
    gt_changes: list[SpeakerChangeGT],
    b0_series: dict[int, list[CausalMatch]],
    added_detector_cuts: list[ObservedCut],
) -> dict[int, list[CausalMatch]]:
    product: dict[int, list[CausalMatch]] = {}
    for deadline in RECALL_DEADLINES_MS:
        baseline_matches = b0_series[deadline]
        matched_gt = {match.gt_index for match in baseline_matches}
        remaining_positions = [index for index in range(len(gt_changes)) if index not in matched_gt]
        remaining_changes = [gt_changes[index] for index in remaining_positions]
        added_matches = causal_ordered_match(
            remaining_changes,
            added_detector_cuts,
            deadline_ms=deadline,
        )
        remapped = [
            CausalMatch(
                gt_index=remaining_positions[match.gt_index],
                cut_index=match.cut_index,
                change=match.change,
                cut=match.cut,
                observation_delay_samples=match.observation_delay_samples,
                localization_error_samples=match.localization_error_samples,
            )
            for match in added_matches
        ]
        product[deadline] = sorted(
            [*baseline_matches, *remapped],
            key=lambda match: match.gt_index,
        )
    return product


def matched_gt_indices(series: dict[int, list[CausalMatch]]) -> dict[int, set[int]]:
    return {deadline: {match.gt_index for match in matches} for deadline, matches in series.items()}


def validate_series(
    gt_changes: list[SpeakerChangeGT],
    cuts: list[ObservedCut],
    series: dict[int, list[CausalMatch]],
) -> None:
    previous_count = -1
    for deadline in RECALL_DEADLINES_MS:
        matches = series[deadline]
        if len(matches) < previous_count:
            raise Phase3MetricError("causal recall count decreased at a later deadline")
        previous_count = len(matches)
        gt_indices = [match.gt_index for match in matches]
        cut_indices = [match.cut_index for match in matches]
        if len(gt_indices) != len(set(gt_indices)) or len(cut_indices) != len(set(cut_indices)):
            raise Phase3MetricError("causal matching is not one-to-one")
        for match in matches:
            if match.observation_delay_samples < 0:
                raise Phase3MetricError("negative causal observation delay")
            if match.observation_delay_samples > deadline * 16:
                raise Phase3MetricError("match exceeds its causal deadline")
            if abs(match.localization_error_samples) > LOCALIZATION_TOLERANCE_MS * 16:
                raise Phase3MetricError("match exceeds localization tolerance")
            if match.change is not gt_changes[match.gt_index]:
                raise Phase3MetricError("match ground-truth index is inconsistent")
            if match.cut is not cuts[match.cut_index]:
                raise Phase3MetricError("match cut index is inconsistent")


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    if confidence != 0.95:
        raise Phase3MetricError("only the pinned 95% Wilson interval is supported")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def poisson_count_interval(count: int, confidence: float = 0.95) -> list[float]:
    if count < 0:
        raise Phase3MetricError("Poisson count cannot be negative")
    if confidence != 0.95:
        raise Phase3MetricError("only the pinned 95% Poisson interval is supported")
    alpha = 1.0 - confidence
    lower = 0.0 if count == 0 else 0.5 * float(chi2.ppf(alpha / 2.0, 2 * count))
    upper = 0.5 * float(chi2.ppf(1.0 - alpha / 2.0, 2 * (count + 1)))
    return [lower, upper]


def rate_with_interval(count: int, exposure_hours: float) -> dict[str, Any]:
    interval = poisson_count_interval(count)
    if exposure_hours <= 0.0:
        value = None if count == 0 else float("inf")
        bounds: list[float | None] = [None, None]
    else:
        value = count / exposure_hours
        bounds = [interval[0] / exposure_hours, interval[1] / exposure_hours]
    return {
        "count": count,
        "exposure_hours": exposure_hours,
        "rate": value,
        "poisson_95": bounds,
    }


def recall_summary(series: dict[int, list[CausalMatch]], gt_count: int) -> dict[str, Any]:
    return {
        str(deadline): {
            "matched": len(series[deadline]),
            "total": gt_count,
            "recall": len(series[deadline]) / gt_count if gt_count else 0.0,
            "wilson_95": wilson_interval(len(series[deadline]), gt_count),
        }
        for deadline in RECALL_DEADLINES_MS
    }


def percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def timing_summary(matches: Iterable[CausalMatch]) -> dict[str, Any]:
    match_list = list(matches)
    localization = [match.localization_error_samples / 16.0 for match in match_list]
    causal = [match.observation_delay_samples / 16.0 for match in match_list]
    lookback = [match.cut.event.event_lookback_ms for match in match_list]
    late = [value for value in localization if value > 0.0]
    if any(value < 0.0 for value in causal):
        raise Phase3MetricError("negative causal delay reached timing summary")
    return {
        "matched_count": len(match_list),
        "localization_error_ms": percentile_summary(localization),
        "causal_audio_delay_ms": percentile_summary(causal),
        "event_lookback_ms": percentile_summary(lookback),
        "late_cut_leakage_ms": percentile_summary(late),
    }
