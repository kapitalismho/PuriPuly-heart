from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(canonical_json(dict(row)) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8", newline="\n")


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return None if not array.size else float(np.percentile(array, q))


def path_has_alias(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink() or (
            hasattr(os.path, "isjunction") and os.path.isjunction(current)
        ):
            return True
    return False


def strict_regular_file(path: Path, field: str) -> Path:
    if path_has_alias(path) or not path.is_file():
        raise ValueError(f"{field} must be a regular non-aliased file")
    return path.resolve()


def weighted_average_precision(
    labels: Iterable[bool | int], scores: Iterable[float], weights: Iterable[float]
) -> float | None:
    truth = np.asarray(list(labels), dtype=np.bool_)
    probability = np.asarray(list(scores), dtype=np.float64)
    weight = np.asarray(list(weights), dtype=np.float64)
    if truth.ndim != 1 or probability.shape != truth.shape or weight.shape != truth.shape:
        raise ValueError("metric vectors must have identical one-dimensional geometry")
    positive_weight = float(weight[truth].sum())
    if positive_weight == 0.0:
        return None
    order = np.argsort(-probability, kind="stable")
    truth = truth[order]
    probability = probability[order]
    weight = weight[order]
    cumulative_tp = np.cumsum(weight * truth)
    cumulative_fp = np.cumsum(weight * ~truth)
    group_ends = np.flatnonzero(np.r_[probability[1:] != probability[:-1], True])
    true_positive = cumulative_tp[group_ends]
    false_positive = cumulative_fp[group_ends]
    recall = true_positive / positive_weight
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_positive) > 0,
    )
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


@dataclass(frozen=True, slots=True)
class ActivityInterval:
    start_sample: int
    end_sample: int
    active_speakers: tuple[str, ...]
    masked: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ActivityInterval:
        return cls(
            start_sample=int(value["start_sample"]),
            end_sample=int(value["end_sample"]),
            active_speakers=tuple(sorted(map(str, value["active_speakers"]))),
            masked=bool(value.get("masked", False)),
        )


@dataclass(frozen=True, slots=True)
class ReplacementEvent:
    source_id: str
    anchor_episode_id: str
    anchor_id: str
    boundary_source_sample: int
    model_evidence_frontier_sample: int
    decoder_emit_sample: int
    compute_lag_ms: float | None
    confirmation_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "anchor_episode_id": self.anchor_episode_id,
            "anchor_id": self.anchor_id,
            "boundary_source_sample": self.boundary_source_sample,
            "model_evidence_frontier_sample": self.model_evidence_frontier_sample,
            "decoder_emit_sample": self.decoder_emit_sample,
            "compute_lag_ms": self.compute_lag_ms,
            "confirmation_samples": self.confirmation_samples,
        }


@dataclass(frozen=True, slots=True)
class AnchorEpisode:
    episode_id: str
    source_id: str
    anchor_speaker: str
    opportunity_start_sample: int
    anchor_emit_sample: int
    end_emit_sample: int
    replacement_boundary_sample: int | None


@dataclass(frozen=True, slots=True)
class GTSessionResult:
    source_id: str
    confirmation_samples: int
    enrollment_samples: int
    silence_reset_samples: int
    events: tuple[ReplacementEvent, ...]
    episodes: tuple[AnchorEpisode, ...]


class _GTSimulator:
    def __init__(
        self,
        source_id: str,
        confirmation_samples: int,
        enrollment_samples: int,
        silence_reset_samples: int,
    ) -> None:
        self.source_id = source_id
        self.confirmation_samples = confirmation_samples
        self.enrollment_samples = enrollment_samples
        self.silence_reset_samples = silence_reset_samples
        self.anchor: str | None = None
        self.candidate: str | None = None
        self.candidate_start: int | None = None
        self.candidate_evidence = 0
        self.silence_evidence = 0
        self.pending_boundary: int | None = None
        self.pending_evidence = 0
        self.episode_counter = 0
        self.current_episode_id: str | None = None
        self.current_opportunity: int | None = None
        self.current_anchor_emit: int | None = None
        self.events: list[ReplacementEvent] = []
        self.episodes: list[AnchorEpisode] = []

    def clear_replacement(self) -> None:
        self.pending_boundary = None
        self.pending_evidence = 0

    def clear_candidate(self) -> None:
        self.candidate = None
        self.candidate_start = None
        self.candidate_evidence = 0

    def open_episode(self, speaker: str, opportunity: int, emit: int) -> None:
        self.episode_counter += 1
        self.current_episode_id = f"{self.source_id}:A{self.episode_counter:05d}"
        self.current_opportunity = opportunity
        self.current_anchor_emit = emit
        self.anchor = speaker

    def close_episode(self, end_sample: int, boundary: int | None) -> None:
        if (
            self.anchor is None
            or self.current_episode_id is None
            or self.current_opportunity is None
            or self.current_anchor_emit is None
        ):
            raise ValueError("anchor episode is incomplete")
        self.episodes.append(
            AnchorEpisode(
                episode_id=self.current_episode_id,
                source_id=self.source_id,
                anchor_speaker=self.anchor,
                opportunity_start_sample=self.current_opportunity,
                anchor_emit_sample=self.current_anchor_emit,
                end_emit_sample=end_sample,
                replacement_boundary_sample=boundary,
            )
        )
        self.current_episode_id = None
        self.current_opportunity = None
        self.current_anchor_emit = None

    def become_unanchored(self) -> None:
        self.anchor = None
        self.silence_evidence = 0
        self.clear_replacement()
        self.clear_candidate()

    def process(self, start: int, end: int, speakers: tuple[str, ...], masked: bool) -> None:
        if end <= start:
            return
        if masked:
            return
        if self.anchor is None:
            self.clear_replacement()
            self.silence_evidence = 0
            if len(speakers) != 1:
                self.clear_candidate()
                return
            speaker = speakers[0]
            if self.candidate != speaker:
                self.candidate = speaker
                self.candidate_start = start
                self.candidate_evidence = 0
            needed = self.enrollment_samples - self.candidate_evidence
            duration = end - start
            if duration < needed:
                self.candidate_evidence += duration
                return
            emit = start + needed
            if self.candidate_start is None:
                raise ValueError("enrollment opportunity is missing")
            self.open_episode(speaker, self.candidate_start, emit)
            self.clear_candidate()
            self.process(emit, end, speakers, False)
            return
        if not speakers:
            self.clear_replacement()
            needed = self.silence_reset_samples - self.silence_evidence
            duration = end - start
            if duration < needed:
                self.silence_evidence += duration
                return
            reset_sample = start + needed
            self.close_episode(reset_sample, None)
            self.become_unanchored()
            self.process(reset_sample, end, speakers, False)
            return
        self.silence_evidence = 0
        if self.anchor in speakers:
            self.clear_replacement()
            return
        if self.pending_boundary is None:
            self.pending_boundary = start
        needed = self.confirmation_samples - self.pending_evidence
        duration = end - start
        if duration < needed:
            self.pending_evidence += duration
            return
        qualifying = start + needed
        if self.current_episode_id is None or self.pending_boundary is None:
            raise ValueError("replacement episode is incomplete")
        event = ReplacementEvent(
            source_id=self.source_id,
            anchor_episode_id=self.current_episode_id,
            anchor_id=self.anchor,
            boundary_source_sample=self.pending_boundary,
            model_evidence_frontier_sample=qualifying,
            decoder_emit_sample=qualifying,
            compute_lag_ms=None,
            confirmation_samples=self.confirmation_samples,
        )
        self.events.append(event)
        self.close_episode(qualifying, event.boundary_source_sample)
        self.become_unanchored()
        self.process(qualifying, end, speakers, False)

    def finish(self, end_sample: int) -> None:
        if self.anchor is not None:
            self.close_episode(end_sample, None)


def intervals_from_manifest(value: dict[str, Any]) -> tuple[ActivityInterval, ...]:
    return tuple(ActivityInterval.from_dict(row) for row in value["intervals"])


def simulate_gt_session(
    value: dict[str, Any],
    *,
    replacement_confirmation_samples: int,
    enrollment_samples: int,
    silence_reset_samples: int,
) -> GTSessionResult:
    intervals = intervals_from_manifest(value)
    simulator = _GTSimulator(
        str(value["source_id"]),
        replacement_confirmation_samples,
        enrollment_samples,
        silence_reset_samples,
    )
    expected = intervals[0].start_sample
    for interval in intervals:
        if interval.start_sample != expected:
            raise ValueError("interval timeline must be contiguous")
        simulator.process(
            interval.start_sample,
            interval.end_sample,
            interval.active_speakers,
            interval.masked,
        )
        expected = interval.end_sample
    simulator.finish(intervals[-1].end_sample)
    return GTSessionResult(
        source_id=str(value["source_id"]),
        confirmation_samples=replacement_confirmation_samples,
        enrollment_samples=enrollment_samples,
        silence_reset_samples=silence_reset_samples,
        events=tuple(simulator.events),
        episodes=tuple(simulator.episodes),
    )


def monotonic_boundary_matches(
    predicted_samples: Sequence[int],
    reference_samples: Sequence[int],
    tolerance_samples: int,
) -> list[tuple[int, int]]:
    predicted = list(map(int, predicted_samples))
    reference = list(map(int, reference_samples))
    rows = len(predicted)
    columns = len(reference)
    counts = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    costs = np.zeros((rows + 1, columns + 1), dtype=np.int64)
    choices = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            candidates = [
                (int(counts[row - 1, column]), int(costs[row - 1, column]), 1),
                (int(counts[row, column - 1]), int(costs[row, column - 1]), 2),
            ]
            delta = abs(predicted[row - 1] - reference[column - 1])
            if delta <= tolerance_samples:
                candidates.append(
                    (
                        int(counts[row - 1, column - 1]) + 1,
                        int(costs[row - 1, column - 1]) + delta,
                        3,
                    )
                )
            best = min(candidates, key=lambda item: (-item[0], item[1], -item[2]))
            counts[row, column] = best[0]
            costs[row, column] = best[1]
            choices[row, column] = best[2]
    matches = []
    row = rows
    column = columns
    while row and column:
        choice = int(choices[row, column])
        if choice == 3:
            matches.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif choice == 1:
            row -= 1
        else:
            column -= 1
    matches.reverse()
    return matches


def exact_episode_contamination_samples(
    intervals: Sequence[ActivityInterval],
    *,
    anchor_speaker: str,
    start_sample: int,
    end_sample: int,
) -> int:
    return sum(
        max(0, min(end_sample, interval.end_sample) - max(start_sample, interval.start_sample))
        for interval in intervals
        if not interval.masked
        and anchor_speaker not in interval.active_speakers
        and interval.active_speakers
        and min(end_sample, interval.end_sample) > max(start_sample, interval.start_sample)
    )


def product_event_metrics(
    *,
    predicted_events: Sequence[ReplacementEvent],
    reference: GTSessionResult,
    intervals: Sequence[ActivityInterval],
    contamination_episodes: Sequence[tuple[str, int, int]],
    tolerance_samples: int,
) -> dict[str, Any]:
    predicted = sorted(predicted_events, key=lambda item: item.boundary_source_sample)
    references = sorted(reference.events, key=lambda item: item.boundary_source_sample)
    matches = monotonic_boundary_matches(
        [item.boundary_source_sample for item in predicted],
        [item.boundary_source_sample for item in references],
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
        (
            predicted[left].model_evidence_frontier_sample
            - references[right].boundary_source_sample
        )
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
    active_samples = sum(
        item.end_sample - item.start_sample for item in intervals if item.active_speakers
    )
    active_hours = active_samples / 16000.0 / 3600.0
    contamination_seconds = sum(contamination_per_replacement)
    logical_contamination = sum(
        exact_episode_contamination_samples(
            intervals,
            anchor_speaker=anchor,
            start_sample=start,
            end_sample=end,
        )
        for anchor, start, end in contamination_episodes
    )
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
            logical_contamination / 16000.0
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


def session_topology(
    manifest: dict[str, Any],
    predicted: Sequence[ReplacementEvent],
    reference: GTSessionResult,
    tolerance_samples: int,
) -> dict[str, Any]:
    predicted_ordered = sorted(predicted, key=lambda item: item.boundary_source_sample)
    reference_ordered = sorted(reference.events, key=lambda item: item.boundary_source_sample)
    result: dict[str, dict[str, int]] = {}
    for window in manifest["topology_windows"]:
        start = int(window["start_sample"])
        end = int(window["end_sample"])
        topology = str(window["primary_topology"])
        chosen_predicted = [
            item for item in predicted_ordered if start <= item.boundary_source_sample < end
        ]
        chosen_reference = [
            item for item in reference_ordered if start <= item.boundary_source_sample < end
        ]
        values = result.setdefault(
            topology,
            {
                "eligible_episode_count": 0,
                "episodes_with_predicted_cut": 0,
                "episodes_with_reference_replacement": 0,
                "episodes_with_aligned_cut": 0,
            },
        )
        values["eligible_episode_count"] += 1
        values["episodes_with_predicted_cut"] += int(bool(chosen_predicted))
        values["episodes_with_reference_replacement"] += int(bool(chosen_reference))
        values["episodes_with_aligned_cut"] += int(
            any(
                0 <= left.boundary_source_sample - right.boundary_source_sample
                <= tolerance_samples
                for left in chosen_predicted
                for right in chosen_reference
            )
        )
    return result


def aggregate_topology(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, dict[str, int]] = {}
    for row in rows:
        for topology, values in row["topology"].items():
            target = counters.setdefault(
                topology,
                {
                    "eligible_episode_count": 0,
                    "episodes_with_predicted_cut": 0,
                    "episodes_with_reference_replacement": 0,
                    "episodes_with_aligned_cut": 0,
                },
            )
            for key in target:
                target[key] += int(values[key])
    result = {}
    for topology, values in sorted(counters.items()):
        count = values["eligible_episode_count"]
        result[topology] = {
            **values,
            "overlap_return_preservation_rate": (
                1.0 - values["episodes_with_predicted_cut"] / count
                if topology == "overlap_return" and count
                else None
            ),
            "overlap_takeover_success_rate": (
                values["episodes_with_aligned_cut"] / count
                if topology == "overlap_takeover" and count
                else None
            ),
        }
    return result
