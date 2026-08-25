from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from experiments.psem_relative_occupancy_gate.contracts import (
    ActivityInterval,
    AnchorEpisode,
    AnchorLifecycle,
    EvaluationCell,
    RelativeState,
    Trace,
    evaluation_cells,
    relative_state,
    sample_trace_at_cells,
)
from experiments.psem_relative_occupancy_gate.decoder import (
    RelativeObservation,
    ReplacementDecoder,
    ReplacementEvent,
    TimelineSpan,
)


@dataclass(frozen=True, slots=True)
class PosteriorCell:
    cell: EvaluationCell
    probabilities: tuple[float, ...]
    slot_alive: tuple[bool, ...]
    evidence_frontier_sample: int
    state_reset: bool
    trace_valid: bool


@dataclass(frozen=True, slots=True)
class OracleAnchorMapping:
    anchor_episode_id: str
    anchor_speaker: str
    slot_index: int
    slot_id: str
    support_scores: tuple[float, ...]
    support_weight_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_episode_id": self.anchor_episode_id,
            "anchor_speaker": self.anchor_speaker,
            "slot_index": self.slot_index,
            "slot_id": self.slot_id,
            "support_scores": list(self.support_scores),
            "support_weight_samples": self.support_weight_samples,
        }


@dataclass(frozen=True, slots=True)
class ModelObservation:
    start_sample: int
    end_sample: int
    probabilities: tuple[float, ...]
    slot_alive: tuple[bool, ...]
    trace_valid: bool
    state_reset: bool
    masked: bool
    speech_present: bool
    evidence_frontier_sample: int

    def __post_init__(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("invalid model observation support")
        if len(self.probabilities) != len(self.slot_alive) or not self.probabilities:
            raise ValueError("model observation slot geometry mismatch")
        if self.evidence_frontier_sample < self.end_sample:
            raise ValueError("model observation frontier precedes its support")
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.probabilities):
            raise ValueError("model observation probabilities are invalid")


@dataclass(frozen=True, slots=True)
class CausalEnrollmentConfig:
    active_threshold: float
    other_low_threshold: float
    confirmation_samples: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.other_low_threshold < self.active_threshold <= 1.0:
            raise ValueError("causal enrollment thresholds are invalid")
        if self.confirmation_samples <= 0:
            raise ValueError("causal enrollment confirmation must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_threshold": self.active_threshold,
            "other_low_threshold": self.other_low_threshold,
            "confirmation_samples": self.confirmation_samples,
        }


@dataclass(frozen=True, slots=True)
class CausalEnrollmentEvent:
    source_id: str
    anchor_episode_id: str
    anchor_slot_index: int
    anchor_slot_id: str
    candidate_start_sample: int
    decoder_emit_sample: int
    model_evidence_frontier_sample: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "anchor_episode_id": self.anchor_episode_id,
            "anchor_slot_index": self.anchor_slot_index,
            "anchor_slot_id": self.anchor_slot_id,
            "candidate_start_sample": self.candidate_start_sample,
            "decoder_emit_sample": self.decoder_emit_sample,
            "model_evidence_frontier_sample": self.model_evidence_frontier_sample,
        }


@dataclass(frozen=True, slots=True)
class CausalAnchorEpisode:
    episode_id: str
    anchor_slot_index: int
    anchor_slot_id: str
    candidate_start_sample: int
    anchor_emit_sample: int
    end_emit_sample: int
    end_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "anchor_slot_index": self.anchor_slot_index,
            "anchor_slot_id": self.anchor_slot_id,
            "candidate_start_sample": self.candidate_start_sample,
            "anchor_emit_sample": self.anchor_emit_sample,
            "end_emit_sample": self.end_emit_sample,
            "end_reason": self.end_reason,
        }


@dataclass(frozen=True, slots=True)
class CausalSessionResult:
    source_id: str
    enrollment_config: CausalEnrollmentConfig
    replacement_confirmation_samples: int
    anchor_threshold: float
    other_threshold: float
    enrollments: tuple[CausalEnrollmentEvent, ...]
    episodes: tuple[CausalAnchorEpisode, ...]
    replacement_events: tuple[ReplacementEvent, ...]
    timeline: tuple[TimelineSpan, ...]
    uncertain_entry_count: int
    final_reset_count: int


def posterior_cells(
    trace: Trace,
    intervals: Sequence[ActivityInterval],
    scored_start_sample: int,
    scored_end_sample: int,
    cell_samples: int = 1600,
) -> tuple[PosteriorCell, ...]:
    cells = evaluation_cells(intervals, scored_start_sample, scored_end_sample, cell_samples)
    sampled = sample_trace_at_cells(trace, cells)
    return tuple(
        PosteriorCell(
            cell=cell,
            probabilities=tuple(float(value) for value in sampled["probabilities"][index]),
            slot_alive=tuple(bool(value) for value in sampled["slot_alive"][index]),
            evidence_frontier_sample=int(sampled["evidence_frontier_samples"][index]),
            state_reset=bool(sampled["state_reset"][index]),
            trace_valid=bool(sampled["trace_valid"][index]),
        )
        for index, cell in enumerate(cells)
    )


def oracle_anchor_mapping(
    episode: AnchorEpisode,
    cells: Sequence[PosteriorCell],
    slot_ids: Sequence[str],
) -> OracleAnchorMapping:
    if not slot_ids:
        raise ValueError("oracle mapping requires model slots")
    weighted = np.zeros(len(slot_ids), dtype=np.float64)
    denominator = 0
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
        if cell.masked or not posterior.trace_valid:
            continue
        if episode.anchor_speaker not in cell.active_speakers:
            continue
        weight = cell.duration_samples
        denominator += weight
        probabilities = np.asarray(posterior.probabilities, dtype=np.float64)
        alive = np.asarray(posterior.slot_alive, dtype=np.bool_)
        weighted += probabilities * alive * weight
    if denominator <= 0:
        raise ValueError(f"oracle episode has no valid anchor-active support: {episode.episode_id}")
    scores = weighted / denominator
    slot_index = int(np.argmax(scores))
    return OracleAnchorMapping(
        anchor_episode_id=episode.episode_id,
        anchor_speaker=episode.anchor_speaker,
        slot_index=slot_index,
        slot_id=str(slot_ids[slot_index]),
        support_scores=tuple(float(value) for value in scores),
        support_weight_samples=denominator,
    )


def relative_probabilities(
    posterior: PosteriorCell,
    anchor_slot_index: int,
) -> tuple[float, float] | None:
    if not posterior.trace_valid:
        return None
    if not 0 <= anchor_slot_index < len(posterior.probabilities):
        raise ValueError("anchor slot index is outside the posterior")
    if not posterior.slot_alive[anchor_slot_index]:
        return None
    p_anchor = posterior.probabilities[anchor_slot_index]
    p_other = max(
        (
            probability
            for index, (probability, alive) in enumerate(
                zip(posterior.probabilities, posterior.slot_alive, strict=True)
            )
            if index != anchor_slot_index and alive
        ),
        default=0.0,
    )
    return float(p_anchor), float(p_other)


def model_observations(
    cells: Sequence[PosteriorCell],
    intervals: Sequence[ActivityInterval],
) -> tuple[ModelObservation, ...]:
    result: list[ModelObservation] = []
    interval_index = 0
    for posterior in cells:
        cell = posterior.cell
        while (
            interval_index < len(intervals)
            and intervals[interval_index].end_sample <= cell.start_sample
        ):
            interval_index += 1
        cursor = cell.start_sample
        local_index = interval_index
        first_piece = True
        while cursor < cell.end_sample:
            if local_index >= len(intervals):
                raise ValueError("activity timeline ends before the evaluation grid")
            interval = intervals[local_index]
            if not interval.start_sample <= cursor < interval.end_sample:
                raise ValueError("activity timeline does not cover the evaluation grid")
            end = min(cell.end_sample, interval.end_sample)
            frontier = (
                max(posterior.evidence_frontier_sample, end) if posterior.trace_valid else end
            )
            result.append(
                ModelObservation(
                    start_sample=cursor,
                    end_sample=end,
                    probabilities=posterior.probabilities,
                    slot_alive=posterior.slot_alive,
                    trace_valid=posterior.trace_valid,
                    state_reset=posterior.state_reset and first_piece,
                    masked=interval.masked,
                    speech_present=bool(interval.active_speakers),
                    evidence_frontier_sample=frontier,
                )
            )
            first_piece = False
            cursor = end
            if cursor == interval.end_sample:
                local_index += 1
        interval_index = max(interval_index, local_index - 1)
    return tuple(result)


class CausalAnchorTracker:
    def __init__(
        self,
        *,
        source_id: str,
        slot_ids: Sequence[str],
        enrollment_config: CausalEnrollmentConfig,
        replacement_confirmation_samples: int,
        anchor_threshold: float,
        other_threshold: float,
        silence_reset_samples: int,
    ) -> None:
        if not slot_ids or len(set(slot_ids)) != len(slot_ids):
            raise ValueError("causal tracker requires unique model slots")
        if replacement_confirmation_samples <= 0 or silence_reset_samples <= 0:
            raise ValueError("causal tracker durations must be positive")
        if not 0.0 <= anchor_threshold <= 1.0 or not 0.0 <= other_threshold <= 1.0:
            raise ValueError("causal activity thresholds are invalid")
        self.source_id = source_id
        self.slot_ids = tuple(slot_ids)
        self.enrollment_config = enrollment_config
        self.replacement_confirmation_samples = replacement_confirmation_samples
        self.anchor_threshold = anchor_threshold
        self.other_threshold = other_threshold
        self.silence_reset_samples = silence_reset_samples
        self.lifecycle = AnchorLifecycle.UNANCHORED
        self.anchor_slot_index: int | None = None
        self.candidate_slot_index: int | None = None
        self.candidate_start_sample: int | None = None
        self.candidate_evidence_samples = 0
        self.silence_evidence_samples = 0
        self.decoder = ReplacementDecoder(source_id, replacement_confirmation_samples)
        self.episode_counter = 0
        self.current_episode_id: str | None = None
        self.current_candidate_start: int | None = None
        self.current_anchor_emit: int | None = None
        self.enrollments: list[CausalEnrollmentEvent] = []
        self.episodes: list[CausalAnchorEpisode] = []
        self.replacement_events: list[ReplacementEvent] = []
        self.timeline: list[TimelineSpan] = []
        self.uncertain_entry_count = 0
        self.final_reset_count = 0

    def _clear_candidate(self) -> None:
        self.candidate_slot_index = None
        self.candidate_start_sample = None
        self.candidate_evidence_samples = 0

    def _append_timeline(
        self,
        observation: ModelObservation,
        *,
        lifecycle: AnchorLifecycle | None = None,
        state: RelativeState | None = None,
    ) -> None:
        selected_lifecycle = lifecycle or self.lifecycle
        anchor_emit = self.current_anchor_emit
        if (
            selected_lifecycle is AnchorLifecycle.ANCHORED
            and anchor_emit is not None
            and observation.start_sample < anchor_emit
        ):
            prefix_end = min(observation.end_sample, anchor_emit)
            self._store_timeline_span(
                TimelineSpan(
                    start_sample=observation.start_sample,
                    end_sample=prefix_end,
                    lifecycle=AnchorLifecycle.UNANCHORED,
                    anchor_id=None,
                    state=None,
                    masked=observation.masked,
                    speech_present=observation.speech_present,
                )
            )
            if prefix_end == observation.end_sample:
                return
            observation = replace(observation, start_sample=prefix_end, state_reset=False)
        anchor_id = (
            self.slot_ids[self.anchor_slot_index]
            if self.anchor_slot_index is not None and selected_lifecycle is AnchorLifecycle.ANCHORED
            else None
        )
        span = TimelineSpan(
            start_sample=observation.start_sample,
            end_sample=observation.end_sample,
            lifecycle=selected_lifecycle,
            anchor_id=anchor_id,
            state=state,
            masked=observation.masked,
            speech_present=observation.speech_present,
        )
        self._store_timeline_span(span)

    def _store_timeline_span(self, span: TimelineSpan) -> None:
        if self.timeline:
            previous = self.timeline[-1]
            if (
                previous.end_sample == span.start_sample
                and previous.lifecycle is span.lifecycle
                and previous.anchor_id == span.anchor_id
                and previous.state is span.state
                and previous.masked == span.masked
                and previous.speech_present == span.speech_present
            ):
                self.timeline[-1] = TimelineSpan(
                    start_sample=previous.start_sample,
                    end_sample=span.end_sample,
                    lifecycle=span.lifecycle,
                    anchor_id=span.anchor_id,
                    state=span.state,
                    masked=span.masked,
                    speech_present=span.speech_present,
                )
                return
        self.timeline.append(span)

    def _open_episode(self, slot_index: int, candidate_start: int, emit: int) -> None:
        self.episode_counter += 1
        self.current_episode_id = f"{self.source_id}:C{self.episode_counter:05d}"
        self.current_candidate_start = candidate_start
        self.current_anchor_emit = emit
        event = CausalEnrollmentEvent(
            source_id=self.source_id,
            anchor_episode_id=self.current_episode_id,
            anchor_slot_index=slot_index,
            anchor_slot_id=self.slot_ids[slot_index],
            candidate_start_sample=candidate_start,
            decoder_emit_sample=emit,
            model_evidence_frontier_sample=emit,
        )
        self.enrollments.append(event)

    def _close_episode(self, emit: int, reason: str) -> None:
        if (
            self.current_episode_id is None
            or self.anchor_slot_index is None
            or self.current_candidate_start is None
            or self.current_anchor_emit is None
        ):
            raise ValueError("causal anchor episode is incomplete")
        emit = max(emit, self.current_anchor_emit)
        self.episodes.append(
            CausalAnchorEpisode(
                episode_id=self.current_episode_id,
                anchor_slot_index=self.anchor_slot_index,
                anchor_slot_id=self.slot_ids[self.anchor_slot_index],
                candidate_start_sample=self.current_candidate_start,
                anchor_emit_sample=self.current_anchor_emit,
                end_emit_sample=emit,
                end_reason=reason,
            )
        )
        self.current_episode_id = None
        self.current_candidate_start = None
        self.current_anchor_emit = None

    def _become_unanchored(self) -> None:
        self.lifecycle = AnchorLifecycle.UNANCHORED
        self.anchor_slot_index = None
        self.silence_evidence_samples = 0
        self.decoder.clear()
        self._clear_candidate()

    def _enter_uncertain(self, observation: ModelObservation) -> None:
        if self.lifecycle is AnchorLifecycle.ANCHORED:
            self._close_episode(observation.evidence_frontier_sample, "slot_continuity_invalid")
        self.lifecycle = AnchorLifecycle.ANCHOR_UNCERTAIN
        self.anchor_slot_index = None
        self.decoder.clear()
        self._clear_candidate()
        self.uncertain_entry_count += 1

    def _singleton_candidate(self, observation: ModelObservation) -> int | None:
        if not observation.trace_valid or not observation.speech_present:
            return None
        candidates = [
            index
            for index, (probability, alive) in enumerate(
                zip(observation.probabilities, observation.slot_alive, strict=True)
            )
            if alive and probability >= self.enrollment_config.active_threshold
        ]
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        if any(
            alive
            and index != candidate
            and probability > self.enrollment_config.other_low_threshold
            for index, (probability, alive) in enumerate(
                zip(observation.probabilities, observation.slot_alive, strict=True)
            )
        ):
            return None
        return candidate

    def _predicted_state(self, observation: ModelObservation) -> RelativeState | None:
        if self.anchor_slot_index is None or not observation.trace_valid:
            return None
        if not observation.slot_alive[self.anchor_slot_index]:
            return None
        p_anchor = observation.probabilities[self.anchor_slot_index]
        p_other = max(
            (
                probability
                for index, (probability, alive) in enumerate(
                    zip(observation.probabilities, observation.slot_alive, strict=True)
                )
                if index != self.anchor_slot_index and alive
            ),
            default=0.0,
        )
        return relative_state(
            "anchor",
            tuple(
                value
                for value, present in (
                    ("anchor", p_anchor >= self.anchor_threshold),
                    ("other", p_other >= self.other_threshold),
                )
                if present
            ),
        )

    def _advance_unanchored(self, observation: ModelObservation) -> bool:
        if observation.masked:
            self._append_timeline(observation)
            return False
        if observation.state_reset:
            self._clear_candidate()
        candidate = self._singleton_candidate(observation)
        if candidate is None:
            self._clear_candidate()
            self._append_timeline(observation)
            return False
        if self.candidate_slot_index != candidate:
            self.candidate_slot_index = candidate
            self.candidate_start_sample = observation.start_sample
            self.candidate_evidence_samples = 0
        needed = self.enrollment_config.confirmation_samples - self.candidate_evidence_samples
        duration = observation.end_sample - observation.start_sample
        if duration < needed:
            self.candidate_evidence_samples += duration
            self._append_timeline(observation)
            return False
        candidate_start = self.candidate_start_sample
        if candidate_start is None:
            raise ValueError("causal candidate start is missing")
        emit = max(
            observation.start_sample + needed,
            observation.evidence_frontier_sample,
        )
        self._append_timeline(observation)
        self.lifecycle = AnchorLifecycle.ANCHORED
        self.anchor_slot_index = candidate
        self.silence_evidence_samples = 0
        self._open_episode(candidate, candidate_start, emit)
        self._clear_candidate()
        return True

    def _advance_anchored(self, observation: ModelObservation) -> bool:
        anchor_emit = self.current_anchor_emit
        if anchor_emit is None:
            raise ValueError("causal anchor emission time is missing")
        if observation.end_sample <= anchor_emit:
            self._append_timeline(observation, lifecycle=AnchorLifecycle.UNANCHORED)
            return False
        if observation.start_sample < anchor_emit:
            self._append_timeline(
                replace(observation, end_sample=anchor_emit),
                lifecycle=AnchorLifecycle.UNANCHORED,
            )
            observation = replace(observation, start_sample=anchor_emit, state_reset=False)
        anchor_slot = self.anchor_slot_index
        if (
            anchor_slot is None
            or observation.state_reset
            or not observation.trace_valid
            or not observation.slot_alive[anchor_slot]
        ):
            self._enter_uncertain(observation)
            self._append_timeline(
                observation,
                lifecycle=AnchorLifecycle.ANCHOR_UNCERTAIN,
            )
            return True
        if not observation.masked and not observation.speech_present:
            duration = observation.end_sample - observation.start_sample
            needed = self.silence_reset_samples - self.silence_evidence_samples
            if duration >= needed:
                reset_sample = observation.start_sample + needed
                head = replace(observation, end_sample=reset_sample)
                self._append_timeline(head, state=self._predicted_state(head))
                self._close_episode(reset_sample, "final_speech_end")
                self._become_unanchored()
                self.final_reset_count += 1
                if reset_sample < observation.end_sample:
                    tail = replace(
                        observation,
                        start_sample=reset_sample,
                        state_reset=False,
                    )
                    self._advance_unanchored(tail)
                return False
            self.silence_evidence_samples += duration
        elif not observation.masked:
            self.silence_evidence_samples = 0
        state = None if observation.masked else self._predicted_state(observation)
        event = self.decoder.advance(
            RelativeObservation(
                observation.start_sample,
                observation.end_sample,
                state,
                observation.masked,
                observation.evidence_frontier_sample,
            ),
            lifecycle=self.lifecycle,
            anchor_id=self.slot_ids[anchor_slot],
            anchor_episode_id=self.current_episode_id,
        )
        self._append_timeline(observation, state=state)
        if event is None:
            return False
        self.replacement_events.append(event)
        self._close_episode(event.decoder_emit_sample, "speaker_induced_cut")
        self._become_unanchored()
        return True

    def advance_group(self, observations: Sequence[ModelObservation]) -> None:
        if not observations:
            raise ValueError("causal observation group is empty")
        frontier = observations[0].evidence_frontier_sample
        if any(value.evidence_frontier_sample != frontier for value in observations):
            raise ValueError("causal observation group has mixed evidence frontiers")
        if self.lifecycle is AnchorLifecycle.ANCHOR_UNCERTAIN:
            self._become_unanchored()
        group_started_anchored = self.lifecycle is AnchorLifecycle.ANCHORED
        terminal_change = False
        remainder_lifecycle: AnchorLifecycle | None = None
        for observation in observations:
            if terminal_change:
                self._append_timeline(
                    observation,
                    lifecycle=remainder_lifecycle,
                )
                continue
            if self.lifecycle is AnchorLifecycle.ANCHORED:
                terminal_change = self._advance_anchored(observation)
                if terminal_change:
                    remainder_lifecycle = self.lifecycle
            else:
                enrolled = self._advance_unanchored(observation)
                if enrolled and not group_started_anchored:
                    terminal_change = True
                    remainder_lifecycle = AnchorLifecycle.UNANCHORED

    def finish(self, scored_end_sample: int) -> CausalSessionResult:
        if self.lifecycle is AnchorLifecycle.ANCHORED:
            self._close_episode(scored_end_sample, "scored_end")
        return CausalSessionResult(
            source_id=self.source_id,
            enrollment_config=self.enrollment_config,
            replacement_confirmation_samples=self.replacement_confirmation_samples,
            anchor_threshold=self.anchor_threshold,
            other_threshold=self.other_threshold,
            enrollments=tuple(self.enrollments),
            episodes=tuple(self.episodes),
            replacement_events=tuple(self.replacement_events),
            timeline=tuple(self.timeline),
            uncertain_entry_count=self.uncertain_entry_count,
            final_reset_count=self.final_reset_count,
        )


def simulate_causal_session(
    *,
    source_id: str,
    slot_ids: Sequence[str],
    observations: Sequence[ModelObservation],
    enrollment_config: CausalEnrollmentConfig,
    replacement_confirmation_samples: int,
    anchor_threshold: float,
    other_threshold: float,
    silence_reset_samples: int,
    scored_end_sample: int,
) -> CausalSessionResult:
    tracker = CausalAnchorTracker(
        source_id=source_id,
        slot_ids=slot_ids,
        enrollment_config=enrollment_config,
        replacement_confirmation_samples=replacement_confirmation_samples,
        anchor_threshold=anchor_threshold,
        other_threshold=other_threshold,
        silence_reset_samples=silence_reset_samples,
    )
    group: list[ModelObservation] = []
    frontier: int | None = None
    for observation in observations:
        if frontier is None or observation.evidence_frontier_sample == frontier:
            group.append(observation)
            frontier = observation.evidence_frontier_sample
            continue
        tracker.advance_group(group)
        group = [observation]
        frontier = observation.evidence_frontier_sample
    if group:
        tracker.advance_group(group)
    return tracker.finish(scored_end_sample)
