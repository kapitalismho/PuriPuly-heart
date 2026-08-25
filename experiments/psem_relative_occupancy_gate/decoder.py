from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from experiments.psem_relative_occupancy_gate.contracts import (
    ActivityInterval,
    AnchorEpisode,
    AnchorLifecycle,
    RelativeState,
    relative_state,
)


@dataclass(frozen=True, slots=True)
class RelativeObservation:
    start_sample: int
    end_sample: int
    state: RelativeState | None
    masked: bool
    evidence_frontier_sample: int

    def __post_init__(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("invalid observation support")
        if self.evidence_frontier_sample < self.end_sample:
            raise ValueError("evidence frontier precedes observation support")
        if self.masked and self.state is not None:
            raise ValueError("masked observations cannot carry a decision state")


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
class EnrollmentEvent:
    source_id: str
    anchor_episode_id: str
    anchor_id: str
    opportunity_start_sample: int
    anchor_emit_sample: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "anchor_episode_id": self.anchor_episode_id,
            "anchor_id": self.anchor_id,
            "opportunity_start_sample": self.opportunity_start_sample,
            "anchor_emit_sample": self.anchor_emit_sample,
        }


@dataclass(frozen=True, slots=True)
class TimelineSpan:
    start_sample: int
    end_sample: int
    lifecycle: AnchorLifecycle
    anchor_id: str | None
    state: RelativeState | None
    masked: bool
    speech_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "lifecycle": self.lifecycle.value,
            "anchor_id": self.anchor_id,
            "state": self.state.value if self.state is not None else None,
            "masked": self.masked,
            "speech_present": self.speech_present,
        }


@dataclass(frozen=True, slots=True)
class GTSessionResult:
    source_id: str
    confirmation_samples: int
    enrollment_samples: int
    silence_reset_samples: int
    events: tuple[ReplacementEvent, ...]
    enrollments: tuple[EnrollmentEvent, ...]
    episodes: tuple[AnchorEpisode, ...]
    timeline: tuple[TimelineSpan, ...]


class ReplacementDecoder:
    def __init__(self, source_id: str, confirmation_samples: int) -> None:
        if confirmation_samples <= 0:
            raise ValueError("confirmation_samples must be positive")
        self.source_id = source_id
        self.confirmation_samples = confirmation_samples
        self.pending_boundary_sample: int | None = None
        self.pending_evidence_samples = 0

    def clear(self) -> None:
        self.pending_boundary_sample = None
        self.pending_evidence_samples = 0

    @property
    def remaining_confirmation_samples(self) -> int:
        return self.confirmation_samples - self.pending_evidence_samples

    def advance(
        self,
        observation: RelativeObservation,
        *,
        lifecycle: AnchorLifecycle,
        anchor_id: str | None,
        anchor_episode_id: str | None,
        compute_lag_ms: float | None = None,
    ) -> ReplacementEvent | None:
        if lifecycle is not AnchorLifecycle.ANCHORED:
            self.clear()
            return None
        if anchor_id is None or anchor_episode_id is None:
            raise ValueError("anchored decoding requires anchor identity and episode")
        if observation.masked:
            return None
        if observation.state is not RelativeState.OTHER_ONLY:
            self.clear()
            return None
        if self.pending_boundary_sample is None:
            self.pending_boundary_sample = observation.start_sample
        duration = observation.end_sample - observation.start_sample
        needed = self.confirmation_samples - self.pending_evidence_samples
        if duration < needed:
            self.pending_evidence_samples += duration
            return None
        qualifying_sample = observation.start_sample + needed
        frontier = observation.evidence_frontier_sample
        event = ReplacementEvent(
            source_id=self.source_id,
            anchor_episode_id=anchor_episode_id,
            anchor_id=anchor_id,
            boundary_source_sample=self.pending_boundary_sample,
            model_evidence_frontier_sample=frontier,
            decoder_emit_sample=max(qualifying_sample, frontier),
            compute_lag_ms=compute_lag_ms,
            confirmation_samples=self.confirmation_samples,
        )
        self.clear()
        return event


class _GTSimulator:
    def __init__(
        self,
        *,
        source_id: str,
        confirmation_samples: int,
        enrollment_samples: int,
        silence_reset_samples: int,
        initial_lifecycle: AnchorLifecycle,
        initial_anchor: str | None,
        scored_start_sample: int,
    ) -> None:
        if enrollment_samples <= 0 or silence_reset_samples <= 0:
            raise ValueError("lifecycle durations must be positive")
        if initial_lifecycle is AnchorLifecycle.ANCHORED and initial_anchor is None:
            raise ValueError("initial anchored lifecycle requires an anchor")
        if initial_lifecycle is not AnchorLifecycle.ANCHORED and initial_anchor is not None:
            raise ValueError("initial anchor requires anchored lifecycle")
        self.source_id = source_id
        self.enrollment_samples = enrollment_samples
        self.silence_reset_samples = silence_reset_samples
        self.lifecycle = initial_lifecycle
        self.anchor = initial_anchor
        self.decoder = ReplacementDecoder(source_id, confirmation_samples)
        self.candidate: str | None = None
        self.candidate_start: int | None = None
        self.candidate_evidence = 0
        self.silence_evidence = 0
        self.episode_counter = 0
        self.current_episode_id: str | None = None
        self.current_opportunity_start: int | None = None
        self.current_anchor_emit: int | None = None
        self.events: list[ReplacementEvent] = []
        self.enrollments: list[EnrollmentEvent] = []
        self.episodes: list[AnchorEpisode] = []
        self.timeline: list[TimelineSpan] = []
        if initial_lifecycle is AnchorLifecycle.ANCHORED:
            self._open_episode(initial_anchor, scored_start_sample, scored_start_sample)

    def _open_episode(self, anchor: str | None, opportunity: int, emit: int) -> None:
        if anchor is None:
            raise ValueError("anchor is required")
        self.episode_counter += 1
        self.current_episode_id = f"{self.source_id}:A{self.episode_counter:05d}"
        self.current_opportunity_start = opportunity
        self.current_anchor_emit = emit
        self.enrollments.append(
            EnrollmentEvent(
                source_id=self.source_id,
                anchor_episode_id=self.current_episode_id,
                anchor_id=anchor,
                opportunity_start_sample=opportunity,
                anchor_emit_sample=emit,
            )
        )

    def _close_episode(self, end_sample: int, replacement_boundary: int | None) -> None:
        if (
            self.current_episode_id is None
            or self.anchor is None
            or self.current_opportunity_start is None
            or self.current_anchor_emit is None
        ):
            raise ValueError("anchor episode is incomplete")
        self.episodes.append(
            AnchorEpisode(
                episode_id=self.current_episode_id,
                source_id=self.source_id,
                anchor_speaker=self.anchor,
                opportunity_start_sample=self.current_opportunity_start,
                anchor_emit_sample=self.current_anchor_emit,
                end_emit_sample=end_sample,
                replacement_boundary_sample=replacement_boundary,
            )
        )
        self.current_episode_id = None
        self.current_opportunity_start = None
        self.current_anchor_emit = None

    def _append(
        self,
        start: int,
        end: int,
        *,
        state: RelativeState | None,
        masked: bool,
        speech_present: bool,
    ) -> None:
        if end <= start:
            return
        span = TimelineSpan(
            start_sample=start,
            end_sample=end,
            lifecycle=self.lifecycle,
            anchor_id=self.anchor,
            state=state,
            masked=masked,
            speech_present=speech_present,
        )
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

    def _clear_candidate(self) -> None:
        self.candidate = None
        self.candidate_start = None
        self.candidate_evidence = 0

    def _become_unanchored(self) -> None:
        self.lifecycle = AnchorLifecycle.UNANCHORED
        self.anchor = None
        self.silence_evidence = 0
        self.decoder.clear()
        self._clear_candidate()

    def process(self, interval: ActivityInterval) -> None:
        self._process_span(
            interval.start_sample,
            interval.end_sample,
            interval.active_speakers,
            interval.masked,
        )

    def _process_span(
        self,
        start: int,
        end: int,
        speakers: tuple[str, ...],
        masked: bool,
    ) -> None:
        if end <= start:
            return
        if masked:
            observation = RelativeObservation(start, end, None, True, end)
            self.decoder.advance(
                observation,
                lifecycle=self.lifecycle,
                anchor_id=self.anchor,
                anchor_episode_id=self.current_episode_id,
            )
            self._append(
                start,
                end,
                state=None,
                masked=True,
                speech_present=bool(speakers),
            )
            return
        if self.lifecycle is not AnchorLifecycle.ANCHORED:
            self.decoder.clear()
            self.silence_evidence = 0
            if len(speakers) != 1:
                self._clear_candidate()
                self._append(
                    start,
                    end,
                    state=None,
                    masked=False,
                    speech_present=bool(speakers),
                )
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
                self._append(
                    start,
                    end,
                    state=None,
                    masked=False,
                    speech_present=True,
                )
                return
            emit = start + needed
            opportunity = self.candidate_start
            if opportunity is None:
                raise ValueError("enrollment opportunity is missing")
            self._append(
                start,
                emit,
                state=None,
                masked=False,
                speech_present=True,
            )
            self.lifecycle = AnchorLifecycle.ANCHORED
            self.anchor = speaker
            self._open_episode(speaker, opportunity, emit)
            self._clear_candidate()
            self._process_span(emit, end, speakers, False)
            return
        if not speakers:
            observation = RelativeObservation(start, end, RelativeState.NONE, False, end)
            self.decoder.advance(
                observation,
                lifecycle=self.lifecycle,
                anchor_id=self.anchor,
                anchor_episode_id=self.current_episode_id,
            )
            needed = self.silence_reset_samples - self.silence_evidence
            duration = end - start
            if duration < needed:
                self.silence_evidence += duration
                self._append(
                    start,
                    end,
                    state=RelativeState.NONE,
                    masked=False,
                    speech_present=False,
                )
                return
            reset_sample = start + needed
            self._append(
                start,
                reset_sample,
                state=RelativeState.NONE,
                masked=False,
                speech_present=False,
            )
            self._close_episode(reset_sample, None)
            self._become_unanchored()
            self._process_span(reset_sample, end, speakers, False)
            return
        self.silence_evidence = 0
        if self.anchor is None:
            raise ValueError("anchored lifecycle has no anchor")
        state = relative_state(self.anchor, speakers)
        observation_end = end
        if state is RelativeState.OTHER_ONLY:
            observation_end = min(
                end,
                start + self.decoder.remaining_confirmation_samples,
            )
        observation = RelativeObservation(
            start,
            observation_end,
            state,
            False,
            observation_end,
        )
        event = self.decoder.advance(
            observation,
            lifecycle=self.lifecycle,
            anchor_id=self.anchor,
            anchor_episode_id=self.current_episode_id,
        )
        if event is None:
            self._append(
                start,
                observation_end,
                state=state,
                masked=False,
                speech_present=True,
            )
            self._process_span(observation_end, end, speakers, False)
            return
        self._append(
            start,
            event.decoder_emit_sample,
            state=state,
            masked=False,
            speech_present=True,
        )
        self.events.append(event)
        self._close_episode(event.decoder_emit_sample, event.boundary_source_sample)
        self._become_unanchored()
        self._process_span(event.decoder_emit_sample, end, speakers, False)

    def finish(self, scored_end_sample: int) -> None:
        if self.lifecycle is AnchorLifecycle.ANCHORED:
            self._close_episode(scored_end_sample, None)


def simulate_gt_session(
    *,
    source_id: str,
    intervals: Sequence[ActivityInterval],
    confirmation_samples: int,
    enrollment_samples: int,
    silence_reset_samples: int,
    initial_lifecycle: AnchorLifecycle = AnchorLifecycle.UNANCHORED,
    initial_anchor: str | None = None,
) -> GTSessionResult:
    if not intervals:
        raise ValueError("intervals are required")
    expected = intervals[0].start_sample
    for interval in intervals:
        if interval.start_sample != expected:
            raise ValueError("interval timeline must be contiguous")
        expected = interval.end_sample
    simulator = _GTSimulator(
        source_id=source_id,
        confirmation_samples=confirmation_samples,
        enrollment_samples=enrollment_samples,
        silence_reset_samples=silence_reset_samples,
        initial_lifecycle=initial_lifecycle,
        initial_anchor=initial_anchor,
        scored_start_sample=intervals[0].start_sample,
    )
    for interval in intervals:
        simulator.process(interval)
    simulator.finish(intervals[-1].end_sample)
    return GTSessionResult(
        source_id=source_id,
        confirmation_samples=confirmation_samples,
        enrollment_samples=enrollment_samples,
        silence_reset_samples=silence_reset_samples,
        events=tuple(simulator.events),
        enrollments=tuple(simulator.enrollments),
        episodes=tuple(simulator.episodes),
        timeline=tuple(simulator.timeline),
    )
