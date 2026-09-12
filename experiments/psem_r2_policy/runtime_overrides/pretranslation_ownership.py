from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis
from puripuly_heart.core.stt.backend import STTTimedToken
from puripuly_heart.domain.models import FinalLanguageRun

PretranslationRelation = Literal["CURRENT", "OTHER", "UNKNOWN"]
PretranslationDisposition = Literal[
    "assigned",
    "disabled",
    "unsplit",
    "already_committed",
    "late",
    "retracted",
    "invalid",
]
PretranslationEvidenceDisposition = Literal["observed", "invalid"]


@dataclass(frozen=True, slots=True)
class PretranslationOwnershipUnit:
    group_id: str
    relation: PretranslationRelation
    text: str
    language_runs: tuple[FinalLanguageRun, ...]
    token_indexes: tuple[int, ...]
    start_source_sample: int | None = None
    end_source_sample: int | None = None


@dataclass(frozen=True, slots=True)
class PretranslationAssignment:
    parent_utterance_id: UUID
    disposition: PretranslationDisposition
    units: tuple[PretranslationOwnershipUnit, ...]
    conserved: bool
    late_ignored: tuple[str, ...] = ()
    unknown_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PretranslationEvidence:
    capture_epoch: int
    start_sample: int
    end_sample: int
    available_at_monotonic_s: float
    relation: PretranslationRelation
    producer_generation: object
    reference_generation: object
    reference_valid: bool


@dataclass(frozen=True, slots=True)
class _ObservedHypothesis:
    hypothesis: ProspectiveSpeakerHypothesis
    disposition: Literal["observed", "retracted", "invalid"]


def _runs_from_tokens(tokens: tuple[STTTimedToken, ...]) -> tuple[FinalLanguageRun, ...]:
    runs: list[FinalLanguageRun] = []
    for token in tokens:
        language = token.language
        if runs and runs[-1].language == language:
            previous = runs[-1]
            runs[-1] = FinalLanguageRun(previous.text + token.text, language)
        else:
            runs.append(FinalLanguageRun(token.text, language))
    return tuple(runs)


def _token_source_interval(
    token: STTTimedToken,
) -> tuple[int | None, int | None, str | None]:
    if token.timing == "invalid":
        return None, None, "invalid"
    if token.timing == "unmapped":
        return None, None, "unmapped"
    if token.timing == "end_only" or token.source_start_sample is None:
        return None, token.source_end_sample, "interval_uncertain"
    if token.source_end_sample is None:
        return None, None, "unmapped"
    return token.source_start_sample, token.source_end_sample, None


def _applicable_hypotheses(
    events: tuple[ProspectiveSpeakerHypothesis, ...],
    *,
    admitted_at_monotonic_s: float,
    capture_epoch: int,
) -> list[ProspectiveSpeakerHypothesis]:
    applicable = [
        hypothesis
        for hypothesis in events
        if (
            not hypothesis.retracted
            and hypothesis.producer_valid
            and hypothesis.reference_valid
            and hypothesis.capture_epoch == capture_epoch
            and hypothesis.available_at_monotonic_s <= admitted_at_monotonic_s
            and hypothesis.support_start_sample
            <= hypothesis.estimated_transition_sample
            <= hypothesis.support_end_sample
            <= hypothesis.observed_frontier_sample
        )
    ]
    return sorted(
        applicable,
        key=lambda item: (item.estimated_transition_sample, item.revision, item.hypothesis_id),
    )


def _transition_segment(
    end_sample: int,
    applicable: list[ProspectiveSpeakerHypothesis],
) -> int:
    segment_n = 0
    for hypothesis in applicable:
        if hypothesis.estimated_transition_sample >= end_sample:
            break
        segment_n += 1
    return segment_n


def _relation_from_evidence(
    start_sample: int,
    end_sample: int,
    evidence: tuple[PretranslationEvidence, ...],
    *,
    admitted_at_monotonic_s: float,
    capture_epoch: int,
) -> tuple[PretranslationRelation, str | None]:
    covering = [
        item
        for item in evidence
        if (
            item.reference_valid
            and item.capture_epoch == capture_epoch
            and item.available_at_monotonic_s <= admitted_at_monotonic_s
            and item.start_sample <= start_sample
            and end_sample <= item.end_sample
        )
    ]
    if not covering:
        return "UNKNOWN", "no_reference_evidence"
    relations = {item.relation for item in covering}
    if "UNKNOWN" in relations:
        return "UNKNOWN", "unknown_evidence"
    if len(relations) > 1:
        return "UNKNOWN", "overlap"
    return next(iter(relations)), None

def _same_generation(
    left_producer: object,
    left_reference: object,
    right_producer: object,
    right_reference: object,
) -> bool:
    return left_producer == right_producer and left_reference == right_reference


def _coverage_generation(
    *,
    start_sample: int,
    end_sample: int,
    applicable: list[ProspectiveSpeakerHypothesis],
    evidence: tuple[PretranslationEvidence, ...],
    admitted_at_monotonic_s: float,
    capture_epoch: int,
) -> tuple[object, object] | None:
    candidate_generations: list[tuple[object, object]] = []
    for hypothesis in applicable:
        candidate = (hypothesis.producer_generation, hypothesis.reference_generation)
        if not any(
            _same_generation(candidate[0], candidate[1], existing[0], existing[1])
            for existing in candidate_generations
        ):
            candidate_generations.append(candidate)

    covering_generations: list[tuple[object, object]] = []
    for producer_generation, reference_generation in candidate_generations:
        intervals = sorted(
            (
                max(item.start_sample, start_sample),
                min(item.end_sample, end_sample),
            )
            for item in evidence
            if (
                item.reference_valid
                and item.capture_epoch == capture_epoch
                and item.available_at_monotonic_s <= admitted_at_monotonic_s
                and item.end_sample > item.start_sample
                and _same_generation(
                    item.producer_generation,
                    item.reference_generation,
                    producer_generation,
                    reference_generation,
                )
                and item.end_sample > start_sample
                and item.start_sample < end_sample
            )
        )
        frontier = start_sample
        for interval_start, interval_end in intervals:
            if interval_start > frontier:
                break
            frontier = max(frontier, interval_end)
            if frontier >= end_sample:
                covering_generations.append((producer_generation, reference_generation))
                break
    if len(covering_generations) != 1:
        return None
    return covering_generations[0]


def _whole_parent_unit(
    tokens: tuple[STTTimedToken, ...],
) -> tuple[PretranslationOwnershipUnit, ...]:
    return (_unit_from_run(tokens, list(range(len(tokens))), "UNKNOWN", 0),)


def _partition_is_requested(
    tokens: tuple[STTTimedToken, ...],
    applicable: list[ProspectiveSpeakerHypothesis],
) -> bool:
    if len(tokens) < 2:
        return False
    starts = [
        token.source_start_sample
        for token in tokens
        if token.source_start_sample is not None
    ]
    ends = [
        token.source_end_sample
        for token in tokens
        if token.source_end_sample is not None
    ]
    if not starts or not ends:
        return bool(applicable)
    parent_start = min(starts)
    parent_end = max(ends)
    return any(
        parent_start < hypothesis.estimated_transition_sample < parent_end
        for hypothesis in applicable
    )


def _partition_coverage_generation(
    tokens: tuple[STTTimedToken, ...],
    applicable: list[ProspectiveSpeakerHypothesis],
    evidence: tuple[PretranslationEvidence, ...],
    *,
    admitted_at_monotonic_s: float,
    capture_epoch: int,
) -> tuple[object, object] | None:
    intervals = [_token_source_interval(token) for token in tokens]
    if any(reason is not None for _start, _end, reason in intervals):
        return None
    starts = [start for start, _end, _reason in intervals if start is not None]
    ends = [end for _start, end, _reason in intervals if end is not None]
    if not starts or not ends:
        return None
    parent_start = min(starts)
    parent_end = max(ends)
    if parent_end <= parent_start:
        return None
    cutting = [
        hypothesis
        for hypothesis in applicable
        if parent_start < hypothesis.estimated_transition_sample < parent_end
    ]
    if not cutting:
        return None
    return _coverage_generation(
        start_sample=parent_start,
        end_sample=parent_end,
        applicable=cutting,
        evidence=evidence,
        admitted_at_monotonic_s=admitted_at_monotonic_s,
        capture_epoch=capture_epoch,
    )




def assign_ownership_units(
    tokens: tuple[STTTimedToken, ...],
    events: tuple[ProspectiveSpeakerHypothesis, ...],
    *,
    admitted_at_monotonic_s: float,
    capture_epoch: int,
    evidence: tuple[PretranslationEvidence, ...] = (),
) -> tuple[tuple[PretranslationOwnershipUnit, ...], tuple[str, ...], tuple[str, ...]]:
    applicable = _applicable_hypotheses(
        events,
        admitted_at_monotonic_s=admitted_at_monotonic_s,
        capture_epoch=capture_epoch,
    )
    late_ignored = tuple(
        hypothesis.hypothesis_id
        for hypothesis in events
        if hypothesis.available_at_monotonic_s > admitted_at_monotonic_s
        and hypothesis.capture_epoch == capture_epoch
        and not hypothesis.retracted
    )
    if len(tokens) > 1 and not _partition_is_requested(tokens, applicable):
        return (
            _whole_parent_unit(tokens),
            tuple(late_ignored),
            ("no_confirmed_transition",),
        )
    if tokens and _partition_is_requested(tokens, applicable):
        coverage_generation = _partition_coverage_generation(
            tokens,
            applicable,
            evidence,
            admitted_at_monotonic_s=admitted_at_monotonic_s,
            capture_epoch=capture_epoch,
        )
        if coverage_generation is None:
            return (
                _whole_parent_unit(tokens),
                tuple(late_ignored),
                ("insufficient_evidence_coverage",),
            )
        applicable = [
            hypothesis
            for hypothesis in applicable
            if _same_generation(
                hypothesis.producer_generation,
                hypothesis.reference_generation,
                coverage_generation[0],
                coverage_generation[1],
            )
        ]
        evidence = tuple(
            item
            for item in evidence
            if _same_generation(
                item.producer_generation,
                item.reference_generation,
                coverage_generation[0],
                coverage_generation[1],
            )
        )
    labels: list[tuple[PretranslationRelation, str, str | None]] = []
    for token in tokens:
        start_sample, end_sample, uncertain = _token_source_interval(token)
        if uncertain is not None:
            labels.append(("UNKNOWN", f"u:{uncertain}", uncertain))
            continue
        assert start_sample is not None and end_sample is not None
        straddle = any(
            start_sample < hypothesis.estimated_transition_sample < end_sample
            for hypothesis in applicable
        )
        if straddle:
            labels.append(("UNKNOWN", "u:straddle", "straddle"))
            continue
        segment_n = _transition_segment(end_sample, applicable)
        relation, evidence_reason = _relation_from_evidence(
            start_sample,
            end_sample,
            evidence,
            admitted_at_monotonic_s=admitted_at_monotonic_s,
            capture_epoch=capture_epoch,
        )
        labels.append((relation, f"s:{segment_n}", evidence_reason))
    if not tokens:
        return (), tuple(late_ignored), ()
    units: list[PretranslationOwnershipUnit] = []
    unknown_reasons: list[str] = []
    run_indexes = [0]
    current = labels[0]
    for index, label in enumerate(labels[1:], start=1):
        if label[0] == current[0] and label[1] == current[1]:
            run_indexes.append(index)
            continue
        units.append(_unit_from_run(tokens, run_indexes, current[0], len(units)))
        if current[2] is not None:
            unknown_reasons.append(current[2])
        run_indexes = [index]
        current = label
    units.append(_unit_from_run(tokens, run_indexes, current[0], len(units)))
    if current[2] is not None:
        unknown_reasons.append(current[2])
    return tuple(units), tuple(late_ignored), tuple(unknown_reasons)


def _unit_from_run(
    tokens: tuple[STTTimedToken, ...],
    indexes: list[int],
    relation: PretranslationRelation,
    unit_index: int,
) -> PretranslationOwnershipUnit:
    selected = tuple(tokens[index] for index in indexes)
    starts = [token.source_start_sample for token in selected if token.source_start_sample is not None]
    ends = [token.source_end_sample for token in selected if token.source_end_sample is not None]
    return PretranslationOwnershipUnit(
        group_id=f"{relation}-{unit_index}",
        relation=relation,
        text="".join(token.text for token in selected),
        language_runs=_runs_from_tokens(selected),
        token_indexes=tuple(indexes),
        start_source_sample=min(starts) if starts else None,
        end_source_sample=max(ends) if ends else None,
    )


class PretranslationOwnershipOwner:
    def __init__(self, *, enabled: bool = False, tombstone_capacity: int = 4096) -> None:
        if tombstone_capacity < 1:
            raise ValueError("pretranslation tombstone capacity must be positive")
        self.enabled = enabled
        self._capacity = tombstone_capacity
        self._observed: OrderedDict[tuple[str, int], _ObservedHypothesis] = OrderedDict()
        self._evidence: OrderedDict[int, PretranslationEvidence] = OrderedDict()
        self._evidence_seq = 0
        self._committed: OrderedDict[UUID, PretranslationAssignment] = OrderedDict()

    def observe(self, hypothesis: ProspectiveSpeakerHypothesis) -> PretranslationDisposition:
        key = (hypothesis.hypothesis_id, hypothesis.revision)
        if hypothesis.retracted:
            disposition: Literal["observed", "retracted", "invalid"] = "retracted"
        elif not hypothesis.producer_valid or not hypothesis.reference_valid:
            disposition = "invalid"
        elif not (
            hypothesis.support_start_sample
            <= hypothesis.estimated_transition_sample
            <= hypothesis.support_end_sample
            <= hypothesis.observed_frontier_sample
        ):
            disposition = "invalid"
        else:
            disposition = "observed"
        self._observed[key] = _ObservedHypothesis(hypothesis, disposition)
        self._observed.move_to_end(key)
        while len(self._observed) > self._capacity:
            self._observed.popitem(last=False)
        if disposition == "retracted":
            return "retracted"
        if disposition == "invalid":
            return "invalid"
        return "assigned"

    def observe_evidence(
        self,
        *,
        capture_epoch: int,
        start_sample: int,
        end_sample: int,
        available_at_monotonic_s: float,
        relation: PretranslationRelation,
        producer_generation: object,
        reference_generation: object,
        reference_valid: bool,
    ) -> PretranslationEvidenceDisposition:
        if start_sample > end_sample:
            raise ValueError("evidence start follows end")
        if relation not in {"CURRENT", "OTHER", "UNKNOWN"}:
            raise ValueError(f"unknown evidence relation: {relation!r}")
        evidence = PretranslationEvidence(
            capture_epoch=capture_epoch,
            start_sample=start_sample,
            end_sample=end_sample,
            available_at_monotonic_s=available_at_monotonic_s,
            relation=relation,
            producer_generation=producer_generation,
            reference_generation=reference_generation,
            reference_valid=reference_valid,
        )
        key = self._evidence_seq
        self._evidence_seq += 1
        self._evidence[key] = evidence
        self._evidence.move_to_end(key)
        while len(self._evidence) > self._capacity:
            self._evidence.popitem(last=False)
        if not reference_valid:
            return "invalid"
        return "observed"

    def assign(
        self,
        *,
        parent_utterance_id: UUID,
        timed_tokens: tuple[STTTimedToken, ...],
        capture_epoch: int,
        admitted_at_monotonic_s: float,
        parent_text: str | None = None,
    ) -> PretranslationAssignment:
        prior = self._committed.get(parent_utterance_id)
        if prior is not None:
            return PretranslationAssignment(
                parent_utterance_id=parent_utterance_id,
                disposition="already_committed",
                units=prior.units,
                conserved=prior.conserved,
                late_ignored=prior.late_ignored,
                unknown_reasons=prior.unknown_reasons,
            )
        if not self.enabled:
            assignment = PretranslationAssignment(
                parent_utterance_id=parent_utterance_id,
                disposition="disabled",
                units=(),
                conserved=True,
            )
            return assignment
        if not timed_tokens:
            return PretranslationAssignment(
                parent_utterance_id=parent_utterance_id,
                disposition="unsplit",
                units=(),
                conserved=True,
            )
        token_join = "".join(token.text for token in timed_tokens)
        if parent_text is not None and token_join != parent_text:
            assignment = PretranslationAssignment(
                parent_utterance_id=parent_utterance_id,
                disposition="unsplit",
                units=(),
                conserved=True,
                unknown_reasons=("unsupported",),
            )
            self._remember_committed(parent_utterance_id, assignment)
            return assignment
        usable = tuple(
            item.hypothesis
            for item in self._observed.values()
            if item.disposition == "observed"
        )
        late_events = tuple(
            item.hypothesis
            for item in self._observed.values()
            if item.hypothesis.capture_epoch == capture_epoch
            and item.hypothesis.available_at_monotonic_s > admitted_at_monotonic_s
            and not item.hypothesis.retracted
        )
        evidence = tuple(self._evidence.values())
        units, late_ignored, unknown_reasons = assign_ownership_units(
            timed_tokens,
            usable,
            admitted_at_monotonic_s=admitted_at_monotonic_s,
            capture_epoch=capture_epoch,
            evidence=evidence,
        )
        late_ignored = tuple(
            dict.fromkeys((*late_ignored, *(item.hypothesis_id for item in late_events)))
        )
        reconstructed = "".join(unit.text for unit in units)
        conserved = reconstructed == token_join
        if parent_text is not None:
            conserved = reconstructed == parent_text
        if not conserved:
            assignment = PretranslationAssignment(
                parent_utterance_id=parent_utterance_id,
                disposition="unsplit",
                units=(),
                conserved=True,
                late_ignored=late_ignored,
                unknown_reasons=(*unknown_reasons, "unsupported"),
            )
            self._remember_committed(parent_utterance_id, assignment)
            return assignment
        assignment = PretranslationAssignment(
            parent_utterance_id=parent_utterance_id,
            disposition="assigned",
            units=units,
            conserved=conserved,
            late_ignored=late_ignored,
            unknown_reasons=unknown_reasons,
        )
        self._remember_committed(parent_utterance_id, assignment)
        return assignment

    def observe_after_commit(
        self,
        parent_utterance_id: UUID,
        hypothesis: ProspectiveSpeakerHypothesis,
    ) -> PretranslationDisposition:
        self.observe(hypothesis)
        if parent_utterance_id in self._committed:
            return "late"
        if hypothesis.retracted:
            return "retracted"
        if not hypothesis.producer_valid or not hypothesis.reference_valid:
            return "invalid"
        return "assigned"

    def committed(self, parent_utterance_id: UUID) -> PretranslationAssignment | None:
        return self._committed.get(parent_utterance_id)

    def reset(self) -> None:
        self._observed.clear()
        self._evidence.clear()
        self._committed.clear()

    def _remember_committed(self, parent_utterance_id: UUID, assignment: PretranslationAssignment) -> None:
        self._committed[parent_utterance_id] = assignment
        self._committed.move_to_end(parent_utterance_id)
        while len(self._committed) > self._capacity:
            self._committed.popitem(last=False)


__all__ = [
    "PretranslationAssignment",
    "PretranslationDisposition",
    "PretranslationEvidence",
    "PretranslationOwnershipOwner",
    "PretranslationOwnershipUnit",
    "PretranslationRelation",
    "assign_ownership_units",
]
