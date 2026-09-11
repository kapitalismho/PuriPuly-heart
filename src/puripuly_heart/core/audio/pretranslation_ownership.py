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


def assign_ownership_units(
    tokens: tuple[STTTimedToken, ...],
    events: tuple[ProspectiveSpeakerHypothesis, ...],
    *,
    admitted_at_monotonic_s: float,
    capture_epoch: int,
) -> tuple[tuple[PretranslationOwnershipUnit, ...], tuple[str, ...], tuple[str, ...]]:
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
    applicable = sorted(
        applicable,
        key=lambda item: (item.estimated_transition_sample, item.revision, item.hypothesis_id),
    )
    late_ignored = tuple(
        hypothesis.hypothesis_id
        for hypothesis in events
        if hypothesis.available_at_monotonic_s > admitted_at_monotonic_s
        and hypothesis.capture_epoch == capture_epoch
        and not hypothesis.retracted
    )
    labels: list[tuple[str, str, str | None]] = []
    for token in tokens:
        start_sample, end_sample, uncertain = _token_source_interval(token)
        if uncertain == "invalid":
            labels.append(("UNKNOWN", "unknown", "invalid"))
            continue
        if uncertain == "unmapped" or end_sample is None:
            labels.append(("UNKNOWN", "unknown", "unmapped"))
            continue
        if uncertain == "interval_uncertain":
            labels.append(("UNKNOWN", "unknown", "interval_uncertain"))
            continue
        assert start_sample is not None
        straddle = any(
            start_sample < hypothesis.estimated_transition_sample < end_sample
            for hypothesis in applicable
        )
        if straddle:
            labels.append(("UNKNOWN", "straddle", "straddle"))
            continue
        relation = "CURRENT"
        segment_id = "CURRENT-0"
        other_n = 0
        for hypothesis in applicable:
            if hypothesis.estimated_transition_sample >= end_sample:
                break
            other_n += 1
            relation = "OTHER"
            segment_id = f"OTHER-{other_n}"
        labels.append((relation, segment_id, None))
    units: list[PretranslationOwnershipUnit] = []
    unknown_reasons: list[str] = []
    if not tokens:
        return (), tuple(late_ignored), ()
    run_indexes = [0]
    current = labels[0]
    for index, label in enumerate(labels[1:], start=1):
        if label[0] == current[0] and label[1] == current[1]:
            run_indexes.append(index)
            continue
        units.append(_unit_from_run(tokens, run_indexes, current[0], current[1]))
        if current[2] is not None:
            unknown_reasons.append(current[2])
        run_indexes = [index]
        current = label
    units.append(_unit_from_run(tokens, run_indexes, current[0], current[1]))
    if current[2] is not None:
        unknown_reasons.append(current[2])
    return tuple(units), tuple(late_ignored), tuple(unknown_reasons)


def _unit_from_run(
    tokens: tuple[STTTimedToken, ...],
    indexes: list[int],
    relation: str,
    group_id: str,
) -> PretranslationOwnershipUnit:
    selected = tuple(tokens[index] for index in indexes)
    starts = [token.source_start_sample for token in selected if token.source_start_sample is not None]
    ends = [token.source_end_sample for token in selected if token.source_end_sample is not None]
    return PretranslationOwnershipUnit(
        group_id=group_id,
        relation=relation,  # type: ignore[arg-type]
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

    def assign(
        self,
        *,
        parent_utterance_id: UUID,
        timed_tokens: tuple[STTTimedToken, ...],
        capture_epoch: int,
        admitted_at_monotonic_s: float,
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
        units, late_ignored, unknown_reasons = assign_ownership_units(
            timed_tokens,
            usable,
            admitted_at_monotonic_s=admitted_at_monotonic_s,
            capture_epoch=capture_epoch,
        )
        late_ignored = tuple(
            dict.fromkeys((*late_ignored, *(item.hypothesis_id for item in late_events)))
        )
        conserved = "".join(unit.text for unit in units) == "".join(
            token.text for token in timed_tokens
        )
        assignment = PretranslationAssignment(
            parent_utterance_id=parent_utterance_id,
            disposition="assigned",
            units=units,
            conserved=conserved,
            late_ignored=late_ignored,
            unknown_reasons=unknown_reasons,
        )
        self._committed[parent_utterance_id] = assignment
        self._committed.move_to_end(parent_utterance_id)
        while len(self._committed) > self._capacity:
            self._committed.popitem(last=False)
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
        self._committed.clear()


__all__ = [
    "PretranslationAssignment",
    "PretranslationDisposition",
    "PretranslationOwnershipOwner",
    "PretranslationOwnershipUnit",
    "PretranslationRelation",
    "assign_ownership_units",
]
