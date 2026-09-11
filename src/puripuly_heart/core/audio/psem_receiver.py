from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

PsemDisposition = Literal[
    "sealed",
    "already_separated",
    "too_late_for_current_scope",
    "invalid_source",
    "invalid_reference",
    "duplicate",
    "retracted",
]


@dataclass(frozen=True, slots=True)
class ProspectiveSpeakerHypothesis:
    hypothesis_id: str
    revision: int
    capture_epoch: int
    support_start_sample: int
    support_end_sample: int
    estimated_transition_sample: int
    observed_frontier_sample: int
    available_at_monotonic_s: float
    producer_generation: object
    reference_generation: object
    producer_valid: bool
    reference_valid: bool
    retracted: bool = False
    local_slot: int | None = None

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("speaker hypothesis requires an identity")
        if self.revision < 0:
            raise ValueError("speaker hypothesis revision must be non-negative")
        if self.capture_epoch < 0:
            raise ValueError("speaker hypothesis capture epoch must be non-negative")
        if self.support_start_sample < 0:
            raise ValueError("speaker hypothesis support start must be non-negative")
        if self.support_end_sample < self.support_start_sample:
            raise ValueError("speaker hypothesis support is reversed")
        if self.observed_frontier_sample < self.support_end_sample:
            raise ValueError("observed frontier precedes speaker support")


@dataclass(frozen=True, slots=True)
class ProspectiveSpeakerApplicationReceipt:
    hypothesis_id: str
    revision: int
    disposition: PsemDisposition
    capture_epoch: int
    requested_transition_sample: int
    actual_applied_sample: int | None
    segment_id: UUID | None
    available_at_monotonic_s: float
    applied_at_monotonic_s: float
    producer_generation: object
    reference_generation: object


class ProspectiveSealPort(Protocol):
    async def seal_prospective_transition(
        self,
        *,
        capture_epoch: int,
        requested_source_sample: int,
    ) -> tuple[str, int | None, UUID | None]: ...


class ProspectiveSpeakerTransitionReceiver:
    def __init__(
        self,
        *,
        delivery: ProspectiveSealPort,
        monotonic_clock,
        tombstone_capacity: int = 4096,
    ) -> None:
        if tombstone_capacity < 1:
            raise ValueError("speaker hypothesis tombstone capacity must be positive")
        self._delivery = delivery
        self._monotonic_clock = monotonic_clock
        self._capacity = tombstone_capacity
        self._receipts: OrderedDict[tuple[str, int], ProspectiveSpeakerApplicationReceipt] = (
            OrderedDict()
        )

    async def receive(
        self,
        hypothesis: ProspectiveSpeakerHypothesis,
    ) -> ProspectiveSpeakerApplicationReceipt:
        key = (hypothesis.hypothesis_id, hypothesis.revision)
        prior = self._receipts.get(key)
        if prior is not None:
            return self._receipt(hypothesis, "duplicate")
        if hypothesis.retracted:
            receipt = self._receipt(hypothesis, "retracted")
        elif not hypothesis.producer_valid:
            receipt = self._receipt(hypothesis, "invalid_source")
        elif not hypothesis.reference_valid:
            receipt = self._receipt(hypothesis, "invalid_reference")
        elif not (
            hypothesis.support_start_sample
            <= hypothesis.estimated_transition_sample
            <= hypothesis.support_end_sample
            <= hypothesis.observed_frontier_sample
        ):
            receipt = self._receipt(hypothesis, "invalid_source")
        else:
            disposition, actual, segment_id = await self._delivery.seal_prospective_transition(
                capture_epoch=hypothesis.capture_epoch,
                requested_source_sample=hypothesis.estimated_transition_sample,
            )
            receipt = self._receipt(
                hypothesis,
                disposition,
                actual_applied_sample=actual,
                segment_id=segment_id,
            )
        self._receipts[key] = receipt
        self._receipts.move_to_end(key)
        while len(self._receipts) > self._capacity:
            self._receipts.popitem(last=False)
        return receipt

    def _receipt(
        self,
        hypothesis: ProspectiveSpeakerHypothesis,
        disposition: PsemDisposition | str,
        *,
        actual_applied_sample: int | None = None,
        segment_id: UUID | None = None,
    ) -> ProspectiveSpeakerApplicationReceipt:
        return ProspectiveSpeakerApplicationReceipt(
            hypothesis_id=hypothesis.hypothesis_id,
            revision=hypothesis.revision,
            disposition=disposition,
            capture_epoch=hypothesis.capture_epoch,
            requested_transition_sample=hypothesis.estimated_transition_sample,
            actual_applied_sample=actual_applied_sample,
            segment_id=segment_id,
            available_at_monotonic_s=hypothesis.available_at_monotonic_s,
            applied_at_monotonic_s=self._monotonic_clock(),
            producer_generation=hypothesis.producer_generation,
            reference_generation=hypothesis.reference_generation,
        )


__all__ = [
    "ProspectiveSealPort",
    "ProspectiveSpeakerApplicationReceipt",
    "ProspectiveSpeakerHypothesis",
    "ProspectiveSpeakerTransitionReceiver",
    "PsemDisposition",
]
