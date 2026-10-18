from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
from puripuly_heart.domain.models import (
    SpeakerAssignment,
    SpeakerAttribution,
    SpeakerKey,
    Transcript,
)

if TYPE_CHECKING:
    from puripuly_heart.core.orchestrator.translation_turn import SourceChild


@dataclass(frozen=True, slots=True)
class SourceSpeakerSnapshot:
    assignment: SpeakerAssignment
    first_readable_at: float | None
    deadline: float | None
    evidence_revision: int


class SpeakerPresentationAssignments(Protocol):
    changed: asyncio.Event

    def refresh(self, *, now: float | None = None) -> None: ...
    def resolve_source(self, source: SourceChild, *, now: float) -> SpeakerAssignment | None: ...
    def mark_readable(self, source: SourceChild, *, now: float) -> SpeakerAssignment | None: ...
    def next_deadline(self) -> float | None: ...
    def retire_source(self, source_id: UUID) -> None: ...
    def output_disposition(self, utterance_id: UUID, *, applied: bool) -> None: ...


@dataclass(slots=True)
class _SourceAssignment:
    assignment: SpeakerAssignment
    proofs: tuple[str, ...] = ()
    first_readable: float | None = None
    settled: bool = False
    historical_identified: bool = False


class PeerSpeakerIdentityAllocator:
    CAPACITY = 4
    MAX_SOURCES = 128
    MAX_ATTEMPTS = 4096
    LATE_WINDOW_S = 1.0

    def __init__(self, evidence_view=None, *, clock=time.monotonic) -> None:
        self.evidence = evidence_view if evidence_view is not None else SpeakerEvidenceView()
        self.clock = clock
        self._scope: tuple[str, str] | None = None
        self._keys: dict[SpeakerKey, int] = {}
        self._sources: dict[UUID, _SourceAssignment] = {}
        self._attempts = OrderedDict()
        self._producer_done = set()
        self._output_rejected = set()
        self._output_applied = set()
        self._cancelled_attempts = set()
        self._retired_order = (-1, -1, -1)
        self._revision = 0
        self._allocation_order = (-1, -1, -1)
        self.changed = asyncio.Event()
        self.evidence.subscribe(self.changed.set)
        self.resolution_delays = []
        self.readable_unresolved = 0
        self.resolved_within_500ms = 0
        self.resolved_within_1s = 0

    def reset(self) -> None:
        self.evidence.release_claims()
        self._scope = None
        self._keys.clear()
        self._sources.clear()
        self._attempts.clear()
        self._producer_done.clear()
        self._output_rejected.clear()
        self._output_applied.clear()
        self._cancelled_attempts.clear()
        self._retired_order = (-1, -1, -1)
        self._allocation_order = (-1, -1, -1)
        self.changed.set()

    def observe(
        self, transcript: Transcript, *, child_sequence: int, defer_settlement: bool = False
    ) -> SpeakerAssignment:
        source = transcript.source_child
        source_id = transcript.utterance_id if source is None else source.source_id
        existing = self._sources.get(source_id)
        self._attempts[transcript.utterance_id] = source_id
        self._attempts.move_to_end(transcript.utterance_id)
        while len(self._attempts) > self.MAX_ATTEMPTS:
            retired_attempt, _ = self._attempts.popitem(last=False)
            self._producer_done.discard(retired_attempt)
            self._output_rejected.discard(retired_attempt)
            self._output_applied.discard(retired_attempt)
            self._cancelled_attempts.discard(retired_attempt)
        if existing is not None:
            return existing.assignment
        run = transcript.final_speaker_runs[0] if len(transcript.final_speaker_runs) == 1 else None
        attribution = (
            run.attribution
            if run is not None
            else SpeakerAttribution("mixed" if transcript.final_speaker_runs else "not_ready")
        )
        order = (
            source.order
            if source is not None
            else (
                transcript.publication_generation or 0,
                transcript.source_order or 0,
                child_sequence,
            )
        )
        proofs = tuple(
            dict.fromkeys(
                origin.speaker_proof_id
                for origin in (() if source is None else source.origins)
                if origin.speaker_proof_id is not None
            )
        )
        for token in proofs:
            self.evidence.adopt(token)
        assignment = SpeakerAssignment(
            attribution,
            scope_order=order,
            source_text_range=transcript.source_text_range,
            source_text_revision=transcript.source_text_revision,
            settled=False,
        )
        if order <= self._retired_order:
            self._release(_SourceAssignment(assignment, proofs))
            return replace(assignment, settled=True)
        while len(self._sources) >= self.MAX_SOURCES:
            oldest = min(self._sources, key=lambda key: self._sources[key].assignment.scope_order)
            self.retire_source(oldest)
        self._sources[source_id] = _SourceAssignment(
            assignment, proofs, historical_identified=attribution.key is not None
        )
        if not defer_settlement:
            self.refresh()
        return self._sources[source_id].assignment

    def _release(self, record):
        tokens, record.proofs = record.proofs, ()
        for token in tokens:
            if not any(token in other.proofs for other in self._sources.values()):
                self.evidence.release(token)

    def _settle(self, record, *, now):
        if record.settled:
            return
        record.settled = True
        self._revision += 1
        record.assignment = replace(record.assignment, revision=self._revision, settled=True)
        if (
            record.first_readable is not None
            and record.assignment.attribution.key is not None
            and (record.assignment.palette_index is not None or record.assignment.palette_overflow)
        ):
            delay = max(0.0, now - record.first_readable)
            self.resolution_delays.append(delay)
            self.resolution_delays = self.resolution_delays[-self.MAX_SOURCES :]
            self.resolved_within_500ms += delay <= 0.5
            self.resolved_within_1s += delay <= self.LATE_WINDOW_S
        self._release(record)
        self.changed.set()

    def refresh(self, *, now=None):
        now = self.clock() if now is None else now
        blocked = False
        for record in sorted(self._sources.values(), key=lambda item: item.assignment.scope_order):
            if record.settled:
                continue
            if (
                record.proofs
                and not record.historical_identified
                and not all(self.evidence.has_proof(token) for token in record.proofs)
            ):
                record.assignment = replace(
                    record.assignment, attribution=SpeakerAttribution("unavailable")
                )
                self._settle(record, now=now)
                continue
            if (
                record.first_readable is not None
                and now >= record.first_readable + self.LATE_WINDOW_S
            ):
                self._settle(record, now=now)
                continue
            if record.assignment.attribution.key is None and record.proofs:
                runs = [self.evidence.resolve(token).attribution for token in record.proofs]
                keys = {run.key for run in runs}
                if len(keys) == 1 and None not in keys:
                    record.assignment = replace(record.assignment, attribution=runs[0])
                elif any(run.state == "mixed" for run in runs):
                    record.assignment = replace(
                        record.assignment, attribution=SpeakerAttribution("mixed")
                    )
                if not all(self.evidence.has_proof(token) for token in record.proofs):
                    self._settle(record, now=now)
                    continue
            attribution = record.assignment.attribution
            key = attribution.key
            if key is None:
                if record.proofs and attribution.state not in (
                    "mixed",
                    "malformed",
                    "non_diarized",
                ):
                    blocked = True
                else:
                    self._settle(record, now=now)
                continue
            if key is not None and record.assignment.scope_order <= self._allocation_order:
                self._settle(record, now=now)
                continue
            if blocked:
                continue
            scope = (attribution.source, attribution.speaker_scope_id)
            if scope != self._scope:
                self._scope = scope
                self._keys.clear()
            index = self._keys.get(key)
            overflow = index is None and len(self._keys) >= self.CAPACITY
            if index is None and not overflow:
                index = len(self._keys)
                self._keys[key] = index
            record.assignment = replace(
                record.assignment, palette_index=index, palette_overflow=overflow
            )
            self._allocation_order = record.assignment.scope_order
            self._settle(record, now=now)

    def assignment_for(self, utterance_id: UUID) -> SpeakerAssignment | None:
        self.refresh()
        record = self._sources.get(self._attempts.get(utterance_id, utterance_id))
        return None if record is None else record.assignment

    def resolve_source(self, source, *, now):
        self.refresh(now=now)
        record = self._sources.get(source.source_id)
        return None if record is None else record.assignment

    def mark_readable(self, source, *, now):
        record = self._sources.get(source.source_id)
        if record is not None and record.first_readable is None:
            record.first_readable = now
            if not record.settled:
                self.readable_unresolved += 1
                self.changed.set()
        self.refresh(now=now)
        return None if record is None else record.assignment

    def next_deadline(self):
        return min(
            (
                record.first_readable + self.LATE_WINDOW_S
                for record in self._sources.values()
                if not record.settled and record.first_readable is not None
            ),
            default=None,
        )

    def retire(self, utterance_id: UUID) -> None:
        if utterance_id not in self._attempts:
            return
        self._producer_done.add(utterance_id)
        self._dispose_rejected_source(self._attempts.get(utterance_id, utterance_id))

    def cancel_attempt(self, utterance_id: UUID) -> None:
        source_id = self._attempts.get(utterance_id)
        if source_id is None:
            return
        self._cancelled_attempts.add(utterance_id)
        attempts = {attempt for attempt, source in self._attempts.items() if source == source_id}
        if attempts <= self._cancelled_attempts:
            self.retire_source(source_id)

    def output_disposition(self, utterance_id: UUID, *, applied: bool) -> None:
        if utterance_id not in self._attempts:
            return
        if applied:
            self._output_applied.add(utterance_id)
            self._output_rejected.discard(utterance_id)
        elif utterance_id not in self._output_applied:
            self._output_rejected.add(utterance_id)
        self._dispose_rejected_source(self._attempts[utterance_id])

    def _dispose_rejected_source(self, source_id):
        attempts = {attempt for attempt, source in self._attempts.items() if source == source_id}
        if attempts and attempts <= self._producer_done and attempts <= self._output_rejected:
            self.retire_source(source_id)

    def retire_source(self, source_id: UUID) -> None:
        record = self._sources.pop(source_id, None)
        if record is not None:
            self._retired_order = max(self._retired_order, record.assignment.scope_order)
            self._release(record)
            self.changed.set()

    def discard(self, utterance_id: UUID) -> None:
        self.retire_source(self._attempts.get(utterance_id, utterance_id))

    def source_snapshot(self, source_id: UUID) -> SourceSpeakerSnapshot | None:
        self.refresh()
        record = self._sources.get(source_id)
        if record is None:
            return None
        return SourceSpeakerSnapshot(
            record.assignment,
            record.first_readable,
            None if record.first_readable is None else record.first_readable + self.LATE_WINDOW_S,
            self.evidence.revision,
        )

    def open(self) -> None:
        self.evidence.subscribe(self.changed.set)

    def close(self) -> None:
        self.reset()
        self.evidence.unsubscribe(self.changed.set)
