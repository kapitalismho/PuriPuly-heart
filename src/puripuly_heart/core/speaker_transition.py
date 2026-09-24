from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from puripuly_heart.domain.models import Transcript

SpeakerComparison = Literal["continuity", "transition", "context_reset", "unavailable"]


@dataclass(frozen=True, slots=True)
class SpeakerTransitionClaim:
    comparison: SpeakerComparison
    claim_id: str

    evidence_available_at: float
    publication_generation: int
    source_order: int

    @property
    def is_transition(self) -> bool:
        return self.comparison == "transition"


@dataclass(frozen=True, slots=True)
class _SpeakerReference:
    speaker_id: str
    session_scope: str
    publication_generation: int
    source_order: int
    child_sequence: int
    source_end_ms: int


class PeerSpeakerTransitionInterpreter:
    def __init__(self) -> None:
        self._reference: _SpeakerReference | None = None
        self._claims: dict[UUID, SpeakerTransitionClaim] = {}

    def reset(self) -> None:
        self._reference = None
        self._claims.clear()

    def observe(self, transcript: Transcript, *, child_sequence: int) -> SpeakerTransitionClaim:
        claim_id = f"peer:{transcript.utterance_id}"
        generation = transcript.publication_generation
        source_order = transcript.source_order
        runs = transcript.final_speaker_runs
        if (
            generation is None
            or source_order is None
            or len(runs) != 1
            or not runs[0].has_ordered_source_evidence
        ):
            claim = SpeakerTransitionClaim(
                "unavailable",
                claim_id,
                transcript.created_at or 0.0,
                generation or 0,
                source_order or 0,
            )
            self._reference = None
            self._claims[transcript.utterance_id] = claim
            return claim
        run = runs[0]
        assert run.speaker_id is not None
        assert run.source_start_ms is not None
        assert run.source_end_ms is not None

        current = _SpeakerReference(
            speaker_id=run.speaker_id,
            session_scope=run.session_scope,
            publication_generation=generation,
            source_order=source_order,
            child_sequence=child_sequence,
            source_end_ms=run.source_end_ms,
        )
        previous = self._reference
        if previous is None:
            comparison: SpeakerComparison = "context_reset"
        elif generation != previous.publication_generation:
            comparison = "context_reset"
        elif current.session_scope != previous.session_scope:
            comparison = "context_reset"
        elif (
            source_order < previous.source_order
            or (source_order == previous.source_order and child_sequence <= previous.child_sequence)
            or run.source_start_ms < previous.source_end_ms
        ):
            comparison = "unavailable"
        elif current.speaker_id != previous.speaker_id:
            comparison = "transition"
        else:
            comparison = "continuity"

        claim = SpeakerTransitionClaim(
            comparison,
            claim_id,
            transcript.created_at or 0.0,
            generation,
            source_order,
        )
        if comparison != "unavailable":
            self._reference = current
        self._claims[transcript.utterance_id] = claim
        return claim

    def claim_for(self, utterance_id: UUID) -> SpeakerTransitionClaim | None:
        return self._claims.get(utterance_id)

    def retire(self, utterance_id: UUID) -> None:
        self._claims.pop(utterance_id, None)
