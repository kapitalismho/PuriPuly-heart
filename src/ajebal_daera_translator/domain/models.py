from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Transcript:
    utterance_id: UUID
    text: str
    is_final: bool
    created_at: float | None = None  # monotonic seconds (Clock)


@dataclass(frozen=True, slots=True)
class Translation:
    utterance_id: UUID
    text: str
    created_at: float | None = None  # monotonic seconds (Clock)


@dataclass(frozen=True, slots=True)
class OSCMessage:
    utterance_id: UUID
    text: str
    created_at: float  # monotonic seconds (Clock)


@dataclass(slots=True)
class UtteranceBundle:
    utterance_id: UUID
    partial: Transcript | None = None
    final: Transcript | None = None
    translation: Translation | None = None

    def with_transcript(self, transcript: Transcript) -> "UtteranceBundle":
        if transcript.utterance_id != self.utterance_id:
            raise ValueError("utterance_id mismatch")

        if transcript.is_final:
            self.final = transcript
            self.partial = None
        else:
            if self.final is None:
                self.partial = transcript
        return self

    def with_translation(self, translation: Translation) -> "UtteranceBundle":
        if translation.utterance_id != self.utterance_id:
            raise ValueError("utterance_id mismatch")
        self.translation = translation
        return self
