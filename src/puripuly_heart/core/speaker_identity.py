from __future__ import annotations

from uuid import UUID

from puripuly_heart.domain.models import (
    FinalSpeakerRun,
    SpeakerAssignment,
    SpeakerAttribution,
    SpeakerKey,
    Transcript,
)


def native_soniox_attribution(text: str, runs: tuple[FinalSpeakerRun, ...]) -> SpeakerAttribution:
    scopes = {run.session_scope for run in runs if run.source == "soniox"}
    scope = next(iter(scopes)) if len(scopes) == 1 else ""
    fallback = SpeakerAttribution("mixed", source="soniox", speaker_scope_id=scope)
    if not runs or "".join(run.text for run in runs) != text:
        return SpeakerAttribution("malformed", source="soniox", speaker_scope_id=scope)
    if len(scopes) != 1 or not scope or any(run.source != "soniox" for run in runs):
        return fallback
    meaningful = tuple(run for run in runs if any(character.isalnum() for character in run.text))
    attributions = tuple(run.attribution for run in meaningful)
    if any(item.state not in ("identified", "missing") for item in attributions):
        state = "malformed" if any(item.state == "malformed" for item in attributions) else "mixed"
        return SpeakerAttribution(state, source="soniox", speaker_scope_id=scope)
    keys = {item.key for item in attributions if item.key is not None}
    if len(keys) > 1:
        return fallback
    if keys:
        return SpeakerAttribution("identified", next(iter(keys)), "soniox", scope)
    return SpeakerAttribution("missing", source="soniox", speaker_scope_id=scope)


class PeerSpeakerIdentityAllocator:
    CAPACITY = 4

    def __init__(self) -> None:
        self._scope: tuple[str, str] | None = None
        self._keys: dict[SpeakerKey, int] = {}
        self._assignments: dict[UUID, SpeakerAssignment] = {}
        self._last_order: tuple[int, int, int] | None = None

    def reset(self) -> None:
        self._scope = None
        self._keys.clear()
        self._assignments.clear()
        self._last_order = None

    def observe(
        self, transcript: Transcript, *, child_sequence: int, native_soniox: bool = False
    ) -> SpeakerAssignment:
        run = transcript.final_speaker_runs[0] if len(transcript.final_speaker_runs) == 1 else None
        if native_soniox:
            attribution = native_soniox_attribution(transcript.text, transcript.final_speaker_runs)
        elif run is not None:
            attribution = run.attribution
        elif transcript.final_speaker_runs:
            attribution = SpeakerAttribution("mixed")
        else:
            attribution = SpeakerAttribution("non_diarized")
        order = (
            transcript.publication_generation or 0,
            transcript.source_order or 0,
            child_sequence,
        )
        key = attribution.key
        scope = (attribution.source, attribution.speaker_scope_id)
        if key is not None and self._scope != scope:
            if self._last_order is None or order > self._last_order:
                self._scope = scope
                self._keys.clear()
        palette_index = None
        palette_overflow = False
        if self._last_order is None or order > self._last_order:
            self._last_order = order
            if key is not None and scope == self._scope:
                if key in self._keys:
                    palette_index = self._keys[key]
                elif len(self._keys) < self.CAPACITY:
                    palette_index = len(self._keys)
                    self._keys[key] = palette_index
                else:
                    palette_overflow = True
        self._assignments[transcript.utterance_id] = SpeakerAssignment(
            attribution,
            scope_order=order,
            palette_index=palette_index,
            palette_overflow=palette_overflow,
            source_text_range=transcript.source_text_range,
            source_text_revision=transcript.source_text_revision,
        )
        return self._assignments[transcript.utterance_id]

    def assignment_for(self, utterance_id: UUID) -> SpeakerAssignment | None:
        return self._assignments.get(utterance_id)

    def retire(self, utterance_id: UUID) -> None:
        self._assignments.pop(utterance_id, None)
