from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.domain.events import STTSessionState
from puripuly_heart.domain.models import Transcript, Translation
from puripuly_heart.ui.event_mapping import MappedEvent


@dataclass(frozen=True, slots=True)
class EventProjectionContext:
    source_language: str | None = None
    target_language: str | None = None
    translation_enabled: bool = False
    stt_state: STTSessionState | None = None


@dataclass(frozen=True, slots=True)
class DashboardTranscriptProjection:
    text: str
    language_code: str | None
    debug_prefix: str | None


@dataclass(frozen=True, slots=True)
class DashboardTranslationProjection:
    text: str
    language_code: str | None
    debug_prefix: str | None


@dataclass(frozen=True, slots=True)
class HistoryProjection:
    source: str
    text: str
    translated: bool = False
    language_code: str | None = None


@dataclass(frozen=True, slots=True)
class TranslationAppliedDiagnostic:
    utterance_id: object | None
    channel: str | None
    source_label: str
    dashboard_target_language: str | None
    translation_target_language: str | None
    text_len: int


@dataclass(frozen=True, slots=True)
class EventProjectionBatch:
    status: str | None = None
    transcript: DashboardTranscriptProjection | None = None
    translation: DashboardTranslationProjection | None = None
    history: tuple[HistoryProjection, ...] = ()
    translation_diagnostic: TranslationAppliedDiagnostic | None = None
    osc_history_language_code: str | None = None


@dataclass(slots=True)
class EventProjectionService:
    _closed: bool = False

    def close(self) -> None:
        self._closed = True

    def project(
        self,
        mapped: MappedEvent,
        context: EventProjectionContext,
    ) -> EventProjectionBatch:
        if self._closed:
            return EventProjectionBatch()
        if mapped.kind == "status":
            return EventProjectionBatch(status=mapped.status)
        if mapped.kind == "transcript" and isinstance(mapped.payload, Transcript):
            return self._project_transcript(mapped, mapped.payload, context)
        if mapped.kind == "translation" and isinstance(mapped.payload, Translation):
            return self._project_translation(mapped, mapped.payload, context)
        if mapped.kind == "osc":
            language_code = (
                context.target_language if context.translation_enabled else context.source_language
            )
            return EventProjectionBatch(osc_history_language_code=language_code)
        return EventProjectionBatch()

    def _project_transcript(
        self,
        mapped: MappedEvent,
        transcript: Transcript,
        context: EventProjectionContext,
    ) -> EventProjectionBatch:
        is_final = mapped.transcript_kind == "final"
        history = ()
        if is_final:
            history = (
                HistoryProjection(
                    mapped.source or "Mic",
                    transcript.text,
                    language_code=context.source_language,
                ),
            )
        return EventProjectionBatch(
            transcript=DashboardTranscriptProjection(
                text=transcript.text,
                language_code=context.source_language,
                debug_prefix=None,
            ),
            history=history,
        )

    def _project_translation(
        self,
        mapped: MappedEvent,
        translation: Translation,
        context: EventProjectionContext,
    ) -> EventProjectionBatch:
        source = mapped.source or "Mic"
        return EventProjectionBatch(
            translation=DashboardTranslationProjection(
                text=translation.text,
                language_code=context.target_language,
                debug_prefix=None,
            ),
            history=(
                HistoryProjection(
                    source,
                    translation.text,
                    translated=True,
                    language_code=context.target_language,
                ),
            ),
            translation_diagnostic=TranslationAppliedDiagnostic(
                utterance_id=translation.utterance_id,
                channel=translation.channel,
                source_label=source,
                dashboard_target_language=context.target_language,
                translation_target_language=translation.target_language,
                text_len=len(translation.text),
            ),
        )


__all__ = [
    "DashboardTranscriptProjection",
    "DashboardTranslationProjection",
    "EventProjectionBatch",
    "EventProjectionContext",
    "EventProjectionService",
    "HistoryProjection",
    "TranslationAppliedDiagnostic",
]
