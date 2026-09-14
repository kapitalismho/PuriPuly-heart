from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

from puripuly_heart.domain.events import STTSessionState
from puripuly_heart.domain.models import Transcript, Translation
from puripuly_heart.ui.event_mapping import MappedEvent

_FINAL_TRANSCRIPT_CACHE_LIMIT = 500


@dataclass(frozen=True, slots=True)
class EventProjectionContext:
    source_language: str | None = None
    target_language: str | None = None
    translation_enabled: bool = False
    runtime_logging_mode: object | None = None
    stt_state: STTSessionState | None = None


@dataclass(frozen=True, slots=True)
class DashboardTranscriptProjection:
    text: str
    language_code: str | None
    utterance_id: object | None
    channel: str | None
    source_text_len: int
    transcript_kind: Literal["partial", "final"]
    should_log: bool
    debug_prefix: str | None


@dataclass(frozen=True, slots=True)
class DashboardTranslationProjection:
    text: str
    language_code: str | None
    update_id: str | None
    origin_wall_clock_ms: int | None
    utterance_id: object | None
    channel: str | None
    session_scope: str | None
    source_text_hash: str | None
    source_text_len: int | None
    logical_turn_key: str | None
    debug_prefix: str | None


@dataclass(frozen=True, slots=True)
class HistoryProjection:
    source: str
    text: str
    translated: bool = False
    language_code: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationProjection:
    source: str
    channel: str
    source_text: str
    translated_text: str
    origin_wall_clock_ms: int | None = None
    utterance_id: object | None = None
    source_language: str | None = None
    target_language: str | None = None
    target_index: int | None = None
    disposition: str = "translated"
    turn_kind: str | None = None


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
    conversation: ConversationProjection | None = None
    translation_diagnostic: TranslationAppliedDiagnostic | None = None
    osc_history_language_code: str | None = None


@dataclass(slots=True)
class EventProjectionService:
    final_transcript_cache_limit: int = _FINAL_TRANSCRIPT_CACHE_LIMIT
    _primary_first_partial_emitted: set[str] = field(default_factory=set)
    _final_self_transcripts: OrderedDict[str, Transcript] = field(default_factory=OrderedDict)
    _closed: bool = False

    @property
    def final_self_transcripts(self) -> OrderedDict[str, Transcript]:
        return self._final_self_transcripts

    def close(self) -> None:
        self._closed = True
        self._primary_first_partial_emitted.clear()
        self._final_self_transcripts.clear()

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
        utterance_key = str(transcript.utterance_id)
        if is_final:
            self._primary_first_partial_emitted.discard(utterance_key)
            should_log = True
            self._remember_final_self_transcript(transcript)
        else:
            should_log = utterance_key not in self._primary_first_partial_emitted
            if should_log:
                self._primary_first_partial_emitted.add(utterance_key)
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
                utterance_id=transcript.utterance_id,
                channel=transcript.channel,
                source_text_len=len(transcript.text),
                transcript_kind="final" if is_final else "partial",
                should_log=should_log,
                debug_prefix=_visual_debug_prefix(
                    channel=transcript.channel,
                    utterance_id=transcript.utterance_id,
                    runtime_logging_mode=context.runtime_logging_mode,
                ),
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
        conversation = (
            None
            if mapped.runtime_log_handled
            else self._conversation_record_projection(translation, source=source)
        )
        return EventProjectionBatch(
            translation=DashboardTranslationProjection(
                text=translation.text,
                language_code=context.target_language,
                update_id=translation.update_id,
                origin_wall_clock_ms=translation.origin_wall_clock_ms,
                utterance_id=translation.utterance_id,
                channel=translation.channel,
                session_scope=translation.session_scope,
                source_text_hash=translation.source_text_hash,
                source_text_len=translation.source_text_len,
                logical_turn_key=translation.logical_turn_key,
                debug_prefix=_visual_debug_prefix(
                    channel=translation.channel,
                    utterance_id=translation.utterance_id,
                    update_id=translation.update_id,
                    runtime_logging_mode=context.runtime_logging_mode,
                ),
            ),
            history=(
                HistoryProjection(
                    source,
                    translation.text,
                    translated=True,
                    language_code=context.target_language,
                ),
            ),
            conversation=conversation,
            translation_diagnostic=TranslationAppliedDiagnostic(
                utterance_id=translation.utterance_id,
                channel=translation.channel,
                source_label=source,
                dashboard_target_language=context.target_language,
                translation_target_language=translation.target_language,
                text_len=len(translation.text),
            ),
        )

    def _remember_final_self_transcript(self, transcript: Transcript) -> None:
        if transcript.channel != "self" or not transcript.is_final:
            return
        key = str(transcript.utterance_id)
        self._final_self_transcripts[key] = transcript
        self._final_self_transcripts.move_to_end(key)
        while len(self._final_self_transcripts) > self.final_transcript_cache_limit:
            self._final_self_transcripts.popitem(last=False)

    def _source_text_for_translation(self, translation: Translation) -> str:
        source_text = translation.source_text.strip()
        if source_text:
            return source_text
        transcript = self._final_self_transcripts.get(str(translation.utterance_id))
        if transcript is None:
            return ""
        return transcript.text.strip()

    def _conversation_record_projection(
        self,
        translation: Translation,
        *,
        source: str,
    ) -> ConversationProjection | None:
        translated_text = translation.text.strip()
        if not translated_text:
            return None
        source_text = self._source_text_for_translation(translation)
        if not source_text:
            return None
        return ConversationProjection(
            source=source,
            channel=translation.channel,
            source_text=source_text,
            translated_text=translated_text,
            origin_wall_clock_ms=translation.origin_wall_clock_ms,
            utterance_id=translation.utterance_id,
            source_language=translation.source_language,
            target_language=translation.target_language,
            disposition="translated",
            turn_kind=translation.channel,
        )


def _visual_debug_prefix(
    *,
    channel: str | None,
    utterance_id: object | None,
    runtime_logging_mode: object | None,
    update_id: str | None = None,
) -> str | None:
    if channel != "peer" or utterance_id is None:
        return None
    mode_value = getattr(runtime_logging_mode, "value", runtime_logging_mode)
    if mode_value != "detailed":
        return None
    turn_token = _short_visual_debug_token(utterance_id)
    stage_token = _short_visual_debug_token(update_id) if update_id else "src"
    return f"[P {turn_token}/{stage_token}]"


def _short_visual_debug_token(value: object | None) -> str:
    text = "" if value is None else str(value).strip()
    normalized = "".join(char for char in text if char.isalnum())
    return (normalized[:4] or "none").lower()


__all__ = [
    "ConversationProjection",
    "DashboardTranscriptProjection",
    "DashboardTranslationProjection",
    "EventProjectionBatch",
    "EventProjectionContext",
    "EventProjectionService",
    "HistoryProjection",
    "TranslationAppliedDiagnostic",
]
