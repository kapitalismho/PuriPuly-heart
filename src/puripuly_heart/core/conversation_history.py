from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime

from puripuly_heart.core.observability import ConversationRecord as ConversationObservation
from puripuly_heart.domain.events import SelfTurnPresentation

MAX_CONVERSATION_RECORDS = 1000
CLEANUP_BATCH = 500


def _format_conversation_timestamp(origin_wall_clock_ms: int | None) -> str:
    timestamp_s = origin_wall_clock_ms / 1000.0 if origin_wall_clock_ms is not None else time.time()
    return datetime.fromtimestamp(timestamp_s).strftime("%H:%M:%S")


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    timestamp_label: str
    source: str
    channel: str
    source_text: str | None
    translated_text: str | None
    utterance_id: str
    source_language: str | None = None
    target_language: str | None = None
    target_index: int | None = None
    disposition: str = "translated"
    turn_kind: str | None = None
    turn_generation: int | None = None
    presentation_revision: int = 0
    source_revision: int = 0
    translation_source_revision: int | None = None
    turn_status: str | None = None
    accepted: bool = True
    parent_utterance_id: str | None = None
    peer_caption_revision: int = 0


class ConversationHistoryModel:
    def __init__(
        self,
        *,
        max_entries: int = MAX_CONVERSATION_RECORDS,
        cleanup_batch: int = CLEANUP_BATCH,
    ) -> None:
        self._max_entries = max_entries
        self._cleanup_batch = cleanup_batch
        self._records: list[ConversationRecord] = []
        self._peer_revisions: OrderedDict[str, tuple[int, bool, bool]] = OrderedDict()

    @property
    def records(self) -> tuple[ConversationRecord, ...]:
        return tuple(self._records)

    def append(self, record: ConversationRecord) -> None:
        if record.channel == "peer" and record.peer_caption_revision > 0:
            self._append_peer_caption(record)
            return
        parent_id = record.parent_utterance_id or record.utterance_id
        matches = [
            index
            for index, previous in enumerate(self._records)
            if previous.channel == record.channel
            and (previous.parent_utterance_id or previous.utterance_id) == parent_id
            and previous.turn_generation is not None
            and (
                (record.target_index is None and record.target_language is None)
                or (
                    record.target_index is not None and previous.target_index == record.target_index
                )
                or (
                    record.target_index is None
                    and previous.target_language == record.target_language
                )
            )
        ]
        if matches:
            for index in matches:
                previous = self._records[index]
                self._records[index] = replace(
                    previous,
                    accepted=True,
                    source_text=record.source_text or previous.source_text,
                    translated_text=record.translated_text or previous.translated_text,
                )
            return
        self._records.append(record)
        if len(self._records) > self._max_entries + self._cleanup_batch:
            del self._records[: self._cleanup_batch]

    def _append_peer_caption(self, record: ConversationRecord) -> None:
        unit_id = record.parent_utterance_id or record.utterance_id
        record = replace(record, utterance_id=unit_id)
        revision = record.peer_caption_revision
        previous_revision = self._peer_revisions.get(unit_id)
        has_source = record.source_text is not None
        has_translation = record.translated_text is not None
        if previous_revision is not None:
            if revision < previous_revision[0]:
                return
            if revision == previous_revision[0]:
                has_source = has_source and not previous_revision[1]
                has_translation = has_translation and not previous_revision[2]
                if not has_source and not has_translation:
                    return
        same_revision = previous_revision is not None and revision == previous_revision[0]
        self._peer_revisions[unit_id] = (
            revision,
            has_source or (same_revision and previous_revision[1]),
            has_translation or (same_revision and previous_revision[2]),
        )
        self._peer_revisions.move_to_end(unit_id)
        while len(self._peer_revisions) > 2 * (self._max_entries + self._cleanup_batch):
            self._peer_revisions.popitem(last=False)
        for index, previous in enumerate(self._records):
            if previous.channel != "peer" or previous.utterance_id != unit_id:
                continue
            self._records[index] = replace(
                record,
                timestamp_label=previous.timestamp_label,
                source=previous.source,
                source_text=(
                    record.source_text
                    if has_source
                    else previous.source_text if same_revision else None
                ),
                translated_text=(
                    record.translated_text
                    if has_translation
                    else previous.translated_text if same_revision else None
                ),
                source_language=record.source_language or previous.source_language,
                target_language=record.target_language or previous.target_language,
                target_index=(
                    record.target_index
                    if record.target_index is not None
                    else previous.target_index
                ),
            )
            return
        self._records.append(record)
        if len(self._records) > self._max_entries + self._cleanup_batch:
            del self._records[: self._cleanup_batch]

    def update_self_turn(self, snapshot: SelfTurnPresentation, *, source: str) -> None:
        utterance_id = str(snapshot.utterance_id)
        existing = [
            record
            for record in self._records
            if record.channel == "self"
            and (record.parent_utterance_id or record.utterance_id) == utterance_id
        ]
        managed = next((record for record in existing if record.turn_generation is not None), None)
        if managed is not None and (
            snapshot.turn_generation != managed.turn_generation
            or snapshot.revision <= managed.presentation_revision
            or snapshot.source_revision < managed.source_revision
            or (managed.turn_status != "active" and snapshot.status != managed.turn_status)
        ):
            return
        timestamp = (
            existing[0].timestamp_label if existing else _format_conversation_timestamp(None)
        )
        replacements = []
        for index, target in enumerate(snapshot.targets or (None,)):
            language = target.target_language if target is not None else None
            previous = next(
                (
                    record
                    for record in existing
                    if (target is not None and record.target_index == index)
                    or record.target_language == language
                    or (
                        target is not None
                        and target.status == "source_only"
                        and record.target_language is None
                        and record.target_index is None
                    )
                ),
                None,
            )
            translation = target.translation if target is not None else None
            replacements.append(
                ConversationRecord(
                    timestamp_label=timestamp,
                    source=source,
                    channel="self",
                    utterance_id=utterance_id,
                    source_text=snapshot.source_text,
                    translated_text=translation.text if translation is not None else None,
                    source_language=snapshot.source_language,
                    target_language=language,
                    target_index=index if target is not None else None,
                    disposition=target.status if target is not None else "source_only",
                    turn_kind="speech",
                    turn_generation=snapshot.turn_generation,
                    presentation_revision=snapshot.revision,
                    source_revision=snapshot.source_revision,
                    translation_source_revision=(
                        target.source_revision if translation is not None else None
                    ),
                    turn_status=snapshot.status,
                    accepted=previous.accepted if previous is not None else False,
                    parent_utterance_id=utterance_id,
                )
            )
        indices = [
            index
            for index, record in enumerate(self._records)
            if record.channel == "self"
            and (record.parent_utterance_id or record.utterance_id) == utterance_id
        ]
        insertion = indices[0] if indices else len(self._records)
        for index in reversed(indices):
            del self._records[index]
        self._records[insertion:insertion] = replacements
        if len(self._records) > self._max_entries + self._cleanup_batch:
            del self._records[: self._cleanup_batch]


class ConversationHistory:
    def __init__(
        self,
        *,
        max_entries: int = MAX_CONVERSATION_RECORDS,
        cleanup_batch: int = CLEANUP_BATCH,
    ) -> None:
        self._model = ConversationHistoryModel(max_entries=max_entries, cleanup_batch=cleanup_batch)
        self._self_turn_observers: list[Callable[[SelfTurnPresentation, str], None]] = []

    @property
    def conversation_records(self) -> tuple[ConversationRecord, ...]:
        return self._model.records

    def observe_self_turns(
        self, callback: Callable[[SelfTurnPresentation, str], None]
    ) -> Callable[[], None]:
        self._self_turn_observers.append(callback)
        return lambda: self._self_turn_observers.remove(callback)

    def observe_conversation(self, record: ConversationObservation) -> None:
        metadata = record.metadata
        revision = metadata.get("peer_caption_revision")
        self.append_conversation_record(
            source=str(
                metadata.get("source") or ("Listen" if record.speaker_channel == "peer" else "Mic")
            ),
            channel=record.speaker_channel,
            utterance_id=record.utterance_id,
            parent_utterance_id=metadata.get("parent_utterance_id"),
            source_text=record.transcript_text,
            translated_text=record.translation_text,
            source_language=record.source_language,
            target_language=record.target_language,
            target_index=metadata.get("target_index"),
            disposition=str(metadata.get("disposition") or "accepted"),
            origin_wall_clock_ms=metadata.get("origin_wall_clock_ms"),
            turn_kind=metadata.get("turn_kind"),
            peer_caption_revision=revision if isinstance(revision, int) and revision > 0 else 0,
        )

    def append_conversation_record(
        self,
        *,
        source: str,
        channel: str,
        utterance_id: str,
        source_text: str | None,
        translated_text: str | None,
        source_language: str | None = None,
        target_language: str | None = None,
        target_index: int | None = None,
        disposition: str = "translated",
        origin_wall_clock_ms: int | None = None,
        turn_kind: str | None = None,
        parent_utterance_id: str | None = None,
        peer_caption_revision: int = 0,
    ) -> None:
        cleaned_source = source_text.strip() if isinstance(source_text, str) else None
        cleaned_translation = translated_text.strip() if isinstance(translated_text, str) else None
        if not cleaned_source and not cleaned_translation:
            return
        self._model.append(
            ConversationRecord(
                timestamp_label=_format_conversation_timestamp(origin_wall_clock_ms),
                source=source.strip() or ("Listen" if channel == "peer" else "Mic"),
                channel=channel,
                utterance_id=utterance_id,
                source_text=cleaned_source or None,
                translated_text=cleaned_translation or None,
                source_language=source_language,
                target_language=target_language,
                target_index=target_index,
                disposition=disposition,
                turn_kind=turn_kind,
                parent_utterance_id=parent_utterance_id,
                peer_caption_revision=peer_caption_revision,
            )
        )

    def update_self_turn(self, snapshot: SelfTurnPresentation, *, source: str) -> None:
        self._model.update_self_turn(snapshot, source=source)
        for callback in tuple(self._self_turn_observers):
            callback(snapshot, source)
