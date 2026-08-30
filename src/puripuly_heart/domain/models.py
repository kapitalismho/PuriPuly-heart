from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

ChannelId = Literal["self", "peer"]


def _validate_channel(channel: str) -> None:
    if channel not in ("self", "peer"):
        raise ValueError(f"invalid channel: {channel!r}")


def _new_update_id() -> str:
    return uuid4().hex


def _wall_clock_ms_now() -> int:
    return int(time.time() * 1000)


def _hash_source_text(source_text: str) -> str | None:
    if not source_text:
        return None
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class FinalLanguageRun:
    text: str
    language: str


@dataclass(frozen=True, slots=True)
class Transcript:
    utterance_id: UUID
    text: str
    is_final: bool
    created_at: float | None = None  # monotonic seconds (Clock)
    channel: ChannelId = "self"
    final_language_runs: tuple[FinalLanguageRun, ...] = ()

    def __post_init__(self) -> None:
        _validate_channel(self.channel)
        if not self.is_final and self.final_language_runs:
            raise ValueError("partial transcripts cannot have final language runs")


@dataclass(frozen=True, slots=True, init=False)
class Translation:
    utterance_id: UUID
    translated_text: str
    source_text: str
    source_language: str | None
    target_language: str | None
    channel: ChannelId
    created_at: float | None = None  # monotonic seconds (Clock)
    update_id: str
    origin_wall_clock_ms: int | None
    session_scope: str | None
    source_text_hash: str | None
    source_text_len: int | None
    logical_turn_key: str | None

    def __init__(
        self,
        utterance_id: UUID,
        text: str | None = None,
        *,
        translated_text: str | None = None,
        source_text: str = "",
        source_language: str | None = None,
        target_language: str | None = None,
        channel: ChannelId = "self",
        created_at: float | None = None,
        update_id: str | None = None,
        origin_wall_clock_ms: int | None = None,
        session_scope: str | None = None,
        source_text_hash: str | None = None,
        source_text_len: int | None = None,
        logical_turn_key: str | None = None,
    ) -> None:
        if text is not None and translated_text is not None and text != translated_text:
            raise ValueError("text and translated_text must match when both are set")

        resolved_text = translated_text if translated_text is not None else text
        if resolved_text is None:
            raise TypeError("Translation requires text or translated_text")

        _validate_channel(channel)

        object.__setattr__(self, "utterance_id", utterance_id)
        object.__setattr__(self, "translated_text", resolved_text)
        object.__setattr__(self, "source_text", source_text)
        object.__setattr__(self, "source_language", source_language)
        object.__setattr__(self, "target_language", target_language)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "update_id", update_id or _new_update_id())
        object.__setattr__(
            self,
            "origin_wall_clock_ms",
            origin_wall_clock_ms if origin_wall_clock_ms is not None else _wall_clock_ms_now(),
        )
        object.__setattr__(self, "session_scope", session_scope)
        object.__setattr__(
            self,
            "source_text_hash",
            source_text_hash if source_text_hash is not None else _hash_source_text(source_text),
        )
        object.__setattr__(
            self,
            "source_text_len",
            (
                source_text_len
                if source_text_len is not None
                else (len(source_text) if source_text else None)
            ),
        )
        object.__setattr__(
            self,
            "logical_turn_key",
            logical_turn_key if logical_turn_key is not None else f"{channel}:{utterance_id}",
        )

    @property
    def text(self) -> str:
        return self.translated_text


@dataclass(frozen=True, slots=True)
class OSCMessage:
    utterance_id: UUID
    text: str
    created_at: float  # monotonic seconds (Clock)
    turn_generation: int | None = None
    turn_order: int | None = None
    presentation_revision: int = 0
    target_indexes: tuple[int, ...] = ()
    target_languages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.presentation_revision, bool) or not isinstance(
            self.presentation_revision, int
        ):
            raise TypeError("presentation_revision must be an integer")
        if (self.turn_generation is None) != (self.turn_order is None):
            raise ValueError("turn generation and order must be provided together")
        if self.turn_generation is not None:
            if isinstance(self.turn_generation, bool) or not isinstance(self.turn_generation, int):
                raise TypeError("turn_generation must be an integer")
            if self.turn_generation < 0:
                raise ValueError("turn_generation must be non-negative")
        if self.turn_order is not None:
            if isinstance(self.turn_order, bool) or not isinstance(self.turn_order, int):
                raise TypeError("turn_order must be an integer")
            if self.turn_order < 0:
                raise ValueError("turn_order must be non-negative")
        if self.presentation_revision < 0:
            raise ValueError("presentation_revision must be non-negative")
        if self.turn_generation is None and self.presentation_revision != 0:
            raise ValueError("presentation_revision requires self turn identity")
        if not isinstance(self.target_indexes, tuple):
            raise TypeError("target_indexes must be a tuple")
        if not isinstance(self.target_languages, tuple):
            raise TypeError("target_languages must be a tuple")
        if bool(self.target_indexes) != bool(self.target_languages):
            raise ValueError("target indexes and languages must be provided together")
        if len(self.target_indexes) != len(self.target_languages):
            raise ValueError("target indexes and languages must have the same length")
        if self.target_indexes and self.turn_generation is None:
            raise ValueError("target metadata requires self turn identity")
        for target_index in self.target_indexes:
            if isinstance(target_index, bool) or not isinstance(target_index, int):
                raise TypeError("target indexes must be integers")
            if target_index < 0:
                raise ValueError("target indexes must be non-negative")
        if len(set(self.target_indexes)) != len(self.target_indexes):
            raise ValueError("target indexes must be unique")
        if tuple(sorted(self.target_indexes)) != self.target_indexes:
            raise ValueError("target indexes must be in configured order")
        for target_language in self.target_languages:
            if not isinstance(target_language, str):
                raise TypeError("target languages must be strings")
            if not target_language or target_language.strip() != target_language:
                raise ValueError("target languages must be normalized and non-empty")
        if len(set(self.target_languages)) != len(self.target_languages):
            raise ValueError("target languages must be unique")

    @property
    def self_turn_key(self) -> tuple[int, int] | None:
        if self.turn_generation is None or self.turn_order is None:
            return None
        return (self.turn_generation, self.turn_order)


@dataclass(slots=True)
class UtteranceBundle:
    utterance_id: UUID
    channel: ChannelId = "self"
    partial: Transcript | None = None
    final: Transcript | None = None
    translation: Translation | None = None

    def __post_init__(self) -> None:
        _validate_channel(self.channel)

    def with_transcript(self, transcript: Transcript) -> "UtteranceBundle":
        if transcript.utterance_id != self.utterance_id:
            raise ValueError("utterance_id mismatch")
        if self.partial is None and self.final is None and self.translation is None:
            self.channel = transcript.channel
        elif transcript.channel != self.channel:
            raise ValueError("channel mismatch")

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
        if self.partial is None and self.final is None and self.translation is None:
            self.channel = translation.channel
        elif translation.channel != self.channel:
            raise ValueError("channel mismatch")
        self.translation = translation
        return self
