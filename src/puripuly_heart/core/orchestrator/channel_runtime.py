from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID

from puripuly_heart.core.orchestrator.configuration import TranslationRuntimeConfigSnapshot
from puripuly_heart.domain.models import ChannelId, UtteranceBundle

_RUNTIME_TO_HUB_ALIAS_FIELDS = {
    "stt": "stt",
    "stt_task": "_stt_task",
    "utterances": "_utterances",
    "translation_tasks": "_translation_tasks",
    "utterance_sources": "_utterance_sources",
    "utterance_start_times": "_utterance_start_times",
    "translation_history": "_translation_history",
    "speech_ended_ids": "_speech_ended_ids",
    "merge_buffer": "_merge_buffer",
}


def _validate_channel(channel: str) -> None:
    if channel not in ("self", "peer"):
        raise ValueError(f"invalid channel: {channel!r}")


@dataclass(frozen=True, slots=True)
class ContextEntry:
    text: str
    source_language: str
    target_language: str
    timestamp: float
    channel: ChannelId = "self"

    def __post_init__(self) -> None:
        _validate_channel(self.channel)


@dataclass(slots=True)
class _MergeBuffer:
    merge_id: UUID
    parts: list[str] = field(default_factory=list)
    utterance_ids: list[UUID] = field(default_factory=list)
    start_time: float | None = None
    last_end_time: float | None = None
    last_final_at: float = 0.0
    spec_task: asyncio.Task[None] | None = None
    spec_text: str | None = None
    spec_translation: object | None = None
    spec_config_snapshot: TranslationRuntimeConfigSnapshot | None = None
    spec_attempts: int = 0
    spec_started_at: float | None = None
    spec_done_at: float | None = None
    spec_latency_stage_times: dict[str, float] = field(default_factory=dict)
    resume_pending: bool = False
    resume_confirmed: bool = False
    resume_utterance_id: UUID | None = None
    resume_chunk_count: int = 0
    resume_started_at: float | None = None
    awaiting_vad_end: bool = False
    awaiting_vad_utterance_id: UUID | None = None
    awaiting_vad_timeout_task: asyncio.Task[None] | None = None
    finalize_wait_task: asyncio.Task[None] | None = None
    finalize_wait_started_at: float | None = None
    resume_end_timeout_task: asyncio.Task[None] | None = None
    resume_end_utterance_id: UUID | None = None


@dataclass(slots=True)
class ChannelRuntime:
    channel: ChannelId
    stt: object | None = None
    stt_task: asyncio.Task[None] | None = None
    utterances: dict[UUID, UtteranceBundle] = field(default_factory=dict)
    translation_tasks: dict[UUID, asyncio.Task[None]] = field(default_factory=dict)
    utterance_sources: dict[UUID, str] = field(default_factory=dict)
    utterance_start_times: dict[UUID, float] = field(default_factory=dict)
    translation_history: list[ContextEntry] = field(default_factory=list)
    speech_ended_ids: set[UUID] = field(default_factory=set)
    merge_buffer: _MergeBuffer | None = None
    alias_target: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_channel(self.channel)

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name == "alias_target":
            return
        alias_target = getattr(self, "alias_target", None)
        if alias_target is None:
            return
        hub_field = _RUNTIME_TO_HUB_ALIAS_FIELDS.get(name)
        if hub_field is None:
            return
        object.__setattr__(alias_target, hub_field, value)

    def get_or_create_bundle(self, utterance_id: UUID) -> UtteranceBundle:
        bundle = self.utterances.get(utterance_id)
        if bundle is None:
            bundle = UtteranceBundle(utterance_id=utterance_id, channel=self.channel)
            self.utterances[utterance_id] = bundle
        return bundle

    def remember_source(self, utterance_id: UUID, source: str | None) -> None:
        if source:
            self.utterance_sources[utterance_id] = source

    def get_source(self, utterance_id: UUID) -> str | None:
        return self.utterance_sources.get(utterance_id)

    def clear_context(self) -> None:
        self.translation_history.clear()

    def remember_context(
        self,
        text: str,
        *,
        timestamp: float,
        source_language: str = "",
        target_language: str = "",
        max_entries: int | None = None,
    ) -> None:
        text_clean = text.strip()
        if len(text_clean) < 2:
            return

        self.translation_history.append(
            ContextEntry(
                text=text_clean,
                source_language=source_language,
                target_language=target_language,
                timestamp=timestamp,
                channel=self.channel,
            )
        )
        if max_entries is not None and max_entries > 0:
            while len(self.translation_history) > max_entries:
                self.translation_history.pop(0)

    def get_valid_context(
        self,
        *,
        now: float,
        source_language: str,
        target_language: str,
        time_window_s: float,
        max_entries: int,
    ) -> list[ContextEntry]:
        history = (
            self.translation_history[-max_entries:] if max_entries > 0 else self.translation_history
        )
        return [
            entry
            for entry in history
            if (now - entry.timestamp) < time_window_s
            and (not entry.source_language or entry.source_language == source_language)
            and (not entry.target_language or entry.target_language == target_language)
            and len(entry.text) >= 2
        ]

    async def clear_live_translation_state(self) -> None:
        translation_task_ids = set(self.translation_tasks)
        translation_tasks = list(self.translation_tasks.values())
        for task in translation_tasks:
            task.cancel()
        if translation_tasks:
            await asyncio.gather(*translation_tasks, return_exceptions=True)
        self.translation_tasks.clear()

        for utterance_id in translation_task_ids:
            self.utterance_start_times.pop(utterance_id, None)
            self.speech_ended_ids.discard(utterance_id)

        if self.merge_buffer is None:
            return

        merge_buffer = self.merge_buffer
        merge_tasks = [
            merge_buffer.spec_task,
            merge_buffer.finalize_wait_task,
            merge_buffer.awaiting_vad_timeout_task,
            merge_buffer.resume_end_timeout_task,
        ]
        for task in merge_tasks:
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in merge_tasks if task is not None), return_exceptions=True
        )

        for utterance_id in set(merge_buffer.utterance_ids):
            self.utterances.pop(utterance_id, None)
            self.utterance_sources.pop(utterance_id, None)
            self.utterance_start_times.pop(utterance_id, None)
            self.speech_ended_ids.discard(utterance_id)

        self.merge_buffer = None

    async def reset_runtime_state(self) -> None:
        await self.clear_live_translation_state()
        self.utterances.clear()
        self.utterance_sources.clear()
        self.utterance_start_times.clear()
        self.translation_history.clear()
        self.speech_ended_ids.clear()
        self.stt_task = None
