from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from puripuly_heart.core.orchestrator.configuration import TranslationRuntimeConfigSnapshot
from puripuly_heart.domain.models import ChannelId, UtteranceBundle


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


class _SpeculativeAttemptStatus(StrEnum):
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"
    STALE = "stale"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class _SpeculativeAttempt:
    source_text: str
    normalized_text: str
    config_snapshot: TranslationRuntimeConfigSnapshot
    provider_generation: int
    sequence: int
    status: _SpeculativeAttemptStatus = _SpeculativeAttemptStatus.RUNNING
    task: asyncio.Task[None] | None = None
    result: object | None = None
    started_at: float | None = None
    completed_at: float | None = None
    terminal_action_started: bool = False
    latency_stage_times: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _MergeBuffer:
    merge_id: UUID
    parts: list[str] = field(default_factory=list)
    utterance_ids: list[UUID] = field(default_factory=list)
    start_time: float | None = None
    last_end_time: float | None = None
    last_final_at: float = 0.0
    speculative_attempt: _SpeculativeAttempt | None = None
    speculative_sequence: int = 0
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

    def __post_init__(self) -> None:
        attempt = self.speculative_attempt
        if attempt is not None:
            self.speculative_sequence = max(self.speculative_sequence, attempt.sequence)


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
    low_latency_committed_utterance_ids: set[UUID] = field(default_factory=set)
    merge_buffer: _MergeBuffer | None = None

    def __post_init__(self) -> None:
        _validate_channel(self.channel)

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
        spec_attempt = merge_buffer.speculative_attempt
        if spec_attempt is not None:
            spec_attempt.status = _SpeculativeAttemptStatus.CANCELLED
        merge_tasks = [
            spec_attempt.task if spec_attempt is not None else None,
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
        self.low_latency_committed_utterance_ids.clear()
        self.stt_task = None
