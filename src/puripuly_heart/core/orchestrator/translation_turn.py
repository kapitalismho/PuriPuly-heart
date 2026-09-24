from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol
from uuid import UUID, uuid5

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
    TranslationRuntimeConfigSnapshotPort,
)
from puripuly_heart.core.translation_policy import (
    TranslationContextPolicy,
    TranslationRuntimePolicy,
)
from puripuly_heart.domain.models import (
    ChannelId,
    FinalLanguageRun,
    FinalSpeakerRun,
    Transcript,
    Translation,
)

logger = logging.getLogger(__name__)

TranslationTurnKind = Literal["manual", "self", "peer"]
TranslationTurnOutcome = Literal["translated", "source_only", "cancelled", "failed"]
_COMPLETED_PARENT_LIMIT = 4096


def _default_config_snapshot() -> TranslationRuntimeConfigSnapshot:
    return TranslationRuntimeConfigSnapshot(
        revision=0,
        value=TranslationRuntimeConfig(),
    )


def _translation_turn_child_id(
    parent_utterance_id: UUID,
    *,
    turn_kind: TranslationTurnKind,
    run_index: int,
    target_index: int,
    run_language: str,
    target_language: str,
    primary_uses_parent_identity: bool,
) -> UUID:
    if primary_uses_parent_identity and target_index == 0:
        return parent_utterance_id
    return uuid5(
        parent_utterance_id,
        f"{turn_kind}:{run_index}:{target_index}:{run_language}:{target_language}",
    )


@dataclass(frozen=True, slots=True)
class _FinalTranscriptSegment:
    text: str
    language: str
    speaker_id: str | None = None
    speaker_session_scope: str = ""
    source_start_ms: int | None = None
    source_end_ms: int | None = None
    speaker_confidence: float | None = None
    overlaps_previous: bool = False


def _final_transcript_segments(
    transcript: Transcript,
    *,
    split_speakers: bool,
) -> tuple[_FinalTranscriptSegment, ...]:
    text = transcript.text
    language_runs = transcript.final_language_runs or (FinalLanguageRun(text, ""),)
    speaker_runs = (
        transcript.final_speaker_runs
        if split_speakers and transcript.final_speaker_runs
        else (FinalSpeakerRun(text, None, ""),)
    )
    if "".join(run.text for run in language_runs) != text:
        language_runs = (FinalLanguageRun(text, ""),)
    if "".join(run.text for run in speaker_runs) != text:
        speaker_runs = (FinalSpeakerRun(text, None, ""),)
    boundaries = {0, len(text)}
    offset = 0
    for run in language_runs:
        offset += len(run.text)
        boundaries.add(offset)
    offset = 0
    for run in speaker_runs:
        offset += len(run.text)
        boundaries.add(offset)
    positions = sorted(boundaries)
    raw: list[_FinalTranscriptSegment] = []
    language_index = 0
    speaker_index = 0
    language_end = len(language_runs[0].text)
    speaker_end = len(speaker_runs[0].text)
    for start, end in zip(positions, positions[1:]):
        while start >= language_end and language_index + 1 < len(language_runs):
            language_index += 1
            language_end += len(language_runs[language_index].text)
        while start >= speaker_end and speaker_index + 1 < len(speaker_runs):
            speaker_index += 1
            speaker_end += len(speaker_runs[speaker_index].text)
        language = language_runs[language_index].language
        speaker = speaker_runs[speaker_index]
        piece = text[start:end]
        if not piece:
            continue
        segment = _FinalTranscriptSegment(
            piece,
            language,
            speaker.speaker_id,
            speaker.session_scope,
            speaker.source_start_ms,
            speaker.source_end_ms,
            speaker.speaker_confidence,
            speaker.overlaps_previous,
        )
        if (
            raw
            and raw[-1].language == segment.language
            and raw[-1].speaker_id == segment.speaker_id
            and raw[-1].speaker_session_scope == segment.speaker_session_scope
        ):
            previous = raw[-1]
            raw[-1] = replace(
                previous,
                text=previous.text + piece,
                source_end_ms=segment.source_end_ms,
                speaker_confidence=(
                    min(previous.speaker_confidence, segment.speaker_confidence)
                    if previous.speaker_confidence is not None
                    and segment.speaker_confidence is not None
                    else None
                ),
                overlaps_previous=previous.overlaps_previous or segment.overlaps_previous,
            )
        else:
            raw.append(segment)
    segments: list[_FinalTranscriptSegment] = []
    leading = ""
    for segment in raw:
        if not any(character.isalnum() for character in segment.text):
            if segments:
                previous = segments[-1]
                segments[-1] = replace(previous, text=previous.text + segment.text)
            else:
                leading += segment.text
            continue
        if leading:
            segment = replace(segment, text=leading + segment.text)
            leading = ""
        segments.append(segment)
    if leading and segments:
        previous = segments[-1]
        segments[-1] = replace(previous, text=previous.text + leading)
    return tuple(segments)


@dataclass(frozen=True, slots=True)
class _PrestartedTranslation:
    task: asyncio.Task[TranslationTurnProcessResult]
    child_utterance_id: UUID
    provider_generation: int
    config_snapshot: TranslationRuntimeConfigSnapshot


@dataclass(frozen=True, slots=True)
class TranslationTurnRequest:
    transcript: Transcript
    source: str
    turn_kind: TranslationTurnKind
    target_languages: tuple[str, ...]
    config_snapshot: TranslationRuntimeConfigSnapshot
    precomputed_translation: Translation | None = None
    prestarted_secondary_translation: _PrestartedTranslation | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.transcript.is_final:
            raise ValueError("translation turn requires a final transcript")
        expected_channel: ChannelId = "peer" if self.turn_kind == "peer" else "self"
        if self.transcript.channel != expected_channel:
            raise ValueError("translation turn kind does not match transcript channel")
        normalized_targets = tuple(
            dict.fromkeys(
                language.strip() for language in self.target_languages if language.strip()
            )
        )
        if not normalized_targets:
            raise ValueError("translation turn requires at least one target language")
        if self.precomputed_translation is not None:
            if self.precomputed_translation.utterance_id != self.transcript.utterance_id:
                raise ValueError("precomputed translation identity mismatch")
            if self.precomputed_translation.channel != self.transcript.channel:
                raise ValueError("precomputed translation channel mismatch")
            precomputed_target = (
                self.precomputed_translation.target_language.strip()
                if isinstance(self.precomputed_translation.target_language, str)
                else ""
            )
            if precomputed_target and precomputed_target != normalized_targets[0]:
                raise ValueError("precomputed translation must target the primary language")
            nonempty_runs = tuple(
                run for run in self.transcript.final_language_runs if run.text.strip()
            )
            if len(nonempty_runs) > 1:
                raise ValueError("precomputed translation requires exactly one language run")
        if self.prestarted_secondary_translation is not None:
            runs = self.transcript.final_language_runs or (
                FinalLanguageRun(text=self.transcript.text, language=""),
            )
            nonempty_runs = tuple(run for run in runs if run.text.strip())
            if len(normalized_targets) != 2:
                raise ValueError("prestarted secondary translation requires two target languages")
            if self.turn_kind != "self" or len(nonempty_runs) != 1:
                raise ValueError(
                    "prestarted secondary translation requires exactly one language run"
                )
            child_id = _translation_turn_child_id(
                self.transcript.utterance_id,
                turn_kind=self.turn_kind,
                run_index=next(index for index, run in enumerate(runs) if run.text.strip()),
                target_index=1,
                run_language=nonempty_runs[0].language,
                target_language=normalized_targets[1],
                primary_uses_parent_identity=True,
            )
            if self.prestarted_secondary_translation.child_utterance_id != child_id:
                raise ValueError("prestarted secondary translation identity mismatch")
        object.__setattr__(self, "target_languages", normalized_targets)


@dataclass(frozen=True, slots=True)
class TranslationTurnChild:
    parent_utterance_id: UUID
    utterance_id: UUID
    sequence: int
    target_index: int
    turn_generation: int
    turn_order: int
    transcript: Transcript
    detected_language: str | None
    target_language: str
    source: str
    turn_kind: TranslationTurnKind
    context_policy: TranslationContextPolicy
    config_snapshot: TranslationRuntimeConfigSnapshot
    precomputed_translation: Translation | None = None
    parent_output_count: int = 1
    prestarted_translation: _PrestartedTranslation | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def channel(self) -> ChannelId:
        return self.transcript.channel


@dataclass(frozen=True, slots=True)
class TranslationOutputSubmission:
    parent_utterance_id: UUID
    child_utterance_id: UUID
    sequence: int
    channel: ChannelId
    source: str
    source_text: str
    source_language: str | None
    target_language: str
    outcome: TranslationTurnOutcome
    config_snapshot: TranslationRuntimeConfigSnapshot
    translation: Translation | None = None
    applied_context_mode: Literal["integrated"] | None = None
    failure_code: str | None = None
    target_index: int = 0
    turn_generation: int | None = None
    turn_order: int | None = None
    publication_generation: int | None = None
    source_order: int | None = None
    turn_kind: TranslationTurnKind | None = None
    parent_output_count: int = 1
    context_texts: tuple[str, ...] | None = None
    speaker_transition: str | None = None
    speaker_transition_claim_id: str | None = None

    def __post_init__(self) -> None:
        if self.outcome == "translated" and self.translation is None:
            raise ValueError("translated output requires a translation")
        if self.translation is not None:
            if self.outcome != "translated":
                raise ValueError("only translated output may include a translation")
            if self.translation.utterance_id != self.child_utterance_id:
                raise ValueError("translation output child identity mismatch")
            if self.translation.channel != self.channel:
                raise ValueError("translation output channel mismatch")
        if (self.turn_generation is None) != (self.turn_order is None):
            raise ValueError("turn generation and order must be provided together")
        for name, value in (
            ("turn_generation", self.turn_generation),
            ("turn_order", self.turn_order),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (self.publication_generation is None) != (self.source_order is None):
            raise ValueError("publication generation and source order must be provided together")
        if self.publication_generation is not None and self.channel != "peer":
            raise ValueError("publication generation is only valid for Peer output")
        if self.channel != "peer" and self.speaker_transition is not None:
            raise ValueError("speaker transition evidence is only valid for Peer output")
        if (self.speaker_transition is None) != (self.speaker_transition_claim_id is None):
            raise ValueError(
                "speaker transition comparison and claim identity must be provided together"
            )


@dataclass(frozen=True, slots=True)
class TranslationTurnProcessResult:
    outcome: TranslationTurnOutcome
    output: TranslationOutputSubmission | None = None

    def __post_init__(self) -> None:
        if self.output is not None and self.output.outcome != self.outcome:
            raise ValueError("translation process result outcome mismatch")


class TranslationOutputSubmissionPort(Protocol):
    async def submit_translation_output(
        self,
        submission: TranslationOutputSubmission,
    ) -> object | None: ...


@dataclass(slots=True)
class _TranslationTurnParent:
    parent_utterance_id: UUID
    channel: ChannelId
    children: tuple[TranslationTurnChild, ...]
    turn_generation: int
    turn_order: int
    admitted_at_monotonic_s: float
    execution_started: bool = False
    waiting_expiry_task: asyncio.Task[None] | None = None
    completed_child_ids: set[UUID] = field(default_factory=set)
    semantic_completed_child_ids: set[UUID] = field(default_factory=set)
    semantic_done_event: asyncio.Event = field(default_factory=asyncio.Event)
    closed_event: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False

    @property
    def child_ids(self) -> tuple[UUID, ...]:
        return tuple(child.utterance_id for child in self.children)


ChildCreated = Callable[[TranslationTurnChild], Awaitable[None]]
ChildStarted = Callable[
    [TranslationTurnChild, asyncio.Task[TranslationTurnProcessResult]], Awaitable[None]
]
ChildCancellationRequested = Callable[[], bool]
ChildProcessor = Callable[
    [TranslationTurnChild, ChildCancellationRequested],
    Awaitable[TranslationTurnProcessResult | TranslationTurnOutcome],
]
ChildTerminal = Callable[[TranslationTurnChild, TranslationTurnOutcome], Awaitable[None]]
ParentClosed = Callable[[UUID], Awaitable[None]]
ParentRejected = Callable[[UUID], Awaitable[None]]
ParentAdmitted = Callable[[tuple[TranslationTurnChild, ...]], Awaitable[None]]
TurnGenerationAdvanced = Callable[[ChannelId, int], None]


@dataclass(slots=True)
class TranslationTurnLifecycleOwner:
    on_child_created: ChildCreated
    on_child_started: ChildStarted
    process_child: ChildProcessor
    on_child_terminal: ChildTerminal
    on_parent_closed: ParentClosed
    on_parent_rejected: ParentRejected
    on_parent_admitted: ParentAdmitted | None = None
    predecessor_wait_observer: Callable[[str, Mapping[str, object]], None] | None = None
    turn_generation_observer: TurnGenerationAdvanced | None = None
    output: TranslationOutputSubmissionPort | None = None
    config_snapshot: TranslationRuntimeConfigSnapshotPort = _default_config_snapshot
    policy: TranslationRuntimePolicy = field(default_factory=TranslationRuntimePolicy)
    _parents: dict[UUID, _TranslationTurnParent] = field(default_factory=dict)
    peer_waiting_capacity: int = 8
    peer_waiting_ttl_s: float = 12.0
    self_speech_waiting_capacity: int = 8
    self_speech_running_capacity: int = 2
    self_speech_waiting_ttl_s: float = 12.0
    _output_submitted_child_ids: set[UUID] = field(default_factory=set, init=False)
    child_watchdog_s: float = 60.0
    _closed_parent_ids: OrderedDict[UUID, tuple[ChannelId, int, int]] = field(
        default_factory=OrderedDict
    )
    _cancelling_parent_ids: set[UUID] = field(default_factory=set)
    _parent_tasks: dict[UUID, asyncio.Task[None]] = field(default_factory=dict)
    _active_tasks: dict[UUID, asyncio.Task[TranslationTurnProcessResult]] = field(
        default_factory=dict
    )
    _channel_tails: dict[ChannelId, _TranslationTurnParent] = field(default_factory=dict)
    _channel_admission_locks: dict[ChannelId, asyncio.Lock] = field(default_factory=dict)
    _channel_turn_generations: dict[ChannelId, int] = field(
        default_factory=lambda: {"self": 0, "peer": 0}
    )
    _channel_next_turn_orders: dict[ChannelId, int] = field(
        default_factory=lambda: {"self": 0, "peer": 0}
    )
    _scope: LifecycleScope = field(init=False)
    _self_speech_slots: asyncio.Semaphore = field(init=False)
    _peer_execution_condition: asyncio.Condition = field(init=False)
    _peer_running_count: int = field(default=0, init=False)
    _peer_active_parent_count: int = field(default=0, init=False)
    _blocked_channels: set[ChannelId] = field(default_factory=set)
    _accepting: bool = True
    _closed: bool = False

    def __post_init__(self) -> None:
        if self.peer_waiting_capacity < 1 or self.self_speech_waiting_capacity < 1:
            raise ValueError("translation waiting capacity must be positive")
        if self.self_speech_running_capacity < 1:
            raise ValueError("Self speech running capacity must be positive")
        if (
            self.peer_waiting_ttl_s <= 0
            or self.self_speech_waiting_ttl_s <= 0
            or self.child_watchdog_s <= 0
        ):
            raise ValueError("translation lifecycle bounds must be positive")
        self._scope = LifecycleScope("translation-turns")
        self._self_speech_slots = asyncio.Semaphore(self.self_speech_running_capacity)
        self._peer_execution_condition = asyncio.Condition()

    @property
    def has_resources(self) -> bool:
        return bool(self._parents or self._parent_tasks or self._active_tasks)

    def is_parent_closed(self, parent_utterance_id: UUID) -> bool:
        return parent_utterance_id in self._closed_parent_ids

    def is_parent_active(self, parent_utterance_id: UUID) -> bool:
        return parent_utterance_id in self._parents

    def is_child_cancellation_requested(self, child: TranslationTurnChild) -> bool:
        return (
            self._closed
            or not self._accepting
            or child.channel in self._blocked_channels
            or child.parent_utterance_id in self._cancelling_parent_ids
        )

    def channel_ingress_open(self, channel: ChannelId) -> bool:
        return self._accepting and not self._closed and channel not in self._blocked_channels

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": "TranslationTurnLifecycleOwner",
            "resource_fields": (
                "parent tasks",
                "child tasks",
                "parent/child terminal state",
            ),
            "ordering": (
                "self parent context admission is serialized and provider execution overlaps",
                "peer source context admission is serialized; provider execution overlaps with ordered output",
            ),
            "stop_ingress": "stop accepting translation turns",
            "shutdown_policy": "cancel parent tasks, terminalize unfinished children, await scope",
            "late_callback_rule": "closed parents reject child completion and output submission",
            "fast_translation_enabled": self.policy.fast_translation_enabled,
            "context_policy": self.policy.context_policy,
        }

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("TranslationTurnLifecycleOwner is closed")

    async def open_channel_ingress(self, channel: ChannelId) -> None:
        if self._closed:
            raise RuntimeError("TranslationTurnLifecycleOwner is closed")
        self._blocked_channels.discard(channel)

    async def close_channel_ingress(self, channel: ChannelId) -> None:
        self._blocked_channels.add(channel)
        self._advance_turn_generation(channel)
        await self._cancel_channel(channel)
        self._blocked_channels.add(channel)

    async def submit(
        self,
        request: TranslationTurnRequest,
        *,
        wait_for_parent: bool = False,
    ) -> tuple[UUID, ...]:
        parent_id = request.transcript.utterance_id
        if parent_id in self._closed_parent_ids or parent_id in self._parents:
            await self._cancel_prestarted_translation(request.prestarted_secondary_translation)
            await self._reject_parent(parent_id)
            return ()
        if (
            not self._accepting
            or self._closed
            or request.transcript.channel in self._blocked_channels
        ):
            await self._cancel_prestarted_translation(request.prestarted_secondary_translation)
            await self._reject_parent(parent_id)
            return ()
        channel = request.transcript.channel
        turn_generation = self._channel_turn_generations[channel]
        turn_order = self._channel_next_turn_orders[channel]
        self._channel_next_turn_orders[channel] = turn_order + 1
        children = self._build_children(
            request,
            turn_generation=turn_generation,
            turn_order=turn_order,
        )
        parent = _TranslationTurnParent(
            parent_utterance_id=parent_id,
            channel=channel,
            children=children,
            turn_generation=turn_generation,
            turn_order=turn_order,
            admitted_at_monotonic_s=asyncio.get_running_loop().time(),
        )
        self._parents[parent_id] = parent
        if not children:
            await self._cancel_prestarted_translation(request.prestarted_secondary_translation)
            await self._close_parent(parent)
            return ()
        try:
            await self.start()
            admission_lock = self._channel_admission_locks.setdefault(
                parent.channel,
                asyncio.Lock(),
            )
            predecessor: _TranslationTurnParent | None = None
            async with admission_lock:
                if self._parent_cancellation_requested(parent):
                    await self._terminalize_parent_remaining(parent, "cancelled")
                else:
                    if self.on_parent_admitted is not None:
                        try:
                            await self.on_parent_admitted(parent.children)
                        except Exception:
                            logger.exception("translation parent admission adapter failed")
                            await self._terminalize_parent_remaining(parent, "failed")
                    if self._parent_cancellation_requested(parent):
                        await self._terminalize_parent_remaining(parent, "cancelled")
                    elif not parent.closed:
                        predecessor = self._channel_tails.get(parent.channel)
                        if predecessor is not None and predecessor.closed:
                            predecessor = None
                        if any(child.turn_kind == "manual" for child in parent.children):
                            predecessor = None
                        self._channel_tails[parent.channel] = parent
            overflow = self._peer_waiting_parents()[: -self.peer_waiting_capacity]
            for waiting_parent in overflow:
                await self._retire_waiting_parent(
                    waiting_parent,
                    "source_only",
                    failure_code="translation_overload",
                )
            if parent.channel == "peer" and not parent.closed:
                parent.waiting_expiry_task = start_lifecycle_task(
                    self._scope,
                    self._expire_waiting_parent(parent),
                    name=f"peer-waiting-expiry:{parent_id}",
                    eager_start=True,
                )
            self_speech_overflow = self._self_speech_waiting_parents()[
                : -self.self_speech_waiting_capacity
            ]
            for waiting_parent in self_speech_overflow:
                await self._retire_waiting_parent(
                    waiting_parent,
                    "source_only",
                    failure_code="translation_overload",
                )
            if (
                parent.channel == "self"
                and any(child.turn_kind == "self" for child in parent.children)
                and not parent.closed
            ):
                parent.waiting_expiry_task = start_lifecycle_task(
                    self._scope,
                    self._expire_waiting_parent(parent),
                    name=f"self-speech-waiting-expiry:{parent_id}",
                    eager_start=True,
                )
            for child in children:
                if parent.closed:
                    break
                try:
                    await self.on_child_created(child)
                except Exception:
                    logger.exception("translation child creation adapter failed")
                    await self._terminalize_child(child, "failed")
            if not parent.closed:
                if self._parent_cancellation_requested(parent):
                    await self._terminalize_parent_remaining(parent, "cancelled")
                else:
                    parent_task = start_lifecycle_task(
                        self._scope,
                        self._run_parent(parent, predecessor),
                        name=f"parent:{parent_id}",
                        eager_start=True,
                    )
                    if not parent_task.done():
                        self._parent_tasks[parent_id] = parent_task
        except (Exception, asyncio.CancelledError) as exc:
            if not parent.closed:
                self._cancelling_parent_ids.add(parent_id)
                await self._terminalize_parent_remaining(
                    parent,
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                )
            raise
        if wait_for_parent:
            await parent.closed_event.wait()
        return parent.child_ids

    async def _reject_parent(self, parent_utterance_id: UUID) -> None:
        try:
            await self.on_parent_rejected(parent_utterance_id)
        except Exception:
            logger.exception("translation parent rejection adapter failed")

    async def submit_parent(
        self,
        transcript: Transcript,
        *,
        source: str,
        turn_kind: TranslationTurnKind | None = None,
        target_languages: tuple[str, ...] = ("und",),
    ) -> tuple[UUID, ...]:
        resolved_kind = turn_kind or ("peer" if transcript.channel == "peer" else "self")
        return await self.submit(
            TranslationTurnRequest(
                transcript=transcript,
                source=source,
                turn_kind=resolved_kind,
                target_languages=target_languages,
                config_snapshot=self.config_snapshot(),
            )
        )

    async def cancel_pending(
        self,
        *,
        channel: ChannelId | None = None,
        turn_kinds: frozenset[TranslationTurnKind] | None = None,
    ) -> None:
        if self._closed:
            return
        if turn_kinds is not None and channel is None:
            raise ValueError("turn-kind cancellation requires a channel")
        if channel is not None:
            if turn_kinds is None:
                self._advance_turn_generation(channel)
            await self._cancel_channel(channel, turn_kinds=turn_kinds)
            return
        self._advance_turn_generation("self")
        self._advance_turn_generation("peer")
        self._accepting = False
        self._request_cancellation()
        await self._drain_admission_locks()
        await self._scope.close()
        await self._terminalize_unfinished_parents("cancelled")
        self._parent_tasks.clear()
        self._active_tasks.clear()
        self._channel_tails.clear()
        self._channel_admission_locks.clear()
        self._accepting = True
        self._scope = LifecycleScope("translation-turns")

    async def _cancel_channel(
        self,
        channel: ChannelId,
        *,
        turn_kinds: frozenset[TranslationTurnKind] | None = None,
    ) -> None:
        if turn_kinds is None:
            self._blocked_channels.add(channel)
        try:
            admission_lock = self._channel_admission_locks.setdefault(
                channel,
                asyncio.Lock(),
            )
            async with admission_lock:
                selected_parents = tuple(
                    parent
                    for parent in self._parents.values()
                    if parent.channel == channel
                    and (
                        turn_kinds is None
                        or any(child.turn_kind in turn_kinds for child in parent.children)
                    )
                )
                if not selected_parents:
                    return
                self._cancelling_parent_ids.update(
                    parent.parent_utterance_id for parent in selected_parents
                )
                current_task = asyncio.current_task()
                tasks_to_await: list[asyncio.Task[None]] = []
                parents_to_await: list[_TranslationTurnParent] = []
                for parent in selected_parents:
                    parent_task = self._parent_tasks.get(parent.parent_utterance_id)
                    if parent_task is current_task:
                        continue
                    if parent_task is not None and not parent_task.done():
                        parent_task.cancel()
                        tasks_to_await.append(parent_task)
                    parents_to_await.append(parent)
                if tasks_to_await:
                    await asyncio.gather(*tasks_to_await, return_exceptions=True)
                await self._terminalize_selected_unfinished(selected_parents, "cancelled")
                if parents_to_await:
                    await asyncio.gather(
                        *(parent.closed_event.wait() for parent in parents_to_await)
                    )
        finally:
            if turn_kinds is None:
                self._blocked_channels.discard(channel)

    async def wait_for_idle(self) -> None:
        while self._parents or self._parent_tasks or self._active_tasks:
            await asyncio.sleep(0)

    async def wait_for_parent(self, parent_utterance_id: UUID) -> None:
        parent = self._parents.get(parent_utterance_id)
        if parent is not None:
            await parent.closed_event.wait()

    async def close(self) -> None:
        if self._closed:
            return
        self._advance_turn_generation("self")
        self._advance_turn_generation("peer")
        self._accepting = False
        self._closed = True
        self._request_cancellation()
        await self._drain_admission_locks()
        await self._scope.close()
        await self._terminalize_unfinished_parents("cancelled")
        self._parent_tasks.clear()
        self._active_tasks.clear()
        self._channel_tails.clear()
        self._channel_admission_locks.clear()

    def _build_children(
        self,
        request: TranslationTurnRequest,
        *,
        turn_generation: int,
        turn_order: int,
    ) -> tuple[TranslationTurnChild, ...]:
        segments = _final_transcript_segments(
            request.transcript,
            split_speakers=request.turn_kind == "peer",
        )
        child_specs = [
            (segment_index, target_index, segment, target_language)
            for segment_index, segment in enumerate(segments)
            if segment.text.strip()
            for target_index, target_language in enumerate(request.target_languages)
        ]
        nonempty_run_count = sum(1 for segment in segments if segment.text.strip())
        primary_uses_parent_identity = nonempty_run_count == 1 and request.turn_kind in {
            "manual",
            "self",
        }
        children: list[TranslationTurnChild] = []
        for sequence, (run_index, target_index, segment, target_language) in enumerate(child_specs):
            child_id = _translation_turn_child_id(
                request.transcript.utterance_id,
                turn_kind=request.turn_kind,
                run_index=run_index,
                target_index=target_index,
                run_language=segment.language,
                target_language=target_language,
                primary_uses_parent_identity=primary_uses_parent_identity,
            )
            children.append(
                TranslationTurnChild(
                    parent_utterance_id=request.transcript.utterance_id,
                    utterance_id=child_id,
                    sequence=sequence,
                    target_index=target_index,
                    turn_generation=turn_generation,
                    turn_order=turn_order,
                    transcript=Transcript(
                        utterance_id=child_id,
                        text=segment.text,
                        is_final=True,
                        created_at=request.transcript.created_at,
                        channel=request.transcript.channel,
                        final_language_runs=(FinalLanguageRun(segment.text, segment.language),),
                        final_speaker_runs=(
                            (
                                FinalSpeakerRun(
                                    segment.text,
                                    segment.speaker_id,
                                    segment.speaker_session_scope,
                                    source_start_ms=segment.source_start_ms,
                                    source_end_ms=segment.source_end_ms,
                                    speaker_confidence=segment.speaker_confidence,
                                    overlaps_previous=segment.overlaps_previous,
                                ),
                            )
                            if segment.speaker_session_scope
                            else ()
                        ),
                        publication_generation=request.transcript.publication_generation,
                        source_order=request.transcript.source_order,
                    ),
                    detected_language=segment.language or None,
                    target_language=target_language,
                    source=request.source,
                    turn_kind=request.turn_kind,
                    context_policy=self.policy.context_policy,
                    precomputed_translation=(
                        self._precomputed_translation_for_child(
                            request.precomputed_translation,
                            child_id=child_id,
                            parent_utterance_id=request.transcript.utterance_id,
                            target_language=target_language,
                        )
                        if request.precomputed_translation is not None and target_index == 0
                        else None
                    ),
                    parent_output_count=len(child_specs),
                    prestarted_translation=(
                        request.prestarted_secondary_translation if target_index == 1 else None
                    ),
                    config_snapshot=request.config_snapshot,
                )
            )
        return tuple(children)

    async def _run_parent(
        self,
        parent: _TranslationTurnParent,
        predecessor: _TranslationTurnParent | None,
    ) -> None:
        speech_slot = False
        try:
            is_dual_target_self = (
                parent.channel == "self"
                and len({child.target_language for child in parent.children}) > 1
            )
            is_self_speech = any(child.turn_kind == "self" for child in parent.children)
            if (
                parent.channel != "peer"
                and not is_dual_target_self
                and not is_self_speech
                and predecessor is not None
            ):
                self._observe_predecessor_wait(
                    "predecessor_wait_start",
                    parent=parent,
                    predecessor=predecessor,
                )
                await predecessor.semantic_done_event.wait()
                self._observe_predecessor_wait(
                    "predecessor_wait_end",
                    parent=parent,
                    predecessor=predecessor,
                )
            if is_self_speech:
                await self._self_speech_slots.acquire()
                speech_slot = True
            if parent.channel == "peer":
                if self._parent_cancellation_requested(parent):
                    raise asyncio.CancelledError
                await self._run_peer_children(parent, predecessor)
                return
            self._mark_parent_execution_started(parent)
            if is_dual_target_self:
                child_runners = tuple(
                    start_lifecycle_task(
                        self._scope,
                        self._run_child(child, None),
                        name=f"self-child-runner:{child.utterance_id}",
                        eager_start=True,
                    )
                    for child in parent.children
                    if child.utterance_id not in parent.completed_child_ids
                )
                await asyncio.gather(*child_runners)
                return
            for child in parent.children:
                if child.utterance_id in parent.completed_child_ids:
                    continue
                if self.is_child_cancellation_requested(child):
                    await self._terminalize_child(child, "cancelled")
                    continue
                await self._run_child(child, predecessor)
        except asyncio.CancelledError:
            await self._terminalize_parent_remaining(parent, "cancelled")
            raise
        except Exception:
            logger.exception("translation parent execution failed")
            await self._terminalize_parent_remaining(parent, "failed")
        finally:
            if speech_slot:
                self._self_speech_slots.release()
            self._parent_tasks.pop(parent.parent_utterance_id, None)

    @staticmethod
    def _precomputed_translation_for_child(
        translation: Translation,
        *,
        child_id: UUID,
        parent_utterance_id: UUID,
        target_language: str,
    ) -> Translation:
        return Translation(
            utterance_id=child_id,
            translated_text=translation.text,
            source_text=translation.source_text,
            source_language=translation.source_language,
            target_language=target_language,
            channel=translation.channel,
            created_at=translation.created_at,
            update_id=translation.update_id,
            origin_wall_clock_ms=translation.origin_wall_clock_ms,
            session_scope=translation.session_scope,
            source_text_hash=translation.source_text_hash,
            source_text_len=translation.source_text_len,
            logical_turn_key=f"{translation.channel}:{parent_utterance_id}",
        )

    def _observe_predecessor_wait(
        self,
        event: str,
        *,
        parent: _TranslationTurnParent,
        predecessor: _TranslationTurnParent,
    ) -> None:
        if self.predecessor_wait_observer is None:
            return
        self.predecessor_wait_observer(
            event,
            {
                "channel": parent.channel,
                "parent_utterance_id": str(parent.parent_utterance_id),
                "predecessor_utterance_id": str(predecessor.parent_utterance_id),
                "active_parent_count": len(self._parents),
            },
        )

    async def _run_child(
        self,
        child: TranslationTurnChild,
        predecessor: _TranslationTurnParent | None,
    ) -> None:
        child_task = start_lifecycle_task(
            self._scope,
            self._execute_started_child(child, predecessor),
            name=f"child:{child.utterance_id}",
            eager_start=True,
        )
        self._active_tasks[child.utterance_id] = child_task
        try:
            await child_task
        except asyncio.CancelledError:
            await self._terminalize_child(child, "cancelled")
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
        except Exception:
            if not child_task.done():
                child_task.cancel()
                await asyncio.gather(child_task, return_exceptions=True)
            logger.exception("translation child execution adapter failed")
            await self._terminalize_child(child, "failed")
        finally:
            self._active_tasks.pop(child.utterance_id, None)

    async def _execute_started_child(
        self,
        child: TranslationTurnChild,
        predecessor: _TranslationTurnParent | None,
    ) -> TranslationTurnProcessResult:
        child_task = asyncio.current_task()
        if child_task is None:
            raise RuntimeError("translation child task is unavailable")
        await self.on_child_started(child, child_task)
        return await self._execute_child(child, predecessor)

    async def _run_peer_children(
        self,
        parent: _TranslationTurnParent,
        predecessor: _TranslationTurnParent | None,
    ) -> None:
        children = tuple(
            child
            for child in parent.children
            if child.utterance_id not in parent.completed_child_ids
        )
        tasks: list[asyncio.Task[TranslationTurnProcessResult]] = []
        await self._acquire_peer_active_parent_slot(parent)
        try:
            if self._parent_cancellation_requested(parent):
                raise asyncio.CancelledError
            for child in children:
                task = start_lifecycle_task(
                    self._scope,
                    self._process_peer_child(parent, child),
                    name=f"peer-child:{child.utterance_id}",
                    eager_start=True,
                )
                tasks.append(task)
                self._active_tasks[child.utterance_id] = task
            for child, task in zip(children, tasks, strict=True):
                result = await task
                if self._parent_cancellation_requested(parent):
                    raise asyncio.CancelledError
                if predecessor is not None:
                    await predecessor.closed_event.wait()
                await self._publish_child_result(child, result)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for child in children:
                self._active_tasks.pop(child.utterance_id, None)
            await self._release_peer_active_parent_slot()

    async def _process_peer_child(
        self,
        parent: _TranslationTurnParent,
        child: TranslationTurnChild,
    ) -> TranslationTurnProcessResult:
        await self._acquire_peer_execution_slot(parent, child)
        try:
            if self.is_child_cancellation_requested(child):
                raise asyncio.CancelledError
            task = asyncio.current_task()
            if task is None:
                raise RuntimeError("Peer translation task is unavailable")
            try:
                await self.on_child_started(child, task)
                result = await self._process_child(child)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("translation child execution adapter failed")
                result = TranslationTurnProcessResult("failed")
            if self.is_child_cancellation_requested(child):
                raise asyncio.CancelledError
            self._mark_child_semantic_done(child)
            return result
        finally:
            await self._release_peer_execution_slot()

    async def _acquire_peer_active_parent_slot(
        self,
        parent: _TranslationTurnParent,
    ) -> None:
        limit = max(
            self.peer_waiting_capacity,
            parent.children[0].config_snapshot.value.concurrency_limit,
        )
        async with self._peer_execution_condition:
            await self._peer_execution_condition.wait_for(
                lambda: self._peer_active_parent_count < limit
            )
            if self._parent_cancellation_requested(parent):
                raise asyncio.CancelledError
            self._peer_active_parent_count += 1

    async def _release_peer_active_parent_slot(self) -> None:
        async with self._peer_execution_condition:
            self._peer_active_parent_count -= 1
            self._peer_execution_condition.notify_all()

    async def _acquire_peer_execution_slot(
        self,
        parent: _TranslationTurnParent,
        child: TranslationTurnChild,
    ) -> None:
        limit = child.config_snapshot.value.concurrency_limit
        async with self._peer_execution_condition:
            await self._peer_execution_condition.wait_for(lambda: self._peer_running_count < limit)
            if self.is_child_cancellation_requested(child):
                raise asyncio.CancelledError
            self._peer_running_count += 1
        self._mark_parent_execution_started(parent)

    async def _release_peer_execution_slot(self) -> None:
        async with self._peer_execution_condition:
            self._peer_running_count -= 1
            self._peer_execution_condition.notify_all()

    async def _execute_child(
        self,
        child: TranslationTurnChild,
        predecessor: _TranslationTurnParent | None,
    ) -> TranslationTurnProcessResult:
        result = await self._process_child(child)
        if self.is_child_cancellation_requested(child):
            await self._terminalize_child(child, "cancelled")
            raise asyncio.CancelledError
        self._mark_child_semantic_done(child)
        if predecessor is not None:
            await predecessor.closed_event.wait()
        return await self._publish_child_result(child, result)

    async def _publish_child_result(
        self,
        child: TranslationTurnChild,
        result: TranslationTurnProcessResult,
    ) -> TranslationTurnProcessResult:
        if self.is_child_cancellation_requested(child):
            raise asyncio.CancelledError
        if result.output is not None and self.output is not None:
            try:
                await self.output.submit_translation_output(result.output)
                self._output_submitted_child_ids.add(child.utterance_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("translation output submission failed")
                result = TranslationTurnProcessResult("failed")
        await self._terminalize_child(child, result.outcome)
        return result

    def _mark_child_semantic_done(self, child: TranslationTurnChild) -> None:
        parent = self._parents.get(child.parent_utterance_id)
        if parent is None or parent.closed:
            return
        parent.semantic_completed_child_ids.add(child.utterance_id)
        if parent.semantic_completed_child_ids == set(parent.child_ids):
            parent.semantic_done_event.set()

    async def _process_child(self, child: TranslationTurnChild) -> TranslationTurnProcessResult:
        try:
            result = await asyncio.wait_for(
                self.process_child(
                    child,
                    lambda: self.is_child_cancellation_requested(child),
                ),
                timeout=self.child_watchdog_s,
            )
        except TimeoutError:
            return TranslationTurnProcessResult("failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            return TranslationTurnProcessResult("failed")
        if isinstance(result, str):
            return TranslationTurnProcessResult(result)
        return result

    def _request_cancellation(self) -> None:
        self._cancelling_parent_ids.update(self._parents)
        current_task = asyncio.current_task()
        for parent_task in tuple(self._parent_tasks.values()):
            if parent_task is not current_task and not parent_task.done():
                parent_task.cancel()

    def _parent_cancellation_requested(self, parent: _TranslationTurnParent) -> bool:
        return (
            parent.closed
            or self._closed
            or not self._accepting
            or parent.channel in self._blocked_channels
            or parent.parent_utterance_id in self._cancelling_parent_ids
        )

    async def _drain_admission_locks(self) -> None:
        for admission_lock in tuple(self._channel_admission_locks.values()):
            async with admission_lock:
                pass

    def _peer_waiting_parents(self) -> list[_TranslationTurnParent]:
        return sorted(
            (
                parent
                for parent in self._parents.values()
                if parent.channel == "peer" and not parent.execution_started and not parent.closed
            ),
            key=lambda parent: (parent.admitted_at_monotonic_s, parent.turn_order),
        )

    def _self_speech_waiting_parents(self) -> list[_TranslationTurnParent]:
        return sorted(
            (
                parent
                for parent in self._parents.values()
                if parent.channel == "self"
                and not parent.closed
                and not parent.execution_started
                and any(child.turn_kind == "self" for child in parent.children)
            ),
            key=lambda parent: (parent.admitted_at_monotonic_s, parent.turn_order),
        )

    async def _expire_waiting_parent(self, parent: _TranslationTurnParent) -> None:
        loop = asyncio.get_running_loop()
        ttl_s = (
            self.self_speech_waiting_ttl_s
            if any(child.turn_kind == "self" for child in parent.children)
            else self.peer_waiting_ttl_s
        )
        deadline = parent.admitted_at_monotonic_s + ttl_s
        await asyncio.sleep(max(0.0, deadline - loop.time()))
        if parent.closed or parent.execution_started:
            return
        await self._retire_waiting_parent(
            parent,
            "source_only",
            failure_code="translation_timeout",
        )

    async def _retire_waiting_parent(
        self,
        parent: _TranslationTurnParent,
        outcome: TranslationTurnOutcome,
        *,
        failure_code: str,
    ) -> None:
        if parent.closed or parent.execution_started:
            return
        try:
            if outcome == "source_only" and self.output is not None:
                for child in parent.children:
                    if child.utterance_id in parent.completed_child_ids:
                        continue
                    try:
                        await self.output.submit_translation_output(
                            TranslationOutputSubmission(
                                parent_utterance_id=child.parent_utterance_id,
                                child_utterance_id=child.utterance_id,
                                sequence=child.sequence,
                                channel=child.channel,
                                source=child.source,
                                source_text=child.transcript.text,
                                source_language=child.detected_language,
                                target_language=child.target_language,
                                outcome="source_only",
                                config_snapshot=child.config_snapshot,
                                failure_code=failure_code,
                                target_index=child.target_index,
                                turn_generation=child.turn_generation,
                                turn_order=child.turn_order,
                                publication_generation=child.transcript.publication_generation,
                                source_order=child.transcript.source_order,
                                turn_kind=child.turn_kind,
                                parent_output_count=child.parent_output_count,
                            )
                        )
                    except Exception:
                        logger.exception("translation retirement output submission failed")
                    else:
                        self._output_submitted_child_ids.add(child.utterance_id)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            self._cancelling_parent_ids.add(parent.parent_utterance_id)
            try:
                await self._terminalize_parent_remaining(parent, outcome)
            finally:
                parent_task = self._parent_tasks.get(parent.parent_utterance_id)
                if (
                    parent_task is not None
                    and parent_task is not asyncio.current_task()
                    and not parent_task.done()
                ):
                    parent_task.cancel()
                    await asyncio.gather(parent_task, return_exceptions=True)
                if parent.closed:
                    self._cancelling_parent_ids.discard(parent.parent_utterance_id)

    def _mark_parent_execution_started(self, parent: _TranslationTurnParent) -> None:
        if parent.execution_started:
            return
        parent.execution_started = True
        task = parent.waiting_expiry_task
        parent.waiting_expiry_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _advance_turn_generation(self, channel: ChannelId) -> None:
        self._channel_turn_generations[channel] += 1
        self._channel_next_turn_orders[channel] = 0
        if self.turn_generation_observer is None:
            return
        try:
            self.turn_generation_observer(
                channel,
                self._channel_turn_generations[channel],
            )
        except Exception:
            logger.exception("translation turn generation observer failed")

    def child_output_was_submitted(self, child_utterance_id: UUID) -> bool:
        return child_utterance_id in self._output_submitted_child_ids

    async def _terminalize_unfinished_parents(
        self,
        outcome: TranslationTurnOutcome,
    ) -> None:
        await self._terminalize_selected_unfinished(tuple(self._parents.values()), outcome)

    async def _terminalize_selected_unfinished(
        self,
        parents: tuple[_TranslationTurnParent, ...],
        outcome: TranslationTurnOutcome,
    ) -> None:
        for parent in parents:
            await self._terminalize_parent_remaining(parent, outcome)

    async def _terminalize_parent_remaining(
        self,
        parent: _TranslationTurnParent,
        outcome: TranslationTurnOutcome,
    ) -> None:
        for child in parent.children:
            if child.utterance_id not in parent.completed_child_ids:
                await self._terminalize_child(child, outcome)

    async def _terminalize_child(
        self,
        child: TranslationTurnChild,
        outcome: TranslationTurnOutcome,
    ) -> None:
        parent = self._parents.get(child.parent_utterance_id)
        if parent is None or parent.closed or child.utterance_id in parent.completed_child_ids:
            return
        if (
            child.prestarted_translation is not None
            and child.prestarted_translation.task is not asyncio.current_task()
        ):
            await self._cancel_prestarted_translation(child.prestarted_translation)
        try:
            await self.on_child_terminal(child, outcome)
        except Exception:
            logger.exception("translation child terminal adapter failed")
        finally:
            self._output_submitted_child_ids.discard(child.utterance_id)
            parent.completed_child_ids.add(child.utterance_id)
            self._mark_child_semantic_done(child)
            if parent.completed_child_ids == set(parent.child_ids):
                await self._close_parent(parent)

    @staticmethod
    async def _cancel_prestarted_translation(
        prestarted: _PrestartedTranslation | None,
    ) -> None:
        if prestarted is None:
            return
        task = prestarted.task
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _close_parent(self, parent: _TranslationTurnParent) -> None:
        if parent.closed:
            return
        expiry_task = parent.waiting_expiry_task
        parent.waiting_expiry_task = None
        if (
            expiry_task is not None
            and expiry_task is not asyncio.current_task()
            and not expiry_task.done()
        ):
            expiry_task.cancel()
        parent.closed = True
        self._parents.pop(parent.parent_utterance_id, None)
        if self._channel_tails.get(parent.channel) is parent:
            self._channel_tails.pop(parent.channel, None)
        self._cancelling_parent_ids.discard(parent.parent_utterance_id)
        self._closed_parent_ids[parent.parent_utterance_id] = (
            parent.channel,
            parent.turn_generation,
            parent.turn_order,
        )
        while len(self._closed_parent_ids) > _COMPLETED_PARENT_LIMIT:
            self._closed_parent_ids.popitem(last=False)
        try:
            try:
                await self.on_parent_closed(parent.parent_utterance_id)
            except Exception:
                logger.exception("translation parent closure adapter failed")
        finally:
            parent.semantic_done_event.set()
            parent.closed_event.set()


__all__ = [
    "TranslationContextPolicy",
    "TranslationOutputSubmission",
    "TranslationOutputSubmissionPort",
    "TranslationRuntimePolicy",
    "TranslationTurnChild",
    "TranslationTurnKind",
    "TranslationTurnLifecycleOwner",
    "TranslationTurnOutcome",
    "TranslationTurnProcessResult",
    "TranslationTurnRequest",
]
