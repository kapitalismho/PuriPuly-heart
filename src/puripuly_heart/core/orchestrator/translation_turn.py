from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import UUID, uuid5

from puripuly_heart.core.diagnostics import emit_failure
from puripuly_heart.core.error_messages import provider_failure_report
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
    TranslationRuntimeConfigSnapshotPort,
)
from puripuly_heart.core.translation_policy import (
    TranslationContextPolicy,
    TranslationRuntimePolicy,
)
from puripuly_heart.domain.events import SelfTurnPresentation, SelfTurnTargetPresentation
from puripuly_heart.domain.models import (
    ChannelId,
    FinalLanguageRun,
    FinalSpeakerRun,
    SpeakerAssignment,
    Transcript,
    Translation,
)
from puripuly_heart.domain.recognition import slice_recognition_origins

if TYPE_CHECKING:
    from puripuly_heart.core.orchestrator.translation_request import PreparedTranslationRequest


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
    overlaps_previous: bool = False
    speaker_source: str = ""
    attribution_state: str = "unavailable"
    source_text_start: int = 0
    source_text_end: int = 0


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
        speaker_runs = (FinalSpeakerRun(text, None, "", attribution_state="malformed"),)
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
            speaker.overlaps_previous,
            speaker.source,
            speaker.attribution_state,
            start,
            end,
        )
        if (
            raw
            and raw[-1].language == segment.language
            and raw[-1].speaker_id == segment.speaker_id
            and raw[-1].speaker_session_scope == segment.speaker_session_scope
            and raw[-1].speaker_source == segment.speaker_source
            and raw[-1].attribution_state == segment.attribution_state
        ):
            previous = raw[-1]
            raw[-1] = replace(
                previous,
                text=previous.text + piece,
                source_text_end=end,
                source_end_ms=segment.source_end_ms,
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
                segments[-1] = replace(segments[-1], source_text_end=segment.source_text_end)
            else:
                leading += segment.text
            continue
        if leading:
            segment = replace(segment, text=leading + segment.text)
            leading = ""
            segment = replace(segment, source_text_start=0)
        segments.append(segment)
    if leading and segments:
        previous = segments[-1]
        segments[-1] = replace(previous, text=previous.text + leading)
        segments[-1] = replace(segments[-1], source_text_end=len(text))
    return tuple(segments)


@dataclass(frozen=True, slots=True)
class TerminalTranslationFailure:
    exception: Exception
    provider_generation: int
    utterance_id: UUID
    text: str
    channel: ChannelId
    source_language: str
    target_language: str
    configuration: TranslationRuntimeConfig


@dataclass(frozen=True, slots=True)
class TranslationTurnRequest:
    transcript: Transcript
    source: str
    turn_kind: TranslationTurnKind
    target_languages: tuple[str, ...]
    config_snapshot: TranslationRuntimeConfigSnapshot
    precomputed_translation: Translation | None = None
    precomputed_failure: TerminalTranslationFailure | None = field(
        default=None, repr=False, compare=False
    )
    operation_id: str | None = None
    request_id: str | None = None

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
    precomputed_failure: TerminalTranslationFailure | None = field(
        default=None, repr=False, compare=False
    )
    parent_output_count: int = 1
    operation_id: str | None = None
    request_id: str | None = None
    draft: bool = False
    provider_generation: int | None = None
    prepared_request: PreparedTranslationRequest | None = field(
        default=None, repr=False, compare=False
    )
    preparation_failure: Exception | None = field(default=None, repr=False, compare=False)
    canonical_attempt: bool = False

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
    speaker_assignment: SpeakerAssignment | None = None
    operation_id: str | None = None
    request_id: str | None = None
    failure_diagnostics: ErrorDiagnostics | None = None

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
        if self.channel != "peer" and self.speaker_assignment is not None:
            raise ValueError("speaker attribution is only valid for Peer output")


@dataclass(frozen=True, slots=True)
class TranslationTurnProcessResult:
    outcome: TranslationTurnOutcome
    output: TranslationOutputSubmission | None = None
    failure_diagnostics: ErrorDiagnostics | None = None
    terminal_failure: TerminalTranslationFailure | None = field(
        default=None, repr=False, compare=False
    )

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
    failure_diagnostics: ErrorDiagnostics | None = None
    source_closed: bool = True
    source_revision: int = 0
    presentation_revision: int = 0
    draft_request: TranslationTurnRequest | None = None
    source_closed_event: asyncio.Event = field(default_factory=asyncio.Event)
    results_changed: asyncio.Event = field(default_factory=asyncio.Event)
    operations: dict[int, asyncio.Task[TranslationTurnProcessResult]] = field(default_factory=dict)
    results: dict[int, TranslationTurnProcessResult] = field(default_factory=dict)
    result_revisions: dict[int, int] = field(default_factory=dict)
    speculative_targets: set[int] = field(default_factory=set)
    final_attempts: set[int] = field(default_factory=set)
    retiring_operations: set[asyncio.Task[TranslationTurnProcessResult]] = field(
        default_factory=set
    )
    secondary_eligible: bool = False
    speculative_occupancy: bool = False
    canonical_slot: bool = False
    terminal_status: Literal["cancelled", "expired", "failed"] | None = None

    def operation_retired(self, task: asyncio.Task[TranslationTurnProcessResult]) -> None:
        self.retiring_operations.discard(task)
        self.results_changed.set()

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
    on_draft_updated: (
        Callable[[SelfTurnPresentation, TranslationTurnRequest], Awaitable[None]] | None
    ) = None
    prepare_draft_child: Callable[[TranslationTurnChild], TranslationTurnChild] | None = None
    _speculative_parent_id: UUID | None = field(default=None, init=False)
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

    def is_draft_parent(self, parent_id: UUID) -> bool:
        parent = self._parents.get(parent_id)
        return parent is not None and parent.draft_request is not None

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

    async def revise_draft(
        self,
        request: TranslationTurnRequest,
        *,
        provider_generation: int,
        compatible: Callable[[str, str], bool],
        configuration_compatible: (
            Callable[[TranslationRuntimeConfig, TranslationRuntimeConfig], bool] | None
        ) = None,
    ) -> None:
        parent_id = request.transcript.utterance_id
        if not self.channel_ingress_open("self") or parent_id in self._closed_parent_ids:
            return
        parent = self._parents.get(parent_id)
        if parent is None:
            generation = self._channel_turn_generations["self"]
            order = self._channel_next_turn_orders["self"]
            self._channel_next_turn_orders["self"] = order + 1
            parent = _TranslationTurnParent(
                parent_id,
                "self",
                (),
                generation,
                order,
                0.0,
                source_closed=False,
                draft_request=request,
            )
            self._parents[parent_id] = parent
            task = start_lifecycle_task(
                self._scope,
                self._run_parent(parent, None),
                name=f"parent:{parent_id}",
                eager_start=True,
            )
            if not task.done():
                self._parent_tasks[parent_id] = task
        elif parent.source_closed:
            raise ValueError("cannot revise a source-closed turn")
        previous = parent.draft_request
        if previous is not None:
            request = replace(
                request,
                transcript=replace(request.transcript, created_at=previous.transcript.created_at),
            )
        children = tuple(
            replace(child, draft=True, provider_generation=provider_generation)
            for child in self._build_children(
                request, turn_generation=parent.turn_generation, turn_order=parent.turn_order
            )
        )
        reusable = bool(
            previous is not None
            and parent.children
            and (
                configuration_compatible(
                    previous.config_snapshot.value, request.config_snapshot.value
                )
                if configuration_compatible is not None
                else previous.config_snapshot.value == request.config_snapshot.value
            )
            and parent.children[0].provider_generation == provider_generation
            and compatible(previous.transcript.text, request.transcript.text)
            and previous.target_languages == request.target_languages
        )
        changed = previous is None or previous.transcript.text != request.transcript.text
        if not parent.children or changed or not reusable:
            parent.source_revision += 1
        if not reusable:
            for task in parent.operations.values():
                if not task.done():
                    task.cancel()
                    parent.retiring_operations.add(task)
                    task.add_done_callback(parent.operation_retired)
            parent.operations.clear()
            parent.results.clear()
            parent.speculative_targets.clear()
            parent.final_attempts.clear()
            parent.result_revisions.clear()
            parent.secondary_eligible = False
        else:
            for index in parent.results:
                parent.result_revisions[index] = parent.source_revision
        parent.children = children
        parent.draft_request = request
        await self._publish_draft(parent)
        self._start_current_draft()

    def prestart_draft_secondary(self, parent_id: UUID) -> None:
        parent = self._parents.get(parent_id)
        if (
            parent is None
            or parent.source_closed
            or parent.turn_generation != self._channel_turn_generations["self"]
            or self._parent_cancellation_requested(parent)
        ):
            return
        parent.secondary_eligible = True
        if parent.speculative_occupancy:
            self._start_draft_operations(parent, secondary=True)

    async def seal_draft(self, parent_id: UUID) -> tuple[UUID, ...]:
        parent = self._parents.get(parent_id)
        if parent is None or parent.draft_request is None or parent.source_closed:
            return ()
        parent.source_closed = True
        if self.prepare_draft_child is not None:
            parent.children = tuple(
                self.prepare_draft_child(
                    replace(child, prepared_request=None, canonical_attempt=True)
                )
                for child in parent.children
            )
        parent.admitted_at_monotonic_s = asyncio.get_running_loop().time()
        lock = self._channel_admission_locks.setdefault("self", asyncio.Lock())
        async with lock:
            if self.on_parent_admitted is not None:
                await self.on_parent_admitted(parent.children)
        if not self._self_speech_slots.locked():
            await self._self_speech_slots.acquire()
            parent.canonical_slot = True
            self._mark_parent_execution_started(parent)
            self._release_speculative_occupancy(parent)
        else:
            parent.waiting_expiry_task = start_lifecycle_task(
                self._scope,
                self._expire_waiting_parent(parent),
                name=f"self-speech-waiting-expiry:{parent_id}",
                eager_start=True,
            )
        for waiting in self._self_speech_waiting_parents()[: -self.self_speech_waiting_capacity]:
            if not waiting.speculative_occupancy:
                await self._retire_waiting_parent(
                    waiting, "source_only", failure_code="translation_overload"
                )
        await self._publish_draft(parent)
        parent.source_closed_event.set()
        return parent.child_ids

    def _start_current_draft(self) -> None:
        if not self.channel_ingress_open("self"):
            return
        parent = max(
            (
                candidate
                for candidate in self._parents.values()
                if candidate.channel == "self"
                and candidate.draft_request is not None
                and candidate.draft_request.turn_kind == "self"
                and candidate.children
                and not candidate.closed
                and candidate.turn_generation == self._channel_turn_generations["self"]
            ),
            key=lambda candidate: candidate.turn_order,
            default=None,
        )
        if (
            parent is None
            or parent.source_closed
            or self._parent_cancellation_requested(parent)
            or self._speculative_parent_id not in (None, parent.parent_utterance_id)
        ):
            return
        self._speculative_parent_id = parent.parent_utterance_id
        parent.speculative_occupancy = True
        self._start_draft_operations(parent, secondary=parent.secondary_eligible)

    def _release_speculative_occupancy(self, parent: _TranslationTurnParent) -> None:
        parent.speculative_occupancy = False
        if self._speculative_parent_id == parent.parent_utterance_id:
            self._speculative_parent_id = None
            self._start_current_draft()

    def _start_draft_operations(self, parent: _TranslationTurnParent, *, secondary: bool) -> None:
        for child in parent.children:
            index = child.target_index
            if index > 0 and not secondary:
                continue
            if index in parent.operations:
                continue
            if not child.canonical_attempt:
                parent.speculative_targets.add(index)
            if self.prepare_draft_child is not None and child.prepared_request is None:
                child = self.prepare_draft_child(child)
                parent.children = tuple(
                    child if candidate.target_index == index else candidate
                    for candidate in parent.children
                )
            task = start_lifecycle_task(
                self._scope,
                self._process_draft_operation(parent, child, parent.source_revision),
                name=f"child:{child.utterance_id}:{parent.source_revision}",
                eager_start=True,
            )
            parent.operations[index] = task

    async def _process_draft_operation(
        self,
        parent: _TranslationTurnParent,
        child: TranslationTurnChild,
        source_revision: int,
    ) -> TranslationTurnProcessResult:
        result = await self._process_child(child)
        current = parent.children[child.target_index]
        if (
            parent.closed
            or self.is_child_cancellation_requested(child)
            or current.provider_generation != child.provider_generation
            or (
                source_revision != parent.source_revision
                and parent.operations.get(child.target_index) is not asyncio.current_task()
            )
        ):
            return result
        parent.results[child.target_index] = result
        parent.result_revisions[child.target_index] = parent.source_revision
        if child.canonical_attempt:
            parent.final_attempts.add(child.target_index)
        await self._publish_draft(parent)
        parent.results_changed.set()
        return result

    async def _publish_draft(self, parent: _TranslationTurnParent) -> None:
        request = parent.draft_request
        if request is None or self.on_draft_updated is None:
            return
        parent.presentation_revision += 1
        targets = []
        for child in parent.children:
            result = parent.results.get(child.target_index)
            status = (
                (parent.terminal_status or "pending")
                if result is None
                else {
                    "translated": "ready",
                    "source_only": "source_only",
                    "failed": "failed",
                    "cancelled": "cancelled",
                }[result.outcome]
            )
            if (
                parent.terminal_status is None
                and parent.source_closed
                and result is not None
                and result.outcome == "failed"
                and child.target_index in parent.speculative_targets
                and child.target_index not in parent.final_attempts
            ):
                status = "pending"
            targets.append(
                SelfTurnTargetPresentation(
                    child.target_language,
                    parent.result_revisions.get(child.target_index, parent.source_revision),
                    status,
                    (
                        result.output.translation
                        if result is not None and result.output is not None and status == "ready"
                        else None
                    ),
                )
            )
        settled = bool(targets) and all(target.status != "pending" for target in targets)
        status = parent.terminal_status or (
            ("failed" if any(target.status == "failed" for target in targets) else "complete")
            if parent.source_closed and settled
            else "active"
        )
        await self.on_draft_updated(
            SelfTurnPresentation(
                parent.parent_utterance_id,
                parent.turn_generation,
                parent.turn_order,
                parent.presentation_revision,
                parent.source_revision,
                request.transcript.text,
                request.config_snapshot.value.source_language,
                parent.source_closed,
                status,
                tuple(targets),
                request.transcript.created_at,
            ),
            request,
        )

    async def _admit_draft_execution(self, parent: _TranslationTurnParent) -> None:
        if parent.canonical_slot:
            return
        if not parent.speculative_occupancy:
            await self._self_speech_slots.acquire()
            parent.canonical_slot = True
            return
        acquire = start_lifecycle_task(
            self._scope,
            self._self_speech_slots.acquire(),
            name=f"self-speech-slot:{parent.parent_utterance_id}",
            eager_start=True,
        )
        changed: asyncio.Task[bool] | None = None
        try:
            while True:
                if acquire.done():
                    await acquire
                    parent.canonical_slot = True
                    return
                if not parent.retiring_operations and all(
                    child.target_index in parent.results
                    and parent.results[child.target_index].outcome in {"translated", "source_only"}
                    for child in parent.children
                ):
                    return
                if (
                    parent.speculative_occupancy
                    and not parent.retiring_operations
                    and all(task.done() for task in parent.operations.values())
                ):
                    self._release_speculative_occupancy(parent)
                    for waiting in self._self_speech_waiting_parents()[
                        : -self.self_speech_waiting_capacity
                    ]:
                        await self._retire_waiting_parent(
                            waiting, "source_only", failure_code="translation_overload"
                        )
                    if parent.closed:
                        return
                    if (
                        asyncio.get_running_loop().time()
                        >= parent.admitted_at_monotonic_s + self.self_speech_waiting_ttl_s
                    ):
                        await self._retire_waiting_parent(
                            parent, "source_only", failure_code="translation_timeout"
                        )
                        return
                    if parent.waiting_expiry_task is None or parent.waiting_expiry_task.done():
                        parent.waiting_expiry_task = start_lifecycle_task(
                            self._scope,
                            self._expire_waiting_parent(parent),
                            name=f"self-speech-waiting-expiry:{parent.parent_utterance_id}",
                            eager_start=True,
                        )
                parent.results_changed.clear()
                changed = start_lifecycle_task(
                    self._scope,
                    parent.results_changed.wait(),
                    name=f"self-result-ready:{parent.parent_utterance_id}",
                    eager_start=True,
                )
                await asyncio.wait((acquire, changed), return_when=asyncio.FIRST_COMPLETED)
                if not changed.done():
                    changed.cancel()
                    await asyncio.gather(changed, return_exceptions=True)
                changed = None
        finally:
            if changed is not None and not changed.done():
                changed.cancel()
                await asyncio.gather(changed, return_exceptions=True)
            if not parent.canonical_slot:
                if not acquire.done():
                    acquire.cancel()
                await asyncio.gather(acquire, return_exceptions=True)
                if not acquire.cancelled() and acquire.exception() is None:
                    self._self_speech_slots.release()

    async def _run_draft_parent(self, parent: _TranslationTurnParent) -> None:
        try:
            await parent.source_closed_event.wait()
            if parent.closed or self._parent_cancellation_requested(parent):
                raise asyncio.CancelledError
            await self._admit_draft_execution(parent)
            if parent.closed:
                return
            self._mark_parent_execution_started(parent)
            self._release_speculative_occupancy(parent)
            self._start_draft_operations(parent, secondary=True)

            async def finish(child: TranslationTurnChild) -> None:
                result = await parent.operations[child.target_index]
                if result.outcome == "failed" and child.target_index in parent.speculative_targets:
                    result = await self._process_child(
                        replace(child, precomputed_failure=result.terminal_failure)
                    )
                    parent.results[child.target_index] = result
                    parent.result_revisions[child.target_index] = parent.source_revision
                    parent.final_attempts.add(child.target_index)
                    await self._publish_draft(parent)
                if result.output is not None:
                    result = replace(
                        result,
                        output=replace(
                            result.output,
                            source_text=child.transcript.text,
                        ),
                    )
                await self._publish_child_result(child, result)

            await asyncio.gather(*(finish(child) for child in parent.children))
        except asyncio.CancelledError:
            if parent.terminal_status is None:
                parent.terminal_status = "cancelled"
            await self._publish_draft(parent)
            await self._terminalize_parent_remaining(parent, "cancelled")
            raise
        except Exception as exc:
            self._record_parent_failure(
                parent, exc, "translation.parent_execution_failed", "parent_execution"
            )
            parent.terminal_status = "failed"
            try:
                await self._publish_draft(parent)
            finally:
                await self._terminalize_parent_remaining(parent, "failed")
        finally:
            tasks = tuple(parent.operations.values()) + tuple(parent.retiring_operations)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._release_speculative_occupancy(parent)
            if parent.canonical_slot:
                parent.canonical_slot = False
                self._self_speech_slots.release()
            self._parent_tasks.pop(parent.parent_utterance_id, None)

    async def submit(
        self,
        request: TranslationTurnRequest,
        *,
        wait_for_parent: bool = False,
    ) -> tuple[UUID, ...]:
        parent_id = request.transcript.utterance_id
        if parent_id in self._closed_parent_ids or parent_id in self._parents:
            await self._reject_parent(request)
            return ()
        if (
            not self._accepting
            or self._closed
            or request.transcript.channel in self._blocked_channels
        ):
            await self._reject_parent(request)
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
                        except Exception as exc:
                            self._record_parent_failure(
                                parent, exc, "translation.parent_admission_failed", "admission"
                            )
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
                except Exception as exc:
                    self._record_child_failure(
                        child, exc, "translation.child_creation_failed", stage="child_creation"
                    )
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
            if not isinstance(exc, asyncio.CancelledError):
                self._record_parent_failure(
                    parent, exc, "translation.parent_admission_failed", "admission"
                )
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

    async def _reject_parent(self, request: TranslationTurnRequest) -> None:
        try:
            await self.on_parent_rejected(request.transcript.utterance_id)
        except Exception as exc:
            diagnostics = replace(
                provider_failure_report(exc, provider="llm", operation="rejection").diagnostics,
                component="translation.turn",
                code="translation.parent_rejection_failed",
                channel=request.transcript.channel,
                turn_id=str(request.transcript.utterance_id),
                operation_id=request.operation_id,
                request_id=request.request_id,
            )
            emit_failure(
                logger, diagnostics, severity="warning", disposition="degraded", stage="rejection"
            )

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
                await asyncio.gather(*(parent.closed_event.wait() for parent in parents_to_await))
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
                                    overlaps_previous=segment.overlaps_previous,
                                    source=segment.speaker_source,
                                    attribution_state=segment.attribution_state,
                                ),
                            )
                            if segment.speaker_session_scope
                            or segment.attribution_state != "unavailable"
                            else ()
                        ),
                        publication_generation=request.transcript.publication_generation,
                        source_order=request.transcript.source_order,
                        source_text_range=(segment.source_text_start, segment.source_text_end),
                        source_text_revision=(
                            request.transcript.source_text_revision
                            or hashlib.sha256(request.transcript.text.encode("utf-8")).hexdigest()[
                                :16
                            ]
                        ),
                        recognition_origins=slice_recognition_origins(
                            request.transcript.recognition_origins,
                            segment.source_text_start,
                            segment.source_text_end,
                        ),
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
                    precomputed_failure=(
                        request.precomputed_failure if target_index == 0 else None
                    ),
                    parent_output_count=len(child_specs),
                    config_snapshot=request.config_snapshot,
                    operation_id=request.operation_id,
                    request_id=request.request_id,
                )
            )
        return tuple(children)

    async def _run_parent(
        self,
        parent: _TranslationTurnParent,
        predecessor: _TranslationTurnParent | None,
    ) -> None:
        if parent.draft_request is not None:
            await self._run_draft_parent(parent)
            return
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
        except Exception as exc:
            self._record_parent_failure(
                parent, exc, "translation.parent_execution_failed", "parent_execution"
            )
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
        except Exception as exc:
            if not child_task.done():
                child_task.cancel()
                await asyncio.gather(child_task, return_exceptions=True)
            self._record_child_failure(child, exc, "translation.child_execution_failed")
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
        running: dict[UUID, asyncio.Task[TranslationTurnProcessResult]] = {}
        finished: dict[UUID, TranslationTurnProcessResult] = {}
        await self._acquire_peer_active_parent_slot(parent)
        next_child = 0
        publish_index = 0
        predecessor_wait = (
            start_lifecycle_task(
                self._scope,
                predecessor.closed_event.wait(),
                name=f"peer-predecessor:{parent.parent_utterance_id}",
                eager_start=True,
            )
            if predecessor is not None and not predecessor.closed_event.is_set()
            else None
        )

        def start_next() -> None:
            nonlocal next_child
            child = children[next_child]
            task = start_lifecycle_task(
                self._scope,
                self._process_peer_child(parent, child),
                name=f"peer-child:{child.utterance_id}",
                eager_start=True,
            )
            running[child.utterance_id] = task
            self._active_tasks[child.utterance_id] = task
            next_child += 1

        try:
            if self._parent_cancellation_requested(parent):
                raise asyncio.CancelledError
            for _ in range(min(len(children), children[0].config_snapshot.value.concurrency_limit)):
                start_next()
            while publish_index < len(children):
                if self._parent_cancellation_requested(parent):
                    raise asyncio.CancelledError
                if predecessor is None or predecessor.closed_event.is_set():
                    while publish_index < len(children):
                        child = children[publish_index]
                        result = finished.pop(child.utterance_id, None)
                        if result is None:
                            break
                        await self._publish_child_result(child, result)
                        publish_index += 1
                    if publish_index == len(children):
                        break
                waiting: set[asyncio.Task[object]] = set(running.values())
                if predecessor_wait is not None and not predecessor_wait.done():
                    waiting.add(predecessor_wait)
                await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                for child_id, task in tuple(running.items()):
                    if not task.done():
                        continue
                    finished[child_id] = await task
                    running.pop(child_id)
                    self._active_tasks.pop(child_id, None)
                    if next_child < len(children):
                        start_next()
        finally:
            if predecessor_wait is not None and not predecessor_wait.done():
                predecessor_wait.cancel()
            for task in running.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                *running.values(),
                *((predecessor_wait,) if predecessor_wait is not None else ()),
                return_exceptions=True,
            )
            for child_id in running:
                self._active_tasks.pop(child_id, None)
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
            except Exception as exc:
                result = TranslationTurnProcessResult(
                    "failed",
                    failure_diagnostics=self._record_child_failure(
                        child, exc, "translation.child_execution_failed"
                    ),
                )
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
            except Exception as exc:
                result = TranslationTurnProcessResult(
                    "failed",
                    failure_diagnostics=self._record_child_failure(
                        child, exc, "translation.output_submission_failed", stage="output_submit"
                    ),
                )
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
        except TimeoutError as exc:
            return TranslationTurnProcessResult(
                "failed",
                failure_diagnostics=self._record_child_failure(
                    child, exc, "translation.child_timeout"
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return TranslationTurnProcessResult(
                "failed",
                failure_diagnostics=self._record_child_failure(
                    child, exc, "translation.child_execution_failed"
                ),
            )
        if isinstance(result, str):
            return TranslationTurnProcessResult(result)
        return result

    def _record_child_failure(
        self,
        child: TranslationTurnChild,
        exception: Exception,
        code: str,
        *,
        stage: str = "child_execution",
        secondary: bool = False,
    ) -> ErrorDiagnostics:
        native = provider_failure_report(exception, provider="llm", operation=stage).diagnostics
        preserved = isinstance(getattr(exception, "failure_diagnostics", None), ErrorDiagnostics)
        parent = self._parents.get(child.parent_utterance_id)
        diagnostics = replace(
            native,
            component=native.component if preserved else "translation.turn",
            code=native.code if preserved else code,
            channel=child.channel,
            turn_id=str(child.utterance_id),
            generation=child.turn_generation,
            operation_id=child.operation_id,
            request_id=child.request_id,
            cause_id=(
                parent.failure_diagnostics.diagnostic_id
                if secondary and parent is not None and parent.failure_diagnostics is not None
                else native.cause_id
            ),
            fields={
                **native.fields,
                "parent_utterance_id": str(child.parent_utterance_id),
                "target_index": child.target_index,
            },
        )
        if parent is not None and not secondary:
            parent.failure_diagnostics = parent.failure_diagnostics or diagnostics
        emit_failure(
            logger,
            diagnostics,
            stage=stage,
            severity="warning" if secondary else "error",
            disposition="degraded" if secondary else "failed",
        )
        return diagnostics

    def _record_parent_failure(
        self,
        parent: _TranslationTurnParent,
        exception: Exception,
        code: str,
        stage: str,
        *,
        secondary: bool = False,
    ) -> None:
        child = parent.children[0] if parent.children else None
        native = provider_failure_report(exception, provider="llm", operation=stage).diagnostics
        diagnostics = replace(
            native,
            component="translation.turn",
            code=code,
            channel=parent.channel,
            turn_id=str(parent.parent_utterance_id),
            generation=parent.turn_generation,
            operation_id=child.operation_id if child is not None else None,
            request_id=child.request_id if child is not None else None,
            cause_id=(
                parent.failure_diagnostics.diagnostic_id
                if secondary and parent.failure_diagnostics is not None
                else native.cause_id
            ),
        )
        if not secondary:
            parent.failure_diagnostics = parent.failure_diagnostics or diagnostics
        emit_failure(
            logger,
            diagnostics,
            stage=stage,
            severity="warning" if secondary else "error",
            disposition="degraded" if secondary else "failed",
        )

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
                and parent.source_closed
                and not parent.speculative_occupancy
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
        if parent.speculative_occupancy:
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
        if parent.draft_request is not None:
            parent.terminal_status = (
                "expired" if failure_code == "translation_timeout" else "failed"
            )
            if outcome == "source_only":
                parent.results = {
                    child.target_index: TranslationTurnProcessResult("source_only")
                    for child in parent.children
                }
                parent.result_revisions = {
                    child.target_index: parent.source_revision for child in parent.children
                }
            await self._publish_draft(parent)
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
                                operation_id=child.operation_id,
                                request_id=child.request_id,
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
        except Exception as exc:
            diagnostics = replace(
                provider_failure_report(
                    exc, provider="llm", operation="turn_generation"
                ).diagnostics,
                component="translation.turn",
                code="translation.turn_generation_observer_failed",
                channel=channel,
                generation=self._channel_turn_generations[channel],
            )
            emit_failure(
                logger,
                diagnostics,
                severity="warning",
                disposition="degraded",
                stage="turn_generation",
            )

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
        try:
            await self.on_child_terminal(child, outcome)
        except Exception as exc:
            self._record_child_failure(
                child,
                exc,
                "translation.child_retirement_failed",
                stage="retirement",
                secondary=True,
            )
        finally:
            self._output_submitted_child_ids.discard(child.utterance_id)
            parent.completed_child_ids.add(child.utterance_id)
            self._mark_child_semantic_done(child)
            if parent.completed_child_ids == set(parent.child_ids):
                await self._close_parent(parent)

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
            except Exception as exc:
                self._record_parent_failure(
                    parent,
                    exc,
                    "translation.parent_retirement_failed",
                    "retirement",
                    secondary=True,
                )
        finally:
            parent.semantic_done_event.set()
            parent.closed_event.set()


__all__ = [
    "TerminalTranslationFailure",
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
