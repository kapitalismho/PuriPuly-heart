from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, cast
from uuid import UUID

from puripuly_heart.core.clock import Clock
from puripuly_heart.core.orchestrator.configuration import TranslationRuntimeConfig
from puripuly_heart.core.orchestrator.context import ContextMode
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    LatencyStageDiagnostic,
    OverlayEmitDiagnostic,
    OverlaySinkDurationDiagnostic,
    RuntimeDiagnostic,
    SelfOverlayDecisionDiagnostic,
    TranslationLatencyDiagnosticsOwner,
    TranslationReadyDiagnostic,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationOutputSubmission,
    TranslationTurnChild,
    TranslationTurnOutcome,
)
from puripuly_heart.core.output.models import OutputRoutingDecision
from puripuly_heart.core.overlay.sink import OverlayEventAdapter, OverlayEventUnion, OverlaySink
from puripuly_heart.core.overlay.state import ActiveSelfOverlayMetadata
from puripuly_heart.core.runtime.output import OutputPublicationResult, OutputRuntime
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import ChannelId, Transcript, Translation

_SOFT_REUSE_PUNCT = {".", ",", "…", "。", "，", "、"}


class TranslationUiMessagePort(Protocol):
    async def publish(self, event: UIEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class TranslationUiMessage:
    event_type: UIEventType
    utterance_id: UUID | None = None
    payload: object | None = None
    source: str | None = None
    channel: ChannelId | None = None
    runtime_log_handled: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptOverlayProjection:
    transcript: Transcript
    source_language: str
    target_language: str
    event_kind: str | None = None


@dataclass(frozen=True, slots=True)
class TranslationOverlayProjection:
    translation: Translation
    source_language: str
    target_language: str
    applied_context_mode: ContextMode | None
    source_text: str = ""
    record_peer_first_emit: bool = False


@dataclass(frozen=True, slots=True)
class ChatboxProjection:
    utterance_id: UUID
    channel: ChannelId
    transcript_text: str
    translation_text: str | None
    include_source: bool
    source: str | None
    presentation_revision: int = 0
    turn_generation: int | None = None
    turn_order: int | None = None


@dataclass(frozen=True, slots=True)
class ActiveSelfProjection:
    merge_id: UUID
    active_text: str
    spec_text: str | None
    spec_translation: Translation | None
    source_language: str
    target_language: str
    resume_pending: bool
    resume_confirmed: bool
    created_at: float | None = None


@dataclass(frozen=True, slots=True)
class ActiveSelfProjectionReceipt:
    secondary_text: str
    source: str
    reuse_mode: str | None
    emitted: bool


@dataclass(frozen=True, slots=True)
class TranslationResultProjectionReceipt:
    clear_runtime_latency_bookkeeping: bool
    complete_peer_logical_turn: bool = False
    record_runtime_translation: bool = True


@dataclass(frozen=True, slots=True)
class SelfTranslationTurnSnapshot:
    parent_utterance_id: UUID
    turn_generation: int
    turn_order: int
    configured_targets: tuple[str, ...]
    completed_translations_by_target: tuple[tuple[str, str], ...]
    failed_targets: frozenset[str]
    revision: int
    terminal: bool
    admitted_at: float = 0.0
    first_success_at: float | None = None
    all_targets_terminal_at: float | None = None

    @property
    def text(self) -> str:
        return "\n".join(text for _target, text in self.completed_translations_by_target)


@dataclass(slots=True)
class _SelfTranslationTurnAggregate:
    parent_utterance_id: UUID
    turn_generation: int
    turn_order: int
    configured_targets: tuple[str, ...]
    expected_children_by_target: tuple[tuple[UUID, ...], ...]
    child_target_indexes: dict[UUID, int]
    child_sequences: dict[UUID, int]
    admitted_at: float
    translations_by_child: dict[UUID, Translation] = field(default_factory=dict)
    terminal_child_ids: set[UUID] = field(default_factory=set)
    failed_child_ids: set[UUID] = field(default_factory=set)
    first_success_at: float | None = None
    all_targets_terminal_at: float | None = None
    revision: int = 0
    visible_signature: tuple[tuple[str, str], ...] = ()
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@dataclass(frozen=True, slots=True)
class _SelfProjectionUpdate:
    accepted: bool
    snapshot: SelfTranslationTurnSnapshot | None = None


@dataclass(slots=True)
class TranslationUiMessageQueue:
    queue: asyncio.Queue[UIEvent] = field(repr=False)

    async def publish(self, event: UIEvent) -> None:
        await self.queue.put(event)


@dataclass(slots=True)
class TranslationOutputProjectionOwner:
    output_runtime: OutputRuntime
    ui_messages: TranslationUiMessagePort = field(repr=False)
    diagnostics: TranslationLatencyDiagnosticsOwner = field(repr=False)
    clock: Clock
    self_turn_tombstone_capacity: int = 4096
    _self_turns: dict[UUID, _SelfTranslationTurnAggregate] = field(default_factory=dict)
    _retired_self_turns: dict[tuple[int, UUID], _SelfTranslationTurnAggregate] = field(
        default_factory=dict
    )
    _self_turn_tombstones: set[tuple[int, UUID]] = field(default_factory=set)
    _self_turn_tombstone_order: deque[tuple[int, UUID]] = field(default_factory=deque)
    _active_self_turn_generation: int = -1
    _latest_visible_self_turn: tuple[int, int] | None = None
    _latest_visible_self_revision: int = 0
    _self_publish_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.self_turn_tombstone_capacity < 1:
            raise ValueError("self_turn_tombstone_capacity must be positive")

    @property
    def self_turn_aggregate_count(self) -> int:
        return len(self._self_turns)

    @property
    def self_turn_tombstone_count(self) -> int:
        return len(self._self_turn_tombstones)

    def retire_turn_generation(self, channel: ChannelId, turn_generation: int) -> None:
        if turn_generation < 0:
            raise ValueError("turn_generation must be non-negative")
        if channel != "self" or turn_generation <= self._active_self_turn_generation:
            return
        self._active_self_turn_generation = turn_generation
        for parent_utterance_id, aggregate in tuple(self._self_turns.items()):
            key = (aggregate.turn_generation, parent_utterance_id)
            self._retired_self_turns[key] = aggregate
            self._remember_self_turn_tombstone(
                aggregate.turn_generation,
                aggregate.parent_utterance_id,
            )
        self._self_turns.clear()

    def self_turn_presentation_revision(self, child: TranslationTurnChild) -> int:
        aggregate = self._self_turns.get(child.parent_utterance_id)
        if aggregate is None or not self._self_child_matches_aggregate(child, aggregate):
            aggregate = self._retired_self_turns.get(
                (child.turn_generation, child.parent_utterance_id)
            )
        if aggregate is None or not self._self_child_matches_aggregate(child, aggregate):
            return 0
        return aggregate.revision

    def admit_self_turn(self, children: tuple[TranslationTurnChild, ...]) -> bool:
        if not children:
            return False
        first = children[0]
        if first.channel != "self" or any(child.channel != "self" for child in children):
            raise ValueError("self projection admission requires Self children")
        if any(
            (child.parent_utterance_id, child.turn_generation, child.turn_order)
            != (first.parent_utterance_id, first.turn_generation, first.turn_order)
            for child in children
        ):
            raise ValueError("self projection admission identity mismatch")
        if first.turn_generation < self._active_self_turn_generation:
            return False
        if first.turn_generation > self._active_self_turn_generation:
            self.retire_turn_generation("self", first.turn_generation)
        target_pairs = tuple(
            sorted({(child.target_index, child.target_language.strip()) for child in children})
        )
        if tuple(index for index, _target in target_pairs) != tuple(range(len(target_pairs))):
            raise ValueError("self projection target indexes must be contiguous")
        configured_targets = tuple(target for _index, target in target_pairs)
        if len(configured_targets) < 2:
            return True
        tombstone_key = (first.turn_generation, first.parent_utterance_id)
        if tombstone_key in self._self_turn_tombstones:
            return False
        existing = self._self_turns.get(first.parent_utterance_id)
        if existing is not None:
            return (
                existing.turn_generation,
                existing.turn_order,
                existing.configured_targets,
            ) == (
                first.turn_generation,
                first.turn_order,
                configured_targets,
            )
        expected_children_by_target = tuple(
            tuple(
                child.utterance_id
                for child in sorted(children, key=lambda item: item.sequence)
                if child.target_index == target_index
            )
            for target_index in range(len(configured_targets))
        )
        aggregate = _SelfTranslationTurnAggregate(
            parent_utterance_id=first.parent_utterance_id,
            turn_generation=first.turn_generation,
            turn_order=first.turn_order,
            configured_targets=configured_targets,
            expected_children_by_target=expected_children_by_target,
            child_target_indexes={child.utterance_id: child.target_index for child in children},
            child_sequences={child.utterance_id: child.sequence for child in children},
            admitted_at=self.clock.now(),
        )
        self._self_turns[first.parent_utterance_id] = aggregate
        self.diagnostics.emit(
            RuntimeDiagnostic(
                message=(
                    "[Detailed][Translation] translation_turn_admitted "
                    "parent_utterance_id=%s turn_generation=%s turn_order=%s "
                    "target_count=%s presentation_revision=0"
                ),
                args=(
                    aggregate.parent_utterance_id,
                    aggregate.turn_generation,
                    aggregate.turn_order,
                    len(aggregate.configured_targets),
                ),
                detailed=True,
            )
        )
        return True

    def set_clock(self, clock: Clock) -> None:
        self.clock = clock
        self.output_runtime.clock = clock
        self.overlay_event_adapter.clock = clock

    @property
    def has_overlay_destination(self) -> bool:
        return self.output_runtime.has_overlay_destination

    @property
    def overlay_sink(self) -> OverlaySink | None:
        return self.output_runtime.overlay_sink

    @property
    def overlay_event_adapter(self) -> OverlayEventAdapter:
        adapter = self.output_runtime.overlay_event_adapter
        if adapter is None:
            raise RuntimeError("OutputRuntime overlay event adapter is unavailable")
        return adapter

    @property
    def routing_decisions(self) -> tuple[OutputRoutingDecision, ...]:
        return self.output_runtime.routing_decisions

    @property
    def output_state(self) -> str:
        return self.output_runtime.state

    @property
    def output_has_resources(self) -> bool:
        return self.output_runtime.has_resources

    @staticmethod
    def chatbox_is_eligible(channel: ChannelId) -> bool:
        return OutputRuntime.chatbox_is_eligible(channel)

    @staticmethod
    def chatbox_is_denied(channel: ChannelId) -> bool:
        return OutputRuntime.chatbox_is_denied(channel)

    async def publish_ui(self, message: TranslationUiMessage) -> None:
        await self.ui_messages.publish(
            UIEvent(
                type=message.event_type,
                utterance_id=message.utterance_id,
                payload=message.payload,
                source=message.source,
                channel=message.channel,
                runtime_log_handled=message.runtime_log_handled,
            )
        )

    def publish_system_immediate(self, text: str) -> OutputPublicationResult:
        return self.output_runtime.publish_system_immediate_chatbox(text=text)

    def publish_system_disclosure(self, text: str) -> OutputPublicationResult:
        self.diagnostics.emit(
            RuntimeDiagnostic(
                message="[Translation] OSC disclosure enqueue: channel=peer text_len=%s",
                args=(len(text),),
                fallback_level=logging.INFO,
                detailed=True,
            )
        )
        return self.output_runtime.publish_system_disclosure_chatbox(text=text)

    def set_self_chatbox_typing_reason(
        self,
        reason: str,
        active: bool,
    ) -> OutputPublicationResult:
        return self.output_runtime.set_self_chatbox_typing_reason(reason, active)

    def clear_self_chatbox_typing_reasons(self) -> OutputPublicationResult:
        return self.output_runtime.clear_self_chatbox_typing_reasons()

    async def replace_overlay_sink(
        self,
        overlay_sink: OverlaySink | None,
        *,
        expected_current: OverlaySink | None = None,
        require_match: bool = False,
    ) -> bool:
        return await self.output_runtime.replace_overlay_sink(
            overlay_sink,
            expected_current=expected_current,
            require_match=require_match,
        )

    def active_self_overlay_metadata(self) -> ActiveSelfOverlayMetadata | None:
        metadata = self.output_runtime.active_self_overlay_metadata()
        if metadata is None:
            return None
        return cast(ActiveSelfOverlayMetadata, metadata)

    async def emit_final_transcript(
        self,
        projection: TranscriptOverlayProjection,
    ) -> None:
        if not self.has_overlay_destination:
            return
        if projection.event_kind is not None:
            self._record_overlay_emit(
                event_kind=projection.event_kind,
                utterance_id=projection.transcript.utterance_id,
                channel=projection.transcript.channel,
                secondary_len=len(projection.transcript.text.strip()),
            )
        await self.publish_overlay_event(
            self.overlay_event_adapter.transcript_final(
                projection.transcript,
                source_language=projection.source_language,
                target_language=projection.target_language,
            )
        )

    async def project_self_final_transcript(
        self,
        *,
        transcript: Transcript,
        source_language: str,
        target_language: str,
        translation_will_follow: bool,
    ) -> bool:
        source_language, target_language = self.self_overlay_languages_for_utterance(
            utterance_id=transcript.utterance_id,
            source_language=source_language,
            target_language=target_language,
        )
        await self.emit_final_transcript(
            TranscriptOverlayProjection(
                transcript=transcript,
                source_language=source_language,
                target_language=target_language,
            )
        )
        if translation_will_follow:
            return False
        return await self.close_overlay_utterance(
            utterance_id=transcript.utterance_id,
            channel=transcript.channel,
            is_final=True,
        )

    async def project_peer_source_only(
        self,
        *,
        transcript: Transcript,
        source_language: str,
        target_language: str,
        close_is_final: bool,
        finalize_latency: bool,
    ) -> bool:
        if self.has_overlay_destination:
            self.diagnostics.record_latency_stage(
                LatencyStageDiagnostic(
                    channel="peer",
                    utterance_id=transcript.utterance_id,
                    stage="peer_overlay_first_emit",
                    overwrite=False,
                )
            )
            await self.emit_final_transcript(
                TranscriptOverlayProjection(
                    transcript=transcript,
                    source_language=source_language,
                    target_language=target_language,
                    event_kind="peer_transcript_final",
                )
            )
        return await self.close_overlay_utterance(
            utterance_id=transcript.utterance_id,
            channel="peer",
            is_final=close_is_final,
            finalize_latency=finalize_latency,
        )

    async def emit_translation(
        self,
        projection: TranslationOverlayProjection,
    ) -> None:
        if not self.has_overlay_destination:
            return
        translation = projection.translation
        self._record_overlay_emit(
            event_kind="translation_final",
            utterance_id=translation.utterance_id,
            channel=translation.channel,
            secondary_len=len(translation.text.strip()),
        )
        if projection.record_peer_first_emit:
            self.diagnostics.record_latency_stage(
                LatencyStageDiagnostic(
                    channel=translation.channel,
                    utterance_id=translation.utterance_id,
                    stage="peer_overlay_first_emit",
                    overwrite=False,
                )
            )
        await self.publish_overlay_event(
            self.overlay_event_adapter.translation_final(
                utterance_id=translation.utterance_id,
                channel=translation.channel,
                text=translation.text,
                source_text=projection.source_text,
                source_language=projection.source_language,
                target_language=projection.target_language,
                applied_context_mode=projection.applied_context_mode,
                created_at=translation.created_at,
                **self._translation_metadata(translation),
            )
        )

    async def close_overlay_utterance(
        self,
        *,
        utterance_id: UUID,
        channel: ChannelId,
        is_final: bool,
        finalize_latency: bool | None = None,
    ) -> bool:
        should_finalize = finalize_latency is True or (
            finalize_latency is None and channel == "peer"
        )
        if self.has_overlay_destination:
            await self.publish_overlay_event(
                self.overlay_event_adapter.utterance_closed(
                    utterance_id=utterance_id,
                    channel=channel,
                    is_final=is_final,
                )
            )
        if should_finalize:
            self.diagnostics.clear_latency_timeline(channel, utterance_id)
        return should_finalize

    async def publish_overlay_event(self, event: OverlayEventUnion) -> None:
        if not self.has_overlay_destination:
            return
        detailed_mode = self.diagnostics.detailed_enabled
        start = time.perf_counter() if detailed_mode else 0.0
        result = await self.output_runtime.publish_overlay_event(event)
        if result.decision.reason == "destination_publish_failed":
            self.diagnostics.record_overlay_sink_failure(
                result.decision.metadata.get("error_type", "Exception")
            )
            return
        if detailed_mode and result.decision.decision == "published":
            elapsed_ms = max(0, int((time.perf_counter() - start) * 1000))
            self.diagnostics.record_overlay_sink_duration(
                OverlaySinkDurationDiagnostic(
                    event_type=type(event).__name__,
                    channel=getattr(event, "channel", None),
                    utterance_id=getattr(event, "utterance_id", None),
                    update_id=getattr(event, "update_id", None),
                    elapsed_ms=elapsed_ms,
                )
            )

    async def sync_active_self(
        self,
        projection: ActiveSelfProjection,
    ) -> ActiveSelfProjectionReceipt:
        if not self.has_overlay_destination or not projection.active_text:
            return ActiveSelfProjectionReceipt("", "blank", None, False)
        metadata = self.active_self_overlay_metadata()
        reuse_mode = (
            self.soft_reuse_mode(projection.spec_text, projection.active_text)
            if projection.spec_translation is not None
            else None
        )
        secondary_text, source = self._active_secondary_decision(
            projection,
            metadata,
            reuse_mode=reuse_mode,
        )
        self.diagnostics.record_self_overlay_decision(
            SelfOverlayDecisionDiagnostic.create(
                merge_id=projection.merge_id,
                source=source,
                active_text=projection.active_text,
                secondary_text=secondary_text,
                spec_text_len=len((projection.spec_text or "").strip()),
                spec_translation_len=(
                    len(projection.spec_translation.text.strip())
                    if projection.spec_translation is not None
                    else 0
                ),
                cached_secondary_len=(
                    len(metadata.secondary_text.strip()) if metadata is not None else 0
                ),
                reuse_mode=reuse_mode,
                resume_pending=projection.resume_pending,
                resume_confirmed=projection.resume_confirmed,
            )
        )
        occupant_key = f"self:{projection.merge_id}"
        translation_metadata = self._secondary_translation_metadata(
            projection,
            source=source,
            secondary_text=secondary_text,
            metadata=metadata,
        )
        source_language, target_language = self._active_languages(
            projection,
            source=source,
            secondary_text=secondary_text,
            metadata=metadata,
        )
        primary_language = source_language.strip() or None
        secondary_language = (target_language.strip() or None) if secondary_text.strip() else None
        if (
            metadata is not None
            and projection.merge_id == metadata.utterance_id
            and occupant_key == metadata.occupant_key
            and projection.active_text == metadata.text
            and secondary_text == metadata.secondary_text
            and primary_language == metadata.primary_language
            and secondary_language == metadata.secondary_language
            and translation_metadata == self._active_translation_metadata(metadata)
        ):
            return ActiveSelfProjectionReceipt(
                secondary_text,
                source,
                reuse_mode,
                False,
            )
        self._record_overlay_emit(
            event_kind="active_self",
            utterance_id=projection.merge_id,
            channel="self",
            secondary_len=len(secondary_text),
        )
        await self.publish_overlay_event(
            self.overlay_event_adapter.self_active_update(
                text=projection.active_text,
                utterance_id=projection.merge_id,
                secondary_text=secondary_text,
                occupant_key=occupant_key,
                source_language=source_language,
                target_language=target_language,
                created_at=projection.created_at,
                **translation_metadata,
            )
        )
        return ActiveSelfProjectionReceipt(
            secondary_text,
            source,
            reuse_mode,
            True,
        )

    def soft_reuse_mode(self, spec_text: str | None, final_text: str) -> str | None:
        if spec_text is None:
            return None
        if spec_text == final_text:
            return "exact"
        normalized_spec = self.normalize_soft_reuse_text(spec_text)
        normalized_final = self.normalize_soft_reuse_text(final_text)
        if not normalized_spec or not normalized_final:
            return None
        if normalized_spec == normalized_final:
            return "soft_boundary"
        return None

    async def reset_overlay_preview(self) -> None:
        if self.active_self_overlay_metadata() is None or not self.has_overlay_destination:
            return
        await self.publish_overlay_event(self.overlay_event_adapter.self_active_clear())

    async def blank_active_self(
        self,
        *,
        utterance_id: UUID,
        text: str,
        source_language: str,
        target_language: str,
        created_at: float,
    ) -> None:
        self._record_overlay_emit(
            event_kind="active_self",
            utterance_id=utterance_id,
            channel="self",
            secondary_len=0,
        )
        await self.publish_overlay_event(
            self.overlay_event_adapter.self_active_update(
                text=text,
                utterance_id=utterance_id,
                secondary_text="",
                occupant_key=f"self:{utterance_id}",
                source_language=source_language,
                target_language=target_language,
                created_at=created_at,
            )
        )

    def should_blank_stale_active_secondary(
        self,
        *,
        final_text: str,
        reuse_mode: str | None,
    ) -> bool:
        metadata = self.active_self_overlay_metadata()
        return (
            reuse_mode is None
            and self.has_overlay_destination
            and metadata is not None
            and metadata.text == final_text
            and bool(metadata.secondary_text.strip())
        )

    def self_overlay_languages_for_utterance(
        self,
        *,
        utterance_id: UUID,
        source_language: str,
        target_language: str,
    ) -> tuple[str, str]:
        metadata = self.active_self_overlay_metadata()
        if (
            metadata is None
            or metadata.utterance_id != utterance_id
            or metadata.occupant_key != f"self:{utterance_id}"
        ):
            return source_language, target_language
        return (
            self._language_or_fallback(metadata.primary_language, source_language),
            self._language_or_fallback(metadata.secondary_language, target_language),
        )

    async def complete_self_target(
        self,
        child: TranslationTurnChild,
        outcome: TranslationTurnOutcome,
    ) -> None:
        aggregate = self._self_turns.get(child.parent_utterance_id)
        retired_key = (child.turn_generation, child.parent_utterance_id)
        if aggregate is None or not self._self_child_matches_aggregate(child, aggregate):
            aggregate = self._retired_self_turns.get(retired_key)
        if aggregate is None or not self._self_child_matches_aggregate(child, aggregate):
            return
        async with aggregate.lock:
            if not self._self_child_matches_aggregate(child, aggregate):
                return
            aggregate.terminal_child_ids.add(child.utterance_id)
            if (
                outcome != "translated"
                and child.utterance_id not in aggregate.translations_by_child
            ):
                aggregate.failed_child_ids.add(child.utterance_id)
            terminal = self._self_aggregate_terminal(aggregate)
            if terminal and aggregate.all_targets_terminal_at is None:
                aggregate.all_targets_terminal_at = self.clock.now()
            self._refresh_self_snapshot(aggregate)
            presentation_revision = aggregate.revision
            first_success_elapsed_ms = self._elapsed_ms(
                aggregate.admitted_at,
                aggregate.first_success_at,
            )
            all_targets_terminal_elapsed_ms = self._elapsed_ms(
                aggregate.admitted_at,
                aggregate.all_targets_terminal_at,
            )
        self.diagnostics.emit(
            RuntimeDiagnostic(
                message=(
                    "[Detailed][Translation] translation_target_completed "
                    "parent_utterance_id=%s turn_generation=%s turn_order=%s "
                    "target_index=%s target_language=%s outcome=%s "
                    "presentation_revision=%s first_success_elapsed_ms=%s "
                    "all_targets_terminal_elapsed_ms=%s"
                ),
                args=(
                    child.parent_utterance_id,
                    child.turn_generation,
                    child.turn_order,
                    child.target_index,
                    child.target_language,
                    outcome,
                    presentation_revision,
                    first_success_elapsed_ms,
                    all_targets_terminal_elapsed_ms,
                ),
                detailed=True,
            )
        )
        if terminal and self._self_turns.get(child.parent_utterance_id) is aggregate:
            self._self_turns.pop(child.parent_utterance_id, None)
            self._remember_self_turn_tombstone(
                aggregate.turn_generation,
                aggregate.parent_utterance_id,
            )
        if terminal and self._retired_self_turns.get(retired_key) is aggregate:
            self._retired_self_turns.pop(retired_key, None)

    async def _record_self_submission(
        self,
        submission: TranslationOutputSubmission,
    ) -> _SelfProjectionUpdate:
        aggregate = self._self_turns.get(submission.parent_utterance_id)
        if aggregate is None:
            retired = self._retired_self_turns.get(
                (submission.turn_generation, submission.parent_utterance_id)
            )
            if retired is not None and self._self_submission_matches_aggregate(submission, retired):
                self._emit_self_stale_submission(submission, retired.revision)
            return _SelfProjectionUpdate(False)
        async with aggregate.lock:
            if (
                submission.turn_generation != self._active_self_turn_generation
                or not self._self_submission_matches_aggregate(submission, aggregate)
                or submission.child_utterance_id in aggregate.terminal_child_ids
            ):
                if submission.turn_generation != self._active_self_turn_generation:
                    self._emit_self_stale_submission(submission, aggregate.revision)
                return _SelfProjectionUpdate(False)
            aggregate.terminal_child_ids.add(submission.child_utterance_id)
            if submission.outcome == "translated":
                translation = submission.translation
                if translation is None:
                    return _SelfProjectionUpdate(False)
                aggregate.translations_by_child[submission.child_utterance_id] = translation
                aggregate.failed_child_ids.discard(submission.child_utterance_id)
            else:
                aggregate.failed_child_ids.add(submission.child_utterance_id)
            if (
                self._self_aggregate_terminal(aggregate)
                and aggregate.all_targets_terminal_at is None
            ):
                aggregate.all_targets_terminal_at = self.clock.now()
            snapshot = self._refresh_self_snapshot(aggregate)
            if snapshot is not None:
                await self._publish_self_snapshot(snapshot, source=submission.source)
            if submission.turn_generation != self._active_self_turn_generation:
                return _SelfProjectionUpdate(False)
        return _SelfProjectionUpdate(True, snapshot)

    def _refresh_self_snapshot(
        self,
        aggregate: _SelfTranslationTurnAggregate,
    ) -> SelfTranslationTurnSnapshot | None:
        completed: list[tuple[str, str]] = []
        failed: set[str] = set()
        for target_index, child_ids in enumerate(aggregate.expected_children_by_target):
            if not set(child_ids).issubset(aggregate.terminal_child_ids):
                continue
            target = aggregate.configured_targets[target_index]
            if any(child_id in aggregate.failed_child_ids for child_id in child_ids):
                failed.add(target)
                continue
            if any(child_id not in aggregate.translations_by_child for child_id in child_ids):
                failed.add(target)
                continue
            translated_parts = tuple(
                aggregate.translations_by_child[child_id].text.strip()
                for child_id in child_ids
                if aggregate.translations_by_child[child_id].text.strip()
            )
            if translated_parts:
                completed.append((target, " ".join(translated_parts)))
            else:
                failed.add(target)
        signature = tuple(completed)
        if signature == aggregate.visible_signature:
            return None
        aggregate.visible_signature = signature
        aggregate.revision += 1
        if signature and aggregate.first_success_at is None:
            aggregate.first_success_at = self.clock.now()
        return SelfTranslationTurnSnapshot(
            parent_utterance_id=aggregate.parent_utterance_id,
            turn_generation=aggregate.turn_generation,
            turn_order=aggregate.turn_order,
            configured_targets=aggregate.configured_targets,
            completed_translations_by_target=signature,
            failed_targets=frozenset(failed),
            revision=aggregate.revision,
            terminal=self._self_aggregate_terminal(aggregate),
            admitted_at=aggregate.admitted_at,
            first_success_at=aggregate.first_success_at,
            all_targets_terminal_at=aggregate.all_targets_terminal_at,
        )

    async def _publish_self_snapshot(
        self,
        snapshot: SelfTranslationTurnSnapshot,
        *,
        source: str | None,
    ) -> OutputPublicationResult | None:
        if not snapshot.text:
            return None
        turn_key = (snapshot.turn_generation, snapshot.turn_order)
        async with self._self_publish_lock:
            if snapshot.turn_generation != self._active_self_turn_generation:
                self._emit_self_stale_snapshot(snapshot)
                return None
            latest = self._latest_visible_self_turn
            if latest is not None and turn_key < latest:
                self._emit_self_stale_snapshot(snapshot)
                return None
            if latest == turn_key and snapshot.revision <= self._latest_visible_self_revision:
                return None
            target_languages = tuple(
                target for target, _text in snapshot.completed_translations_by_target
            )
            target_indexes = tuple(
                snapshot.configured_targets.index(target) for target in target_languages
            )
            stage = (
                "translation_complete_result_published"
                if len(snapshot.completed_translations_by_target)
                == len(snapshot.configured_targets)
                else "translation_first_result_published"
            )
            visible_elapsed_ms = self._elapsed_ms(snapshot.admitted_at, self.clock.now())
            result = await self.publish_chatbox(
                ChatboxProjection(
                    utterance_id=snapshot.parent_utterance_id,
                    channel="self",
                    transcript_text="",
                    translation_text=snapshot.text,
                    include_source=False,
                    source=source,
                    presentation_revision=snapshot.revision,
                    turn_generation=snapshot.turn_generation,
                    turn_order=snapshot.turn_order,
                )
            )
            if result.decision.decision != "published":
                self.diagnostics.emit(
                    RuntimeDiagnostic(
                        message=(
                            "[Detailed][Translation] %s parent_utterance_id=%s "
                            "turn_generation=%s turn_order=%s target_indexes=%s "
                            "target_languages=%s revision=%s terminal=%s reason=%s "
                            "first_success_elapsed_ms=%s "
                            "all_targets_terminal_elapsed_ms=%s "
                            "publication_attempt_elapsed_ms=%s "
                            "first_visible_elapsed_ms=None "
                            "complete_visible_elapsed_ms=None"
                        ),
                        args=(
                            stage.replace("_published", "_publication_denied"),
                            snapshot.parent_utterance_id,
                            snapshot.turn_generation,
                            snapshot.turn_order,
                            target_indexes,
                            target_languages,
                            snapshot.revision,
                            snapshot.terminal,
                            result.decision.reason,
                            self._elapsed_ms(
                                snapshot.admitted_at,
                                snapshot.first_success_at,
                            ),
                            self._elapsed_ms(
                                snapshot.admitted_at,
                                snapshot.all_targets_terminal_at,
                            ),
                            visible_elapsed_ms,
                        ),
                        detailed=True,
                    )
                )
                return result
            self._latest_visible_self_turn = turn_key
            self._latest_visible_self_revision = snapshot.revision
            self.diagnostics.emit(
                RuntimeDiagnostic(
                    message=(
                        "[Detailed][Translation] %s parent_utterance_id=%s "
                        "turn_generation=%s turn_order=%s target_indexes=%s "
                        "target_languages=%s revision=%s terminal=%s "
                        "first_success_elapsed_ms=%s all_targets_terminal_elapsed_ms=%s "
                        "first_visible_elapsed_ms=%s complete_visible_elapsed_ms=%s"
                    ),
                    args=(
                        stage,
                        snapshot.parent_utterance_id,
                        snapshot.turn_generation,
                        snapshot.turn_order,
                        target_indexes,
                        target_languages,
                        snapshot.revision,
                        snapshot.terminal,
                        self._elapsed_ms(snapshot.admitted_at, snapshot.first_success_at),
                        self._elapsed_ms(
                            snapshot.admitted_at,
                            snapshot.all_targets_terminal_at,
                        ),
                        (
                            visible_elapsed_ms
                            if stage == "translation_first_result_published"
                            else None
                        ),
                        (
                            visible_elapsed_ms
                            if stage == "translation_complete_result_published"
                            else None
                        ),
                    ),
                    detailed=True,
                )
            )
            return result

    def _emit_self_stale_snapshot(self, snapshot: SelfTranslationTurnSnapshot) -> None:
        target_languages = tuple(
            target for target, _text in snapshot.completed_translations_by_target
        )
        self.diagnostics.emit(
            RuntimeDiagnostic(
                message=(
                    "[Detailed][Translation] translation_result_suppressed_stale_turn "
                    "parent_utterance_id=%s turn_generation=%s turn_order=%s "
                    "target_indexes=%s target_languages=%s revision=%s"
                ),
                args=(
                    snapshot.parent_utterance_id,
                    snapshot.turn_generation,
                    snapshot.turn_order,
                    tuple(snapshot.configured_targets.index(target) for target in target_languages),
                    target_languages,
                    snapshot.revision,
                ),
                detailed=True,
            )
        )

    def _emit_self_stale_submission(
        self,
        submission: TranslationOutputSubmission,
        presentation_revision: int,
    ) -> None:
        self.diagnostics.emit(
            RuntimeDiagnostic(
                message=(
                    "[Detailed][Translation] translation_result_suppressed_stale_turn "
                    "parent_utterance_id=%s turn_generation=%s turn_order=%s "
                    "target_indexes=%s target_languages=%s revision=%s"
                ),
                args=(
                    submission.parent_utterance_id,
                    submission.turn_generation,
                    submission.turn_order,
                    (submission.target_index,),
                    (submission.target_language,),
                    presentation_revision,
                ),
                detailed=True,
            )
        )

    @staticmethod
    def _self_submission_matches_aggregate(
        submission: TranslationOutputSubmission,
        aggregate: _SelfTranslationTurnAggregate,
    ) -> bool:
        target_index = aggregate.child_target_indexes.get(submission.child_utterance_id)
        return (
            submission.turn_generation == aggregate.turn_generation
            and submission.turn_order == aggregate.turn_order
            and target_index == submission.target_index
            and target_index is not None
            and aggregate.configured_targets[target_index] == submission.target_language.strip()
        )

    @staticmethod
    def _self_child_matches_aggregate(
        child: TranslationTurnChild,
        aggregate: _SelfTranslationTurnAggregate,
    ) -> bool:
        return (
            child.turn_generation == aggregate.turn_generation
            and child.turn_order == aggregate.turn_order
            and aggregate.child_target_indexes.get(child.utterance_id) == child.target_index
        )

    @staticmethod
    def _self_aggregate_terminal(aggregate: _SelfTranslationTurnAggregate) -> bool:
        return set(aggregate.child_target_indexes).issubset(aggregate.terminal_child_ids)

    @staticmethod
    def _elapsed_ms(started_at: float, completed_at: float | None) -> int | None:
        if completed_at is None:
            return None
        return max(0, int((completed_at - started_at) * 1000))

    def _remember_self_turn_tombstone(
        self,
        turn_generation: int,
        parent_utterance_id: UUID,
    ) -> None:
        key = (turn_generation, parent_utterance_id)
        if key in self._self_turn_tombstones:
            return
        if len(self._self_turn_tombstone_order) >= self.self_turn_tombstone_capacity:
            evicted = self._self_turn_tombstone_order.popleft()
            self._self_turn_tombstones.discard(evicted)
            self._retired_self_turns.pop(evicted, None)
        self._self_turn_tombstone_order.append(key)
        self._self_turn_tombstones.add(key)

    def emit_translation_ready(
        self,
        translation: Translation,
        *,
        parent_utterance_id: UUID | None = None,
        target_index: int | None = None,
        turn_generation: int | None = None,
        turn_order: int | None = None,
    ) -> bool:
        return self.diagnostics.emit_translation_ready(
            TranslationReadyDiagnostic(
                channel=translation.channel,
                utterance_id=translation.utterance_id,
                update_id=translation.update_id,
                origin_wall_clock_ms=translation.origin_wall_clock_ms,
                session_scope=translation.session_scope,
                source_text_hash=translation.source_text_hash,
                source_text_len=translation.source_text_len,
                logical_turn_key=translation.logical_turn_key,
                translation_len=len(translation.text),
                parent_utterance_id=parent_utterance_id,
                target_index=target_index,
                target_language=translation.target_language,
                turn_generation=turn_generation,
                turn_order=turn_order,
            )
        )

    async def project_translation_result(
        self,
        submission: TranslationOutputSubmission,
    ) -> TranslationResultProjectionReceipt:
        configuration = submission.config_snapshot.value
        utterance_id = submission.child_utterance_id
        channel = submission.channel
        source_language = self._source_language_for(channel, configuration)
        target_language = self._target_language_for(channel, configuration)
        publish_to_chatbox = self.chatbox_is_eligible(channel)
        deny_peer_chatbox_attempt = self.chatbox_is_denied(channel)
        dual_target_self = channel == "self" and len(configuration.self_target_languages) == 2
        self_update = _SelfProjectionUpdate(True)
        if dual_target_self:
            self_update = await self._record_self_submission(submission)
            if not self_update.accepted:
                return TranslationResultProjectionReceipt(
                    True,
                    record_runtime_translation=False,
                )

        if submission.outcome == "source_only":
            if channel == "peer":
                await self.project_peer_source_only(
                    transcript=Transcript(
                        utterance_id=utterance_id,
                        text=submission.source_text,
                        is_final=True,
                        created_at=self.clock.now(),
                        channel="peer",
                    ),
                    source_language=source_language,
                    target_language=target_language,
                    close_is_final=True,
                    finalize_latency=True,
                )
                await self.publish_peer_chatbox_denial(utterance_id)
            elif dual_target_self:
                self.diagnostics.clear_latency_timeline(channel, utterance_id)
            elif publish_to_chatbox:
                await self.publish_chatbox(
                    ChatboxProjection(
                        utterance_id=utterance_id,
                        channel=channel,
                        transcript_text=submission.source_text,
                        translation_text=None,
                        include_source=configuration.chatbox_include_source,
                        source=submission.source,
                    )
                )
            else:
                self.diagnostics.clear_latency_timeline(channel, utterance_id)
            return TranslationResultProjectionReceipt(True)

        if submission.outcome == "failed":
            if dual_target_self:
                await self.close_overlay_utterance(
                    utterance_id=utterance_id,
                    channel=channel,
                    is_final=False,
                    finalize_latency=True,
                )
                return TranslationResultProjectionReceipt(True)
            if submission.failure_code == "stale_provider_completion":
                if channel == "peer":
                    await self.publish_peer_chatbox_denial(utterance_id)
                    return TranslationResultProjectionReceipt(
                        True,
                        complete_peer_logical_turn=True,
                    )
                await self.close_overlay_utterance(
                    utterance_id=utterance_id,
                    channel=channel,
                    is_final=False,
                    finalize_latency=True,
                )
                return TranslationResultProjectionReceipt(True)

            fallback_to_chatbox = configuration.fallback_transcript_only and publish_to_chatbox
            denied_fallback_to_chatbox = (
                configuration.fallback_transcript_only and deny_peer_chatbox_attempt
            )
            if channel == "self":
                await self.close_overlay_utterance(
                    utterance_id=utterance_id,
                    channel=channel,
                    is_final=False,
                    finalize_latency=not fallback_to_chatbox,
                )
            else:
                await self.project_peer_source_only(
                    transcript=Transcript(
                        utterance_id=utterance_id,
                        text=submission.source_text,
                        is_final=True,
                        created_at=self.clock.now(),
                        channel="peer",
                    ),
                    source_language=source_language,
                    target_language=target_language,
                    close_is_final=False,
                    finalize_latency=not denied_fallback_to_chatbox,
                )
            if fallback_to_chatbox:
                await self.publish_chatbox(
                    ChatboxProjection(
                        utterance_id=utterance_id,
                        channel=channel,
                        transcript_text=submission.source_text,
                        translation_text=None,
                        include_source=configuration.chatbox_include_source,
                        source=submission.source,
                    )
                )
            elif deny_peer_chatbox_attempt:
                await self.publish_peer_chatbox_denial(utterance_id)
            elif channel != "peer":
                self.diagnostics.clear_latency_timeline(channel, utterance_id)
            return TranslationResultProjectionReceipt(True)

        translation = submission.translation
        if translation is None:
            raise ValueError("translated submission requires a translation")
        self.emit_translation_ready(
            translation,
            parent_utterance_id=submission.parent_utterance_id,
            target_index=submission.target_index,
            turn_generation=submission.turn_generation,
            turn_order=submission.turn_order,
        )
        if channel == "peer" and self.has_overlay_destination:
            await self.emit_translation(
                TranslationOverlayProjection(
                    translation=translation,
                    source_text=translation.source_text,
                    source_language=self._language_or_fallback(
                        translation.source_language,
                        source_language,
                    ),
                    target_language=self._language_or_fallback(
                        translation.target_language,
                        target_language,
                    ),
                    applied_context_mode=submission.applied_context_mode,
                    record_peer_first_emit=True,
                )
            )
            await self.close_overlay_utterance(
                utterance_id=utterance_id,
                channel=channel,
                is_final=True,
                finalize_latency=not (publish_to_chatbox or deny_peer_chatbox_attempt),
            )
        await self.publish_ui(
            TranslationUiMessage(
                event_type=UIEventType.TRANSLATION_DONE,
                utterance_id=utterance_id,
                payload=translation,
                source=submission.source,
            )
        )
        if channel == "self":
            await self.emit_translation(
                TranslationOverlayProjection(
                    translation=translation,
                    source_language=self._language_or_fallback(
                        translation.source_language,
                        source_language,
                    ),
                    target_language=self._language_or_fallback(
                        translation.target_language,
                        target_language,
                    ),
                    applied_context_mode=submission.applied_context_mode,
                )
            )
            await self.close_overlay_utterance(
                utterance_id=utterance_id,
                channel=channel,
                is_final=True,
                finalize_latency=not publish_to_chatbox,
            )
        if dual_target_self:
            if self_update.snapshot is None:
                self.diagnostics.clear_latency_timeline(channel, utterance_id)
        elif publish_to_chatbox:
            await self.publish_chatbox(
                ChatboxProjection(
                    utterance_id=utterance_id,
                    channel=channel,
                    transcript_text=submission.source_text,
                    translation_text=translation.text,
                    include_source=configuration.chatbox_include_source,
                    source=submission.source,
                    turn_generation=submission.turn_generation,
                    turn_order=submission.turn_order,
                )
            )
        elif deny_peer_chatbox_attempt:
            await self.publish_peer_chatbox_denial(utterance_id)
        else:
            self.diagnostics.clear_latency_timeline(channel, utterance_id)
        return TranslationResultProjectionReceipt(True)

    async def publish_chatbox(
        self,
        projection: ChatboxProjection,
    ) -> OutputPublicationResult:
        result = await self.output_runtime.publish_chatbox(
            publication_id=projection.utterance_id,
            channel=projection.channel,
            transcript_text=projection.transcript_text,
            translation_text=projection.translation_text,
            include_source=projection.include_source,
            presentation_revision=projection.presentation_revision,
            turn_generation=projection.turn_generation,
            turn_order=projection.turn_order,
        )
        if result.decision.decision != "published":
            self.diagnostics.emit(
                RuntimeDiagnostic(
                    message="[Translation] OSC enqueue skipped: channel=%s route=%s reason=%s",
                    args=(
                        projection.channel,
                        result.decision.route,
                        result.decision.reason,
                    ),
                    fallback_level=logging.INFO,
                    detailed=True,
                )
            )
            self.diagnostics.clear_latency_timeline(
                projection.channel,
                projection.utterance_id,
            )
            return result
        message = result.message
        assert message is not None
        self.diagnostics.emit(
            RuntimeDiagnostic(
                message=(
                    "[Translation] OSC enqueue preview: channel=%s text_len=%s "
                    "translation_text_present=%s include_source=%s"
                ),
                args=(
                    projection.channel,
                    len(message.text),
                    projection.translation_text is not None,
                    projection.include_source,
                ),
                fallback_level=logging.INFO,
                detailed=True,
            )
        )
        if projection.channel == "self":
            self.diagnostics.record_latency_stage(
                LatencyStageDiagnostic(
                    channel="self",
                    utterance_id=projection.utterance_id,
                    stage="self_chatbox_enqueue",
                )
            )
        await self.publish_ui(
            TranslationUiMessage(
                event_type=UIEventType.OSC_SENT,
                utterance_id=projection.utterance_id,
                payload=message,
                source=projection.source,
                channel=projection.channel,
            )
        )
        self.diagnostics.clear_latency_timeline(
            projection.channel,
            projection.utterance_id,
        )
        return result

    async def publish_peer_chatbox_denial(
        self,
        utterance_id: UUID,
    ) -> OutputPublicationResult:
        result = await self.output_runtime.publish_chatbox(
            publication_id=utterance_id,
            channel="peer",
            transcript_text="",
            translation_text=None,
            include_source=False,
        )
        self.diagnostics.emit(
            RuntimeDiagnostic(
                message="[Translation] OSC enqueue skipped: channel=%s route=%s reason=%s",
                args=("peer", result.decision.route, result.decision.reason),
                fallback_level=logging.INFO,
                detailed=True,
            )
        )
        self.diagnostics.clear_latency_timeline("peer", utterance_id)
        return result

    @staticmethod
    def _translation_metadata(translation: Translation) -> dict[str, object]:
        return {
            "update_id": translation.update_id,
            "origin_wall_clock_ms": translation.origin_wall_clock_ms,
            "session_scope": translation.session_scope,
            "source_text_hash": translation.source_text_hash,
            "source_text_len": translation.source_text_len,
            "logical_turn_key": translation.logical_turn_key,
        }

    @staticmethod
    def _active_translation_metadata(
        metadata: ActiveSelfOverlayMetadata | None,
    ) -> dict[str, object]:
        if metadata is None:
            return {
                "update_id": None,
                "origin_wall_clock_ms": None,
                "session_scope": None,
                "source_text_hash": None,
                "source_text_len": None,
                "logical_turn_key": None,
            }
        return {
            "update_id": metadata.update_id,
            "origin_wall_clock_ms": metadata.origin_wall_clock_ms,
            "session_scope": metadata.session_scope,
            "source_text_hash": metadata.source_text_hash,
            "source_text_len": metadata.source_text_len,
            "logical_turn_key": metadata.logical_turn_key,
        }

    def _active_secondary_decision(
        self,
        projection: ActiveSelfProjection,
        metadata: ActiveSelfOverlayMetadata | None,
        *,
        reuse_mode: str | None,
    ) -> tuple[str, str]:
        translation = projection.spec_translation
        if translation is not None and reuse_mode is not None:
            return translation.text.strip(), "spec"
        if metadata is not None and metadata.secondary_text.strip():
            return metadata.secondary_text.strip(), "sticky_cache"
        return "", "blank"

    def _secondary_translation_metadata(
        self,
        projection: ActiveSelfProjection,
        *,
        source: str,
        secondary_text: str,
        metadata: ActiveSelfOverlayMetadata | None,
    ) -> dict[str, object]:
        if not secondary_text:
            return self._active_translation_metadata(None)
        if source == "spec" and projection.spec_translation is not None:
            return self._translation_metadata(projection.spec_translation)
        if (
            source == "sticky_cache"
            and metadata is not None
            and metadata.utterance_id == projection.merge_id
        ):
            return self._active_translation_metadata(metadata)
        return self._active_translation_metadata(None)

    def _active_languages(
        self,
        projection: ActiveSelfProjection,
        *,
        source: str,
        secondary_text: str,
        metadata: ActiveSelfOverlayMetadata | None,
    ) -> tuple[str, str]:
        if source == "spec" and projection.spec_translation is not None:
            return (
                self._language_or_fallback(
                    projection.spec_translation.source_language,
                    projection.source_language,
                ),
                self._language_or_fallback(
                    projection.spec_translation.target_language,
                    projection.target_language,
                ),
            )
        metadata_matches = (
            metadata is not None
            and metadata.utterance_id == projection.merge_id
            and metadata.occupant_key == f"self:{projection.merge_id}"
        )
        if secondary_text and source == "sticky_cache" and metadata_matches:
            return (
                self._language_or_fallback(
                    metadata.primary_language,
                    projection.source_language,
                ),
                self._language_or_fallback(
                    metadata.secondary_language,
                    projection.target_language,
                ),
            )
        if not secondary_text and metadata_matches:
            return (
                self._language_or_fallback(
                    metadata.primary_language,
                    projection.source_language,
                ),
                projection.target_language,
            )
        return projection.source_language, projection.target_language

    @staticmethod
    def _language_or_fallback(language: str | None, fallback: str) -> str:
        if language is not None and language.strip():
            return language
        return fallback

    @staticmethod
    def _source_language_for(
        channel: ChannelId,
        configuration: TranslationRuntimeConfig,
    ) -> str:
        if channel == "peer" and configuration.peer_source_language:
            return configuration.peer_source_language
        return configuration.source_language

    @staticmethod
    def _target_language_for(
        channel: ChannelId,
        configuration: TranslationRuntimeConfig,
    ) -> str:
        if channel == "peer" and configuration.peer_target_language:
            return configuration.peer_target_language
        return configuration.target_language

    @staticmethod
    def normalize_soft_reuse_text(text: str) -> str:
        start = 0
        end = len(text)
        while start < end and TranslationOutputProjectionOwner._is_soft_reuse_boundary_char(
            text[start]
        ):
            start += 1
        while end > start and TranslationOutputProjectionOwner._is_soft_reuse_boundary_char(
            text[end - 1]
        ):
            end -= 1
        return text[start:end]

    @staticmethod
    def _is_soft_reuse_boundary_char(character: str) -> bool:
        return character.isspace() or character in _SOFT_REUSE_PUNCT

    def _record_overlay_emit(
        self,
        *,
        event_kind: str,
        utterance_id: UUID,
        channel: ChannelId,
        secondary_len: int,
    ) -> None:
        self.diagnostics.record_overlay_emit(
            OverlayEmitDiagnostic(
                event_kind=event_kind,
                utterance_id=utterance_id,
                channel=channel,
                secondary_len=secondary_len,
                sink_type=(
                    type(self.output_runtime.overlay_sink).__name__
                    if self.has_overlay_destination
                    else None
                ),
            )
        )


__all__ = [
    "ActiveSelfProjection",
    "ActiveSelfProjectionReceipt",
    "ChatboxProjection",
    "SelfTranslationTurnSnapshot",
    "TranscriptOverlayProjection",
    "TranslationOutputProjectionOwner",
    "TranslationOverlayProjection",
    "TranslationResultProjectionReceipt",
    "TranslationUiMessage",
    "TranslationUiMessagePort",
    "TranslationUiMessageQueue",
]
