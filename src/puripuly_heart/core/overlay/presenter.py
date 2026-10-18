from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol
from uuid import UUID, uuid5

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.speaker_identity import SpeakerPresentationAssignments

from .presenter_acceptance import (
    PRESENTER_AGGREGATE_BYTE_LIMIT,
    PRESENTER_ENTRY_LIMIT,
    PRESENTER_EVENT_BYTE_LIMIT,
    PresenterAcceptanceLedger,
)
from .presenter_projection import NativeRetryIntentProjection
from .presenter_sources import PresenterSourceLedger, SourceCaptionUnit
from .protocol import (
    SPEAKER_IDENTITY_STYLES,
    OverlayPresentationBlock,
    OverlayPresentationCalibration,
    OverlayPresentationSnapshot,
)
from .sink import (
    OverlayApplicationReceipt,
    OverlayEventUnion,
    OverlaySink,
    PeerActiveUpdate,
    PeerTranscriptFinal,
    SelfActiveClear,
    SelfActiveUpdate,
    SelfTranscriptFinal,
    TranslationFinal,
    TranslationStreamUpdate,
    UtteranceClosed,
)
from .state import (
    ActiveSelfOverlayMetadata,
    OverlayEntryRemovalRecord,
    OverlayPresentationState,
    OverlayReductionResult,
)
from .state import (
    OverlayLogicalTurnEntry as _LogicalTurnEntry,
)

VISIBLE_WINDOW_TARGET_BLOCKS = 2

LATE_ARRIVAL_WINDOW_SECONDS = 5.0
VISIBLE_TTL_SECONDS = 8.0
SleepFn = Callable[[float], Awaitable[None]]


class OverlayPresentationTransport(Protocol):
    async def replace_snapshot(
        self,
        snapshot: OverlayPresentationSnapshot,
        *,
        block_expirations: Mapping[str, float | None] | None = None,
    ) -> object: ...

    async def broadcast_shutdown(self) -> None: ...


@dataclass(slots=True)
class OverlayPresenter(OverlaySink):
    calibration: OverlayCalibration
    bridge: OverlayPresentationTransport | None = None
    clock: Clock = field(default_factory=SystemClock)
    sleep: SleepFn = asyncio.sleep
    visible_window_target_blocks: int = VISIBLE_WINDOW_TARGET_BLOCKS
    show_translation: bool = True
    show_peer_original: bool = True
    translation_enabled: bool = True
    native_retry_enabled: bool = False
    task_factory: Any | None = None
    speaker_identities: SpeakerPresentationAssignments | None = None
    _speaker_task: asyncio.Task[None] | None = field(init=False, default=None)

    _acceptance: PresenterAcceptanceLedger = field(
        init=False,
        default_factory=PresenterAcceptanceLedger,
    )
    _sources: PresenterSourceLedger = field(init=False, default_factory=PresenterSourceLedger)
    _expiration_tasks: dict[tuple[str, UUID], asyncio.Task[None]] = field(
        init=False,
        default_factory=dict,
    )
    _revision: int = field(init=False, default=0)
    _appearance_seq: int = field(init=False, default=0)
    _retry_projection: NativeRetryIntentProjection = field(
        init=False,
        default_factory=NativeRetryIntentProjection,
    )
    _presentation_state: OverlayPresentationState = field(init=False)
    _ownership_transition_lock: asyncio.Lock = field(
        init=False,
        default_factory=asyncio.Lock,
    )
    _closing: bool = field(init=False, default=False)
    _peer_admission_changed: asyncio.Event = field(init=False, default_factory=asyncio.Event)
    _closed: bool = field(init=False, default=False)
    _active_speaker_scope: tuple[str, str] | None = field(init=False, default=None)
    _active_speaker_order: tuple[int, int, int] = field(init=False, default=(0, 0, 0))

    def __post_init__(self) -> None:
        self._presentation_state = OverlayPresentationState()
        self._retry_projection.reset(enabled=self.native_retry_enabled)
        self._presentation_state.generate_snapshot(
            revision=0,
            calibration=_calibration_from_overlay(self.calibration),
            rendered_entries=[],
        )

    @property
    def _entries(self) -> dict[tuple[str, UUID], _LogicalTurnEntry]:
        return self._presentation_state.entries

    @property
    def _retired_preview_self_seqs(self) -> OrderedDict[tuple[str, UUID], int]:
        return self._presentation_state.retired_preview_self_seqs

    @property
    def _live_self_turn_key(self) -> tuple[str, UUID] | None:
        return self._presentation_state.live_self_turn_key

    @_live_self_turn_key.setter
    def _live_self_turn_key(self, key: tuple[str, UUID] | None) -> None:
        self._presentation_state.live_self_turn_key = key

    @property
    def _live_peer_turn_key(self) -> tuple[str, UUID] | None:
        return self._presentation_state.live_peer_turn_key

    @_live_peer_turn_key.setter
    def _live_peer_turn_key(self, key: tuple[str, UUID] | None) -> None:
        self._presentation_state.live_peer_turn_key = key

    def active_self_overlay_metadata(self) -> ActiveSelfOverlayMetadata | None:
        return self._presentation_state.active_self_overlay_metadata()

    def _finish_reduction_result(self, result: OverlayReductionResult) -> bool:
        self._drain_presentation_state_removals()
        return result.changed

    def _terminal_update_reason(
        self,
        channel: str | None,
        utterance_id: UUID | None,
    ) -> str | None:
        return self._acceptance.terminal_reason(self._entry_key(channel, utterance_id))

    def attach_bridge(self, bridge: OverlayPresentationTransport) -> None:
        self.bridge = bridge

    def detach_bridge(self) -> None:
        self.bridge = None

    def bind_speaker_identities(self, identities: SpeakerPresentationAssignments) -> None:
        if self.speaker_identities is not None and self.speaker_identities is not identities:
            self.speaker_identities.changed.set()
        self.speaker_identities = identities
        if (self._speaker_task is None or self._speaker_task.done()) and not self._closed:
            self._speaker_task = self._create_task(
                self._watch_speaker_assignments(), task_name="overlay-speaker-assignments"
            )

    async def _watch_speaker_assignments(self) -> None:
        while not self._closed:
            identities = self.speaker_identities
            deadline = identities.next_deadline()
            timeout = None if deadline is None else max(0.0, deadline - self.clock.now())
            try:
                await asyncio.wait_for(identities.changed.wait(), timeout=timeout)
            except TimeoutError:
                pass
            identities.changed.clear()
            async with self._ownership_transition_lock:
                if self._closed:
                    return
                await self._publish_if_changed()

    def snapshot(self) -> OverlayPresentationSnapshot:
        return self._presentation_state.snapshot()

    def reset_scene(self) -> None:
        self._cancel_all_expiration_tasks()
        self._clear_entries_for_reason("scene_reset")
        self._acceptance.reset()
        self._sources.reset()
        self._retired_preview_self_seqs.clear()
        self._live_self_turn_key = None
        self._live_peer_turn_key = None
        self._revision = 0
        self._signal_peer_admission_change()
        self._appearance_seq = 0
        self._active_speaker_scope = None
        self._active_speaker_order = (0, 0, 0)
        self._retry_projection.clear_scene()
        self._presentation_state.generate_snapshot(
            revision=0,
            calibration=_calibration_from_overlay(self.calibration),
            rendered_entries=[],
        )

    async def clear_for_runtime_detach(self) -> None:
        await self._cancel_all_expiration_tasks_and_wait()
        self._clear_entries_for_reason("scene_reset")
        self._acceptance.reset()
        self._sources.reset()
        self._signal_peer_admission_change()
        self._retired_preview_self_seqs.clear()
        self._live_self_turn_key = None
        self._live_peer_turn_key = None
        self._active_speaker_scope = None
        self._active_speaker_order = (0, 0, 0)
        self._revision += 1
        self._retry_projection.clear_scene()
        snapshot = self._presentation_state.generate_snapshot(
            revision=self._revision,
            calibration=_calibration_from_overlay(self.calibration),
            rendered_entries=[],
        )
        if self.bridge is not None:
            await self.bridge.replace_snapshot(snapshot, block_expirations={})

    def application_receipt(self, publication_id: str) -> OverlayApplicationReceipt | None:
        return self._acceptance.receipt(publication_id)

    async def emit(self, event: OverlayEventUnion) -> OverlayApplicationReceipt:
        receipt_event = event
        existing = self._acceptance.retained_receipt(receipt_event)
        if existing is not None:
            return existing
        async with self._ownership_transition_lock:
            existing = self._acceptance.retained_receipt(receipt_event)
            if existing is not None:
                return existing
            if self._sources.key_for(event) is None:
                event = self._acceptance.normalize_sequence(event, self._entries)
            rejection_reason = self._application_rejection_reason(event)
            if rejection_reason is not None:
                await self._publish_if_changed()
                receipt = OverlayApplicationReceipt(
                    stage="application_accepted",
                    outcome="stale" if rejection_reason == "stale" else "not_applied",
                    publication_id=event.event_id,
                    scene_revision=self._revision,
                    cause=rejection_reason,
                )
                self._acceptance.remember_receipt(receipt_event, receipt)
                return receipt
            await self._emit_serialized(event)
            receipt = OverlayApplicationReceipt(
                stage="application_accepted",
                outcome="applied",
                publication_id=event.event_id,
                scene_revision=self._revision,
            )
            self._acceptance.remember_receipt(receipt_event, receipt)
            return receipt

    async def emit_peer_when_admissible(
        self,
        event: OverlayEventUnion,
        *,
        on_wait: Callable[[str], None] | None = None,
    ) -> OverlayApplicationReceipt:
        if event.channel != "peer" or not isinstance(
            event, (PeerTranscriptFinal, TranslationFinal)
        ):
            return await self.emit(event)
        wait_reported = False
        while True:
            wake: asyncio.Event | None = None
            wait_to_report: str | None = None
            async with self._ownership_transition_lock:
                existing = self._acceptance.retained_receipt(event)
                if existing is not None:
                    return existing
                normalized = (
                    event
                    if self._sources.key_for(event) is not None
                    else self._acceptance.normalize_sequence(event, self._entries)
                )
                rejection_reason = self._application_rejection_reason(normalized)
                if rejection_reason is not None:
                    await self._publish_if_changed()
                    receipt = OverlayApplicationReceipt(
                        stage="application_accepted",
                        outcome="stale" if rejection_reason == "stale" else "not_applied",
                        publication_id=normalized.event_id,
                        scene_revision=self._revision,
                        cause=rejection_reason,
                    )
                    self._acceptance.remember_receipt(event, receipt)
                    return receipt
                if self._peer_rows_protected(normalized):
                    wait_reason = "protected_rows"
                    wake = self._peer_admission_changed
                else:
                    await self._emit_serialized(normalized)
                    receipt = OverlayApplicationReceipt(
                        stage="application_accepted",
                        outcome="applied",
                        publication_id=normalized.event_id,
                        scene_revision=self._revision,
                    )
                    self._acceptance.remember_receipt(event, receipt)
                    return receipt
                if not wait_reported and on_wait is not None:
                    wait_to_report = wait_reason
                    wait_reported = True
            if wait_to_report is not None and on_wait is not None:
                try:
                    on_wait(wait_to_report)
                except Exception:
                    pass
            if wake is None:
                continue
            await wake.wait()

    def _peer_rows_protected(self, event: OverlayEventUnion) -> bool:
        source_key = self._sources.key_for(event)
        if source_key is not None:
            unit = self._sources.units.get(source_key)
            key = unit.card_key if unit is not None else None
        else:
            key = self._entry_key(event.channel, event.utterance_id)
        selected_ids = {block.id for block in self.snapshot().blocks}
        if key is not None and f"{key[0]}:{key[1]}" in selected_ids:
            return False
        if len(selected_ids) < self.visible_window_target_blocks:
            return False
        protected = {
            candidate
            for candidate in (self._live_self_turn_key, self._live_peer_turn_key)
            if candidate is not None and f"{candidate[0]}:{candidate[1]}" in selected_ids
        }
        return len(protected) >= self.visible_window_target_blocks

    def _signal_peer_admission_change(self) -> None:
        previous = self._peer_admission_changed
        self._peer_admission_changed = asyncio.Event()
        previous.set()

    async def _emit_serialized(self, event: OverlayEventUnion) -> None:
        changed = self._apply_event(event)
        source_key = self._sources.key_for(event)
        if source_key is None:
            self._acceptance.record_entry_ordering(event, self._entries)
        elif source_key is not None:
            unit = self._sources.units.get(source_key)
            if unit is not None:
                self._acceptance.record_entry_ordering(
                    replace(event, utterance_id=source_key[1]), {source_key: unit.content}
                )
        await self._publish_if_changed(
            fresh_render_event=event,
            event_changed=changed,
        )

    def _application_rejection_reason(self, event: OverlayEventUnion) -> str | None:
        self._expire_closed_entries(now=self.clock.now())
        if self._sources.key_for(event) is not None:
            if self._closing or self._closed:
                return "closed"
            if event.channel is not None and event.utterance_id is not None:
                attempt_key = (event.channel, event.utterance_id)
                if (
                    attempt_key not in self._entries
                    and attempt_key not in self._sources.attempts
                    and self._acceptance.terminal_reason(attempt_key) is not None
                ):
                    self._sources.retire_attempt(attempt_key)
            reason = self._sources.rejection_reason(event)
            if reason is not None:
                return reason
            if self._acceptance.event_payload_bytes(event) > PRESENTER_EVENT_BYTE_LIMIT:
                return "presenter_payload_exhausted"
            source_key = self._sources.key_for(event)
            if (
                len(self._entries) >= PRESENTER_ENTRY_LIMIT
                and source_key not in self._sources.units
            ):
                return "presenter_overload"
            retained = sum(
                self._acceptance.entry_payload_bytes(entry) for entry in self._entries.values()
            ) + sum(
                self._acceptance.entry_payload_bytes(unit.content)
                for unit in self._sources.units.values()
            )
            if (
                retained + 2 * self._acceptance.event_payload_bytes(event)
                > PRESENTER_AGGREGATE_BYTE_LIMIT
            ):
                return "presenter_payload_exhausted"
            return None
        return self._acceptance.rejection_reason(
            event,
            self._entries,
            closed=self._closing or self._closed,
        )

    async def update_calibration(self, calibration: OverlayCalibration) -> None:
        async with self._ownership_transition_lock:
            if calibration == self.calibration:
                return
            self.calibration = calibration.copy()
            await self._publish_if_changed()

    async def update_display_preferences(
        self,
        *,
        show_translation: bool,
        show_peer_original: bool,
    ) -> None:
        async with self._ownership_transition_lock:
            next_show_translation = bool(show_translation)
            next_show_peer_original = bool(show_peer_original)
            if (
                next_show_translation == self.show_translation
                and next_show_peer_original == self.show_peer_original
            ):
                return
            self.show_translation = next_show_translation
            self.show_peer_original = next_show_peer_original
            await self._publish_if_changed()

    async def update_translation_enabled(self, enabled: bool) -> None:
        async with self._ownership_transition_lock:
            next_enabled = bool(enabled)
            if next_enabled == self.translation_enabled:
                return
            self.translation_enabled = next_enabled
            await self._publish_if_changed()

    async def begin_native_retry_epoch(self, *, enabled: bool) -> None:
        async with self._ownership_transition_lock:
            self.native_retry_enabled = bool(enabled) and not self._closing and not self._closed
            self._retry_projection.begin(
                enabled=self.native_retry_enabled,
                snapshot=self.snapshot(),
            )
            await self._publish_if_changed(force_protocol_publish=True)
            self._rearm_expiration_tasks_from_original_deadlines()

    async def broadcast_shutdown(self) -> None:
        if self.bridge is None:
            return
        await self.bridge.broadcast_shutdown()

    async def close(self) -> None:
        async with self._ownership_transition_lock:
            if self._closed:
                return
            self._closing = True
            self._signal_peer_admission_change()
            await self.clear_for_runtime_detach()
            self.native_retry_enabled = False
            self._retry_projection.reset()
            self._closed = True
        if self._speaker_task is not None:
            self._speaker_task.cancel()
            await asyncio.gather(self._speaker_task, return_exceptions=True)
            self._speaker_task = None

    def _apply_event(self, event: OverlayEventUnion) -> bool:
        now = self.clock.now()
        self._expire_closed_entries(now=now)
        if self._sources.key_for(event) is not None:
            return self._apply_source_event(event, now=now)

        if event.channel == "self":
            return self._apply_self_event(event, now=now)
        if event.channel == "peer":
            return self._apply_peer_event(event, now=now)

        return False

    def _source_layout_key(self) -> tuple[object, ...]:
        return (
            self.translation_enabled,
            self.show_translation,
            self.show_peer_original,
            self.calibration.text_scale,
        )

    def _compatible_source_card(
        self,
        unit: SourceCaptionUnit,
    ) -> tuple[str, UUID] | None:
        source = unit.source
        content = unit.content
        assignment = content.speaker_assignment
        if source.channel != "peer" or source.display_group_id is None or assignment is None:
            return None
        attribution = assignment.attribution
        if attribution.state != "identified" or attribution.key is None:
            return None
        if assignment.scope_order > self._active_speaker_order:
            self._active_speaker_order = assignment.scope_order
            self._active_speaker_scope = (attribution.source, attribution.speaker_scope_id)
        colored_old_scope = any(
            block.channel == "peer"
            and (entry := self._entries.get(("peer", UUID(block.id.split(":", 1)[1])))) is not None
            and entry.speaker_style not in (None, "gray")
            and entry.speaker_assignment is not None
            and (
                entry.speaker_assignment.attribution.source,
                entry.speaker_assignment.attribution.speaker_scope_id,
            )
            != self._active_speaker_scope
            for block in self.snapshot().blocks
        )
        style, gray_reason = self._first_readable_speaker_style(
            content, colored_old_scope=colored_old_scope
        )
        for block in reversed(self.snapshot().blocks):
            if block.channel != "peer" or block.block_variant != "finalized":
                continue
            key = ("peer", UUID(block.id.split(":", 1)[1]))
            entry = self._entries.get(key)
            if (
                entry is not None
                and entry.source_managed
                and key != self._live_peer_turn_key
                and entry.display_group_id == source.display_group_id
                and entry.speaker_assignment is not None
                and entry.speaker_assignment.attribution.state == "identified"
                and entry.speaker_assignment.attribution.key == attribution.key
                and entry.speaker_style == style
                and entry.speaker_gray_reason == gray_reason
                and entry.layout_key == self._source_layout_key()
                and entry.original_language == content.original_language
                and entry.translation_language == content.translation_language
            ):
                return key
        return None

    def _apply_source_event(self, event: OverlayEventUnion, *, now: float) -> bool:
        source_key = self._sources.key_for(event)
        if source_key is None:
            return False
        unit = self._sources.units.get(source_key)
        if isinstance(event, UtteranceClosed):
            if unit is None:
                return False
            self._sources.remember_attempt(event, source_key)
            entry = self._entries.get(unit.card_key) if unit.card_key is not None else None
            if entry is None or entry.closed_at is not None:
                return False
            entry.closed_at = now
            entry.closed_seq = event.seq
            self._schedule_expiration(unit.card_key, entry)
            return False
        if not isinstance(
            event,
            (SelfTranscriptFinal, PeerTranscriptFinal, TranslationStreamUpdate, TranslationFinal),
        ):
            return False
        new_source = unit is None
        if unit is None:
            source = event.source_child
            if source is None:
                return False
            content = _LogicalTurnEntry(
                channel=source.channel,
                utterance_id=source.source_id,
                original_text=source.effective_text,
                original_language=source.language,
                translation_language=event.target_language.strip() or None,
                speaker_assignment=event.speaker_assignment,
            )
            unit = SourceCaptionUnit(source, content)
            self._sources.units[source_key] = unit
        content = unit.content
        if unit.source.language is None and event.source_language.strip():
            content.original_language = event.source_language.strip()
        self._sources.remember_attempt(event, source_key)
        if not unit.applied and event.speaker_assignment is not None:
            content.speaker_assignment = event.speaker_assignment
        if isinstance(event, (TranslationStreamUpdate, TranslationFinal)):
            content.translation_text = event.text
            content.translation_language = event.target_language.strip() or None
            content.translation_update_id = event.update_id
            content.translation_origin_wall_clock_ms = event.origin_wall_clock_ms
            content.translation_session_scope = event.session_scope
            content.translation_source_text_hash = event.source_text_hash
            content.translation_source_text_len = event.source_text_len
            content.translation_logical_turn_key = event.logical_turn_key
        if event.turn_generation is not None and event.turn_order is not None:
            unit.publication_ordering = (
                event.turn_kind or str(event.channel),
                event.turn_generation,
                event.turn_order,
                event.target_index,
            )
        new_application = not unit.applied and self._presentation_state.entry_is_publishable(
            content,
            show_peer_original=self.show_peer_original,
            translation_enabled=self.translation_enabled,
        )
        if not unit.readable and self.speaker_identities is not None:
            assignment = self.speaker_identities.resolve_source(unit.source, now=now)
            if assignment is not None:
                content.speaker_assignment = assignment
        if unit.card_key is not None and new_application:
            compatible_key = self._compatible_source_card(unit)
            if compatible_key is not None and compatible_key != unit.card_key:
                pending_key = unit.card_key
                self._sources.card_sources.pop(pending_key, None)
                self._remove_entry(pending_key, reason="source_applied_to_visible_card", now=now)
                unit.card_key = compatible_key
                self._sources.card_sources.setdefault(compatible_key, []).append(source_key)
        if unit.card_key is None:
            card_key = self._compatible_source_card(unit) if new_application else None
            if card_key is None:
                attempt_key = self._entry_key(event.channel, event.utterance_id)
                card_key = (
                    attempt_key
                    if event.channel == "self" and attempt_key in self._entries
                    else (unit.source.channel, uuid5(unit.source.source_id, "overlay-card"))
                )
                entry = self._presentation_state.entry_for(*card_key)
                if event.channel == "self" and entry.last_meaningful_visible_at is not None:
                    new_application = False
                    unit.applied = True
                entry.source_managed = True
                entry.display_group_id = unit.source.display_group_id
                entry.layout_key = self._source_layout_key()
                entry.speaker_assignment = content.speaker_assignment
                if entry.publication_ordering is None:
                    entry.publication_ordering = unit.publication_ordering
            unit.card_key = card_key
            self._sources.card_sources.setdefault(card_key, []).append(source_key)
        entry = self._entries[unit.card_key]
        if entry.speaker_style is None and content.speaker_assignment is not None:
            entry.speaker_assignment = content.speaker_assignment
        previous = (
            entry.original_text,
            entry.translation_text,
            entry.translation_language,
            entry.translation_update_id,
            entry.translation_origin_wall_clock_ms,
            entry.translation_session_scope,
            entry.translation_source_text_hash,
            entry.translation_source_text_len,
            entry.translation_logical_turn_key,
        )
        units = [self._sources.units[key] for key in self._sources.card_sources[unit.card_key]]
        units.sort(key=lambda candidate: candidate.source.order)
        entry.original_text = " ".join(
            candidate.source.effective_text.strip() for candidate in units
        )
        entry.translation_text = " ".join(
            candidate.content.translation_text.strip()
            for candidate in units
            if candidate.content.translation_text.strip()
        )
        entry.original_language = content.original_language
        entry.translation_language = content.translation_language
        entry.translation_update_id = content.translation_update_id
        entry.translation_origin_wall_clock_ms = content.translation_origin_wall_clock_ms
        entry.translation_session_scope = content.translation_session_scope
        entry.translation_source_text_hash = content.translation_source_text_hash
        entry.translation_source_text_len = content.translation_source_text_len
        entry.translation_logical_turn_key = content.translation_logical_turn_key
        entry.original_seq = event.seq
        entry.last_updated_seq = max(entry.last_updated_seq, event.seq)
        if event.channel == "self" and self._live_self_turn_key == unit.card_key:
            self._presentation_state._clear_self_live_payload(entry)
            self._live_self_turn_key = None
        current = (
            entry.original_text,
            entry.translation_text,
            entry.translation_language,
            entry.translation_update_id,
            entry.translation_origin_wall_clock_ms,
            entry.translation_session_scope,
            entry.translation_source_text_hash,
            entry.translation_source_text_len,
            entry.translation_logical_turn_key,
        )
        if new_application:
            self._appearance_seq = max(
                self._appearance_seq,
                event.seq,
                *(block.appearance_seq for block in self.snapshot().blocks),
            )
            entry.appearance_seq = self._next_appearance_seq()
            entry.publishable_seq = entry.appearance_seq
            entry.publication_ordering = unit.publication_ordering
            entry.source_application_pending = True
            unit.applied = True
        changed = new_application or new_source or current != previous
        if changed:
            entry.card_revision += 1
        return changed

    def _apply_self_event(self, event: OverlayEventUnion, *, now: float) -> bool:
        if isinstance(event, SelfActiveUpdate):
            return self._apply_self_active_update(event, now=now)
        if isinstance(event, SelfActiveClear):
            return self._apply_self_active_clear(event, now=now)
        if isinstance(event, SelfTranscriptFinal):
            return self._apply_self_finalized_update(event, now=now)
        if isinstance(event, (TranslationStreamUpdate, TranslationFinal)):
            return self._apply_self_translation_update(event, now=now)
        if isinstance(event, UtteranceClosed):
            return self._apply_self_utterance_closed(event, now=now)
        return False

    def _apply_peer_event(self, event: OverlayEventUnion, *, now: float) -> bool:
        if isinstance(event, PeerActiveUpdate):
            # Reserved compatibility/fallback path. Normal product peer overlay
            # rows become primary-visible when translation arrives, not from
            # source-only active speech.
            return self._apply_peer_active_update(event, now=now)
        if isinstance(event, PeerTranscriptFinal):
            return self._apply_peer_finalized_update(event, now=now)
        if isinstance(event, (TranslationStreamUpdate, TranslationFinal)):
            return self._apply_peer_translation_update(event, now=now)
        if isinstance(event, UtteranceClosed):
            return self._apply_peer_utterance_closed(event, now=now)
        return False

    def _apply_self_active_update(self, event: SelfActiveUpdate, *, now: float) -> bool:
        result = self._presentation_state.apply_self_active_update(
            event,
            now=now,
            terminal_update_reason=self._terminal_update_reason,
        )
        return self._finish_reduction_result(result)

    def _apply_peer_active_update(self, event: PeerActiveUpdate, *, now: float) -> bool:
        result = self._presentation_state.apply_peer_active_update(
            event,
            now=now,
            show_peer_original=self.show_peer_original,
            next_appearance_seq=self._next_appearance_seq,
            terminal_update_reason=self._terminal_update_reason,
            translation_enabled=self.translation_enabled,
        )
        return self._finish_peer_reduction_result(result, event)

    def _apply_peer_finalized_update(
        self,
        event: PeerTranscriptFinal,
        *,
        now: float,
    ) -> bool:
        result = self._presentation_state.apply_peer_finalized_update(
            event,
            now=now,
            show_peer_original=self.show_peer_original,
            next_appearance_seq=self._next_appearance_seq,
            terminal_update_reason=self._terminal_update_reason,
            translation_enabled=self.translation_enabled,
        )
        return self._finish_peer_reduction_result(result, event)

    def _apply_peer_translation_update(
        self,
        event: TranslationStreamUpdate | TranslationFinal,
        *,
        now: float,
    ) -> bool:
        result = self._presentation_state.apply_peer_translation_update(
            event,
            now=now,
            show_peer_original=self.show_peer_original,
            next_appearance_seq=self._next_appearance_seq,
            terminal_update_reason=self._terminal_update_reason,
            translation_enabled=self.translation_enabled,
        )
        return self._finish_peer_reduction_result(result, event)

    def _apply_peer_utterance_closed(self, event: UtteranceClosed, *, now: float) -> bool:
        result = self._presentation_state.apply_peer_utterance_closed(
            event,
            now=now,
            is_tombstoned=self._is_tombstoned,
        )
        return self._finish_peer_reduction_result(result, event)

    def _finish_peer_reduction_result(
        self,
        result: OverlayReductionResult,
        event: OverlayEventUnion,
    ) -> bool:
        changed = self._finish_reduction_result(result)
        if changed:
            key = self._entry_key(event.channel, event.utterance_id)
            entry = self._entries.get(key)
            if entry is not None and (entry.closed_seq is not None or entry.retained_hidden):
                self._schedule_expiration(key, entry)
        return changed

    def _apply_self_active_clear(self, event: SelfActiveClear, *, now: float) -> bool:
        result = self._presentation_state.apply_self_active_clear(
            event,
            now=now,
        )
        return self._finish_reduction_result(result)

    def _apply_self_finalized_update(
        self,
        event: SelfTranscriptFinal,
        *,
        now: float,
    ) -> bool:
        result = self._presentation_state.apply_self_finalized_update(
            event,
            now=now,
            next_appearance_seq=self._next_appearance_seq,
            terminal_update_reason=self._terminal_update_reason,
        )
        changed = self._finish_reduction_result(result)
        if changed:
            key = self._entry_key(event.channel, event.utterance_id)
            entry = self._entries.get(key)
            if entry is not None and (entry.closed_seq is not None or entry.retained_hidden):
                self._schedule_expiration(key, entry)
        return changed

    def _apply_self_translation_update(
        self,
        event: TranslationStreamUpdate | TranslationFinal,
        *,
        now: float,
    ) -> bool:
        result = self._presentation_state.apply_self_translation_update(
            event,
            now=now,
            next_appearance_seq=self._next_appearance_seq,
            terminal_update_reason=self._terminal_update_reason,
        )
        changed = self._finish_reduction_result(result)
        if changed:
            key = self._entry_key(event.channel, event.utterance_id)
            entry = self._entries.get(key)
            if entry is not None and (entry.closed_seq is not None or entry.retained_hidden):
                self._schedule_expiration(key, entry)
        return changed

    def _apply_self_utterance_closed(self, event: UtteranceClosed, *, now: float) -> bool:
        result = self._presentation_state.apply_self_utterance_closed(
            event,
            now=now,
            is_tombstoned=self._is_tombstoned,
        )
        changed = self._finish_reduction_result(result)
        if changed:
            key = self._entry_key(event.channel, event.utterance_id)
            entry = self._entries.get(key)
            if entry is not None:
                self._schedule_expiration(key, entry)
        return changed

    def _entry_key(self, channel: str | None, utterance_id: UUID | None) -> tuple[str, UUID]:
        if channel not in ("self", "peer"):
            raise ValueError(f"invalid overlay channel: {channel!r}")
        if utterance_id is None:
            raise ValueError("overlay presenter requires utterance_id for finalized entries")
        return (channel, utterance_id)

    def _is_tombstoned(self, channel: str | None, utterance_id: UUID | None) -> bool:
        return self._acceptance.is_tombstoned(self._entry_key(channel, utterance_id))

    def _live_turn_key_for_channel(self, channel: str) -> tuple[str, UUID] | None:
        if channel == "self":
            return self._live_self_turn_key
        if channel == "peer":
            return self._live_peer_turn_key
        raise ValueError(f"invalid overlay channel: {channel!r}")

    def _set_live_turn_key_for_channel(
        self,
        channel: str,
        key: tuple[str, UUID] | None,
    ) -> None:
        if channel == "self":
            self._live_self_turn_key = key
            return
        if channel == "peer":
            self._live_peer_turn_key = key
            return
        raise ValueError(f"invalid overlay channel: {channel!r}")

    def _live_entry_for_channel(
        self,
        channel: str,
    ) -> tuple[tuple[str, UUID], _LogicalTurnEntry] | None:
        live_key = self._live_turn_key_for_channel(channel)
        if live_key is None:
            return None
        entry = self._entries.get(live_key)
        if entry is None:
            self._set_live_turn_key_for_channel(channel, None)
            return None
        return live_key, entry

    def _live_self_entry(self) -> tuple[tuple[str, UUID], _LogicalTurnEntry] | None:
        return self._live_entry_for_channel("self")

    def _live_peer_entry(self) -> tuple[tuple[str, UUID], _LogicalTurnEntry] | None:
        return self._live_entry_for_channel("peer")

    def _prepare_unreadable_sources(self, *, now: float) -> None:
        self.speaker_identities.refresh(now=now)
        for unit in sorted(tuple(self._sources.units.values()), key=lambda item: item.source.order):
            if unit.readable:
                continue
            assignment = self.speaker_identities.resolve_source(unit.source, now=now)
            if assignment is not None:
                unit.content.speaker_assignment = assignment
            entry = self._entries.get(unit.card_key)
            if entry is None:
                continue
            if entry.speaker_style is None:
                entry.speaker_assignment = unit.content.speaker_assignment
            if not self._presentation_state.entry_is_publishable(
                unit.content,
                show_peer_original=self.show_peer_original,
                translation_enabled=self.translation_enabled,
            ):
                continue
            compatible = self._compatible_source_card(unit)
            if compatible is None or compatible == unit.card_key:
                continue
            old_key = unit.card_key
            source_key = (unit.source.channel, unit.source.source_id)
            self._sources.card_sources.pop(old_key, None)
            self._remove_entry(old_key, reason="source_applied_to_visible_card", now=now)
            unit.card_key = compatible
            self._sources.card_sources.setdefault(compatible, []).append(source_key)
            units = sorted(
                (self._sources.units[key] for key in self._sources.card_sources[compatible]),
                key=lambda item: item.source.order,
            )
            target = self._entries[compatible]
            target.original_text = " ".join(item.source.effective_text.strip() for item in units)
            target.translation_text = " ".join(
                item.content.translation_text.strip()
                for item in units
                if item.content.translation_text.strip()
            )
            target.original_language = unit.content.original_language
            target.translation_language = unit.content.translation_language
            target.translation_update_id = unit.content.translation_update_id
            target.translation_origin_wall_clock_ms = unit.content.translation_origin_wall_clock_ms
            target.translation_session_scope = unit.content.translation_session_scope
            target.translation_source_text_hash = unit.content.translation_source_text_hash
            target.translation_source_text_len = unit.content.translation_source_text_len
            target.translation_logical_turn_key = unit.content.translation_logical_turn_key
            target.original_seq = entry.original_seq
            target.last_updated_seq = max(target.last_updated_seq, entry.last_updated_seq)
            target.appearance_seq = self._next_appearance_seq()
            target.publishable_seq = target.appearance_seq
            target.publication_ordering = unit.publication_ordering
            target.source_application_pending = True
            target.card_revision += 1
            unit.applied = True

    async def _publish_if_changed(
        self,
        *,
        fresh_render_event: OverlayEventUnion | None = None,
        event_changed: bool = False,
        force_protocol_publish: bool = False,
    ) -> None:
        now = self.clock.now()
        self._expire_closed_entries(now=now)
        if self.speaker_identities is not None:
            self._prepare_unreadable_sources(now=now)
        previous_snapshot = self.snapshot()
        selection = self._presentation_state.visible_block_selection(
            entries=self._entries,
            live_self_entry=self._live_self_entry(),
            live_peer_entry=self._live_peer_entry(),
            visible_window_target_blocks=self.visible_window_target_blocks,
            show_translation=self.show_translation,
            show_peer_original=self.show_peer_original,
            next_appearance_seq=self._next_appearance_seq,
            translation_enabled=self.translation_enabled,
        )
        self._mark_entries_visible(selection.selected_keys)
        self._prune_displaced_finalized_entries(
            set(selection.selected_keys),
            candidate_keys=selection.candidate_keys,
        )
        for protected_key in selection.protected_keys:
            active_entry = self._entries.get(protected_key)
            if active_entry is not None:
                active_entry.ever_visible = True
        rendered_entries = selection.rendered_entries
        rendered_entries = self._apply_speaker_presentation(rendered_entries)
        speaker_divider = self._speaker_divider(rendered_entries)
        previous_blocks = {block.id: block for block in previous_snapshot.blocks}
        for index, (key, block) in enumerate(rendered_entries):
            entry = self._entries.get(key)
            previous = previous_blocks.get(block.id)
            previous_revision = previous.card_revision if previous is not None else 0
            if (
                entry is not None
                and block.card_revision == previous_revision
                and (
                    previous is None
                    or self._presentation_state.rendered_block_signature(block)
                    != self._presentation_state.rendered_block_signature(previous)
                )
            ):
                entry.card_revision += 1
                rendered_entries[index] = (key, replace(block, card_revision=entry.card_revision))
        next_blocks = [block for _, block in rendered_entries]
        next_calibration = _calibration_from_overlay(self.calibration)
        fresh_render_channel = self._eligible_fresh_render_channel(
            fresh_render_event,
            event_changed=event_changed,
            rendered_entries=rendered_entries,
            previous_snapshot=previous_snapshot,
        )
        previous_rendered_signature = self._presentation_state.rendered_blocks_signature(
            previous_snapshot.blocks
        )
        next_rendered_signature = self._presentation_state.rendered_blocks_signature(next_blocks)
        self._refresh_visible_expiration_deadlines(
            rendered_entries,
            previous_blocks=previous_snapshot.blocks,
            now=now,
        )
        if (
            next_rendered_signature == previous_rendered_signature
            and speaker_divider == previous_snapshot.speaker_divider
            and next_calibration == previous_snapshot.calibration
            and fresh_render_channel is None
            and not force_protocol_publish
        ):
            self._signal_peer_admission_change()
            return

        if fresh_render_channel is not None and self.native_retry_enabled:
            source_key = (
                self._sources.key_for(fresh_render_event)
                if fresh_render_event is not None
                else None
            )
            unit = self._sources.units.get(source_key) if source_key is not None else None
            card = self._entries.get(unit.card_key) if unit is not None else None
            self._retry_projection.advance(
                fresh_render_channel,
                fresh_render_event,
                target=card.block_id if card is not None else None,
            )
        self._retry_projection.prune(next_blocks)
        self._revision += 1
        snapshot = self._presentation_state.generate_snapshot(
            revision=self._revision,
            calibration=next_calibration,
            rendered_entries=rendered_entries,
            native_fresh_render_generations=(
                self._retry_projection.generations if self.native_retry_enabled else None
            ),
            native_fresh_render_targets=(
                self._retry_projection.targets if self.native_retry_enabled else None
            ),
            native_quiet_tail_episodes=(
                self._retry_projection.episodes if self.native_retry_enabled else None
            ),
            entry_ordering=self._acceptance.entry_ordering,
            semantic_retirement_frontiers=self._acceptance.retired_turn_frontiers,
            speaker_divider=speaker_divider,
        )
        self._signal_peer_admission_change()
        if self.bridge is not None:
            block_expirations = {
                block.id: self._entry_expiration_deadline(entry)
                for (key, block) in rendered_entries
                if (entry := self._entries.get(key)) is not None
            }
            await self.bridge.replace_snapshot(
                snapshot,
                block_expirations=block_expirations,
            )

    def _refresh_visible_expiration_deadlines(
        self,
        rendered_entries: list[tuple[tuple[str, UUID], OverlayPresentationBlock]],
        *,
        previous_blocks: list[OverlayPresentationBlock],
        now: float,
    ) -> None:
        previous_signatures = {
            block.id: self._presentation_state.visible_block_content_signature(block)
            for block in previous_blocks
        }
        for key, block in rendered_entries:
            entry = self._entries.get(key)
            if entry is None:
                continue
            if entry.source_managed:
                if (
                    not entry.source_application_pending
                    and entry.last_meaningful_visible_at is not None
                ):
                    continue
                entry.source_application_pending = False
                for source_key in self._sources.card_sources.get(key, []):
                    self._sources.units[source_key].applied = True
            elif (
                entry.last_meaningful_visible_at is not None and block.block_variant == "finalized"
            ):
                continue
            elif previous_signatures.get(
                block.id
            ) == self._presentation_state.visible_block_content_signature(block):
                continue
            entry.ever_visible = True
            if entry.visible_since is None:
                entry.visible_since = now
            entry.last_meaningful_visible_at = now
            self._schedule_expiration(key, entry)

    def _prune_displaced_finalized_entries(
        self,
        visible_entry_keys: set[tuple[str, UUID]],
        *,
        candidate_keys: list[tuple[str, UUID]],
    ) -> None:
        displaced_keys = [
            key
            for key in candidate_keys
            if (entry := self._entries.get(key)) is not None
            and self._presentation_state.entry_is_selectable(
                entry,
                show_peer_original=self.show_peer_original,
                translation_enabled=self.translation_enabled,
            )
            and key not in visible_entry_keys
        ]
        for key in displaced_keys:
            entry = self._entries.get(key)
            if entry is None:
                continue
            self._remove_entry(
                key,
                reason="evicted_by_newer_turn",
                now=self.clock.now(),
                tombstone_seq=entry.last_updated_seq,
            )

    def _mark_entries_visible(self, visible_entry_keys: list[tuple[str, UUID]]) -> None:
        for key in visible_entry_keys:
            entry = self._entries.get(key)
            if entry is not None:
                if entry.retained_hidden:
                    entry.retained_hidden = False
                    entry.window_evicted_at = None
                    self._schedule_expiration(key, entry)
                entry.ever_visible = True

    def _apply_speaker_presentation(
        self,
        rendered_entries: list[tuple[tuple[str, UUID], OverlayPresentationBlock]],
    ) -> list[tuple[tuple[str, UUID], OverlayPresentationBlock]]:
        if self.speaker_identities is not None:
            now = self.clock.now()
            for key, block in rendered_entries:
                if block.channel != "peer":
                    continue
                entry = self._entries.get(key)
                if entry is None:
                    continue
                for source_key in self._sources.card_sources.get(key, []):
                    unit = self._sources.units[source_key]
                    assignment = self.speaker_identities.mark_readable(unit.source, now=now)
                    unit.readable = True
                    if assignment is None:
                        continue
                    previous = entry.speaker_assignment
                    if entry.speaker_style is None or (
                        entry.speaker_style == "gray"
                        and entry.speaker_gray_reason
                        in ("not_ready", "unavailable", "uncertain", "unallocated", "missing")
                        and assignment.palette_index is not None
                        and (previous is None or assignment.revision > previous.revision)
                    ):
                        entry.speaker_assignment = assignment
                        unit.content.speaker_assignment = assignment
                        entry.speaker_style = None
                        if previous != assignment:
                            entry.card_revision += 1
        for key, block in rendered_entries:
            if block.channel != "peer":
                continue
            entry = self._entries.get(key)
            assignment = None if entry is None else entry.speaker_assignment
            if assignment is None:
                continue
            attribution = assignment.attribution
            if (
                attribution.state == "identified"
                and attribution.source
                and attribution.speaker_scope_id
                and assignment.scope_order > self._active_speaker_order
            ):
                self._active_speaker_order = assignment.scope_order
                self._active_speaker_scope = (
                    attribution.source,
                    attribution.speaker_scope_id,
                )

        colored_old_scope = any(
            block.channel == "peer"
            and (entry := self._entries.get(key)) is not None
            and entry.speaker_style not in (None, "gray")
            and entry.speaker_assignment is not None
            and entry.speaker_assignment.attribution.state == "identified"
            and (
                entry.speaker_assignment.attribution.source,
                entry.speaker_assignment.attribution.speaker_scope_id,
            )
            != self._active_speaker_scope
            for key, block in rendered_entries
        )
        styled: list[tuple[tuple[str, UUID], OverlayPresentationBlock]] = []
        for key, block in rendered_entries:
            entry = self._entries.get(key)
            if entry is not None:
                block = replace(block, card_revision=entry.card_revision)
            if block.channel != "peer":
                styled.append((key, block))
                continue
            entry = self._entries.get(key)
            if entry is None:
                continue
            if entry.speaker_style is None:
                entry.speaker_style, entry.speaker_gray_reason = self._first_readable_speaker_style(
                    entry,
                    colored_old_scope=colored_old_scope,
                )
            styled.append((key, replace(block, speaker_style=entry.speaker_style)))
        return styled

    def _first_readable_speaker_style(
        self,
        entry: _LogicalTurnEntry,
        *,
        colored_old_scope: bool,
    ) -> tuple[str, str | None]:
        assignment = entry.speaker_assignment
        if assignment is None:
            return "gray", "not_ready"
        attribution = assignment.attribution
        if attribution.state == "non_diarized":
            return "gold", None
        if attribution.state != "identified":
            return "gray", attribution.state
        if attribution.key is None:
            return "gray", "malformed"
        scope = (attribution.source, attribution.speaker_scope_id)
        if scope != self._active_speaker_scope or colored_old_scope:
            return "gray", "scope_handoff"
        index = assignment.palette_index
        if index is not None and 0 <= index < len(SPEAKER_IDENTITY_STYLES):
            return SPEAKER_IDENTITY_STYLES[index], None
        if assignment.palette_overflow:
            return "gray", "palette_overflow"
        return "gray", "unallocated"

    def _speaker_divider(
        self,
        rendered_entries: list[tuple[tuple[str, UUID], OverlayPresentationBlock]],
    ) -> bool:
        if len(rendered_entries) != 2:
            return False
        speaker_keys = []
        for key, block in rendered_entries:
            entry = self._entries.get(key)
            if (
                block.channel != "peer"
                or entry is None
                or entry.speaker_gray_reason != "palette_overflow"
                or entry.speaker_assignment is None
                or entry.speaker_assignment.attribution.key is None
            ):
                return False
            speaker_keys.append(entry.speaker_assignment.attribution.key)
        first, second = speaker_keys
        return (
            first.source == second.source
            and first.speaker_scope_id == second.speaker_scope_id
            and first != second
        )

    def _next_appearance_seq(self) -> int:
        self._appearance_seq += 1
        return self._appearance_seq

    def _rearm_expiration_tasks_from_original_deadlines(self) -> None:
        for key, entry in tuple(self._entries.items()):
            self._schedule_expiration(key, entry)

    def _schedule_expiration(
        self,
        key: tuple[str, UUID],
        entry: _LogicalTurnEntry,
    ) -> None:
        self._cancel_expiration_task(key)
        if self._entry_expiration_deadline(entry) is None:
            return
        entry.expiration_revision += 1
        self._expiration_tasks[key] = self._create_task(
            self._expire_entry_after_ttl(key, entry.expiration_revision),
            task_name=f"presenter-expiration:{key[0]}:{key[1]}",
        )

    async def _expire_entry_after_ttl(
        self, key: tuple[str, UUID], expiration_revision: int
    ) -> None:
        try:
            while True:
                entry = self._entries.get(key)
                if entry is None or entry.expiration_revision != expiration_revision:
                    return

                deadline = self._entry_expiration_deadline(entry)
                if deadline is None:
                    return
                remaining = deadline - self.clock.now()
                if remaining > 0:
                    await self.sleep(remaining)
                    continue

                self._remove_entry(
                    key,
                    reason="expired",
                    now=self.clock.now(),
                    current_task=self._current_task(),
                    tombstone_seq=entry.last_updated_seq if entry.closed_seq is None else None,
                )
                await self._publish_if_changed()
                return
        except asyncio.CancelledError:
            raise
        finally:
            current_task = self._current_task()
            if current_task is not None and self._expiration_tasks.get(key) is current_task:
                self._expiration_tasks.pop(key, None)

    def _expire_closed_entries(self, *, now: float) -> None:
        current_task = self._current_task()
        self._presentation_state.expire_entries(
            now=now,
            late_arrival_window_seconds=LATE_ARRIVAL_WINDOW_SECONDS,
            visible_ttl_seconds=VISIBLE_TTL_SECONDS,
        )
        self._drain_presentation_state_removals(current_task=current_task)

    def _entry_expiration_deadline(self, entry: _LogicalTurnEntry) -> float | None:
        return self._presentation_state.entry_expiration_deadline(
            entry,
            late_arrival_window_seconds=LATE_ARRIVAL_WINDOW_SECONDS,
            visible_ttl_seconds=VISIBLE_TTL_SECONDS,
        )

    def _remove_entry(
        self,
        key: tuple[str, UUID],
        *,
        reason: str,
        now: float | None = None,
        current_task: asyncio.Task[None] | None = None,
        tombstone_seq: int | None = None,
    ) -> None:
        if self._expiration_tasks.get(key) is not current_task:
            self._cancel_expiration_task(key)
        self._presentation_state.remove_entry(
            key,
            reason=reason,
            now=now,
            tombstone_seq=tombstone_seq,
        )
        self._drain_presentation_state_removals(current_task=current_task)

    def _drain_presentation_state_removals(
        self,
        *,
        current_task: asyncio.Task[None] | None = None,
    ) -> None:
        for record in self._presentation_state.drain_pending_removals():
            if self._expiration_tasks.get(record.key) is not current_task:
                self._cancel_expiration_task(record.key)
            self._record_removed_entry(record)

    def _record_removed_entry(self, record: OverlayEntryRemovalRecord) -> None:
        key = record.key
        entry = record.entry
        if not entry.source_managed and record.reason in {"expired", "evicted_by_newer_turn"}:
            self._sources.retire_attempt(key)
        self._acceptance.retire_entry(key)
        for source_key in self._sources.card_sources.get(key, []):
            self._acceptance.retire_entry(source_key)
            if (
                self.speaker_identities is not None
                and record.reason != "source_applied_to_visible_card"
            ):
                self.speaker_identities.retire_source(source_key[1])
        self._sources.retire_card(key)
        seq = record.tombstone_seq if record.tombstone_seq is not None else entry.closed_seq
        if record.reason == "expired" and entry.ever_visible:
            self._acceptance.remember_scene_terminal(key, record.reason)
        if record.reason == "evicted_by_newer_turn":
            self._acceptance.remember_scene_terminal(key, record.reason)
        if seq is not None:
            self._acceptance.remember_tombstone(key, seq)

    def _cancel_expiration_task(self, key: tuple[str, UUID]) -> None:
        task = self._expiration_tasks.pop(key, None)
        if task is not None and not task.done():
            task.cancel()

    def _cancel_all_expiration_tasks(self) -> None:
        for task in self._expiration_tasks.values():
            if not task.done():
                task.cancel()
        self._expiration_tasks.clear()

    async def _cancel_all_expiration_tasks_and_wait(self) -> None:
        tasks = tuple(self._expiration_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        self._expiration_tasks.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        task_name: str,
    ) -> asyncio.Task[Any]:
        if self.task_factory is not None:
            return self.task_factory(coroutine, task_name=task_name)
        return asyncio.create_task(coroutine, name=f"OverlayPresenter:{task_name}")

    def _eligible_fresh_render_channel(
        self,
        event: OverlayEventUnion | None,
        *,
        event_changed: bool,
        rendered_entries: list[tuple[tuple[str, UUID], OverlayPresentationBlock]],
        previous_snapshot: OverlayPresentationSnapshot,
    ) -> str | None:
        if event is None or event.utterance_id is None:
            return None
        key = (event.channel, event.utterance_id)
        source_key = self._sources.key_for(event)
        if source_key is not None:
            unit = self._sources.units.get(source_key)
            if unit is None or unit.card_key is None:
                return None
            key = unit.card_key
        if event.channel == "self":
            if not event_changed or not isinstance(
                event,
                (SelfActiveUpdate, SelfTranscriptFinal, TranslationFinal),
            ):
                return None
        elif event.channel == "peer":
            if not event_changed or not isinstance(
                event,
                (
                    PeerActiveUpdate,
                    PeerTranscriptFinal,
                    TranslationStreamUpdate,
                    TranslationFinal,
                ),
            ):
                return None
        else:
            return None
        for rendered_key, block in rendered_entries:
            if rendered_key != key:
                continue
            if block.primary_text.strip():
                if isinstance(event, SelfActiveUpdate):
                    previous_block = next(
                        (
                            previous
                            for previous in previous_snapshot.blocks
                            if previous.id == block.id
                        ),
                        None,
                    )
                    if previous_block is not None and (
                        self._presentation_state.visible_block_content_signature(previous_block)
                        == self._presentation_state.visible_block_content_signature(block)
                    ):
                        return None
                if event.channel == "self" and block.block_variant == "finalized":
                    previous_block = self._visible_finalized_self_block_in_snapshot(
                        previous_snapshot,
                        key,
                    )
                    previous_signature = (
                        self._presentation_state.visible_block_content_signature(previous_block)
                        if previous_block is not None
                        else None
                    )
                    current_signature = self._presentation_state.visible_block_content_signature(
                        block
                    )
                    if previous_signature == current_signature:
                        return None
                return event.channel
        return None

    def _visible_finalized_self_block_in_snapshot(
        self,
        snapshot: OverlayPresentationSnapshot,
        key: tuple[str, UUID],
    ) -> OverlayPresentationBlock | None:
        if key[0] != "self":
            return None
        block_id = f"self:{key[1]}"
        for block in snapshot.blocks:
            if block.channel != "self" or block.id != block_id:
                continue
            if block.block_variant == "finalized" and block.primary_text.strip():
                return block
        return None

    def _current_task(self) -> asyncio.Task[None] | None:
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None

    def _clear_entries_for_reason(self, reason: str) -> None:
        for key in list(self._entries):
            self._remove_entry(key, reason=reason, now=self.clock.now())


def _calibration_from_overlay(
    calibration: OverlayCalibration,
) -> OverlayPresentationCalibration:
    return OverlayPresentationCalibration(
        anchor=calibration.anchor,
        offset_x=calibration.offset_x,
        offset_y=calibration.offset_y,
        distance=calibration.distance,
        text_scale=calibration.text_scale,
        background_alpha=calibration.background_alpha,
    )
