from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol
from uuid import UUID

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.clock import Clock, SystemClock

from .presenter_acceptance import PresenterAcceptanceLedger
from .presenter_projection import NativeRetryIntentProjection
from .protocol import (
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
SELF_TRANSLATION_MIN_VISIBLE_SECONDS = 4.0
SleepFn = Callable[[float], Awaitable[None]]
PEER_REPLACEMENT_INTERVAL_SECONDS = 1.0
PeerPacingWaitObserver = Callable[[str], None]


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
    speaker_transition_mode: str = "A"
    task_factory: Any | None = None

    _acceptance: PresenterAcceptanceLedger = field(
        init=False,
        default_factory=PresenterAcceptanceLedger,
    )
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
    _last_new_occupant_at: float | None = field(init=False, default=None)
    _peer_admission_changed: asyncio.Event = field(init=False, default_factory=asyncio.Event)
    _closed: bool = field(init=False, default=False)
    _speaker_seen_readable: set[str] = field(init=False, default_factory=set)
    _speaker_colors: dict[str, str] = field(init=False, default_factory=dict)
    _speaker_emphasis_id: str | None = field(init=False, default=None)
    _peer_run_color: str = field(init=False, default="gold")

    def __post_init__(self) -> None:
        if self.speaker_transition_mode not in {"A", "C", "E"}:
            raise ValueError("speaker transition mode must be A, C, or E")
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

    def snapshot(self) -> OverlayPresentationSnapshot:
        return self._presentation_state.snapshot()

    def reset_scene(self) -> None:
        self._cancel_all_expiration_tasks()
        self._clear_entries_for_reason("scene_reset")
        self._acceptance.reset()
        self._retired_preview_self_seqs.clear()
        self._live_self_turn_key = None
        self._live_peer_turn_key = None
        self._revision = 0
        self._last_new_occupant_at = None
        self._signal_peer_admission_change()
        self._appearance_seq = 0
        self._speaker_seen_readable.clear()
        self._speaker_colors.clear()
        self._speaker_emphasis_id = None
        self._peer_run_color = "gold"
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
        self._last_new_occupant_at = None
        self._signal_peer_admission_change()
        self._retired_preview_self_seqs.clear()
        self._live_self_turn_key = None
        self._live_peer_turn_key = None
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
            event = self._acceptance.normalize_sequence(event, self._entries)
            rejection_reason = self._application_rejection_reason(event)
            if rejection_reason is not None:
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
        on_wait: PeerPacingWaitObserver | None = None,
    ) -> OverlayApplicationReceipt:
        if event.channel != "peer" or not isinstance(
            event, (PeerTranscriptFinal, TranslationFinal)
        ):
            return await self.emit(event)
        wait_reported = False
        while True:
            wake: asyncio.Event | None = None
            delay: float | None = None
            wait_to_report: str | None = None
            async with self._ownership_transition_lock:
                existing = self._acceptance.retained_receipt(event)
                if existing is not None:
                    return existing
                normalized = self._acceptance.normalize_sequence(event, self._entries)
                rejection_reason = self._application_rejection_reason(normalized)
                if rejection_reason is not None:
                    receipt = OverlayApplicationReceipt(
                        stage="application_accepted",
                        outcome="stale" if rejection_reason == "stale" else "not_applied",
                        publication_id=normalized.event_id,
                        scene_revision=self._revision,
                        cause=rejection_reason,
                    )
                    self._acceptance.remember_receipt(event, receipt)
                    return receipt
                delay = self._peer_replacement_delay(normalized)
                if delay is None:
                    wait_reason = "protected_rows"
                    wake = self._peer_admission_changed
                elif delay <= 0:
                    await self._emit_serialized(normalized)
                    receipt = OverlayApplicationReceipt(
                        stage="application_accepted",
                        outcome="applied",
                        publication_id=normalized.event_id,
                        scene_revision=self._revision,
                    )
                    self._acceptance.remember_receipt(event, receipt)
                    return receipt
                else:
                    wait_reason = "replacement_gate"
                    wake = self._peer_admission_changed
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
            if delay is None:
                await wake.wait()
            else:
                wake_task = asyncio.create_task(wake.wait())
                deadline_task = asyncio.create_task(self.sleep(delay))
                owned_tasks = (wake_task, deadline_task)
                try:
                    done, _ = await asyncio.wait(
                        owned_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        await task
                finally:
                    for task in owned_tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*owned_tasks, return_exceptions=True)

    def _peer_replacement_delay(self, event: OverlayEventUnion) -> float | None:
        key = self._entry_key(event.channel, event.utterance_id)
        selected_ids = {block.id for block in self.snapshot().blocks}
        if f"{key[0]}:{key[1]}" in selected_ids:
            return 0.0
        if len(selected_ids) < self.visible_window_target_blocks:
            return 0.0
        protected = {
            candidate
            for candidate in (self._live_self_turn_key, self._live_peer_turn_key)
            if candidate is not None and f"{candidate[0]}:{candidate[1]}" in selected_ids
        }
        if len(protected) >= self.visible_window_target_blocks:
            return None
        if self._last_new_occupant_at is None:
            return 0.0
        return max(
            0.0,
            self._last_new_occupant_at + PEER_REPLACEMENT_INTERVAL_SECONDS - self.clock.now(),
        )

    def _signal_peer_admission_change(self) -> None:
        previous = self._peer_admission_changed
        self._peer_admission_changed = asyncio.Event()
        previous.set()

    async def _emit_serialized(self, event: OverlayEventUnion) -> None:
        changed = self._apply_event(event)
        self._acceptance.record_entry_ordering(event, self._entries)
        if changed:
            await self._publish_if_changed(
                fresh_render_event=event,
                event_changed=True,
            )

    def _application_rejection_reason(self, event: OverlayEventUnion) -> str | None:
        self._expire_closed_entries(now=self.clock.now())
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

    def _apply_event(self, event: OverlayEventUnion) -> bool:
        now = self.clock.now()
        self._expire_closed_entries(now=now)

        if event.channel == "self":
            return self._apply_self_event(event, now=now)
        if event.channel == "peer":
            return self._apply_peer_event(event, now=now)

        return False

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
            show_translation=self.show_translation,
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
            show_translation=self.show_translation,
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
            show_translation=self.show_translation,
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
            show_translation=self.show_translation,
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

    async def _publish_if_changed(
        self,
        *,
        fresh_render_event: OverlayEventUnion | None = None,
        event_changed: bool = False,
        force_protocol_publish: bool = False,
    ) -> None:
        now = self.clock.now()
        self._expire_closed_entries(now=now)
        previous_snapshot = self.snapshot()
        previous_occupants = {block.id for block in previous_snapshot.blocks}
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
            and next_calibration == previous_snapshot.calibration
            and fresh_render_channel is None
            and not force_protocol_publish
        ):
            self._signal_peer_admission_change()
            return

        if fresh_render_channel is not None and self.native_retry_enabled:
            self._retry_projection.advance(fresh_render_channel, fresh_render_event)
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
        )
        next_occupants = {block.id for block in snapshot.blocks}
        if next_occupants - previous_occupants:
            self._last_new_occupant_at = now
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
            if previous_signatures.get(
                block.id
            ) == self._presentation_state.visible_block_content_signature(block):
                continue
            entry = self._entries.get(key)
            if entry is None:
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
        visible_ids = {block.id for _, block in rendered_entries}
        for key, block in rendered_entries:
            if block.id in self._speaker_seen_readable:
                continue
            self._speaker_seen_readable.add(block.id)
            self._speaker_emphasis_id = None
            if block.channel != "peer":
                continue
            entry = self._entries.get(key)
            comparison = None if entry is None else entry.speaker_transition
            if comparison == "transition":
                self._peer_run_color = "cyan" if self._peer_run_color == "gold" else "gold"
                if self.speaker_transition_mode == "E":
                    self._speaker_emphasis_id = block.id
            elif comparison in {"context_reset", "unavailable", None}:
                surviving_cyan = any(
                    self._speaker_colors.get(visible_id) == "cyan"
                    for visible_id in visible_ids
                    if visible_id != block.id
                )
                self._peer_run_color = "cyan" if surviving_cyan else "gold"
            self._speaker_colors[block.id] = self._peer_run_color

        styled: list[tuple[tuple[str, UUID], OverlayPresentationBlock]] = []
        for key, block in rendered_entries:
            if block.channel != "peer":
                styled.append((key, block))
                continue
            entry = self._entries.get(key)
            transition = entry is not None and entry.speaker_transition == "transition"
            if self.speaker_transition_mode == "C":
                speaker_style = self._speaker_colors.get(block.id, "gold")
            elif self.speaker_transition_mode == "E" and self._speaker_emphasis_id == block.id:
                speaker_style = "cyan"
            else:
                speaker_style = "gold"
            styled.append(
                (
                    key,
                    replace(
                        block,
                        speaker_style=speaker_style,
                        speaker_boundary=transition and self.speaker_transition_mode in {"A", "E"},
                    ),
                )
            )
        live_ids = {entry.block_id for entry in self._entries.values()}
        self._speaker_colors = {
            block_id: color
            for block_id, color in self._speaker_colors.items()
            if block_id in live_ids
        }
        self._speaker_seen_readable.intersection_update(live_ids)
        return styled

    async def update_speaker_transition_mode(self, mode: str) -> None:
        async with self._ownership_transition_lock:
            if mode not in {"A", "C", "E"}:
                raise ValueError("speaker transition mode must be A, C, or E")
            if mode == self.speaker_transition_mode:
                return
            self.speaker_transition_mode = mode
            self._speaker_emphasis_id = None
            await self._publish_if_changed(force_protocol_publish=True)

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
            show_translation=self.show_translation,
            late_arrival_window_seconds=LATE_ARRIVAL_WINDOW_SECONDS,
            visible_ttl_seconds=VISIBLE_TTL_SECONDS,
            self_translation_min_visible_seconds=SELF_TRANSLATION_MIN_VISIBLE_SECONDS,
        )
        self._drain_presentation_state_removals(current_task=current_task)

    def _entry_expiration_deadline(self, entry: _LogicalTurnEntry) -> float | None:
        return self._entry_expiration_components(entry)[0]

    def _entry_expiration_components(
        self,
        entry: _LogicalTurnEntry,
    ) -> tuple[float | None, float | None, float | None]:
        return self._presentation_state.entry_expiration_components(
            entry,
            show_translation=self.show_translation,
            late_arrival_window_seconds=LATE_ARRIVAL_WINDOW_SECONDS,
            visible_ttl_seconds=VISIBLE_TTL_SECONDS,
            self_translation_min_visible_seconds=SELF_TRANSLATION_MIN_VISIBLE_SECONDS,
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
        self._acceptance.retire_entry(key)
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
        if event.channel == "self":
            if not event_changed or not isinstance(
                event,
                (SelfTranscriptFinal, TranslationFinal),
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
