from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol
from uuid import UUID

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.clock import Clock, SystemClock

from .diagnostics import OverlayDiagnosticsRecorder
from .protocol import (
    U64_MAX,
    NativeFreshRenderGenerations,
    NativeFreshRenderTargets,
    NativeQuietTailEpisode,
    NativeQuietTailEpisodes,
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
    OverlayTurnDecisionRecord,
)
from .state import (
    OverlayLogicalTurnEntry as _LogicalTurnEntry,
)

VISIBLE_WINDOW_TARGET_BLOCKS = 2
_CLOSED_TOMBSTONE_LIMIT = 64
_PRESENTER_ENTRY_LIMIT = 64
_PRESENTER_EVENT_BYTE_LIMIT = 1024 * 1024
_PRESENTER_AGGREGATE_BYTE_LIMIT = 16 * 1024 * 1024
_PRESENTER_RECEIPT_LIMIT = 4096
LATE_ARRIVAL_WINDOW_SECONDS = 5.0
VISIBLE_TTL_SECONDS = 8.0
SELF_TRANSLATION_MIN_VISIBLE_SECONDS = 4.0
# LOAD-BEARING: The peer presentation refresh burst is product-permanent unless
# Stage 2 HMD QA proves an alternative. The 2026-04-28 submit-only resubmit
# regression showed repeated stored-frame SetOverlayTexture calls are not
# equivalent; each cadence tick must drive fresh snapshot/render/GPU work.
PEER_PRESENTATION_REFRESH_BURST_SECONDS = 2.0
PEER_PRESENTATION_REFRESH_BURST_INTERVAL_SECONDS = 0.1
SleepFn = Callable[[float], Awaitable[None]]


class OverlayPresentationTransport(Protocol):
    async def replace_snapshot(
        self,
        snapshot: OverlayPresentationSnapshot,
        *,
        block_expirations: Mapping[str, float | None] | None = None,
    ) -> object: ...

    async def broadcast_shutdown(self) -> None: ...


class RuntimeDetailedLogger(Protocol):
    def __call__(self, message: str, *, level: int = logging.INFO) -> bool: ...


@dataclass(slots=True)
class OverlayPresenter(OverlaySink):
    calibration: OverlayCalibration
    bridge: OverlayPresentationTransport | None = None
    diagnostics: OverlayDiagnosticsRecorder | None = None
    runtime_log_detailed: RuntimeDetailedLogger | None = None
    clock: Clock = field(default_factory=SystemClock)
    sleep: SleepFn = asyncio.sleep
    visible_window_target_blocks: int = VISIBLE_WINDOW_TARGET_BLOCKS
    show_translation: bool = True
    show_peer_original: bool = True
    translation_enabled: bool = True
    peer_presentation_refresh_burst: bool = True
    self_presentation_refresh_burst: bool = True
    native_retry_trigger_emission: bool = False
    task_factory: Any | None = None

    _terminal_registry: OrderedDict[tuple[str, UUID], int] = field(
        init=False,
        default_factory=OrderedDict,
    )
    _scene_terminal_keys: OrderedDict[tuple[str, UUID], str] = field(
        init=False,
        default_factory=OrderedDict,
    )
    _application_receipts: OrderedDict[tuple[object, ...], OverlayApplicationReceipt] = field(
        init=False,
        default_factory=OrderedDict,
    )
    _application_receipts_by_id: dict[str, OverlayApplicationReceipt] = field(
        init=False,
        default_factory=dict,
    )
    _entry_ordering: dict[tuple[str, UUID], tuple[str, int, int]] = field(
        init=False,
        default_factory=dict,
    )
    _entry_sequence_namespaces: dict[tuple[str, UUID], tuple[int, int]] = field(
        init=False,
        default_factory=dict,
    )
    _retired_turn_frontiers: dict[tuple[str, int], int] = field(
        init=False,
        default_factory=dict,
    )
    _expiration_tasks: dict[tuple[str, UUID], asyncio.Task[None]] = field(
        init=False,
        default_factory=dict,
    )
    _revision: int = field(init=False, default=0)
    _appearance_seq: int = field(init=False, default=0)
    _native_fresh_render_generations: NativeFreshRenderGenerations = field(
        init=False, default_factory=NativeFreshRenderGenerations
    )
    _native_fresh_render_targets: NativeFreshRenderTargets = field(
        init=False, default_factory=NativeFreshRenderTargets
    )
    _native_quiet_tail_episodes: NativeQuietTailEpisodes = field(
        init=False, default_factory=NativeQuietTailEpisodes
    )
    _native_quiet_tail_self_target: str | None = field(init=False, default=None)
    _native_quiet_tail_peer_target: str | None = field(init=False, default=None)
    _native_quiet_tail_self_generation: int | None = field(init=False, default=None)
    _native_quiet_tail_peer_generation: int | None = field(init=False, default=None)
    _presentation_state: OverlayPresentationState = field(init=False)
    _last_visible_window_signature: tuple[object, ...] | None = field(init=False, default=None)
    _peer_presentation_refresh_burst_task: asyncio.Task[None] | None = field(
        init=False,
        default=None,
    )
    _self_presentation_refresh_burst_task: asyncio.Task[None] | None = field(
        init=False,
        default=None,
    )
    _self_presentation_refresh_burst_cancel_reasons: dict[asyncio.Task[None], str] = field(
        init=False, default_factory=dict
    )
    _self_presentation_refresh_burst_cancel_cleanup_counts: dict[asyncio.Task[None], int] = field(
        init=False, default_factory=dict
    )
    _ownership_transition_lock: asyncio.Lock = field(
        init=False,
        default_factory=asyncio.Lock,
    )
    _closing: bool = field(init=False, default=False)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._presentation_state = OverlayPresentationState()
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

    def _emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool:
        if self.runtime_log_detailed is None:
            return False
        try:
            return self.runtime_log_detailed(message, level=level)
        except Exception:
            return False

    def _emit_detailed_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
    ) -> bool:
        runtime_log_detailed = self.runtime_log_detailed
        if runtime_log_detailed is None:
            return False

        owner = getattr(runtime_log_detailed, "__self__", None)
        try:
            if owner is not None:
                emit_detailed_lazy = getattr(owner, "emit_detailed_lazy", None)
                if callable(emit_detailed_lazy):
                    return emit_detailed_lazy(build_message, level=level)
                log_detailed_lazy = getattr(owner, "log_detailed_lazy", None)
                if callable(log_detailed_lazy):
                    return log_detailed_lazy(build_message, level=level)
            return runtime_log_detailed(build_message(), level=level)
        except Exception:
            return False

    def _emit_turn_decision(
        self,
        decision: str,
        *,
        disposition: str | None = None,
        key: tuple[str, UUID] | None = None,
        entry: _LogicalTurnEntry | None = None,
        block: OverlayPresentationBlock | None = None,
        extras: dict[str, object] | None = None,
    ) -> bool:
        def build_message() -> str:
            resolved_key = key
            if resolved_key is None and entry is not None:
                resolved_key = (entry.channel, entry.utterance_id)
            parts = [f"decision={decision}"]
            if disposition is not None:
                parts.append(f"disposition={disposition}")
            if resolved_key is not None:
                parts.append(f"entry={self._format_entry_key(resolved_key)}")
            if entry is not None:
                publishable = self._presentation_state.entry_is_publishable(
                    entry,
                    show_peer_original=self.show_peer_original,
                    translation_enabled=self.translation_enabled,
                )
                parts.extend(
                    [
                        f"channel={entry.channel}",
                        f"publishable={publishable}",
                        f"ever_visible={entry.ever_visible}",
                        "ever_visible_with_translation="
                        f"{entry.translation_observed_visible_since is not None}",
                        f"retained_hidden={entry.retained_hidden}",
                    ]
                )
            if block is not None:
                parts.extend(
                    [
                        f"block_variant={block.block_variant}",
                        f"primary_len={len(block.primary_text)}",
                        f"secondary_len={len(block.secondary_text)}",
                    ]
                )
            if extras is not None:
                for field_name, value in extras.items():
                    parts.append(f"{field_name}={value}")
            return f"[OverlayPresenter][Decision] {' '.join(parts)}"

        return self._emit_detailed_lazy(build_message)

    def _emit_pair_state(
        self,
        key: tuple[str, UUID],
        entry: _LogicalTurnEntry,
        block: OverlayPresentationBlock,
        *,
        publish_kind: str,
    ) -> bool:
        def build_message() -> str:
            rendered_primary_source, rendered_secondary_source = self._rendered_text_sources(
                entry,
                block,
            )
            parts = [
                "[OverlayPresenter][PairState]",
                f"entry={self._format_entry_key(key)}",
                f"channel={entry.channel}",
                f"block_variant={block.block_variant}",
                f"publish_kind={publish_kind}",
                f"update_id={block.update_id}",
                f"origin_wall_clock_ms={block.origin_wall_clock_ms}",
                f"source_text_hash={block.source_text_hash}",
                f"source_text_len={block.source_text_len}",
                f"original_seq={entry.original_seq}",
                f"translation_seq={entry.translation_seq}",
                "rendered_pair_state="
                f"{self._rendered_pair_state(rendered_primary_source, rendered_secondary_source)}",
                f"rendered_primary_source={rendered_primary_source}",
                f"rendered_secondary_source={rendered_secondary_source}",
                f"appearance_seq={block.appearance_seq}",
                f"primary_len={len(block.primary_text)}",
                f"secondary_len={len(block.secondary_text) if block.secondary_enabled else 0}",
            ]
            elapsed_ms = self._elapsed_from_origin_wall_clock_ms(block.origin_wall_clock_ms)
            if elapsed_ms is not None:
                parts.append(f"elapsed_ms={elapsed_ms}")
            return " ".join(parts)

        return self._emit_detailed_lazy(build_message)

    def _emit_skip_disposition(
        self,
        *,
        decision: str,
        disposition: str,
        key: tuple[str, UUID] | None = None,
        entry: _LogicalTurnEntry | None = None,
        extras: dict[str, object] | None = None,
    ) -> bool:
        return self._emit_turn_decision(
            decision,
            disposition=disposition,
            key=key,
            entry=entry,
            extras=extras,
        )

    def _emit_reduction_decisions(
        self,
        decisions: tuple[OverlayTurnDecisionRecord, ...],
    ) -> None:
        for decision in decisions:
            self._emit_turn_decision(
                decision.decision,
                disposition=decision.disposition,
                key=decision.key,
                entry=decision.entry,
                block=decision.block,
                extras=decision.extras,
            )

    def _finish_reduction_result(self, result: OverlayReductionResult) -> bool:
        self._emit_reduction_decisions(result.decisions)
        self._drain_presentation_state_removals()
        return result.changed

    def _elapsed_from_origin_wall_clock_ms(self, origin_wall_clock_ms: int | None) -> int | None:
        if origin_wall_clock_ms is None:
            return None
        return max(0, int(time.time() * 1000) - origin_wall_clock_ms)

    def _rendered_text_sources(
        self,
        entry: _LogicalTurnEntry,
        block: OverlayPresentationBlock,
    ) -> tuple[str, str]:
        if not self.translation_enabled and entry.channel == "peer":
            if block.block_variant == "active_peer":
                return "live_text", "none"
            return "original_text", "none"
        if entry.channel == "peer" and block.block_variant == "active_peer":
            secondary_visible = block.secondary_enabled and bool(block.secondary_text.strip())
            return "blank", "source" if secondary_visible else "blank"
        if entry.channel == "peer" and block.block_variant == "finalized":
            if entry.translation_text.strip():
                secondary_visible = block.secondary_enabled and bool(block.secondary_text.strip())
                return "translation", "source" if secondary_visible else "blank"
            secondary_visible = block.secondary_enabled and bool(block.secondary_text.strip())
            return "blank", "source" if secondary_visible else "blank"

        secondary_source = "none"
        if block.secondary_enabled and block.secondary_text:
            if entry.channel == "peer":
                secondary_source = "original_text"
            elif block.block_variant == "active_self" and entry.live_secondary_text.strip():
                secondary_source = "live_secondary_text"
            else:
                secondary_source = "translation_text"

        if block.block_variant == "active_self":
            return "live_text", secondary_source
        if entry.channel == "peer":
            return "translation_text", secondary_source
        return "original_text", secondary_source

    def _rendered_pair_state(self, primary_source: str, secondary_source: str) -> str:
        if primary_source == "live_text":
            if secondary_source == "live_secondary_text":
                return "live_with_preview_translation"
            if secondary_source == "translation_text":
                return "live_with_translation"
            return "live_only"
        if primary_source == "translation_text":
            if secondary_source == "original_text":
                return "translation_with_original"
            return "translation_only"
        if primary_source == "translation":
            if secondary_source == "source":
                return "translation_with_original"
            return "translation_only"
        if primary_source == "blank" and secondary_source == "source":
            return "source_only"
        if secondary_source in {"translation_text", "live_secondary_text"}:
            return "original_with_translation"
        return "original_only"

    def _terminal_update_reason(
        self,
        channel: str | None,
        utterance_id: UUID | None,
    ) -> str | None:
        key = self._entry_key(channel, utterance_id)
        reason = self._scene_terminal_keys.get(key)
        if reason is not None:
            return reason
        if key in self._terminal_registry:
            return ""
        return None

    def _remember_scene_terminal_reason(self, key: tuple[str, UUID], *, reason: str) -> None:
        self._scene_terminal_keys.pop(key, None)
        self._scene_terminal_keys[key] = reason
        while len(self._scene_terminal_keys) > _CLOSED_TOMBSTONE_LIMIT:
            self._scene_terminal_keys.popitem(last=False)

    def attach_bridge(self, bridge: OverlayPresentationTransport) -> None:
        self.bridge = bridge

    def detach_bridge(self) -> None:
        self.bridge = None

    def snapshot(self) -> OverlayPresentationSnapshot:
        return self._presentation_state.snapshot()

    def reset_scene(self) -> None:
        self._cancel_peer_presentation_refresh_burst_task()
        self._cancel_self_presentation_refresh_burst_task(reason="scene_reset")
        self._cancel_all_expiration_tasks()
        self._clear_entries_for_reason("scene_reset")
        self._terminal_registry.clear()
        self._scene_terminal_keys.clear()
        self._application_receipts.clear()
        self._application_receipts_by_id.clear()
        self._entry_ordering.clear()
        self._entry_sequence_namespaces.clear()
        self._retired_turn_frontiers.clear()
        self._retired_preview_self_seqs.clear()
        self._live_self_turn_key = None
        self._live_peer_turn_key = None
        self._revision = 0
        self._appearance_seq = 0
        self._native_fresh_render_generations = NativeFreshRenderGenerations()
        self._native_fresh_render_targets = NativeFreshRenderTargets()
        self._native_quiet_tail_episodes = NativeQuietTailEpisodes()
        self._native_quiet_tail_self_target = None
        self._native_quiet_tail_peer_target = None
        self._last_visible_window_signature = None
        peer_refresh_key = self._presentation_state.peer_presentation_refresh_target_key
        if peer_refresh_key is not None:
            self._presentation_state.end_peer_presentation_refresh(peer_refresh_key)
        self_refresh_key = self._presentation_state.self_presentation_refresh_target_key
        if self_refresh_key is not None:
            self._presentation_state.end_self_presentation_refresh(self_refresh_key)
        self._presentation_state.generate_snapshot(
            revision=0,
            calibration=_calibration_from_overlay(self.calibration),
            rendered_entries=[],
        )

    async def clear_for_runtime_detach(self) -> None:
        await self._cancel_peer_presentation_refresh_burst_task_and_wait()
        await self._cancel_self_presentation_refresh_burst_task_and_wait(reason="runtime_detach")
        await self._cancel_all_expiration_tasks_and_wait()
        self._clear_entries_for_reason("scene_reset")
        self._terminal_registry.clear()
        self._scene_terminal_keys.clear()
        self._application_receipts.clear()
        self._application_receipts_by_id.clear()
        self._entry_ordering.clear()
        self._retired_turn_frontiers.clear()
        self._entry_sequence_namespaces.clear()
        self._retired_preview_self_seqs.clear()
        self._live_self_turn_key = None
        self._live_peer_turn_key = None
        self._revision += 1
        self._native_fresh_render_generations = NativeFreshRenderGenerations()
        self._native_fresh_render_targets = NativeFreshRenderTargets()
        self._native_quiet_tail_episodes = NativeQuietTailEpisodes()
        self._native_quiet_tail_self_target = None
        self._native_quiet_tail_peer_target = None
        self._last_visible_window_signature = None
        peer_refresh_key = self._presentation_state.peer_presentation_refresh_target_key
        if peer_refresh_key is not None:
            self._presentation_state.end_peer_presentation_refresh(peer_refresh_key)
        self_refresh_key = self._presentation_state.self_presentation_refresh_target_key
        if self_refresh_key is not None:
            self._presentation_state.end_self_presentation_refresh(self_refresh_key)
        snapshot = self._presentation_state.generate_snapshot(
            revision=self._revision,
            calibration=_calibration_from_overlay(self.calibration),
            rendered_entries=[],
        )
        if self.bridge is not None:
            await self.bridge.replace_snapshot(snapshot, block_expirations={})

    def application_receipt(self, publication_id: str) -> OverlayApplicationReceipt | None:
        return self._application_receipts_by_id.get(publication_id)

    async def emit(self, event: OverlayEventUnion) -> OverlayApplicationReceipt:
        receipt_key = self._application_receipt_key(event)
        existing = self._application_receipts.get(receipt_key)
        if existing is not None:
            return existing
        async with self._ownership_transition_lock:
            existing = self._application_receipts.get(receipt_key)
            if existing is not None:
                return existing
            event = self._normalize_event_sequence(event)
            rejection_reason = self._application_rejection_reason(event)
            if rejection_reason is not None:
                if (
                    rejection_reason == "stale"
                    and event.channel in {"self", "peer"}
                    and event.utterance_id is not None
                    and (event.channel, event.utterance_id) not in self._entries
                ):
                    key = (event.channel, event.utterance_id)
                    terminal_reason = self._terminal_update_reason(*key)
                    idle_hidden = terminal_reason == "expired"
                    self._emit_turn_decision(
                        (
                            "overlay_turn_late_update_ignored_after_idle_hide"
                            if idle_hidden
                            else "overlay_turn_late_update_ignored_after_eviction"
                        ),
                        disposition="hidden_idle_ttl" if idle_hidden else "evicted",
                        key=key,
                        extras={
                            "event_seq": event.seq,
                            "terminal_reason": terminal_reason or "retired_frontier",
                        },
                    )
                elif (
                    rejection_reason == "stale"
                    and event.channel in {"self", "peer"}
                    and event.utterance_id is not None
                ):
                    key = (event.channel, event.utterance_id)
                    self._emit_turn_decision(
                        "overlay_turn_superseded",
                        disposition="superseded",
                        key=key,
                        entry=self._entries.get(key),
                        extras={"event_seq": event.seq},
                    )
                receipt = OverlayApplicationReceipt(
                    stage="application_accepted",
                    outcome="stale" if rejection_reason == "stale" else "not_applied",
                    publication_id=event.event_id,
                    scene_revision=self._revision,
                    cause=rejection_reason,
                )
                self._remember_application_receipt(receipt_key, receipt)
                return receipt
            await self._emit_serialized(event)
            receipt = OverlayApplicationReceipt(
                stage="application_accepted",
                outcome="applied",
                publication_id=event.event_id,
                scene_revision=self._revision,
            )
            self._remember_application_receipt(receipt_key, receipt)
            return receipt

    async def _emit_serialized(self, event: OverlayEventUnion) -> None:
        previous_snapshot = self.snapshot()
        changed = self._apply_event(event)
        self._record_entry_ordering(event)
        peer_event_is_current = self._peer_presentation_refresh_event_is_current(event)
        peer_event_is_visible = (
            peer_event_is_current
            and event.utterance_id is not None
            and self._snapshot_has_refreshable_peer_key(("peer", event.utterance_id))
        )
        if changed or peer_event_is_visible:
            await self._publish_if_changed(
                fresh_render_event=event,
                event_changed=changed,
            )
        if changed or peer_event_is_current:
            await self._start_peer_presentation_refresh_burst_for_event(event)
        if changed:
            await self._start_self_presentation_refresh_burst_for_event(
                event,
                previous_snapshot=previous_snapshot,
            )

    def _application_rejection_reason(self, event: OverlayEventUnion) -> str | None:
        if self._closing or self._closed:
            return "closed"
        now = self.clock.now()
        self._expire_closed_entries(now=now)
        event_payload_bytes = self._overlay_event_payload_bytes(event)
        if event_payload_bytes > _PRESENTER_EVENT_BYTE_LIMIT:
            return "presenter_payload_exhausted"
        key = (
            (event.channel, event.utterance_id)
            if event.channel in {"self", "peer"} and event.utterance_id is not None
            else None
        )
        entry = self._entries.get(key) if key is not None else None
        if (
            key is not None
            and entry is None
            and event.turn_generation is not None
            and event.turn_order is not None
        ):
            scope = event.turn_kind or str(event.channel)
            if event.turn_order <= self._retired_turn_frontiers.get(
                (scope, event.turn_generation),
                -1,
            ):
                return "stale"
        if key is not None and entry is None and self._terminal_update_reason(*key) is not None:
            return "stale"
        if entry is not None and event.seq <= entry.last_updated_seq:
            return "stale"
        creates_entry = not isinstance(event, (SelfActiveClear, UtteranceClosed))
        if (
            key is not None
            and entry is None
            and creates_entry
            and len(self._entries) >= _PRESENTER_ENTRY_LIMIT
        ):
            return "presenter_overload"
        current_payload_bytes = sum(
            self._entry_payload_bytes(candidate) for candidate in self._entries.values()
        )
        projected_entry_bytes = self._projected_entry_payload_bytes(entry, event)
        previous_entry_bytes = self._entry_payload_bytes(entry) if entry is not None else 0
        if (
            current_payload_bytes - previous_entry_bytes + projected_entry_bytes
            > _PRESENTER_AGGREGATE_BYTE_LIMIT
        ):
            return "presenter_payload_exhausted"
        return None

    def _normalize_event_sequence(self, event: OverlayEventUnion) -> OverlayEventUnion:
        if (
            event.sequence_namespace == 0
            or event.channel not in {"self", "peer"}
            or event.utterance_id is None
        ):
            return event
        key = (event.channel, event.utterance_id)
        namespace = self._entry_sequence_namespaces.get(key)
        if namespace is None or namespace[0] != event.sequence_namespace:
            entry = self._entries.get(key)
            offset = entry.last_updated_seq if entry is not None else 0
            namespace = (event.sequence_namespace, offset)
            self._entry_sequence_namespaces[key] = namespace
        return replace(event, seq=namespace[1] + event.seq)

    def _record_entry_ordering(self, event: OverlayEventUnion) -> None:
        if (
            event.channel not in {"self", "peer"}
            or event.utterance_id is None
            or event.turn_generation is None
            or event.turn_order is None
        ):
            return
        key = (event.channel, event.utterance_id)
        if key not in self._entries:
            return
        self._entry_ordering[key] = (
            event.turn_kind or str(event.channel),
            event.turn_generation,
            event.turn_order,
        )

    def _remember_application_receipt(
        self,
        key: tuple[object, ...],
        receipt: OverlayApplicationReceipt,
    ) -> None:
        self._application_receipts.pop(key, None)
        self._application_receipts[key] = receipt
        self._application_receipts_by_id[receipt.publication_id] = receipt
        while len(self._application_receipts) > _PRESENTER_RECEIPT_LIMIT:
            _, evicted = self._application_receipts.popitem(last=False)
            if self._application_receipts_by_id.get(evicted.publication_id) is evicted:
                self._application_receipts_by_id.pop(evicted.publication_id, None)

    @staticmethod
    def _application_receipt_key(event: OverlayEventUnion) -> tuple[object, ...]:
        if event.turn_generation is not None:
            return (
                event.turn_kind or event.channel,
                event.turn_generation,
                event.event_id,
            )
        return (
            event.event_id,
            event.type,
            event.channel,
            event.utterance_id,
            event.seq,
            event.created_at,
            getattr(event, "text", None),
            getattr(event, "secondary_text", None),
            getattr(event, "update_id", None),
        )

    @staticmethod
    def _entry_payload_bytes(entry: _LogicalTurnEntry | None) -> int:
        if entry is None:
            return 0
        total = 256
        for field_name in (
            "live_text",
            "live_secondary_text",
            "live_primary_language",
            "live_secondary_language",
            "live_update_id",
            "live_session_scope",
            "live_source_text_hash",
            "live_logical_turn_key",
            "original_text",
            "original_language",
            "translation_text",
            "translation_language",
            "translation_update_id",
            "translation_session_scope",
            "translation_source_text_hash",
            "translation_logical_turn_key",
            "occupant_key",
        ):
            value = getattr(entry, field_name)
            if isinstance(value, str):
                total += len(value.encode("utf-8"))
        return total

    @staticmethod
    def _overlay_event_payload_bytes(event: OverlayEventUnion) -> int:
        total = 256
        for field_name in (
            "event_id",
            "update_id",
            "session_scope",
            "source_text_hash",
            "logical_turn_key",
            "text",
            "source_text",
            "secondary_text",
            "source_language",
            "target_language",
            "occupant_key",
        ):
            value = getattr(event, field_name, None)
            if isinstance(value, str):
                total += len(value.encode("utf-8"))
        return total

    def _projected_entry_payload_bytes(
        self,
        entry: _LogicalTurnEntry | None,
        event: OverlayEventUnion,
    ) -> int:
        if entry is None:
            return self._overlay_event_payload_bytes(event)
        current = self._entry_payload_bytes(entry)
        replacements: tuple[tuple[str, str | None], ...] = ()
        if isinstance(event, (SelfTranscriptFinal, PeerTranscriptFinal)):
            replacements = (
                ("original_text", event.text),
                ("original_language", event.source_language),
            )
        elif isinstance(event, (TranslationStreamUpdate, TranslationFinal)):
            replacements = (
                ("translation_text", event.text),
                ("translation_language", event.target_language),
                ("translation_update_id", event.update_id),
                ("translation_session_scope", event.session_scope),
                ("translation_source_text_hash", event.source_text_hash),
                ("translation_logical_turn_key", event.logical_turn_key),
            )
        elif isinstance(event, (SelfActiveUpdate, PeerActiveUpdate)):
            replacements = (
                ("live_text", event.text),
                ("live_update_id", event.update_id),
                ("live_session_scope", event.session_scope),
                ("live_source_text_hash", event.source_text_hash),
                ("live_logical_turn_key", event.logical_turn_key),
                ("occupant_key", event.occupant_key),
            )
            if isinstance(event, SelfActiveUpdate):
                replacements += (("live_secondary_text", event.secondary_text),)
        for field_name, replacement in replacements:
            current_value = getattr(entry, field_name)
            if isinstance(current_value, str):
                current -= len(current_value.encode("utf-8"))
            if isinstance(replacement, str):
                current += len(replacement.encode("utf-8"))
        return max(0, current)

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

    async def update_peer_presentation_refresh_burst(self, enabled: bool) -> None:
        async with self._ownership_transition_lock:
            await self._update_peer_presentation_refresh_burst_serialized(enabled)

    async def _update_peer_presentation_refresh_burst_serialized(self, enabled: bool) -> None:
        next_enabled = bool(enabled)
        if next_enabled == self.peer_presentation_refresh_burst:
            return
        self.peer_presentation_refresh_burst = next_enabled
        if not next_enabled:
            await self._cancel_peer_presentation_refresh_burst_task_and_wait()
            peer_refresh_key = self._presentation_state.peer_presentation_refresh_target_key
            if (
                peer_refresh_key is not None
                and self._presentation_state.end_peer_presentation_refresh(peer_refresh_key)
            ):
                await self._publish_if_changed()

    async def update_self_presentation_refresh_burst(self, enabled: bool) -> None:
        async with self._ownership_transition_lock:
            await self._update_self_presentation_refresh_burst_serialized(enabled)

    async def _update_self_presentation_refresh_burst_serialized(self, enabled: bool) -> None:
        next_enabled = bool(enabled)
        if next_enabled == self.self_presentation_refresh_burst:
            return
        self.self_presentation_refresh_burst = next_enabled
        if not next_enabled:
            await self._cancel_self_presentation_refresh_burst_task_and_wait(
                reason="disabled",
                allow_task_cleanup=True,
            )
            self_refresh_key = self._presentation_state.self_presentation_refresh_target_key
            if (
                self_refresh_key is not None
                and self._presentation_state.end_self_presentation_refresh(self_refresh_key)
            ):
                await self._publish_if_changed()

    async def discard_epoch_retry_intent(self) -> None:
        async with self._ownership_transition_lock:
            await self._cancel_peer_presentation_refresh_burst_task_and_wait()
            await self._cancel_self_presentation_refresh_burst_task_and_wait(reason="epoch_discard")
            self.native_retry_trigger_emission = False
            self.peer_presentation_refresh_burst = False
            self.self_presentation_refresh_burst = False
            self._native_fresh_render_generations = NativeFreshRenderGenerations()
            self._native_fresh_render_targets = NativeFreshRenderTargets()
            self._native_quiet_tail_episodes = NativeQuietTailEpisodes()
            self._native_quiet_tail_self_target = None
            self._native_quiet_tail_peer_target = None
            self._native_quiet_tail_self_generation = None
            self._native_quiet_tail_peer_generation = None
            peer_refresh_key = self._presentation_state.peer_presentation_refresh_target_key
            if peer_refresh_key is not None:
                self._presentation_state.end_peer_presentation_refresh(peer_refresh_key)
            self_refresh_key = self._presentation_state.self_presentation_refresh_target_key
            if self_refresh_key is not None:
                self._presentation_state.end_self_presentation_refresh(self_refresh_key)
            await self._publish_if_changed(force_protocol_publish=True)

    async def update_native_retry_ownership(self, confirmed: bool) -> None:
        async with self._ownership_transition_lock:
            await self._update_native_retry_ownership_serialized(confirmed)

    async def _update_native_retry_ownership_serialized(self, confirmed: bool) -> None:
        confirmed = bool(confirmed)
        if self._closing or self._closed:
            self.native_retry_trigger_emission = False
            self._native_fresh_render_generations = NativeFreshRenderGenerations()
            self._native_fresh_render_targets = NativeFreshRenderTargets()
            return
        if confirmed:
            if (
                self.native_retry_trigger_emission
                and not self.peer_presentation_refresh_burst
                and not self.self_presentation_refresh_burst
            ):
                return
            active_targets = self._active_python_retry_targets()
            if not active_targets:
                active_targets = self._active_native_retry_targets()
            await self._update_peer_presentation_refresh_burst_serialized(False)
            await self._update_self_presentation_refresh_burst_serialized(False)
            self.native_retry_trigger_emission = True
            self._synchronize_native_retry_targets(active_targets)
            await self._publish_if_changed(force_protocol_publish=True)
            return
        if (
            not self.native_retry_trigger_emission
            and self.peer_presentation_refresh_burst
            and self.self_presentation_refresh_burst
            and self.snapshot().native_fresh_render_generations is None
            and self.snapshot().native_fresh_render_targets is None
        ):
            return
        active_targets = self._active_native_retry_targets()
        self.native_retry_trigger_emission = False
        self._native_fresh_render_generations = NativeFreshRenderGenerations()
        self._native_fresh_render_targets = NativeFreshRenderTargets()
        await self._publish_if_changed(force_protocol_publish=True)
        await self._update_peer_presentation_refresh_burst_serialized(True)
        await self._update_self_presentation_refresh_burst_serialized(True)
        await self._restart_python_retry_targets(active_targets)

    async def broadcast_shutdown(self) -> None:
        if self.bridge is None:
            return
        await self.bridge.broadcast_shutdown()

    async def close(self) -> None:
        async with self._ownership_transition_lock:
            if self._closed:
                return
            self._closing = True
            await self.clear_for_runtime_detach()
            self.native_retry_trigger_emission = False
            self._native_fresh_render_generations = NativeFreshRenderGenerations()
            self._native_fresh_render_targets = NativeFreshRenderTargets()
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
        key = self._entry_key(channel, utterance_id)
        return key in self._scene_terminal_keys or key in self._terminal_registry

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
        selection = self._presentation_state.visible_block_selection(
            entries=self._entries,
            live_self_entry=self._live_self_entry(),
            live_peer_entry=self._live_peer_entry(),
            visible_window_target_blocks=self.visible_window_target_blocks,
            show_translation=self.show_translation,
            show_peer_original=self.show_peer_original,
            peer_presentation_refresh_burst=self.peer_presentation_refresh_burst,
            self_presentation_refresh_burst=self.self_presentation_refresh_burst,
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
        self._record_visible_window_selection(
            active_self_present=selection.active_self_present,
            finalized_limit=selection.finalized_limit,
            candidate_keys=selection.candidate_keys,
            selected_keys=selection.selected_keys,
            protected_selected=selection.protected_keys,
            retained_hidden=selection.retained_hidden,
        )
        rendered_entries = selection.rendered_entries
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
        previous_signatures = {
            block.id: self._presentation_state.rendered_block_signature(block)
            for block in previous_snapshot.blocks
        }
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
            self._emit_turn_decision(
                "overlay_turn_no_visible_change",
                disposition="rendered_signature_unchanged",
                extras={"block_count": len(next_blocks)},
            )
            return

        for key, block in rendered_entries:
            entry = self._entries.get(key)
            if entry is None:
                continue
            previous_signature = previous_signatures.get(block.id)
            if previous_signature is None:
                self._emit_turn_decision(
                    "overlay_turn_first_visible",
                    key=key,
                    entry=entry,
                    block=block,
                )
                self._emit_pair_state(key, entry, block, publish_kind="first_visible")
                continue
            if previous_signature != self._presentation_state.rendered_block_signature(block):
                self._emit_turn_decision(
                    "overlay_turn_updated",
                    key=key,
                    entry=entry,
                    block=block,
                )
                self._emit_pair_state(key, entry, block, publish_kind="visible_update")

        if fresh_render_channel is not None and self.native_retry_trigger_emission:
            self._increment_native_fresh_render_generation(
                fresh_render_channel,
                event=fresh_render_event,
            )
            self._advance_native_quiet_tail_episode(
                fresh_render_channel,
                event=fresh_render_event,
            )
        self._prune_native_quiet_tail_targets(rendered_entries)
        self._revision += 1
        snapshot = self._presentation_state.generate_snapshot(
            revision=self._revision,
            calibration=next_calibration,
            rendered_entries=rendered_entries,
            native_fresh_render_generations=(
                self._native_fresh_render_generations
                if self.native_retry_trigger_emission
                else None
            ),
            native_fresh_render_targets=(
                self._native_fresh_render_targets if self.native_retry_trigger_emission else None
            ),
            native_quiet_tail_episodes=(
                self._native_quiet_tail_episodes if self.native_retry_trigger_emission else None
            ),
            entry_ordering=self._entry_ordering,
            semantic_retirement_frontiers=self._retired_turn_frontiers,
        )
        blocks_summary = [
            {
                "id": block.id,
                "variant": block.block_variant,
                "update_id": block.update_id,
                "origin_wall_clock_ms": block.origin_wall_clock_ms,
                "session_scope": block.session_scope,
                "primary_len": len(block.primary_text),
                "secondary_len": len(block.secondary_text),
            }
            for block in next_blocks
        ]
        self._emit_detailed_lazy(
            lambda: "[OverlayPresenter] Snapshot publish: revision=%s block_count=%s bridge_attached=%s blocks=%s"
            % (
                snapshot.revision,
                len(next_blocks),
                self.bridge is not None,
                blocks_summary,
            )
        )
        if self.diagnostics is not None:
            self.diagnostics.record_presenter(
                "snapshot_publish",
                revision=snapshot.revision,
                block_count=len(next_blocks),
                bridge_attached=self.bridge is not None,
                blocks=blocks_summary,
            )
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

    def _next_appearance_seq(self) -> int:
        self._appearance_seq += 1
        return self._appearance_seq

    def _remember_tombstone(self, key: tuple[str, UUID], closed_seq: int) -> None:
        self._terminal_registry.pop(key, None)
        self._terminal_registry[key] = closed_seq
        while len(self._terminal_registry) > _CLOSED_TOMBSTONE_LIMIT:
            self._terminal_registry.popitem(last=False)

    def _schedule_expiration(
        self,
        key: tuple[str, UUID],
        entry: _LogicalTurnEntry,
    ) -> None:
        self._cancel_expiration_task(key)
        if self._entry_expiration_deadline(entry) is None:
            return
        entry.expiration_revision += 1
        self._record_deadline(entry)
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
        ordering = self._entry_ordering.pop(key, None)
        self._entry_sequence_namespaces.pop(key, None)
        if ordering is not None:
            scope, generation, order = ordering
            frontier_key = (scope, generation)
            self._retired_turn_frontiers[frontier_key] = max(
                order,
                self._retired_turn_frontiers.get(frontier_key, -1),
            )
        effective_deadline, visible_deadline, translation_deadline = (
            self._entry_expiration_components(entry)
        )
        removal_time = record.now if record.now is not None else self.clock.now()
        extra_fields: dict[str, object] = {}
        if entry.channel == "self":
            lifetime_ms = 0.0
            if entry.visible_since is not None:
                lifetime_ms = max(0.0, (removal_time - entry.visible_since) * 1000.0)
            translated_lifetime_ms = 0.0
            if entry.translation_observed_visible_since is not None:
                translated_lifetime_ms = max(
                    0.0,
                    (removal_time - entry.translation_observed_visible_since) * 1000.0,
                )
            extra_fields = {
                "lifetime_ms": lifetime_ms,
                "translated_lifetime_ms": translated_lifetime_ms,
                "had_translation": bool(entry.translation_text.strip()),
                "ever_visible_with_translation": entry.translation_observed_visible_since
                is not None,
                "translation_observed_visible_since": entry.translation_observed_visible_since,
            }
        if self.diagnostics is not None:
            self.diagnostics.record_presenter_removal(
                reason=record.reason,
                entry_key=self._format_entry_key(key),
                appearance_seq=entry.appearance_seq,
                channel=entry.channel,
                primary_len=len(entry.original_text.strip()),
                secondary_len=len(entry.translation_text.strip()),
                visible_since=entry.visible_since,
                translation_visible_since=entry.translation_visible_since,
                closed_at=entry.closed_at,
                now=removal_time,
                visible_deadline=visible_deadline,
                translation_deadline=translation_deadline,
                effective_deadline=effective_deadline,
                **extra_fields,
            )
        seq = record.tombstone_seq if record.tombstone_seq is not None else entry.closed_seq
        if record.reason == "expired" and entry.ever_visible:
            self._remember_scene_terminal_reason(key, reason=record.reason)
            self._emit_turn_decision(
                "overlay_turn_hidden_idle_ttl",
                disposition="hidden_idle_ttl",
                key=key,
                entry=entry,
                extras={"deadline": effective_deadline},
            )
        if record.reason == "evicted_by_newer_turn":
            self._remember_scene_terminal_reason(key, reason=record.reason)
            self._emit_turn_decision(
                "overlay_turn_evicted_by_newer_turn",
                disposition="evicted",
                key=key,
                entry=entry,
            )
        if seq is not None:
            self._remember_tombstone(key, seq)

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
                    previous_block = self._refreshable_self_block_in_snapshot(
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

    def _advance_native_quiet_tail_episode(
        self,
        channel: str,
        *,
        event: OverlayEventUnion | None,
    ) -> None:
        if event is None or event.utterance_id is None:
            return
        phase = (
            "final"
            if isinstance(
                event,
                (SelfTranscriptFinal, PeerTranscriptFinal, TranslationFinal),
            )
            else "stream"
        )
        current = (
            self._native_quiet_tail_episodes.self
            if channel == "self"
            else self._native_quiet_tail_episodes.peer
        )
        current_target = (
            self._native_quiet_tail_self_target
            if channel == "self"
            else self._native_quiet_tail_peer_target
        )
        target = f"{channel}:{event.utterance_id}"
        replace = current is None or current.phase != phase or current_target != target
        if channel == "self" and isinstance(event, TranslationFinal):
            replace = True
        episode = (
            NativeQuietTailEpisode(
                phase=phase,
                generation=self._next_native_fresh_render_generation(
                    self._native_quiet_tail_self_generation
                    if channel == "self"
                    else self._native_quiet_tail_peer_generation
                ),
            )
            if replace
            else current
        )
        if channel == "self":
            self._native_quiet_tail_self_target = target
            if replace:
                self._native_quiet_tail_self_generation = episode.generation
            self._native_quiet_tail_episodes = NativeQuietTailEpisodes(
                self=episode,
                peer=self._native_quiet_tail_episodes.peer,
            )
        else:
            self._native_quiet_tail_peer_target = target
            if replace:
                self._native_quiet_tail_peer_generation = episode.generation
            self._native_quiet_tail_episodes = NativeQuietTailEpisodes(
                self=self._native_quiet_tail_episodes.self,
                peer=episode,
            )

    def _prune_native_quiet_tail_targets(
        self,
        rendered_entries: list[tuple[tuple[str, UUID], OverlayPresentationBlock]],
    ) -> None:
        visible = {block.id for _, block in rendered_entries if block.primary_text.strip()}
        self_target = self._native_fresh_render_targets.self
        peer_target = self._native_fresh_render_targets.peer
        if self_target is not None and self_target not in visible:
            self_target = None
            self._native_quiet_tail_self_target = None
        if peer_target is not None and peer_target not in visible:
            peer_target = None
            self._native_quiet_tail_peer_target = None
        self._native_fresh_render_targets = NativeFreshRenderTargets(
            self=self_target,
            peer=peer_target,
        )
        self._native_quiet_tail_episodes = NativeQuietTailEpisodes(
            self=(self._native_quiet_tail_episodes.self if self_target is not None else None),
            peer=(self._native_quiet_tail_episodes.peer if peer_target is not None else None),
        )

    def _increment_native_fresh_render_generation(
        self,
        channel: str,
        *,
        event: OverlayEventUnion | None,
    ) -> None:
        if event is None or event.utterance_id is None:
            return
        target = f"{channel}:{event.utterance_id}"
        generations = self._native_fresh_render_generations
        targets = self._native_fresh_render_targets
        if channel == "self":
            self._native_fresh_render_generations = NativeFreshRenderGenerations(
                self=self._next_native_fresh_render_generation(generations.self),
                peer=generations.peer,
            )
            self._native_fresh_render_targets = NativeFreshRenderTargets(
                self=target,
                peer=targets.peer,
            )
            return
        self._native_fresh_render_generations = NativeFreshRenderGenerations(
            self=generations.self,
            peer=self._next_native_fresh_render_generation(generations.peer),
        )
        self._native_fresh_render_targets = NativeFreshRenderTargets(
            self=targets.self,
            peer=target,
        )

    def _active_python_retry_targets(self) -> dict[str, tuple[str, UUID]]:
        targets: dict[str, tuple[str, UUID]] = {}
        self_target = self._presentation_state.self_presentation_refresh_target_key
        peer_target = self._presentation_state.peer_presentation_refresh_target_key
        if self_target is not None and self._snapshot_has_refreshable_self_key(self_target):
            targets["self"] = self_target
        if peer_target is not None and self._snapshot_has_refreshable_peer_key(peer_target):
            targets["peer"] = peer_target
        return targets

    def _active_native_retry_targets(self) -> dict[str, tuple[str, UUID]]:
        targets: dict[str, tuple[str, UUID]] = {}
        for channel, target in (
            ("self", self._native_fresh_render_targets.self),
            ("peer", self._native_fresh_render_targets.peer),
        ):
            if target is None:
                continue
            prefix, separator, raw_id = target.partition(":")
            if separator and prefix == channel:
                try:
                    targets[channel] = (channel, UUID(raw_id))
                except ValueError:
                    continue
        return targets

    async def _restart_python_retry_targets(
        self,
        targets: dict[str, tuple[str, UUID]],
    ) -> None:
        self_target = targets.get("self")
        if self_target is not None and self._snapshot_has_refreshable_self_key(self_target):
            self._presentation_state.begin_self_presentation_refresh(self_target)
            self._record_self_presentation_refresh_burst_start(
                self_target,
                reason="native_ownership_released",
            )
            self._self_presentation_refresh_burst_task = (
                self._create_self_presentation_refresh_burst_task(self_target)
            )
        peer_target = targets.get("peer")
        if peer_target is not None and self._snapshot_has_refreshable_peer_key(peer_target):
            self._presentation_state.begin_peer_presentation_refresh(peer_target)
            self._peer_presentation_refresh_burst_task = self._create_task(
                self._run_peer_presentation_refresh_burst(peer_target),
                task_name="presenter-peer-refresh-burst",
            )

    def _synchronize_native_retry_targets(
        self,
        active_targets: dict[str, tuple[str, UUID]],
    ) -> None:
        targets: dict[str, str] = {}
        self_target = active_targets.get("self")
        if self_target is not None and self._snapshot_has_refreshable_self_key(self_target):
            targets["self"] = f"{self_target[0]}:{self_target[1]}"
        peer_target = active_targets.get("peer")
        if peer_target is not None and self._snapshot_has_refreshable_peer_key(peer_target):
            targets["peer"] = f"{peer_target[0]}:{peer_target[1]}"
        generations = self._native_fresh_render_generations
        self._native_fresh_render_generations = NativeFreshRenderGenerations(
            self=(
                self._next_native_fresh_render_generation(generations.self)
                if "self" in targets
                else None
            ),
            peer=(
                self._next_native_fresh_render_generation(generations.peer)
                if "peer" in targets
                else None
            ),
        )
        self._native_fresh_render_targets = NativeFreshRenderTargets(
            self=targets.get("self"),
            peer=targets.get("peer"),
        )

    def _next_native_fresh_render_generation(self, current: int | None) -> int:
        if current is None:
            return 1
        if current == U64_MAX:
            return 0
        return current + 1

    def _self_presentation_refresh_key_for_event(
        self,
        event: OverlayEventUnion,
    ) -> tuple[str, UUID] | None:
        if not self.self_presentation_refresh_burst:
            return None
        if event.channel != "self" or event.utterance_id is None:
            return None
        if isinstance(event, (SelfTranscriptFinal, TranslationFinal)):
            return ("self", event.utterance_id)
        return None

    def _snapshot_has_refreshable_self_key(self, key: tuple[str, UUID]) -> bool:
        return self._refreshable_self_block_in_snapshot(self.snapshot(), key) is not None

    def _self_presentation_refresh_request_key_for_event(
        self,
        event: OverlayEventUnion,
        *,
        previous_snapshot: OverlayPresentationSnapshot,
    ) -> tuple[str, UUID] | None:
        key = self._self_presentation_refresh_key_for_event(event)
        if key is None:
            return None
        current_block = self._refreshable_self_block_in_snapshot(self.snapshot(), key)
        if current_block is None:
            return None
        previous_block = self._refreshable_self_block_in_snapshot(previous_snapshot, key)
        previous_signature = (
            self._presentation_state.visible_block_content_signature(previous_block)
            if previous_block is not None
            else None
        )
        current_signature = self._presentation_state.visible_block_content_signature(current_block)
        if previous_signature == current_signature:
            return None
        return key

    def _refreshable_self_block_in_snapshot(
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

    def _peer_presentation_refresh_key_for_event(
        self,
        event: OverlayEventUnion,
    ) -> tuple[str, UUID] | None:
        if event.channel != "peer" or event.utterance_id is None:
            return None
        if isinstance(
            event,
            (
                PeerActiveUpdate,
                PeerTranscriptFinal,
                TranslationStreamUpdate,
                TranslationFinal,
            ),
        ):
            return ("peer", event.utterance_id)
        return None

    def _snapshot_has_refreshable_peer_key(self, key: tuple[str, UUID]) -> bool:
        block_id = f"{key[0]}:{key[1]}"
        for block in self.snapshot().blocks:
            if block.channel != "peer" or block.id != block_id:
                continue
            # Only normal product rows made primary-visible by peer translation
            # arrival are refreshable. Source-only finalized rows and reserved
            # active_peer compatibility rows must not start the burst.
            if block.block_variant == "finalized" and block.primary_text.strip():
                return True
        return False

    def _peer_presentation_refresh_event_is_current(self, event: OverlayEventUnion) -> bool:
        key = self._peer_presentation_refresh_key_for_event(event)
        if key is None:
            return False
        entry = self._entries.get(key)
        return entry is not None and entry.last_updated_seq == event.seq

    async def _start_peer_presentation_refresh_burst_for_event(
        self,
        event: OverlayEventUnion,
    ) -> None:
        if not self.peer_presentation_refresh_burst:
            return
        key = self._peer_presentation_refresh_key_for_event(event)
        if key is None or not self._snapshot_has_refreshable_peer_key(key):
            return
        needs_clean_publish = self._presentation_state.begin_peer_presentation_refresh(key)
        self._cancel_peer_presentation_refresh_burst_task()
        if needs_clean_publish:
            await self._publish_if_changed()
        self._peer_presentation_refresh_burst_task = self._create_task(
            self._run_peer_presentation_refresh_burst(key),
            task_name="presenter-peer-refresh-burst",
        )

    async def _start_self_presentation_refresh_burst_for_event(
        self,
        event: OverlayEventUnion,
        *,
        previous_snapshot: OverlayPresentationSnapshot,
    ) -> None:
        key = self._self_presentation_refresh_request_key_for_event(
            event,
            previous_snapshot=previous_snapshot,
        )
        if key is None:
            return
        needs_clean_publish = self._presentation_state.begin_self_presentation_refresh(key)
        self._cancel_self_presentation_refresh_burst_task(
            reason="target_replaced",
            cleanup_publish_count=1 if needs_clean_publish else 0,
        )
        if needs_clean_publish:
            await self._publish_if_changed()
        self._record_self_presentation_refresh_burst_start(
            key,
            reason="eligible_finalized_self_update",
        )
        self._self_presentation_refresh_burst_task = (
            self._create_self_presentation_refresh_burst_task(key)
        )

    async def _run_peer_presentation_refresh_burst(self, key: tuple[str, UUID]) -> None:
        deadline = self.clock.now() + PEER_PRESENTATION_REFRESH_BURST_SECONDS
        try:
            while self.peer_presentation_refresh_burst and self.clock.now() < deadline:
                await self.sleep(PEER_PRESENTATION_REFRESH_BURST_INTERVAL_SECONDS)
                if not self.peer_presentation_refresh_burst:
                    return
                if self._presentation_state.peer_presentation_refresh_target_key != key:
                    return
                if not self._snapshot_has_refreshable_peer_key(key):
                    return
                if not self._presentation_state.tick_peer_presentation_refresh(key):
                    return
                await self._publish_if_changed()
        except asyncio.CancelledError:
            raise
        finally:
            current_task = self._current_task()
            if (
                current_task is not None
                and self._peer_presentation_refresh_burst_task is current_task
            ):
                self._peer_presentation_refresh_burst_task = None
                if self._presentation_state.end_peer_presentation_refresh(key):
                    await self._publish_if_changed()

    async def _run_self_presentation_refresh_burst(self, key: tuple[str, UUID]) -> None:
        deadline = self.clock.now() + PEER_PRESENTATION_REFRESH_BURST_SECONDS
        tick_count = 0
        cleanup_publish_count = 0
        end_reason = "deadline_expired"
        current_task = self._current_task()
        try:
            while self.self_presentation_refresh_burst and self.clock.now() < deadline:
                await self.sleep(PEER_PRESENTATION_REFRESH_BURST_INTERVAL_SECONDS)
                if not self.self_presentation_refresh_burst:
                    end_reason = "disabled"
                    return
                if self._presentation_state.self_presentation_refresh_target_key != key:
                    end_reason = "target_replaced"
                    return
                if not self._snapshot_has_refreshable_self_key(key):
                    end_reason = "target_invalid"
                    return
                if not self._presentation_state.tick_self_presentation_refresh(key):
                    end_reason = "target_replaced"
                    return
                tick_count += 1
                await self._publish_if_changed()
            end_reason = "deadline_expired"
        except asyncio.CancelledError:
            if current_task is not None:
                end_reason = self._self_presentation_refresh_burst_cancel_reasons.get(
                    current_task,
                    "cancelled",
                )
            else:
                end_reason = "cancelled"
            raise
        finally:
            active_task = current_task is not None and (
                self._self_presentation_refresh_burst_task is current_task
            )
            if active_task:
                self._self_presentation_refresh_burst_task = None
                if self._presentation_state.end_self_presentation_refresh(key):
                    await self._publish_if_changed()
                    cleanup_publish_count += 1
            if current_task is not None:
                cleanup_publish_count += (
                    self._self_presentation_refresh_burst_cancel_cleanup_counts.pop(
                        current_task,
                        0,
                    )
                )
                self._self_presentation_refresh_burst_cancel_reasons.pop(current_task, None)
            self._record_self_presentation_refresh_burst_end(
                key,
                reason=end_reason,
                tick_count=tick_count,
                cleanup_publish_count=cleanup_publish_count,
            )

    def _create_self_presentation_refresh_burst_task(
        self,
        key: tuple[str, UUID],
    ) -> asyncio.Task[None]:
        task = self._create_task(
            self._run_self_presentation_refresh_burst(key),
            task_name="presenter-self-refresh-burst",
        )

        def record_unstarted_cancel_end(completed_task: asyncio.Task[None]) -> None:
            self._record_unstarted_self_presentation_refresh_cancel_end(
                completed_task,
                key,
            )

        task.add_done_callback(record_unstarted_cancel_end)
        return task

    def _record_unstarted_self_presentation_refresh_cancel_end(
        self,
        task: asyncio.Task[None],
        key: tuple[str, UUID],
    ) -> None:
        has_cancel_metadata = (
            task in self._self_presentation_refresh_burst_cancel_reasons
            or task in self._self_presentation_refresh_burst_cancel_cleanup_counts
        )
        if not has_cancel_metadata:
            return
        reason = self._self_presentation_refresh_burst_cancel_reasons.pop(
            task,
            "cancelled",
        )
        cleanup_publish_count = self._self_presentation_refresh_burst_cancel_cleanup_counts.pop(
            task, 0
        )
        if self._self_presentation_refresh_burst_task is task:
            self._self_presentation_refresh_burst_task = None
        self._record_self_presentation_refresh_burst_end(
            key,
            reason=reason,
            tick_count=0,
            cleanup_publish_count=cleanup_publish_count,
        )

    def _cancel_peer_presentation_refresh_burst_task(self) -> None:
        task = self._peer_presentation_refresh_burst_task
        self._peer_presentation_refresh_burst_task = None
        if task is not None and not task.done():
            task.cancel()

    def _cancel_self_presentation_refresh_burst_task(
        self,
        *,
        reason: str = "cancelled",
        cleanup_publish_count: int = 0,
    ) -> None:
        task = self._self_presentation_refresh_burst_task
        self._self_presentation_refresh_burst_task = None
        if task is None:
            return
        self._self_presentation_refresh_burst_cancel_reasons[task] = reason
        self._self_presentation_refresh_burst_cancel_cleanup_counts[task] = cleanup_publish_count
        if not task.done():
            task.cancel()
        else:
            self._self_presentation_refresh_burst_cancel_reasons.pop(task, None)
            self._self_presentation_refresh_burst_cancel_cleanup_counts.pop(task, None)

    async def _cancel_peer_presentation_refresh_burst_task_and_wait(self) -> None:
        task = self._peer_presentation_refresh_burst_task
        self._peer_presentation_refresh_burst_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _cancel_self_presentation_refresh_burst_task_and_wait(
        self,
        *,
        reason: str = "cancelled",
        allow_task_cleanup: bool = False,
        cleanup_publish_count: int = 0,
    ) -> None:
        task = self._self_presentation_refresh_burst_task
        if task is None:
            return
        self._self_presentation_refresh_burst_cancel_reasons[task] = reason
        self._self_presentation_refresh_burst_cancel_cleanup_counts[task] = cleanup_publish_count
        if not allow_task_cleanup:
            self._self_presentation_refresh_burst_task = None
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        else:
            self._self_presentation_refresh_burst_cancel_reasons.pop(task, None)
            self._self_presentation_refresh_burst_cancel_cleanup_counts.pop(task, None)
        if allow_task_cleanup and self._self_presentation_refresh_burst_task is task:
            self._self_presentation_refresh_burst_task = None

    def _current_task(self) -> asyncio.Task[None] | None:
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None

    def _clear_entries_for_reason(self, reason: str) -> None:
        for key in list(self._entries):
            self._remove_entry(key, reason=reason, now=self.clock.now())

    def _record_visible_window_selection(
        self,
        *,
        active_self_present: bool,
        finalized_limit: int,
        candidate_keys: list[tuple[str, UUID]],
        selected_keys: list[tuple[str, UUID]],
        protected_selected: list[tuple[str, UUID]],
        retained_hidden: list[tuple[str, UUID]],
    ) -> None:
        if self.diagnostics is None:
            return
        candidate_labels = [self._format_entry_key(key) for key in candidate_keys]
        selected_labels = [self._format_entry_key(key) for key in selected_keys]
        dropped_labels = [label for label in candidate_labels if label not in selected_labels]
        protected_labels = [self._format_entry_key(key) for key in protected_selected]
        retained_hidden_labels = [self._format_entry_key(key) for key in retained_hidden]
        signature = (
            active_self_present,
            finalized_limit,
            tuple(candidate_labels),
            tuple(selected_labels),
            tuple(dropped_labels),
            tuple(protected_labels),
            tuple(retained_hidden_labels),
        )
        if signature == self._last_visible_window_signature:
            return
        self._last_visible_window_signature = signature
        self.diagnostics.record_presenter(
            "visible_window",
            active_self_present=active_self_present,
            finalized_limit=finalized_limit,
            candidate_keys=candidate_labels,
            selected_keys=selected_labels,
            dropped_keys=dropped_labels,
            protected_selected=protected_labels,
            retained_hidden=retained_hidden_labels,
        )

    def _record_deadline(self, entry: _LogicalTurnEntry) -> None:
        if self.diagnostics is None:
            return
        effective_deadline, visible_deadline, translation_deadline = (
            self._entry_expiration_components(entry)
        )
        self.diagnostics.record_presenter(
            "deadline_scheduled",
            entry_key=self._format_entry_key((entry.channel, entry.utterance_id)),
            channel=entry.channel,
            visible_since=entry.visible_since,
            translation_visible_since=entry.translation_visible_since,
            closed_at=entry.closed_at,
            visible_deadline=visible_deadline,
            translation_deadline=translation_deadline,
            effective_deadline=effective_deadline,
        )

    def _record_self_presentation_refresh_burst_start(
        self,
        key: tuple[str, UUID],
        *,
        reason: str,
    ) -> None:
        target_key = self._format_entry_key(key)
        self._emit_detailed_lazy(
            lambda: "[OverlayPresenter][SelfPresentationRefresh] start reason=%s target_key=%s"
            % (reason, target_key)
        )
        if self.diagnostics is not None:
            self.diagnostics.record_presenter(
                "self_presentation_refresh_burst_start",
                reason=reason,
                target_key=target_key,
            )

    def _record_self_presentation_refresh_burst_end(
        self,
        key: tuple[str, UUID],
        *,
        reason: str,
        tick_count: int,
        cleanup_publish_count: int,
    ) -> None:
        target_key = self._format_entry_key(key)
        self._emit_detailed_lazy(
            lambda: "[OverlayPresenter][SelfPresentationRefresh] end "
            "reason=%s target_key=%s tick_count=%s cleanup_publish_count=%s"
            % (reason, target_key, tick_count, cleanup_publish_count)
        )
        if self.diagnostics is not None:
            self.diagnostics.record_presenter(
                "self_presentation_refresh_burst_end",
                reason=reason,
                target_key=target_key,
                tick_count=tick_count,
                cleanup_publish_count=cleanup_publish_count,
            )

    def _format_entry_key(self, key: tuple[str, UUID]) -> str:
        return f"{key[0]}:{key[1]}"


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
