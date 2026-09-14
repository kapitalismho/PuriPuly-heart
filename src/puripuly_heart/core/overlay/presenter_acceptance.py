from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Mapping
from uuid import UUID

from .sink import (
    OverlayApplicationReceipt,
    OverlayEventUnion,
    PeerActiveUpdate,
    PeerTranscriptFinal,
    SelfActiveClear,
    SelfActiveUpdate,
    SelfTranscriptFinal,
    TranslationFinal,
    TranslationStreamUpdate,
    UtteranceClosed,
)
from .state import OverlayLogicalTurnEntry

CLOSED_TOMBSTONE_LIMIT = 64
PRESENTER_ENTRY_LIMIT = 64
PRESENTER_EVENT_BYTE_LIMIT = 1024 * 1024
PRESENTER_AGGREGATE_BYTE_LIMIT = 16 * 1024 * 1024
PRESENTER_RECEIPT_LIMIT = 4096

EntryKey = tuple[str, UUID]


@dataclass(slots=True)
class PresenterAcceptanceLedger:
    """Bookkeeping for local presenter acceptance and semantic retirement."""

    terminal_registry: OrderedDict[EntryKey, int] = field(default_factory=OrderedDict)
    scene_terminal_keys: OrderedDict[EntryKey, str] = field(default_factory=OrderedDict)
    receipts: OrderedDict[tuple[object, ...], OverlayApplicationReceipt] = field(
        default_factory=OrderedDict
    )
    receipts_by_id: dict[str, OverlayApplicationReceipt] = field(default_factory=dict)
    entry_ordering: dict[EntryKey, tuple[str, int, int]] = field(default_factory=dict)
    sequence_namespaces: dict[EntryKey, tuple[int, int]] = field(default_factory=dict)
    retired_turn_frontiers: dict[tuple[str, int], int] = field(default_factory=dict)

    def reset(self) -> None:
        self.terminal_registry.clear()
        self.scene_terminal_keys.clear()
        self.receipts.clear()
        self.receipts_by_id.clear()
        self.entry_ordering.clear()
        self.sequence_namespaces.clear()
        self.retired_turn_frontiers.clear()

    def receipt(self, publication_id: str) -> OverlayApplicationReceipt | None:
        return self.receipts_by_id.get(publication_id)

    def retained_receipt(self, event: OverlayEventUnion) -> OverlayApplicationReceipt | None:
        return self.receipts.get(self.receipt_key(event))

    def remember_receipt(
        self,
        event: OverlayEventUnion,
        receipt: OverlayApplicationReceipt,
    ) -> None:
        key = self.receipt_key(event)
        self.receipts.pop(key, None)
        self.receipts[key] = receipt
        self.receipts_by_id[receipt.publication_id] = receipt
        while len(self.receipts) > PRESENTER_RECEIPT_LIMIT:
            _, evicted = self.receipts.popitem(last=False)
            if self.receipts_by_id.get(evicted.publication_id) is evicted:
                self.receipts_by_id.pop(evicted.publication_id, None)

    def normalize_sequence(
        self,
        event: OverlayEventUnion,
        entries: Mapping[EntryKey, OverlayLogicalTurnEntry],
    ) -> OverlayEventUnion:
        if (
            event.sequence_namespace == 0
            or event.channel not in {"self", "peer"}
            or event.utterance_id is None
        ):
            return event
        key = (event.channel, event.utterance_id)
        namespace = self.sequence_namespaces.get(key)
        if namespace is None or namespace[0] != event.sequence_namespace:
            entry = entries.get(key)
            offset = entry.last_updated_seq if entry is not None else 0
            namespace = (event.sequence_namespace, offset)
            self.sequence_namespaces[key] = namespace
        return replace(event, seq=namespace[1] + event.seq)

    def rejection_reason(
        self,
        event: OverlayEventUnion,
        entries: Mapping[EntryKey, OverlayLogicalTurnEntry],
        *,
        closed: bool,
    ) -> str | None:
        if closed:
            return "closed"
        if self.event_payload_bytes(event) > PRESENTER_EVENT_BYTE_LIMIT:
            return "presenter_payload_exhausted"
        key = (
            (event.channel, event.utterance_id)
            if event.channel in {"self", "peer"} and event.utterance_id is not None
            else None
        )
        entry = entries.get(key) if key is not None else None
        if (
            key is not None
            and entry is None
            and event.turn_generation is not None
            and event.turn_order is not None
        ):
            scope = event.turn_kind or str(event.channel)
            if event.turn_order <= self.retired_turn_frontiers.get(
                (scope, event.turn_generation), -1
            ):
                return "stale"
        if key is not None and entry is None and self.terminal_reason(key) is not None:
            return "stale"
        if entry is not None and event.seq <= entry.last_updated_seq:
            return "stale"
        creates_entry = not isinstance(event, (SelfActiveClear, UtteranceClosed))
        if (
            key is not None
            and entry is None
            and creates_entry
            and len(entries) >= PRESENTER_ENTRY_LIMIT
        ):
            return "presenter_overload"
        current_payload_bytes = sum(
            self.entry_payload_bytes(candidate) for candidate in entries.values()
        )
        projected = self.projected_entry_payload_bytes(entry, event)
        previous = self.entry_payload_bytes(entry) if entry is not None else 0
        if current_payload_bytes - previous + projected > PRESENTER_AGGREGATE_BYTE_LIMIT:
            return "presenter_payload_exhausted"
        return None

    def record_entry_ordering(
        self,
        event: OverlayEventUnion,
        entries: Mapping[EntryKey, OverlayLogicalTurnEntry],
    ) -> None:
        if (
            event.channel not in {"self", "peer"}
            or event.utterance_id is None
            or event.turn_generation is None
            or event.turn_order is None
        ):
            return
        key = (event.channel, event.utterance_id)
        if key in entries:
            self.entry_ordering[key] = (
                event.turn_kind or str(event.channel),
                event.turn_generation,
                event.turn_order,
            )

    def retire_entry(self, key: EntryKey) -> None:
        ordering = self.entry_ordering.pop(key, None)
        self.sequence_namespaces.pop(key, None)
        if ordering is None:
            return
        scope, generation, order = ordering
        frontier_key = (scope, generation)
        self.retired_turn_frontiers[frontier_key] = max(
            order, self.retired_turn_frontiers.get(frontier_key, -1)
        )

    def terminal_reason(self, key: EntryKey) -> str | None:
        reason = self.scene_terminal_keys.get(key)
        if reason is not None:
            return reason
        if key in self.terminal_registry:
            return ""
        return None

    def remember_scene_terminal(self, key: EntryKey, reason: str) -> None:
        self.scene_terminal_keys.pop(key, None)
        self.scene_terminal_keys[key] = reason
        while len(self.scene_terminal_keys) > CLOSED_TOMBSTONE_LIMIT:
            self.scene_terminal_keys.popitem(last=False)

    def remember_tombstone(self, key: EntryKey, closed_seq: int) -> None:
        self.terminal_registry.pop(key, None)
        self.terminal_registry[key] = closed_seq
        while len(self.terminal_registry) > CLOSED_TOMBSTONE_LIMIT:
            self.terminal_registry.popitem(last=False)

    def is_tombstoned(self, key: EntryKey) -> bool:
        return key in self.scene_terminal_keys or key in self.terminal_registry

    @staticmethod
    def receipt_key(event: OverlayEventUnion) -> tuple[object, ...]:
        if event.turn_generation is not None:
            return (event.turn_kind or event.channel, event.turn_generation, event.event_id)
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
    def entry_payload_bytes(entry: OverlayLogicalTurnEntry | None) -> int:
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
    def event_payload_bytes(event: OverlayEventUnion) -> int:
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

    @classmethod
    def projected_entry_payload_bytes(
        cls,
        entry: OverlayLogicalTurnEntry | None,
        event: OverlayEventUnion,
    ) -> int:
        if entry is None:
            return cls.event_payload_bytes(event)
        current = cls.entry_payload_bytes(entry)
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
