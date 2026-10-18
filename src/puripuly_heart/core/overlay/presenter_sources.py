from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .sink import OverlayEventUnion
from .state import OverlayEntryKey, OverlayLogicalTurnEntry

if TYPE_CHECKING:
    from puripuly_heart.core.orchestrator.translation_turn import SourceChild

SOURCE_UNIT_LIMIT = 64
SOURCE_ATTEMPT_LIMIT = 4096


@dataclass(slots=True)
class SourceCaptionUnit:
    source: SourceChild
    content: OverlayLogicalTurnEntry
    card_key: OverlayEntryKey | None = None
    publication_ordering: tuple[str, int, int, int] | None = None
    applied: bool = False
    readable: bool = False


@dataclass(slots=True)
class PresenterSourceLedger:
    units: dict[OverlayEntryKey, SourceCaptionUnit] = field(default_factory=dict)
    attempts: OrderedDict[OverlayEntryKey, OverlayEntryKey | None] = field(
        default_factory=OrderedDict
    )
    attempt_sequences: OrderedDict[OverlayEntryKey, tuple[int, int]] = field(
        default_factory=OrderedDict
    )
    card_sources: dict[OverlayEntryKey, list[OverlayEntryKey]] = field(default_factory=dict)
    retired_orders: dict[str, tuple[int, int, int]] = field(default_factory=dict)

    def key_for(self, event: OverlayEventUnion) -> OverlayEntryKey | None:
        if event.source_child is not None:
            return (event.source_child.channel, event.source_child.source_id)
        if event.channel is not None and event.utterance_id is not None:
            return self.attempts.get((event.channel, event.utterance_id))
        return None

    def rejection_reason(self, event: OverlayEventUnion) -> str | None:
        key = self.key_for(event)
        source = event.source_child
        if key is None:
            return None
        if source is not None and source.channel != event.channel:
            return "stale"
        if event.channel is not None and event.utterance_id is not None:
            attempt = (event.channel, event.utterance_id)
            if attempt in self.attempts and self.attempts[attempt] is None:
                if source is not None:
                    self.retire_source(source)
                return "stale"
        unit = self.units.get(key)
        if unit is None:
            if source is None or source.order <= self.retired_orders.get(key[0], (-1, -1, -1)):
                return "stale"
            if len(self.units) >= SOURCE_UNIT_LIMIT:
                return "presenter_overload"
        elif source is not None and source != unit.source:
            return "stale"
        if event.channel is not None and event.utterance_id is not None:
            attempt = (event.channel, event.utterance_id)
            bound = self.attempts.get(attempt)
            if bound is not None and bound != key:
                return "stale"
            sequence = self.attempt_sequences.get(attempt)
            if (
                sequence is not None
                and sequence[0] == event.sequence_namespace
                and event.seq <= sequence[1]
            ):
                return "stale"
        return None

    def remember_attempt(self, event: OverlayEventUnion, key: OverlayEntryKey) -> None:
        if event.channel is None or event.utterance_id is None:
            return
        attempt = (event.channel, event.utterance_id)
        self.attempts[attempt] = key
        self.attempts.move_to_end(attempt)
        self.attempt_sequences[attempt] = (event.sequence_namespace, event.seq)
        self.attempt_sequences.move_to_end(attempt)
        self._trim_attempts()

    def retire_attempt(self, key: OverlayEntryKey) -> None:
        self.attempts[key] = None
        self.attempts.move_to_end(key)
        self.attempt_sequences.pop(key, None)
        self._trim_attempts()

    def _trim_attempts(self) -> None:
        while len(self.attempts) > SOURCE_ATTEMPT_LIMIT:
            retired, _ = self.attempts.popitem(last=False)
            self.attempt_sequences.pop(retired, None)

    def retire_source(self, source: SourceChild) -> None:
        self.retired_orders[source.channel] = max(
            source.order, self.retired_orders.get(source.channel, (-1, -1, -1))
        )

    def retire_card(self, key: OverlayEntryKey) -> None:
        for source_key in self.card_sources.pop(key, []):
            unit = self.units.pop(source_key, None)
            if unit is not None:
                self.retire_source(unit.source)

    def reset(self) -> None:
        self.units.clear()
        self.attempts.clear()
        self.attempt_sequences.clear()
        self.card_sources.clear()
        self.retired_orders.clear()
