from __future__ import annotations

import asyncio
import contextlib
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from puripuly_heart.core.overlay.sink import (
    OverlayEventUnion,
    SelfActiveClear,
    SelfActiveUpdate,
    SelfTranscriptFinal,
)
from puripuly_heart.domain.models import ChannelId

OutputDestination = Literal["overlay", "ui", "chatbox"]

OUTPUT_BATCH_MAX_UNSENT = 8
OUTPUT_BATCH_MAX_BYTES = 1024 * 1024
OUTPUT_SCOPE_MAX_BYTES = 9 * 1024 * 1024
COMPLETED_BATCH_LIMIT = 4096


@dataclass(slots=True)
class DestinationBatch:
    scope: str
    destination: OutputDestination
    parent_id: str
    channel: ChannelId
    managed_parent: bool
    target_count: int
    turn_generation: int | None
    turn_order: int | None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    reserved_bytes: int = 0
    retained_payloads: dict[int, str] = field(default_factory=dict)
    seen_targets: set[int] = field(default_factory=set)
    expected_targets: frozenset[int] = frozenset()
    completed_targets: set[int] = field(default_factory=set)
    active: bool = False
    disposition: str | None = None


@dataclass(slots=True)
class DestinationBatchAdmission:
    """Destination-neutral parent batch policy subordinate to OutputRuntime.

    The component owns reservation, queue activation, release, and retirement state.
    OutputRuntime owns route selection, destination objects, and delivery tasks.
    Callers serialize compound routing transitions with ``lock``.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    batches: dict[tuple[str, str], DestinationBatch] = field(default_factory=dict)
    waiting: dict[str, deque[tuple[str, str]]] = field(default_factory=dict)
    active: dict[str, tuple[str, str]] = field(default_factory=dict)
    reserved_bytes: dict[str, int] = field(default_factory=dict)
    terminal_batches: OrderedDict[tuple[str, str], str] = field(default_factory=OrderedDict)
    retired_order_ranges: dict[tuple[str, int], list[tuple[int, int]]] = field(default_factory=dict)
    retired_turn_generations: dict[ChannelId, int] = field(
        default_factory=lambda: {"self": -1, "peer": -1}
    )

    @staticmethod
    def scope_for(origin: str, destination: OutputDestination) -> str:
        return origin if destination == "overlay" else f"{origin}:{destination}"

    @staticmethod
    def normalized_payloads(values: Iterable[str]) -> dict[int, str]:
        return {id(value): value for value in values if value}

    @classmethod
    def payload_bytes(cls, values: Iterable[str]) -> int:
        return sum(len(value.encode("utf-8")) for value in values)

    def snapshot(self) -> dict[str, object]:
        return {
            "active": len(self.active),
            "unsent": sum(len(waiting) for waiting in self.waiting.values()),
            "batches": len(self.batches),
            "reserved_bytes": sum(self.reserved_bytes.values()),
            "scopes": {
                scope: {
                    "active": int(scope in self.active),
                    "unsent": len(self.waiting.get(scope, ())),
                    "reserved_bytes": self.reserved_bytes.get(scope, 0),
                }
                for scope in set(self.active) | set(self.waiting)
            },
        }

    def admit_parent(
        self,
        *,
        parent_id: str,
        channel: ChannelId,
        origin: str,
        turn_generation: int,
        turn_order: int,
        payloads: dict[int, str],
        destination_targets: Mapping[OutputDestination, frozenset[int]],
    ) -> frozenset[OutputDestination]:
        retained_payload_bytes = self.payload_bytes(payloads.values())
        if retained_payload_bytes > OUTPUT_BATCH_MAX_BYTES:
            return frozenset()
        admitted: set[OutputDestination] = set()
        for destination, targets in destination_targets.items():
            if not targets:
                continue
            scope = self.scope_for(origin, destination)
            key = (scope, parent_id)
            if key in self.batches or key in self.terminal_batches:
                continue
            waiting = self.waiting.setdefault(scope, deque())
            pressured = self._scope_is_pressured(scope, waiting, retained_payload_bytes)
            if pressured and origin == "manual":
                self.remember_terminal(key, "output_overload", (turn_generation, turn_order))
                self._prune_scope(scope)
                continue
            while pressured and waiting:
                evicted_key = waiting[0]
                evicted = self.batches.get(evicted_key)
                if evicted is None:
                    waiting.popleft()
                else:
                    self.release(evicted, "output_overload")
                waiting = self.waiting.setdefault(scope, deque())
                pressured = self._scope_is_pressured(scope, waiting, retained_payload_bytes)
            if pressured:
                self.remember_terminal(key, "output_overload", (turn_generation, turn_order))
                self._prune_scope(scope)
                continue
            batch = DestinationBatch(
                scope=scope,
                destination=destination,
                parent_id=parent_id,
                channel=channel,
                managed_parent=True,
                target_count=len(targets),
                turn_generation=turn_generation,
                turn_order=turn_order,
                reserved_bytes=retained_payload_bytes,
                retained_payloads=dict(payloads),
                expected_targets=targets,
            )
            self._insert(batch)
            admitted.add(destination)
        return frozenset(admitted)

    def resize_parent(
        self,
        *,
        parent_id: str,
        origin: str,
        payloads: dict[int, str],
        destinations: Iterable[OutputDestination],
    ) -> frozenset[OutputDestination]:
        admitted: set[OutputDestination] = set()
        for destination in destinations:
            scope = self.scope_for(origin, destination)
            batch = self.batches.get((scope, parent_id))
            if batch is None or batch.disposition is not None:
                continue
            next_payloads = dict(batch.retained_payloads)
            next_payloads.update(payloads)
            next_reserved_bytes = self.payload_bytes(next_payloads.values())
            additional = next_reserved_bytes - batch.reserved_bytes
            if next_reserved_bytes > OUTPUT_BATCH_MAX_BYTES or (
                self.reserved_bytes.get(scope, 0) + additional > OUTPUT_SCOPE_MAX_BYTES
            ):
                self.release(batch, "output_payload_exhausted")
                continue
            batch.retained_payloads = next_payloads
            batch.reserved_bytes = next_reserved_bytes
            self.reserved_bytes[scope] = self.reserved_bytes.get(scope, 0) + additional
            admitted.add(destination)
        return frozenset(admitted)

    def parent_batches(
        self,
        *,
        parent_id: str,
        origin: str,
        destinations: frozenset[OutputDestination] | None,
    ) -> tuple[DestinationBatch, ...]:
        return tuple(
            batch
            for (scope, candidate_parent_id), batch in self.batches.items()
            if candidate_parent_id == parent_id
            and (scope == origin or scope.startswith(f"{origin}:"))
            and (destinations is None or batch.destination in destinations)
        )

    def complete_target(
        self,
        *,
        parent_id: str,
        origin: str,
        destination_indexes: Mapping[OutputDestination, int],
        destinations: frozenset[OutputDestination] | None,
    ) -> None:
        for destination, target_index in destination_indexes.items():
            if destinations is not None and destination not in destinations:
                continue
            scope = self.scope_for(origin, destination)
            batch = self.batches.get((scope, parent_id))
            if batch is None or batch.disposition is not None:
                continue
            batch.completed_targets.add(target_index)
            expected = batch.expected_targets
            if (expected and expected.issubset(batch.completed_targets)) or (
                not expected and len(batch.completed_targets) >= batch.target_count
            ):
                self.release(batch, "applied")

    def register_overlay_event(
        self,
        event: OverlayEventUnion,
        *,
        payloads: dict[int, str],
    ) -> tuple[DestinationBatch | None, str | None]:
        scope = self.overlay_batch_scope(event)
        managed_fields = (
            event.parent_utterance_id,
            event.turn_kind,
            event.turn_generation,
            event.turn_order,
        )
        has_managed_identity = any(value is not None for value in managed_fields)
        if has_managed_identity:
            if any(value is None for value in managed_fields):
                return None, "unauthorized_parent"
            parent_id = str(event.parent_utterance_id)
            key = (scope, parent_id)
            terminal = self.terminal_batches.get(key)
            if terminal is not None:
                return None, terminal
            assert event.turn_generation is not None
            assert event.turn_order is not None
            if event.turn_generation <= self.retired_turn_generations.get(event.channel, -1):
                return None, "stale_retired"
            if self.order_is_retired(scope, event.turn_generation, event.turn_order):
                return None, "stale_retired"
            batch = self.batches.get(key)
            if (
                batch is None
                or not batch.managed_parent
                or batch.destination != "overlay"
                or batch.turn_generation != event.turn_generation
                or batch.turn_order != event.turn_order
                or event.target_index not in batch.expected_targets
            ):
                return None, "unauthorized_parent"
            batch.seen_targets.add(event.target_index)
            return batch, batch.disposition

        parent_id = event.event_id
        key = (scope, parent_id)
        terminal = self.terminal_batches.get(key)
        if terminal is not None:
            return None, terminal
        payload_size = self.payload_bytes(payloads.values())
        if payload_size > OUTPUT_BATCH_MAX_BYTES:
            return None, "output_payload_exhausted"
        waiting = self.waiting.setdefault(scope, deque())
        if scope == "preview:self" and scope in self.active:
            for superseded_key in tuple(waiting):
                superseded = self.batches.get(superseded_key)
                if superseded is not None:
                    self.release(superseded, "preview_superseded")
            waiting = self.waiting.setdefault(scope, deque())
        elif scope in self.active and len(waiting) >= OUTPUT_BATCH_MAX_UNSENT:
            evicted = self.batches.get(waiting[0])
            if evicted is not None:
                self.release(evicted, "output_overload")
            waiting = self.waiting.setdefault(scope, deque())
        if self.reserved_bytes.get(scope, 0) + payload_size > OUTPUT_SCOPE_MAX_BYTES:
            return None, "output_payload_exhausted"
        if event.channel is None:
            return None, "invalid_channel"
        batch = DestinationBatch(
            scope=scope,
            destination="overlay",
            parent_id=parent_id,
            channel=event.channel,
            managed_parent=False,
            target_count=1,
            turn_generation=None,
            turn_order=None,
            reserved_bytes=payload_size,
            retained_payloads=payloads,
            seen_targets={0},
            expected_targets=frozenset({0}),
        )
        self._insert(batch)
        return batch, None

    def cancel_waiting(self, batch: DestinationBatch) -> None:
        if batch.active or batch.disposition is not None:
            return
        waiting = self.waiting.get(batch.scope)
        key = (batch.scope, batch.parent_id)
        if waiting is not None:
            with contextlib.suppress(ValueError):
                waiting.remove(key)
        self.release(batch, "cancelled_local")

    def release(self, batch: DestinationBatch, disposition: str) -> None:
        key = (batch.scope, batch.parent_id)
        if self.batches.get(key) is not batch:
            return
        self.batches.pop(key, None)
        remaining = max(0, self.reserved_bytes.get(batch.scope, 0) - batch.reserved_bytes)
        if remaining:
            self.reserved_bytes[batch.scope] = remaining
        else:
            self.reserved_bytes.pop(batch.scope, None)
        batch.disposition = disposition
        batch.ready.set()
        waiting = self.waiting.get(batch.scope)
        if waiting is not None:
            with contextlib.suppress(ValueError):
                waiting.remove(key)
        if batch.managed_parent:
            self.remember_terminal(key, disposition, (batch.turn_generation, batch.turn_order))
        if self.active.get(batch.scope) != key:
            self._prune_scope(batch.scope)
            return
        self.active.pop(batch.scope, None)
        while waiting:
            successor_key = waiting.popleft()
            successor = self.batches.get(successor_key)
            if successor is None or successor.disposition is not None:
                continue
            self.active[batch.scope] = successor_key
            successor.active = True
            successor.ready.set()
            break
        self._prune_scope(batch.scope)

    def release_destination(self, destination: OutputDestination, disposition: str) -> None:
        for batch in tuple(self.batches.values()):
            if batch.destination == destination:
                self.release(batch, disposition)

    def release_all(self, disposition: str) -> None:
        for batch in tuple(self.batches.values()):
            self.release(batch, disposition)
        self.batches.clear()
        self.waiting.clear()
        self.active.clear()
        self.reserved_bytes.clear()

    def remember_terminal(
        self,
        key: tuple[str, str],
        disposition: str,
        ordering: tuple[int | None, int | None],
    ) -> None:
        self.terminal_batches.pop(key, None)
        self.terminal_batches[key] = disposition
        generation, order = ordering
        if generation is not None and order is not None:
            self.remember_retired_order(key[0], generation, order)
        while len(self.terminal_batches) > COMPLETED_BATCH_LIMIT:
            self.terminal_batches.popitem(last=False)

    def remember_retired_order(self, scope: str, generation: int, order: int) -> None:
        key = (scope, generation)
        ranges = self.retired_order_ranges.setdefault(key, [])
        start = end = order
        merged: list[tuple[int, int]] = []
        inserted = False
        for existing_start, existing_end in ranges:
            if existing_end + 1 < start:
                merged.append((existing_start, existing_end))
            elif end + 1 < existing_start:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((existing_start, existing_end))
            else:
                start = min(start, existing_start)
                end = max(end, existing_end)
        if not inserted:
            merged.append((start, end))
        self.retired_order_ranges[key] = merged

    def order_is_retired(self, scope: str, generation: int, order: int) -> bool:
        return any(
            start <= order <= end
            for start, end in self.retired_order_ranges.get((scope, generation), ())
        )

    def retire_turn_generation(self, channel: ChannelId, turn_generation: int) -> None:
        previous = self.retired_turn_generations.get(channel, -1)
        if turn_generation - 1 <= previous:
            return
        self.retired_turn_generations[channel] = turn_generation - 1
        for batch in tuple(self.batches.values()):
            if batch.channel == channel and (
                batch.turn_generation is None or batch.turn_generation < turn_generation
            ):
                self.release(batch, "generation_retired")
        for key in tuple(self.retired_order_ranges):
            if key[1] < turn_generation and (
                key[0] == channel
                or key[0] == "manual"
                or key[0].startswith(f"{channel}:")
                or key[0].startswith("manual:")
            ):
                self.retired_order_ranges.pop(key, None)

    @staticmethod
    def overlay_batch_scope(event: OverlayEventUnion) -> str:
        if (
            event.turn_kind is None
            and event.parent_utterance_id is None
            and isinstance(event, (SelfActiveUpdate, SelfActiveClear))
        ):
            return "preview:self"
        if (
            event.turn_kind is None
            and event.parent_utterance_id is None
            and isinstance(event, SelfTranscriptFinal)
        ):
            return "self:original"
        return event.turn_kind or str(event.channel)

    def _scope_is_pressured(
        self,
        scope: str,
        waiting: deque[tuple[str, str]],
        retained_payload_bytes: int,
    ) -> bool:
        return scope in self.active and (
            len(waiting) >= OUTPUT_BATCH_MAX_UNSENT
            or self.reserved_bytes.get(scope, 0) + retained_payload_bytes > OUTPUT_SCOPE_MAX_BYTES
        )

    def _insert(self, batch: DestinationBatch) -> None:
        key = (batch.scope, batch.parent_id)
        self.batches[key] = batch
        self.reserved_bytes[batch.scope] = (
            self.reserved_bytes.get(batch.scope, 0) + batch.reserved_bytes
        )
        if batch.scope not in self.active:
            self.active[batch.scope] = key
            batch.active = True
            batch.ready.set()
        else:
            self.waiting.setdefault(batch.scope, deque()).append(key)

    def _prune_scope(self, scope: str) -> None:
        waiting = self.waiting.get(scope)
        if waiting is not None and not waiting:
            self.waiting.pop(scope, None)
        if self.reserved_bytes.get(scope) == 0:
            self.reserved_bytes.pop(scope, None)
