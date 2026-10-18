from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import replace
from io import BytesIO
from typing import Literal
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.stt.backend import (
    STTAudioBoundary,
    STTNativeProvenance,
    STTProviderAudioBoundary,
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnEvent,
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
)
from puripuly_heart.core.stt.scoped_event_buffer import (
    STTProviderEventBuffer,
    STTProviderEventBufferClosed,
)
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity


class ResourceConsumer:
    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self.live: dict[str, BytesIO] = {}
        self.owners: dict[int, tuple[STTProviderTurnEvent, tuple[str, ...]]] = {}
        self.discarded: list[STTProviderTurnEvent] = []
        self.retained: list[tuple[STTProviderTurnEvent, ...]] = []
        self.consumed: list[tuple[STTProviderTurnEvent, tuple[bytes, ...]]] = []

    def bind(self, event: STTProviderTurnEvent, *tokens: str) -> None:
        for token in tokens:
            if token not in self.live:
                assert len(self.live) < self.capacity
                self.live[token] = BytesIO(token.encode())
        self.owners[id(event)] = (event, tokens)

    def tokens(self, event: STTProviderTurnEvent) -> tuple[str, ...]:
        owner = self.owners.get(id(event))
        return owner[1] if owner is not None else ()

    def release(
        self, event: STTProviderTurnEvent, retained_events: tuple[STTProviderTurnEvent, ...]
    ) -> None:
        retained_tokens = {token for retained in retained_events for token in self.tokens(retained)}
        for token in self.tokens(event):
            if token not in retained_tokens:
                resource = self.live.pop(token, None)
                if resource is not None:
                    resource.close()

    def discard(
        self, event: STTProviderTurnEvent, retained_events: tuple[STTProviderTurnEvent, ...]
    ) -> None:
        self.discarded.append(event)
        self.retained.append(retained_events)
        self.release(event, retained_events)

    def consume(
        self, event: STTProviderTurnEvent, retained_events: tuple[STTProviderTurnEvent, ...] = ()
    ) -> None:
        payloads = tuple(self.live[token].getvalue() for token in self.tokens(event))
        assert payloads == tuple(token.encode() for token in self.tokens(event))
        self.consumed.append((event, payloads))
        self.release(event, retained_events)

    def close(self) -> None:
        for resource in self.live.values():
            resource.close()
        self.live.clear()


@pytest.fixture
def resources() -> Iterator[ResourceConsumer]:
    consumer = ResourceConsumer()
    try:
        yield consumer
    finally:
        consumer.close()


def identity(epoch: str = "epoch", turn: str = "turn") -> STTProviderTurnIdentity:
    return STTProviderTurnIdentity(AudioSegmentIdentity(1, 1, uuid4(), 1), epoch, turn)


def update(
    turn: STTProviderTurnIdentity,
    text: str,
    *,
    stability: Literal["provisional", "stable"] = "stable",
    assembly: Literal["append", "replace"] = "append",
) -> STTProviderTurnUpdate:
    return STTProviderTurnUpdate(turn, 1, stability, assembly, text)


def terminal(turn: STTProviderTurnIdentity, text: str = "final") -> STTProviderTurnTerminal:
    return STTProviderTurnTerminal(turn, "final", text, text_authority="authoritative")


def event_of_kind(kind: str, turn: STTProviderTurnIdentity) -> STTProviderTurnEvent:
    if kind == "update":
        return update(turn, "update")
    if kind == "terminal":
        return terminal(turn)
    if kind == "boundary":
        return STTProviderAudioBoundary(turn, STTAudioBoundary("boundary", 1, 1, 1.0, "natural"))
    if kind == "input":
        return STTProviderInputTerminal(turn, "submitted", "peer")
    if kind == "epoch":
        return STTProviderEpochEnded(turn.provider_epoch_id, True, "completed")
    stream = RecognitionStreamIdentity("peer", 1, 1, turn.provider_epoch_id, ())
    unit = STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), "unit")
    if kind == "unit":
        return unit
    assert kind == "unit_terminal"
    return STTRecognitionUnitTerminal(unit, "final")


def drain(
    buffer: STTProviderEventBuffer, resources: ResourceConsumer
) -> list[STTProviderTurnEvent]:
    async def collect() -> list[STTProviderTurnEvent]:
        return [event async for event in buffer.events()]

    buffer.close()
    delivered = asyncio.run(collect())
    for index, event in enumerate(delivered):
        resources.consume(event, tuple(delivered[index + 1 :]))
    assert not resources.live
    assert buffer.depth == 0
    with pytest.raises(STTProviderEventBufferClosed):
        asyncio.run(buffer.get())
    return delivered


def test_default_callback_preserves_coalescing_order_and_healthy_close() -> None:
    turn = identity()
    first = update(turn, "old", stability="provisional", assembly="replace")
    latest = replace(first, sequence=2, text="latest")
    final = terminal(turn)
    buffer = STTProviderEventBuffer(max_events=2)
    assert buffer.put(first)
    assert buffer.put(latest)
    assert buffer.put(final)
    assert buffer.max_events == 2
    assert drain(buffer, ResourceConsumer()) == [latest, final]


def test_healthy_close_preserves_queued_resources_until_delivery(
    resources: ResourceConsumer,
) -> None:
    buffer = STTProviderEventBuffer(max_events=2, on_discard=resources.discard)
    progress = update(identity(), "recognized")
    final = terminal(progress.identity)
    resources.bind(progress, "progress")
    resources.bind(final, "final")
    handles = tuple(resources.live.values())
    assert buffer.put(progress)
    assert buffer.put(final)
    buffer.close()
    assert all(not resource.closed for resource in handles)
    assert resources.discarded == []
    assert drain(buffer, resources) == [progress, final]
    assert all(resource.closed for resource in handles)
    assert [payload for _, payload in resources.consumed] == [(b"progress",), (b"final",)]


def test_superseded_provisional_protects_incoming_shared_resource_and_reuses_capacity(
    resources: ResourceConsumer,
) -> None:
    resources.capacity = 3
    turn = identity()
    old = update(turn, "old", stability="provisional", assembly="replace")
    latest = replace(old, sequence=2, text="latest")
    resources.bind(old, "old-only", "shared")
    old_handle = resources.live["old-only"]
    shared_handle = resources.live["shared"]
    resources.bind(latest, "shared", "latest-only")
    buffer = STTProviderEventBuffer(max_events=1, on_discard=resources.discard)
    assert buffer.put(old)
    assert buffer.put(latest)
    assert old_handle.closed
    assert not shared_handle.closed
    assert resources.discarded == [old]
    assert resources.retained == [(latest,)]
    delivered = asyncio.run(buffer.get())
    assert delivered is latest
    resources.consume(delivered)
    assert shared_handle.closed
    next_event = terminal(identity("next"))
    resources.bind(next_event, "reused-1", "reused-2", "reused-3")
    assert buffer.put(next_event)
    assert drain(buffer, resources) == [next_event]
    assert [event for event, _ in resources.consumed] == [latest, next_event]


@pytest.mark.parametrize("shared_owner", ["incoming", "queued"])
def test_room_making_eviction_protects_final_queued_owners(
    resources: ResourceConsumer, shared_owner: str
) -> None:
    provisional = update(identity("old"), "old", stability="provisional", assembly="replace")
    queued = terminal(identity("queued"), "queued")
    incoming = terminal(identity("incoming"), "incoming")
    resources.bind(provisional, "evicted-only", "shared")
    resources.bind(queued, "shared" if shared_owner == "queued" else "queued-only")
    resources.bind(incoming, "shared" if shared_owner == "incoming" else "incoming-only")
    evicted = resources.live["evicted-only"]
    shared = resources.live["shared"]
    buffer = STTProviderEventBuffer(max_events=2, on_discard=resources.discard)
    assert buffer.put(provisional)
    assert buffer.put(queued)
    assert buffer.put(incoming)
    assert buffer.depth == 2
    assert evicted.closed
    assert not shared.closed
    assert resources.discarded == [provisional]
    assert resources.retained == [(queued, incoming)]
    assert drain(buffer, resources) == [queued, incoming]
    assert shared.closed


@pytest.mark.parametrize("kind", ["provisional_replace", "provisional_append", "boundary"])
def test_nonretiring_rejection_preserves_healthy_shared_owner(
    resources: ResourceConsumer, kind: str
) -> None:
    turn = identity()
    queued = terminal(turn, "queued")
    if kind == "boundary":
        rejected = event_of_kind(kind, turn)
    else:
        rejected = update(
            turn,
            "rejected",
            stability="provisional",
            assembly="replace" if kind == "provisional_replace" else "append",
        )
    resources.bind(queued, "shared")
    resources.bind(rejected, "shared", "rejected-only")
    rejected_handle = resources.live["rejected-only"]
    buffer = STTProviderEventBuffer(max_events=1, on_discard=resources.discard)
    assert buffer.put(queued)
    assert not buffer.put(rejected)
    assert rejected_handle.closed
    assert set(resources.live) == {"shared"}
    assert resources.discarded == [rejected]
    assert resources.retained == [(queued,)]
    delivered = asyncio.run(buffer.get())
    assert delivered is queued
    resources.consume(delivered)
    later = terminal(turn, "valid progress")
    resources.bind(later, "later")
    assert buffer.put(later)
    assert drain(buffer, resources) == [later]


@pytest.mark.parametrize("kind", ["update", "terminal"])
def test_scoped_overflow_releases_removed_updates_boundaries_and_rejected_resources(
    resources: ResourceConsumer, kind: str
) -> None:
    turn = identity()
    removed_update = update(turn, "removed")
    removed_boundary = event_of_kind("boundary", turn)
    retained_terminal = terminal(turn, "already queued")
    other = update(identity("other"), "other progress")
    incoming = event_of_kind(kind, turn)
    resources.bind(removed_update, "update")
    resources.bind(removed_boundary, "boundary")
    resources.bind(retained_terminal, "shared")
    resources.bind(other, "other")
    resources.bind(incoming, "shared", "rejected")
    removed_handles = [resources.live[token] for token in ("update", "boundary", "rejected")]
    buffer = STTProviderEventBuffer(max_events=4, on_discard=resources.discard)
    for event in (removed_update, removed_boundary, retained_terminal, other):
        assert buffer.put(event)
    assert not buffer.put(incoming)
    failure = STTProviderTurnTerminal(
        turn,
        "failed",
        failure_reason="provider_event_buffer_overflow",
        epoch_disposition="retire",
    )
    assert all(handle.closed for handle in removed_handles)
    assert set(resources.live) == {"shared", "other"}
    assert resources.discarded == [removed_update, removed_boundary, incoming]
    assert resources.retained == [(retained_terminal, other, failure)] * 3
    progress = terminal(identity("next"), "new epoch progress")
    resources.bind(progress, "next")
    assert buffer.put(progress)
    assert buffer.max_events == 4
    assert drain(buffer, resources) == [retained_terminal, other, failure, progress]


@pytest.mark.parametrize(
    "kind", ["update", "terminal", "boundary", "input", "unit", "unit_terminal", "epoch"]
)
def test_retired_epoch_rejects_every_event_kind_and_allows_other_epoch_progress(
    resources: ResourceConsumer, kind: str
) -> None:
    turn = identity()
    queued = update(turn, "queued")
    overflow = terminal(turn)
    late = event_of_kind(kind, turn)
    resources.bind(queued, "queued")
    resources.bind(overflow, "overflow")
    resources.bind(late, "late")
    handles = tuple(resources.live.values())
    buffer = STTProviderEventBuffer(max_events=1, on_discard=resources.discard)
    assert buffer.put(queued)
    assert not buffer.put(overflow)
    assert not buffer.put(late)
    assert all(handle.closed for handle in handles)
    assert resources.discarded == [queued, overflow, late]
    failure = asyncio.run(buffer.get())
    assert failure == STTProviderTurnTerminal(
        turn,
        "failed",
        failure_reason="provider_event_buffer_overflow",
        epoch_disposition="retire",
    )
    progress = terminal(identity("other"), "valid progress")
    resources.bind(progress, "progress")
    assert buffer.put(progress)
    assert drain(buffer, resources) == [progress]


@pytest.mark.parametrize("kind", ["input", "unit", "unit_terminal"])
def test_independent_count_overflow_preserves_queued_delivery_and_retires_epoch(
    resources: ResourceConsumer, kind: str
) -> None:
    turn = identity()
    queued = terminal(turn, "queued")
    rejected = event_of_kind(kind, turn)
    resources.bind(queued, "shared")
    resources.bind(rejected, "shared", "rejected")
    rejected_handle = resources.live["rejected"]
    buffer = STTProviderEventBuffer(max_events=1, on_discard=resources.discard)
    assert buffer.put(queued)
    assert not buffer.put(rejected)
    retirement = STTProviderEpochEnded("epoch", False, "provider_event_buffer_overflow")
    assert rejected_handle.closed
    assert set(resources.live) == {"shared"}
    assert resources.discarded == [rejected]
    assert resources.retained == [(queued, retirement)]
    assert buffer.depth == 2
    assert drain(buffer, resources) == [queued, retirement]


@pytest.mark.parametrize("kind", ["unit", "unit_terminal", "update", "terminal"])
def test_byte_overflow_releases_evictions_and_rejection_but_preserves_other_epoch(
    resources: ResourceConsumer, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    turn = identity()
    removed_update = update(turn, "removed")
    removed_boundary = event_of_kind("boundary", turn)
    healthy = terminal(identity("healthy"), "healthy")
    rejected = event_of_kind(kind, turn)
    if isinstance(rejected, STTRecognitionUnit):
        rejected = replace(rejected, text="123456789")
    elif isinstance(rejected, STTRecognitionUnitTerminal):
        rejected = replace(rejected, unit=replace(rejected.unit, text="123456789"))
    else:
        provenance = STTNativeProvenance(unresolved_source_text="123456789")
        rejected = replace(
            rejected,
            provenance=provenance if isinstance(rejected, STTProviderTurnUpdate) else (provenance,),
        )
    resources.bind(removed_update, "removed-update")
    resources.bind(removed_boundary, "removed-boundary")
    resources.bind(healthy, "shared")
    resources.bind(rejected, "shared", "rejected-only")
    removed = [
        resources.live[token] for token in ("removed-update", "removed-boundary", "rejected-only")
    ]
    buffer = STTProviderEventBuffer(max_events=4, on_discard=resources.discard)
    monkeypatch.setattr(buffer, "MAX_RECOGNITION_BYTES", 8)
    assert buffer.put(removed_update)
    assert buffer.put(removed_boundary)
    assert buffer.put(healthy)
    assert not buffer.put(rejected)
    retirement = STTProviderEpochEnded("epoch", False, "provider_event_buffer_overflow")
    assert all(handle.closed for handle in removed)
    assert set(resources.live) == {"shared"}
    assert resources.discarded == [removed_update, removed_boundary, rejected]
    assert resources.retained == [(healthy, retirement)] * 3
    assert drain(buffer, resources) == [healthy, retirement]


def test_recognition_byte_budget_is_recovered_by_consumption(
    resources: ResourceConsumer, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = event_of_kind("unit", identity())
    assert isinstance(first, STTRecognitionUnit)
    first = replace(first, text="12345678")
    second = replace(first, identity=replace(first.identity, unit_id=uuid4(), receipt_sequence=2))
    rejected = replace(
        first, identity=replace(first.identity, unit_id=uuid4(), receipt_sequence=3), text="x"
    )
    resources.bind(first, "first")
    resources.bind(second, "second")
    resources.bind(rejected, "rejected")
    rejected_handle = resources.live["rejected"]
    buffer = STTProviderEventBuffer(on_discard=resources.discard)
    monkeypatch.setattr(buffer, "MAX_RECOGNITION_BYTES", 8)
    assert buffer.put(first)
    delivered = asyncio.run(buffer.get())
    assert delivered is first
    resources.consume(delivered)
    assert buffer.put(second)
    assert not buffer.put(rejected)
    assert rejected_handle.closed
    assert resources.discarded == [rejected]
    assert drain(buffer, resources) == [
        second,
        STTProviderEpochEnded("epoch", False, "provider_event_buffer_overflow"),
    ]


def test_retirement_admission_and_refusal_only_discard_actual_removals(
    resources: ResourceConsumer,
) -> None:
    queued = terminal(identity("queued"), "queued")
    overflow = event_of_kind("input", identity("overflow"))
    refused = STTProviderEpochEnded("refused", True, "completed")
    resources.bind(queued, "queued")
    resources.bind(overflow, "overflow")
    resources.bind(refused, "refused")
    overflow_handle = resources.live["overflow"]
    refused_handle = resources.live["refused"]
    buffer = STTProviderEventBuffer(max_events=1, on_discard=resources.discard)
    assert buffer.put(queued)
    assert not buffer.put(overflow)
    assert not buffer.put(refused)
    assert overflow_handle.closed and refused_handle.closed
    assert set(resources.live) == {"queued"}
    assert resources.discarded == [overflow, refused]
    assert drain(buffer, resources) == [
        queued,
        STTProviderEpochEnded("overflow", False, "provider_event_buffer_overflow"),
    ]


def test_accepted_retirement_disposes_epoch_updates_and_boundaries(
    resources: ResourceConsumer,
) -> None:
    turn = identity()
    removed_update = update(turn, "removed")
    removed_boundary = event_of_kind("boundary", turn)
    healthy = terminal(identity("other"), "healthy")
    retirement = STTProviderEpochEnded("epoch", True, "completed")
    for token, event in (
        ("update", removed_update),
        ("boundary", removed_boundary),
        ("healthy", healthy),
    ):
        resources.bind(event, token)
    removed = [resources.live["update"], resources.live["boundary"]]
    buffer = STTProviderEventBuffer(max_events=3, on_discard=resources.discard)
    for event in (removed_update, removed_boundary, healthy):
        assert buffer.put(event)
    assert buffer.put(retirement)
    assert all(handle.closed for handle in removed)
    assert resources.discarded == [removed_update, removed_boundary]
    assert resources.retained == [(healthy, retirement)] * 2
    assert drain(buffer, resources) == [healthy, retirement]


def test_closed_put_releases_rejection_without_releasing_healthy_queued_owner(
    resources: ResourceConsumer,
) -> None:
    queued = terminal(identity(), "queued")
    rejected = terminal(queued.identity, "closed rejection")
    resources.bind(queued, "shared")
    resources.bind(rejected, "shared", "rejected")
    rejected_handle = resources.live["rejected"]
    buffer = STTProviderEventBuffer(on_discard=resources.discard)
    assert buffer.put(queued)
    buffer.close()
    with pytest.raises(STTProviderEventBufferClosed):
        buffer.put(rejected)
    assert rejected_handle.closed
    assert set(resources.live) == {"shared"}
    assert resources.discarded == [rejected]
    assert resources.retained == [(queued,)]
    assert drain(buffer, resources) == [queued]


def test_disposal_exception_propagates_after_final_admission_state(
    resources: ResourceConsumer,
) -> None:
    turn = identity()
    old = update(turn, "old", stability="provisional", assembly="replace")
    incoming = replace(old, sequence=2, text="accepted")
    resources.bind(old, "old", "shared")
    resources.bind(incoming, "shared")
    old_handle = resources.live["old"]

    def discard(event: STTProviderTurnEvent, retained: tuple[STTProviderTurnEvent, ...]) -> None:
        resources.discard(event, retained)
        raise RuntimeError("owner disposal failed")

    buffer = STTProviderEventBuffer(on_discard=discard)
    assert buffer.put(old)
    with pytest.raises(RuntimeError, match="owner disposal failed"):
        buffer.put(incoming)
    assert old_handle.closed
    assert set(resources.live) == {"shared"}
    assert resources.retained == [(incoming,)]
    assert drain(buffer, resources) == [incoming]


def test_full_buffer_rejects_boundary_without_evicting_deliverable_provisional(
    resources: ResourceConsumer,
) -> None:
    turn = identity()
    provisional = update(turn, "valid partial", stability="provisional", assembly="replace")
    rejected = event_of_kind("boundary", turn)
    resources.bind(provisional, "shared", "partial")
    resources.bind(rejected, "shared", "rejected")
    rejected_handle = resources.live["rejected"]
    buffer = STTProviderEventBuffer(max_events=1, on_discard=resources.discard)
    assert buffer.put(provisional)
    assert not buffer.put(rejected)
    assert rejected_handle.closed
    assert set(resources.live) == {"shared", "partial"}
    assert resources.discarded == [rejected]
    assert resources.retained == [(provisional,)]
    delivered = asyncio.run(buffer.get())
    assert delivered is provisional
    resources.consume(delivered)
    progress = terminal(turn, "valid final")
    resources.bind(progress, "progress")
    assert buffer.put(progress)
    assert drain(buffer, resources) == [progress]


def test_retired_epoch_rejection_protects_shared_queued_resource(
    resources: ResourceConsumer,
) -> None:
    turn = identity()
    queued = terminal(turn, "retained final")
    overflow = event_of_kind("input", turn)
    late = terminal(turn, "late")
    resources.bind(queued, "shared")
    resources.bind(overflow, "overflow")
    resources.bind(late, "shared", "late-only")
    overflow_handle = resources.live["overflow"]
    late_handle = resources.live["late-only"]
    buffer = STTProviderEventBuffer(max_events=1, on_discard=resources.discard)
    assert buffer.put(queued)
    assert not buffer.put(overflow)
    assert not buffer.put(late)
    retirement = STTProviderEpochEnded("epoch", False, "provider_event_buffer_overflow")
    assert overflow_handle.closed and late_handle.closed
    assert set(resources.live) == {"shared"}
    assert resources.discarded == [overflow, late]
    assert resources.retained == [(queued, retirement)] * 2
    assert drain(buffer, resources) == [queued, retirement]
