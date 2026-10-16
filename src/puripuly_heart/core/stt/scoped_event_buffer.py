from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator

from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnEvent,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
)


class STTProviderEventBufferClosed(RuntimeError):
    __slots__ = ()


class STTProviderEventBuffer:
    MAX_EVENTS = 256
    MAX_RECOGNITION_BYTES = 1024 * 1024

    def __init__(self, *, max_events: int = MAX_EVENTS) -> None:
        if max_events <= 0 or max_events > self.MAX_EVENTS:
            raise ValueError(f"max_events must be between 1 and {self.MAX_EVENTS}")
        self._max_events = max_events
        self._events: deque[STTProviderTurnEvent] = deque()
        self._recognition_bytes = 0
        self._wake = asyncio.Event()
        self._closed = False
        self._retired_epochs: set[str] = set()
        self._retired_epoch_order: deque[str] = deque()

    @property
    def depth(self) -> int:
        return len(self._events)

    @property
    def max_events(self) -> int:
        return self._max_events

    def put(self, event: STTProviderTurnEvent) -> bool:
        if self._closed:
            raise STTProviderEventBufferClosed("provider event buffer is closed")
        epoch_id = _event_epoch_id(event)
        if epoch_id in self._retired_epochs:
            return False
        size = _recognition_bytes(event)
        if size + self._recognition_bytes > self.MAX_RECOGNITION_BYTES:
            self._fail_epoch(epoch_id)
            return False
        if _is_provisional_snapshot(event):
            self._remove_matching_provisional_snapshots(event)
        if len(self._events) >= self._max_events:
            self._discard_provisional_snapshots_until_room()
        if len(self._events) >= self._max_events:
            if isinstance(event, STTProviderTurnUpdate) and event.stability == "provisional":
                return False
            if isinstance(event, (STTProviderTurnUpdate, STTProviderTurnTerminal)):
                self._fail_scoped_turn(event)
                return False
            if isinstance(
                event, (STTProviderInputTerminal, STTRecognitionUnit, STTRecognitionUnitTerminal)
            ):
                self._fail_epoch(epoch_id)
                return False
            if isinstance(event, STTProviderEpochEnded):
                return self._append_retirement(event, event.provider_epoch_id)
        self._events.append(event)
        self._recognition_bytes += size
        self._wake.set()
        return True

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    async def get(self) -> STTProviderTurnEvent:
        while True:
            if self._events:
                event = self._events.popleft()
                self._recognition_bytes -= _recognition_bytes(event)
                if not self._events:
                    self._wake.clear()
                return event
            if self._closed:
                raise STTProviderEventBufferClosed("provider event buffer is closed")
            self._wake.clear()
            if self._events:
                self._wake.set()
                continue
            await self._wake.wait()

    async def events(self) -> AsyncIterator[STTProviderTurnEvent]:
        while True:
            try:
                yield await self.get()
            except STTProviderEventBufferClosed:
                return

    def _remove_matching_provisional_snapshots(self, incoming: STTProviderTurnUpdate) -> None:
        self._events = deque(
            event
            for event in self._events
            if not (_is_provisional_snapshot(event) and event.identity == incoming.identity)
        )
        self._recognition_bytes = sum(_recognition_bytes(event) for event in self._events)

    def _discard_provisional_snapshots_until_room(self) -> None:
        retained: deque[STTProviderTurnEvent] = deque()
        removed = False
        while self._events:
            event = self._events.popleft()
            if not removed and _is_provisional_snapshot(event):
                removed = True
                continue
            retained.append(event)
        self._events = retained
        self._recognition_bytes = sum(_recognition_bytes(event) for event in self._events)

    def _fail_scoped_turn(
        self,
        event: STTProviderTurnUpdate | STTProviderTurnTerminal,
    ) -> None:
        identity = event.identity
        self._append_retirement(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="failed",
                text_authority="none",
                failure_reason="provider_event_buffer_overflow",
                epoch_disposition="retire",
                is_current=event.is_current,
            ),
            identity.provider_epoch_id,
        )

    def _fail_epoch(self, epoch_id: str) -> None:
        self._append_retirement(
            STTProviderEpochEnded(epoch_id, orderly=False, reason="provider_event_buffer_overflow"),
            epoch_id,
        )

    def _append_retirement(self, event: STTProviderTurnEvent, epoch_id: str) -> bool:
        retained = deque(
            queued
            for queued in self._events
            if not (
                _event_epoch_id(queued) == epoch_id and isinstance(queued, STTProviderTurnUpdate)
            )
        )
        if len(retained) >= self._max_events + 1:
            return False
        self._events = retained
        self._events.append(event)
        self._retire_epoch(epoch_id)
        self._recognition_bytes = sum(_recognition_bytes(queued) for queued in self._events)
        self._wake.set()
        return True

    def _retire_epoch(self, epoch_id: str) -> None:
        if epoch_id in self._retired_epochs:
            return
        self._retired_epochs.add(epoch_id)
        self._retired_epoch_order.append(epoch_id)
        while len(self._retired_epoch_order) > 4096:
            self._retired_epochs.discard(self._retired_epoch_order.popleft())


def _is_provisional_snapshot(event: STTProviderTurnEvent) -> bool:
    return (
        isinstance(event, STTProviderTurnUpdate)
        and event.stability == "provisional"
        and event.assembly == "replace"
    )


def _event_epoch_id(event: STTProviderTurnEvent) -> str:
    if isinstance(event, STTProviderEpochEnded):
        return event.provider_epoch_id
    if isinstance(event, STTRecognitionUnitTerminal):
        return event.unit.identity.stream.provider_epoch_id
    if isinstance(event, STTRecognitionUnit):
        return event.identity.stream.provider_epoch_id
    return event.identity.provider_epoch_id


def _recognition_bytes(event: STTProviderTurnEvent) -> int:
    if isinstance(event, STTRecognitionUnitTerminal):
        return event.unit.retained_bytes
    if isinstance(event, STTRecognitionUnit):
        return event.retained_bytes
    if (
        isinstance(event, (STTProviderTurnUpdate, STTProviderTurnTerminal))
        and event.recognition_origins
    ):
        return len(event.text.encode("utf-8")) + sum(
            128 + (origin.native.retained_bytes if origin.native is not None else 0)
            for origin in event.recognition_origins
        )
    return 0


__all__ = [
    "STTProviderEventBuffer",
    "STTProviderEventBufferClosed",
]
