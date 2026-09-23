from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from puripuly_heart.core.stt.backend import (
    STTBackendTranscriptEvent,
    STTProviderEpochEnded,
    STTProviderTurnEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_event_buffer import (
    STTProviderEventBuffer,
    STTProviderEventBufferClosed,
)

LegacySTTEvent = STTBackendTranscriptEvent | BaseException | None


class STTSessionEventProjection:
    def __init__(
        self,
        projection: STTSessionProjection,
        *,
        allows_sealed_turn_overlap: bool = False,
    ) -> None:
        self._projection = projection
        self._allows_sealed_turn_overlap = allows_sealed_turn_overlap
        self._legacy_events: asyncio.Queue[LegacySTTEvent] | None = None
        self._scoped_events: STTProviderEventBuffer | None = None
        if projection.mode == "legacy":
            self._legacy_events = asyncio.Queue()
        else:
            self._scoped_events = STTProviderEventBuffer()
        self._active_identity: STTProviderTurnIdentity | None = None
        self._payload_sequences: dict[STTProviderTurnIdentity, int] = {}
        self._update_sequences: dict[STTProviderTurnIdentity, int] = {}
        self._sealed_identities: set[STTProviderTurnIdentity] = set()
        self._retired = False
        self._epoch_ended = False
        self._closed = False

    @property
    def projection(self) -> STTSessionProjection:
        return self._projection

    @property
    def is_legacy(self) -> bool:
        return self._projection.mode == "legacy"

    @property
    def is_scoped(self) -> bool:
        return self._projection.mode == "scoped"

    @property
    def provider_epoch_id(self) -> str | None:
        return self._projection.provider_epoch_id

    @property
    def active_identity(self) -> STTProviderTurnIdentity | None:
        return self._active_identity

    @property
    def identities(self) -> tuple[STTProviderTurnIdentity, ...]:
        return tuple(self._payload_sequences)

    @property
    def sealed(self) -> bool:
        identity = self._active_identity
        return (
            identity in self._sealed_identities
            if identity is not None
            else bool(self._sealed_identities)
        )

    @property
    def retired(self) -> bool:
        return self._retired

    @property
    def scoped_event_depth(self) -> int:
        return self._scoped_buffer().depth

    def begin(self, request: STTProviderTurnRequest) -> None:
        self._require_scoped_projection()
        if self._closed:
            raise RuntimeError("STT provider session is closed")
        if self._retired:
            raise RuntimeError("STT provider epoch is retired")
        if request.identity.provider_epoch_id != self.provider_epoch_id:
            raise RuntimeError("STT turn belongs to a different provider epoch")
        active = self._active_identity
        if active is not None and (
            not self._allows_sealed_turn_overlap or active not in self._sealed_identities
        ):
            raise RuntimeError("STT session already has an unresolved turn")
        if request.identity in self._payload_sequences:
            raise RuntimeError("STT provider turn identity is already active")
        self._active_identity = request.identity
        self._payload_sequences[request.identity] = 0
        self._update_sequences[request.identity] = 0

    def require_open(self, identity: STTProviderTurnIdentity) -> None:
        self._require_scoped_projection()
        if identity not in self._payload_sequences:
            raise RuntimeError("unknown or retired STT provider turn")

    def validate_payload(
        self,
        identity: STTProviderTurnIdentity,
        payload_sequence: int,
    ) -> None:
        self.require_open(identity)
        if identity in self._sealed_identities:
            raise RuntimeError("STT provider turn is already sealed")
        if payload_sequence != self._payload_sequences[identity] + 1:
            raise ValueError("payload_sequence must be contiguous")

    def payload_written(
        self,
        identity: STTProviderTurnIdentity,
        payload_sequence: int,
    ) -> None:
        self.validate_payload(identity, payload_sequence)
        self._payload_sequences[identity] = payload_sequence

    def seal(self, identity: STTProviderTurnIdentity) -> None:
        self.require_open(identity)
        if identity in self._sealed_identities:
            raise RuntimeError("STT provider turn is already sealed")
        self._sealed_identities.add(identity)

    def next_update_sequence(self, identity: STTProviderTurnIdentity) -> int | None:
        if not self.is_current(identity):
            return None
        sequence = self._update_sequences[identity] + 1
        self._update_sequences[identity] = sequence
        return sequence

    def is_current(self, identity: STTProviderTurnIdentity) -> bool:
        return identity in self._payload_sequences

    def put_legacy(self, event: LegacySTTEvent) -> bool:
        queue = self._legacy_events
        if queue is None or self._closed:
            return False
        queue.put_nowait(event)
        return True

    def discard_legacy(self) -> None:
        queue = self._legacy_events
        if queue is None:
            return
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def put_update(self, event: STTProviderTurnUpdate) -> bool:
        if not self.is_current(event.identity):
            return False
        buffer = self._scoped_buffer()
        try:
            accepted = buffer.put(event)
        except STTProviderEventBufferClosed:
            return False
        if not accepted and event.stability != "provisional":
            self._retire_after_overflow(event.identity)
        return accepted

    def terminal(self, event: STTProviderTurnTerminal) -> bool:
        if not self.is_current(event.identity):
            return False
        buffer = self._scoped_buffer()
        try:
            accepted = buffer.put(event)
        except STTProviderEventBufferClosed:
            accepted = False
        if event.epoch_disposition == "retire" or not accepted:
            self._retired = True
        self._payload_sequences.pop(event.identity, None)
        self._update_sequences.pop(event.identity, None)
        self._sealed_identities.discard(event.identity)
        if self._active_identity == event.identity:
            self._active_identity = None
        return accepted

    def retire(self) -> None:
        self._retired = True

    def end_epoch(
        self,
        *,
        orderly: bool,
        reason: str,
        provider_turn_id: str | None = None,
    ) -> bool:
        if not self.is_scoped or self._epoch_ended or self._closed:
            return False
        epoch_id = self.provider_epoch_id
        if epoch_id is None:
            raise RuntimeError("scoped STT projection is missing its provider epoch")
        self._retired = True
        self._epoch_ended = True
        event = STTProviderEpochEnded(
            provider_epoch_id=epoch_id,
            orderly=orderly,
            reason=reason,
            provider_turn_id=provider_turn_id,
        )
        try:
            return self._scoped_buffer().put(event)
        except STTProviderEventBufferClosed:
            return False

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        queue = self._legacy_events
        if queue is None:
            raise RuntimeError("legacy STT events are unavailable for a scoped session")
        while True:
            event = await queue.get()
            if event is None:
                return
            if isinstance(event, BaseException):
                raise event
            yield event

    async def turn_events(self) -> AsyncIterator[STTProviderTurnEvent]:
        buffer = self._scoped_events
        if buffer is None:
            raise RuntimeError("scoped STT events are unavailable for a legacy session")
        async for event in buffer.events():
            yield event

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._legacy_events is not None:
            self._legacy_events.put_nowait(None)
        if self._scoped_events is not None:
            self._scoped_events.close()

    def _require_scoped_projection(self) -> None:
        if not self.is_scoped:
            raise RuntimeError("scoped STT operations are unavailable for a legacy session")

    def _scoped_buffer(self) -> STTProviderEventBuffer:
        buffer = self._scoped_events
        if buffer is None:
            raise RuntimeError("scoped STT events are unavailable for a legacy session")
        return buffer

    def _retire_after_overflow(self, identity: STTProviderTurnIdentity) -> None:
        self._retired = True
        self._payload_sequences.pop(identity, None)
        self._update_sequences.pop(identity, None)
        self._sealed_identities.discard(identity)
        if self._active_identity == identity:
            self._active_identity = None


__all__ = ["STTSessionEventProjection"]
