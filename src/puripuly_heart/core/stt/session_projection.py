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
    def __init__(self, projection: STTSessionProjection) -> None:
        self._projection = projection
        self._legacy_events: asyncio.Queue[LegacySTTEvent] | None = None
        self._scoped_events: STTProviderEventBuffer | None = None
        if projection.mode == "legacy":
            self._legacy_events = asyncio.Queue()
        else:
            self._scoped_events = STTProviderEventBuffer()
        self._active_identity: STTProviderTurnIdentity | None = None
        self._payload_sequence = 0
        self._update_sequence = 0
        self._sealed = False
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
    def sealed(self) -> bool:
        return self._sealed

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
        if self._active_identity is not None:
            raise RuntimeError("STT session already has an unresolved turn")
        self._active_identity = request.identity
        self._payload_sequence = 0
        self._update_sequence = 0
        self._sealed = False

    def require_open(self, identity: STTProviderTurnIdentity) -> None:
        self._require_scoped_projection()
        if self._active_identity != identity:
            raise RuntimeError("unknown or retired STT provider turn")

    def validate_payload(
        self,
        identity: STTProviderTurnIdentity,
        payload_sequence: int,
    ) -> None:
        self.require_open(identity)
        if self._sealed:
            raise RuntimeError("STT provider turn is already sealed")
        if payload_sequence != self._payload_sequence + 1:
            raise ValueError("payload_sequence must be contiguous")

    def payload_written(
        self,
        identity: STTProviderTurnIdentity,
        payload_sequence: int,
    ) -> None:
        self.validate_payload(identity, payload_sequence)
        self._payload_sequence = payload_sequence

    def seal(self, identity: STTProviderTurnIdentity) -> None:
        self.require_open(identity)
        if self._sealed:
            raise RuntimeError("STT provider turn is already sealed")
        self._sealed = True

    def next_update_sequence(self, identity: STTProviderTurnIdentity) -> int | None:
        if not self.is_current(identity):
            return None
        self._update_sequence += 1
        return self._update_sequence

    def is_current(self, identity: STTProviderTurnIdentity) -> bool:
        return not self._retired and self._active_identity == identity

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
        if self._active_identity == event.identity:
            self._active_identity = None
            self._payload_sequence = 0
            self._update_sequence = 0
            self._sealed = False
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
        if self._active_identity == identity:
            self._active_identity = None
            self._payload_sequence = 0
            self._update_sequence = 0
            self._sealed = False


__all__ = ["STTSessionEventProjection"]
