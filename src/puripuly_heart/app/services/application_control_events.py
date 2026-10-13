from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from uuid import UUID

from puripuly_heart.core.messages import UserErrorReport
from puripuly_heart.domain.events import STTSessionState, UIEvent, UIEventType
from puripuly_heart.domain.models import OSCMessage, Transcript, Translation


@dataclass(slots=True)
class ControlEvents:
    """One consumer of the runtime queue; independent bounded subscriber projections."""

    history: deque[dict] = field(default_factory=lambda: deque(maxlen=256))
    subscribers: set[asyncio.Queue[int]] = field(default_factory=set)
    sequence: int = 0
    instance_id: str | None = None
    diagnostic_recorder: Callable[..., dict] | None = None
    subscriber_gaps: int = 0
    disconnected_consumers: int = 0

    def publish(self, event: dict) -> None:
        self.sequence += 1
        entry = {**deepcopy(event), "sequence": self.sequence, "instance_id": self.instance_id}
        self.history.append(entry)
        for subscriber in self.subscribers:
            if subscriber.empty():
                subscriber.put_nowait(self.sequence)

    def delivery_health(self) -> dict:
        return {
            "active_subscribers": len(self.subscribers),
            "retained_events": len(self.history),
            "history_capacity": self.history.maxlen,
            "latest_sequence": self.sequence,
            "oldest_sequence": self.history[0]["sequence"] if self.history else self.sequence + 1,
            "evicted_events": max(0, self.sequence - len(self.history)),
            "subscriber_gaps": self.subscriber_gaps,
            "disconnected_consumers": self.disconnected_consumers,
            "restart_resets_history": True,
        }

    def publish_ui(self, event: UIEvent) -> None:
        payload = event.payload
        kind = {
            UIEventType.TRANSCRIPT_PARTIAL: "transcript",
            UIEventType.TRANSCRIPT_FINAL: "transcript",
            UIEventType.TRANSLATION_DONE: "translation",
        }.get(event.type, event.type.value.lower())
        item: dict = {
            "topic": kind,
            "channel": event.channel if event.channel in (None, "self", "peer") else None,
            "source": (
                event.source
                if event.source
                in {
                    "capture",
                    "translation",
                    "osc",
                    "stt",
                    "llm",
                    "self",
                    "peer",
                    "runtime",
                    "manual",
                    "provider",
                }
                else "unknown"
            ),
        }
        if event.type == UIEventType.ERROR:
            item["error"] = {"code": "runtime_error"}
            if isinstance(payload, UserErrorReport) and self.diagnostic_recorder is not None:
                diagnostics = payload.diagnostics
                if diagnostics.channel is None and event.channel in {"self", "peer"}:
                    diagnostics = replace(diagnostics, channel=event.channel)
                if diagnostics.turn_id is None and isinstance(event.utterance_id, UUID):
                    diagnostics = replace(diagnostics, turn_id=str(event.utterance_id))
                diagnostic = self.diagnostic_recorder(
                    diagnostics,
                    severity=payload.message.severity,
                    disposition="failed",
                )
                item["error"]["code"] = diagnostic["code"]
                item["error"]["diagnostics"] = diagnostic
                item["diagnostic_id"] = diagnostic["diagnostic_id"]
        elif event.type in {
            UIEventType.TRANSCRIPT_PARTIAL,
            UIEventType.TRANSCRIPT_FINAL,
            UIEventType.TRANSLATION_DONE,
        }:
            content = (
                isinstance(payload, Transcript)
                if event.type in {UIEventType.TRANSCRIPT_PARTIAL, UIEventType.TRANSCRIPT_FINAL}
                else isinstance(payload, Translation)
            )
            if content:
                item["text"] = payload.text
            utterance_id = payload.utterance_id if content else event.utterance_id
            item["utterance_id"] = str(utterance_id) if isinstance(utterance_id, UUID) else ""
        elif event.type == UIEventType.SESSION_STATE_CHANGED:
            if isinstance(payload, STTSessionState):
                item["state"] = payload.value
        elif event.type == UIEventType.OSC_SENT:
            utterance_id = event.utterance_id
            if utterance_id is None and isinstance(payload, OSCMessage):
                utterance_id = payload.utterance_id
            if isinstance(utterance_id, UUID):
                item["utterance_id"] = str(utterance_id)
        self.publish(item)

    async def subscribe(
        self,
        *,
        topics: list[str],
        channel: str | None,
        include_transcripts: bool,
        include_translations: bool,
        after: int | None,
        cursor_instance_id: str | None = None,
    ) -> AsyncIterator[dict]:
        queue: asyncio.Queue[int] = asyncio.Queue(maxsize=32)
        cursor = self.sequence if after is None else after
        foreign = cursor_instance_id is not None and cursor_instance_id != self.instance_id
        if foreign or (after is not None and after > self.sequence):
            self.subscriber_gaps += 1
            yield {
                "sequence": self.sequence,
                "instance_id": self.instance_id,
                "topic": "gap",
                "after": after,
                "reason": "foreign_instance" if foreign else "invalid_cursor",
                "snapshot_required": True,
            }
            cursor = self.sequence
        self.subscribers.add(queue)
        try:
            while True:
                oldest = self.history[0]["sequence"] if self.history else self.sequence + 1
                if cursor < oldest - 1:
                    self.subscriber_gaps += 1
                    yield {
                        "sequence": oldest - 1,
                        "topic": "gap",
                        "instance_id": self.instance_id,
                        "reason": "history_evicted",
                        "after": cursor,
                        "snapshot_required": True,
                    }
                    cursor = oldest - 1
                    continue
                if cursor < self.sequence:
                    pending = tuple(
                        self.history[sequence - oldest]
                        for sequence in range(cursor + 1, self.sequence + 1)
                    )
                    for entry in pending:
                        sequence = entry["sequence"]
                        if sequence <= cursor:
                            continue
                        cursor = sequence
                        if topics and entry.get("topic") not in topics:
                            continue
                        if channel is not None and entry.get("channel") not in (None, channel):
                            continue
                        projected = deepcopy(entry)
                        if not (
                            (entry.get("topic") == "transcript" and include_transcripts)
                            or (entry.get("topic") == "translation" and include_translations)
                        ):
                            projected.pop("text", None)
                        yield projected
                await queue.get()
        finally:
            self.subscribers.discard(queue)
            self.disconnected_consumers += 1


class ObservedEventQueue:
    """Observe dequeue without competing with the UI event bridge."""

    def __init__(self, queue: asyncio.Queue, events: ControlEvents) -> None:
        self._queue = queue
        self._events = events

    async def get(self) -> UIEvent:
        event = await self._queue.get()
        self._events.publish_ui(event)
        return event

    def task_done(self) -> None:
        self._queue.task_done()
