from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from puripuly_heart.domain.events import STTSessionState, UIEvent, UIEventType
from puripuly_heart.domain.models import OSCMessage, Transcript, Translation

MappedEventKind = Literal["status", "transcript", "translation", "osc", "error"]


@dataclass(frozen=True, slots=True)
class MappedEvent:
    kind: MappedEventKind
    payload: object
    source: str | None
    status: str | None = None
    transcript_kind: Literal["partial", "final"] | None = None
    runtime_log_handled: bool = False


def map_ui_event(event: UIEvent) -> MappedEvent | None:
    if event.type == UIEventType.SESSION_STATE_CHANGED:
        return MappedEvent(
            kind="status",
            payload=event.payload,
            source=event.source,
            status=_status_for_session_state(event.payload),
        )
    if event.type in (UIEventType.TRANSCRIPT_PARTIAL, UIEventType.TRANSCRIPT_FINAL):
        if not isinstance(event.payload, Transcript):
            return None
        transcript_kind: Literal["partial", "final"] = (
            "final" if event.type == UIEventType.TRANSCRIPT_FINAL else "partial"
        )
        return MappedEvent(
            kind="transcript",
            payload=event.payload,
            source=event.source,
            transcript_kind=transcript_kind,
        )
    if event.type == UIEventType.TRANSLATION_DONE:
        if not isinstance(event.payload, Translation):
            return None
        return MappedEvent(
            kind="translation",
            payload=event.payload,
            source=event.source,
            runtime_log_handled=event.runtime_log_handled,
        )
    if event.type == UIEventType.OSC_SENT:
        if not isinstance(event.payload, OSCMessage):
            return None
        return MappedEvent(kind="osc", payload=event.payload, source=event.source)
    if event.type == UIEventType.ERROR:
        return MappedEvent(kind="error", payload=event.payload, source=event.source)
    return None


def _status_for_session_state(state: object) -> str:
    if state == STTSessionState.CONNECTING or getattr(state, "name", "") == "CONNECTING":
        return "connecting"
    if state == STTSessionState.STREAMING or getattr(state, "name", "") == "STREAMING":
        return "connected"
    if state == STTSessionState.DRAINING or getattr(state, "name", "") == "DRAINING":
        return "stopping"
    return "disconnected"


__all__ = ["MappedEvent", "MappedEventKind", "map_ui_event"]
