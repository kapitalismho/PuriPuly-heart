from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from puripuly_heart.domain.events import SelfTurnPresentation, UIEvent, UIEventType
from puripuly_heart.domain.models import OSCMessage, Transcript, Translation

MappedEventKind = Literal["transcript", "translation", "self_turn", "osc", "error"]


@dataclass(frozen=True, slots=True)
class MappedEvent:
    kind: MappedEventKind
    payload: object
    source: str | None
    transcript_kind: Literal["partial", "final"] | None = None
    runtime_log_handled: bool = False
    peer_caption_revision: int = 0


def map_ui_event(event: UIEvent) -> MappedEvent | None:
    if event.type == UIEventType.SELF_TURN_UPDATED:
        if not isinstance(event.payload, SelfTurnPresentation):
            return None
        return MappedEvent(kind="self_turn", payload=event.payload, source=event.source)
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
            peer_caption_revision=event.peer_caption_revision,
        )
    if event.type == UIEventType.TRANSLATION_DONE:
        if not isinstance(event.payload, Translation):
            return None
        return MappedEvent(
            kind="translation",
            payload=event.payload,
            source=event.source,
            runtime_log_handled=event.runtime_log_handled,
            peer_caption_revision=event.peer_caption_revision,
        )
    if event.type == UIEventType.OSC_SENT:
        if not isinstance(event.payload, OSCMessage):
            return None
        return MappedEvent(kind="osc", payload=event.payload, source=event.source)
    if event.type == UIEventType.ERROR:
        return MappedEvent(kind="error", payload=event.payload, source=event.source)
    return None


__all__ = ["MappedEvent", "MappedEventKind", "map_ui_event"]
