from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias
from uuid import UUID

from puripuly_heart.core.messages import ErrorDiagnostics, UserErrorReport, UserMessageRef

from .models import ChannelId, Transcript


def _validate_channel(channel: str) -> None:
    if channel not in ("self", "peer"):
        raise ValueError(f"invalid channel: {channel!r}")


class STTSessionState(str, Enum):
    CONNECTING = "CONNECTING"
    DISCONNECTED = "DISCONNECTED"
    STREAMING = "STREAMING"
    DRAINING = "DRAINING"


class STTEventType(str, Enum):
    PARTIAL = "STT_PARTIAL"
    FINAL = "STT_FINAL"
    ERROR = "STT_ERROR"
    SESSION_STATE = "STT_SESSION_STATE"


@dataclass(frozen=True, slots=True)
class STTPartialEvent:
    utterance_id: UUID
    transcript: Transcript
    type: STTEventType = STTEventType.PARTIAL

    def __post_init__(self) -> None:
        if self.transcript.is_final:
            raise ValueError("STTPartialEvent requires transcript.is_final == False")

    @property
    def channel(self) -> ChannelId:
        return self.transcript.channel


@dataclass(frozen=True, slots=True)
class STTFinalEvent:
    utterance_id: UUID
    transcript: Transcript
    type: STTEventType = STTEventType.FINAL

    def __post_init__(self) -> None:
        if not self.transcript.is_final:
            raise ValueError("STTFinalEvent requires transcript.is_final == True")

    @property
    def channel(self) -> ChannelId:
        return self.transcript.channel


@dataclass(frozen=True, slots=True)
class STTErrorEvent:
    message: str | UserMessageRef | None = None
    utterance_id: UUID | None = None
    channel: ChannelId = "self"
    runtime_log_handled: bool = False
    diagnostics: ErrorDiagnostics | None = None
    type: STTEventType = STTEventType.ERROR

    def __post_init__(self) -> None:
        _validate_channel(self.channel)


@dataclass(frozen=True, slots=True)
class STTSessionStateEvent:
    state: STTSessionState
    utterance_id: None = None
    channel: ChannelId = "self"
    type: STTEventType = STTEventType.SESSION_STATE

    def __post_init__(self) -> None:
        _validate_channel(self.channel)


STTEvent = STTPartialEvent | STTFinalEvent | STTErrorEvent | STTSessionStateEvent
UIErrorPayload: TypeAlias = UserMessageRef | UserErrorReport | str


class UIEventType(str, Enum):
    SESSION_STATE_CHANGED = "SESSION_STATE_CHANGED"
    TRANSCRIPT_PARTIAL = "TRANSCRIPT_PARTIAL"
    TRANSCRIPT_FINAL = "TRANSCRIPT_FINAL"
    TRANSLATION_DONE = "TRANSLATION_DONE"
    OSC_SENT = "OSC_SENT"
    ERROR = "ERROR"


class UIEventDeliveryAuthority(Protocol):
    def is_current(self) -> bool: ...
    def claim_delivery(self) -> bool: ...

    def claim_presentation(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class UIEvent:
    type: UIEventType
    utterance_id: UUID | None = None
    payload: object | None = None
    source: str | None = None
    channel: ChannelId | None = None
    runtime_log_handled: bool = False
    delivery_authority: UIEventDeliveryAuthority | None = None

    def __post_init__(self) -> None:
        resolved_channel = self.channel
        if resolved_channel is None:
            payload_channel = getattr(self.payload, "channel", None)
            if payload_channel is not None:
                _validate_channel(payload_channel)
                resolved_channel = payload_channel
            else:
                resolved_channel = "self"
        else:
            _validate_channel(resolved_channel)

        object.__setattr__(self, "channel", resolved_channel)
