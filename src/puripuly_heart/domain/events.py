from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from .models import Transcript


class STTSessionState(str, Enum):
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


@dataclass(frozen=True, slots=True)
class STTFinalEvent:
    utterance_id: UUID
    transcript: Transcript
    type: STTEventType = STTEventType.FINAL

    def __post_init__(self) -> None:
        if not self.transcript.is_final:
            raise ValueError("STTFinalEvent requires transcript.is_final == True")


@dataclass(frozen=True, slots=True)
class STTErrorEvent:
    message: str
    utterance_id: UUID | None = None
    type: STTEventType = STTEventType.ERROR


@dataclass(frozen=True, slots=True)
class STTSessionStateEvent:
    state: STTSessionState
    utterance_id: None = None
    type: STTEventType = STTEventType.SESSION_STATE


STTEvent = STTPartialEvent | STTFinalEvent | STTErrorEvent | STTSessionStateEvent


class UIEventType(str, Enum):
    SESSION_STATE_CHANGED = "SESSION_STATE_CHANGED"
    TRANSCRIPT_PARTIAL = "TRANSCRIPT_PARTIAL"
    TRANSCRIPT_FINAL = "TRANSCRIPT_FINAL"
    TRANSLATION_DONE = "TRANSLATION_DONE"
    OSC_SENT = "OSC_SENT"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class UIEvent:
    type: UIEventType
    utterance_id: UUID | None = None
    payload: object | None = None
    source: str | None = None
