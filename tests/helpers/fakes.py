from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puripuly_heart.domain.models import OSCMessage


@dataclass(slots=True)
class FakeSender:
    sent: list[str]
    typing: list[bool]

    def __init__(self) -> None:
        self.sent = []
        self.typing = []

    def send_chatbox(self, text: str) -> None:
        self.sent.append(text)

    def send_typing(self, is_typing: bool) -> None:
        self.typing.append(is_typing)


@dataclass(slots=True)
class RecordingOscQueue:
    messages: list[OSCMessage]
    typing: list[bool]
    immediate_messages: list[str]
    process_due_calls: int
    immediate_result: bool

    def __init__(self, *, immediate_result: bool = True) -> None:
        self.messages = []
        self.typing = []
        self.immediate_messages = []
        self.process_due_calls = 0
        self.immediate_result = immediate_result

    def enqueue(self, message: OSCMessage) -> None:
        self.messages.append(message)

    def send_typing(self, is_typing: bool) -> None:
        self.typing.append(is_typing)

    def set_typing_reason(self, reason: str, active: bool) -> None:
        self.typing.append(active)

    def send_immediate(self, text: str) -> tuple[bool, None]:
        self.immediate_messages.append(text)
        return self.immediate_result, None

    def process_due(self) -> None:
        self.process_due_calls += 1


def samples(value: float, n: int = 512) -> np.ndarray:
    return np.full((n,), value, dtype=np.float32)
