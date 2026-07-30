from __future__ import annotations

from typing import Protocol


class SelfChatboxTypingPort(Protocol):
    def set_self_chatbox_typing_reason(self, reason: str, active: bool) -> None: ...

    def clear_self_chatbox_typing_reasons(self) -> None: ...
