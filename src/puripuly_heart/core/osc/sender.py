from __future__ import annotations

from typing import Protocol


class OscSender(Protocol):
    def send_chatbox(self, text: str) -> None: ...
    def send_typing(self, is_typing: bool) -> None: ...
