from __future__ import annotations

from typing import Protocol

from puripuly_heart.core.osc.encoding import OscArg


class OscSender(Protocol):
    def send_message(self, address: str, *values: OscArg) -> None: ...

    def send_chatbox(self, text: str) -> None: ...

    def send_typing(self, is_typing: bool) -> None: ...
