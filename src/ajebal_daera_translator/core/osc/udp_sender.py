from __future__ import annotations

import socket
from dataclasses import dataclass, field

from ajebal_daera_translator.core.osc.encoding import encode_message
from ajebal_daera_translator.core.osc.sender import OscSender


@dataclass(slots=True)
class VrchatOscUdpSender(OscSender):
    host: str = "127.0.0.1"
    port: int = 9000
    chatbox_address: str = "/chatbox/input"
    chatbox_send: bool = True
    chatbox_clear: bool = False
    _sock: socket.socket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must be non-empty")
        if not (0 < self.port <= 65535):
            raise ValueError("port must be in 1..65535")
        if not self.chatbox_address or not self.chatbox_address.startswith("/"):
            raise ValueError("chatbox_address must start with '/'")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self._sock.close()

    def send_chatbox(self, text: str) -> None:
        packet = encode_message(
            self.chatbox_address,
            [text, self.chatbox_send, self.chatbox_clear],
        )
        self._sock.sendto(packet, (self.host, self.port))

