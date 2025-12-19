from __future__ import annotations

import socket
from dataclasses import dataclass, field

from ajebal_daera_translator.core.osc.sender import OscSender
from pythonosc.osc_message_builder import OscMessageBuilder


@dataclass(slots=True)
class VrchatOscUdpSender(OscSender):
    host: str = "127.0.0.1"
    port: int = 9000
    chatbox_address: str = "/chatbox/input"
    typing_address: str = "/chatbox/typing"
    chatbox_send: bool = True
    chatbox_clear: bool = False
    _sock: socket.socket = field(init=False, repr=False)
    _OscMessageBuilder: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must be non-empty")
        if not (0 < self.port <= 65535):
            raise ValueError("port must be in 1..65535")
        if not self.chatbox_address or not self.chatbox_address.startswith("/"):
            raise ValueError("chatbox_address must start with '/'")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._OscMessageBuilder = OscMessageBuilder

    def close(self) -> None:
        self._sock.close()

    def send_chatbox(self, text: str) -> None:
        builder = self._OscMessageBuilder(address=self.chatbox_address)
        builder.add_arg(text)
        builder.add_arg(self.chatbox_send)
        builder.add_arg(self.chatbox_clear)
        packet = builder.build().dgram
        self._sock.sendto(packet, (self.host, self.port))

    def send_typing(self, is_typing: bool) -> None:
        """Send typing indicator to VRChat chatbox."""
        builder = self._OscMessageBuilder(address=self.typing_address)
        builder.add_arg(is_typing)
        packet = builder.build().dgram
        self._sock.sendto(packet, (self.host, self.port))
