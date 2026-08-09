from __future__ import annotations

import socket
from dataclasses import dataclass, field

from pythonosc.osc_message_builder import OscMessageBuilder

from puripuly_heart.core.osc.encoding import OscArg
from puripuly_heart.core.osc.sender import OscSender


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

    def set_destination(self, host: str, port: int) -> None:
        if not host:
            raise ValueError("OSC destination host must be non-empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("OSC destination port must be in 1..65535")
        self.host = host
        self.port = int(port)

    def send_message(self, address: str, *values: OscArg) -> None:
        if not address or not address.startswith("/"):
            raise ValueError("OSC address must start with '/'")
        builder = self._OscMessageBuilder(address=address)
        for value in values:
            builder.add_arg(value)
        self._sock.sendto(builder.build().dgram, (self.host, self.port))

    def send_chatbox(self, text: str) -> None:
        self.send_message(self.chatbox_address, text, self.chatbox_send, self.chatbox_clear)

    def send_typing(self, is_typing: bool) -> None:
        """Send typing indicator to VRChat chatbox."""
        self.send_message(self.typing_address, is_typing)
