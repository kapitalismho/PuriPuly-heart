from __future__ import annotations

import socket

from ajebal_daera_translator.core.osc.encoding import encode_message
from ajebal_daera_translator.core.osc.udp_sender import VrchatOscUdpSender


def _read_osc_string(packet: bytes, offset: int) -> tuple[str, int]:
    end = packet.index(b"\0", offset)
    value = packet[offset:end].decode("utf-8")
    next_offset = end + 1
    next_offset += (-next_offset) % 4
    return value, next_offset


def test_encode_message_chatbox_input():
    packet = encode_message("/chatbox/input", ["hello", True, False])
    address, offset = _read_osc_string(packet, 0)
    assert address == "/chatbox/input"

    tags, offset = _read_osc_string(packet, offset)
    assert tags == ",sTF"

    text, offset = _read_osc_string(packet, offset)
    assert text == "hello"

    assert offset == len(packet)


def test_vrchat_udp_sender_sends_packet():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(1.0)
    host, port = server.getsockname()
    assert host == "127.0.0.1"

    sender = VrchatOscUdpSender(host="127.0.0.1", port=port)
    try:
        sender.send_chatbox("hello")
        packet, _addr = server.recvfrom(65535)
    finally:
        sender.close()
        server.close()

    address, offset = _read_osc_string(packet, 0)
    assert address == "/chatbox/input"

    tags, offset = _read_osc_string(packet, offset)
    assert tags == ",sTF"

    text, offset = _read_osc_string(packet, offset)
    assert text == "hello"

    assert offset == len(packet)

