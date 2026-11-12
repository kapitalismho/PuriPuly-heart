from __future__ import annotations

import asyncio
import socket
from uuid import uuid4

from puripuly_heart.core.clock import FakeClock, SystemClock
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.domain.models import OSCMessage

UNENCODABLE = "x" * 145 + "\ud800"
GOOD_LONG = "y" * 300


class _RaisingSender:
    def send_message(self, address: str, *values: object) -> None:
        raise ValueError("unencodable payload")

    def send_chatbox(self, text: str) -> None:
        raise ValueError("unencodable payload")

    def send_typing(self, is_typing: bool) -> None:
        raise ValueError("unencodable payload")


def _capture() -> tuple[VrchatOscUdpSender, socket.socket]:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(0.2)
    sender = VrchatOscUdpSender(host="127.0.0.1", port=receiver.getsockname()[1])
    return sender, receiver


def _drain(receiver: socket.socket) -> list[bytes]:
    payloads: list[bytes] = []
    while True:
        try:
            data, _ = receiver.recvfrom(65535)
        except socket.timeout:
            break
        payloads.append(data)
    return payloads


def test_unencodable_tail_page_is_dropped_and_pagination_continues() -> None:
    sender, receiver = _capture()
    try:
        clock = FakeClock()
        paginator = ChatboxPaginator(sender=sender, clock=clock)
        statuses: list[tuple[str, str]] = []
        paginator.bind_delivery_observer(
            lambda message, status: statuses.append((str(message.utterance_id), status))
        )

        bad_id = uuid4()
        paginator.enqueue(OSCMessage(utterance_id=bad_id, text=UNENCODABLE, created_at=clock.now()))
        head_payloads = _drain(receiver)
        assert any(b"x" * 16 in payload for payload in head_payloads)

        clock.advance(3.0)
        paginator.process_due()

        good_id = uuid4()
        paginator.enqueue(OSCMessage(utterance_id=good_id, text=GOOD_LONG, created_at=clock.now()))
        clock.advance(3.0)
        paginator.process_due()
        clock.advance(3.0)
        paginator.process_due()

        payloads = _drain(receiver)
        good_pages = [payload for payload in payloads if b"yyyy" in payload]
        assert len(good_pages) == 3
        assert (str(good_id), "first_page_sent") in statuses
    finally:
        sender.close()
        receiver.close()


async def test_output_runtime_chatbox_survives_unencodable_page_and_close_succeeds() -> None:
    sender, receiver = _capture()
    try:
        runtime = OutputRuntime(
            chatbox=ChatboxPaginator(sender=sender, clock=SystemClock(), page_interval_s=0.1),
            clock=SystemClock(),
            flush_interval_s=0.05,
        )
        await runtime.start(auto_flush_chatbox=True)

        await runtime.publish_chatbox(
            publication_id=uuid4(),
            channel="self",
            transcript_text=UNENCODABLE,
            translation_text=None,
            include_source=False,
        )
        await runtime.publish_chatbox(
            publication_id=uuid4(),
            channel="self",
            transcript_text=GOOD_LONG,
            translation_text=None,
            include_source=False,
        )
        await asyncio.sleep(1.2)

        payloads = _drain(receiver)
        good_pages = [payload for payload in payloads if b"yyyy" in payload]
        assert len(good_pages) == 3

        await runtime.close()
    finally:
        sender.close()
        receiver.close()


def test_send_failures_outside_oserror_are_reported_not_raised() -> None:
    clock = FakeClock()
    paginator = ChatboxPaginator(sender=_RaisingSender(), clock=clock)

    published, diagnostics = paginator.send_immediate("hello")
    assert published is False
    assert diagnostics is not None

    paginator.send_typing(True)
    paginator.send_typing(False)
