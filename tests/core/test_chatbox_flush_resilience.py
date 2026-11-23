from __future__ import annotations

import asyncio
import socket
from contextlib import closing
from uuid import uuid4

import pytest
from pythonosc.osc_message import OscMessage

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
    try:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.2)
        sender = VrchatOscUdpSender(host="127.0.0.1", port=receiver.getsockname()[1])
    except BaseException:
        receiver.close()
        raise
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
            lambda message, status, diagnostics: statuses.append(
                (str(message.utterance_id), status)
            )
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


async def test_output_runtime_sends_first_chatbox_tail_after_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender, receiver = _capture()
    with closing(sender), receiver:
        clock = SystemClock()
        page_interval_s = 0.05
        paginator = ChatboxPaginator(
            sender=sender, clock=clock, max_chars=5, page_interval_s=page_interval_s
        )
        runtime = OutputRuntime(chatbox=paginator, clock=clock)
        idle = asyncio.Event()
        schedule_changed = paginator._schedule_changed
        original_wait = schedule_changed.wait
        send_times: list[float] = []
        original_send_chatbox = VrchatOscUdpSender.send_chatbox

        async def observe_wait() -> bool:
            if not schedule_changed.is_set():
                idle.set()
            return await original_wait()

        def record_send(udp_sender: VrchatOscUdpSender, text: str) -> None:
            send_times.append(clock.now())
            original_send_chatbox(udp_sender, text)

        monkeypatch.setattr(schedule_changed, "wait", observe_wait)
        monkeypatch.setattr(VrchatOscUdpSender, "send_chatbox", record_send)
        try:
            await runtime.start(auto_flush_chatbox=True)
            await asyncio.wait_for(idle.wait(), timeout=2.0)

            await runtime.publish_chatbox(
                publication_id=uuid4(),
                channel="self",
                transcript_text="alpha bravo",
                translation_text=None,
                include_source=False,
            )
            head_packet, _ = receiver.recvfrom(65535)
            head = OscMessage(head_packet)
            assert (head.address, head.params) == ("/chatbox/input", ["alpha", True, False])

            receiver.setblocking(False)
            try:
                async with asyncio.timeout(2.0):
                    tail_packet, _ = await asyncio.get_running_loop().sock_recvfrom(receiver, 65535)
            except TimeoutError:
                pytest.fail("Chatbox tail 'bravo' was not received after inline head 'alpha'")
            tail = OscMessage(tail_packet)
            assert (tail.address, tail.params) == ("/chatbox/input", ["bravo", True, False])
            head_at, tail_at = send_times
            assert tail_at >= head_at + page_interval_s, "Chatbox tail was sent before eligibility"
        finally:
            await asyncio.wait_for(runtime.close(), timeout=1.0)


async def test_output_runtime_chatbox_survives_unencodable_page_and_close_succeeds() -> None:
    sender, receiver = _capture()
    runtime: OutputRuntime | None = None
    try:
        runtime = OutputRuntime(
            chatbox=ChatboxPaginator(sender=sender, clock=SystemClock(), page_interval_s=0.1),
            clock=SystemClock(),
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
        receiver.setblocking(False)
        loop = asyncio.get_running_loop()
        good_pages: list[bytes] = []
        async with asyncio.timeout(5.0):
            while len(good_pages) < 3:
                payload, _ = await loop.sock_recvfrom(receiver, 65535)
                if b"yyyy" in payload:
                    good_pages.append(payload)

        receiver.settimeout(0.2)
        good_pages.extend(payload for payload in _drain(receiver) if b"yyyy" in payload)
        assert len(good_pages) == 3

    finally:
        try:
            if runtime is not None:
                await asyncio.wait_for(runtime.close(), timeout=1.0)
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


def test_queued_send_cause_survives_failing_observer_without_stopping_pagination():
    paginator = ChatboxPaginator(sender=_RaisingSender(), clock=FakeClock(), max_chars=5)
    receipts = []

    def observe(message, status, diagnostics):
        receipts.append((status, diagnostics))
        raise OSError(5, "private observer detail")

    paginator.bind_delivery_observer(observe)
    paginator.enqueue(OSCMessage(utterance_id=uuid4(), text="synthetic pages", created_at=0.0))
    primary = paginator.last_failure_diagnostics
    assert receipts[0] == ("output_send_failed", primary)
    assert primary.fields["exception_class"] == "ValueError"
    assert paginator.delivery_observer_failure.cause_id == primary.diagnostic_id
    assert paginator.delivery_observer_failure.fields["errno"] == 5
    paginator.clock.advance(3.0)
    paginator.process_due()
    assert len(receipts) == 2
    assert "private" not in repr((primary, paginator.delivery_observer_failure))
    paginator.drop_pending()
