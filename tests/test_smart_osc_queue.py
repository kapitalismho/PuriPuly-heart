from __future__ import annotations

import uuid
from dataclasses import dataclass

from ajebal_daera_translator.core.clock import FakeClock
from ajebal_daera_translator.core.osc.smart_queue import SmartOscQueue
from ajebal_daera_translator.domain.models import OSCMessage


@dataclass(slots=True)
class FakeSender:
    sent: list[str]

    def __init__(self) -> None:
        self.sent = []

    def send_chatbox(self, text: str) -> None:
        self.sent.append(text)


def test_smart_queue_cooldown_and_flush():
    clock = FakeClock()
    sender = FakeSender()
    queue = SmartOscQueue(sender=sender, clock=clock, cooldown_s=1.5, ttl_s=100.0)

    queue.enqueue(OSCMessage(uuid.uuid4(), text="hello", created_at=clock.now()))
    assert sender.sent == ["hello"]

    clock.advance(0.5)
    queue.enqueue(OSCMessage(uuid.uuid4(), text="world", created_at=clock.now()))
    assert sender.sent == ["hello"]

    clock.advance(1.0)  # now=1.5
    queue.process_due()
    assert sender.sent == ["hello", "world"]


def test_smart_queue_splits_and_carries_over():
    clock = FakeClock()
    sender = FakeSender()
    queue = SmartOscQueue(sender=sender, clock=clock, cooldown_s=1.0, ttl_s=100.0, max_chars=10)

    uid = uuid.uuid4()
    queue.enqueue(OSCMessage(uid, text="one two three four", created_at=clock.now()))
    assert len(sender.sent) == 1

    clock.advance(1.0)
    queue.process_due()
    assert len(sender.sent) == 2


def test_smart_queue_ttl_drop():
    clock = FakeClock()
    sender = FakeSender()
    queue = SmartOscQueue(sender=sender, clock=clock, cooldown_s=1.5, ttl_s=1.0)

    queue.enqueue(OSCMessage(uuid.uuid4(), text="first", created_at=clock.now()))
    clock.advance(0.1)
    queue.enqueue(OSCMessage(uuid.uuid4(), text="stale", created_at=clock.now()))

    clock.advance(2.0)
    queue.process_due()

    assert sender.sent == ["first"]
