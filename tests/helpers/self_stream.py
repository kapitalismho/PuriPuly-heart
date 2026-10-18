from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np

from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import Transcript, Translation


@dataclass
class Provider:
    calls: list[dict] = field(default_factory=list)
    releases: dict[tuple[str, str], asyncio.Event] = field(default_factory=dict)
    cancelled: list[tuple[str, str]] = field(default_factory=list)

    async def translate(self, *, utterance_id, text, target_language, context="", **kwargs):
        self.calls.append({"text": text, "target": target_language, "context": context})
        key = (text, target_language)
        try:
            await self.releases.setdefault(key, asyncio.Event()).wait()
        except asyncio.CancelledError:
            self.cancelled.append(key)
            raise
        return Translation(utterance_id=utterance_id, text=f"{target_language}:{text}")

    def release(self, text, target="en"):
        self.releases.setdefault((text, target), asyncio.Event()).set()

    async def close(self):
        return None


@dataclass
class Sender:
    packets: list[str] = field(default_factory=list)

    def send_chatbox(self, text):
        self.packets.append(text)

    def send_typing(self, value):
        return None


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def contribution(harness, identity, text):
    await harness.dispatch_stt_event(
        STTFinalEvent(identity, Transcript(utterance_id=identity, text=text, is_final=True))
    )


async def begin(harness, text, *, ended=False):
    identity = uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(
            identity, pre_roll=np.empty(0, dtype=np.float32), chunk=np.ones(512, dtype=np.float32)
        )
    )
    if ended:
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
    await contribution(harness, identity, text)
    return identity
