from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.core.stt.batch_speaker import CausalBatchSession
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession


class InterruptedInner:
    def __init__(self, mode):
        self.mode = mode
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.events = STTProviderEventBuffer()
        self.close_calls = 0

    async def turn_events(self):
        async for event in self.events.events():
            yield event

    async def interrupt(self):
        self.entered.set()
        if self.mode == "error":
            raise OSError("unconfirmed inner cleanup")
        await self.release.wait()

    async def stop(self):
        await self.interrupt()
        self.events.close()

    async def close(self):
        self.close_calls += 1
        await self.interrupt()
        self.events.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper", [StreamingSpeakerSession, CausalBatchSession])
@pytest.mark.parametrize("operation", ["stop", "close"])
@pytest.mark.parametrize("mode", ["error", "cancel"])
async def test_interrupted_wrapper_cleanup_settles_own_consumer_and_retries_inner(
    wrapper, operation, mode
):
    inner = InterruptedInner(mode)
    session = wrapper(inner)
    consumer = session._consumer
    pending = asyncio.create_task(getattr(session, operation)())
    try:
        await inner.entered.wait()
        if mode == "cancel":
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            with pytest.raises(OSError, match="unconfirmed inner cleanup"):
                await pending
        assert consumer.done()
        attempts = inner.close_calls
        inner.mode = "healthy"
        inner.release.set()
        await session.close()
        assert inner.close_calls == attempts + 1
        await session.close()
        assert inner.close_calls == attempts + 1
        assert consumer.done()
    finally:
        inner.mode = "healthy"
        inner.release.set()
        await inner.close()
        if not consumer.done():
            consumer.cancel()
        await asyncio.gather(consumer, pending, return_exceptions=True)
