from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.core.stt.backend import STTProviderTurnTerminal
from puripuly_heart.core.stt.batch_speaker import CausalBatchSession
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from tests.core.test_causal_batch_speaker import evidence, payload, request, seal
from tests.core.test_stt_scoped_engine import ControlledScopedSession


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


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "close"])
async def test_healthy_batch_cleanup_drains_owned_terminal_while_rejecting_new_ingress(operation):
    class DrainingInner(ControlledScopedSession):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.emitted = False

        async def complete_owned(self):
            if not self.emitted:
                self.entered.set()
                await self.release.wait()
                self.emit(
                    STTProviderTurnTerminal(self.requests[0].identity, "final", "Owned completion.")
                )
                self.emitted = True

        async def stop(self):
            await self.complete_owned()
            await super().stop()

        async def close(self):
            await self.complete_owned()
            await super().close()

    inner = DrainingInner()
    session = CausalBatchSession(inner)
    req = request()
    pending = None
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 1600, 1)
        await session.observe_speaker_evidence((evidence(0, 1600, (0,)),))
        await seal(session, req)
        pending = asyncio.create_task(getattr(session, operation)())
        await asyncio.wait_for(inner.entered.wait(), 1)
        with pytest.raises(RuntimeError, match="batch capture request is unavailable"):
            await session.begin_turn(request(2))
        await session.observe_speaker_evidence((evidence(0, 1600, (1,)),))
        assert not pending.done()
        inner.release.set()
        await asyncio.wait_for(pending, 1)
        received = [event async for event in session.turn_events()]
        assert len(received) == 1
        terminal = received[0]
        assert terminal.identity == req.identity
        assert terminal.outcome == "final" and terminal.text == "Owned completion."
        assert terminal.final_speaker_runs[0].speaker_id == "0"
        assert session._consumer.done()
        await session.close()
    finally:
        inner.release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await session.close()
