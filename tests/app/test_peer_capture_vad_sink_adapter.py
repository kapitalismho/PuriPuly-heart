from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.app.adapters.peer_capture_vad_sink import (
    PeerCaptureVadSinkAdapter,
)
from puripuly_heart.app.wiring import create_peer_capture_vad_sink_adapter


class Runtime:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def handle_peer_vad_event(self, event: object) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_adapter_resolves_current_runtime_for_each_peer_event() -> None:
    first = Runtime()
    second = Runtime()
    current = [first]
    adapter = PeerCaptureVadSinkAdapter(runtime_provider=lambda: current[0])
    first_event = object()
    second_event = object()

    await adapter.handle_vad_event(first_event)
    current[0] = second
    await adapter.handle_vad_event(second_event)

    assert first.events == [first_event]
    assert second.events == [second_event]


@pytest.mark.asyncio
async def test_adapter_rejects_event_without_current_runtime() -> None:
    adapter = PeerCaptureVadSinkAdapter(runtime_provider=lambda: None)

    with pytest.raises(RuntimeError, match="Peer VAD sink requires the Peer translation owner"):
        await adapter.handle_vad_event(object())


@pytest.mark.asyncio
async def test_adapter_propagates_downstream_exception_and_cancellation() -> None:
    class FailingRuntime:
        async def handle_peer_vad_event(self, _event: object) -> None:
            raise RuntimeError("peer event failed")

    adapter = PeerCaptureVadSinkAdapter(runtime_provider=FailingRuntime)
    with pytest.raises(RuntimeError, match="peer event failed"):
        await adapter.handle_vad_event(object())

    class CancellingRuntime:
        async def handle_peer_vad_event(self, _event: object) -> None:
            raise asyncio.CancelledError

    adapter = PeerCaptureVadSinkAdapter(runtime_provider=CancellingRuntime)
    with pytest.raises(asyncio.CancelledError):
        await adapter.handle_vad_event(object())


def test_wiring_factory_composes_internal_peer_vad_sink_adapter() -> None:
    runtime = Runtime()
    adapter = create_peer_capture_vad_sink_adapter(runtime_provider=lambda: runtime)

    assert isinstance(adapter, PeerCaptureVadSinkAdapter)
    assert adapter.runtime_provider() is runtime
