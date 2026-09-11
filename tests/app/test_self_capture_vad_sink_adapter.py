from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.app.adapters.self_capture_vad_sink import (
    SelfCaptureVadSinkAdapter,
)

from puripuly_heart.app.wiring import create_self_capture_vad_sink_adapter


class Runtime:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.rejections: list[tuple[object, str, str]] = []
        self.failures: list[tuple[object, str]] = []

    async def handle_vad_event(self, event: object) -> None:
        self.events.append(event)

    async def reject_owned_segment(
        self,
        event: object,
        *,
        reason: str,
        outcome: str,
    ) -> None:
        self.rejections.append((event, reason, outcome))

    async def fail_owned_segment(self, event: object, *, reason: str) -> None:
        self.failures.append((event, reason))


@pytest.mark.asyncio
async def test_adapter_resolves_current_runtime_for_each_self_event() -> None:
    first = Runtime()
    second = Runtime()
    current = [first]
    adapter = SelfCaptureVadSinkAdapter(runtime_provider=lambda: current[0])
    first_event = object()
    second_event = object()

    await adapter.handle_vad_event(first_event)
    current[0] = second
    await adapter.handle_vad_event(second_event)

    assert first.events == [first_event]
    assert second.events == [second_event]


@pytest.mark.asyncio
async def test_adapter_forwards_required_recognition_terminal_operations() -> None:
    runtime = Runtime()
    adapter = SelfCaptureVadSinkAdapter(runtime_provider=lambda: runtime)
    rejected = object()
    failed = object()

    await adapter.reject_owned_segment(
        rejected,
        reason="recognition_admission_timeout",
        outcome="expired",
    )
    await adapter.fail_owned_segment(failed, reason="buffer_exhausted")

    assert runtime.rejections == [
        (rejected, "recognition_admission_timeout", "expired")
    ]
    assert runtime.failures == [(failed, "buffer_exhausted")]


@pytest.mark.asyncio
async def test_adapter_rejects_event_without_current_runtime() -> None:
    adapter = SelfCaptureVadSinkAdapter(runtime_provider=lambda: None)

    with pytest.raises(RuntimeError, match="Self VAD sink requires the Self translation owner"):
        await adapter.handle_vad_event(object())


@pytest.mark.asyncio
async def test_adapter_propagates_downstream_exception_and_cancellation() -> None:
    class FailingRuntime:
        async def handle_vad_event(self, _event: object) -> None:
            raise RuntimeError("self event failed")

    adapter = SelfCaptureVadSinkAdapter(runtime_provider=FailingRuntime)
    with pytest.raises(RuntimeError, match="self event failed"):
        await adapter.handle_vad_event(object())

    class CancellingRuntime:
        async def handle_vad_event(self, _event: object) -> None:
            raise asyncio.CancelledError

    adapter = SelfCaptureVadSinkAdapter(runtime_provider=CancellingRuntime)
    with pytest.raises(asyncio.CancelledError):
        await adapter.handle_vad_event(object())


def test_wiring_factory_composes_internal_self_vad_sink_adapter() -> None:
    runtime = Runtime()
    adapter = create_self_capture_vad_sink_adapter(runtime_provider=lambda: runtime)

    assert isinstance(adapter, SelfCaptureVadSinkAdapter)
    assert adapter.runtime_provider() is runtime
