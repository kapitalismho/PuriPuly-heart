from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.app.adapters.self_capture_audio_loop import (
    SelfCaptureAudioLoopAdapter,
)

from puripuly_heart.app.wiring import create_self_capture_audio_loop_adapter


@pytest.mark.asyncio
async def test_adapter_forwards_loop_inputs_with_current_gate_and_self_activity_log() -> None:
    runner_calls: list[dict[str, object]] = []
    basic_logs: list[str] = []
    current_gate = [object()]
    source = object()
    vad = object()
    sink = object()

    async def runner(**kwargs: object) -> None:
        runner_calls.append(dict(kwargs))

    adapter = SelfCaptureAudioLoopAdapter(
        runner=runner,
        audio_gate_provider=lambda: current_gate[0],
        log_basic=basic_logs.append,
    )

    await adapter(
        source=source,
        vad=vad,
        sink=sink,
        target_sample_rate_hz=24000,
    )
    first_gate = current_gate[0]
    current_gate[0] = object()
    await adapter(
        source=source,
        vad=vad,
        sink=sink,
        target_sample_rate_hz=24000,
    )

    assert runner_calls[0]["source"] is source
    assert runner_calls[0]["vad"] is vad
    assert runner_calls[0]["sink"] is sink
    assert runner_calls[0]["target_sample_rate_hz"] == 24000
    assert runner_calls[0]["audio_gate"] is first_gate
    assert runner_calls[0]["channel_label"] == "self"
    assert runner_calls[1]["audio_gate"] is current_gate[0]
    log_basic = runner_calls[0]["log_basic"]
    assert callable(log_basic)
    log_basic("[Capture] progress channel=self state=no_frames")
    assert basic_logs == ["[Capture] progress channel=self state=no_frames"]


@pytest.mark.asyncio
async def test_adapter_propagates_cancellation_to_owned_runner_call() -> None:
    started = asyncio.Event()
    released = asyncio.Event()

    async def runner(**_kwargs: object) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    adapter = SelfCaptureAudioLoopAdapter(
        runner=runner,
        audio_gate_provider=lambda: object(),
        log_basic=lambda _message: None,
    )
    task = asyncio.create_task(adapter(source=object(), vad=object(), sink=object()))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert released.is_set()


def test_wiring_factory_composes_internal_self_audio_loop_adapter() -> None:
    adapter = create_self_capture_audio_loop_adapter(
        audio_gate_provider=lambda: None,
        log_basic=lambda _message: None,
    )

    assert isinstance(adapter, SelfCaptureAudioLoopAdapter)
    assert adapter.runner.__name__ == "run_audio_vad_loop"
