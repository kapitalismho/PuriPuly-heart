from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.app.adapters.peer_capture_audio_loop import (
    PeerCaptureAudioLoopAdapter,
)

from puripuly_heart.app.wiring import create_peer_capture_audio_loop_adapter


@pytest.mark.asyncio
async def test_adapter_forwards_loop_inputs_with_peer_activity_log() -> None:
    runner_calls: list[dict[str, object]] = []
    basic_logs: list[str] = []
    source = object()
    vad = object()
    sink = object()

    async def runner(**kwargs: object) -> None:
        runner_calls.append(dict(kwargs))

    adapter = PeerCaptureAudioLoopAdapter(
        runner=runner,
        log_basic=basic_logs.append,
    )

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
    assert runner_calls[0]["channel_label"] == "peer"
    log_basic = runner_calls[0]["log_basic"]
    assert callable(log_basic)
    log_basic("[Capture] progress channel=peer state=no_frames")
    assert basic_logs == ["[Capture] progress channel=peer state=no_frames"]


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

    adapter = PeerCaptureAudioLoopAdapter(
        runner=runner,
        log_basic=lambda _message: None,
    )
    task = asyncio.create_task(adapter(source=object(), vad=object(), sink=object()))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert released.is_set()


def test_wiring_factory_composes_internal_peer_audio_loop_adapter() -> None:
    adapter = create_peer_capture_audio_loop_adapter(
        log_basic=lambda _message: None,
    )

    assert isinstance(adapter, PeerCaptureAudioLoopAdapter)
    assert adapter.runner.__name__ == "run_audio_vad_loop"
