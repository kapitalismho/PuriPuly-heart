from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.app.adapters.peer_capture_audio_loop import (
    PeerCaptureAudioLoopAdapter,
)
from puripuly_heart.app.wiring import create_peer_capture_audio_loop_adapter


@pytest.mark.asyncio
async def test_adapter_forwards_loop_inputs_with_peer_diagnostic_effects() -> None:
    runner_calls: list[dict[str, object]] = []
    logs: list[str] = []
    detailed = [False]
    source = object()
    vad = object()
    sink = object()

    async def runner(**kwargs: object) -> None:
        runner_calls.append(dict(kwargs))

    adapter = PeerCaptureAudioLoopAdapter(
        runner=runner,
        log_detailed=logs.append,
        is_detailed_enabled=lambda: detailed[0],
    )

    await adapter(
        source=source,
        vad=vad,
        sink=sink,
        target_sample_rate_hz=24000,
    )

    assert runner_calls == [
        {
            "source": source,
            "vad": vad,
            "sink": sink,
            "target_sample_rate_hz": 24000,
            "channel_label": "peer",
            "is_detailed_enabled": runner_calls[0]["is_detailed_enabled"],
            "log_detailed": runner_calls[0]["log_detailed"],
        }
    ]
    is_detailed_enabled = runner_calls[0]["is_detailed_enabled"]
    assert callable(is_detailed_enabled)
    assert is_detailed_enabled() is False
    detailed[0] = True
    assert is_detailed_enabled() is True
    log_detailed = runner_calls[0]["log_detailed"]
    assert callable(log_detailed)
    log_detailed("[AudioDiag][AudioVadLoop][peer] probe")
    assert logs == ["[AudioDiag][AudioVadLoop][peer] probe"]


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
        log_detailed=lambda _message: None,
        is_detailed_enabled=lambda: False,
    )
    task = asyncio.create_task(adapter(source=object(), vad=object(), sink=object()))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert released.is_set()


def test_wiring_factory_composes_internal_peer_audio_loop_adapter() -> None:
    adapter = create_peer_capture_audio_loop_adapter(
        log_detailed=lambda _message: None,
        is_detailed_enabled=lambda: False,
    )

    assert isinstance(adapter, PeerCaptureAudioLoopAdapter)
    assert adapter.runner.__name__ == "run_audio_vad_loop"
