from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.app.adapters.self_capture_audio_loop import (
    SelfCaptureAudioLoopAdapter,
)

from puripuly_heart.app.wiring import create_self_capture_audio_loop_adapter


@pytest.mark.asyncio
async def test_adapter_forwards_loop_inputs_with_current_gate_and_self_diagnostics() -> None:
    runner_calls: list[dict[str, object]] = []
    logs: list[str] = []
    detailed = [False]
    current_gate = [object()]
    source = object()
    vad = object()
    sink = object()

    async def runner(**kwargs: object) -> None:
        runner_calls.append(dict(kwargs))

    adapter = SelfCaptureAudioLoopAdapter(
        runner=runner,
        audio_gate_provider=lambda: current_gate[0],
        log_detailed=logs.append,
        is_detailed_enabled=lambda: detailed[0],
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

    assert runner_calls[0] == {
        "source": source,
        "vad": vad,
        "sink": sink,
        "target_sample_rate_hz": 24000,
        "audio_gate": first_gate,
        "channel_label": "self",
        "is_detailed_enabled": runner_calls[0]["is_detailed_enabled"],
        "log_detailed": runner_calls[0]["log_detailed"],
    }
    assert runner_calls[1]["audio_gate"] is current_gate[0]
    is_detailed_enabled = runner_calls[0]["is_detailed_enabled"]
    assert callable(is_detailed_enabled)
    assert is_detailed_enabled() is False
    detailed[0] = True
    assert is_detailed_enabled() is True
    log_detailed = runner_calls[0]["log_detailed"]
    assert callable(log_detailed)
    log_detailed("[AudioDiag][AudioVadLoop][self] probe")
    assert logs == ["[AudioDiag][AudioVadLoop][self] probe"]


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
        log_detailed=lambda _message: None,
        is_detailed_enabled=lambda: False,
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
        log_detailed=lambda _message: None,
        is_detailed_enabled=lambda: False,
    )

    assert isinstance(adapter, SelfCaptureAudioLoopAdapter)
    assert adapter.runner.__name__ == "run_audio_vad_loop"
