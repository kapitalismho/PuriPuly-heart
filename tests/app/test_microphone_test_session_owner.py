from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.app.services.microphone_test import (
    MicrophoneTestSessionOwner,
    MicrophoneTestSessionRequest,
)
from puripuly_heart.core.runtime.mic_test import MicTestRuntime


@pytest.mark.asyncio
async def test_owner_starts_stops_and_owns_meter_and_audio_signature() -> None:
    prepared = 0
    capture_started = asyncio.Event()
    capture_cancelled = asyncio.Event()
    meter_values: list[float] = []

    async def prepare() -> bool:
        nonlocal prepared
        prepared += 1
        return True

    async def capture(
        generation: int,
        meter_callback,
        level_log_interval_s: float,
    ) -> None:
        assert generation == owner.runtime.generation
        assert level_log_interval_s == 2.5
        await owner.set_meter_level(
            0.4,
            meter_callback,
            generation=generation,
        )
        capture_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            capture_cancelled.set()
            raise

    owner = MicrophoneTestSessionOwner(
        prepare_capture=prepare,
        capture_session=capture,
    )
    request = MicrophoneTestSessionRequest(
        audio_signature=("wasapi", "microphone", 16000, 1),
        meter_callback=meter_values.append,
        level_log_interval_s=2.5,
    )

    assert await owner.start(request) is True
    await capture_started.wait()

    assert prepared == 1
    assert owner.active is True
    assert owner.meter_level == 0.4
    assert owner.audio_signature == ("wasapi", "microphone", 16000, 1)
    assert meter_values == [0.4]

    await owner.stop()

    assert capture_cancelled.is_set()
    assert owner.active is False
    assert owner.meter_level == 0.0


@pytest.mark.asyncio
async def test_owner_rejects_parallel_and_active_direct_capture_sessions() -> None:
    capture_started = asyncio.Event()

    async def capture(_generation: int, _meter_callback, _interval: float) -> None:
        capture_started.set()
        await asyncio.Event().wait()

    owner = MicrophoneTestSessionOwner(
        prepare_capture=lambda: asyncio.sleep(0, result=True),
        capture_session=capture,
    )
    request = MicrophoneTestSessionRequest(audio_signature=("signature",))

    assert await owner.start(request) is True
    await capture_started.wait()
    assert await owner.start(request) is False
    await owner.stop()

    generation = owner.runtime.begin_direct_capture()
    assert await owner.start(request) is False
    owner.runtime.end_direct_capture(generation)
    await owner.close()


@pytest.mark.asyncio
async def test_owner_rejects_prepare_failure_without_starting_runtime_task() -> None:
    capture_calls = 0

    async def capture(_generation: int, _meter_callback, _interval: float) -> None:
        nonlocal capture_calls
        capture_calls += 1

    owner = MicrophoneTestSessionOwner(
        prepare_capture=lambda: asyncio.sleep(0, result=False),
        capture_session=capture,
    )

    assert await owner.start(MicrophoneTestSessionRequest(audio_signature=("signature",))) is False
    assert capture_calls == 0
    assert owner.active is False


@pytest.mark.asyncio
async def test_owner_drops_stale_meter_updates_and_contains_callback_failure() -> None:
    diagnostics: list[tuple[str, dict[str, object], BaseException | None]] = []
    owner = MicrophoneTestSessionOwner(
        prepare_capture=lambda: asyncio.sleep(0, result=True),
        capture_session=lambda _generation, _meter_callback, _interval: asyncio.sleep(0),
        diagnostics_sink=lambda event, metadata, exception: diagnostics.append(
            (event, dict(metadata), exception)
        ),
    )
    generation = owner.runtime.begin_direct_capture()

    def fail(_value: float) -> None:
        raise RuntimeError("callback detail")

    await owner.set_meter_level(2.0, fail, generation=generation)
    owner.runtime.end_direct_capture(generation)
    await owner.set_meter_level(0.2, None, generation=generation)

    assert owner.meter_level == 1.0
    assert diagnostics[0][0] == "meter_callback_failed"
    assert diagnostics[0][1] == {"error_type": "RuntimeError"}
    assert isinstance(diagnostics[0][2], RuntimeError)


@pytest.mark.asyncio
async def test_owner_contains_session_failure_and_closes_runtime() -> None:
    diagnostics: list[tuple[str, dict[str, object], BaseException | None]] = []

    async def fail_capture(
        _generation: int,
        _meter_callback,
        _interval: float,
    ) -> None:
        raise RuntimeError("capture detail")

    owner = MicrophoneTestSessionOwner(
        prepare_capture=lambda: asyncio.sleep(0, result=True),
        capture_session=fail_capture,
        diagnostics_sink=lambda event, metadata, exception: diagnostics.append(
            (event, dict(metadata), exception)
        ),
    )

    assert await owner.start(MicrophoneTestSessionRequest(audio_signature=("signature",))) is True
    task = owner.runtime.session_task
    assert task is not None
    await task

    assert diagnostics[0][0] == "session_failed"
    assert diagnostics[0][1] == {"error_type": "RuntimeError"}
    assert isinstance(diagnostics[0][2], RuntimeError)

    await owner.close()

    assert owner.runtime.is_closed is True
    assert owner.meter_level == 0.0
    assert (
        await owner.start(MicrophoneTestSessionRequest(audio_signature=("after-close",))) is False
    )


@pytest.mark.asyncio
async def test_owner_retries_completed_stale_runtime_resources_before_start() -> None:
    class RecordingSource:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    runtime = MicTestRuntime()
    generation = runtime.begin_direct_capture()
    source = RecordingSource()
    assert runtime.attach_source(source, generation=generation) is True
    runtime.end_direct_capture(generation)
    owner = MicrophoneTestSessionOwner(
        prepare_capture=lambda: asyncio.sleep(0, result=True),
        capture_session=lambda _generation, _meter_callback, _interval: asyncio.sleep(0),
        runtime_factory=lambda: runtime,
    )

    assert await owner.start(MicrophoneTestSessionRequest(audio_signature=("signature",))) is True
    task = runtime.session_task
    assert task is not None
    await task

    assert source.closed == 1
    assert runtime.source is None
