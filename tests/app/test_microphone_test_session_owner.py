from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCaptureRequest,
    MicrophoneTestRuntimePort,
)
from puripuly_heart.app.services.microphone_test import (
    MicrophoneTestSelfCaptureState,
    MicrophoneTestSessionOwner,
    MicrophoneTestSessionRequest,
)
from puripuly_heart.core.runtime.mic_test import MicTestRuntime

CaptureCallback = Callable[
    [int, Callable[[float], object] | None, float],
    Awaitable[None],
]


class CallbackCapturePort:
    def __init__(self, callback: CaptureCallback) -> None:
        self.callback = callback

    async def capture(
        self,
        request: MicrophoneTestCaptureRequest,
        *,
        runtime: MicrophoneTestRuntimePort,
    ) -> None:
        assert request.generation is not None
        await self.callback(
            request.generation,
            request.meter_callback,
            request.level_log_interval_s,
        )


def _capture_request(
    generation: int,
    meter_callback: Callable[[float], object] | None,
    level_log_interval_s: float,
) -> MicrophoneTestCaptureRequest:
    return MicrophoneTestCaptureRequest(
        saved_host_api="Windows WASAPI",
        requested_device="Microphone",
        internal_channels=1,
        generation=generation,
        meter_callback=meter_callback,
        level_log_interval_s=level_log_interval_s,
    )


def _owner(
    capture_callback: CaptureCallback,
    **kwargs: Any,
) -> MicrophoneTestSessionOwner:
    return MicrophoneTestSessionOwner(
        capture_port=CallbackCapturePort(capture_callback),
        capture_request_factory=_capture_request,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_owner_starts_stops_and_owns_meter_and_audio_signature() -> None:
    self_capture_active = True
    disable_calls = 0
    capture_started = asyncio.Event()
    capture_cancelled = asyncio.Event()
    meter_values: list[float] = []
    logs: list[str] = []

    def self_capture_snapshot() -> MicrophoneTestSelfCaptureState:
        return MicrophoneTestSelfCaptureState(
            stop_required=self_capture_active,
            source_open=self_capture_active,
        )

    async def disable_self_capture() -> None:
        nonlocal disable_calls, self_capture_active
        disable_calls += 1
        self_capture_active = False

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

    owner = _owner(
        capture,
        self_capture_snapshot=self_capture_snapshot,
        disable_self_capture=disable_self_capture,
        log_sink=logs.append,
    )
    request = MicrophoneTestSessionRequest(
        audio_signature=("wasapi", "microphone", 16000, 1),
        meter_callback=meter_values.append,
        level_log_interval_s=2.5,
    )

    assert await owner.start(request) is True
    await capture_started.wait()

    assert disable_calls == 1
    assert logs == [
        "[MicTest] stt_auto_off requested=True completed=True "
        "exception_class=None exception_message=None"
    ]
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

    owner = _owner(capture)
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
async def test_owner_rejects_retained_self_capture_failure_without_starting_runtime_task() -> None:
    capture_calls = 0
    logs: list[str] = []
    close_failure = RuntimeError("close detail")

    async def capture(_generation: int, _meter_callback, _interval: float) -> None:
        nonlocal capture_calls
        capture_calls += 1

    owner = _owner(
        capture,
        self_capture_snapshot=lambda: MicrophoneTestSelfCaptureState(
            stop_required=False,
            source_open=False,
            close_exception=close_failure,
        ),
        log_sink=logs.append,
    )

    assert await owner.start(MicrophoneTestSessionRequest(audio_signature=("signature",))) is False
    assert capture_calls == 0
    assert owner.active is False
    assert logs == [
        "[MicTest] stt_auto_off requested=False completed=False "
        "exception_class='RuntimeError' exception_message='close detail'"
    ]


@pytest.mark.asyncio
async def test_owner_contains_self_capture_disable_failure() -> None:
    logs: list[str] = []

    async def fail_disable() -> None:
        raise RuntimeError("disable detail")

    owner = _owner(
        lambda _generation, _meter_callback, _interval: asyncio.sleep(0),
        self_capture_snapshot=lambda: MicrophoneTestSelfCaptureState(
            stop_required=True,
            source_open=True,
        ),
        disable_self_capture=fail_disable,
        log_sink=logs.append,
    )

    assert await owner.start(MicrophoneTestSessionRequest(audio_signature=("signature",))) is False
    assert owner.active is False
    assert logs == [
        "[MicTest] stt_auto_off requested=True completed=False "
        "exception_class='RuntimeError' exception_message='disable detail'"
    ]


@pytest.mark.asyncio
async def test_owner_rejects_source_that_remains_open_after_disable() -> None:
    snapshots = iter(
        (
            MicrophoneTestSelfCaptureState(
                stop_required=True,
                source_open=True,
            ),
            MicrophoneTestSelfCaptureState(
                stop_required=False,
                source_open=True,
            ),
        )
    )
    logs: list[str] = []

    owner = _owner(
        lambda _generation, _meter_callback, _interval: asyncio.sleep(0),
        self_capture_snapshot=lambda: next(snapshots),
        disable_self_capture=lambda: asyncio.sleep(0),
        log_sink=logs.append,
    )

    assert await owner.start(MicrophoneTestSessionRequest(audio_signature=("signature",))) is False
    assert owner.active is False
    assert logs == [
        "[MicTest] stt_auto_off requested=True completed=False "
        "exception_class='RuntimeError' "
        "exception_message='self microphone source still open after STT auto-off'"
    ]


@pytest.mark.asyncio
async def test_owner_drops_stale_meter_updates_and_contains_callback_failure() -> None:
    diagnostics: list[tuple[str, dict[str, object], BaseException | None]] = []
    owner = _owner(
        lambda _generation, _meter_callback, _interval: asyncio.sleep(0),
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

    owner = _owner(
        fail_capture,
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
    owner = _owner(
        lambda _generation, _meter_callback, _interval: asyncio.sleep(0),
        runtime_factory=lambda: runtime,
    )

    assert await owner.start(MicrophoneTestSessionRequest(audio_signature=("signature",))) is True
    task = runtime.session_task
    assert task is not None
    await task

    assert source.closed == 1
    assert runtime.source is None
