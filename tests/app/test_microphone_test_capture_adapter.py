from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from puripuly_heart.app.adapters.microphone_test_capture import (
    MicrophoneTestCaptureAdapter,
)
from puripuly_heart.app.ports.microphone_test import MicrophoneTestCaptureRequest
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.source import (
    MicrophoneTestRouteObservation,
    SelfMicCaptureChannelDecision,
    SoundDeviceInputMetadata,
)
from puripuly_heart.core.runtime.mic_test import MicTestRuntime


@dataclass
class FakeClock:
    value: float = 0.0

    def now(self) -> float:
        return self.value


class OneFrameSource:
    opened_channels = 1
    frame_channels = 1
    actual_sample_rate_hz = 16000
    queue_drop_count = 0
    callback_status_count = 0

    def __init__(self) -> None:
        self.close_calls = 0

    async def frames(self):
        yield AudioFrameF32(
            samples=np.asarray([0.0, 0.5, -0.25], dtype=np.float32),
            sample_rate_hz=16000,
        )

    async def close(self) -> None:
        self.close_calls += 1


def _route(*, should_attempt_open: bool = True) -> MicrophoneTestRouteObservation:
    return MicrophoneTestRouteObservation(
        saved_host_api="Windows WASAPI",
        actual_host_api="Windows WASAPI",
        requested_device="Microphone",
        hostapi_index=1,
        resolved_device_idx=2 if should_attempt_open else None,
        resolved_device_name="Microphone" if should_attempt_open else None,
        resolution_exception_class=None,
        resolution_exception_message=None,
        should_attempt_open=should_attempt_open,
        wasapi_auto_convert=True,
        wasapi_exclusive=False,
    )


def _decision() -> SelfMicCaptureChannelDecision:
    return SelfMicCaptureChannelDecision(
        device_idx=2,
        internal_channels=1,
        preferred_capture_channels=1,
        metadata=SoundDeviceInputMetadata(
            device_idx=2,
            name="Microphone",
            max_input_channels=1,
            default_samplerate=48000.0,
            metadata_status="ok",
        ),
    )


@pytest.mark.asyncio
async def test_adapter_captures_frames_updates_meter_logs_and_closes_source() -> None:
    logs: list[str] = []
    meter: list[tuple[float, int | None]] = []
    source = OneFrameSource()

    async def set_meter(
        value: float,
        _callback,
        generation: int | None,
    ) -> None:
        meter.append((value, generation))

    adapter = MicrophoneTestCaptureAdapter(
        clock=FakeClock(),
        log_sink=logs.append,
        meter_sink=set_meter,
        route_observer=lambda **_kwargs: _route(),
        channel_decision=lambda **_kwargs: _decision(),
        source_factory=lambda **_kwargs: source,
    )
    runtime = MicTestRuntime()

    await adapter.capture(
        MicrophoneTestCaptureRequest(
            saved_host_api="Windows WASAPI",
            requested_device="Microphone",
            internal_channels=1,
            level_log_interval_s=0.0,
        ),
        runtime=runtime,
    )

    assert meter[0][0] == 0.0
    assert meter[1][0] == 0.5
    assert meter[-1][0] == 0.0
    assert all(generation is not None for _, generation in meter)
    assert any(message.startswith("[MicTest] route ") for message in logs)
    assert any("[MicTest] open attempted=True opened=True" in message for message in logs)
    assert any("[MicTest] level " in message and "frames=1" in message for message in logs)
    assert any("[MicTest] end opened=True frames_total=1" in message for message in logs)
    assert source.close_calls == 1
    assert runtime.source is None


@pytest.mark.asyncio
async def test_adapter_reports_route_miss_without_opening_source() -> None:
    logs: list[str] = []
    meter: list[float] = []

    async def set_meter(
        value: float,
        _callback,
        _generation: int | None,
    ) -> None:
        meter.append(value)

    def fail_source(**_kwargs):
        raise AssertionError("source must not open")

    adapter = MicrophoneTestCaptureAdapter(
        clock=FakeClock(),
        log_sink=logs.append,
        meter_sink=set_meter,
        route_observer=lambda **_kwargs: _route(should_attempt_open=False),
        channel_decision=lambda **_kwargs: _decision(),
        source_factory=fail_source,
    )

    await adapter.capture(
        MicrophoneTestCaptureRequest(
            saved_host_api="Windows WASAPI",
            requested_device="Missing",
            internal_channels=1,
        ),
        runtime=MicTestRuntime(),
    )

    assert meter == [0.0, 0.0]
    assert any("[MicTest] open attempted=False opened=False" in message for message in logs)
    assert any("[MicTest] level " in message and "frames=0" in message for message in logs)
    assert any("[MicTest] end opened=False frames_total=0" in message for message in logs)


@pytest.mark.asyncio
async def test_adapter_contains_eligible_source_open_failure_and_clears_meter() -> None:
    raw_message = "bad failure usable near_silence 마이크"
    logs: list[str] = []
    meter: list[float] = []

    async def set_meter(
        value: float,
        _callback,
        _generation: int | None,
    ) -> None:
        meter.append(value)

    def fail_source(**_kwargs):
        raise RuntimeError(raw_message)

    adapter = MicrophoneTestCaptureAdapter(
        clock=FakeClock(),
        log_sink=logs.append,
        meter_sink=set_meter,
        route_observer=lambda **_kwargs: _route(),
        channel_decision=lambda **_kwargs: _decision(),
        source_factory=fail_source,
    )
    runtime = MicTestRuntime()

    await adapter.capture(
        MicrophoneTestCaptureRequest(
            saved_host_api="Windows WASAPI",
            requested_device="Microphone",
            internal_channels=1,
        ),
        runtime=runtime,
    )

    open_messages = [message for message in logs if message.startswith("[MicTest] open ")]
    end_messages = [message for message in logs if message.startswith("[MicTest] end ")]
    assert meter == [0.0, 0.0]
    assert runtime.source is None
    assert runtime.has_active_direct_capture is False
    assert len(open_messages) == 1
    assert "attempted=True opened=False" in open_messages[0]
    assert "exception_class='RuntimeError'" in open_messages[0]
    assert f"exception_message={raw_message!r}" in open_messages[0]
    assert len(end_messages) == 1
    assert "opened=False frames_total=0" in end_messages[0]
    assert "exception_class='RuntimeError'" in end_messages[0]
    assert f"exception_message={raw_message!r}" in end_messages[0]
