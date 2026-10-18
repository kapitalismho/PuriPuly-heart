from __future__ import annotations

import asyncio

import numpy as np
import pytest

from puripuly_heart.app.adapters.audio_test_capture import AudioTestCaptureAdapter
from puripuly_heart.app.ports.audio_test import (
    AudioTestCaptureRequest,
    AudioTestOpenedSource,
    AudioTestSourceUnavailable,
)
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.runtime.audio_test import AudioTestCaptureRuntime


def _frame() -> AudioFrameF32:
    return AudioFrameF32(samples=np.zeros(160, dtype=np.float32), sample_rate_hz=16000)


class ScriptedSource:
    def __init__(self, frames: int | None, *, stall: bool = False) -> None:
        self.frames_left = frames
        self.stall = stall
        self.closed = asyncio.Event()

    async def frames(self):
        while self.frames_left is None or self.frames_left > 0:
            if self.frames_left is not None:
                self.frames_left -= 1
            yield _frame()
            await asyncio.sleep(0)
        if self.stall:
            await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed.set()


class Opener:
    def __init__(self, result: object) -> None:
        self.result = result

    async def open(self, *, on_acquired) -> AudioTestOpenedSource:
        if isinstance(self.result, BaseException):
            raise self.result
        on_acquired(self.result)
        return AudioTestOpenedSource(source=self.result, device={"name": "Mic"})


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.frames = 0
        self.ready = asyncio.Event()

    def request(
        self,
        opener: Opener,
        generation: int,
        *,
        ready_on_open: bool = False,
        frame_timeout_s: float | None = None,
        frame_error: Exception | None = None,
    ) -> AudioTestCaptureRequest:
        def on_frame(_generation: int, _frame: object) -> None:
            self.frames += 1
            if frame_error is not None:
                raise frame_error

        def on_ready(gen: int) -> None:
            self.events.append(("ready", gen, self.frames))
            self.ready.set()

        return AudioTestCaptureRequest(
            channel="self",
            generation=generation,
            opener=opener,
            ready_on_open=ready_on_open,
            frame_timeout_s=frame_timeout_s,
            opened_callback=lambda gen, device: self.events.append(("opened", gen, device)),
            ready_callback=on_ready,
            failure_callback=lambda gen, reason, cause: self.events.append(
                ("failed", gen, reason, cause)
            ),
            frame_callback=on_frame,
        )

    def failures(self) -> list[tuple[object, ...]]:
        return [event for event in self.events if event[0] == "failed"]


async def _run(request_factory, runtime: AudioTestCaptureRuntime, logs: list[str]):
    adapter = AudioTestCaptureAdapter(log_sink=logs.append)
    task = runtime.start(
        lambda generation: adapter.capture(request_factory(generation), runtime=runtime)
    )
    return task


@pytest.mark.asyncio
async def test_capture_reports_device_ready_on_first_frame_and_closes_on_stop() -> None:
    runtime = AudioTestCaptureRuntime(cancel_timeout_s=0.5)
    recorder = Recorder()
    source = ScriptedSource(frames=None)
    logs: list[str] = []

    await _run(lambda gen: recorder.request(Opener(source), gen), runtime, logs)
    await recorder.ready.wait()
    while recorder.frames < 3:
        await asyncio.sleep(0)
    generation = runtime.generation

    assert recorder.events[0] == ("opened", generation, {"name": "Mic"})
    assert recorder.events[1] == ("ready", generation, 0)
    assert runtime.source is source

    await runtime.stop()

    assert source.closed.is_set()
    assert runtime.source is None
    assert recorder.failures() == []


@pytest.mark.asyncio
async def test_loopback_style_capture_is_ready_when_opened_without_frames() -> None:
    runtime = AudioTestCaptureRuntime(cancel_timeout_s=0.5)
    recorder = Recorder()
    source = ScriptedSource(frames=0, stall=True)

    await _run(
        lambda gen: recorder.request(Opener(source), gen, ready_on_open=True),
        runtime,
        [],
    )
    await recorder.ready.wait()

    assert recorder.frames == 0
    assert recorder.failures() == []
    await runtime.stop()
    assert source.closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            AudioTestSourceUnavailable("input_route_unavailable", "LookupError"),
            ("input_route_unavailable", "LookupError"),
        ),
        (
            AudioTestSourceUnavailable("target_unavailable", "process_not_running"),
            ("target_unavailable", "process_not_running"),
        ),
        (OSError("device busy"), ("source_open_failed", "OSError")),
    ],
)
async def test_open_failures_are_reported_without_readiness_or_source(error, expected) -> None:
    runtime = AudioTestCaptureRuntime(cancel_timeout_s=0.5)
    recorder = Recorder()
    logs: list[str] = []

    task = await _run(lambda gen: recorder.request(Opener(error), gen), runtime, logs)
    await task

    assert recorder.failures() == [("failed", 1, *expected)]
    assert not recorder.ready.is_set()
    assert runtime.source is None
    assert any(f"reason={expected[0]}" in line for line in logs)


@pytest.mark.asyncio
async def test_missing_frames_time_out_and_release_the_source() -> None:
    runtime = AudioTestCaptureRuntime(cancel_timeout_s=0.5)
    recorder = Recorder()
    source = ScriptedSource(frames=0, stall=True)

    task = await _run(
        lambda gen: recorder.request(Opener(source), gen, frame_timeout_s=0.05),
        runtime,
        [],
    )
    await task

    assert recorder.failures() == [("failed", 1, "no_audio_frames", None)]
    assert source.closed.is_set()
    assert runtime.source is None


@pytest.mark.asyncio
async def test_source_end_and_frame_processing_errors_are_explicit_failures() -> None:
    ended = Recorder()
    runtime = AudioTestCaptureRuntime(cancel_timeout_s=0.5)
    task = await _run(lambda gen: ended.request(Opener(ScriptedSource(frames=2)), gen), runtime, [])
    await task
    assert ended.failures() == [("failed", 1, "capture_ended", None)]

    broken = Recorder()
    runtime = AudioTestCaptureRuntime(cancel_timeout_s=0.5)
    source = ScriptedSource(frames=None)
    task = await _run(
        lambda gen: broken.request(
            Opener(source),
            gen,
            frame_error=ValueError("capture_format_changed"),
        ),
        runtime,
        [],
    )
    await task
    assert broken.failures() == [("failed", 1, "capture_failed", "ValueError")]
    assert source.closed.is_set()


@pytest.mark.asyncio
async def test_stale_generation_closes_opened_source_without_reporting() -> None:
    runtime = AudioTestCaptureRuntime(cancel_timeout_s=0.5)
    recorder = Recorder()
    source = ScriptedSource(frames=None)
    adapter = AudioTestCaptureAdapter(log_sink=lambda _line: None)

    await adapter.capture(recorder.request(Opener(source), generation=99), runtime=runtime)

    assert source.closed.is_set()
    assert recorder.events == []
    assert runtime.source is None
