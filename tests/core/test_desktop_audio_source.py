from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import puripuly_heart.core.audio.desktop_source as desktop_source_module
from puripuly_heart.core.audio.desktop_source import (
    DesktopLoopbackAudioSource,
    DesktopLoopbackDeviceResolver,
)


@pytest.mark.asyncio
async def test_desktop_loopback_source_yields_float32_frames(monkeypatch):
    stream_ref: dict[str, object] = {}
    manager_ref: dict[str, object] = {}

    class FakeStream:
        def __init__(self, *, stream_callback, **kwargs):
            self.stream_callback = stream_callback
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            self.closed = False
            stream_ref["stream"] = self

        def start_stream(self):
            self.started = True

        def stop_stream(self):
            self.stopped = True

        def close(self):
            self.closed = True

    class FakePyAudioManager:
        def __init__(self):
            self.terminated = False
            manager_ref["manager"] = self

        def get_loopback_device_info_generator(self):
            yield {
                "index": 7,
                "name": "Headphones (Loopback)",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }

        def get_default_wasapi_loopback(self):
            return {
                "index": 7,
                "name": "Headphones (Loopback)",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }

        def open(self, **kwargs):
            return FakeStream(**kwargs)

        def terminate(self):
            self.terminated = True

    fake_pyaudio = SimpleNamespace(
        PyAudio=FakePyAudioManager,
        paContinue=0,
        paFloat32=1,
    )

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setitem(__import__("sys").modules, "pyaudiowpatch", fake_pyaudio)

    source = DesktopLoopbackAudioSource(device_name="Headphones (Loopback)")
    stream = stream_ref["stream"]
    samples = np.array([0.25, -0.25, 0.5, -0.5], dtype=np.float32)
    stream.stream_callback(samples.tobytes(), 2, None, 0)

    frame = await source.frames().__anext__()
    assert source.resolved_device_name == "Headphones (Loopback)"
    assert source.resolved_device_index == 7
    assert source.resolved_channels == 2
    assert source.actual_sample_rate_hz == 48000
    assert source.used_default_fallback is False
    assert frame.sample_rate_hz == 48000
    assert frame.channels == 2
    assert frame.samples.dtype == np.float32
    assert frame.samples.ndim == 1
    np.testing.assert_allclose(frame.samples, samples)
    assert frame.capture is not None
    assert frame.capture.capture_epoch == 0
    assert frame.capture.callback_sequence == 0
    assert frame.capture.source_start_sample == 0
    assert frame.capture.source_end_sample == 2

    await source.close()
    assert stream.started is True
    assert stream.stopped is True
    assert stream.closed is True
    assert manager_ref["manager"].terminated is True


@pytest.mark.asyncio
async def test_desktop_loopback_source_exposes_default_fallback_metadata(monkeypatch):
    stream_ref: dict[str, object] = {}

    class FakeStream:
        def __init__(self, *, stream_callback, **kwargs):
            self.stream_callback = stream_callback
            stream_ref["stream"] = self

        def start_stream(self):
            return None

        def stop_stream(self):
            return None

        def close(self):
            return None

    class FakePyAudioManager:
        def get_loopback_device_info_generator(self):
            yield {
                "index": 10,
                "name": "Default Speakers [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }

        def get_default_wasapi_loopback(self):
            return {
                "index": 10,
                "name": "Default Speakers [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }

        def open(self, **kwargs):
            return FakeStream(**kwargs)

        def terminate(self):
            return None

    fake_pyaudio = SimpleNamespace(PyAudio=FakePyAudioManager, paContinue=0, paFloat32=1)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setitem(__import__("sys").modules, "pyaudiowpatch", fake_pyaudio)

    source = DesktopLoopbackAudioSource(device_name="Missing Device")
    try:
        assert source.resolved_device_name == "Default Speakers [Loopback]"
        assert source.used_default_fallback is True
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_desktop_loopback_callback_tracks_status_and_drops_without_logging(monkeypatch):
    stream_ref: dict[str, object] = {}
    reports: list[tuple[tuple[object, ...], int]] = []
    allow_status_stringification = False

    class StatusWithoutCallbackStringification:
        def __str__(self) -> str:
            if not allow_status_stringification:
                raise AssertionError("callback must not stringify status")
            return "loopback-overflow"

    class FakeStream:
        def __init__(self, *, stream_callback, **kwargs):
            self.stream_callback = stream_callback
            stream_ref["stream"] = self

        def start_stream(self):
            return None

        def stop_stream(self):
            return None

        def close(self):
            return None

    class FakePyAudioManager:
        def get_loopback_device_info_generator(self):
            yield {
                "index": 10,
                "name": "Steam Streaming Speakers [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }

        def get_default_wasapi_loopback(self):
            return {
                "index": 10,
                "name": "Steam Streaming Speakers [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }

        def open(self, **kwargs):
            return FakeStream(**kwargs)

        def terminate(self):
            return None

    continue_flag = 123
    fake_pyaudio = SimpleNamespace(
        PyAudio=FakePyAudioManager,
        paContinue=continue_flag,
        paFloat32=1,
    )
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setitem(__import__("sys").modules, "pyaudiowpatch", fake_pyaudio)

    def capture_report(_logger, _message, *args, level):
        reports.append((args, level))

    monkeypatch.setattr(desktop_source_module, "emit_basic_log", capture_report)

    source = DesktopLoopbackAudioSource(
        device_name="Steam Streaming Speakers [Loopback]", max_queue_frames=1
    )
    try:
        stream = stream_ref["stream"]
        samples = np.ones(4, dtype=np.float32).tobytes()
        status = StatusWithoutCallbackStringification()
        normal_result = stream.stream_callback(samples, 2, None, status)
        overflow_result = stream.stream_callback(samples, 2, None, 0)

        assert normal_result == (None, continue_flag)
        assert overflow_result == (None, continue_flag)
        assert source.callback_status_count == 1
        assert source.queue_drop_count == 1
        assert source.last_callback_status is status
        assert reports == []

        allow_status_stringification = True
        frame = await source.frames().__anext__()

        np.testing.assert_allclose(frame.samples, np.ones(4, dtype=np.float32))
        assert reports == [((1, 1, 1, 1), logging.WARNING)]
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_desktop_loopback_callback_warning_reporting_is_rate_limited(monkeypatch) -> None:
    stream_ref: dict[str, object] = {}
    reports: list[tuple[tuple[object, ...], int]] = []
    clock = SimpleNamespace(value=0.0)

    class FakeStream:
        def __init__(self, *, stream_callback, **kwargs):
            _ = kwargs
            self.stream_callback = stream_callback
            stream_ref["stream"] = self

        def start_stream(self):
            return None

        def stop_stream(self):
            return None

        def close(self):
            return None

    class FakePyAudioManager:
        def get_loopback_device_info_generator(self):
            yield {
                "index": 10,
                "name": "Steam Streaming Speakers [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }

        def get_default_wasapi_loopback(self):
            return {
                "index": 10,
                "name": "Steam Streaming Speakers [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }

        def open(self, **kwargs):
            return FakeStream(**kwargs)

        def terminate(self):
            return None

    def capture_report(_logger, _message, *args, level):
        reports.append((args, level))

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setitem(
        __import__("sys").modules,
        "pyaudiowpatch",
        SimpleNamespace(PyAudio=FakePyAudioManager, paContinue=123, paFloat32=1),
    )
    monkeypatch.setattr(
        desktop_source_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock.value),
        raising=False,
    )
    monkeypatch.setattr(desktop_source_module, "emit_basic_log", capture_report)

    source = DesktopLoopbackAudioSource(
        device_name="Steam Streaming Speakers [Loopback]", max_queue_frames=1
    )

    def trigger_status_and_drop() -> None:
        stream = stream_ref["stream"]
        samples = np.ones(4, dtype=np.float32).tobytes()
        stream.stream_callback(samples, 2, None, "loopback-overflow")
        stream.stream_callback(samples, 2, None, "loopback-overflow")

    try:
        trigger_status_and_drop()
        await source.frames().__anext__()
        assert reports == [((2, 2, 1, 1), logging.WARNING)]

        trigger_status_and_drop()
        await source.frames().__anext__()
        assert len(reports) == 1

        clock.value = 1.1
        trigger_status_and_drop()
        await source.frames().__anext__()

        assert reports == [
            ((2, 2, 1, 1), logging.WARNING),
            ((6, 4, 3, 2), logging.WARNING),
        ]
    finally:
        await source.close()


def test_desktop_loopback_source_falls_back_to_default_output_when_saved_device_missing():
    resolver = DesktopLoopbackDeviceResolver(
        devices=["Default Speakers"],
        default_device="Default Speakers",
    )

    resolved = resolver.resolve(saved_device_name="Missing Headphones")

    assert resolved == "Default Speakers"


def test_desktop_loopback_source_prefers_exact_saved_device_match():
    resolver = DesktopLoopbackDeviceResolver(
        devices=["Default Speakers", "Headphones (Loopback)"],
        default_device="Default Speakers",
    )

    resolved = resolver.resolve(saved_device_name="Headphones (Loopback)")

    assert resolved == "Headphones (Loopback)"


@pytest.mark.asyncio
async def test_desktop_loopback_source_uses_output_channel_count_when_input_channels_missing(
    monkeypatch,
):
    stream_ref: dict[str, object] = {}

    class FakeStream:
        def __init__(self, *, stream_callback, **kwargs):
            self.stream_callback = stream_callback
            self.kwargs = kwargs
            stream_ref["stream"] = self

        def start_stream(self):
            return None

        def stop_stream(self):
            return None

        def close(self):
            return None

    class FakePyAudioManager:
        def get_loopback_device_info_generator(self):
            yield {
                "index": 11,
                "name": "Speakers (Loopback)",
                "maxOutputChannels": 6,
                "defaultSampleRate": 48000.0,
            }

        def get_default_wasapi_loopback(self):
            return {
                "index": 11,
                "name": "Speakers (Loopback)",
                "maxOutputChannels": 6,
                "defaultSampleRate": 48000.0,
            }

        def open(self, **kwargs):
            return FakeStream(**kwargs)

        def terminate(self):
            return None

    fake_pyaudio = SimpleNamespace(
        PyAudio=FakePyAudioManager,
        paContinue=0,
        paFloat32=1,
    )

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setitem(__import__("sys").modules, "pyaudiowpatch", fake_pyaudio)

    source = DesktopLoopbackAudioSource(device_name="Speakers (Loopback)")
    assert source.resolved_device_name == "Speakers (Loopback)"
    assert stream_ref["stream"].kwargs["channels"] == 6
    await source.close()
