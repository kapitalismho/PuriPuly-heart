from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import puripuly_heart.core.audio.source as audio_source_module
from puripuly_heart.config.audio_host_api import (
    WINDOWS_DIRECTSOUND_HOST_API,
    WINDOWS_MME_HOST_API,
    WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
    WINDOWS_WASAPI_HOST_API,
)
from puripuly_heart.core.audio.source import (
    SelfMicCaptureChannelDecision,
    SoundDeviceAudioSource,
    SoundDeviceInputMetadata,
    determine_self_mic_capture_channels,
    query_sounddevice_input_metadata,
    resolve_sounddevice_input_device,
)


def _observe_microphone_test_route(**kwargs):
    assert hasattr(audio_source_module, "observe_microphone_test_route")
    return audio_source_module.observe_microphone_test_route(**kwargs)


def _microphone_test_route_observation_type():
    observation_type = getattr(audio_source_module, "MicrophoneTestRouteObservation", None)
    assert observation_type is not None
    return observation_type


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"sample_rate_hz": 0}, "sample_rate_hz"),
        ({"channels": 0}, "channels"),
        ({"max_queue_frames": 0}, "max_queue_frames"),
    ],
)
def test_sounddevice_audio_source_rejects_invalid_params(kwargs, error):
    with pytest.raises(ValueError, match=error):
        SoundDeviceAudioSource(**kwargs)


@pytest.mark.asyncio
async def test_sounddevice_callback_tracks_status_and_drops_without_logging(monkeypatch):
    stream_ref: dict[str, object] = {}
    warnings: list[str] = []
    allow_status_stringification = False

    class StatusWithoutCallbackStringification:
        def __str__(self) -> str:
            if not allow_status_stringification:
                raise AssertionError("callback must not stringify status")
            return "input-overflow"

    class FakeInputStream:
        def __init__(self, *, callback, **kwargs):
            self.callback = callback
            self.samplerate = 48000
            stream_ref["stream"] = self

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    fake_sd = SimpleNamespace(InputStream=FakeInputStream)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    def fake_warning(message, *args, **kwargs):
        _ = kwargs
        warnings.append(message % args if args else message)

    monkeypatch.setattr(
        "puripuly_heart.core.audio.source.logger.warning",
        fake_warning,
    )

    source = SoundDeviceAudioSource(sample_rate_hz=None, channels=1, max_queue_frames=1)
    try:
        stream = stream_ref["stream"]
        status = StatusWithoutCallbackStringification()
        stream.callback(np.ones((4,), dtype=np.float32), None, None, status)
        stream.callback(np.ones((4,), dtype=np.float32), None, None, None)

        assert source.callback_status_count == 1
        assert source.queue_drop_count == 1
        assert source.last_callback_status is status
        assert warnings == []

        allow_status_stringification = True
        frame = await source.frames().__anext__()

        np.testing.assert_allclose(frame.samples, np.ones((4,), dtype=np.float32))
        assert any("[AudioDiag][Drop][self]" in message for message in warnings)
        assert any("source=sounddevice" in message for message in warnings)
        assert any("callback_status_total=1" in message for message in warnings)
        assert any("queue_drop_total=1" in message for message in warnings)
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_sounddevice_callback_warning_reporting_is_rate_limited(monkeypatch) -> None:
    stream_ref: dict[str, object] = {}
    warnings: list[str] = []
    clock = SimpleNamespace(value=0.0)

    class FakeInputStream:
        def __init__(self, *, callback, **kwargs):
            _ = kwargs
            self.callback = callback
            self.samplerate = 48000
            stream_ref["stream"] = self

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    def fake_warning(message, *args, **kwargs):
        _ = kwargs
        warnings.append(message % args if args else message)

    monkeypatch.setitem(
        __import__("sys").modules, "sounddevice", SimpleNamespace(InputStream=FakeInputStream)
    )
    monkeypatch.setattr(
        audio_source_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock.value),
        raising=False,
    )
    monkeypatch.setattr(audio_source_module.logger, "warning", fake_warning)

    source = SoundDeviceAudioSource(sample_rate_hz=None, channels=1, max_queue_frames=1)

    def trigger_status_and_drop() -> None:
        stream = stream_ref["stream"]
        stream.callback(np.ones((4,), dtype=np.float32), None, None, "input-overflow")
        stream.callback(np.ones((4,), dtype=np.float32), None, None, "input-overflow")

    try:
        trigger_status_and_drop()
        await source.frames().__anext__()
        assert len(warnings) == 1
        assert "callback_status_total=2" in warnings[0]
        assert "callback_status_new=2" in warnings[0]
        assert "queue_drop_total=1" in warnings[0]
        assert "queue_drop_new=1" in warnings[0]

        trigger_status_and_drop()
        await source.frames().__anext__()
        assert len(warnings) == 1

        clock.value = 1.1
        trigger_status_and_drop()
        await source.frames().__anext__()

        assert len(warnings) == 2
        assert "callback_status_total=6" in warnings[1]
        assert "callback_status_new=4" in warnings[1]
        assert "queue_drop_total=3" in warnings[1]
        assert "queue_drop_new=2" in warnings[1]
    finally:
        await source.close()


def test_resolve_sounddevice_input_device_prefers_hostapi_default(monkeypatch):
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [{"name": "WASAPI", "default_input_device": 1}],
        query_devices=lambda: [
            {"max_input_channels": 0, "hostapi": 0, "name": "Out"},
            {"max_input_channels": 2, "hostapi": 0, "name": "Mic"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    assert resolve_sounddevice_input_device(host_api="WASAPI") == 1


def test_resolve_sounddevice_input_device_by_name(monkeypatch):
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [{"name": "ALSA", "default_input_device": 0}],
        query_devices=lambda: [
            {"max_input_channels": 2, "hostapi": 0, "name": "Mic"},
            {"max_input_channels": 0, "hostapi": 0, "name": "Out"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    assert resolve_sounddevice_input_device(device="Mic") == 0


def test_resolve_sounddevice_input_device_returns_none_when_blank() -> None:
    assert resolve_sounddevice_input_device() is None


def test_observe_microphone_test_route_true_auto_allows_system_default_without_querying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_query(*_args, **_kwargs):
        raise AssertionError("true Auto should not query PortAudio before opening system default")

    fake_sd = SimpleNamespace(
        query_hostapis=unexpected_query,
        query_devices=unexpected_query,
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    observation = _observe_microphone_test_route(saved_host_api="", requested_device="")

    assert isinstance(observation, _microphone_test_route_observation_type())
    assert observation.saved_host_api == ""
    assert observation.actual_host_api == ""
    assert observation.requested_device == ""
    assert observation.hostapi_index is None
    assert observation.resolved_device_idx is None
    assert observation.resolved_device_name is None
    assert observation.resolution_exception_class is None
    assert observation.resolution_exception_message is None
    assert observation.should_attempt_open is True
    assert observation.wasapi_auto_convert is False
    assert observation.wasapi_exclusive is False


def test_observe_microphone_test_route_uses_host_api_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [
            {"name": WINDOWS_WASAPI_HOST_API, "default_input_device": 1},
        ],
        query_devices=lambda: [
            {"max_input_channels": 0, "hostapi": 0, "name": "Out"},
            {"max_input_channels": 2, "hostapi": 0, "name": "Default Mic"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    observation = _observe_microphone_test_route(
        saved_host_api=WINDOWS_WASAPI_HOST_API,
        requested_device="",
    )

    assert observation.hostapi_index == 0
    assert observation.resolved_device_idx == 1
    assert observation.resolved_device_name == "Default Mic"
    assert observation.should_attempt_open is True
    assert observation.actual_host_api == WINDOWS_WASAPI_HOST_API
    assert observation.wasapi_auto_convert is False
    assert observation.wasapi_exclusive is False


def test_observe_microphone_test_route_host_api_default_miss_does_not_open_global_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [
            {"name": WINDOWS_WASAPI_HOST_API, "default_input_device": -1},
        ],
        query_devices=lambda: [
            {"max_input_channels": 2, "hostapi": 0, "name": "Available Mic"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    observation = _observe_microphone_test_route(
        saved_host_api=WINDOWS_WASAPI_HOST_API,
        requested_device="",
    )

    assert observation.hostapi_index == 0
    assert observation.resolved_device_idx is None
    assert observation.resolved_device_name is None
    assert observation.should_attempt_open is False
    assert observation.resolution_exception_class is None
    assert observation.resolution_exception_message is None


def test_observe_microphone_test_route_explicit_device_miss_does_not_open_global_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [{"name": WINDOWS_MME_HOST_API, "default_input_device": 0}],
        query_devices=lambda: [
            {"max_input_channels": 2, "hostapi": 0, "name": "마이크"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    observation = _observe_microphone_test_route(
        saved_host_api=WINDOWS_MME_HOST_API,
        requested_device="Missing Mic",
    )

    assert observation.saved_host_api == WINDOWS_MME_HOST_API
    assert observation.actual_host_api == WINDOWS_MME_HOST_API
    assert observation.requested_device == "Missing Mic"
    assert observation.hostapi_index == 0
    assert observation.resolved_device_idx is None
    assert observation.resolved_device_name is None
    assert observation.should_attempt_open is False
    assert observation.resolution_exception_class is None
    assert observation.resolution_exception_message is None


@pytest.mark.parametrize(
    ("host_api", "device_name"),
    [
        (WINDOWS_MME_HOST_API, "MME Mic"),
        (WINDOWS_DIRECTSOUND_HOST_API, "DirectSound Mic"),
    ],
)
def test_observe_microphone_test_route_resolves_mme_and_directsound_devices(
    monkeypatch: pytest.MonkeyPatch,
    host_api: str,
    device_name: str,
) -> None:
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [
            {"name": WINDOWS_MME_HOST_API, "default_input_device": 0},
            {"name": WINDOWS_DIRECTSOUND_HOST_API, "default_input_device": 1},
        ],
        query_devices=lambda: [
            {"max_input_channels": 2, "hostapi": 0, "name": "MME Mic"},
            {"max_input_channels": 2, "hostapi": 1, "name": "DirectSound Mic"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    observation = _observe_microphone_test_route(
        saved_host_api=host_api,
        requested_device=device_name.lower(),
    )

    assert observation.saved_host_api == host_api
    assert observation.actual_host_api == host_api
    assert observation.resolved_device_name == device_name
    assert observation.resolved_device_idx in {0, 1}
    assert observation.should_attempt_open is True
    assert observation.wasapi_auto_convert is False
    assert observation.wasapi_exclusive is False


def test_observe_microphone_test_route_maps_wasapi_compatibility_to_actual_wasapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [
            {"name": WINDOWS_WASAPI_HOST_API, "default_input_device": 2},
        ],
        query_devices=lambda: [
            {"max_input_channels": 0, "hostapi": 0, "name": "Out"},
            {"max_input_channels": 0, "hostapi": 0, "name": "Other Out"},
            {"max_input_channels": 2, "hostapi": 0, "name": "Compat Mic"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    observation = _observe_microphone_test_route(
        saved_host_api=WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
        requested_device="Compat Mic",
    )

    assert observation.saved_host_api == WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    assert observation.actual_host_api == WINDOWS_WASAPI_HOST_API
    assert observation.hostapi_index == 0
    assert observation.resolved_device_idx == 2
    assert observation.resolved_device_name == "Compat Mic"
    assert observation.should_attempt_open is True
    assert observation.wasapi_auto_convert is True
    assert observation.wasapi_exclusive is False


def test_observe_microphone_test_route_records_resolution_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_query_error():
        raise RuntimeError("portaudio unavailable")

    fake_sd = SimpleNamespace(query_hostapis=raise_query_error, query_devices=lambda: [])
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    observation = _observe_microphone_test_route(
        saved_host_api=WINDOWS_MME_HOST_API,
        requested_device="마이크",
    )

    assert observation.saved_host_api == WINDOWS_MME_HOST_API
    assert observation.actual_host_api == WINDOWS_MME_HOST_API
    assert observation.requested_device == "마이크"
    assert observation.hostapi_index is None
    assert observation.resolved_device_idx is None
    assert observation.resolved_device_name is None
    assert observation.should_attempt_open is False
    assert observation.resolution_exception_class == "RuntimeError"
    assert observation.resolution_exception_message == "portaudio unavailable"


@pytest.mark.parametrize(
    ("max_input_channels", "expected_channels"),
    [
        (1, 1),
        (2, 2),
        (8, 2),
    ],
)
def test_determine_self_mic_capture_channels_uses_positive_metadata(
    max_input_channels: int,
    expected_channels: int,
) -> None:
    metadata = SoundDeviceInputMetadata(
        device_idx=5,
        name="Mic",
        max_input_channels=max_input_channels,
        default_samplerate=48000.0,
        metadata_status="ok",
    )

    decision = determine_self_mic_capture_channels(
        device_idx=5,
        internal_channels=1,
        metadata=metadata,
    )

    assert decision == SelfMicCaptureChannelDecision(
        device_idx=5,
        internal_channels=1,
        preferred_capture_channels=expected_channels,
        metadata=metadata,
    )


@pytest.mark.parametrize(
    "metadata",
    [
        SoundDeviceInputMetadata(
            device_idx=3,
            name="No Channels",
            max_input_channels=0,
            default_samplerate=48000.0,
            metadata_status="invalid",
        ),
        SoundDeviceInputMetadata(
            device_idx=3,
            name="Missing Channels",
            max_input_channels=None,
            default_samplerate=None,
            metadata_status="unavailable",
        ),
        SoundDeviceInputMetadata(
            device_idx=3,
            name=None,
            max_input_channels=None,
            default_samplerate=None,
            metadata_status="query_failed",
            metadata_error="boom",
        ),
    ],
)
def test_determine_self_mic_capture_channels_falls_back_to_internal_channels(
    metadata: SoundDeviceInputMetadata,
) -> None:
    decision = determine_self_mic_capture_channels(
        device_idx=metadata.device_idx,
        internal_channels=1,
        metadata=metadata,
    )

    assert decision.preferred_capture_channels == 1
    assert decision.internal_channels == 1
    assert decision.metadata is metadata


def test_determine_self_mic_capture_channels_rejects_invalid_internal_channels() -> None:
    metadata = SoundDeviceInputMetadata(
        device_idx=1,
        name="Mic",
        max_input_channels=2,
        default_samplerate=48000.0,
        metadata_status="ok",
    )

    with pytest.raises(ValueError, match="internal_channels"):
        determine_self_mic_capture_channels(
            device_idx=1,
            internal_channels=0,
            metadata=metadata,
        )


def test_query_sounddevice_input_metadata_for_explicit_device(monkeypatch):
    fake_sd = SimpleNamespace(
        query_devices=lambda *args, **kwargs: [
            {"name": "Out", "max_input_channels": 0, "default_samplerate": 48000.0},
            {"name": "마이크", "max_input_channels": 2, "default_samplerate": 44100.0},
        ]
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    metadata = query_sounddevice_input_metadata(1)

    assert metadata.device_idx == 1
    assert metadata.name == "마이크"
    assert metadata.max_input_channels == 2
    assert metadata.default_samplerate == 44100.0
    assert metadata.metadata_status == "ok"
    assert metadata.metadata_error is None


def test_query_sounddevice_input_metadata_for_default_input(monkeypatch):
    query_calls: list[dict[str, object]] = []

    def fake_query_devices(*args, **kwargs):
        query_calls.append({"args": args, "kwargs": kwargs})
        return {"name": "Default Mic", "max_input_channels": 2, "default_samplerate": 48000.0}

    fake_sd = SimpleNamespace(query_devices=fake_query_devices)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    metadata = query_sounddevice_input_metadata(None)

    assert query_calls == [{"args": (), "kwargs": {"kind": "input"}}]
    assert metadata.device_idx is None
    assert metadata.name == "Default Mic"
    assert metadata.max_input_channels == 2
    assert metadata.default_samplerate == 48000.0
    assert metadata.metadata_status == "default_resolved"


def test_query_sounddevice_input_metadata_reports_invalid_explicit_device(monkeypatch):
    fake_sd = SimpleNamespace(query_devices=lambda *args, **kwargs: [])
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    metadata = query_sounddevice_input_metadata(99)

    assert metadata.device_idx == 99
    assert metadata.max_input_channels is None
    assert metadata.metadata_status == "invalid"
    assert "99" in (metadata.metadata_error or "")


def test_query_sounddevice_input_metadata_reports_query_failure(monkeypatch):
    def fake_query_devices(*args, **kwargs):
        raise RuntimeError("no devices")

    fake_sd = SimpleNamespace(query_devices=fake_query_devices)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    metadata = query_sounddevice_input_metadata(None)

    assert metadata.device_idx is None
    assert metadata.max_input_channels is None
    assert metadata.metadata_status == "query_failed"
    assert metadata.metadata_error == "no devices"


def test_resolve_sounddevice_input_device_by_index_with_hostapi(monkeypatch):
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [{"name": "WASAPI", "default_input_device": 1}],
        query_devices=lambda: [
            {"max_input_channels": 2, "hostapi": 0, "name": "Mic0"},
            {"max_input_channels": 2, "hostapi": 0, "name": "Mic1"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    assert resolve_sounddevice_input_device(host_api="WASAPI", device="1") == 1


def test_resolve_sounddevice_input_device_rejects_mismatched_index(monkeypatch):
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [
            {"name": "ALSA", "default_input_device": 0},
            {"name": "WASAPI", "default_input_device": 1},
        ],
        query_devices=lambda: [
            {"max_input_channels": 0, "hostapi": 0, "name": "Out"},
            {"max_input_channels": 2, "hostapi": 1, "name": "Mic"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    assert resolve_sounddevice_input_device(host_api="ALSA", device="1") is None


def test_resolve_sounddevice_input_device_matches_name_with_hostapi(monkeypatch):
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [{"name": "ALSA", "default_input_device": 0}],
        query_devices=lambda: [
            {"max_input_channels": 2, "hostapi": 0, "name": "Mic"},
            {"max_input_channels": 2, "hostapi": 0, "name": "Mic2"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    assert resolve_sounddevice_input_device(host_api="ALSA", device="mic2") == 1


def test_resolve_sounddevice_input_device_handles_missing_hostapi(monkeypatch):
    fake_sd = SimpleNamespace(
        query_hostapis=lambda: [{"name": "ALSA", "default_input_device": 0}],
        query_devices=lambda: [
            {"max_input_channels": 1, "hostapi": None, "name": "Mic"},
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    assert resolve_sounddevice_input_device(device="0") == 0
    assert resolve_sounddevice_input_device(host_api="ALSA", device="Mic") is None


@pytest.mark.asyncio
async def test_sounddevice_audio_source_frames_and_close(monkeypatch):
    stream_ref: dict[str, object] = {}

    class FakeInputStream:
        def __init__(self, *, samplerate, channels, dtype, callback, device, blocksize):
            _ = (channels, dtype, device, blocksize)
            self.callback = callback
            self.samplerate = samplerate or 48000
            self.started = False
            self.stopped = False
            self.closed = False
            stream_ref["stream"] = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    fake_sd = SimpleNamespace(InputStream=FakeInputStream)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    source = SoundDeviceAudioSource(sample_rate_hz=None, channels=1, max_queue_frames=1)
    try:
        stream = stream_ref["stream"]
        stream.callback(np.ones((4,), dtype=np.float32), None, None, "warn")
        stream.callback(np.ones((4,), dtype=np.float32), None, None, None)

        frame = await source.frames().__anext__()
        assert frame.sample_rate_hz == 48000
        assert frame.channels == 1
        np.testing.assert_allclose(frame.samples, np.ones((4,), dtype=np.float32))

        stopped_frames = source.frames()
        source._queue.sync_q.put_nowait(None)
        with pytest.raises(StopAsyncIteration):
            await stopped_frames.__anext__()

        source._queue.sync_q.put_nowait(None)
        await source.close()
        assert stream.stopped is True
        assert stream.closed is True

        await source.close()
        stream.callback(np.ones((2,), dtype=np.float32), None, None, None)
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_sounddevice_audio_source_exposes_format_metadata(monkeypatch):
    stream_ref: dict[str, object] = {}

    class FakeInputStream:
        def __init__(self, *, samplerate, channels, dtype, callback, device, blocksize):
            assert channels == 2
            assert dtype == "float32"
            assert device == 3
            assert blocksize == 0
            self.callback = callback
            self.samplerate = samplerate or 48000
            stream_ref["stream"] = self

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = SimpleNamespace(InputStream=FakeInputStream)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    source = SoundDeviceAudioSource(sample_rate_hz=None, channels=2, device=3)
    try:
        assert source.actual_sample_rate_hz == 48000
        assert source.requested_channels == 2
        assert source.opened_channels == 2
        assert source.frame_channels == 2

        stream = stream_ref["stream"]
        stream.callback(np.ones((4, 2), dtype=np.float32), None, None, None)

        frame = await source.frames().__anext__()
        assert frame.sample_rate_hz == 48000
        assert frame.channels == 2
        assert source.frame_channels == 2
        np.testing.assert_allclose(frame.samples, np.ones((4, 2), dtype=np.float32))
    finally:
        await source.close()


def test_sounddevice_audio_source_closes_stream_when_start_fails(monkeypatch):
    closed: list[str] = []
    stopped: list[str] = []

    class FakeInputStream:
        samplerate = 48000

        def __init__(self, **kwargs):
            _ = kwargs

        def start(self):
            raise RuntimeError("start failed")

        def stop(self):
            stopped.append("stopped")

        def close(self):
            closed.append("closed")

    fake_sd = SimpleNamespace(InputStream=FakeInputStream)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    with pytest.raises(RuntimeError, match="start failed"):
        SoundDeviceAudioSource(sample_rate_hz=None, channels=2, device=3)

    assert stopped == ["stopped"]
    assert closed == ["closed"]


def test_sounddevice_audio_source_does_not_pass_wasapi_settings_by_default(monkeypatch):
    stream_kwargs: dict[str, object] = {}

    class FakeInputStream:
        def __init__(self, **kwargs):
            stream_kwargs.update(kwargs)
            self.samplerate = 48000

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = SimpleNamespace(InputStream=FakeInputStream)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    source = SoundDeviceAudioSource()
    try:
        assert "extra_settings" not in stream_kwargs
        assert stream_kwargs["blocksize"] == 0
        assert stream_kwargs["samplerate"] is None
        assert stream_kwargs["channels"] == 1
        assert stream_kwargs["dtype"] == "float32"
    finally:
        asyncio.run(source.close())


def test_sounddevice_audio_source_passes_wasapi_auto_convert_settings(monkeypatch):
    stream_kwargs: dict[str, object] = {}
    wasapi_settings: list[object] = []

    class FakeWasapiSettings:
        def __init__(self, *, exclusive, auto_convert):
            self.exclusive = exclusive
            self.auto_convert = auto_convert
            wasapi_settings.append(self)

    class FakeInputStream:
        def __init__(self, **kwargs):
            stream_kwargs.update(kwargs)
            self.samplerate = 48000

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = SimpleNamespace(
        InputStream=FakeInputStream,
        WasapiSettings=FakeWasapiSettings,
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    source = SoundDeviceAudioSource(wasapi_auto_convert=True, wasapi_exclusive=False)
    try:
        assert len(wasapi_settings) == 1
        assert stream_kwargs["extra_settings"] is wasapi_settings[0]
        assert wasapi_settings[0].exclusive is False
        assert wasapi_settings[0].auto_convert is True
    finally:
        asyncio.run(source.close())


def test_sounddevice_audio_source_rejects_wasapi_settings_when_unavailable(monkeypatch):
    stream_created = False

    class FakeInputStream:
        def __init__(self, **kwargs):
            nonlocal stream_created
            stream_created = True
            self.samplerate = 48000

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = SimpleNamespace(InputStream=FakeInputStream)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)

    with pytest.raises(RuntimeError, match="WASAPI settings support is unavailable"):
        SoundDeviceAudioSource(wasapi_auto_convert=True)

    assert stream_created is False
