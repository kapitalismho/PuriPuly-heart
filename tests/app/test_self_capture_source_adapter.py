from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pytest

from puripuly_heart.app.adapters.self_capture_source import SelfCaptureSourceAdapter
from puripuly_heart.app.wiring import create_self_capture_source_adapter
from puripuly_heart.config.audio_host_api import (
    WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
    normalize_input_host_api,
)
from puripuly_heart.core.audio.source import (
    SelfMicCaptureChannelDecision,
    SoundDeviceInputMetadata,
)
from puripuly_heart.core.self_capture import SelfCaptureSessionConfig


def _config(
    *,
    host_api: str = WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
    device: str = "Compat Mic",
) -> SelfCaptureSessionConfig:
    return SelfCaptureSessionConfig(
        provider_id="deepgram",
        provider_signature=("provider",),
        runtime_signature=("runtime",),
        capture_signature=("capture",),
        target_sample_rate_hz=16000,
        input_host_api=host_api,
        input_device=device,
        internal_channels=1,
    )


def _decision(
    *,
    device_idx: int | None,
    preferred_channels: int = 1,
) -> SelfMicCaptureChannelDecision:
    return SelfMicCaptureChannelDecision(
        device_idx=device_idx,
        internal_channels=1,
        preferred_capture_channels=preferred_channels,
        metadata=SoundDeviceInputMetadata(
            device_idx=device_idx,
            name="Compat Mic",
            max_input_channels=preferred_channels,
            default_samplerate=48000.0,
            metadata_status="ok",
        ),
    )


@dataclass
class AdapterHarness:
    resolve_device: object
    source_factory: object
    preferred_channels: int = 1
    logs: list[tuple[int, str]] = field(default_factory=list)
    wrapped: list[object] = field(default_factory=list)

    def adapter(self) -> SelfCaptureSourceAdapter:
        return SelfCaptureSourceAdapter(
            normalize_host_api=normalize_input_host_api,
            resolve_device=self.resolve_device,
            channel_decision=lambda *, device_idx, internal_channels: _decision(
                device_idx=device_idx,
                preferred_channels=self.preferred_channels,
            ),
            source_factory=self.source_factory,
            log_detailed=lambda message, level=logging.INFO: self.logs.append((level, message)),
            wrap_source=self._wrap_source,
        )

    def _wrap_source(self, source: object) -> object:
        self.wrapped.append(source)
        return ("wrapped", source)


def test_adapter_contains_device_resolution_failure_and_opens_default_index() -> None:
    source_calls: list[dict[str, object]] = []
    source = object()

    def fail_resolution(**_kwargs) -> int:
        raise RuntimeError("device query failed")

    def create_source(**kwargs) -> object:
        source_calls.append(dict(kwargs))
        return source

    harness = AdapterHarness(
        resolve_device=fail_resolution,
        source_factory=create_source,
    )

    result = harness.adapter()(_config())

    assert result == ("wrapped", source)
    assert harness.wrapped == [source]
    assert source_calls == [
        {
            "sample_rate_hz": None,
            "channels": 1,
            "device": None,
            "wasapi_auto_convert": True,
            "wasapi_exclusive": False,
        }
    ]
    assert any(
        level == logging.WARNING and "device query failed" in message
        for level, message in harness.logs
    )


def test_adapter_exhausts_primary_name_and_system_default_in_order() -> None:
    source_calls: list[dict[str, object]] = []
    resolve_calls: list[tuple[str, str]] = []

    def resolve_device(*, host_api: str, device: str) -> int:
        resolve_calls.append((host_api, device))
        return 7

    def fail_source(**kwargs) -> object:
        source_calls.append(dict(kwargs))
        raise RuntimeError(f"open failed {len(source_calls)}")

    harness = AdapterHarness(
        resolve_device=resolve_device,
        source_factory=fail_source,
    )

    with pytest.raises(RuntimeError, match="All microphone attempts failed"):
        harness.adapter()(_config())

    assert resolve_calls == [
        ("Windows WASAPI", "Compat Mic"),
        ("", "Compat Mic"),
    ]
    assert [
        (
            call["device"],
            call["wasapi_auto_convert"],
            call["wasapi_exclusive"],
        )
        for call in source_calls
    ] == [
        (7, True, False),
        (7, False, False),
        (None, False, False),
    ]
    assert harness.wrapped == []
    error_messages = [message for level, message in harness.logs if level == logging.ERROR]
    assert len(error_messages) == 3
    assert "Microphone open detail" in error_messages[0]
    assert "Fallback microphone detail" in error_messages[1]
    assert "System default microphone detail" in error_messages[2]


def test_adapter_uses_requested_channels_when_source_metadata_is_invalid() -> None:
    class InvalidMetadataSource:
        @property
        def opened_channels(self) -> int:
            raise ValueError("invalid opened channels")

        frame_channels = "invalid"
        actual_sample_rate_hz = "invalid"

    harness = AdapterHarness(
        resolve_device=lambda **_kwargs: 4,
        source_factory=lambda **_kwargs: InvalidMetadataSource(),
        preferred_channels=2,
    )

    harness.adapter()(_config())

    format_log = next(
        message for _level, message in harness.logs if "Microphone capture format" in message
    )
    assert "requested_channels=2" in format_log
    assert "opened_channels=2" in format_log
    assert "frame_channels=2" in format_log
    assert "actual_sample_rate_hz=None" in format_log


def test_wiring_factory_composes_internal_self_capture_source_adapter() -> None:
    adapter = create_self_capture_source_adapter(
        log_detailed=lambda *_args, **_kwargs: None,
        wrap_source=lambda source: source,
    )

    assert isinstance(adapter, SelfCaptureSourceAdapter)
    assert adapter.normalize_host_api is normalize_input_host_api
    assert adapter.resolve_device.__name__ == "resolve_sounddevice_input_device"
    assert adapter.channel_decision.__name__ == "determine_self_mic_capture_channels"
    assert adapter.source_factory.__name__ == "SoundDeviceAudioSource"
