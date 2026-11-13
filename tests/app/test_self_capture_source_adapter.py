from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pytest

from puripuly_heart.app.adapters.self_capture.self_capture_source import SelfCaptureSourceAdapter
from puripuly_heart.config.audio_host_api import (
    WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
    normalize_input_host_api,
)
from puripuly_heart.core.audio.source import (
    AudioSourceOpenFailure,
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
            log_diagnostic=lambda message, level=logging.INFO: self.logs.append((level, message)),
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
    assert any(level == logging.WARNING for level, _message in harness.logs)
    assert all("device query failed" not in message for _level, message in harness.logs)


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
    assert all(
        f"open failed {attempt}" not in message
        for attempt in range(1, 4)
        for message in error_messages
    )


def test_partial_native_open_cleanup_debt_prevents_mono_and_route_fallbacks():
    calls = []
    partial_source = object()
    failure = AudioSourceOpenFailure(partial_source, OSError("retained native close failure"))

    def create_source(**kwargs):
        calls.append(kwargs)
        raise failure

    harness = AdapterHarness(
        resolve_device=lambda **_kwargs: 7,
        source_factory=create_source,
        preferred_channels=2,
    )
    with pytest.raises(AudioSourceOpenFailure) as raised:
        harness.adapter()(_config())
    assert raised.value is failure
    assert len(calls) == 1 and calls[0]["channels"] == 2
    assert harness.wrapped == []


def test_all_microphone_causes_survive_mono_routes_and_failed_logging() -> None:
    from dataclasses import replace

    from puripuly_heart.core.diagnostics import serialize_diagnostics

    calls = []

    def fail_source(**kwargs):
        calls.append(kwargs)
        raise OSError(len(calls), "private microphone detail")

    def fail_log(*args, **kwargs):
        raise RuntimeError("private log detail")

    harness = AdapterHarness(
        resolve_device=lambda **kwargs: 7,
        source_factory=fail_source,
        preferred_channels=2,
    )
    adapter = replace(harness.adapter(), log_diagnostic=fail_log)
    with pytest.raises(RuntimeError, match="All microphone attempts failed") as raised:
        adapter(_config())
    diagnostics = raised.value.failure_diagnostics
    assert len(calls) == 6
    assert [cause.fields["errno"] for cause in diagnostics.causes] == list(range(1, 7))
    assert [cause.fields["capture_attempt"] for cause in diagnostics.causes] == [
        "primary",
        "primary_mono_retry",
        "name_fallback",
        "name_fallback_mono_retry",
        "system_default",
        "system_default_mono_retry",
    ]
    assert "private" not in repr(serialize_diagnostics(diagnostics))


def test_successful_microphone_fallback_remains_success_when_logging_fails() -> None:
    from dataclasses import replace

    calls = []
    source = object()

    def create_source(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OSError(13, "private open detail")
        return source

    def fail_log(*args, **kwargs):
        raise RuntimeError("private log detail")

    adapter = AdapterHarness(
        resolve_device=lambda **kwargs: 7,
        source_factory=create_source,
        preferred_channels=2,
    ).adapter()
    adapter = replace(adapter, log_diagnostic=fail_log)
    assert adapter(_config()) == ("wrapped", source)
    assert len(calls) == 2
    assert adapter.last_open_recovered is True
    assert adapter.last_open_diagnostics.causes[0].fields["errno"] == 13
