from __future__ import annotations

from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters.audio_test_sources import (
    LoopbackAudioTestSourceOpener,
    MicrophoneAudioTestSourceOpener,
)
from puripuly_heart.app.ports.audio_test import AudioTestSourceUnavailable
from puripuly_heart.core.audio.source import MicrophoneTestRouteObservation
from puripuly_heart.core.peer_capture import (
    PeerCaptureResolvedTarget,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)


def _route(*, attempt: bool = True, error: str | None = None) -> MicrophoneTestRouteObservation:
    return MicrophoneTestRouteObservation(
        saved_host_api="Windows WASAPI",
        actual_host_api="Windows WASAPI",
        requested_device="Desk Mic",
        hostapi_index=2,
        resolved_device_idx=7 if attempt else None,
        resolved_device_name="Desk Mic" if attempt else None,
        resolution_exception_class=error,
        resolution_exception_message=None,
        should_attempt_open=attempt,
    )


def _microphone(route, source_factory, logs=None) -> MicrophoneAudioTestSourceOpener:
    return MicrophoneAudioTestSourceOpener(
        saved_host_api="Windows WASAPI",
        requested_device="Desk Mic",
        internal_channels=1,
        route_observer=lambda **_kwargs: route,
        channel_decision=lambda **_kwargs: SimpleNamespace(preferred_capture_channels=2),
        source_factory=source_factory,
        log_sink=(logs if logs is not None else []).append,
    )


@pytest.mark.asyncio
async def test_microphone_route_miss_is_unavailable_without_opening() -> None:
    opened: list[object] = []

    with pytest.raises(AudioTestSourceUnavailable) as raised:
        await _microphone(_route(attempt=False, error="LookupError"), opened.append).open(
            on_acquired=lambda _source: None
        )

    assert (raised.value.reason, raised.value.cause) == ("input_route_unavailable", "LookupError")
    assert opened == []


@pytest.mark.asyncio
async def test_microphone_falls_back_to_internal_channels_and_reports_opened_device() -> None:
    calls: list[int] = []
    logs: list[str] = []

    class Source:
        def opened_input_metadata(self) -> dict[str, object]:
            return {"name": "Desk Mic (2- USB)", "host_api": "Windows WASAPI", "channels": 1}

    def factory(**kwargs):
        calls.append(kwargs["channels"])
        assert kwargs["device"] == 7
        if kwargs["channels"] == 2:
            raise OSError("invalid channel count")
        return Source()

    opened = await _microphone(_route(), factory, logs).open(on_acquired=lambda _source: None)

    assert calls == [2, 1]
    assert opened.device == {
        "kind": "input_device",
        "name": "Desk Mic (2- USB)",
        "host_api": "Windows WASAPI",
        "default_fallback": False,
    }
    assert any("channel fallback" in line for line in logs)


class Resolver:
    def __init__(self, resolution: PeerCaptureTargetResolution) -> None:
        self.resolution = resolution
        self.targets: list[PeerCaptureTargetIntent] = []

    async def resolve(self, target: PeerCaptureTargetIntent) -> PeerCaptureTargetResolution:
        self.targets.append(target)
        return self.resolution


@pytest.mark.asyncio
async def test_loopback_unresolved_target_is_unavailable() -> None:
    target = PeerCaptureTargetIntent(kind="process", process_kind="vrchat")
    resolver = Resolver(
        PeerCaptureTargetResolution(
            status=PeerCaptureTargetStatus.UNAVAILABLE,
            reason="process_not_running",
        )
    )
    opened: list[object] = []

    with pytest.raises(AudioTestSourceUnavailable) as raised:
        await LoopbackAudioTestSourceOpener(
            target=target,
            output_device="",
            target_resolver=resolver,
            source_factory=lambda *args: opened.append(args),
        ).open(on_acquired=lambda _source: None)

    assert (raised.value.reason, raised.value.cause) == (
        "target_unavailable",
        "process_not_running",
    )
    assert resolver.targets == [target]
    assert opened == []


@pytest.mark.asyncio
async def test_loopback_opens_resolved_targets_with_the_live_target_and_device_facts() -> None:
    device_target = PeerCaptureTargetIntent(kind="named_output_device", device_name="Headset")
    resolved = PeerCaptureResolvedTarget(intent=device_target)
    calls: list[tuple[object, str]] = []
    source = SimpleNamespace(resolved_device_name="Speakers", used_default_fallback=True)

    def factory(target: PeerCaptureResolvedTarget, output_device: str) -> object:
        calls.append((target, output_device))
        return source

    opened = await LoopbackAudioTestSourceOpener(
        target=device_target,
        output_device="Headset",
        target_resolver=Resolver(
            PeerCaptureTargetResolution(status=PeerCaptureTargetStatus.RESOLVED, target=resolved)
        ),
        source_factory=factory,
    ).open(on_acquired=lambda _source: None)

    assert calls == [(resolved, "Headset")]
    assert opened.source is source
    assert opened.device == {
        "kind": "output_device",
        "name": "Speakers",
        "default_fallback": True,
    }

    process_target = PeerCaptureTargetIntent(
        kind="process",
        process_kind="vrchat",
        executable_basename="VRChat.exe",
    )
    opened = await LoopbackAudioTestSourceOpener(
        target=process_target,
        output_device="",
        target_resolver=Resolver(
            PeerCaptureTargetResolution(
                status=PeerCaptureTargetStatus.RESOLVED,
                target=PeerCaptureResolvedTarget(intent=process_target),
            )
        ),
        source_factory=lambda *_args: object(),
    ).open(on_acquired=lambda _source: None)

    assert opened.device["kind"] == "process"
    assert opened.device["name"] == "VRChat.exe"
