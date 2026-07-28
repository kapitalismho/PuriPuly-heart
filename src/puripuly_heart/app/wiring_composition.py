from __future__ import annotations

from collections.abc import Awaitable, Callable

from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCapturePort,
    MicrophoneTestMeterCallback,
)
from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort


def create_microphone_test_capture_adapter(
    *,
    clock: object,
    log_sink: Callable[[str], None],
    meter_sink: Callable[
        [float, MicrophoneTestMeterCallback | None, int | None],
        Awaitable[None],
    ],
    route_observer: Callable[..., object],
    channel_decision: Callable[..., object],
    source_factory: Callable[..., object],
) -> MicrophoneTestCapturePort:
    from puripuly_heart.app.adapters.microphone_test_capture import (
        MicrophoneTestCaptureAdapter,
    )

    return MicrophoneTestCaptureAdapter(
        clock=clock,
        log_sink=log_sink,
        meter_sink=meter_sink,
        route_observer=route_observer,
        channel_decision=channel_decision,
        source_factory=source_factory,
    )


def create_provider_verifier() -> ProviderVerifierPort:
    from puripuly_heart.app.adapters.provider_verifier import ProviderVerifierAdapter

    return ProviderVerifierAdapter()
