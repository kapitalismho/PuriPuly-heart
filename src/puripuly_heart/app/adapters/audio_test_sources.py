from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.audio_test import (
    AudioTestOpenedSource,
    AudioTestSourceUnavailable,
)
from puripuly_heart.core.audio.source import (
    MicrophoneTestRouteObservation,
    SelfMicCaptureChannelDecision,
)
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.peer_capture import (
    PeerCaptureResolvedTarget,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolverPort,
    PeerCaptureTargetStatus,
)

MicrophoneRouteObserver = Callable[..., MicrophoneTestRouteObservation]
MicrophoneChannelDecision = Callable[..., SelfMicCaptureChannelDecision]
MicrophoneSourceFactory = Callable[..., object]
LoopbackSourceFactory = Callable[[PeerCaptureResolvedTarget, str], object]
AudioTestLogSink = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class MicrophoneAudioTestSourceOpener:
    saved_host_api: str
    requested_device: str
    internal_channels: int
    route_observer: MicrophoneRouteObserver
    channel_decision: MicrophoneChannelDecision
    source_factory: MicrophoneSourceFactory
    log_sink: AudioTestLogSink

    async def open(self, *, on_acquired: Callable[[object], None]) -> AudioTestOpenedSource:
        source: object | None = None

        def open_sync() -> AudioTestOpenedSource:
            nonlocal source
            observation = self.route_observer(
                saved_host_api=self.saved_host_api,
                requested_device=self.requested_device,
            )
            if not observation.should_attempt_open:
                raise AudioTestSourceUnavailable(
                    "input_route_unavailable",
                    observation.resolution_exception_class or "unavailable",
                )
            decision = self.channel_decision(
                device_idx=observation.resolved_device_idx,
                internal_channels=self.internal_channels,
            )
            source = self._open(observation, decision.preferred_capture_channels)
            return AudioTestOpenedSource(source=source, device=self._device(source, observation))

        try:
            return await run_owned_thread_call(open_sync)
        finally:
            if source is not None:
                on_acquired(source)

    def _open(self, observation: MicrophoneTestRouteObservation, channels: int) -> object:
        def open_channels(count: int) -> object:
            return self.source_factory(
                sample_rate_hz=None,
                channels=count,
                device=observation.resolved_device_idx,
                wasapi_auto_convert=observation.wasapi_auto_convert,
                wasapi_exclusive=observation.wasapi_exclusive,
            )

        try:
            return open_channels(channels)
        except Exception as exc:
            if channels <= self.internal_channels:
                raise
            self.log_sink(
                "[AudioTest] channel=self capture channel fallback "
                f"requested={channels} cause={type(exc).__name__}"
            )
            return open_channels(self.internal_channels)

    @staticmethod
    def _device(source: object, observation: MicrophoneTestRouteObservation) -> dict[str, object]:
        metadata = getattr(source, "opened_input_metadata", None)
        try:
            opened = dict(metadata()) if callable(metadata) else {}
        except Exception:
            opened = {}
        return {
            "kind": "input_device",
            "name": opened.get("name") or observation.resolved_device_name,
            "host_api": opened.get("host_api") or observation.actual_host_api or None,
            "default_fallback": False,
        }


@dataclass(frozen=True, slots=True)
class LoopbackAudioTestSourceOpener:
    target: PeerCaptureTargetIntent
    output_device: str
    target_resolver: PeerCaptureTargetResolverPort
    source_factory: LoopbackSourceFactory

    async def open(self, *, on_acquired: Callable[[object], None]) -> AudioTestOpenedSource:
        resolution = await self.target_resolver.resolve(self.target)
        if resolution.status is not PeerCaptureTargetStatus.RESOLVED or resolution.target is None:
            raise AudioTestSourceUnavailable(
                "target_unavailable",
                resolution.reason or "unavailable",
            )
        source: object | None = None

        def open_sync() -> AudioTestOpenedSource:
            nonlocal source
            source = self.source_factory(resolution.target, self.output_device)
            return AudioTestOpenedSource(source=source, device=self._device(source))

        try:
            return await run_owned_thread_call(open_sync)
        finally:
            if source is not None:
                on_acquired(source)

    def _device(self, source: object) -> dict[str, object]:
        if self.target.kind == "process":
            return {
                "kind": "process",
                "name": self.target.executable_basename or self.target.process_kind,
                "process_kind": self.target.process_kind,
                "default_fallback": False,
            }
        return {
            "kind": "output_device",
            "name": getattr(source, "resolved_device_name", None)
            or self.target.device_name
            or self.output_device
            or None,
            "default_fallback": bool(getattr(source, "used_default_fallback", False)),
        }


__all__ = [
    "LoopbackAudioTestSourceOpener",
    "MicrophoneAudioTestSourceOpener",
]
