from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.audio_test import (
    AudioTestOpenedSource,
    AudioTestSourceUnavailable,
)
from puripuly_heart.core.diagnostics import exception_diagnostics, serialize_diagnostics
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.audio.source import (
    AudioSourceOpenFailure,
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
    expected_device_name: str | None = None
    allow_channel_fallback: bool = True

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
                    (
                        observation.failure_diagnostics.fields.get("exception_class")
                        if observation.failure_diagnostics is not None else None
                    ),
                    failure_diagnostics=observation.failure_diagnostics,
                )
            if (
                self.expected_device_name is not None
                and observation.resolved_device_name != self.expected_device_name
            ):
                raise AudioTestSourceUnavailable("input_route_changed")
            decision = self.channel_decision(
                device_idx=observation.resolved_device_idx,
                internal_channels=self.internal_channels,
            )
            try:
                source, recovery = self._open(observation, decision.preferred_capture_channels)
            except AudioSourceOpenFailure as exc:
                source = exc.source
                raise
            return AudioTestOpenedSource(
                source=source, device=self._device(source, observation),
                recovery_diagnostics=recovery,
            )

        try:
            return await run_owned_thread_call(open_sync)
        finally:
            if source is not None:
                on_acquired(source)

    def _open(
        self, observation: MicrophoneTestRouteObservation, channels: int
    ) -> tuple[object, tuple[ErrorDiagnostics, ...]]:
        def open_channels(count: int) -> object:
            return self.source_factory(
                sample_rate_hz=None,
                channels=count,
                device=observation.resolved_device_idx,
                wasapi_auto_convert=observation.wasapi_auto_convert,
                wasapi_exclusive=observation.wasapi_exclusive,
            )

        try:
            return open_channels(channels), ()
        except AudioSourceOpenFailure:
            raise
        except Exception as exc:
            if not self.allow_channel_fallback or channels <= self.internal_channels:
                raise
            try:
                self.log_sink(
                    "[AudioTest] channel=self capture channel fallback "
                    f"requested={channels} cause={type(exc).__name__}"
                )
            except Exception:
                pass
            recovery = exception_diagnostics(
                exc, component="audio_test", operation="audio.test", code="source_open_failed"
            )
            try:
                return open_channels(self.internal_channels), (recovery,)
            except AudioSourceOpenFailure as failure:
                failure.failure_diagnostics = exception_diagnostics(
                    ExceptionGroup("Audio test source attempts failed", [exc, failure]),
                    component="audio_test", operation="audio.test", code="source_open_failed",
                )
                raise
            except Exception as failure:
                raise ExceptionGroup("Audio test source attempts failed", [exc, failure]) from None

    @staticmethod
    def _device(source: object, observation: MicrophoneTestRouteObservation) -> dict[str, object]:
        metadata = getattr(source, "opened_input_metadata", None)
        result: dict[str, object] = {
            "kind": "input_device",
            "name": None,
            "host_api": None,
            "default_fallback": False,
            "metadata_availability": "unavailable",
        }
        if not callable(metadata):
            return result
        try:
            opened = dict(metadata())
        except Exception as exc:
            result["diagnostics"] = serialize_diagnostics(
                exception_diagnostics(
                    exc,
                    component="capture.self",
                    operation="device_metadata",
                    code="capture.self.device_metadata_failed",
                )
            )
        else:
            result.update(
                name=opened.get("name"),
                host_api=opened.get("host_api"),
                metadata_availability="observed",
            )
        return result


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
