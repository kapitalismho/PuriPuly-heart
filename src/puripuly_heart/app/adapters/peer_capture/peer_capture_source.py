from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.core.peer_capture import (
    PeerCaptureResolvedTarget,
    PeerCaptureSessionConfig,
)

PeerCaptureLoopbackSourceFactory = Callable[..., object]
PeerCaptureProcessSourceFactory = Callable[..., object]
PeerCaptureProcessWatcherFactory = Callable[[], object]
PeerCapturePipelineFactory = Callable[..., object]
PeerCaptureDetailedLog = Callable[[str], object]
PeerCaptureSourceWrapper = Callable[[object], object]
PeerCaptureDetailedEnabled = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class PeerCaptureSourceAdapter:
    loopback_source_factory: PeerCaptureLoopbackSourceFactory
    process_source_factory: PeerCaptureProcessSourceFactory
    process_watcher_factory: PeerCaptureProcessWatcherFactory
    pipeline_factory: PeerCapturePipelineFactory
    log_detailed: PeerCaptureDetailedLog
    wrap_source: PeerCaptureSourceWrapper
    is_detailed_enabled: PeerCaptureDetailedEnabled

    def __call__(
        self,
        config: PeerCaptureSessionConfig,
        resolved_target: PeerCaptureResolvedTarget,
    ) -> object:
        target = resolved_target.intent
        if target.kind == "process":
            return self._create_process_pipeline(config, resolved_target)

        device_name = target.device_name or config.output_device
        raw_source = self.loopback_source_factory(device_name=device_name)
        self.log_detailed(
            "[AudioDiag][Loopback][peer] "
            f"requested_device={device_name!r} "
            f"resolved_device_name={getattr(raw_source, 'resolved_device_name', None)!r} "
            f"resolved_device_index={getattr(raw_source, 'resolved_device_index', None)} "
            f"resolved_channels={getattr(raw_source, 'resolved_channels', None)} "
            f"actual_sample_rate_hz={getattr(raw_source, 'actual_sample_rate_hz', None)} "
            f"used_default_fallback={getattr(raw_source, 'used_default_fallback', None)}"
        )
        return self._create_pipeline(config, raw_source)

    def _create_process_pipeline(
        self,
        config: PeerCaptureSessionConfig,
        resolved_target: PeerCaptureResolvedTarget,
    ) -> object:
        resolution = resolved_target.capture_descriptor
        identity = getattr(resolution, "identity", resolution)
        if identity is None:
            raise RuntimeError("resolved process capture requires a process identity")
        raw_source = self.process_source_factory(
            identity=identity,
            watcher=self.process_watcher_factory(),
        )
        self.log_detailed(
            "[AudioDiag][ProcessCapture][peer] "
            f"target_kind={config.capture_target.process_kind} capture=process"
        )
        return self._create_pipeline(config, raw_source)

    def _create_pipeline(
        self,
        config: PeerCaptureSessionConfig,
        raw_source: object,
    ) -> object:
        return self.pipeline_factory(
            source=self.wrap_source(raw_source),
            target_sample_rate_hz=config.target_sample_rate_hz,
            is_detailed_enabled=self.is_detailed_enabled,
            log_detailed=lambda message: self.log_detailed(message),
        )


__all__ = [
    "PeerCaptureDetailedEnabled",
    "PeerCaptureDetailedLog",
    "PeerCaptureLoopbackSourceFactory",
    "PeerCapturePipelineFactory",
    "PeerCaptureProcessSourceFactory",
    "PeerCaptureProcessWatcherFactory",
    "PeerCaptureSourceAdapter",
    "PeerCaptureSourceWrapper",
]
