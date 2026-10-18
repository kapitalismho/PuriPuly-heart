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
PeerCaptureDiagnosticLog = Callable[[str], object]
PeerCaptureSourceWrapper = Callable[[object], object]


@dataclass(frozen=True, slots=True)
class PeerCaptureSourceAdapter:
    loopback_source_factory: PeerCaptureLoopbackSourceFactory
    process_source_factory: PeerCaptureProcessSourceFactory
    process_watcher_factory: PeerCaptureProcessWatcherFactory
    pipeline_factory: PeerCapturePipelineFactory
    log_diagnostic: PeerCaptureDiagnosticLog
    wrap_source: PeerCaptureSourceWrapper

    def __call__(
        self,
        config: PeerCaptureSessionConfig,
        resolved_target: PeerCaptureResolvedTarget,
    ) -> object:
        raw_source = self.open_raw_source(resolved_target, output_device=config.output_device)
        return self.pipeline_factory(
            source=self.wrap_source(raw_source),
            target_sample_rate_hz=config.target_sample_rate_hz,
        )

    def open_raw_source(
        self,
        resolved_target: PeerCaptureResolvedTarget,
        *,
        output_device: str,
    ) -> object:
        target = resolved_target.intent
        if target.kind == "process":
            return self._open_process_source(resolved_target)

        device_name = target.device_name or output_device
        raw_source = self.loopback_source_factory(device_name=device_name)
        self.log_diagnostic(
            "[AudioDiag][Loopback][peer] "
            f"requested_device={device_name!r} "
            f"resolved_device_name={getattr(raw_source, 'resolved_device_name', None)!r} "
            f"resolved_device_index={getattr(raw_source, 'resolved_device_index', None)} "
            f"resolved_channels={getattr(raw_source, 'resolved_channels', None)} "
            f"actual_sample_rate_hz={getattr(raw_source, 'actual_sample_rate_hz', None)} "
            f"used_default_fallback={getattr(raw_source, 'used_default_fallback', None)}"
        )
        return raw_source

    def _open_process_source(self, resolved_target: PeerCaptureResolvedTarget) -> object:
        resolution = resolved_target.capture_descriptor
        identity = getattr(resolution, "identity", resolution)
        if identity is None:
            raise RuntimeError("resolved process capture requires a process identity")
        raw_source = self.process_source_factory(
            identity=identity,
            watcher=self.process_watcher_factory(),
        )
        self.log_diagnostic(
            "[AudioDiag][ProcessCapture][peer] "
            f"target_kind={resolved_target.intent.process_kind} capture=process"
        )
        return raw_source


__all__ = [
    "PeerCaptureDiagnosticLog",
    "PeerCaptureLoopbackSourceFactory",
    "PeerCapturePipelineFactory",
    "PeerCaptureProcessSourceFactory",
    "PeerCaptureProcessWatcherFactory",
    "PeerCaptureSourceAdapter",
    "PeerCaptureSourceWrapper",
]
