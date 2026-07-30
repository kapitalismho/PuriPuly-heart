from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.core.peer_capture import PeerCaptureSessionConfig

PeerCaptureVadModelPathResolver = Callable[[], Path]
PeerCaptureVadEngineFactory = Callable[..., object]
PeerCaptureVadGatingFactory = Callable[..., object]
PeerCaptureVadDetailedLog = Callable[[str], object]
PeerCaptureVadDiagnosticsEnabled = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class PeerCaptureVadAdapter:
    model_path_resolver: PeerCaptureVadModelPathResolver
    engine_factory: PeerCaptureVadEngineFactory
    gating_factory: PeerCaptureVadGatingFactory
    log_detailed: PeerCaptureVadDetailedLog
    diagnostics_enabled: PeerCaptureVadDiagnosticsEnabled

    def __call__(self, config: PeerCaptureSessionConfig) -> object:
        return self.gating_factory(
            engine=self.engine_factory(model_path=self.model_path_resolver()),
            sample_rate_hz=config.target_sample_rate_hz,
            ring_buffer_ms=config.vad_pre_roll_ms,
            speech_threshold=config.vad_speech_threshold,
            hangover_ms=config.vad_hangover_ms,
            diagnostic_event_callback=lambda message: self.log_detailed(message),
            diagnostics_enabled=self.diagnostics_enabled,
            diagnostic_label="peer",
        )


__all__ = [
    "PeerCaptureVadAdapter",
    "PeerCaptureVadDetailedLog",
    "PeerCaptureVadDiagnosticsEnabled",
    "PeerCaptureVadEngineFactory",
    "PeerCaptureVadGatingFactory",
    "PeerCaptureVadModelPathResolver",
]
