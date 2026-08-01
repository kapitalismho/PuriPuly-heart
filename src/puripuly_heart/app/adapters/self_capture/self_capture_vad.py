from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.core.self_capture import SelfCaptureSessionConfig
from puripuly_heart.core.vad.smart_turn import SmartTurnExperimentConfig

SelfCaptureVadModelPathResolver = Callable[[], Path]
SelfCaptureVadEngineFactory = Callable[..., object]
SelfCaptureVadGatingFactory = Callable[..., object]
SelfCaptureVadDetailedLog = Callable[[str], object]
SelfCaptureVadDiagnosticsEnabled = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class SelfCaptureVadAdapter:
    model_path_resolver: SelfCaptureVadModelPathResolver
    engine_factory: SelfCaptureVadEngineFactory
    gating_factory: SelfCaptureVadGatingFactory
    log_detailed: SelfCaptureVadDetailedLog
    diagnostics_enabled: SelfCaptureVadDiagnosticsEnabled
    smart_turn_config_provider: Callable[[], SmartTurnExperimentConfig] | None = None

    def __call__(self, config: SelfCaptureSessionConfig) -> object:
        smart_turn_config = (
            self.smart_turn_config_provider() if self.smart_turn_config_provider else None
        )
        return self.gating_factory(
            engine=self.engine_factory(model_path=self.model_path_resolver()),
            sample_rate_hz=config.target_sample_rate_hz,
            ring_buffer_ms=config.ring_buffer_ms,
            speech_threshold=config.vad_speech_threshold,
            hangover_ms=(
                smart_turn_config.hard_end_ms
                if smart_turn_config is not None and smart_turn_config.stage == "active"
                else config.vad_hangover_ms
            ),
            diagnostic_event_callback=lambda message: self.log_detailed(message),
            diagnostics_enabled=self.diagnostics_enabled,
            diagnostic_label="self",
        )


__all__ = [
    "SelfCaptureVadAdapter",
    "SelfCaptureVadDetailedLog",
    "SelfCaptureVadDiagnosticsEnabled",
    "SelfCaptureVadEngineFactory",
    "SelfCaptureVadGatingFactory",
    "SelfCaptureVadModelPathResolver",
]
