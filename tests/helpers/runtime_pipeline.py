from __future__ import annotations

from puripuly_heart.app.wiring.wiring_runtime_pipeline import (
    RuntimePipelineResolvedInputs,
    runtime_pipeline_inputs_from_vnext,
)
from puripuly_heart.config.settings_vnext.migration import from_legacy_app_settings


def pipeline_inputs_from_legacy(settings) -> RuntimePipelineResolvedInputs:
    return runtime_pipeline_inputs_from_vnext(
        from_legacy_app_settings(settings),
        peer_translation_enabled=settings.ui.peer_translation_enabled,
    )
