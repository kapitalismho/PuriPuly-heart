from __future__ import annotations

from puripuly_heart.app.wiring.wiring_runtime_pipeline import (
    RuntimePipelineResolvedInputs,
    runtime_pipeline_inputs_from_vnext,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


def pipeline_inputs_from_vnext(
    settings: AppSettingsVNext,
    *,
    peer_translation_enabled: bool = False,
) -> RuntimePipelineResolvedInputs:
    return runtime_pipeline_inputs_from_vnext(
        settings,
        peer_translation_enabled=peer_translation_enabled,
    )
