from __future__ import annotations

from pathlib import Path

from puripuly_heart.app.ports.ui_application import UiApplicationPort
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresencePort
from puripuly_heart.composition.application_runtime import (
    compose_application_runtime,
)
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks


def compose_ui_application(
    *,
    presentation: UiPresentationPort,
    config_path: Path,
    allow_stable_settings_import: bool = False,
    runtime_logging_sinks: RuntimeLoggingSinks | None = None,
    vrchat_osc_presence: VrchatOscPresencePort | None = None,
) -> UiApplicationPort:
    return compose_application_runtime(
        presentation=presentation,
        config_path=config_path,
        allow_stable_settings_import=allow_stable_settings_import,
        runtime_logging_sinks=runtime_logging_sinks,
        vrchat_osc_presence=vrchat_osc_presence,
    )
