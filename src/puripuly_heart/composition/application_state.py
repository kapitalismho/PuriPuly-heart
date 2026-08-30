from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.desktop_overlay_application import (
    DesktopOverlayApplicationOwner,
)
from puripuly_heart.app.services.overlay_calibration_application import (
    OverlayCalibrationApplicationOwner,
)
from puripuly_heart.app.wiring_managed_account import ManagedAccountComponents
from puripuly_heart.app.wiring_microphone_test import MicrophoneTestRuntime
from puripuly_heart.app.wiring_runtime_pipeline import RuntimePipelineHandle
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


@dataclass(slots=True)
class ApplicationUiStateAdapter:
    config_path: Path
    settings: SettingsOwner
    pipeline: RuntimePipelineHandle
    desktop_overlay: DesktopOverlayApplicationOwner
    managed: ManagedAccountComponents
    calibration: OverlayCalibrationApplicationOwner
    microphone: Callable[[], MicrophoneTestRuntime | None]

    @property
    def compatibility_settings(self) -> AppSettingsVNext | None:
        return self.settings.canonical

    @property
    def translation_enabled(self) -> bool:
        owner = self.pipeline.translation_runtime_configuration
        return owner.snapshot().value.translation_enabled if owner is not None else False

    @property
    def translation_runtime_ready(self) -> bool | None:
        runtime = self.pipeline.llm_runtime
        return runtime.provider is not None if runtime is not None else None

    @property
    def stt_state(self) -> object | None:
        projection = self.pipeline.stt_sessions
        return projection.state("self") if projection is not None else None

    @property
    def peer_translation_eula_accepted(self) -> bool | None:
        settings = self.settings.canonical
        return (
            bool(settings.state.peer_translation.eula_accepted) if settings is not None else None
        )

    @property
    def microphone_test_active(self) -> bool:
        microphone = self.microphone()
        return microphone.active if microphone is not None else False

    @property
    def provider_name(self) -> str | None:
        settings = self.settings.canonical
        if settings is None:
            return None
        from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
            provider_llm_for_translation,
        )

        return provider_llm_for_translation(
            settings.intent.translation.model,
            settings.intent.translation.connection,
        )

    @property
    def overlay_target(self) -> str | None:
        settings = self.settings.canonical
        return str(settings.intent.overlay.target) if settings is not None else None

    @property
    def desktop_overlay_captions_locked(self) -> bool:
        return self.desktop_overlay.captions_locked

    @property
    def managed_auth_referral_bonus_applied(self) -> bool:
        return self.managed.auth.last_referral_bonus_applied is True

    @property
    def overlay_calibration(self) -> object:
        return self.calibration.current
