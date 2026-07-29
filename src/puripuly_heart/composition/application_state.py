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
    def compatibility_settings(self) -> object | None:
        return self.settings.current

    @property
    def translation_enabled(self) -> bool:
        hub = self.pipeline.hub
        return hub.translation_enabled if hub is not None else False

    @property
    def translation_runtime_ready(self) -> bool | None:
        hub = self.pipeline.hub
        return hub.llm is not None if hub is not None else None

    @property
    def stt_state(self) -> object | None:
        hub = self.pipeline.hub
        return hub.stt_session_state("self") if hub is not None else None

    @property
    def peer_translation_eula_accepted(self) -> bool | None:
        settings = self.settings.current
        return bool(settings.ui.peer_translation_eula_accepted) if settings is not None else None

    @property
    def microphone_test_active(self) -> bool:
        microphone = self.microphone()
        return microphone.active if microphone is not None else False

    @property
    def provider_name(self) -> str | None:
        settings = self.settings.current
        return settings.provider.llm.value if settings is not None else None

    @property
    def overlay_target(self) -> str | None:
        settings = self.settings.current
        return str(settings.overlay.target) if settings is not None else None

    @property
    def desktop_overlay_captions_locked(self) -> bool:
        return self.desktop_overlay.captions_locked

    @property
    def managed_auth_referral_bonus_applied(self) -> bool:
        return self.managed.auth.last_referral_bonus_applied is True

    @property
    def overlay_calibration(self) -> object:
        return self.calibration.current
