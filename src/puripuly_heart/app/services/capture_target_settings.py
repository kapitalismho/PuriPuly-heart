from __future__ import annotations

from pathlib import Path

from puripuly_heart.app.ports.canonical_settings_persistence import (
    CanonicalSettingsPersistenceError,
)
from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.config.settings_vnext.schema import CaptureTargetIntent


class CaptureTargetSettingsError(RuntimeError):
    def __init__(self, status: object) -> None:
        self.status = str(getattr(status, "value", status))
        super().__init__(f"capture_target_settings_{self.status}")


def persist_desktop_audio_capture_target(
    path: Path,
    settings: AppSettings,
    capture_target: CaptureTargetIntent,
) -> AppSettings:
    owner = compose_settings_owner(path)
    if path.exists():
        try:
            owner.start()
        except CanonicalSettingsPersistenceError as exc:
            raise CaptureTargetSettingsError(exc.status) from None
        except Exception:
            raise CaptureTargetSettingsError("load_failed") from None
    else:
        try:
            owner.project(
                settings,
                authoritative=False,
            )
        except Exception:
            raise CaptureTargetSettingsError("migration_failed") from None
    try:
        return owner.update_capture_target(settings, capture_target)
    except CanonicalSettingsPersistenceError as exc:
        raise CaptureTargetSettingsError(exc.status) from None
    except Exception:
        raise CaptureTargetSettingsError("save_failed") from None


__all__ = ["CaptureTargetSettingsError", "persist_desktop_audio_capture_target"]
