from __future__ import annotations

from pathlib import Path

from puripuly_heart.app.ports.canonical_settings_persistence import (
    CanonicalSettingsPersistenceError,
)
from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.config.settings_vnext.defaults import new_settings_for_first_run
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    CaptureTargetIntent,
)


class CaptureTargetSettingsError(RuntimeError):
    def __init__(self, status: object) -> None:
        self.status = str(getattr(status, "value", status))
        super().__init__(f"capture_target_settings_{self.status}")


def persist_desktop_audio_capture_target(
    path: Path,
    capture_target: CaptureTargetIntent,
) -> AppSettingsVNext:
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
            owner.canonical = new_settings_for_first_run()
        except Exception:
            raise CaptureTargetSettingsError("migration_failed") from None
    try:
        owner.apply_capture_target(capture_target)
    except CanonicalSettingsPersistenceError as exc:
        raise CaptureTargetSettingsError(exc.status) from None
    except Exception:
        raise CaptureTargetSettingsError("save_failed") from None
    if owner.canonical is None:
        raise CaptureTargetSettingsError("save_failed")
    return owner.canonical


__all__ = ["CaptureTargetSettingsError", "persist_desktop_audio_capture_target"]
