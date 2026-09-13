from __future__ import annotations

from pathlib import Path

from puripuly_heart.config.settings_vnext.compat import load_vnext_settings, save_vnext_settings
from puripuly_heart.config.settings_vnext.defaults import new_settings_for_first_run
from puripuly_heart.config.settings_vnext.schema import with_telemetry_enabled


def persist_installer_telemetry_preference(path: Path, enabled: bool) -> None:
    if path.exists():
        load_result = load_vnext_settings(path)
        if load_result.settings is None:
            message = (
                load_result.error.message
                if load_result.error is not None
                else str(load_result.status)
            )
            raise RuntimeError(message)
        settings = load_result.settings
    else:
        settings = new_settings_for_first_run()
    expected = with_telemetry_enabled(settings, enabled)
    save_result = save_vnext_settings(path, expected)
    if not save_result.ok:
        message = (
            save_result.error.message if save_result.error is not None else str(save_result.status)
        )
        raise RuntimeError(message)
    verified = load_vnext_settings(path)
    if verified.settings != expected:
        message = (
            verified.error.message
            if verified.error is not None
            else "persisted telemetry preference did not verify"
        )
        raise RuntimeError(message)


__all__ = ["persist_installer_telemetry_preference"]
