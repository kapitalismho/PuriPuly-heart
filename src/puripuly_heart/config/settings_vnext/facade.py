from __future__ import annotations

from pathlib import Path
from typing import Any

from puripuly_heart.config.settings_vnext import compat as vnext_compat
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


def load_vnext_settings(path: Path, **kwargs: Any) -> vnext_compat.VNextSettingsLoadResult:
    return vnext_compat.load_vnext_settings(path, **kwargs)


def save_vnext_settings(
    path: Path,
    settings: AppSettingsVNext,
) -> vnext_compat.VNextSettingsSaveResult:
    return vnext_compat.save_vnext_settings(path, settings)


__all__ = [
    "load_vnext_settings",
    "save_vnext_settings",
]
