from __future__ import annotations

from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


def load_application_settings(
    *,
    settings: SettingsOwner,
) -> AppSettingsVNext:
    return settings.start().settings
