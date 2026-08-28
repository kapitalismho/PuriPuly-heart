from __future__ import annotations

from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner


def load_application_settings(
    *,
    settings: SettingsOwner,
) -> object:
    return settings.start().settings
