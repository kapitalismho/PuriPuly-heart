from __future__ import annotations

import contextlib
from pathlib import Path

from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.wiring import copy_stable_secrets_to_vnext_namespace


def load_application_settings(
    *,
    settings: SettingsOwner,
    config_path: Path,
    allow_stable_settings_import: bool,
) -> object:
    result = settings.start(
        allow_stable_settings_import=allow_stable_settings_import,
    )
    if result.stable_source_path is not None and result.imported_settings is not None:
        with contextlib.suppress(Exception):
            copy_stable_secrets_to_vnext_namespace(
                (result.stable_source_settings or result.imported_settings).intent.secrets,
                stable_config_path=result.stable_source_path,
                vnext_config_path=config_path,
                vnext_settings=result.imported_settings.intent.secrets,
            )
    return result.settings
