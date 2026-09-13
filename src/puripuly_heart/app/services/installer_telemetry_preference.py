from __future__ import annotations

from pathlib import Path

from puripuly_heart.app.services.canonical_settings_persistence import (
    compose_settings_owner,
)


def persist_installer_telemetry_preference(path: Path, enabled: bool) -> None:
    compose_settings_owner(path).persist_telemetry_preference(enabled)


__all__ = ["persist_installer_telemetry_preference"]
