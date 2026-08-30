from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from puripuly_heart.config.settings_vnext.schema import VNEXT_SETTINGS_SCHEMA_VERSION

CANONICAL_SETTINGS_TOP_LEVEL_KEYS = {"settings_version", "intent", "state"}


def load_raw_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_raw_vnext_settings_file(path: Path) -> dict[str, Any]:
    data = load_raw_json_file(path)
    assert isinstance(data, dict)
    assert set(data) == CANONICAL_SETTINGS_TOP_LEVEL_KEYS
    assert data["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION
    return data


def legacy_projected_settings_dict(raw_vnext: dict[str, Any]) -> dict[str, Any]:
    return assert_raw_vnext_dict(raw_vnext)


def assert_raw_vnext_dict(raw_vnext: dict[str, Any]) -> dict[str, Any]:
    assert set(raw_vnext) == CANONICAL_SETTINGS_TOP_LEVEL_KEYS
    assert raw_vnext["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION
    return raw_vnext


def legacy_projected_settings_file(path: Path) -> dict[str, Any]:
    return legacy_projected_settings_dict(assert_raw_vnext_settings_file(path))
