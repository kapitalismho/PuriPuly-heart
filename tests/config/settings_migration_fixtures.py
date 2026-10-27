from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

DYNAMIC_MAPPING_PATHS = frozenset(
    {
        "local_llm.extra_body",
        "stt.custom_terms",
        "translation.connection_history",
        "custom_stt.extra",
    }
)

_FIXTURES_DIR = Path(__file__).with_name("fixtures")


def serialized_field_paths(data: dict[str, Any], prefix: str = "") -> Iterator[str]:
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and path in DYNAMIC_MAPPING_PATHS:
            yield path
        elif isinstance(value, dict) and value:
            yield from serialized_field_paths(value, path)
        else:
            yield path


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES_DIR / name).read_text(encoding="utf-8"))


def maximal_v24_settings_fixture() -> dict[str, Any]:
    return _load_fixture("maximal_v24_settings.json")


def legacy_compatibility_settings_fixture() -> dict[str, Any]:
    return _load_fixture("legacy_compatibility_settings.json")
