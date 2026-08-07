from __future__ import annotations

import json

from tests.helpers.paths import REPO_ROOT
from tests.ui.test_desktop_overlay_i18n import (
    DESKTOP_OVERLAY_RECOVERY_I18N_KEYS,
    SHIPPING_DESKTOP_OVERLAY_I18N_KEYS,
)

I18N_DIR = REPO_ROOT / "src" / "puripuly_heart" / "data" / "i18n"
RUNTIME_SOURCE_DIR = REPO_ROOT / "src" / "puripuly_heart"

DYNAMIC_I18N_PREFIXES = (
    "language.",
    "locale.",
    "provider.",
    "region.",
    "settings.subtab.",
    "settings.overlay.calibration.anchor.",
    "settings.overlay.calibration.text_scale.",
    "settings.overlay.failure.",
    "settings.overlay.status.",
    "settings.peer_translation.status.",
    "logs.mode.",
    "settings.translation_model.",
)

# Overlay target labels are selected with a runtime suffix; keep this exact so target typos fail.
EXACT_DYNAMIC_I18N_KEYS = frozenset(
    {
        "settings.overlay.target.desktop",
        "settings.overlay.target.steamvr",
    }
)

# Desktop-overlay copy seeds product-standard keys before every key is referenced
# in runtime code.
# Keep this exact, temporary allowlist narrow so typo or stale seeded keys still fail.
TEMPORARILY_ALLOWED_UNREFERENCED_I18N_KEYS = frozenset(
    SHIPPING_DESKTOP_OVERLAY_I18N_KEYS | DESKTOP_OVERLAY_RECOVERY_I18N_KEYS
)


def _load_bundles() -> dict[str, dict[str, str]]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(I18N_DIR.glob("*.json"))
    }


def _runtime_python_source() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(RUNTIME_SOURCE_DIR.rglob("*.py"))
    )


def _unused_i18n_keys(keys: list[str], runtime_source: str) -> list[str]:
    return [
        key
        for key in keys
        if key not in runtime_source
        and not key.startswith(DYNAMIC_I18N_PREFIXES)
        and key not in EXACT_DYNAMIC_I18N_KEYS
        and key not in TEMPORARILY_ALLOWED_UNREFERENCED_I18N_KEYS
    ]


def test_i18n_bundles_share_the_same_keys() -> None:
    bundles = _load_bundles()
    assert "en" in bundles

    expected_keys = set(bundles["en"])
    mismatches = {
        locale: {
            "missing": sorted(expected_keys - set(bundle)),
            "extra": sorted(set(bundle) - expected_keys),
        }
        for locale, bundle in bundles.items()
        if set(bundle) != expected_keys
    }

    assert mismatches == {}


def test_i18n_bundles_do_not_keep_unused_runtime_keys() -> None:
    bundles = _load_bundles()
    all_keys = sorted(set().union(*(bundle.keys() for bundle in bundles.values())))
    runtime_source = _runtime_python_source()

    unused_keys = _unused_i18n_keys(all_keys, runtime_source)

    assert unused_keys == []
