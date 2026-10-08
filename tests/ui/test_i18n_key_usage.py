from __future__ import annotations

import json

from tests.helpers.paths import REPO_ROOT

I18N_DIR = REPO_ROOT / "src" / "puripuly_heart" / "data" / "i18n"

# Overlay failure reasons are composed at runtime as f"settings.overlay.failure.{reason}",
# so no individual key is referenced literally in source; every bundle must ship all of them.
OVERLAY_FAILURE_I18N_KEYS = frozenset(
    {
        "settings.overlay.failure.bridge_auth_failed",
        "settings.overlay.failure.contract_mismatch",
        "settings.overlay.failure.hmd_not_found",
        "settings.overlay.failure.manifest_invalid",
        "settings.overlay.failure.missing_executable",
        "settings.overlay.failure.openvr_dll_hash_mismatch",
        "settings.overlay.failure.openvr_init_failed",
        "settings.overlay.failure.packaged_openvr_dll_missing",
        "settings.overlay.failure.renderer_init_failed",
        "settings.overlay.failure.runtime_control_invalid",
        "settings.overlay.failure.runtime_crashed",
        "settings.overlay.failure.shutdown_not_acknowledged",
        "settings.overlay.failure.runtime_exit_nonzero",
        "settings.overlay.failure.shutdown_forced",
        "settings.overlay.failure.shutdown_cleanup_failed",
        "settings.overlay.failure.runtime_disconnected",
        "settings.overlay.failure.spawn_failed",
        "settings.overlay.failure.stale_overlay_build",
        "settings.overlay.failure.startup_timeout",
        "settings.overlay.failure.steamvr_not_installed",
        "settings.overlay.failure.steamvr_not_running",
        "settings.overlay.failure.unknown",
        "settings.overlay.failure.vendored_openvr_dll_missing",
        "settings.overlay.failure.window_configuration_failed",
    }
)


def _load_bundles() -> dict[str, dict[str, str]]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(I18N_DIR.glob("*.json"))
    }


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


def test_dynamic_overlay_failure_keys_exist_in_every_bundle() -> None:
    bundles = _load_bundles()

    missing = {
        locale: sorted(OVERLAY_FAILURE_I18N_KEYS - set(bundle))
        for locale, bundle in bundles.items()
    }

    assert missing == {locale: [] for locale in bundles}
