from __future__ import annotations

import json

from tests.helpers.paths import REPO_ROOT

I18N_DIR = REPO_ROOT / "src" / "puripuly_heart" / "data" / "i18n"

EXPECTED_SPEC_LOCALES = {"en", "ko", "ja", "zh-CN"}

SHIPPING_DESKTOP_OVERLAY_I18N_KEYS = {
    "settings.overlay.desktop.size.title",
    "settings.overlay.desktop.size.option.tiny",
    "settings.overlay.desktop.size.option.xsmall",
    "settings.overlay.desktop.size.option.small",
    "settings.overlay.desktop.size.option.medium",
    "settings.overlay.desktop.size.option.large",
    "settings.overlay.desktop.size.option.xlarge",
    "settings.overlay.desktop.swap_caption_languages.title",
    "settings.overlay.desktop.lock.title",
    "settings.overlay.desktop.background_alpha.title",
    "settings.overlay.desktop.lock.value.move",
    "settings.overlay.desktop.lock.value.locked",
    "settings.overlay.desktop.empty_state.action.lock",
    "settings.overlay.position_reset.title",
    "settings.overlay.position_reset.vr.title",
    "settings.overlay.position_reset.desktop.title",
    "settings.overlay.position_reset.action.vr",
    "settings.overlay.position_reset.action.desktop",
}

DESKTOP_OVERLAY_RECOVERY_I18N_KEYS = {
    "settings.overlay.desktop.recovery.message.reopen",
    "settings.overlay.desktop.recovery.message.retry",
    "settings.overlay.desktop.recovery.action.reopen",
    "settings.overlay.desktop.recovery.action.retry",
    "settings.overlay.desktop.recovery.action.view_details",
}

DOCUMENTED_DESKTOP_OVERLAY_FAILURE_REASONS = {
    "missing_executable",
    "spawn_failed",
    "manifest_invalid",
    "contract_mismatch",
    "startup_timeout",
    "bridge_auth_failed",
    "renderer_init_failed",
    "gpu_readiness_late",
    "gpu_readiness_cancelled",
    "gpu_query_failed",
    "gpu_stalled",
    "runtime_disconnected",
    "window_configuration_failed",
    "runtime_control_invalid",
    "runtime_crashed",
    "shutdown_not_acknowledged",
    "runtime_exit_nonzero",
    "shutdown_forced",
    "shutdown_cleanup_failed",
    "unknown",
}


def _load_bundles() -> dict[str, dict[str, str]]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(I18N_DIR.glob("*.json"))
    }


def test_desktop_overlay_i18n_keys_are_present_in_every_locale_bundle() -> None:
    bundles = _load_bundles()

    assert EXPECTED_SPEC_LOCALES <= set(bundles)

    for locale in EXPECTED_SPEC_LOCALES:
        bundle = bundles[locale]
        missing = sorted(SHIPPING_DESKTOP_OVERLAY_I18N_KEYS - set(bundle))
        assert missing == [], locale
        for key in SHIPPING_DESKTOP_OVERLAY_I18N_KEYS:
            assert bundle[key].strip(), (locale, key)
            assert bundle[key] != key, (locale, key)


def test_desktop_overlay_recovery_i18n_copy_is_user_facing() -> None:
    bundles = _load_bundles()

    for locale in EXPECTED_SPEC_LOCALES:
        bundle = bundles[locale]
        missing = sorted(DESKTOP_OVERLAY_RECOVERY_I18N_KEYS - set(bundle))
        assert missing == [], locale

    technical_fragments = ("executable", "bridge", "renderer", "runtime", "logs")
    en_bundle = bundles["en"]
    recovery_copy = [en_bundle[key] for key in DESKTOP_OVERLAY_RECOVERY_I18N_KEYS]
    assert not [
        (text, fragment)
        for text in recovery_copy
        for fragment in technical_fragments
        if fragment in text.lower()
    ]


def test_overlay_failure_i18n_keys_cover_documented_desktop_overlay_reasons() -> None:
    bundles = _load_bundles()
    required_keys = {
        f"settings.overlay.failure.{reason}"
        for reason in DOCUMENTED_DESKTOP_OVERLAY_FAILURE_REASONS
    }

    for locale, bundle in bundles.items():
        missing = sorted(required_keys - set(bundle))
        assert missing == [], locale
        for key in required_keys:
            assert bundle[key].strip(), (locale, key)
            assert bundle[key] != key, (locale, key)


def test_desktop_overlay_i18n_english_copy_uses_product_language() -> None:
    bundle = _load_bundles()["en"]

    user_facing_copy = [
        bundle[key]
        for key in SHIPPING_DESKTOP_OVERLAY_I18N_KEYS
        | {
            f"settings.overlay.failure.{reason}"
            for reason in DOCUMENTED_DESKTOP_OVERLAY_FAILURE_REASONS
        }
    ]
    banned_fragments = ("Flet renderer", "pass-through", "pass clicks")
    assert not [
        (text, fragment)
        for text in user_facing_copy
        for fragment in banned_fragments
        if fragment in text
    ]
