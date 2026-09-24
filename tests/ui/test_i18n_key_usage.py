from __future__ import annotations

import json
import re

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


_CALL_SITE_I18N_KEY_RE = re.compile(r'\bt\(\s*"([a-z][A-Za-z0-9_.]*)"')
_T_FOR_LOCALE_I18N_KEY_RE = re.compile(
    r'\bt_for_locale\(\s*[A-Za-z_][A-Za-z0-9_.]*\s*,\s*"([a-z][A-Za-z0-9_.]*)"'
)


def _referenced_literal_i18n_keys(runtime_source: str) -> set[str]:
    return set(_CALL_SITE_I18N_KEY_RE.findall(runtime_source)) | set(
        _T_FOR_LOCALE_I18N_KEY_RE.findall(runtime_source)
    )


def test_i18n_literal_call_sites_exist_in_every_bundle() -> None:
    bundles = _load_bundles()
    referenced = {
        key
        for key in _referenced_literal_i18n_keys(_runtime_python_source())
        if not key.startswith(DYNAMIC_I18N_PREFIXES)
    }

    missing = {locale: sorted(referenced - set(bundle)) for locale, bundle in bundles.items()}

    assert missing == {locale: [] for locale in bundles}


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


def test_qwen_audio_and_speaker_recognition_copy_is_consistent_for_every_locale() -> None:
    bundles = _load_bundles()
    expected = {
        "en": ("Good at recognizing Chinese", "Speaker Recognition"),
        "ja": ("中国語の認識が得意です", "話者認識"),
        "ko": ("중국어를 잘해요", "화자 인식"),
        "ru": ("Хорошо распознаёт китайскую речь", "Распознавание говорящего"),
        "zh-CN": ("中文识别效果好", "说话人识别"),
    }

    assert set(bundles) == set(expected)
    for locale, (qwen_audio_description, speaker_recognition_label) in expected.items():
        assert bundles[locale]["provider.qwen_audio"] == "Qwen Audio 3.1"
        assert bundles[locale]["provider.qwen_audio.description"] == qwen_audio_description
        assert bundles[locale]["settings.soniox_speaker_diarization"] == (
            speaker_recognition_label
        )


def test_i18n_bundles_do_not_keep_unused_runtime_keys() -> None:
    bundles = _load_bundles()
    all_keys = sorted(set().union(*(bundle.keys() for bundle in bundles.values())))
    runtime_source = _runtime_python_source()

    unused_keys = _unused_i18n_keys(all_keys, runtime_source)

    assert unused_keys == []


def test_dynamic_overlay_failure_keys_exist_in_every_bundle() -> None:
    bundles = _load_bundles()

    missing = {
        locale: sorted(OVERLAY_FAILURE_I18N_KEYS - set(bundle))
        for locale, bundle in bundles.items()
    }

    assert missing == {locale: [] for locale in bundles}
