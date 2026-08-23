from __future__ import annotations

import json

from tests.helpers.paths import REPO_ROOT

I18N_DIR = REPO_ROOT / "src" / "puripuly_heart" / "data" / "i18n"

EXPECTED_SPEC_LOCALES = {"en", "ko", "ja", "zh-CN"}

CANONICAL_DESKTOP_OVERLAY_I18N_COPY = {
    "en": {
        "settings.overlay.desktop.size.title": "Overlay size",
        "settings.overlay.desktop.size.option.tiny": "Tiny",
        "settings.overlay.desktop.size.option.xsmall": "Extra small",
        "settings.overlay.desktop.size.option.small": "Small",
        "settings.overlay.desktop.size.option.medium": "Medium",
        "settings.overlay.desktop.size.option.large": "Large",
        "settings.overlay.desktop.size.option.xlarge": "Extra large",
        "settings.overlay.desktop.swap_caption_languages.title": "Reverse language order",
        "settings.overlay.desktop.lock.title": "Overlay lock",
        "settings.overlay.desktop.background_alpha.title": "Background transparency",
        "settings.overlay.desktop.lock.value.move": "Move",
        "settings.overlay.desktop.lock.value.locked": "Locked",
        "settings.overlay.desktop.empty_state.action.lock": "Lock",
        "settings.overlay.position_reset.title": "Reset overlay position",
        "settings.overlay.position_reset.vr.title": "Reset position",
        "settings.overlay.position_reset.desktop.title": "Reset position",
        "settings.overlay.position_reset.action.vr": "Reset VR",
        "settings.overlay.position_reset.action.desktop": "Reset desktop",
    },
    "ko": {
        "settings.overlay.desktop.size.title": "오버레이 크기",
        "settings.overlay.desktop.size.option.tiny": "아주 작게",
        "settings.overlay.desktop.size.option.xsmall": "더 작게",
        "settings.overlay.desktop.size.option.small": "작게",
        "settings.overlay.desktop.size.option.medium": "보통",
        "settings.overlay.desktop.size.option.large": "크게",
        "settings.overlay.desktop.size.option.xlarge": "더 크게",
        "settings.overlay.desktop.swap_caption_languages.title": "언어 순서 바꾸기",
        "settings.overlay.desktop.lock.title": "오버레이 잠금",
        "settings.overlay.desktop.background_alpha.title": "배경 투명도",
        "settings.overlay.desktop.lock.value.move": "이동",
        "settings.overlay.desktop.lock.value.locked": "고정",
        "settings.overlay.desktop.empty_state.action.lock": "고정하기",
        "settings.overlay.position_reset.title": "오버레이 위치 초기화",
        "settings.overlay.position_reset.vr.title": "위치 초기화",
        "settings.overlay.position_reset.desktop.title": "위치 초기화",
        "settings.overlay.position_reset.action.vr": "VR 초기화",
        "settings.overlay.position_reset.action.desktop": "데스크톱 초기화",
    },
    "ja": {
        "settings.overlay.desktop.size.title": "オーバーレイサイズ",
        "settings.overlay.desktop.size.option.tiny": "最小",
        "settings.overlay.desktop.size.option.xsmall": "さらに小さく",
        "settings.overlay.desktop.size.option.small": "小さめ",
        "settings.overlay.desktop.size.option.medium": "標準",
        "settings.overlay.desktop.size.option.large": "大きめ",
        "settings.overlay.desktop.size.option.xlarge": "さらに大きく",
        "settings.overlay.desktop.swap_caption_languages.title": "言語の順番を入れ替え",
        "settings.overlay.desktop.lock.title": "オーバーレイ固定",
        "settings.overlay.desktop.background_alpha.title": "背景の透明度",
        "settings.overlay.desktop.lock.value.move": "移動",
        "settings.overlay.desktop.lock.value.locked": "固定",
        "settings.overlay.desktop.empty_state.action.lock": "固定する",
        "settings.overlay.position_reset.title": "オーバーレイ位置をリセット",
        "settings.overlay.position_reset.vr.title": "位置をリセット",
        "settings.overlay.position_reset.desktop.title": "位置をリセット",
        "settings.overlay.position_reset.action.vr": "VRをリセット",
        "settings.overlay.position_reset.action.desktop": "デスクトップをリセット",
    },
    "zh-CN": {
        "settings.overlay.desktop.size.title": "叠加层大小",
        "settings.overlay.desktop.size.option.tiny": "极小",
        "settings.overlay.desktop.size.option.xsmall": "特小",
        "settings.overlay.desktop.size.option.small": "小",
        "settings.overlay.desktop.size.option.medium": "中",
        "settings.overlay.desktop.size.option.large": "大",
        "settings.overlay.desktop.size.option.xlarge": "特大",
        "settings.overlay.desktop.swap_caption_languages.title": "对调字幕语言顺序",
        "settings.overlay.desktop.lock.title": "叠加层锁定",
        "settings.overlay.desktop.background_alpha.title": "背景透明度",
        "settings.overlay.desktop.lock.value.move": "移动",
        "settings.overlay.desktop.lock.value.locked": "固定",
        "settings.overlay.desktop.empty_state.action.lock": "固定",
        "settings.overlay.position_reset.title": "重置叠加层位置",
        "settings.overlay.position_reset.vr.title": "重置位置",
        "settings.overlay.position_reset.desktop.title": "重置位置",
        "settings.overlay.position_reset.action.vr": "重置 VR",
        "settings.overlay.position_reset.action.desktop": "重置桌面",
    },
}

SHIPPING_DESKTOP_OVERLAY_I18N_KEYS = set(CANONICAL_DESKTOP_OVERLAY_I18N_COPY["ko"])

DESKTOP_OVERLAY_RECOVERY_I18N_COPY = {
    "en": {
        "settings.overlay.desktop.recovery.message.reopen": "Captions paused. Reopen them when you're ready.",
        "settings.overlay.desktop.recovery.message.retry": "Captions paused. Try again when you're ready.",
        "settings.overlay.desktop.recovery.action.reopen": "Reopen",
        "settings.overlay.desktop.recovery.action.retry": "Try again",
        "settings.overlay.desktop.recovery.action.view_details": "View details",
    },
    "ko": {
        "settings.overlay.desktop.recovery.message.reopen": "자막이 잠시 멈췄어요. 준비되면 다시 열어주세요.",
        "settings.overlay.desktop.recovery.message.retry": "자막이 잠시 멈췄어요. 준비되면 다시 시도해 주세요.",
        "settings.overlay.desktop.recovery.action.reopen": "다시 열기",
        "settings.overlay.desktop.recovery.action.retry": "다시 시도",
        "settings.overlay.desktop.recovery.action.view_details": "자세히 보기",
    },
    "ja": {
        "settings.overlay.desktop.recovery.message.reopen": "字幕が一時停止しました。準備できたら開き直してください。",
        "settings.overlay.desktop.recovery.message.retry": "字幕が一時停止しました。準備できたらもう一度お試しください。",
        "settings.overlay.desktop.recovery.action.reopen": "開き直す",
        "settings.overlay.desktop.recovery.action.retry": "もう一度試す",
        "settings.overlay.desktop.recovery.action.view_details": "詳しく見る",
    },
    "zh-CN": {
        "settings.overlay.desktop.recovery.message.reopen": "字幕暂停了。准备好后重新打开。",
        "settings.overlay.desktop.recovery.message.retry": "字幕暂停了。准备好后再试一次。",
        "settings.overlay.desktop.recovery.action.reopen": "重新打开",
        "settings.overlay.desktop.recovery.action.retry": "再试一次",
        "settings.overlay.desktop.recovery.action.view_details": "查看详情",
    },
}

DESKTOP_OVERLAY_RECOVERY_I18N_KEYS = set(DESKTOP_OVERLAY_RECOVERY_I18N_COPY["ko"])

OLD_OVERLAY_LOCAL_DESKTOP_KEYS = {
    "settings.overlay.desktop.action.lock_captions",
    "settings.overlay.desktop.action.edit_position",
    "settings.overlay.desktop.action.move_captions",
    "settings.overlay.desktop.action.reset_position",
    "settings.overlay.desktop.helper.locked",
    "settings.overlay.desktop.placeholder.no_captions",
    "settings.overlay.desktop.hint.resize",
    "settings.overlay.desktop.preview.outline_width",
    "settings.overlay.desktop.preview.text_scale",
    "settings.overlay.desktop.preview.text_scale.auto",
    "settings.overlay.desktop.preview.text_scale.small",
    "settings.overlay.desktop.preview.text_scale.normal",
    "settings.overlay.desktop.preview.text_scale.large",
}

DOCUMENTED_DESKTOP_OVERLAY_FAILURE_REASONS = {
    "missing_executable",
    "spawn_failed",
    "manifest_invalid",
    "contract_mismatch",
    "startup_timeout",
    "bridge_auth_failed",
    "renderer_init_failed",
    "runtime_disconnected",
    "window_configuration_failed",
    "runtime_control_invalid",
    "runtime_crashed",
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


def test_desktop_overlay_i18n_copy_matches_product_standard() -> None:
    bundles = _load_bundles()

    for locale, expected_copy in CANONICAL_DESKTOP_OVERLAY_I18N_COPY.items():
        bundle = bundles[locale]
        actual_copy = {key: bundle[key] for key in expected_copy}

        assert actual_copy == expected_copy


def test_desktop_overlay_recovery_i18n_copy_is_user_facing() -> None:
    bundles = _load_bundles()

    for locale, expected_copy in DESKTOP_OVERLAY_RECOVERY_I18N_COPY.items():
        bundle = bundles[locale]
        missing = sorted(DESKTOP_OVERLAY_RECOVERY_I18N_KEYS - set(bundle))
        assert missing == [], locale
        actual_copy = {key: bundle[key] for key in expected_copy}
        assert actual_copy == expected_copy

    technical_fragments = ("executable", "bridge", "renderer", "runtime", "logs")
    en_bundle = bundles["en"]
    recovery_copy = [en_bundle[key] for key in DESKTOP_OVERLAY_RECOVERY_I18N_KEYS]
    assert not [
        (text, fragment)
        for text in recovery_copy
        for fragment in technical_fragments
        if fragment in text.lower()
    ]


def test_shipping_desktop_overlay_i18n_requirements_exclude_old_overlay_local_copy() -> None:
    old_shipping_keys = sorted(SHIPPING_DESKTOP_OVERLAY_I18N_KEYS & OLD_OVERLAY_LOCAL_DESKTOP_KEYS)

    assert old_shipping_keys == []


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

    assert bundle["settings.overlay.target.steamvr"] == "VR"
    assert bundle["settings.overlay.target.desktop"] == "Desktop"
    assert bundle["settings.overlay.desktop.size.title"] == "Overlay size"
    assert bundle["settings.overlay.desktop.lock.value.move"] == "Move"
    assert bundle["settings.overlay.desktop.lock.value.locked"] == "Locked"
    assert bundle["settings.overlay.desktop.empty_state.action.lock"] == "Lock"
    assert bundle["settings.overlay.position_reset.action.desktop"] == "Reset desktop"

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
