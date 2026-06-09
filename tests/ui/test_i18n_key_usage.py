from __future__ import annotations

import json
from pathlib import Path

from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.i18n import available_locales, source_label
from tests.ui.test_desktop_overlay_i18n import (
    DESKTOP_OVERLAY_RECOVERY_I18N_KEYS,
    SHIPPING_DESKTOP_OVERLAY_I18N_KEYS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
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

GITHUB_STAR_SNACKBAR_KEYS = (
    "github_star.snackbar.message",
    "github_star.snackbar.action",
)
OVERLAY_STEAMVR_NOT_RUNNING_KEY = "settings.overlay.failure.steamvr_not_running"

EXPECTED_GITHUB_STAR_SNACKBAR_KO_COPY = {
    "github_star.snackbar.message": "PuriPuly가 도움이 됐다면 GitHub에서 Star를 눌러주세요! 큰 힘이 되어요!",
    "github_star.snackbar.action": "이동",
}
EXPECTED_OVERLAY_STEAMVR_NOT_RUNNING_COPY = {
    "en": "SteamVR is off. If you want to use the desktop overlay, change it in Settings.",
    "ko": "SteamVR이 꺼져 있어요. 혹시 데스크톱 오버레이를 쓰고싶다면 설정을 바꿔주세요.",
    "zh-CN": "SteamVR 尚未运行。如果您想使用桌面叠加层，请在设置中更改。",
    "ja": "SteamVRがオフです。デスクトップオーバーレイを使いたい場合は、設定を変更してください。",
}

# Overlay target labels are selected with a runtime suffix; keep this exact so target typos fail.
EXACT_DYNAMIC_I18N_KEYS = frozenset(
    {
        "settings.overlay.target.desktop",
        "settings.overlay.target.steamvr",
    }
)

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


def test_available_locales_use_product_display_order() -> None:
    assert available_locales() == ("en", "ko", "zh-CN", "ja")


def test_clipboard_source_and_setting_keys_are_localized() -> None:
    bundles = _load_bundles()
    required_keys = {
        "source.clipboard",
        "settings.clipboard_auto_translate",
        "settings.clipboard_auto_translate.on",
        "settings.clipboard_auto_translate.off",
    }

    for locale, bundle in bundles.items():
        missing = sorted(required_keys - set(bundle))
        assert missing == [], locale
        for key in required_keys:
            assert bundle[key].strip()
            assert bundle[key] != key

    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale("ko")
        assert source_label("Clipboard") == "클립보드"
    finally:
        i18n_module.set_locale(previous_locale)


def test_logs_conversation_keys_are_localized() -> None:
    bundles = _load_bundles()
    required_keys = {
        "logs.conversation.show",
        "logs.conversation.hide",
        "logs.conversation.empty",
    }

    for locale, bundle in bundles.items():
        missing = sorted(required_keys - set(bundle))
        assert missing == [], locale
        for key in required_keys:
            assert bundle[key].strip()
            assert bundle[key] != key

    assert bundles["ko"]["logs.conversation.show"] == "대화록 보기"


def test_github_star_snackbar_keys_are_localized_for_all_supported_locales() -> None:
    bundles = _load_bundles()
    supported_locales = set(available_locales())

    assert set(bundles) == supported_locales
    for locale, bundle in bundles.items():
        missing = sorted(set(GITHUB_STAR_SNACKBAR_KEYS) - set(bundle))
        assert missing == [], locale
        for key in GITHUB_STAR_SNACKBAR_KEYS:
            assert bundle[key].strip()
            assert bundle[key] != key


def test_github_star_snackbar_korean_copy_matches_source_spec() -> None:
    ko = _load_bundles()["ko"]

    assert {
        key: ko[key] for key in GITHUB_STAR_SNACKBAR_KEYS
    } == EXPECTED_GITHUB_STAR_SNACKBAR_KO_COPY


def test_overlay_steamvr_not_running_copy_points_to_desktop_overlay_setting() -> None:
    bundles = _load_bundles()

    for locale, expected in EXPECTED_OVERLAY_STEAMVR_NOT_RUNNING_COPY.items():
        assert bundles[locale][OVERLAY_STEAMVR_NOT_RUNNING_KEY] == expected


def test_local_llm_keys_are_localized() -> None:
    bundles = _load_bundles()
    required_keys = {
        "provider.local_llms",
        "provider.local_llm",
        "settings.translation_model.local_llm.description",
        "settings.translation_connection.ollama",
        "settings.translation_connection.ollama.description",
        "settings.local_llm.connection",
        "settings.local_llm.base_url",
        "settings.local_llm.base_url.invalid",
        "settings.local_llm.model",
        "settings.local_llm.model.required",
        "settings.local_llm.api_key",
        "settings.local_llm.api_key.description",
        "settings.local_llm.api_key.save_failed",
        "settings.local_llm.extra_body",
        "settings.local_llm.extra_body.description",
        "settings.local_llm.extra_body.invalid_json",
        "settings.local_llm.extra_body.must_be_object",
        "settings.local_llm.extra_body.reserved_key",
        "settings.local_llm.extra_body.sensitive_key",
        "settings.local_llm.extra_body.not_serializable",
    }

    for locale in ("en", "ko", "ja", "zh-CN"):
        bundle = bundles[locale]
        missing = sorted(required_keys - set(bundle))
        assert missing == [], locale
        for key in required_keys:
            if key in {
                "settings.local_llm.api_key.description",
            }:
                assert bundle[key] == ""
                continue
            assert bundle[key].strip()
            assert bundle[key] != key

    assert bundles["en"]["settings.translation_connection.ollama"] == "OpenAI-compatible API"
    assert bundles["ko"]["settings.translation_connection.ollama"] == "OpenAI 호환 API"
    expected_local_llm_descriptions = {
        "en": "You can use an OpenAI-compatible API",
        "ko": "OpenAI 호환 API를 사용할 수 있어요",
        "ja": "OpenAI互換APIを使用できます",
        "zh-CN": "可以使用 OpenAI 兼容 API",
    }
    expected_gemini31_flash_lite_descriptions = {
        "en": "Translation speed may be unstable",
        "ko": "번역 속도가 불안정할 수 있어요",
        "ja": "翻訳速度が不安定になることがあります",
        "zh-CN": "翻译速度可能不稳定",
    }
    for locale, expected in expected_local_llm_descriptions.items():
        assert bundles[locale]["settings.translation_model.local_llm.description"] == expected
    for locale, expected in expected_gemini31_flash_lite_descriptions.items():
        assert (
            bundles[locale]["settings.translation_model.gemini31_flash_lite.description"]
            == expected
        )
    assert bundles["ko"]["settings.local_llm.connection"] == "OpenAI 호환 LLM 서버"
    assert bundles["ko"]["settings.local_llm.base_url"] == "Base URL"
    expected_model_copy = {
        "en": ("Model ID", "Enter a model ID."),
        "ko": ("모델 ID", "모델 ID를 입력해 주세요."),
        "ja": ("モデルID", "モデルIDを入力してください。"),
        "zh-CN": ("模型 ID", "请输入模型 ID。"),
    }
    for locale, (model_label, required_label) in expected_model_copy.items():
        assert bundles[locale]["settings.local_llm.model"] == model_label
        assert bundles[locale]["settings.local_llm.model.required"] == required_label
    assert bundles["ko"]["settings.local_llm.api_key"] == "서버 API 키 (선택)"
    assert bundles["ko"]["settings.local_llm.api_key.description"] == ""
    assert bundles["ko"]["settings.local_llm.extra_body.description"].startswith("낮은 지연시간")
    assert "서버 API 키" in bundles["ko"]["settings.local_llm.extra_body.sensitive_key"]


def test_zh_cn_qwen_labels_use_qwen_brand_name() -> None:
    zh_cn = _load_bundles()["zh-CN"]

    expected_qwen_labels = {
        "settings.alibaba_api_key_beijing": "Qwen API密钥（北京）",
        "settings.alibaba_api_key_singapore": "Qwen API密钥（新加坡）",
        "settings.qwen_region": "Qwen 服务区域：",
        "provider.qwen": "Qwen 3.5",
        "provider.qwen35_flash": "Qwen 3.5 Flash",
        "provider.qwen35_plus": "Qwen 3.5 Plus",
        "provider.qwen_asr": "Qwen ASR",
    }

    for key, expected in expected_qwen_labels.items():
        assert zh_cn[key] == expected

    for value in zh_cn.values():
        assert "通义千问" not in value


def test_deepseek_v4_pro_keys_are_localized_with_blank_descriptions() -> None:
    bundles = _load_bundles()
    required_keys = {
        "provider.deepseek_v4_pro",
        "provider.deepseek_v4_pro.description",
        "settings.translation_model.deepseek_v4_pro.description",
    }
    forbidden_keys = {
        "provider.deepseek_v4_pro_openrouter",
        "provider.deepseek_v4_pro_openrouter.description",
    }

    for locale, bundle in bundles.items():
        missing = sorted(required_keys - set(bundle))
        assert missing == [], locale
        present_forbidden = sorted(forbidden_keys.intersection(bundle))
        assert present_forbidden == [], locale
        assert bundle["provider.deepseek_v4_pro"].strip()
        assert bundle["provider.deepseek_v4_pro"] != "provider.deepseek_v4_pro"
        assert bundle["provider.deepseek_v4_pro.description"] == ""
        assert bundle["settings.translation_model.deepseek_v4_pro.description"] == ""


def test_managed_key_card_keys_are_localized() -> None:
    bundles = _load_bundles()
    required_keys = {
        "settings.managed_key.title",
        "settings.managed_key.referral_id.label",
        "settings.managed_key.referral_id.empty",
        "settings.managed_key.referral_id.pending_helper",
        "settings.managed_key.referral_id.helper",
        "settings.managed_key.invite_progress.label",
    }

    for locale, bundle in bundles.items():
        missing = sorted(required_keys - set(bundle))
        assert missing == [], locale
        for key in required_keys:
            assert bundle[key].strip()
            assert bundle[key] != key

    assert bundles["en"]["settings.managed_key.title"] == "Managed Key"
    assert bundles["en"]["settings.managed_key.referral_id.label"] == "Talk Together Pass ID"
    assert bundles["en"]["settings.managed_key.referral_id.empty"] == "—"
    assert bundles["ko"]["settings.managed_key.title"] == "매니지드 키"
    ko = bundles["ko"]
    assert ko["settings.managed_key.referral_id.helper"] == (
        "친구에게 Pass ID를 공유하면 함께 추가 사용량을 받을 수 있어요."
    )
    assert ko["settings.managed_key.invite_progress.label"] == "친구 초대"

    for locale_name, bundle in bundles.items():
        for key in (
            "settings.managed_key.referral_id.label",
            "settings.managed_key.referral_id.helper",
            "settings.managed_key.invite_progress.label",
            "discord_auth.referral_id.label",
            "discord_auth.referral_reward_applied",
        ):
            value = bundle[key]
            assert "Referral ID" not in value, (locale_name, key, value)
            assert "Referral reward" not in value, (locale_name, key, value)


def test_qq_key_unavailable_error_key_is_localized_for_all_supported_locales() -> None:
    bundles = _load_bundles()
    runtime_source = _runtime_python_source()
    key = "qq_auth.error.key_unavailable"

    assert key in runtime_source
    for locale, bundle in bundles.items():
        assert key in bundle, locale
        assert bundle[key].strip()
        assert bundle[key] != key


def test_i18n_bundles_do_not_keep_unused_runtime_keys() -> None:
    bundles = _load_bundles()
    all_keys = sorted(set().union(*(bundle.keys() for bundle in bundles.values())))
    runtime_source = _runtime_python_source()

    unused_keys = _unused_i18n_keys(all_keys, runtime_source)

    assert unused_keys == []


def test_unused_key_guard_flags_desktop_overlay_typos() -> None:
    runtime_source = ""
    typo_like_keys = {
        "debug_preview.desktop_overlay_typo",
        "settings.overlay.caption_location.extra",
        "settings.overlay.desktop.typo",
        "settings.overlay.target.typo",
    }

    unused_keys = _unused_i18n_keys(sorted(typo_like_keys), runtime_source)

    assert unused_keys == sorted(typo_like_keys)


def test_desktop_overlay_seed_keys_are_exactly_allowlisted() -> None:
    unused_keys = _unused_i18n_keys(
        sorted(TEMPORARILY_ALLOWED_UNREFERENCED_I18N_KEYS),
        runtime_source="",
    )

    assert unused_keys == []
