from __future__ import annotations

import json

from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.i18n import available_locales, source_label
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

GITHUB_STAR_SNACKBAR_KEYS = (
    "github_star.snackbar.message",
    "github_star.snackbar.action",
)
OVERLAY_STEAMVR_NOT_RUNNING_KEY = "settings.overlay.failure.steamvr_not_running"
SETTINGS_MUTATION_MESSAGE_KEYS = ("settings.mutation.runtime_apply_failed",)
MESSAGE_ERROR_BOUNDARY_KEYS = (
    "provider.failure",
    "stt.failure",
    "settings.mutation.runtime_apply_failed",
)
QQ_AUTH_I18N_KEYS = (
    "qq_auth.body",
    "qq_auth.submit",
    "qq_auth.close",
    "qq_auth.cancel",
    "qq_auth.waiting_body",
    "qq_auth.success",
    "qq_auth.qq_identity.label",
    "qq_auth.qq_identity.helper",
    "qq_auth.credential.label",
    "qq_auth.credential.helper",
    "qq_auth.error.invalid_input",
    "qq_auth.error.credential_mismatch",
    "qq_auth.error.lifetime_used",
    "qq_auth.error.already_claimed_discord",
    "qq_auth.error.retry",
    "qq_auth.error.key_unavailable",
    "qq_auth.error.rate_limited",
    "qq_auth.error.broker_unavailable",
    "qq_auth.error.settings_commit_failed",
    "qq_auth.error.secret_write_failed",
    "discord_auth.error.already_claimed_qq",
)

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
CUSTOM_VOCABULARY_TAG_EDITOR_I18N_KEYS = (
    "settings.custom_vocabulary.description",
    "settings.custom_vocabulary.add_placeholder",
    "settings.custom_vocabulary.add_action",
    "settings.custom_vocabulary.empty",
    "settings.custom_vocabulary.remove_hint",
)
CUSTOM_VOCABULARY_EXISTING_I18N_KEYS = (
    "settings.section.custom_vocabulary",
    "snackbar.custom_vocabulary_limit",
)
EXPECTED_CUSTOM_VOCABULARY_TAG_EDITOR_COPY = {
    "en": {
        "settings.custom_vocabulary.description": "Currently, only Soniox and Deepgram support this.",
        "settings.custom_vocabulary.add_placeholder": "",
        "settings.custom_vocabulary.add_action": "Add",
        "settings.custom_vocabulary.empty": "No hints yet.",
        "settings.custom_vocabulary.remove_hint": "Remove {term}",
    },
    "ko": {
        "settings.custom_vocabulary.description": "현재는 Soniox와 Deepgram만 지원해요.",
        "settings.custom_vocabulary.add_placeholder": "",
        "settings.custom_vocabulary.add_action": "추가",
        "settings.custom_vocabulary.empty": "아직 추가된 힌트가 없어요.",
        "settings.custom_vocabulary.remove_hint": "{term} 삭제",
    },
    "zh-CN": {
        "settings.custom_vocabulary.description": "目前仅支持 Soniox 和 Deepgram。",
        "settings.custom_vocabulary.add_placeholder": "",
        "settings.custom_vocabulary.add_action": "添加",
        "settings.custom_vocabulary.empty": "还没有添加提示。",
        "settings.custom_vocabulary.remove_hint": "删除 {term}",
    },
    "ja": {
        "settings.custom_vocabulary.description": "現在は Soniox と Deepgram のみ対応しています。",
        "settings.custom_vocabulary.add_placeholder": "",
        "settings.custom_vocabulary.add_action": "追加",
        "settings.custom_vocabulary.empty": "追加されたヒントはまだありません。",
        "settings.custom_vocabulary.remove_hint": "{term} を削除",
    },
}
EXPECTED_VAD_THRESHOLD_SECTION_COPY = {
    "en": {
        "settings.section.self_vad_sensitivity": "Your VAD Threshold",
        "settings.section.peer_vad_sensitivity": "Peer VAD Threshold",
    },
    "ko": {
        "settings.section.self_vad_sensitivity": "내 음성 감지 임계값",
        "settings.section.peer_vad_sensitivity": "상대방 음성 감지 임계값",
    },
    "ja": {
        "settings.section.self_vad_sensitivity": "自分の音声検出しきい値",
        "settings.section.peer_vad_sensitivity": "相手の音声検出しきい値",
    },
    "zh-CN": {
        "settings.section.self_vad_sensitivity": "自己的语音检测阈值",
        "settings.section.peer_vad_sensitivity": "对方语音检测阈值",
    },
}

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


def test_available_locales_use_product_display_order() -> None:
    assert available_locales() == ("en", "ko", "zh-CN", "ja", "ru")


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


def test_settings_mutation_message_keys_are_localized_for_all_supported_locales() -> None:
    bundles = _load_bundles()
    supported_locales = set(available_locales())

    assert set(bundles) == supported_locales
    for locale, bundle in bundles.items():
        missing = sorted(set(SETTINGS_MUTATION_MESSAGE_KEYS) - set(bundle))
        assert missing == [], locale
        for key in SETTINGS_MUTATION_MESSAGE_KEYS:
            assert bundle[key].strip()
            assert bundle[key] != key


def test_message_error_boundary_keys_are_localized_for_all_supported_locales() -> None:
    bundles = _load_bundles()
    supported_locales = set(available_locales())

    assert set(bundles) == supported_locales
    for locale, bundle in bundles.items():
        missing = sorted(set(MESSAGE_ERROR_BOUNDARY_KEYS) - set(bundle))
        assert missing == [], locale
        for key in MESSAGE_ERROR_BOUNDARY_KEYS:
            assert bundle[key].strip()
            assert bundle[key] != key


def test_qq_auth_copy_keys_are_localized_for_all_supported_locales() -> None:
    bundles = _load_bundles()

    for locale in ("en", "ko", "ja", "zh-CN"):
        bundle = bundles[locale]
        missing = sorted(set(QQ_AUTH_I18N_KEYS) - set(bundle))
        assert missing == [], locale
        for key in QQ_AUTH_I18N_KEYS:
            assert bundle[key].strip(), (locale, key)
            assert bundle[key] != key, (locale, key)
        assert "647594597" in bundle["qq_auth.body"]
        assert "700" in bundle["qq_auth.body"]
        assert "QQ" in bundle["qq_auth.body"]
        assert "Discord" in bundle["qq_auth.error.already_claimed_discord"]
        assert "QQ" in bundle["discord_auth.error.already_claimed_qq"]
        assert "Managed" in bundle["qq_auth.error.key_unavailable"]
        assert (
            "Translation" in bundle["qq_auth.error.key_unavailable"]
            or "번역" in bundle["qq_auth.error.key_unavailable"]
            or "翻訳" in bundle["qq_auth.error.key_unavailable"]
            or "翻译" in bundle["qq_auth.error.key_unavailable"]
        )


def test_custom_vocabulary_tag_editor_copy_is_localized_for_all_supported_locales() -> None:
    bundles = _load_bundles()
    supported_locales = set(available_locales())
    required_keys = set(CUSTOM_VOCABULARY_TAG_EDITOR_I18N_KEYS) | set(
        CUSTOM_VOCABULARY_EXISTING_I18N_KEYS
    )

    assert set(bundles) == supported_locales
    for locale, expected_copy in EXPECTED_CUSTOM_VOCABULARY_TAG_EDITOR_COPY.items():
        bundle = bundles[locale]
        missing = sorted(required_keys - set(bundle))
        assert missing == [], locale
        assert {key: bundle[key] for key in CUSTOM_VOCABULARY_TAG_EDITOR_I18N_KEYS} == expected_copy
        for key in CUSTOM_VOCABULARY_EXISTING_I18N_KEYS:
            assert bundle[key].strip()
            assert bundle[key] != key
        assert "{term}" in bundle["settings.custom_vocabulary.remove_hint"]
        assert "Soniox" in bundle["settings.custom_vocabulary.description"]
        assert "Deepgram" in bundle["settings.custom_vocabulary.description"]
        assert "Qwen" not in bundle["settings.custom_vocabulary.description"]


def test_vad_section_copy_uses_threshold_not_sensitivity() -> None:
    bundles = _load_bundles()

    for locale, expected_copy in EXPECTED_VAD_THRESHOLD_SECTION_COPY.items():
        assert {
            key: bundles[locale][key]
            for key in (
                "settings.section.self_vad_sensitivity",
                "settings.section.peer_vad_sensitivity",
            )
        } == expected_copy


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
    expected_deepseek_v4_pro_descriptions = {
        "en": "Translation speed is slow",
        "ko": "번역 속도가 느려요",
        "ja": "翻訳速度が遅いです",
        "zh-CN": "翻译速度较慢",
    }
    for locale, expected in expected_local_llm_descriptions.items():
        assert bundles[locale]["settings.translation_model.local_llm.description"] == expected
    for locale in ("en", "ko", "ja", "zh-CN"):
        assert bundles[locale]["settings.translation_model.gemini31_flash_lite.description"] == ""
    for locale, expected in expected_deepseek_v4_pro_descriptions.items():
        assert bundles[locale]["settings.translation_model.deepseek_v4_pro.description"] == expected
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
        "provider.qwen_asr": "Qwen3 ASR Flash Realtime",
    }

    for key, expected in expected_qwen_labels.items():
        assert zh_cn[key] == expected

    for value in zh_cn.values():
        assert "通义千问" not in value


def test_deepseek_v4_pro_keys_are_localized_with_blank_provider_description() -> None:
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
    expected_model_descriptions = {
        "en": "Translation speed is slow",
        "ko": "번역 속도가 느려요",
        "ja": "翻訳速度が遅いです",
        "zh-CN": "翻译速度较慢",
        "ru": "Скорость перевода низкая",
    }

    for locale, bundle in bundles.items():
        missing = sorted(required_keys - set(bundle))
        assert missing == [], locale
        present_forbidden = sorted(forbidden_keys.intersection(bundle))
        assert present_forbidden == [], locale
        assert bundle["provider.deepseek_v4_pro"].strip()
        assert bundle["provider.deepseek_v4_pro"] != "provider.deepseek_v4_pro"
        assert bundle["provider.deepseek_v4_pro.description"] == ""
        assert (
            bundle["settings.translation_model.deepseek_v4_pro.description"]
            == expected_model_descriptions[locale]
        )


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


def test_temporarily_allowed_seed_keys_are_exactly_allowlisted() -> None:
    unused_keys = _unused_i18n_keys(
        sorted(TEMPORARILY_ALLOWED_UNREFERENCED_I18N_KEYS),
        runtime_source="",
    )

    assert unused_keys == []
