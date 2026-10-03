from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REQUIRED_DISCORD_AUTH_KEYS = [
    "discord_auth.body",
    "discord_auth.continue",
    "discord_auth.cancel",
    "discord_auth.waiting_body",
    "discord_auth.callback_received_body",
    "discord_auth.recovering_body",
    "discord_auth.action_required_body",
    "discord_auth.success",
    "discord_auth.referral_reward_applied",
    "discord_auth.error.email_unverified",
    "discord_auth.error.account_too_new",
    "discord_auth.error.lifetime_used",
    "discord_auth.error.hardware_duplicate",
    "discord_auth.error.daily_cap",
    "discord_auth.error.expired",
    "discord_auth.error.loopback_unavailable",
    "discord_auth.error.retry",
    "discord_auth.error.recovery_pending",
    "discord_auth.error.action_required",
    "discord_auth.error.authorization_expired",
    "debug_preview.discord_auth",
]


_EXPECTED_EXACT_STRINGS = {
    "en": {
        "discord_auth.body": "PuriPuly uses AI for high-quality translation.\nA quick verification is all it takes to start.\n\nThere are two ways:\n1. Verify via Discord: free tokens for 800 uses\n2. Sign in with ChatGPT: requires Plus or higher\n\nYou can try both, so no need to overthink it.\nThe developer pays for the tokens personally.\nWe don't keep personal information.",
        "discord_auth.success": "Discord verification is complete.",
    },
    "ko": {
        "discord_auth.body": "PuriPuly는 고품질 번역을 위해 AI를 사용해요.\n간단한 인증을 통해서 사용할 수 있어요.\n\n두가지 방법이 있어요.\n1. Discord로 인증: 800회 분량의 토큰을 무료로 지급\n2. ChatGPT로 로그인: Plus 구독 이상 필요\n\n둘 다 해볼 수 있으니 고민하지 않아도 되어요.\n토큰은 개발자의 사비로 지불합니다.\n개인 정보는 보관하지 않아요.",
        "discord_auth.success": "Discord 인증이 완료되었어요.",
    },
    "ja": {
        "discord_auth.body": "PuriPulyは高品質な翻訳のためにAIを使っています。\n簡単な認証で使えるようになります。\n\n方法は2つあります。\n1. Discordで認証：800回分のトークンを無料で付与\n2. ChatGPTでログイン：Plus以上のプランが必要\n\nどちらも試せるので、迷わなくて大丈夫です。\nトークンは開発者が自費で負担しています。\n個人情報は保存しません。",
        "discord_auth.success": "Discord認証が完了しました。",
    },
    "zh-CN": {
        "discord_auth.body": "PuriPuly 使用 AI 提供高质量翻译。\n只需简单认证即可使用。\n\n有两种方式：\n1. 使用 Discord 认证：免费获得 800 次用量的 Token\n2. 使用 ChatGPT 登录：需要 Plus 及以上订阅\n\n两种都可以尝试，不用纠结。\nToken 由开发者自费承担。\n我们不会保存个人信息。",
        "discord_auth.success": "Discord 认证已完成。",
    },
}

_FORBIDDEN_DISCORD_AUTH_COPY_PATTERNS = {
    "currency or dollar amounts": re.compile(r"(?:\$|USD|usd|dollars?|달러|원|円|엔|美元|美金)"),
    "referral terminology": re.compile(r"Referral", re.IGNORECASE),
}

_EXPECTED_TALK_TOGETHER_PASS_STRINGS = {
    "en": {
        "discord_auth.referral_reward_applied": "You and your friend got 200 extra uses.",
    },
    "ko": {
        "discord_auth.referral_reward_applied": "친구와 함께 200회 추가 사용량을 받았어요.",
    },
    "ja": {
        "discord_auth.referral_reward_applied": "友だちと一緒に200回分の追加使用量を受け取りました。",
    },
    "zh-CN": {
        "discord_auth.referral_reward_applied": "你和朋友已获得 200 次额外使用量。",
    },
}


def _load_bundle(locale: str) -> dict[str, str]:
    i18n_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "puripuly_heart"
        / "data"
        / "i18n"
        / f"{locale}.json"
    )
    return json.loads(i18n_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("locale", ["en", "ko", "zh-CN", "ja"])
def test_discord_auth_i18n_keys_exist_and_are_not_empty(locale: str) -> None:
    bundle = _load_bundle(locale)

    missing = [key for key in REQUIRED_DISCORD_AUTH_KEYS if key not in bundle]
    empty = [key for key in REQUIRED_DISCORD_AUTH_KEYS if bundle.get(key) == ""]

    assert missing == []
    assert empty == []


@pytest.mark.parametrize("locale", ["en", "ko", "zh-CN", "ja"])
def test_discord_auth_i18n_uses_planned_title_body_and_success_copy(
    locale: str,
) -> None:
    bundle = _load_bundle(locale)

    for key, expected_value in _EXPECTED_EXACT_STRINGS[locale].items():
        assert bundle[key] == expected_value


@pytest.mark.parametrize("locale", ["en", "ko", "zh-CN", "ja"])
def test_discord_auth_copy_uses_pass_terms_without_referral_or_currency(locale: str) -> None:
    bundle = _load_bundle(locale)
    checked_copy = "\n".join(
        [
            bundle["discord_auth.body"],
            bundle["discord_auth.referral_reward_applied"],
        ]
    )

    violations = {
        label: pattern.pattern
        for label, pattern in _FORBIDDEN_DISCORD_AUTH_COPY_PATTERNS.items()
        if pattern.search(checked_copy)
    }

    assert violations == {}


@pytest.mark.parametrize("locale", ["en", "ko", "zh-CN", "ja"])
def test_discord_auth_talk_together_pass_i18n_uses_planned_copy(locale: str) -> None:
    bundle = _load_bundle(locale)

    for key, expected_value in _EXPECTED_TALK_TOGETHER_PASS_STRINGS[locale].items():
        assert bundle[key] == expected_value
