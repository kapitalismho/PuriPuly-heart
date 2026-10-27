from __future__ import annotations

import json
from pathlib import Path

import pytest

REQUIRED_AUTH_KEYS = [
    "discord_auth.body",
    "discord_auth.continue",
    "discord_auth.cancel",
    "discord_auth.close",
    "discord_auth.retry",
    "discord_auth.resume",
    "discord_auth.waiting_body",
    "discord_auth.callback_received_body",
    "discord_auth.recovering_body",
    "discord_auth.action_required_body",
    "discord_auth.success",
    "discord_auth.error.email_unverified",
    "discord_auth.error.account_too_new",
    "discord_auth.error.lifetime_used",
    "discord_auth.error.hardware_duplicate",
    "discord_auth.error.daily_cap",
    "discord_auth.error.expired",
    "discord_auth.error.loopback_unavailable",
    "discord_auth.error.browser_launch_failed",
    "discord_auth.error.network_configuration",
    "discord_auth.error.translation_start_failed",
    "discord_auth.error.retry",
    "discord_auth.error.recovery_pending",
    "discord_auth.error.action_required",
    "discord_auth.error.authorization_expired",
    "qq_auth.retry",
    "qq_auth.resume",
    "qq_auth.error.network_configuration",
    "qq_auth.error.translation_start_failed",
    "qq_auth.error.delivery_ack_pending",
    "qq_auth.error.recovery_pending",
    "debug_preview.discord_auth",
    "settings.managed_key.title",
    "settings.managed_trial_usage.remaining",
]


def _load_bundle(locale: str) -> dict[str, str]:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "puripuly_heart"
        / "data"
        / "i18n"
        / f"{locale}.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("locale", ["en", "ko", "zh-CN", "ja", "ru"])
def test_auth_transitions_have_translated_actions_and_errors(locale: str) -> None:
    bundle = _load_bundle(locale)
    assert all(isinstance(bundle.get(key), str) and bundle[key] for key in REQUIRED_AUTH_KEYS)
