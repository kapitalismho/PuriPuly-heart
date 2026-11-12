from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from puripuly_heart.app.ports.chatgpt_account import CHATGPT_USAGE_SETTINGS_URL
from puripuly_heart.app.ports.settings_secrets import SettingsSecretKey

RECOGNITION_ALLOWANCES = MappingProxyType(
    {
        "deepgram": "trial_credit",
        "elevenlabs_scribe": "recurring_free_allowance",
        "gemini_transcribe": "ample_free_tier",
    }
)


class ProviderSetupLinkKind(StrEnum):
    KEY_CONSOLE = "key_console"
    KEY_DOCUMENTATION = "key_documentation"


@dataclass(frozen=True, slots=True)
class ProviderSetupLink:
    kind: ProviderSetupLinkKind
    url: str

    def action_id(self, provider: str) -> str:
        return f"provider.{provider}.{self.kind.value}"


@dataclass(frozen=True, slots=True)
class ProviderSetup:
    provider: str
    secret_key: SettingsSecretKey
    links: tuple[ProviderSetupLink, ...]
    browser_authorization: bool = False

    def snapshot(self) -> dict:
        result = {
            "provider": self.provider,
            "secure_field": {
                "section": "api",
                "provider": self.provider,
                "secret_key": self.secret_key.value,
                "request_kind": "provider_api_key",
                "masked": True,
                "submit_action": "save_and_verify",
                "cancel_action": "cancel",
                "settings_alternative": True,
            },
            "links": [
                {
                    "action_id": link.action_id(self.provider),
                    "kind": link.kind.value,
                    "url": link.url,
                    "requires_user_click": True,
                }
                for link in self.links
            ],
        }
        allowance = RECOGNITION_ALLOWANCES.get(
            "gemini_transcribe" if self.provider == "gemini" else self.provider
        )
        if allowance is not None:
            result["cost"] = {
                "category": "free_credit_or_allowance",
                "allowance_kind": allowance,
                "remaining": None,
                "billing_state": "unknown",
            }
        if self.browser_authorization:
            result["browser_authorization"] = {
                "method": "oauth_pkce",
                "entry": "user_action_in_app_settings",
                "requires_selected_connection": f"{self.provider}_byok",
                "applies_selection_on_success": True,
                "manual_key_fallback": True,
            }
        return result


PROVIDER_SETUPS = MappingProxyType(
    {
        setup.provider: setup
        for setup in (
            ProviderSetup(
                "openrouter",
                SettingsSecretKey.OPENROUTER_API_KEY,
                (
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_CONSOLE, "https://openrouter.ai/keys"
                    ),
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_DOCUMENTATION,
                        "https://openrouter.ai/docs/api-reference/authentication",
                    ),
                ),
                browser_authorization=True,
            ),
            ProviderSetup(
                "openai",
                SettingsSecretKey.OPENAI_API_KEY,
                (
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_CONSOLE, "https://platform.openai.com/api-keys"
                    ),
                ),
            ),
            ProviderSetup(
                "deepseek",
                SettingsSecretKey.DEEPSEEK_API_KEY,
                (
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_CONSOLE, "https://platform.deepseek.com/api_keys"
                    ),
                ),
            ),
            ProviderSetup(
                "deepgram",
                SettingsSecretKey.DEEPGRAM_API_KEY,
                (
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_CONSOLE,
                        "https://console.deepgram.com/",
                    ),
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_DOCUMENTATION,
                        "https://developers.deepgram.com/docs/create-additional-api-keys",
                    ),
                ),
            ),
            ProviderSetup(
                "gemini",
                SettingsSecretKey.GEMINI_API_KEY,
                (
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_CONSOLE,
                        "https://aistudio.google.com/apikey",
                    ),
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_DOCUMENTATION,
                        "https://ai.google.dev/gemini-api/docs/api-key",
                    ),
                ),
            ),
            ProviderSetup(
                "elevenlabs_scribe",
                SettingsSecretKey.ELEVENLABS_SCRIBE_API_KEY,
                (
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_DOCUMENTATION,
                        "https://elevenlabs.io/docs/overview/administration/workspaces/api-keys",
                    ),
                ),
            ),
            ProviderSetup(
                "soniox",
                SettingsSecretKey.SONIOX_API_KEY,
                (
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_CONSOLE, "https://console.soniox.com/"
                    ),
                ),
            ),
            ProviderSetup(
                "alibaba_beijing",
                SettingsSecretKey.ALIBABA_API_KEY_BEIJING,
                (
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_CONSOLE,
                        "https://bailian.console.aliyun.com/cn-beijing",
                    ),
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_DOCUMENTATION,
                        "https://www.alibabacloud.com/help/en/model-studio/get-api-key",
                    ),
                ),
            ),
            ProviderSetup(
                "alibaba_singapore",
                SettingsSecretKey.ALIBABA_API_KEY_SINGAPORE,
                (
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_CONSOLE,
                        "https://bailian.console.alibabacloud.com",
                    ),
                    ProviderSetupLink(
                        ProviderSetupLinkKind.KEY_DOCUMENTATION,
                        "https://www.alibabacloud.com/help/en/model-studio/get-api-key",
                    ),
                ),
            ),
        )
    }
)
NAVIGATION_PROVIDERS = tuple(PROVIDER_SETUPS)


@dataclass(frozen=True, slots=True)
class AccountSetupAction:
    id: str
    provider: str
    kind: str
    url: str | None = None

    def snapshot(self) -> dict:
        return {
            "action_id": self.id,
            "provider": self.provider,
            "kind": self.kind,
            "requires_user_click": True,
            "credentials_owned_by": "existing_app_auth",
        }


ACCOUNT_SETUP_ACTIONS = MappingProxyType(
    {
        action.id: action
        for action in (
            AccountSetupAction("account.chatgpt.setup", "chatgpt", "account_setup"),
            AccountSetupAction("account.qq.setup", "qq", "account_setup"),
            AccountSetupAction("account.discord.setup", "discord", "account_setup"),
            AccountSetupAction(
                "account.chatgpt.usage", "chatgpt", "public_usage", CHATGPT_USAGE_SETTINGS_URL
            ),
        )
    }
)


def official_link_destination(action_id: str) -> str | None:
    for setup in PROVIDER_SETUPS.values():
        for link in setup.links:
            if link.action_id(setup.provider) == action_id:
                return link.url
    action = ACCOUNT_SETUP_ACTIONS.get(action_id)
    return action.url if action is not None else None
