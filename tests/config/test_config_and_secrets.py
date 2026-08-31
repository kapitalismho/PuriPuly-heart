from __future__ import annotations

import json

import pytest

from puripuly_heart.config.llm_profiles import (
    LEGACY_OPENROUTER_SELECTION_ALIAS_GEMINI31_FLASH_LITE_BYOK,
    OPENROUTER_FALLBACK_SELECTION_ALIASES,
    OPENROUTER_MAIN_SELECTION_ALIASES,
    OPENROUTER_MODEL_GEMINI_37_FLASH,
    OPENROUTER_SELECTION_ALIAS_GEMINI37_FLASH_BYOK,
    get_openrouter_llm_profile,
    openrouter_alias_for_fields,
)
from puripuly_heart.config.provider_values import (
    LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS,
    LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS,
    LLMProviderName,
    LocalLLMBackend,
    OpenRouterCredentialSource,
    OpenRouterFallbackSelectionAlias,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
)
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
    default_translation_connection,
    supported_translation_connections,
)
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    KeyringSecretStore,
    mask_secret,
)


def test_translation_model_public_member_names_and_values_match_plan() -> None:
    assert tuple((member.name, member.value) for member in TranslationModel) == (
        ("GEMMA4_26B_31B", "gemma4_26b_31b"),
        ("GEMMA4_31B", "gemma4_31b"),
        ("GEMMA4", "gemma4"),
        ("DEEPSEEK_V4_FLASH", "deepseek_v4_flash"),
        ("GEMINI_37_FLASH", "gemini37_flash"),
        ("QWEN_38_FLASH", "qwen38_flash"),
        ("MANAGED_GEMMA", "managed_gemma"),
        ("MANAGED_GEMMA_12B", "managed_gemma_12b"),
        ("LOCAL_LLM", "local_llm"),
        ("CUSTOM_HTTP", "custom_http"),
    )


def test_local_llm_enum_values_are_stable() -> None:
    assert LLMProviderName.LOCAL_LLM.value == "local_llm"
    assert TranslationModel.LOCAL_LLM.value == "local_llm"
    assert TranslationConnection.OLLAMA.value == "ollama"
    assert LocalLLMBackend.OLLAMA.value == "ollama"


def test_public_translation_connection_helpers_match_model_matrix() -> None:
    assert supported_translation_connections(TranslationModel.GEMMA4) == (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
    )
    assert supported_translation_connections(TranslationModel.DEEPSEEK_V4_FLASH) == (
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
        TranslationConnection.OPENROUTER,
        TranslationConnection.OFFICIAL_BYOK,
    )
    assert supported_translation_connections(TranslationModel.GEMINI_37_FLASH) == (
        TranslationConnection.OFFICIAL_BYOK,
        TranslationConnection.OPENROUTER,
    )
    assert supported_translation_connections(TranslationModel.QWEN_38_FLASH) == (
        TranslationConnection.OFFICIAL_BYOK,
    )
    assert supported_translation_connections(TranslationModel.LOCAL_LLM) == (
        TranslationConnection.OLLAMA,
    )
    assert supported_translation_connections(TranslationModel.GEMMA4_31B) == (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
        TranslationConnection.CEREBRAS,
    )
    assert default_translation_connection(TranslationModel.GEMMA4) == TranslationConnection.MANAGED
    assert (
        default_translation_connection(TranslationModel.GEMINI_37_FLASH)
        == TranslationConnection.OFFICIAL_BYOK
    )
    assert (
        default_translation_connection(TranslationModel.LOCAL_LLM) == TranslationConnection.OLLAMA
    )
    assert (
        default_translation_connection(TranslationModel.GEMMA4_31B) == TranslationConnection.MANAGED
    )


def test_supported_translation_connections_include_ollama_for_local_model() -> None:
    assert supported_translation_connections(TranslationModel.LOCAL_LLM) == (
        TranslationConnection.OLLAMA,
    )
    assert (
        default_translation_connection(TranslationModel.LOCAL_LLM) == TranslationConnection.OLLAMA
    )


def test_local_llm_extra_body_key_sets_match_provider_constants() -> None:
    from puripuly_heart.providers.llm.local_openai import (
        LOCAL_OPENAI_RESERVED_EXTRA_BODY_KEYS,
        LOCAL_OPENAI_SENSITIVE_EXTRA_BODY_KEYS,
    )

    assert LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS == LOCAL_OPENAI_RESERVED_EXTRA_BODY_KEYS
    assert LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS == LOCAL_OPENAI_SENSITIVE_EXTRA_BODY_KEYS


def test_openrouter_alias_for_fields_does_not_expose_deepseek_v4_pro() -> None:
    assert getattr(OpenRouterLLMModel, "DEEPSEEK_V4_PRO", None) is None
    assert getattr(OpenRouterSelectionAlias, "DEEPSEEK_V4_PRO_BYOK", None) is None
    assert (
        openrouter_alias_for_fields(
            model="deepseek/deepseek-v4-pro",
            source=OpenRouterCredentialSource.BYOK.value,
        )
        is None
    )
    assert (
        openrouter_alias_for_fields(
            model="deepseek/deepseek-v4-pro",
            source=OpenRouterCredentialSource.MANAGED.value,
        )
        is None
    )
    assert get_openrouter_llm_profile("deepseek_v4_pro_byok") is None


def test_legacy_gemini_byok_alias_is_compatibility_only() -> None:
    profile = get_openrouter_llm_profile(LEGACY_OPENROUTER_SELECTION_ALIAS_GEMINI31_FLASH_LITE_BYOK)

    assert profile is not None
    assert profile.openrouter_model == OPENROUTER_MODEL_GEMINI_37_FLASH
    assert profile.openrouter_source == OpenRouterCredentialSource.BYOK.value
    assert (
        LEGACY_OPENROUTER_SELECTION_ALIAS_GEMINI31_FLASH_LITE_BYOK
        not in OPENROUTER_MAIN_SELECTION_ALIASES
    )
    assert profile.alias == LEGACY_OPENROUTER_SELECTION_ALIAS_GEMINI31_FLASH_LITE_BYOK
    assert OPENROUTER_SELECTION_ALIAS_GEMINI37_FLASH_BYOK in OPENROUTER_MAIN_SELECTION_ALIASES


def test_openrouter_fallback_aliases_include_curated_openrouter_models() -> None:
    deepseek_fallback = getattr(OpenRouterFallbackSelectionAlias, "DEEPSEEK_V4_FLASH", None)
    deepseek_china_fallback = getattr(
        OpenRouterFallbackSelectionAlias, "DEEPSEEK_V4_FLASH_CHINA", None
    )
    assert deepseek_fallback is not None
    assert deepseek_china_fallback is not None

    assert OPENROUTER_FALLBACK_SELECTION_ALIASES == (
        OpenRouterFallbackSelectionAlias.NONE.value,
        deepseek_fallback.value,
        deepseek_china_fallback.value,
        OpenRouterFallbackSelectionAlias.GEMMA4_26B_31B.value,
        OpenRouterFallbackSelectionAlias.GEMMA4_31B.value,
    )
    assert OpenRouterFallbackSelectionAlias.QWEN35_FLASH.value not in (
        OPENROUTER_FALLBACK_SELECTION_ALIASES
    )


def test_mask_secret():
    assert mask_secret("sk-123456") == "sk-****"
    assert mask_secret("abc", unmasked_prefix=3) == "***"


def test_encrypted_file_secret_store_roundtrip(tmp_path):
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="pw")
    store.set("google_api_key", "sk-SECRET")

    assert store.get("google_api_key") == "sk-SECRET"
    store.delete("google_api_key")
    assert store.get("google_api_key") is None


def test_keyring_secret_store_delete_propagates_unexpected_backend_exceptions() -> None:
    class PasswordDeleteError(Exception):
        pass

    class BrokenKeyring:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.errors = type("errors", (), {"PasswordDeleteError": PasswordDeleteError})

        def delete_password(self, service_name: str, key: str) -> None:
            self.calls.append((service_name, key))
            raise RuntimeError("keyring delete failed")

    fake_keyring = BrokenKeyring()

    class FakeKeyringSecretStore(KeyringSecretStore):
        def _keyring(self):
            return fake_keyring

    store = FakeKeyringSecretStore(service_name="test-service")

    with pytest.raises(RuntimeError, match="keyring delete failed"):
        store.delete("local_llm_api_key")

    assert fake_keyring.calls == [("test-service", "local_llm_api_key")]


def test_keyring_secret_store_delete_ignores_password_delete_error() -> None:
    class PasswordDeleteError(Exception):
        pass

    class MissingPasswordKeyring:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []
            self.errors = type("errors", (), {"PasswordDeleteError": PasswordDeleteError})

        def delete_password(self, service_name: str, key: str) -> None:
            self.calls.append(("delete", service_name, key))
            raise PasswordDeleteError("password not found")

        def get_password(self, service_name: str, key: str) -> str | None:
            self.calls.append(("get", service_name, key))
            return None

    fake_keyring = MissingPasswordKeyring()

    class FakeKeyringSecretStore(KeyringSecretStore):
        def _keyring(self):
            return fake_keyring

    store = FakeKeyringSecretStore(service_name="test-service")

    store.delete("local_llm_api_key")

    assert fake_keyring.calls == [
        ("delete", "test-service", "local_llm_api_key"),
        ("get", "test-service", "local_llm_api_key"),
    ]


def test_keyring_secret_store_delete_reraises_password_delete_error_when_secret_remains() -> None:
    class PasswordDeleteError(Exception):
        pass

    class StillPresentKeyring:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []
            self.errors = type("errors", (), {"PasswordDeleteError": PasswordDeleteError})

        def delete_password(self, service_name: str, key: str) -> None:
            self.calls.append(("delete", service_name, key))
            raise PasswordDeleteError("delete failed")

        def get_password(self, service_name: str, key: str) -> str | None:
            self.calls.append(("get", service_name, key))
            return "still-present"

    fake_keyring = StillPresentKeyring()

    class FakeKeyringSecretStore(KeyringSecretStore):
        def _keyring(self):
            return fake_keyring

    store = FakeKeyringSecretStore(service_name="test-service")

    with pytest.raises(PasswordDeleteError, match="delete failed"):
        store.delete("local_llm_api_key")

    assert fake_keyring.calls == [
        ("delete", "test-service", "local_llm_api_key"),
        ("get", "test-service", "local_llm_api_key"),
    ]


def test_encrypted_file_secret_store_does_not_store_plaintext(tmp_path):
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="pw")
    store.set("k", "sk-SECRET")

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "sk-SECRET" not in json.dumps(raw)


def test_encrypted_file_secret_store_rejects_wrong_passphrase(tmp_path):
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="pw")
    store.set("k", "sk-SECRET")

    wrong = EncryptedFileSecretStore(path, passphrase="wrong")
    with pytest.raises(ValueError):
        wrong.get("k")
