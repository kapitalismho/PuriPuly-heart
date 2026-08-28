from __future__ import annotations

from pathlib import Path

from puripuly_heart.app.wiring_managed_auth_factory import (
    ManagedAuthRuntimeAdapter,
    _managed_connection_auth_settings_values,
)

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    TranslationConnection,
)
from puripuly_heart.config.settings_vnext.migration import from_legacy_app_settings


class SecretStore:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


def _owner(settings: AppSettings) -> SettingsOwner:
    return SettingsOwner(
        path=Path("settings.json"),
        persistence=SettingsVNextCanonicalPersistenceAdapter(),
        canonical=from_legacy_app_settings(settings),
        current=settings,
        authoritative=True,
    )


def _adapter(
    settings: AppSettings,
    *,
    secret_store_factory,
    runtime_presence: tuple[bool, bool] = (True, True),
) -> ManagedAuthRuntimeAdapter:
    return ManagedAuthRuntimeAdapter(
        config_path=Path("settings.json"),
        secret_store_factory=secret_store_factory,
        settings=_owner(settings),
        release_service_provider=lambda: object(),
        settings_repository_factory=lambda _base, _committed, _surface: object(),
        runtime_presence_provider=lambda: runtime_presence,
        ingress_provider=lambda: False,
    )


def test_state_resolves_managed_selection_and_secret_through_injected_store() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation.connection = TranslationConnection.MANAGED
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    adapter = _adapter(
        settings,
        secret_store_factory=lambda *_args, **_kwargs: SecretStore(
            {"openrouter_managed_api_key": "managed-key"}
        ),
    )

    state = adapter.state()

    assert state.settings_available is True
    assert state.managed_selected is True
    assert state.managed_china is False
    assert state.local_key_available is True
    assert state.release_service_available is True
    assert state.runtime_available is True


def test_state_contains_secret_resolution_failure_as_key_unavailable() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    def fail(*_args, **_kwargs):
        raise RuntimeError("secret store unavailable")

    state = _adapter(settings, secret_store_factory=fail).state()

    assert state.managed_selected is True
    assert state.managed_china is True
    assert state.local_key_available is False


def test_transaction_settings_values_exclude_raw_secrets() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation.connection = TranslationConnection.MANAGED
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.active_managed_credential_ref = "managed-ref"

    values = _managed_connection_auth_settings_values(from_legacy_app_settings(settings))
    serialized = repr(values).lower()

    assert values["state"]["managed_connection"]["active_managed_credential_ref"] == ("managed-ref")
    assert "api_key" not in serialized
    assert "credential_value" not in serialized
    assert "secret" not in serialized
