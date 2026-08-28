from __future__ import annotations

import pytest

from puripuly_heart.app import wiring_secrets_factory
from puripuly_heart.app.wiring import create_secret_store
from puripuly_heart.config.provider_values import SecretsBackend
from puripuly_heart.config.settings_vnext.schema import SecretsIntent
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    KeyringSecretStore,
)


def test_create_secret_store_keyring_returns_keyring_store(tmp_path):
    store = create_secret_store(
        SecretsIntent(backend=SecretsBackend.KEYRING.value),
        config_path=tmp_path / "settings.json",
    )

    assert isinstance(store, KeyringSecretStore)
    assert store.service_name == wiring_secrets_factory.STABLE_KEYRING_SERVICE_NAME


def test_create_secret_store_encrypted_file_resolves_relative_path(tmp_path):
    store = create_secret_store(
        SecretsIntent(
            backend=SecretsBackend.ENCRYPTED_FILE.value,
            encrypted_file_path="secrets.json",
        ),
        config_path=tmp_path / "settings.json",
        passphrase="pw",
    )

    assert isinstance(store, EncryptedFileSecretStore)
    assert store.path == tmp_path / "secrets.json"


def test_create_secret_store_encrypted_file_requires_passphrase(tmp_path):
    with pytest.raises(ValueError):
        create_secret_store(
            SecretsIntent(
                backend=SecretsBackend.ENCRYPTED_FILE.value,
                encrypted_file_path="secrets.json",
            ),
            config_path=tmp_path / "settings.json",
        )
