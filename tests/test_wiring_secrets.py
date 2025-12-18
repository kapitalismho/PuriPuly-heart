from __future__ import annotations

from pathlib import Path

import pytest

from ajebal_daera_translator.app.wiring import create_secret_store
from ajebal_daera_translator.config.settings import SecretsBackend, SecretsSettings
from ajebal_daera_translator.core.storage.secrets import (
    EncryptedFileSecretStore,
    KeyringSecretStore,
)


def test_create_secret_store_keyring_returns_keyring_store(tmp_path):
    store = create_secret_store(
        SecretsSettings(backend=SecretsBackend.KEYRING),
        config_path=tmp_path / "settings.json",
    )

    assert isinstance(store, KeyringSecretStore)


def test_create_secret_store_encrypted_file_resolves_relative_path(tmp_path):
    store = create_secret_store(
        SecretsSettings(backend=SecretsBackend.ENCRYPTED_FILE, encrypted_file_path="secrets.json"),
        config_path=tmp_path / "settings.json",
        passphrase="pw",
    )

    assert isinstance(store, EncryptedFileSecretStore)
    assert store.path == tmp_path / "secrets.json"


def test_create_secret_store_encrypted_file_requires_passphrase(tmp_path):
    with pytest.raises(ValueError):
        create_secret_store(
            SecretsSettings(backend=SecretsBackend.ENCRYPTED_FILE, encrypted_file_path="secrets.json"),
            config_path=tmp_path / "settings.json",
        )
