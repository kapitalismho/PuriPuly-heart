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


def test_create_secret_store_encrypted_file_requires_passphrase(tmp_path, monkeypatch):
    monkeypatch.delenv(wiring_secrets_factory.SECRETS_PASSPHRASE_ENV, raising=False)
    with pytest.raises(ValueError) as captured:
        create_secret_store(
            SecretsIntent(
                backend=SecretsBackend.ENCRYPTED_FILE.value,
                encrypted_file_path="secrets.json",
            ),
            config_path=tmp_path / "settings.json",
        )
    assert captured.value.diagnostics.code == "secret_store_passphrase_missing"
    assert captured.value.diagnostics.fields["phase"] == "create_store"


def test_create_secret_store_native_failure_retains_numeric_cause(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise PermissionError(13, "synthetic_secret_path_password")

    monkeypatch.setattr(wiring_secrets_factory, "EncryptedFileSecretStore", fail)
    with pytest.raises(ValueError) as captured:
        create_secret_store(
            SecretsIntent(backend="encrypted_file", encrypted_file_path="isolated.json"),
            config_path=tmp_path / "settings.json",
            passphrase="synthetic",
        )
    diagnostic = captured.value.diagnostics
    assert diagnostic.fields["exception_class"] == "PermissionError"
    assert diagnostic.fields["errno"] == 13
    assert "synthetic_secret_path_password" not in repr(diagnostic)
