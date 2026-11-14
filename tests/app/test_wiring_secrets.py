from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

from puripuly_heart.app.wiring import create_secret_store, wiring_secrets_factory
from puripuly_heart.config.provider_values import SecretsBackend
from puripuly_heart.config.settings_vnext.schema import SecretsIntent
from puripuly_heart.core.storage import secrets
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    KeyringSecretStore,
)


class SyntheticKeyring:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], str] = {}
        self.calls: list[tuple[str, str]] = []

    def get_password(self, service: str, key: str) -> str | None:
        self.calls.append(("get", key))
        return self.items.get((service, key))

    def set_password(self, service: str, key: str, value: str) -> None:
        self.calls.append(("set", key))
        self.items[service, key] = value

    def delete_password(self, service: str, key: str) -> None:
        self.calls.append(("delete", key))
        self.items.pop((service, key), None)


@pytest.fixture(autouse=True)
def synthetic_keyring(monkeypatch):
    backend = SyntheticKeyring()
    monkeypatch.setattr(KeyringSecretStore, "_keyring", lambda self: backend)
    return backend


def _seed_store(tmp_path: Path, backend: str):
    if backend == SecretsBackend.KEYRING.value:
        return KeyringSecretStore()
    return EncryptedFileSecretStore(tmp_path / "secrets.json", passphrase="synthetic")


def _create_store(tmp_path: Path, backend: str = SecretsBackend.ENCRYPTED_FILE.value):
    return create_secret_store(
        SecretsIntent(backend=backend, encrypted_file_path="secrets.json"),
        config_path=tmp_path / "settings.json",
        passphrase="synthetic",
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


@pytest.mark.parametrize("backend", ["encrypted_file", "keyring"])
@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        ({}, None),
        ({"gemini_transcribe_api_key": "synthetic-stt"}, "synthetic-stt"),
        ({"google_api_key": "synthetic-retired"}, None),
        (
            {
                "gemini_transcribe_api_key": "synthetic-stt",
                "google_api_key": "synthetic-retired",
            },
            "synthetic-stt",
        ),
        (
            {
                "gemini_api_key": "synthetic-canonical",
                "gemini_transcribe_api_key": "synthetic-stt",
                "google_api_key": "synthetic-retired",
            },
            "synthetic-canonical",
        ),
        (
            {
                "gemini_api_key": "",
                "gemini_transcribe_api_key": "synthetic-stt",
                "google_api_key": "synthetic-retired",
            },
            "synthetic-stt",
        ),
        (
            {"gemini_transcribe_api_key": "", "google_api_key": "synthetic-retired"},
            None,
        ),
    ],
)
def test_factory_migrates_only_transcribe_and_retires_old_keys_idempotently(
    tmp_path, backend, existing, expected, synthetic_keyring
):
    seeded = _seed_store(tmp_path, backend)
    seeded.set("unrelated", "synthetic-unrelated")
    for key, value in existing.items():
        seeded.set(key, value)

    store = _create_store(tmp_path, backend)

    assert type(store) is type(seeded)
    assert store.get("gemini_api_key") == expected
    assert seeded.get("gemini_api_key") == expected
    assert store.get("gemini_transcribe_api_key") is None
    assert store.get("google_api_key") is None
    assert store.get("unrelated") == "synthetic-unrelated"
    if backend == "encrypted_file":
        before_restart = store.path.read_bytes()
    synthetic_keyring.calls.clear()

    restarted = _create_store(tmp_path, backend)

    assert restarted.get("gemini_api_key") == expected
    assert restarted.get("gemini_transcribe_api_key") is None
    assert restarted.get("google_api_key") is None
    assert restarted.get("unrelated") == "synthetic-unrelated"
    if backend == "encrypted_file":
        assert store.path.read_bytes() == before_restart
    else:
        assert all(operation == "get" for operation, _ in synthetic_keyring.calls)


def test_encrypted_migration_survives_fresh_process_restart(tmp_path):
    store = _seed_store(tmp_path, "encrypted_file")
    store.set("gemini_transcribe_api_key", "synthetic-stt")
    store.set("google_api_key", "synthetic-retired")
    store.set("unrelated", "synthetic-unrelated")
    _create_store(tmp_path)
    code = """
import json
import sys
from pathlib import Path
from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
from puripuly_heart.config.settings_vnext.schema import SecretsIntent
store = create_secret_store(
    SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
    config_path=Path(sys.argv[1]) / "settings.json",
    passphrase="synthetic",
)
keys = ("gemini_api_key", "gemini_transcribe_api_key", "google_api_key", "unrelated")
print(json.dumps({key: store.get(key) for key in keys}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        timeout=30,
        check=True,
    )

    assert json.loads(result.stdout) == {
        "gemini_api_key": "synthetic-stt",
        "gemini_transcribe_api_key": None,
        "google_api_key": None,
        "unrelated": "synthetic-unrelated",
    }


def test_encrypted_migration_failed_canonical_replace_preserves_sources(tmp_path, monkeypatch):
    store = _seed_store(tmp_path, "encrypted_file")
    store.set("gemini_transcribe_api_key", "synthetic-stt")
    store.set("google_api_key", "synthetic-retired")
    store.set("unrelated", "synthetic-unrelated")
    before = store.path.read_bytes()

    def denied(source, target):
        raise PermissionError(13, "synthetic-sensitive-backend-message")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", denied)
        with pytest.raises(wiring_secrets_factory.SecretStoreCreationError) as captured:
            _create_store(tmp_path)

    assert captured.value.diagnostics.fields["exception_class"] == "PermissionError"
    assert "synthetic-sensitive-backend-message" not in repr(captured.value.diagnostics)
    assert "synthetic-sensitive-backend-message" not in "".join(
        traceback.format_exception(captured.value)
    )
    assert store.path.read_bytes() == before
    assert store.get("gemini_api_key") is None
    assert store.get("gemini_transcribe_api_key") == "synthetic-stt"
    assert store.get("google_api_key") == "synthetic-retired"
    assert store.get("unrelated") == "synthetic-unrelated"
    assert _create_store(tmp_path).get("gemini_api_key") == "synthetic-stt"


def test_encrypted_migration_commits_canonical_before_cleanup_and_can_retry(tmp_path, monkeypatch):
    store = _seed_store(tmp_path, "encrypted_file")
    store.set("gemini_transcribe_api_key", "synthetic-stt")
    store.set("google_api_key", "synthetic-retired")
    store.set("unrelated", "synthetic-unrelated")
    original = secrets._atomic_write_json

    def fail_cleanup(path, data):
        if "gemini_transcribe_api_key" not in data["items"]:
            raise PermissionError("synthetic cleanup denied")
        original(path, data)

    with monkeypatch.context() as scoped:
        scoped.setattr(secrets, "_atomic_write_json", fail_cleanup)
        with pytest.raises(wiring_secrets_factory.SecretStoreCreationError):
            _create_store(tmp_path)

    fresh = _seed_store(tmp_path, "encrypted_file")
    assert fresh.get("gemini_api_key") == "synthetic-stt"
    assert fresh.get("gemini_transcribe_api_key") == "synthetic-stt"
    assert fresh.get("google_api_key") == "synthetic-retired"
    assert fresh.get("unrelated") == "synthetic-unrelated"
    canonical_token = json.loads(store.path.read_text(encoding="utf-8"))["items"]["gemini_api_key"]

    restarted = _create_store(tmp_path)

    assert restarted.get("gemini_api_key") == "synthetic-stt"
    assert restarted.get("gemini_transcribe_api_key") is None
    assert restarted.get("google_api_key") is None
    assert json.loads(store.path.read_text(encoding="utf-8"))["items"]["gemini_api_key"] == (
        canonical_token
    )


@pytest.mark.parametrize(
    "damage",
    ["wrong_passphrase", "gemini_api_key", "gemini_transcribe_api_key", "google_api_key"],
)
def test_encrypted_migration_read_failures_do_not_change_file(tmp_path, damage):
    store = _seed_store(tmp_path, "encrypted_file")
    for key in ("gemini_api_key", "gemini_transcribe_api_key", "google_api_key", "unrelated"):
        store.set(key, f"synthetic-{key}")
    if damage != "wrong_passphrase":
        raw = json.loads(store.path.read_text(encoding="utf-8"))
        raw["items"][damage] = "invalid-synthetic-token"
        store.path.write_text(json.dumps(raw), encoding="utf-8")
    before = store.path.read_bytes()

    with pytest.raises(wiring_secrets_factory.SecretStoreCreationError) as captured:
        create_secret_store(
            SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            config_path=tmp_path / "settings.json",
            passphrase="wrong-synthetic" if damage == "wrong_passphrase" else "synthetic",
        )

    assert captured.value.diagnostics.fields["exception_class"] == "ValueError"
    assert store.path.read_bytes() == before
    assert store.get("unrelated") == "synthetic-unrelated"


@pytest.mark.parametrize(
    "failed_key", ["gemini_api_key", "gemini_transcribe_api_key", "google_api_key"]
)
def test_keyring_migration_read_failure_preserves_all_entries(
    tmp_path, monkeypatch, synthetic_keyring, failed_key
):
    store = _seed_store(tmp_path, "keyring")
    store.set("gemini_transcribe_api_key", "synthetic-stt")
    store.set("google_api_key", "synthetic-retired")
    store.set("unrelated", "synthetic-unrelated")
    before = dict(synthetic_keyring.items)
    original = synthetic_keyring.get_password

    def fail_read(service, key):
        if key == failed_key:
            raise PermissionError(13, "synthetic-sensitive-backend-message")
        return original(service, key)

    monkeypatch.setattr(synthetic_keyring, "get_password", fail_read)
    synthetic_keyring.calls.clear()
    with pytest.raises(wiring_secrets_factory.SecretStoreCreationError) as captured:
        _create_store(tmp_path, "keyring")

    assert synthetic_keyring.items == before
    assert all(operation == "get" for operation, _ in synthetic_keyring.calls)
    assert "synthetic-sensitive-backend-message" not in repr(captured.value.diagnostics)
    assert "synthetic-sensitive-backend-message" not in "".join(
        traceback.format_exception(captured.value)
    )


@pytest.mark.parametrize("failure", ["raises", "not_retained", "readback"])
def test_keyring_migration_requires_durable_canonical_before_deleting_sources(
    tmp_path, monkeypatch, synthetic_keyring, failure
):
    store = _seed_store(tmp_path, "keyring")
    store.set("gemini_transcribe_api_key", "synthetic-stt")
    store.set("google_api_key", "synthetic-retired")
    original_set = synthetic_keyring.set_password
    original_get = synthetic_keyring.get_password

    def fail_write(service, key, value):
        if failure == "raises":
            raise PermissionError("synthetic write denied")
        if failure == "readback":
            original_set(service, key, value)

    def fail_readback(service, key):
        if key == "gemini_api_key" and (service, key) in synthetic_keyring.items:
            raise PermissionError("synthetic readback denied")
        return original_get(service, key)

    with monkeypatch.context() as scoped:
        scoped.setattr(synthetic_keyring, "set_password", fail_write)
        if failure == "readback":
            scoped.setattr(synthetic_keyring, "get_password", fail_readback)
        synthetic_keyring.calls.clear()
        with pytest.raises(wiring_secrets_factory.SecretStoreCreationError):
            _create_store(tmp_path, "keyring")

    assert not any(operation == "delete" for operation, _ in synthetic_keyring.calls)
    assert store.get("gemini_transcribe_api_key") == "synthetic-stt"
    assert store.get("google_api_key") == "synthetic-retired"
    expected_canonical = "synthetic-stt" if failure == "readback" else None
    assert store.get("gemini_api_key") == expected_canonical
    assert _create_store(tmp_path, "keyring").get("gemini_api_key") == "synthetic-stt"


@pytest.mark.parametrize("failed_key", ["gemini_transcribe_api_key", "google_api_key"])
def test_keyring_migration_cleanup_failure_preserves_canonical_and_can_retry(
    tmp_path, monkeypatch, synthetic_keyring, failed_key
):
    store = _seed_store(tmp_path, "keyring")
    store.set("gemini_transcribe_api_key", "synthetic-stt")
    store.set("google_api_key", "synthetic-retired")
    store.set("unrelated", "synthetic-unrelated")
    original = synthetic_keyring.delete_password

    def fail_delete(service, key):
        assert store.get("gemini_api_key") == "synthetic-stt"
        if key == failed_key:
            raise PermissionError("synthetic cleanup denied")
        original(service, key)

    with monkeypatch.context() as scoped:
        scoped.setattr(synthetic_keyring, "delete_password", fail_delete)
        with pytest.raises(wiring_secrets_factory.SecretStoreCreationError):
            _create_store(tmp_path, "keyring")

    assert store.get("gemini_api_key") == "synthetic-stt"
    assert store.get(failed_key) == (
        "synthetic-stt" if failed_key == "gemini_transcribe_api_key" else "synthetic-retired"
    )
    assert store.get("unrelated") == "synthetic-unrelated"
    restarted = _create_store(tmp_path, "keyring")
    assert restarted.get("gemini_api_key") == "synthetic-stt"
    assert restarted.get("gemini_transcribe_api_key") is None
    assert restarted.get("google_api_key") is None
