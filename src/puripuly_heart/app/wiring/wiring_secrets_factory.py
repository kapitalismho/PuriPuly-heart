from __future__ import annotations

import contextlib
import os
from pathlib import Path

from puripuly_heart.config.paths import STABLE_APP_DIR_NAME
from puripuly_heart.config.settings import SecretsBackend, SecretsSettings
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    KeyringSecretStore,
    SecretStore,
)

SECRETS_PASSPHRASE_ENV = "PURIPULY_HEART_SECRETS_PASSPHRASE"
STABLE_KEYRING_SERVICE_NAME = STABLE_APP_DIR_NAME


def create_secret_store(
    settings: SecretsSettings,
    *,
    config_path: Path,
    passphrase: str | None = None,
    keyring_service_name: str = STABLE_KEYRING_SERVICE_NAME,
) -> SecretStore:
    passphrase = passphrase or os.getenv(SECRETS_PASSPHRASE_ENV)

    backend = _secrets_backend_value(settings.backend)
    if backend == SecretsBackend.KEYRING.value:
        return KeyringSecretStore(service_name=keyring_service_name)

    if backend == SecretsBackend.ENCRYPTED_FILE.value:
        if not passphrase:
            raise ValueError(
                "encrypted_file secrets backend requires a passphrase; "
                f"set {SECRETS_PASSPHRASE_ENV} or pass passphrase explicitly"
            )
        path = Path(settings.encrypted_file_path)
        if not path.is_absolute():
            path = config_path.parent / path
        return EncryptedFileSecretStore(path=path, passphrase=passphrase)

    raise ValueError(f"Unsupported secrets backend: {settings.backend}")


def _secrets_backend_value(value: object) -> str:
    if isinstance(value, SecretsBackend):
        return value.value
    return str(value)


def _get_secret(
    secrets: SecretStore,
    *,
    key: str,
    env_var: str,
) -> str | None:
    value = secrets.get(key)
    if value:
        return value
    env = os.getenv(env_var)
    if env:
        return env
    return None


def _get_secret_any(
    secrets: SecretStore,
    *,
    key: str,
    env_vars: tuple[str, ...],
    legacy_keys: tuple[str, ...] = (),
) -> str | None:
    value = secrets.get(key)
    if value:
        return value
    for legacy_key in legacy_keys:
        legacy_value = secrets.get(legacy_key)
        if legacy_value:
            # Backfill to the new key so subsequent runs do not rely on fallback.
            with contextlib.suppress(Exception):
                secrets.set(key, legacy_value)
            return legacy_value
    for env_var in env_vars:
        env = os.getenv(env_var)
        if env:
            return env
    return None


def require_secret_any(
    secrets: SecretStore,
    *,
    key: str,
    env_vars: tuple[str, ...],
    legacy_keys: tuple[str, ...] = (),
) -> str:
    value = _get_secret_any(secrets, key=key, env_vars=env_vars, legacy_keys=legacy_keys)
    if value:
        return value
    env_list = ", ".join(env_vars)
    raise ValueError(f"Missing secret `{key}` (or env vars {env_list})")


def require_secret(
    secrets: SecretStore,
    *,
    key: str,
    env_var: str,
) -> str:
    value = _get_secret(secrets, key=key, env_var=env_var)
    if value:
        return value
    raise ValueError(f"Missing secret `{key}` (or env var {env_var})")
