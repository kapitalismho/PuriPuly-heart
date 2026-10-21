from __future__ import annotations

import contextlib
import os
from pathlib import Path

from puripuly_heart.config.paths import STABLE_APP_DIR_NAME
from puripuly_heart.config.provider_values import SecretsBackend
from puripuly_heart.config.settings_vnext.schema import SecretsIntent
from puripuly_heart.core.diagnostics import safe_exception_fields
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    KeyringSecretStore,
    SecretStore,
)

SECRETS_PASSPHRASE_ENV = "PURIPULY_HEART_SECRETS_PASSPHRASE"
STABLE_KEYRING_SERVICE_NAME = STABLE_APP_DIR_NAME


class SecretStoreCreationError(ValueError):
    def __init__(self, code: str, exception: BaseException | None = None) -> None:
        self.diagnostics = ErrorDiagnostics(
            component="settings_secrets",
            operation="create_store",
            code=code,
            category="transaction",
            visibility="basic",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={
                "phase": "create_store",
                **(safe_exception_fields(exception) if exception is not None else {}),
            },
        )
        super().__init__(code)


def create_secret_store(
    settings: SecretsIntent,
    *,
    config_path: Path,
    passphrase: str | None = None,
    keyring_service_name: str = STABLE_KEYRING_SERVICE_NAME,
) -> SecretStore:
    passphrase = passphrase or os.getenv(SECRETS_PASSPHRASE_ENV)

    backend = _secrets_backend_value(settings.backend)
    if backend == SecretsBackend.KEYRING.value:
        try:
            return KeyringSecretStore(service_name=keyring_service_name)
        except Exception as exc:
            raise SecretStoreCreationError("settings_secret_failed", exc) from exc

    if backend == SecretsBackend.ENCRYPTED_FILE.value:
        if not passphrase:
            raise SecretStoreCreationError("secret_store_passphrase_missing")
        path = Path(settings.encrypted_file_path)
        if not path.is_absolute():
            path = config_path.parent / path
        try:
            return EncryptedFileSecretStore(path=path, passphrase=passphrase)
        except Exception as exc:
            raise SecretStoreCreationError("settings_secret_failed", exc) from exc

    raise SecretStoreCreationError("secret_store_backend_unsupported")


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
