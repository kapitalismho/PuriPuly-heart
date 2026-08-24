from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.config.paths import STABLE_APP_DIR_NAME, VNEXT_APP_DIR_NAME
from puripuly_heart.config.settings import SecretsBackend, SecretsSettings
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    KeyringSecretStore,
    SecretStore,
)

SECRETS_PASSPHRASE_ENV = "PURIPULY_HEART_SECRETS_PASSPHRASE"
STABLE_KEYRING_SERVICE_NAME = STABLE_APP_DIR_NAME
VNEXT_KEYRING_SERVICE_NAME = VNEXT_APP_DIR_NAME
VNEXT_IMPORT_SECRET_KEYS = (
    "google_api_key",
    "openrouter_api_key",
    "deepseek_api_key",
    "deepgram_api_key",
    "soniox_api_key",
    "alibaba_api_key_beijing",
    "alibaba_api_key_singapore",
    "alibaba_api_key",
    "local_llm_api_key",
    "custom_stt_api_key",
)


@dataclass(frozen=True, slots=True)
class SecretNamespaceImportResult:
    copied_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    failed_keys: tuple[str, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.failed_keys


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


def copy_stable_secrets_to_vnext_namespace(
    settings: object,
    *,
    stable_config_path: Path,
    vnext_config_path: Path,
    vnext_settings: object | None = None,
    passphrase: str | None = None,
    keys: tuple[str, ...] = VNEXT_IMPORT_SECRET_KEYS,
) -> SecretNamespaceImportResult:
    if vnext_settings is None:
        vnext_settings = settings
    if _stable_encrypted_file_missing(settings, config_path=stable_config_path):
        return SecretNamespaceImportResult(
            copied_keys=(),
            skipped_keys=keys,
            failed_keys=(),
        )
    try:
        stable_store = create_secret_store(
            settings,
            config_path=stable_config_path,
            passphrase=passphrase,
            keyring_service_name=STABLE_KEYRING_SERVICE_NAME,
        )
        vnext_store = create_secret_store(
            vnext_settings,
            config_path=vnext_config_path,
            passphrase=passphrase,
            keyring_service_name=VNEXT_KEYRING_SERVICE_NAME,
        )
    except Exception as exc:
        return SecretNamespaceImportResult(
            copied_keys=(),
            skipped_keys=keys,
            failed_keys=(),
            error=f"{type(exc).__name__}: {exc}",
        )

    copied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for key in keys:
        try:
            value = stable_store.get(key)
            if not value or vnext_store.get(key):
                skipped.append(key)
                continue
            vnext_store.set(key, value)
            copied.append(key)
        except Exception:
            failed.append(key)
    return SecretNamespaceImportResult(
        copied_keys=tuple(copied),
        skipped_keys=tuple(skipped),
        failed_keys=tuple(failed),
    )


def _secrets_backend_value(value: object) -> str:
    if isinstance(value, SecretsBackend):
        return value.value
    return str(value)


def _stable_encrypted_file_missing(settings: object, *, config_path: Path) -> bool:
    backend = _secrets_backend_value(settings.backend)
    if backend != SecretsBackend.ENCRYPTED_FILE.value:
        return False
    return not _encrypted_file_path(settings, config_path=config_path).exists()


def _encrypted_file_path(settings: object, *, config_path: Path) -> Path:
    path = Path(settings.encrypted_file_path)
    if path.is_absolute():
        return path
    return config_path.parent / path


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
