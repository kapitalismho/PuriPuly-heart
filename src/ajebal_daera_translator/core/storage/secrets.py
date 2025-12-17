from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class SecretStore(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    def delete(self, key: str) -> None: ...


@dataclass(slots=True)
class InMemorySecretStore:
    _items: dict[str, str]

    def __init__(self) -> None:
        self._items = {}

    def get(self, key: str) -> str | None:
        return self._items.get(key)

    def set(self, key: str, value: str) -> None:
        self._items[key] = value

    def delete(self, key: str) -> None:
        self._items.pop(key, None)


@dataclass(slots=True)
class KeyringSecretStore:
    service_name: str = "ajebal-daera-translator"

    def _keyring(self):
        try:
            import keyring  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "keyring is required for KeyringSecretStore; install with `pip install keyring`"
            ) from exc
        return keyring

    def get(self, key: str) -> str | None:
        keyring = self._keyring()
        return keyring.get_password(self.service_name, key)

    def set(self, key: str, value: str) -> None:
        keyring = self._keyring()
        keyring.set_password(self.service_name, key, value)

    def delete(self, key: str) -> None:
        keyring = self._keyring()
        try:
            keyring.delete_password(self.service_name, key)
        except Exception:
            return


def mask_secret(value: str, *, unmasked_prefix: int = 3) -> str:
    if not value:
        return value
    if len(value) <= unmasked_prefix:
        return "*" * len(value)
    return value[:unmasked_prefix] + "****"


@dataclass(slots=True)
class EncryptedFileSecretStore:
    path: Path
    _fernet: Fernet
    _items: dict[str, str]

    def __init__(self, path: Path, *, passphrase: str) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            salt_b64 = raw["salt"]
            items = raw.get("items", {})
        else:
            salt = os.urandom(16)
            salt_b64 = base64.b64encode(salt).decode("ascii")
            items = {}
            _atomic_write_json(path, {"version": 1, "salt": salt_b64, "items": items})

        salt = base64.b64decode(salt_b64)
        key = _derive_key(passphrase=passphrase, salt=salt)
        self._fernet = Fernet(key)
        self._items = dict(items)

    def get(self, key: str) -> str | None:
        token = self._items.get(key)
        if token is None:
            return None
        try:
            plaintext = self._fernet.decrypt(token.encode("ascii"))
        except InvalidToken as exc:
            raise ValueError("invalid passphrase or corrupted secrets file") from exc
        return plaintext.decode("utf-8")

    def set(self, key: str, value: str) -> None:
        token = self._fernet.encrypt(value.encode("utf-8")).decode("ascii")
        self._items[key] = token
        self._save()

    def delete(self, key: str) -> None:
        if key in self._items:
            del self._items[key]
            self._save()

    def _save(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["items"] = self._items
        _atomic_write_json(self.path, raw)


def _derive_key(*, passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    key_bytes = kdf.derive(passphrase.encode("utf-8"))
    return base64.urlsafe_b64encode(key_bytes)


def _atomic_write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

