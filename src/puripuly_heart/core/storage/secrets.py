from __future__ import annotations

import base64
import errno
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
    service_name: str = "puripuly-heart"

    def _keyring(self):
        import keyring  # type: ignore

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
        except Exception as exc:
            errors = getattr(keyring, "errors", None)
            password_delete_error = getattr(errors, "PasswordDeleteError", None)
            if password_delete_error is not None and isinstance(exc, password_delete_error):
                if keyring.get_password(self.service_name, key) is None:
                    return
                raise
            raise


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
    _salt_b64: str

    def __init__(self, path: Path, *, passphrase: str) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with _secret_file_lock(path):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                salt_b64 = base64.b64encode(os.urandom(16)).decode("ascii")
                raw = {"version": 1, "salt": salt_b64, "items": {}}
                _atomic_write_json(path, raw)
            self._salt_b64 = raw["salt"]
            salt = base64.b64decode(self._salt_b64)
            self._fernet = Fernet(_derive_key(passphrase=passphrase, salt=salt))

    def get(self, key: str) -> str | None:
        with _secret_file_lock(self.path):
            token = self._read()["items"].get(key)
            if token is None:
                return None
            try:
                plaintext = self._fernet.decrypt(token.encode("ascii"))
            except InvalidToken as exc:
                raise ValueError("invalid passphrase or corrupted secrets file") from exc
            return plaintext.decode("utf-8")

    def set(self, key: str, value: str) -> None:
        token = self._fernet.encrypt(value.encode("utf-8")).decode("ascii")
        with _secret_file_lock(self.path):
            raw = self._read()
            raw["items"][key] = token
            _atomic_write_json(self.path, raw)

    def delete(self, key: str) -> None:
        with _secret_file_lock(self.path):
            raw = self._read()
            if key in raw["items"]:
                del raw["items"][key]
                _atomic_write_json(self.path, raw)

    def _read(self) -> dict[str, Any]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw["salt"] != self._salt_b64:
            raise ValueError("secrets file salt changed")
        return raw


@contextmanager
def _secret_file_lock(path: Path) -> Iterator[None]:
    with path.with_suffix(path.suffix + ".lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _derive_key(*, passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    key_bytes = kdf.derive(passphrase.encode("utf-8"))
    return base64.urlsafe_b64encode(key_bytes)


def _atomic_write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
