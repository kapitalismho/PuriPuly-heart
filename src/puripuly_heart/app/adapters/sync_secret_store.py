from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol

from puripuly_heart.app.ports.secret_store import (
    SecretReadResult,
    SecretSnapshot,
    SecretWriteResult,
)
from puripuly_heart.core.owned_thread import run_owned_thread_call


class SyncSecretStore(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...

    def delete(self, key: str) -> None: ...


@dataclass(slots=True)
class SyncSecretStoreAdapter:
    store: SyncSecretStore

    async def get_secret(self, key: str) -> SecretReadResult:
        value = await run_owned_thread_call(lambda: self.store.get(key))
        return SecretReadResult(
            key=key,
            value=value,
            revision=None,
            message=None,
            diagnostics=None,
        )

    async def set_secret(self, key: str, value: str) -> SecretWriteResult:
        await run_owned_thread_call(lambda: self.store.set(key, value))
        return SecretWriteResult(
            succeeded=True,
            key=key,
            revision=None,
            message=None,
            diagnostics=None,
        )

    async def clear_secret(self, key: str) -> SecretWriteResult:
        await run_owned_thread_call(lambda: self.store.delete(key))
        return SecretWriteResult(
            succeeded=True,
            key=key,
            revision=None,
            message=None,
            diagnostics=None,
        )

    async def snapshot_secret(self, key: str) -> SecretSnapshot:
        value = await run_owned_thread_call(lambda: self.store.get(key))
        return SecretSnapshot(
            key=key,
            value=value,
            revision=None,
            existed=value is not None,
        )

    async def restore_secret(self, snapshot: SecretSnapshot) -> SecretWriteResult:
        if snapshot.existed and snapshot.value is not None:
            await run_owned_thread_call(partial(self.store.set, snapshot.key, snapshot.value))
        else:
            await run_owned_thread_call(lambda: self.store.delete(snapshot.key))
        return SecretWriteResult(
            succeeded=True,
            key=snapshot.key,
            revision=None,
            message=None,
            diagnostics=None,
        )


__all__ = ["SyncSecretStore", "SyncSecretStoreAdapter"]
