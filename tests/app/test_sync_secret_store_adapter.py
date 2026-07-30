from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from puripuly_heart.app.adapters.sync_secret_store import SyncSecretStoreAdapter
from puripuly_heart.app.ports.secret_store import SecretSnapshot
from puripuly_heart.app.wiring import create_sync_secret_store_adapter


@dataclass
class MemorySyncSecretStore:
    values: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


@pytest.mark.asyncio
async def test_adapter_maps_sync_secret_store_to_async_port_contract() -> None:
    store = MemorySyncSecretStore({"provider": "old"})
    adapter = SyncSecretStoreAdapter(store)

    read = await adapter.get_secret("provider")
    snapshot = await adapter.snapshot_secret("provider")
    write = await adapter.set_secret("provider", "new")
    clear = await adapter.clear_secret("provider")

    assert read.value == "old"
    assert read.key == "provider"
    assert snapshot == SecretSnapshot(
        key="provider",
        value="old",
        revision=None,
        existed=True,
    )
    assert write.succeeded is True
    assert clear.succeeded is True
    assert store.values == {}


@pytest.mark.asyncio
async def test_adapter_restores_existing_and_absent_secret_snapshots() -> None:
    store = MemorySyncSecretStore({"existing": "changed", "absent": "temporary"})
    adapter = SyncSecretStoreAdapter(store)

    existing = await adapter.restore_secret(
        SecretSnapshot(
            key="existing",
            value="original",
            revision=None,
            existed=True,
        )
    )
    absent = await adapter.restore_secret(
        SecretSnapshot(
            key="absent",
            value=None,
            revision=None,
            existed=False,
        )
    )

    assert existing.succeeded is True
    assert absent.succeeded is True
    assert store.values == {"existing": "original"}


@pytest.mark.asyncio
async def test_adapter_propagates_sync_store_failure() -> None:
    class FailingStore(MemorySyncSecretStore):
        def set(self, key: str, value: str) -> None:
            raise OSError(f"{key}:{value}")

    adapter = SyncSecretStoreAdapter(FailingStore())

    with pytest.raises(OSError, match="provider:secret"):
        await adapter.set_secret("provider", "secret")


def test_wiring_factory_exposes_secret_store_port_without_ui_dependency() -> None:
    store = MemorySyncSecretStore()

    adapter = create_sync_secret_store_adapter(store)

    assert isinstance(adapter, SyncSecretStoreAdapter)
    assert adapter.store is store
