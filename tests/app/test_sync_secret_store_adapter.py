from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from puripuly_heart.app.adapters.sync_secret_store import SyncSecretStoreAdapter
from puripuly_heart.app.ports.secret_store import SecretSnapshot
from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore


async def test_adapter_roundtrip_snapshot_and_restore(tmp_path: Path) -> None:
    store = EncryptedFileSecretStore(tmp_path / "secrets.json", passphrase="synthetic")
    adapter = SyncSecretStoreAdapter(store)
    absent = await adapter.snapshot_secret("managed")
    assert absent.existed is False
    assert (await adapter.set_secret("managed", "first")).succeeded is True
    saved = await adapter.snapshot_secret("managed")
    assert saved.existed is True
    assert (await adapter.set_secret("managed", "replacement")).succeeded is True
    assert (await adapter.restore_secret(saved)).succeeded is True
    assert (await adapter.get_secret("managed")).value == "first"
    assert (await adapter.clear_secret("managed")).succeeded is True
    assert (await adapter.get_secret("managed")).value is None
    await adapter.set_secret("managed", "replacement")
    assert (await adapter.restore_secret(absent)).succeeded is True
    assert (await adapter.get_secret("managed")).value is None


@pytest.mark.parametrize("method", ["get_secret", "snapshot_secret"])
async def test_adapter_preserves_failed_observation_instead_of_absence(
    tmp_path: Path, method: str
) -> None:
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="synthetic")
    store.set("managed", "keep")
    adapter = SyncSecretStoreAdapter(store)
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError):
        await getattr(adapter, method)("managed")
    assert path.read_text(encoding="utf-8") == "not-json"


@pytest.mark.parametrize("operation", ["set", "clear", "restore_present", "restore_absent"])
async def test_cancelled_adapter_mutation_settles_before_task_finishes(
    tmp_path: Path, operation: str
) -> None:
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="synthetic")
    store.set("managed", "old")
    store.set("unrelated", "keep")
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockedStore:
        def get(self, key: str) -> str | None:
            return store.get(key)

        def set(self, key: str, value: str) -> None:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10)
            try:
                store.set(key, value)
            finally:
                finished.set()

        def delete(self, key: str) -> None:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10)
            try:
                store.delete(key)
            finally:
                finished.set()

    adapter = SyncSecretStoreAdapter(BlockedStore())
    if operation == "set":
        work = adapter.set_secret("managed", "new")
    elif operation == "clear":
        work = adapter.clear_secret("managed")
    else:
        present = operation == "restore_present"
        work = adapter.restore_secret(
            SecretSnapshot("managed", "restored" if present else None, None, present)
        )
    task = asyncio.create_task(work)
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not finished.is_set()
        assert EncryptedFileSecretStore(path, passphrase="synthetic").get("managed") == "old"
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert finished.is_set()
    expected = {"set": "new", "clear": None, "restore_present": "restored", "restore_absent": None}
    fresh = EncryptedFileSecretStore(path, passphrase="synthetic")
    assert fresh.get("managed") == expected[operation]
    assert fresh.get("unrelated") == "keep"
