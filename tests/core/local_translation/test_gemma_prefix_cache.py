from __future__ import annotations

from pathlib import Path

from puripuly_heart.core.local_translation.prefix_cache import (
    PREFIX_CACHE_MAX_ENTRIES,
    GemmaPrefixCache,
)


def _identity(index: int) -> str:
    return f"{index:064x}"


def test_remember_keeps_four_most_recent_and_deletes_oldest_file(tmp_path: Path) -> None:
    cache = GemmaPrefixCache(tmp_path, max_entries=PREFIX_CACHE_MAX_ENTRIES)
    files = []
    for index in range(5):
        identity = _identity(index)
        filename = cache.filename_for(identity, "cpu")
        path = cache.cache_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"kv")
        files.append(path)
        cache.remember(identity, "cpu")

    assert files[0].is_file() is False
    assert all(path.is_file() for path in files[1:])
    assert cache.has(_identity(0), "cpu") is False
    assert cache.has(_identity(4), "cpu") is True


def test_touch_promotes_existing_entry_ahead_of_eviction(tmp_path: Path) -> None:
    cache = GemmaPrefixCache(tmp_path, max_entries=2)
    first = _identity(1)
    second = _identity(2)
    third = _identity(3)
    for identity in (first, second):
        path = cache.cache_dir / cache.filename_for(identity, "cpu")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"kv")
        cache.remember(identity, "cpu")

    cache.touch(first, "cpu")
    path = cache.cache_dir / cache.filename_for(third, "cpu")
    path.write_bytes(b"kv")
    cache.remember(third, "cpu")

    assert cache.has(first, "cpu") is True
    assert cache.has(second, "cpu") is False
    assert cache.has(third, "cpu") is True


def test_cpu_and_vulkan_entries_are_distinct(tmp_path: Path) -> None:
    cache = GemmaPrefixCache(tmp_path)
    identity = _identity(1)
    for backend in ("cpu", "vulkan"):
        path = cache.cache_dir / cache.filename_for(identity, backend)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"kv")
        cache.remember(identity, backend)

    assert cache.has(identity, "cpu") is True
    assert cache.has(identity, "vulkan") is True
    assert cache.filename_for(identity, "cpu") != cache.filename_for(identity, "vulkan")
