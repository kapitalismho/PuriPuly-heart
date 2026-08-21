from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.config.paths import user_config_dir

PREFIX_CACHE_DIRNAME = "gemma-prefix-cache"
PREFIX_CACHE_INDEX_FILENAME = "index.json"
PREFIX_CACHE_MAX_ENTRIES = 4
_IDENTITY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BACKENDS = frozenset({"cpu", "vulkan"})


def default_gemma_prefix_cache_dir() -> Path:
    return user_config_dir() / PREFIX_CACHE_DIRNAME


@dataclass(frozen=True, slots=True)
class PrefixCacheEntry:
    identity: str
    backend: str
    filename: str


class GemmaPrefixCache:
    def __init__(
        self,
        cache_dir: Path,
        *,
        max_entries: int = PREFIX_CACHE_MAX_ENTRIES,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_entries = max(1, int(max_entries))
        self._entries = self._load()

    def filename_for(self, identity: str, backend: str) -> str:
        if _IDENTITY_PATTERN.fullmatch(identity) is None:
            raise ValueError("prefix cache identity must be a sha256 hex digest")
        if backend not in _BACKENDS:
            raise ValueError("prefix cache backend must be cpu or vulkan")
        return f"{identity}.{backend}.bin"

    def has(self, identity: str, backend: str) -> bool:
        try:
            filename = self.filename_for(identity, backend)
        except ValueError:
            return False
        return (self.cache_dir / filename).is_file()

    def touch(self, identity: str, backend: str) -> None:
        self.remember(identity, backend)

    def remember(self, identity: str, backend: str) -> None:
        filename = self.filename_for(identity, backend)
        entry = PrefixCacheEntry(identity=identity, backend=backend, filename=filename)
        self._entries = [
            item
            for item in self._entries
            if not (item.identity == identity and item.backend == backend)
        ]
        self._entries.insert(0, entry)
        while len(self._entries) > self.max_entries:
            old = self._entries.pop()
            (self.cache_dir / old.filename).unlink(missing_ok=True)
        self._save()

    def _index_path(self) -> Path:
        return self.cache_dir / PREFIX_CACHE_INDEX_FILENAME

    def _load(self) -> list[PrefixCacheEntry]:
        path = self._index_path()
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        items = raw.get("entries") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return []
        entries: list[PrefixCacheEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            identity = item.get("identity")
            backend = item.get("backend")
            filename = item.get("filename")
            if not isinstance(identity, str) or not isinstance(backend, str):
                continue
            try:
                expected = self.filename_for(identity, backend)
            except ValueError:
                continue
            if filename != expected:
                continue
            if not (self.cache_dir / expected).is_file():
                continue
            entries.append(PrefixCacheEntry(identity=identity, backend=backend, filename=expected))
        return entries[: self.max_entries]

    def _save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": [
                {
                    "identity": entry.identity,
                    "backend": entry.backend,
                    "filename": entry.filename,
                }
                for entry in self._entries
            ]
        }
        self._index_path().write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "PREFIX_CACHE_MAX_ENTRIES",
    "GemmaPrefixCache",
    "PrefixCacheEntry",
    "default_gemma_prefix_cache_dir",
]
