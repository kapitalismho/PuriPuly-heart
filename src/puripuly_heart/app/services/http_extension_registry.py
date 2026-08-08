from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.config.paths import default_http_extensions_dir
from puripuly_heart.core.http_extensions import (
    HttpExtensionRegistry,
    HttpExtensionRegistrySnapshot,
)


@dataclass(slots=True)
class HttpExtensionRegistryService:
    registry: HttpExtensionRegistry

    @classmethod
    def from_default_directory(cls) -> HttpExtensionRegistryService:
        registry = HttpExtensionRegistry(default_http_extensions_dir())
        registry.reload()
        return cls(registry)

    @property
    def directory(self) -> Path:
        return self.registry.directory

    @property
    def snapshot(self) -> HttpExtensionRegistrySnapshot:
        return self.registry.snapshot

    def reload(self) -> HttpExtensionRegistrySnapshot:
        return self.registry.reload()

    def get(self, http_extension_id: str) -> object | None:
        return self.registry.get(http_extension_id)


__all__ = ["HttpExtensionRegistryService"]
