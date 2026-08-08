from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .schema import (
    HttpExtension,
    HttpExtensionValidationError,
    parse_http_extension,
)


@dataclass(frozen=True, slots=True)
class LoadedHttpExtension:
    definition: HttpExtension
    source_path: Path
    fingerprint: str


@dataclass(frozen=True, slots=True)
class HttpExtensionLoadError:
    source_path: Path
    message: str


@dataclass(frozen=True, slots=True)
class HttpExtensionRegistrySnapshot:
    extensions: tuple[LoadedHttpExtension, ...] = ()
    errors: tuple[HttpExtensionLoadError, ...] = ()

    def get(self, http_extension_id: str) -> LoadedHttpExtension | None:
        return next(
            (
                extension
                for extension in self.extensions
                if extension.definition.id == http_extension_id
            ),
            None,
        )


@dataclass(slots=True)
class HttpExtensionRegistry:
    directory: Path
    _snapshot: HttpExtensionRegistrySnapshot = field(
        init=False,
        default_factory=HttpExtensionRegistrySnapshot,
        repr=False,
    )

    @property
    def snapshot(self) -> HttpExtensionRegistrySnapshot:
        return self._snapshot

    def reload(self) -> HttpExtensionRegistrySnapshot:
        self.directory.mkdir(parents=True, exist_ok=True)
        loaded: list[LoadedHttpExtension] = []
        errors: list[HttpExtensionLoadError] = []
        for source_path in sorted(
            (
                path
                for path in self.directory.iterdir()
                if path.is_file() and path.suffix == ".json"
            ),
            key=lambda path: path.name.casefold(),
        ):
            try:
                raw = json.loads(source_path.read_text(encoding="utf-8"))
                definition = parse_http_extension(raw, source_path=source_path)
            except (
                OSError,
                TypeError,
                UnicodeError,
                json.JSONDecodeError,
                HttpExtensionValidationError,
            ) as exc:
                errors.append(
                    HttpExtensionLoadError(
                        source_path=source_path,
                        message=_safe_error_message(exc),
                    )
                )
                continue
            loaded.append(
                LoadedHttpExtension(
                    definition=definition,
                    source_path=source_path,
                    fingerprint=definition.fingerprint,
                )
            )

        by_id: dict[str, list[LoadedHttpExtension]] = {}
        for extension in loaded:
            by_id.setdefault(extension.definition.id, []).append(extension)
        duplicate_ids = {
            http_extension_id for http_extension_id, items in by_id.items() if len(items) > 1
        }
        if duplicate_ids:
            retained: list[LoadedHttpExtension] = []
            for extension in loaded:
                if extension.definition.id not in duplicate_ids:
                    retained.append(extension)
                    continue
                errors.append(
                    HttpExtensionLoadError(
                        source_path=extension.source_path,
                        message=f"duplicate extension id: {extension.definition.id}",
                    )
                )
            loaded = retained

        self._snapshot = HttpExtensionRegistrySnapshot(
            extensions=tuple(sorted(loaded, key=lambda item: item.definition.id)),
            errors=tuple(sorted(errors, key=lambda item: item.source_path.name.casefold())),
        )
        return self._snapshot

    def get(self, http_extension_id: str) -> LoadedHttpExtension | None:
        return self._snapshot.get(http_extension_id)


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON at line {exc.lineno}, column {exc.colno}"
    message = str(exc)
    return message or type(exc).__name__


__all__ = [
    "LoadedHttpExtension",
    "HttpExtensionLoadError",
    "HttpExtensionRegistry",
    "HttpExtensionRegistrySnapshot",
]
