from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .schema import (
    TranslationExtension,
    TranslationExtensionValidationError,
    parse_translation_extension,
)


@dataclass(frozen=True, slots=True)
class LoadedTranslationExtension:
    definition: TranslationExtension
    source_path: Path
    fingerprint: str


@dataclass(frozen=True, slots=True)
class TranslationExtensionLoadError:
    source_path: Path
    message: str


@dataclass(frozen=True, slots=True)
class TranslationExtensionRegistrySnapshot:
    extensions: tuple[LoadedTranslationExtension, ...] = ()
    errors: tuple[TranslationExtensionLoadError, ...] = ()

    def get(self, extension_id: str) -> LoadedTranslationExtension | None:
        return next(
            (extension for extension in self.extensions if extension.definition.id == extension_id),
            None,
        )


@dataclass(slots=True)
class TranslationExtensionRegistry:
    directory: Path
    _snapshot: TranslationExtensionRegistrySnapshot = field(
        init=False,
        default_factory=TranslationExtensionRegistrySnapshot,
        repr=False,
    )

    @property
    def snapshot(self) -> TranslationExtensionRegistrySnapshot:
        return self._snapshot

    def reload(self) -> TranslationExtensionRegistrySnapshot:
        self.directory.mkdir(parents=True, exist_ok=True)
        loaded: list[LoadedTranslationExtension] = []
        errors: list[TranslationExtensionLoadError] = []
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
                definition = parse_translation_extension(raw, source_path=source_path)
            except (
                OSError,
                TypeError,
                UnicodeError,
                json.JSONDecodeError,
                TranslationExtensionValidationError,
            ) as exc:
                errors.append(
                    TranslationExtensionLoadError(
                        source_path=source_path,
                        message=_safe_error_message(exc),
                    )
                )
                continue
            loaded.append(
                LoadedTranslationExtension(
                    definition=definition,
                    source_path=source_path,
                    fingerprint=definition.fingerprint,
                )
            )

        by_id: dict[str, list[LoadedTranslationExtension]] = {}
        for extension in loaded:
            by_id.setdefault(extension.definition.id, []).append(extension)
        duplicate_ids = {extension_id for extension_id, items in by_id.items() if len(items) > 1}
        if duplicate_ids:
            retained: list[LoadedTranslationExtension] = []
            for extension in loaded:
                if extension.definition.id not in duplicate_ids:
                    retained.append(extension)
                    continue
                errors.append(
                    TranslationExtensionLoadError(
                        source_path=extension.source_path,
                        message=f"duplicate extension id: {extension.definition.id}",
                    )
                )
            loaded = retained

        self._snapshot = TranslationExtensionRegistrySnapshot(
            extensions=tuple(sorted(loaded, key=lambda item: item.definition.id)),
            errors=tuple(sorted(errors, key=lambda item: item.source_path.name.casefold())),
        )
        return self._snapshot

    def get(self, extension_id: str) -> LoadedTranslationExtension | None:
        return self._snapshot.get(extension_id)


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON at line {exc.lineno}, column {exc.colno}"
    message = str(exc)
    return message or type(exc).__name__


__all__ = [
    "LoadedTranslationExtension",
    "TranslationExtensionLoadError",
    "TranslationExtensionRegistry",
    "TranslationExtensionRegistrySnapshot",
]
