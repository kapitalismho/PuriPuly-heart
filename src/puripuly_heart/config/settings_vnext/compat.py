from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from puripuly_heart.config.settings_vnext import migration, serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext

logger = logging.getLogger(__name__)


class SettingsPersistenceStatus(str, Enum):
    SUCCESS = "success"
    PARSE_FAILED = "parse_failed"
    MIGRATION_FAILED = "migration_failed"
    BACKUP_FAILED = "backup_failed"
    SAVE_FAILED = "save_failed"


@dataclass(frozen=True, slots=True)
class SettingsPersistenceError:
    stage: SettingsPersistenceStatus
    message: str


@dataclass(frozen=True, slots=True)
class VNextSettingsLoadResult:
    status: SettingsPersistenceStatus
    settings: AppSettingsVNext | None = None
    migrated: bool = False
    backup_path: Path | None = None
    error: SettingsPersistenceError | None = None

    @property
    def ok(self) -> bool:
        return self.status == SettingsPersistenceStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class VNextSettingsSaveResult:
    status: SettingsPersistenceStatus
    error: SettingsPersistenceError | None = None

    @property
    def ok(self) -> bool:
        return self.status == SettingsPersistenceStatus.SUCCESS


class BackupCreationError(OSError):
    pass


def load_vnext_settings(
    path: Path,
    *,
    now: datetime | None = None,
    max_backup_attempts: int = 100,
) -> VNextSettingsLoadResult:
    try:
        original_bytes = path.read_bytes()
        raw = json.loads(original_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _log_migration_failure("unknown", SettingsPersistenceStatus.PARSE_FAILED)
        return VNextSettingsLoadResult(
            status=SettingsPersistenceStatus.PARSE_FAILED,
            error=_error(SettingsPersistenceStatus.PARSE_FAILED, exc),
        )
    if not isinstance(raw, dict):
        _log_migration_failure("unknown", SettingsPersistenceStatus.PARSE_FAILED)
        return VNextSettingsLoadResult(
            status=SettingsPersistenceStatus.PARSE_FAILED,
            error=SettingsPersistenceError(
                SettingsPersistenceStatus.PARSE_FAILED,
                "settings file must contain a JSON object",
            ),
        )

    source_shape = "canonical" if migration.is_vnext_settings_dict(raw) else "legacy"
    try:
        settings = migration.from_dict(raw)
    except Exception as exc:
        _log_migration_failure(source_shape, SettingsPersistenceStatus.MIGRATION_FAILED)
        return VNextSettingsLoadResult(
            status=SettingsPersistenceStatus.MIGRATION_FAILED,
            error=_error(SettingsPersistenceStatus.MIGRATION_FAILED, exc),
        )

    if not _requires_canonical_save(raw, settings):
        return VNextSettingsLoadResult(
            status=SettingsPersistenceStatus.SUCCESS,
            settings=settings,
            migrated=False,
        )

    previous_schema_version = previous_schema_version_label(raw)
    try:
        backup_path = create_pre_migration_backup(
            path,
            original_bytes,
            previous_schema_version=previous_schema_version,
            now=now,
            max_attempts=max_backup_attempts,
        )
    except Exception as exc:
        _log_migration_failure(source_shape, SettingsPersistenceStatus.BACKUP_FAILED)
        return VNextSettingsLoadResult(
            status=SettingsPersistenceStatus.BACKUP_FAILED,
            error=_error(SettingsPersistenceStatus.BACKUP_FAILED, exc),
        )

    save_result = save_vnext_settings(path, settings)
    if not save_result.ok:
        _log_migration_failure(source_shape, SettingsPersistenceStatus.SAVE_FAILED)
        return VNextSettingsLoadResult(
            status=SettingsPersistenceStatus.SAVE_FAILED,
            backup_path=backup_path,
            error=save_result.error,
        )

    try:
        _validate_persisted_settings(path, settings)
    except Exception as exc:
        try:
            _atomic_write_bytes(path, original_bytes)
        except Exception as restore_exc:
            exc = RuntimeError(
                f"{type(exc).__name__}: persisted validation failed; "
                f"{type(restore_exc).__name__}: source restoration failed"
            )
        _log_migration_failure(source_shape, SettingsPersistenceStatus.SAVE_FAILED)
        return VNextSettingsLoadResult(
            status=SettingsPersistenceStatus.SAVE_FAILED,
            backup_path=backup_path,
            error=_error(SettingsPersistenceStatus.SAVE_FAILED, exc),
        )

    logger.info(
        "settings_migration source_shape=%s destination_shape=canonical status=success",
        source_shape,
    )
    return VNextSettingsLoadResult(
        status=SettingsPersistenceStatus.SUCCESS,
        settings=settings,
        migrated=True,
        backup_path=backup_path,
    )


def save_vnext_settings(path: Path, settings: AppSettingsVNext) -> VNextSettingsSaveResult:
    try:
        _save_vnext_settings_or_raise(path, settings)
    except Exception as exc:
        return VNextSettingsSaveResult(
            status=SettingsPersistenceStatus.SAVE_FAILED,
            error=_error(SettingsPersistenceStatus.SAVE_FAILED, exc),
        )
    return VNextSettingsSaveResult(status=SettingsPersistenceStatus.SUCCESS)


def _save_vnext_settings_or_raise(path: Path, settings: AppSettingsVNext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = serialization.to_json_text(settings)
    _validate_canonical_text(content, settings)
    _atomic_write_text(path, content, encoding="utf-8")


def _requires_canonical_save(raw: dict[str, Any], settings: AppSettingsVNext) -> bool:
    canonical = json.loads(serialization.to_json_text(settings))
    normalized_raw = serialization.normalize_persisted_dict(raw)
    return _without_settings_version(normalized_raw) != _without_settings_version(canonical)


def _without_settings_version(data: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(data)
    comparable.pop("settings_version", None)
    return comparable


def create_pre_migration_backup(
    path: Path,
    original_bytes: bytes,
    *,
    previous_schema_version: str,
    now: datetime | None = None,
    max_attempts: int = 100,
) -> Path:
    if max_attempts < 1:
        raise BackupCreationError("max backup attempts must be at least 1")
    timestamp = backup_timestamp(now)
    for collision_index in range(max_attempts):
        candidate = backup_candidate_path(
            path,
            previous_schema_version=previous_schema_version,
            timestamp=timestamp,
            collision_index=collision_index,
        )
        try:
            with candidate.open("xb") as handle:
                try:
                    handle.write(original_bytes)
                except Exception:
                    try:
                        candidate.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise
            return candidate
        except FileExistsError:
            continue
    raise BackupCreationError(
        f"could not create exclusive pre-migration backup after {max_attempts} attempts"
    )


def backup_candidate_path(
    path: Path,
    *,
    previous_schema_version: str,
    timestamp: str,
    collision_index: int,
) -> Path:
    suffix = "" if collision_index == 0 else f".{collision_index}"
    return path.with_name(f"{path.name}.pre-v{previous_schema_version}.{timestamp}{suffix}.bak")


def backup_timestamp(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def previous_schema_version_label(raw: dict[str, Any]) -> str:
    value = raw.get("settings_version", "unknown")
    if isinstance(value, bool):
        return "unknown"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip().isdigit():
        return str(int(value.strip()))
    return "unknown"


def _atomic_write_text(path: Path, content: str, *, encoding: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(content, encoding=encoding)
        _validate_canonical_text(tmp_path.read_text(encoding=encoding), None)
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    tmp_path = path.with_suffix(path.suffix + ".restore.tmp")
    try:
        tmp_path.write_bytes(content)
        if tmp_path.read_bytes() != content:
            raise OSError("restored settings bytes failed validation")
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _validate_canonical_text(content: str, expected: AppSettingsVNext | None) -> None:
    raw = json.loads(content)
    if not isinstance(raw, dict) or not migration.is_vnext_settings_dict(raw):
        raise ValueError("persisted canonical settings must contain intent and state")
    restored = migration.from_dict(raw)
    if expected is not None and serialization.to_dict(restored) != serialization.to_dict(expected):
        raise ValueError("persisted canonical settings failed semantic validation")


def _validate_persisted_settings(path: Path, expected: AppSettingsVNext) -> None:
    _validate_canonical_text(path.read_text(encoding="utf-8"), expected)


def _log_migration_failure(
    source_shape: str,
    status: SettingsPersistenceStatus,
) -> None:
    logger.warning(
        "settings_migration source_shape=%s destination_shape=canonical status=failure "
        "failure_category=%s",
        source_shape,
        status.value,
    )


def safe_persistence_error(
    status: SettingsPersistenceStatus,
    exc: Exception,
) -> SettingsPersistenceError:
    return SettingsPersistenceError(
        status,
        f"{status.value}:{type(exc).__name__}",
    )


def _error(status: SettingsPersistenceStatus, exc: Exception) -> SettingsPersistenceError:
    return safe_persistence_error(status, exc)


__all__ = [
    "SettingsPersistenceError",
    "SettingsPersistenceStatus",
    "VNextSettingsLoadResult",
    "VNextSettingsSaveResult",
    "backup_candidate_path",
    "backup_timestamp",
    "create_pre_migration_backup",
    "load_vnext_settings",
    "previous_schema_version_label",
    "save_vnext_settings",
]
