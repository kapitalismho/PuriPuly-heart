from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar

from puripuly_heart.app.ports._settings_values import freeze_settings_values
from puripuly_heart.core.messages import ErrorDiagnostics, UserMessageRef


@dataclass(frozen=True, slots=True)
class SettingsSnapshot:
    values: Mapping[str, object]
    revision: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_settings_values(self.values))


@dataclass(frozen=True, slots=True)
class SettingsCommitRequest:
    values: Mapping[str, object]
    expected_revision: str | None
    reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_settings_values(self.values))


@dataclass(frozen=True, slots=True)
class SettingsCommitResult:
    succeeded: bool
    snapshot: SettingsSnapshot | None
    message: UserMessageRef | None
    diagnostics: ErrorDiagnostics | None


class SettingsRepositoryPort(Protocol):
    async def load(self) -> SettingsSnapshot: ...

    async def save(self, request: SettingsCommitRequest) -> SettingsCommitResult: ...


CommittedSettingsT = TypeVar("CommittedSettingsT")


class CommittedSettingsRepositoryPort(
    SettingsRepositoryPort,
    Protocol[CommittedSettingsT],
):
    @property
    def committed_settings(self) -> CommittedSettingsT: ...


__all__ = [
    "CommittedSettingsRepositoryPort",
    "SettingsCommitRequest",
    "SettingsCommitResult",
    "SettingsRepositoryPort",
    "SettingsSnapshot",
]
