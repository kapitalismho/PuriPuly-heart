from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

LegacySettingsT = TypeVar("LegacySettingsT", contravariant=True)
CanonicalSettingsT = TypeVar("CanonicalSettingsT")


class CanonicalSettingsPersistenceError(RuntimeError):
    def __init__(self, status: str, message: str | None = None) -> None:
        self.status = status
        super().__init__(message or status)


@dataclass(frozen=True, slots=True)
class CanonicalSettingsLoadResult:
    compatibility_settings: object
    canonical_settings: object
    migrated: bool
    backup_path: Path | None


@dataclass(frozen=True, slots=True)
class ProviderVerificationBinding:
    provider: str
    secret_key: str
    secret_revision: str | None
    secret_fingerprint: str | None
    verifier_context: Mapping[str, object] = field(default_factory=dict)
    verifier_evidence: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class CanonicalSettingsPersistencePort(Protocol[LegacySettingsT, CanonicalSettingsT]):
    def load_active(self, path: Path) -> CanonicalSettingsLoadResult: ...

    def compatibility_projection(
        self,
        settings: CanonicalSettingsT,
    ) -> LegacySettingsT: ...

    def persist(self, path: Path, settings: CanonicalSettingsT) -> None: ...

    def project(
        self,
        settings: LegacySettingsT,
        *,
        canonical: CanonicalSettingsT | None,
        authoritative: bool,
    ) -> CanonicalSettingsT: ...

    def apply_legacy_delta(
        self,
        *,
        canonical: CanonicalSettingsT | None,
        base_settings: LegacySettingsT | None,
        next_settings: LegacySettingsT,
    ) -> CanonicalSettingsT: ...

    def bind_provider_verification(
        self,
        canonical: CanonicalSettingsT,
        binding: ProviderVerificationBinding,
    ) -> CanonicalSettingsT: ...

    def snapshot(self, canonical: CanonicalSettingsT | None) -> CanonicalSettingsT | None: ...

    def rollback(self, snapshot: CanonicalSettingsT | None) -> CanonicalSettingsT | None: ...


__all__ = [
    "CanonicalSettingsPersistenceError",
    "CanonicalSettingsLoadResult",
    "CanonicalSettingsPersistencePort",
    "ProviderVerificationBinding",
]
