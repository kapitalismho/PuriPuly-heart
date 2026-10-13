from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.messages import ErrorDiagnostics


class CanonicalSettingsPersistenceError(RuntimeError):
    def __init__(
        self,
        status: str,
        message: str | None = None,
        *,
        diagnostics: ErrorDiagnostics | None = None,
        secondary_diagnostics: tuple[ErrorDiagnostics, ...] = (),
        secondary_omitted_count: int = 0,
    ) -> None:
        self.status = status
        self.diagnostics = diagnostics
        self.secondary_diagnostics = secondary_diagnostics[:4]
        self.secondary_omitted_count = secondary_omitted_count + max(
            0, len(secondary_diagnostics) - 4
        )
        super().__init__(message or status)


@dataclass(frozen=True, slots=True)
class CanonicalSettingsLoadResult:
    canonical_settings: AppSettingsVNext
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
class CanonicalSettingsPersistencePort(Protocol):
    def load_active(self, path: Path) -> CanonicalSettingsLoadResult: ...

    def persist(self, path: Path, settings: AppSettingsVNext) -> None: ...

    def bind_provider_verification(
        self,
        canonical: AppSettingsVNext,
        binding: ProviderVerificationBinding,
    ) -> AppSettingsVNext: ...

    def snapshot(self, canonical: AppSettingsVNext | None) -> AppSettingsVNext | None: ...

    def rollback(self, snapshot: AppSettingsVNext | None) -> AppSettingsVNext | None: ...


__all__ = [
    "CanonicalSettingsPersistenceError",
    "CanonicalSettingsLoadResult",
    "CanonicalSettingsPersistencePort",
    "ProviderVerificationBinding",
]
