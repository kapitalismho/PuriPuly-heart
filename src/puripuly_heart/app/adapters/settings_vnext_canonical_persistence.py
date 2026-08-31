from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from puripuly_heart.app.ports.canonical_settings_persistence import (
    CanonicalSettingsLoadResult,
    CanonicalSettingsPersistenceError,
    ProviderVerificationBinding,
)
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings, save_vnext_settings
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    ProviderVerificationEntry,
)


class SettingsVNextCanonicalPersistenceAdapter:
    def load_active(self, path: Path) -> CanonicalSettingsLoadResult:
        result = load_vnext_settings(path)
        if result.settings is None:
            status = getattr(result.status, "value", result.status)
            message = result.error.message if result.error is not None else status
            raise CanonicalSettingsPersistenceError(str(status), message)
        return CanonicalSettingsLoadResult(
            canonical_settings=result.settings,
            migrated=result.migrated,
            backup_path=result.backup_path,
        )

    def persist(self, path: Path, settings: AppSettingsVNext) -> None:
        result = save_vnext_settings(path, settings)
        if not result.ok:
            status = getattr(result.status, "value", result.status)
            message = result.error.message if result.error is not None else status
            raise CanonicalSettingsPersistenceError(str(status), message)

    def bind_provider_verification(
        self,
        canonical: AppSettingsVNext,
        binding: ProviderVerificationBinding,
    ) -> AppSettingsVNext:
        verification = canonical.state.provider_verification
        if not hasattr(verification, binding.provider):
            raise ValueError(f"unsupported provider verification binding: {binding.provider}")
        entry = ProviderVerificationEntry(
            status="verified",
            provider=binding.provider,
            secret_key=binding.secret_key,
            secret_revision=binding.secret_revision,
            secret_fingerprint=binding.secret_fingerprint,
            verifier_context=dict(binding.verifier_context),
            verifier_evidence=dict(binding.verifier_evidence),
        )
        return replace(
            canonical,
            state=replace(
                canonical.state,
                provider_verification=replace(
                    verification,
                    **{binding.provider: entry},
                ),
            ),
        )

    def snapshot(self, canonical: AppSettingsVNext | None) -> AppSettingsVNext | None:
        return copy.deepcopy(canonical)

    def rollback(self, snapshot: AppSettingsVNext | None) -> AppSettingsVNext | None:
        return copy.deepcopy(snapshot)
