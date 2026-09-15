from __future__ import annotations

import copy
import logging
from dataclasses import replace
from pathlib import Path

from puripuly_heart.app.ports.application_runtime_logging import ApplicationRuntimeLoggingPort
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
from puripuly_heart.core.local_translation.assets import remove_retired_managed_gemma_installs

logger = logging.getLogger(__name__)


class SettingsVNextCanonicalPersistenceAdapter:
    def __init__(
        self,
        *,
        retired_asset_cleanup_logging: ApplicationRuntimeLoggingPort | None = None,
    ) -> None:
        self._retired_asset_cleanup_logging = retired_asset_cleanup_logging

    def load_active(self, path: Path) -> CanonicalSettingsLoadResult:
        result = load_vnext_settings(path)
        if result.settings is None:
            status = getattr(result.status, "value", result.status)
            message = result.error.message if result.error is not None else status
            raise CanonicalSettingsPersistenceError(str(status), message)
        if result.migrated:
            failure_types: set[str] = set()
            failed_count = 0

            def _record_cleanup_failure(_retired_path: Path, exc: OSError) -> None:
                nonlocal failed_count
                failed_count += 1
                failure_types.add(type(exc).__name__)

            removed = remove_retired_managed_gemma_installs(on_failure=_record_cleanup_failure)
            self._emit_retired_asset_cleanup_receipt(
                removed_count=len(removed),
                failed_count=failed_count,
                failure_types=tuple(sorted(failure_types)),
            )
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

    def _emit_retired_asset_cleanup_receipt(
        self,
        *,
        removed_count: int,
        failed_count: int,
        failure_types: tuple[str, ...],
    ) -> None:
        if removed_count == 0 and failed_count == 0:
            return
        outcome = "completed" if failed_count == 0 else "partial"
        failure_summary = ",".join(failure_types[:3]) if failure_types else "none"
        message = (
            "[Settings][RetiredAssetCleanup] "
            f"outcome={outcome} removed={removed_count} failed={failed_count} "
            f"failure_types={failure_summary}"
        )
        cleanup_logging = self._retired_asset_cleanup_logging
        level = logging.INFO if failed_count == 0 else logging.WARNING
        if cleanup_logging is not None:
            try:
                cleanup_logging.emit_basic(message, level=level)
                return
            except Exception:
                pass
        logger.log(level, "%s", message)
