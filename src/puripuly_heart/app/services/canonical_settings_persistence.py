from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import cast

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.canonical_settings_persistence import (
    CanonicalSettingsPersistencePort,
    ProviderVerificationBinding,
)
from puripuly_heart.app.ports.settings_repository import (
    SettingsCommitRequest,
    SettingsCommitResult,
    SettingsSnapshot,
)
from puripuly_heart.app.services.provider_runtime_apply import (
    _settings_mutation_diagnostics,
)
from puripuly_heart.app.services.settings_mutation_legacy import (
    _apply_settings_path_patch,
)
from puripuly_heart.config.profile_bootstrap import import_stable_settings_if_missing
from puripuly_heart.config.settings import AppSettings, new_settings_for_first_run
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    CaptureTargetIntent,
    with_capture_target,
)
from puripuly_heart.core.translation_policy import (
    FIXED_TRANSLATION_POLICY,
    TranslationRuntimePolicy,
)


def _legacy_settings_snapshot_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _legacy_settings_snapshot_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, tuple | list):
        return [_legacy_settings_snapshot_value(item) for item in value]
    return copy.deepcopy(value)


def legacy_settings_snapshot_values(settings: AppSettings) -> dict[str, object]:
    return cast(dict[str, object], _legacy_settings_snapshot_value(asdict(settings)))


def _apply_managed_pending_delivery_ack_patch(
    settings: AppSettings,
    values: Mapping[str, object],
) -> None:
    state = values.get("state")
    if not isinstance(state, Mapping):
        return
    managed = state.get("managed_connection")
    if not isinstance(managed, Mapping):
        return
    field_map = {
        "installation_id": "installation_id",
        "release_token": "release_token",
        "release_token_expires_at": "release_token_expires_at",
        "verified_hardware_hash": "verified_hardware_hash",
        "verified_hardware_hash_salt_version": "verified_hardware_hash_salt_version",
        "active_managed_credential_ref": "active_managed_credential_ref",
        "active_managed_expires_at": "active_managed_expires_at",
        "founder_letter_seen_credential_ref": "founder_letter_seen_credential_ref",
        "referral_id": "referral_id",
        "local_managed_claim_sources": "local_managed_claim_sources",
        "pending_delivery_ack_source": "pending_delivery_ack_source",
        "pending_delivery_ack_delivery_id": "pending_delivery_ack_delivery_id",
        "pending_delivery_ack_managed_credential_ref": (
            "pending_delivery_ack_managed_credential_ref"
        ),
        "pending_delivery_ack_expires_at": "pending_delivery_ack_expires_at",
    }
    for source_key, attr_name in field_map.items():
        if source_key not in managed:
            continue
        value = managed[source_key]
        if attr_name == "local_managed_claim_sources":
            setattr(
                settings.managed_identity,
                attr_name,
                tuple(value) if isinstance(value, list) else (),
            )
        elif attr_name == "verified_hardware_hash_salt_version":
            setattr(
                settings.managed_identity,
                attr_name,
                value if type(value) is int else None,
            )
        else:
            setattr(
                settings.managed_identity,
                attr_name,
                value if isinstance(value, str) else None,
            )


@dataclass(frozen=True, slots=True)
class LegacySettingsPatchCallbacks:
    current_settings: Callable[[], AppSettings | None]
    canonical_projection_snapshot: Callable[[], AppSettings | None]
    begin_canonical_mutation: Callable[[], None]
    update_canonical_from_legacy_delta: Callable[[AppSettings, AppSettings], None]
    bind_provider_verification: Callable[[ProviderVerificationBinding], None]
    persist_settings: Callable[[AppSettings], None]
    rollback_canonical_mutation: Callable[[], None]
    save_failure_sink: Callable[[str], None]
    remember_canonical_projection: Callable[[AppSettings], None]


@dataclass(slots=True)
class LegacySettingsPatchRepository:
    callbacks: LegacySettingsPatchCallbacks
    committed_settings: AppSettings
    base_settings: AppSettings | None = None
    surface: str = "translation_provider"
    provider_verification_binding: ProviderVerificationBinding | None = None

    async def load(self) -> SettingsSnapshot:
        settings = self.callbacks.current_settings() or self.committed_settings
        return SettingsSnapshot(
            values=legacy_settings_snapshot_values(settings),
            revision=None,
        )

    async def save(self, request: SettingsCommitRequest) -> SettingsCommitResult:
        base_settings = self.base_settings
        next_settings = copy.deepcopy(base_settings or self.committed_settings)
        if (
            base_settings is None
            and "state" not in request.values
            and "intent" not in request.values
        ):
            next_settings = copy.deepcopy(self.committed_settings)
        elif all(isinstance(path, str) and "." in path for path in request.values):
            _apply_settings_path_patch(next_settings, request.values)
        elif "state" in request.values or "intent" in request.values:
            _apply_managed_pending_delivery_ack_patch(next_settings, request.values)
        else:
            next_settings = copy.deepcopy(self.committed_settings)
        self.callbacks.begin_canonical_mutation()
        try:
            self.callbacks.update_canonical_from_legacy_delta(
                self.callbacks.canonical_projection_snapshot() or base_settings or next_settings,
                next_settings,
            )
            if self.provider_verification_binding is not None:
                self.callbacks.bind_provider_verification(self.provider_verification_binding)
            await asyncio.to_thread(self.callbacks.persist_settings, next_settings)
        except Exception:
            self.callbacks.rollback_canonical_mutation()
            self.callbacks.save_failure_sink("Failed to save settings mutation")
            return SettingsCommitResult(
                succeeded=False,
                snapshot=None,
                message=None,
                diagnostics=_settings_mutation_diagnostics(
                    component="settings_repository",
                    operation="save",
                    code="settings_save_failed",
                    surface=self.surface,
                ),
            )
        self.committed_settings = next_settings
        self.callbacks.remember_canonical_projection(next_settings)
        return SettingsCommitResult(
            succeeded=True,
            snapshot=SettingsSnapshot(
                values=legacy_settings_snapshot_values(self.committed_settings),
                revision=None,
            ),
            message=None,
            diagnostics=None,
        )


def compose_canonical_settings_persistence() -> (
    CanonicalSettingsPersistencePort[AppSettings, AppSettingsVNext]
):
    return SettingsVNextCanonicalPersistenceAdapter()


@dataclass(frozen=True, slots=True)
class SettingsOwnerStartResult:
    settings: AppSettings
    migrated: bool
    backup_path: Path | None
    stable_source_path: Path | None = None
    stable_source_settings: AppSettingsVNext | None = None
    imported_settings: AppSettingsVNext | None = None


@dataclass(slots=True)
class SettingsOwner:
    path: Path
    persistence: CanonicalSettingsPersistencePort[AppSettings, AppSettingsVNext]
    policy: TranslationRuntimePolicy = FIXED_TRANSLATION_POLICY
    canonical: AppSettingsVNext | None = None
    _rollback_snapshot: AppSettingsVNext | None = None
    _mutation_depth: int = 0

    def start(self, *, allow_stable_settings_import: bool = False) -> SettingsOwnerStartResult:
        stable_source_path: Path | None = None
        stable_source_settings: AppSettingsVNext | None = None
        imported_settings: AppSettingsVNext | None = None
        if not self.path.exists() and allow_stable_settings_import:
            imported = import_stable_settings_if_missing(self.path)
            if imported.error is not None:
                raise RuntimeError(
                    "failed to import stable settings into vNext profile: "
                    f"{imported.error.message}"
                )
            if imported.imported and imported.settings is not None:
                stable_source_path = imported.source_path
                stable_source_settings = imported.source_settings
                imported_settings = imported.settings
        if not self.path.exists():
            settings = new_settings_for_first_run()
            self.canonical = self.persistence.project(
                settings,
                canonical=None,
                authoritative=False,
            )
            self.persistence.persist(self.path, self.canonical)
            return SettingsOwnerStartResult(
                settings=self.persistence.compatibility_projection(self.canonical),
                migrated=False,
                backup_path=None,
            )
        loaded = self.persistence.load_active(self.path)
        self.canonical = loaded.canonical_settings
        return SettingsOwnerStartResult(
            settings=loaded.compatibility_settings,
            migrated=loaded.migrated,
            backup_path=loaded.backup_path,
            stable_source_path=stable_source_path,
            stable_source_settings=stable_source_settings,
            imported_settings=imported_settings,
        )

    def persist(self) -> None:
        if self.canonical is None:
            raise RuntimeError("settings owner has no canonical settings")
        self.persistence.persist(self.path, self.canonical)

    def project(self, settings: AppSettings, *, authoritative: bool) -> AppSettingsVNext:
        self.normalize_compatibility(settings)
        projected = self.persistence.project(
            settings,
            canonical=self.canonical,
            authoritative=authoritative,
        )
        if not authoritative:
            self.canonical = projected
        return projected

    def apply_legacy_delta(
        self,
        base_settings: AppSettings | None,
        next_settings: AppSettings,
    ) -> AppSettingsVNext:
        self.normalize_compatibility(next_settings)
        self.canonical = self.persistence.apply_legacy_delta(
            canonical=self.canonical,
            base_settings=base_settings,
            next_settings=next_settings,
        )
        return self.canonical

    def project_legacy_delta(
        self,
        base_settings: AppSettings | None,
        next_settings: AppSettings,
    ) -> AppSettingsVNext:
        normalized = copy.deepcopy(next_settings)
        self.normalize_compatibility(normalized)
        return self.persistence.apply_legacy_delta(
            canonical=self.canonical,
            base_settings=base_settings,
            next_settings=normalized,
        )

    def normalize_compatibility(self, settings: AppSettings) -> None:
        settings.stt.low_latency_mode = self.policy.fast_translation_enabled
        settings.ui.integrated_context_enabled = (
            self.policy.context_policy == "integrated_preferred"
        )

    def bind_provider_verification(self, binding: ProviderVerificationBinding) -> None:
        if self.canonical is None:
            raise RuntimeError("settings owner has no canonical settings")
        self.canonical = self.persistence.bind_provider_verification(self.canonical, binding)

    def compatibility_projection(self) -> AppSettings:
        if self.canonical is None:
            raise RuntimeError("settings owner has no canonical settings")
        return self.persistence.compatibility_projection(self.canonical)

    @staticmethod
    def legacy_snapshot_values(settings: AppSettings) -> dict[str, object]:
        return legacy_settings_snapshot_values(settings)

    def create_legacy_patch_repository(
        self,
        *,
        current_settings: Callable[[], AppSettings | None],
        canonical_projection_snapshot: Callable[[], AppSettings | None],
        begin_canonical_mutation: Callable[[], None],
        update_canonical_from_legacy_delta: Callable[[AppSettings, AppSettings], None],
        bind_provider_verification: Callable[[ProviderVerificationBinding], None],
        persist_settings: Callable[[AppSettings], None],
        rollback_canonical_mutation: Callable[[], None],
        save_failure_sink: Callable[[str], None],
        remember_canonical_projection: Callable[[AppSettings], None],
        committed_settings: AppSettings,
        base_settings: AppSettings | None = None,
        surface: str = "translation_provider",
        provider_verification_binding: ProviderVerificationBinding | None = None,
    ) -> LegacySettingsPatchRepository:
        return LegacySettingsPatchRepository(
            callbacks=LegacySettingsPatchCallbacks(
                current_settings=current_settings,
                canonical_projection_snapshot=canonical_projection_snapshot,
                begin_canonical_mutation=begin_canonical_mutation,
                update_canonical_from_legacy_delta=update_canonical_from_legacy_delta,
                bind_provider_verification=bind_provider_verification,
                persist_settings=persist_settings,
                rollback_canonical_mutation=rollback_canonical_mutation,
                save_failure_sink=save_failure_sink,
                remember_canonical_projection=remember_canonical_projection,
            ),
            committed_settings=committed_settings,
            base_settings=base_settings,
            surface=surface,
            provider_verification_binding=provider_verification_binding,
        )

    def update_capture_target(
        self,
        compatibility_settings: AppSettings,
        capture_target: CaptureTargetIntent,
    ) -> AppSettings:
        if self.canonical is None:
            self.canonical = self.persistence.project(
                compatibility_settings,
                canonical=None,
                authoritative=False,
            )
        snapshot = self.persistence.snapshot(self.canonical)
        desktop_audio = self.canonical.intent.desktop_audio
        self.canonical = with_capture_target(
            replace(
                self.canonical,
                intent=replace(
                    self.canonical.intent,
                    desktop_audio=replace(
                        desktop_audio,
                        vad_speech_threshold=(
                            compatibility_settings.desktop_audio.vad_speech_threshold
                        ),
                        vad_hangover_ms=compatibility_settings.desktop_audio.vad_hangover_ms,
                        vad_pre_roll_ms=compatibility_settings.desktop_audio.vad_pre_roll_ms,
                    ),
                ),
            ),
            capture_target,
        )
        try:
            projected = self.compatibility_projection()
            self.persist()
        except Exception:
            self.canonical = self.persistence.rollback(snapshot)
            raise
        return projected

    def begin(self) -> None:
        if self._mutation_depth == 0:
            self._rollback_snapshot = self.persistence.snapshot(self.canonical)
        self._mutation_depth += 1

    def rollback(self) -> None:
        if self._mutation_depth == 0:
            return
        self.canonical = self.persistence.rollback(self._rollback_snapshot)
        self._mutation_depth = 1
        self.complete()

    def complete(self) -> None:
        if self._mutation_depth == 0:
            return
        self._mutation_depth -= 1
        if self._mutation_depth == 0:
            self._rollback_snapshot = None


def compose_settings_owner(path: Path) -> SettingsOwner:
    return SettingsOwner(path=path, persistence=compose_canonical_settings_persistence())


__all__ = [
    "LegacySettingsPatchCallbacks",
    "LegacySettingsPatchRepository",
    "SettingsOwner",
    "SettingsOwnerStartResult",
    "compose_canonical_settings_persistence",
    "compose_settings_owner",
    "legacy_settings_snapshot_values",
]
