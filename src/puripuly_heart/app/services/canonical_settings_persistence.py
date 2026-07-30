from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
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


def _managed_identity_delta(baseline: object, current: object) -> dict[str, object]:
    baseline_values = asdict(baseline)
    current_values = asdict(current)
    return {
        field_name: copy.deepcopy(value)
        for field_name, value in current_values.items()
        if baseline_values.get(field_name) != value
    }


def _apply_managed_identity_delta(
    settings: AppSettings,
    values: Mapping[str, object],
) -> None:
    for field_name, value in values.items():
        setattr(settings.managed_identity, field_name, copy.deepcopy(value))


def _restore_managed_identity(settings: AppSettings, snapshot: object) -> None:
    for field_name, value in asdict(snapshot).items():
        setattr(settings.managed_identity, field_name, copy.deepcopy(value))


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


@dataclass(slots=True)
class LegacySettingsPatchRepository:
    owner: SettingsOwner
    committed_settings: AppSettings
    base_settings: AppSettings | None = None
    surface: str = "translation_provider"
    provider_verification_binding: ProviderVerificationBinding | None = None
    save_failure_sink: Callable[[str], None] | None = None
    commit_succeeded: bool = False

    async def load(self) -> SettingsSnapshot:
        settings = self.owner.current or self.committed_settings
        return SettingsSnapshot(
            values=legacy_settings_snapshot_values(settings),
            revision=None,
        )

    async def save(self, request: SettingsCommitRequest) -> SettingsCommitResult:
        self.commit_succeeded = False
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
        self.owner.begin()
        try:
            self.owner.apply_legacy_delta(
                self.owner.projection_snapshot or base_settings or next_settings,
                next_settings,
            )
            if self.provider_verification_binding is not None:
                self.owner.bind_provider_verification(self.provider_verification_binding)
            await asyncio.to_thread(self.owner.persist)
        except asyncio.CancelledError:
            self.owner.rollback()
            raise
        except Exception:
            self.owner.rollback()
            if self.save_failure_sink is not None:
                self.save_failure_sink("Failed to save settings mutation")
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
        self.owner.remember_projection(next_settings)
        self.commit_succeeded = True
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
    current: AppSettings | None = None
    authoritative: bool = False
    projection_snapshot: AppSettings | None = None
    _rollback_snapshot: AppSettingsVNext | None = None
    _rollback_legacy_snapshot: AppSettings | None = None
    _rollback_active_settings: AppSettings | None = None
    _rollback_authoritative: bool = False
    _rollback_pending: bool = False
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
            self.current = self.persistence.compatibility_projection(self.canonical)
            return SettingsOwnerStartResult(
                settings=self.current,
                migrated=False,
                backup_path=None,
            )
        loaded = self.persistence.load_active(self.path)
        self.canonical = loaded.canonical_settings
        self.current = loaded.compatibility_settings
        return SettingsOwnerStartResult(
            settings=self.current,
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

    def save_current(
        self,
        *,
        failure_sink: Callable[[BaseException], None] | None = None,
    ) -> bool:
        try:
            self.persist_current()
        except Exception as exc:
            if failure_sink is not None:
                failure_sink(exc)
            return False
        return True

    def persist_current(self) -> None:
        if self.current is None:
            raise RuntimeError("settings owner has no compatibility settings")
        owns_mutation = self.mutation_depth == 0
        baseline = self.projection_snapshot or self.current
        if owns_mutation:
            self.begin(legacy_snapshot=baseline)
        self.apply_legacy_delta(baseline, self.current)
        try:
            self.persist()
        except Exception:
            self.rollback()
            raise
        self.remember_projection(self.current)
        if owns_mutation:
            self.complete()

    def managed_identity_persistence_callback(
        self,
        bound_settings: AppSettings,
    ) -> Callable[[AppSettings], None]:
        bound_snapshot = copy.deepcopy(bound_settings.managed_identity)

        def persist(settings: AppSettings) -> None:
            nonlocal bound_snapshot
            self.persist_managed_identity(
                settings,
                bound_managed_snapshot=bound_snapshot,
            )
            bound_snapshot = copy.deepcopy(settings.managed_identity)

        return persist

    def persist_managed_identity(
        self,
        settings: AppSettings,
        *,
        bound_managed_snapshot: object | None = None,
    ) -> None:
        active_settings = self.current or settings
        baseline = self.projection_snapshot or active_settings
        managed_baseline = (
            bound_managed_snapshot
            if bound_managed_snapshot is not None
            else baseline.managed_identity
        )
        managed_delta = _managed_identity_delta(
            managed_baseline,
            settings.managed_identity,
        )
        next_settings = copy.deepcopy(active_settings)
        _apply_managed_identity_delta(next_settings, managed_delta)
        self.begin(legacy_snapshot=baseline)
        self.apply_legacy_delta(baseline, next_settings)
        try:
            self.persist()
        except Exception:
            self.rollback()
            _restore_managed_identity(settings, managed_baseline)
            raise
        self.current = next_settings
        self.remember_projection(next_settings)
        self.complete()

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
        self.authoritative = True
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

    def persist_provider_verification(
        self,
        *,
        provider: str,
        key: str,
        success: bool,
        binding: ProviderVerificationBinding | None,
        active_secret: str | None,
    ) -> None:
        if self.current is None:
            raise RuntimeError("settings owner has no compatibility settings")
        baseline = copy.deepcopy(self.projection_snapshot or self.current)
        active_settings = self.current
        self.begin(legacy_snapshot=baseline)
        setattr(active_settings.api_key_verified, provider, success)
        try:
            self.apply_legacy_delta(baseline, active_settings)
            if success:
                if binding is None:
                    raise RuntimeError("verified provider requires evidence binding")
                if active_secret != key:
                    raise RuntimeError(
                        "verified credential does not match the active SecretStore value"
                    )
                self.bind_provider_verification(binding)
            self.persist()
        except Exception:
            self.rollback()
            raise
        self.remember_projection(active_settings)
        self.complete()

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
        committed_settings: AppSettings,
        base_settings: AppSettings | None = None,
        surface: str = "translation_provider",
        provider_verification_binding: ProviderVerificationBinding | None = None,
        save_failure_sink: Callable[[str], None] | None = None,
    ) -> LegacySettingsPatchRepository:
        return LegacySettingsPatchRepository(
            owner=self,
            committed_settings=committed_settings,
            base_settings=base_settings,
            surface=surface,
            provider_verification_binding=provider_verification_binding,
            save_failure_sink=save_failure_sink,
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

    def remember_projection(self, settings: AppSettings) -> None:
        self.projection_snapshot = copy.deepcopy(settings)

    def begin(self, *, legacy_snapshot: AppSettings | None = None) -> None:
        if self._mutation_depth == 0:
            self._rollback_snapshot = self.persistence.snapshot(self.canonical)
            self._rollback_active_settings = self.current
            self._rollback_legacy_snapshot = copy.deepcopy(
                legacy_snapshot if legacy_snapshot is not None else self.current
            )
            self._rollback_authoritative = self.authoritative
            self._rollback_pending = True
        self._mutation_depth += 1

    def rollback(self) -> None:
        if not self._rollback_pending:
            return
        self.canonical = self.persistence.rollback(self._rollback_snapshot)
        active_settings = self._rollback_active_settings
        legacy_snapshot = self._rollback_legacy_snapshot
        if active_settings is not None and legacy_snapshot is not None:
            for settings_field in fields(AppSettings):
                setattr(
                    active_settings,
                    settings_field.name,
                    copy.deepcopy(getattr(legacy_snapshot, settings_field.name)),
                )
            self.current = active_settings
        else:
            self.current = legacy_snapshot
        self.authoritative = self._rollback_authoritative
        self._mutation_depth = 1
        self.complete()

    def complete(self) -> None:
        if self._mutation_depth == 0:
            return
        self._mutation_depth -= 1
        if self._mutation_depth == 0:
            self._rollback_snapshot = None
            self._rollback_legacy_snapshot = None
            self._rollback_active_settings = None
            self._rollback_authoritative = False
            self._rollback_pending = False

    @property
    def mutation_depth(self) -> int:
        return self._mutation_depth

    @property
    def rollback_pending(self) -> bool:
        return self._rollback_pending


def compose_settings_owner(path: Path) -> SettingsOwner:
    return SettingsOwner(path=path, persistence=compose_canonical_settings_persistence())


__all__ = [
    "LegacySettingsPatchRepository",
    "SettingsOwner",
    "SettingsOwnerStartResult",
    "compose_canonical_settings_persistence",
    "compose_settings_owner",
    "legacy_settings_snapshot_values",
]
