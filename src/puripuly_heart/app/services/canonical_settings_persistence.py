from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.canonical_settings_persistence import (
    CanonicalSettingsPersistencePort,
    ProviderVerificationBinding,
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
    "SettingsOwner",
    "SettingsOwnerStartResult",
    "compose_canonical_settings_persistence",
    "compose_settings_owner",
]
