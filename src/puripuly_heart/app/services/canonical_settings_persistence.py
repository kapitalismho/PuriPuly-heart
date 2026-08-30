from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

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
    apply_settings_path_patch,
)
from puripuly_heart.config.settings_vnext.defaults import new_settings_for_first_run
from puripuly_heart.config.settings_vnext.migration import (
    apply_canonical_delta,
    merge_canonical_payload,
)
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    CaptureTargetIntent,
    ProviderVerificationEntry,
    with_capture_target,
    with_telemetry_enabled,
    with_translation_runtime_policy,
)
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.core.translation_policy import (
    FIXED_TRANSLATION_POLICY,
    TranslationRuntimePolicy,
)


def canonical_snapshot_values(settings: AppSettingsVNext) -> dict[str, object]:
    return serialization.to_dict(settings)


def _managed_connection_delta(baseline: object, current: object) -> dict[str, object]:
    from dataclasses import asdict

    baseline_values = asdict(baseline)
    current_values = asdict(current)
    return {
        field_name: copy.deepcopy(value)
        for field_name, value in current_values.items()
        if baseline_values.get(field_name) != value
    }


@dataclass(slots=True)
class CanonicalSettingsPatchRepository:
    owner: SettingsOwner
    committed_settings: AppSettingsVNext
    base_settings: AppSettingsVNext | None = None
    surface: str = "translation_provider"
    provider_verification_binding: ProviderVerificationBinding | None = None
    save_failure_sink: Callable[[str], None] | None = None
    commit_succeeded: bool = False

    async def load(self) -> SettingsSnapshot:
        settings = self.owner.canonical or self.committed_settings
        return SettingsSnapshot(
            values=canonical_snapshot_values(settings),
            revision=None,
        )

    async def save(self, request: SettingsCommitRequest) -> SettingsCommitResult:
        self.commit_succeeded = False
        base_settings = self.base_settings
        source = base_settings or self.committed_settings
        if (
            base_settings is None
            and "state" not in request.values
            and "intent" not in request.values
        ):
            next_settings = copy.deepcopy(self.committed_settings)
        elif request.values and all(
            isinstance(path, str) and "." in path for path in request.values
        ):
            next_settings = apply_settings_path_patch(source, request.values)
        elif "state" in request.values or "intent" in request.values:
            next_settings = merge_canonical_payload(source, request.values)
        else:
            next_settings = copy.deepcopy(self.committed_settings)
        if not isinstance(next_settings, AppSettingsVNext):
            raise TypeError("canonical patch repository requires AppSettingsVNext")
        self.owner.begin()
        try:
            self.owner.apply_canonical_delta(
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
                values=canonical_snapshot_values(self.committed_settings),
                revision=None,
            ),
            message=None,
            diagnostics=None,
        )


def compose_canonical_settings_persistence() -> CanonicalSettingsPersistencePort:
    return SettingsVNextCanonicalPersistenceAdapter()


@dataclass(frozen=True, slots=True)
class SettingsOwnerStartResult:
    settings: AppSettingsVNext
    migrated: bool
    backup_path: Path | None


@dataclass(slots=True)
class SettingsOwner:
    path: Path
    persistence: CanonicalSettingsPersistencePort
    policy: TranslationRuntimePolicy = FIXED_TRANSLATION_POLICY
    canonical: AppSettingsVNext | None = None
    authoritative: bool = False
    projection_snapshot: AppSettingsVNext | None = None
    _overlay_enabled: bool = False
    _peer_translation_enabled: bool = False
    _overlay_desktop_locked: bool = False
    _rollback_snapshot: AppSettingsVNext | None = None
    _rollback_overlay_enabled: bool = False
    _rollback_peer_translation_enabled: bool = False
    _rollback_overlay_desktop_locked: bool = False
    _rollback_authoritative: bool = False
    _rollback_pending: bool = False
    _mutation_depth: int = 0

    def require_canonical(self) -> AppSettingsVNext:
        if self.canonical is None:
            raise RuntimeError("settings owner has no canonical settings")
        return self.canonical

    def projected_canonical(self) -> AppSettingsVNext | None:
        return self.canonical

    def normalize(self, settings: AppSettingsVNext) -> AppSettingsVNext:
        return with_translation_runtime_policy(settings, self.policy)

    def overlay_enabled(self) -> bool:
        return self._overlay_enabled

    def set_overlay_enabled(self, enabled: bool) -> None:
        self._overlay_enabled = bool(enabled)

    def overlay_desktop_locked(self) -> bool:
        return self._overlay_desktop_locked

    def set_overlay_desktop_locked(self, locked: bool) -> None:
        self._overlay_desktop_locked = bool(locked)

    def peer_translation_enabled(self) -> bool:
        return self._peer_translation_enabled

    def set_peer_translation_enabled(self, enabled: bool) -> None:
        self._peer_translation_enabled = bool(enabled)

    def start(self) -> SettingsOwnerStartResult:
        if not self.path.exists():
            self.canonical = new_settings_for_first_run()
            self.persistence.persist(self.path, self.canonical)
            loaded = self.persistence.load_active(self.path)
            self.canonical = loaded.canonical_settings
            return SettingsOwnerStartResult(
                settings=self.canonical,
                migrated=False,
                backup_path=None,
            )
        loaded = self.persistence.load_active(self.path)
        self.canonical = loaded.canonical_settings
        return SettingsOwnerStartResult(
            settings=self.canonical,
            migrated=loaded.migrated,
            backup_path=loaded.backup_path,
        )

    def persist(self) -> None:
        self.persistence.persist(self.path, self.require_canonical())

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
        live = self.require_canonical()
        owns_mutation = self.mutation_depth == 0
        baseline = self.projection_snapshot or live
        if owns_mutation:
            self.begin(legacy_snapshot=baseline)
        self.apply_canonical_delta(baseline, live)
        try:
            self.persist()
        except Exception:
            self.rollback()
            raise
        self.remember_projection(live)
        if owns_mutation:
            self.complete()

    def with_telemetry_enabled(
        self,
        settings: AppSettingsVNext,
        enabled: bool,
    ) -> AppSettingsVNext:
        return with_telemetry_enabled(settings, enabled)

    def build_managed_openrouter_byok_target(
        self,
        current_settings: AppSettingsVNext | None = None,
    ) -> AppSettingsVNext | None:
        from puripuly_heart.config.translation_values import (
            provider_llm_for_translation,
        )
        from puripuly_heart.config.llm_profiles import (
            get_openrouter_llm_profile,
            get_openrouter_selection_alias_for_model_and_source,
        )

        settings = current_settings if current_settings is not None else self.canonical
        if settings is None:
            return None
        translation = settings.intent.translation
        if provider_llm_for_translation(translation.model, translation.connection) != "openrouter":
            return None
        if translation.openrouter_selected_source != "managed":
            return None
        openrouter_model = translation.openrouter_model
        alias = translation.openrouter_selection_alias
        if alias is not None:
            profile = get_openrouter_llm_profile(alias)
            if profile is not None:
                openrouter_model = profile.openrouter_model
        alias_value = get_openrouter_selection_alias_for_model_and_source(
            openrouter_model,
            "byok",
        )
        if alias_value is None:
            return None
        history = dict(translation.connection_history)
        history[translation.model] = "openrouter"
        return replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(
                    translation,
                    connection="openrouter",
                    connection_history=history,
                    openrouter_selection_alias=alias_value,
                    openrouter_selected_source="byok",
                    openrouter_model=openrouter_model,
                    openrouter_provider_routing="default",
                ),
            ),
        )

    def persist_managed_identity(
        self,
        settings: AppSettingsVNext,
        *,
        bound_managed_snapshot: object | None = None,
    ) -> None:
        active_settings = self.canonical or settings
        baseline = self.projection_snapshot or active_settings
        managed_baseline = (
            bound_managed_snapshot
            if bound_managed_snapshot is not None
            else baseline.state.managed_connection
        )
        managed_delta = _managed_connection_delta(
            managed_baseline,
            settings.state.managed_connection,
        )
        next_state = replace(
            active_settings.state,
            managed_connection=replace(
                active_settings.state.managed_connection,
                **managed_delta,
            ),
        )
        next_settings = replace(active_settings, state=next_state)
        self.begin()
        self.apply_canonical_delta(baseline, next_settings)
        try:
            self.persist()
        except Exception:
            self.rollback()
            raise
        self.canonical = next_settings
        self.remember_projection(next_settings)
        self.complete()

    def managed_identity_persistence_callback(
        self,
        bound_settings: AppSettingsVNext,
    ) -> Callable[[AppSettingsVNext], None]:
        bound_snapshot = copy.deepcopy(bound_settings.state.managed_connection)

        def persist(settings: AppSettingsVNext) -> None:
            nonlocal bound_snapshot
            self.persist_managed_identity(
                settings,
                bound_managed_snapshot=bound_snapshot,
            )
            bound_snapshot = copy.deepcopy(settings.state.managed_connection)

        return persist

    def apply_canonical_delta(
        self,
        base_settings: AppSettingsVNext | None,
        next_settings: AppSettingsVNext,
    ) -> AppSettingsVNext:
        normalized = self.normalize(next_settings)
        live = self.canonical
        if live is None or base_settings is None:
            self.canonical = normalized
        else:
            self.canonical = apply_canonical_delta(live, self.normalize(base_settings), normalized)
        self.authoritative = True
        return self.require_canonical()

    def bind_provider_verification(self, binding: ProviderVerificationBinding) -> None:
        self.canonical = self.persistence.bind_provider_verification(
            self.require_canonical(),
            binding,
        )

    def persist_provider_verification(
        self,
        *,
        provider: str,
        key: str,
        success: bool,
        binding: ProviderVerificationBinding | None,
        active_secret: str | None,
    ) -> None:
        live = self.require_canonical()
        baseline = copy.deepcopy(self.projection_snapshot or live)
        self.begin()
        verification = live.state.provider_verification
        if success:
            next_entry = ProviderVerificationEntry(status="unknown")
        else:
            next_entry = ProviderVerificationEntry(status="unknown")
        next_settings = replace(
            live,
            state=replace(
                live.state,
                provider_verification=replace(verification, **{provider: next_entry}),
            ),
        )
        try:
            self.apply_canonical_delta(baseline, next_settings)
            if success:
                if binding is None:
                    raise RuntimeError("verified provider requires evidence binding")
                if active_secret != key:
                    raise RuntimeError(
                        "verified credential does not match the active SecretStore value"
                    )
                self.bind_provider_verification(binding)
            else:
                self.canonical = next_settings
            self.persist()
        except Exception:
            self.rollback()
            raise
        self.remember_projection(self.require_canonical())
        self.complete()

    @staticmethod
    def snapshot_values(settings: AppSettingsVNext) -> dict[str, object]:
        return canonical_snapshot_values(settings)

    def create_canonical_patch_repository(
        self,
        *,
        committed_settings: AppSettingsVNext,
        base_settings: AppSettingsVNext | None = None,
        surface: str = "translation_provider",
        provider_verification_binding: ProviderVerificationBinding | None = None,
        save_failure_sink: Callable[[str], None] | None = None,
    ) -> CanonicalSettingsPatchRepository:
        return CanonicalSettingsPatchRepository(
            owner=self,
            committed_settings=committed_settings,
            base_settings=base_settings,
            surface=surface,
            provider_verification_binding=provider_verification_binding,
            save_failure_sink=save_failure_sink,
        )

    def apply_capture_target(self, capture_target: CaptureTargetIntent) -> AppSettingsVNext:
        if self.canonical is None:
            self.canonical = new_settings_for_first_run()
        snapshot = self.persistence.snapshot(self.canonical)
        self.canonical = with_capture_target(self.canonical, capture_target)
        try:
            self.persist()
        except Exception:
            self.canonical = self.persistence.rollback(snapshot)
            raise
        return self.require_canonical()

    def update_capture_target(
        self,
        settings: AppSettingsVNext,
        capture_target: CaptureTargetIntent,
    ) -> AppSettingsVNext:
        if self.canonical is None:
            self.canonical = settings
        snapshot = self.persistence.snapshot(self.canonical)
        desktop_audio = self.canonical.intent.desktop_audio
        self.canonical = with_capture_target(
            replace(
                self.canonical,
                intent=replace(
                    self.canonical.intent,
                    desktop_audio=replace(
                        desktop_audio,
                        vad_speech_threshold=settings.intent.desktop_audio.vad_speech_threshold,
                        vad_hangover_ms=settings.intent.desktop_audio.vad_hangover_ms,
                        vad_pre_roll_ms=settings.intent.desktop_audio.vad_pre_roll_ms,
                    ),
                ),
            ),
            capture_target,
        )
        try:
            self.persist()
        except Exception:
            self.canonical = self.persistence.rollback(snapshot)
            raise
        return self.require_canonical()

    def materialize_translation(self, settings: AppSettingsVNext) -> AppSettingsVNext:
        return materialize_canonical_translation_settings(settings)

    def normalize_compatibility(self, settings: AppSettingsVNext) -> AppSettingsVNext:
        return self.normalize(settings)

    def project(
        self,
        settings: AppSettingsVNext,
        *,
        authoritative: bool,
    ) -> AppSettingsVNext:
        normalized = self.normalize(settings)
        if not authoritative:
            self.canonical = normalized
        return normalized

    def project_legacy_delta(
        self,
        base_settings: AppSettingsVNext | None,
        next_settings: AppSettingsVNext,
    ) -> AppSettingsVNext:
        live = self.canonical
        if live is None or base_settings is None:
            return self.normalize(next_settings)
        return apply_canonical_delta(
            live,
            self.normalize(base_settings),
            self.normalize(next_settings),
        )

    def apply_legacy_delta(
        self,
        base_settings: AppSettingsVNext | None,
        next_settings: AppSettingsVNext,
    ) -> AppSettingsVNext:
        return self.apply_canonical_delta(base_settings, next_settings)

    @staticmethod
    def legacy_snapshot_values(settings: AppSettingsVNext) -> dict[str, object]:
        return canonical_snapshot_values(settings)

    def create_legacy_patch_repository(
        self,
        *,
        committed_settings: AppSettingsVNext,
        base_settings: AppSettingsVNext | None = None,
        surface: str = "translation_provider",
        provider_verification_binding: ProviderVerificationBinding | None = None,
        save_failure_sink: Callable[[str], None] | None = None,
    ) -> CanonicalSettingsPatchRepository:
        return self.create_canonical_patch_repository(
            committed_settings=committed_settings,
            base_settings=base_settings,
            surface=surface,
            provider_verification_binding=provider_verification_binding,
            save_failure_sink=save_failure_sink,
        )

    def remember_projection(self, settings: AppSettingsVNext) -> None:
        self.projection_snapshot = copy.deepcopy(settings)

    def begin(self, *, legacy_snapshot: AppSettingsVNext | None = None) -> None:
        if self._mutation_depth == 0:
            self._rollback_snapshot = self.persistence.snapshot(
                legacy_snapshot if legacy_snapshot is not None else self.canonical
            )
            self._rollback_overlay_enabled = self._overlay_enabled
            self._rollback_peer_translation_enabled = self._peer_translation_enabled
            self._rollback_overlay_desktop_locked = self._overlay_desktop_locked
            self._rollback_authoritative = self.authoritative
            self._rollback_pending = True
            if legacy_snapshot is not None and self.projection_snapshot is None:
                self.projection_snapshot = copy.deepcopy(legacy_snapshot)
        self._mutation_depth += 1

    def rollback(self) -> None:
        if not self._rollback_pending:
            return
        self.canonical = self.persistence.rollback(self._rollback_snapshot)
        self._overlay_enabled = self._rollback_overlay_enabled
        self._peer_translation_enabled = self._rollback_peer_translation_enabled
        self._overlay_desktop_locked = self._rollback_overlay_desktop_locked
        self.authoritative = self._rollback_authoritative
        self._mutation_depth = 1
        self.complete()

    def complete(self) -> None:
        if self._mutation_depth == 0:
            return
        self._mutation_depth -= 1
        if self._mutation_depth == 0:
            self._rollback_snapshot = None
            self._rollback_overlay_enabled = False
            self._rollback_peer_translation_enabled = False
            self._rollback_overlay_desktop_locked = False
            self._rollback_authoritative = False
            self._rollback_pending = False

    @property
    def mutation_depth(self) -> int:
        return self._mutation_depth

    @property
    def rollback_pending(self) -> bool:
        return self._rollback_pending


def materialize_canonical_translation_settings(settings: AppSettingsVNext) -> AppSettingsVNext:
    from puripuly_heart.config.llm_profiles import (
        OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
        OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK,
        OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED,
        OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK,
        OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED,
        openrouter_alias_for_fields,
    )

    translation = settings.intent.translation
    model = translation.model
    connection = translation.connection
    if model == "custom_http":
        if connection == "custom_http":
            return settings
        return replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(translation, connection="custom_http"),
            ),
        )
    updates: dict[str, object] = {}
    if model == "gemma4_26b_31b":
        selected_source = "managed" if connection == "managed" else "byok"
        updates = {
            "openrouter_model": "google/gemma-4-26b-a4b-it",
            "openrouter_provider_routing": "gemma4_26b_31b_latency",
            "openrouter_selected_source": selected_source,
            "openrouter_selection_alias": (
                OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED
                if connection == "managed"
                else OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK
            ),
        }
    elif model == "gemma4_31b":
        if connection == "cerebras":
            updates = {
                "openrouter_provider_routing": "default",
                "cerebras": replace(translation.cerebras, llm_model="gemma-4-31b"),
            }
        else:
            selected_source = "managed" if connection == "managed" else "byok"
            updates = {
                "openrouter_model": "google/gemma-4-31b-it",
                "openrouter_provider_routing": "gemma4_31b_latency",
                "openrouter_selected_source": selected_source,
                "openrouter_selection_alias": (
                    OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED
                    if connection == "managed"
                    else OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK
                ),
            }
    elif model == "gemma4":
        selected_source = "managed" if connection == "managed" else "byok"
        openrouter_model = "google/gemma-4-26b-a4b-it"
        updates = {
            "openrouter_model": openrouter_model,
            "openrouter_provider_routing": "gemma4_26b_latency",
            "openrouter_selected_source": selected_source,
            "openrouter_selection_alias": openrouter_alias_for_fields(
                model=openrouter_model,
                source=selected_source,
            ),
        }
    elif model == "deepseek_v4_flash":
        if connection == "official_byok":
            updates = {
                "openrouter_provider_routing": "default",
                "deepseek": replace(translation.deepseek, llm_model="deepseek-v4-flash"),
            }
        else:
            selected_source = "managed" if connection in {"managed", "managed_china"} else "byok"
            openrouter_model = OPENROUTER_MODEL_DEEPSEEK_V4_FLASH
            updates = {
                "openrouter_model": openrouter_model,
                "openrouter_provider_routing": (
                    "deepseek_only" if connection == "managed_china" else "default"
                ),
                "openrouter_selected_source": selected_source,
                "openrouter_selection_alias": openrouter_alias_for_fields(
                    model=openrouter_model,
                    source=selected_source,
                ),
            }
    elif model == "gemini37_flash":
        if connection == "openrouter":
            openrouter_model = "google/gemini-3.7-flash"
            updates = {
                "openrouter_model": openrouter_model,
                "openrouter_provider_routing": "google_gemini_latency",
                "openrouter_selected_source": "byok",
                "openrouter_selection_alias": openrouter_alias_for_fields(
                    model=openrouter_model,
                    source="byok",
                ),
            }
        else:
            updates = {
                "openrouter_provider_routing": "default",
                "gemini": replace(translation.gemini, llm_model="gemini-3.7-flash"),
            }
    elif model == "gemini31_flash_lite":
        if connection == "openrouter":
            openrouter_model = "google/gemini-3.1-flash-lite"
            updates = {
                "openrouter_model": openrouter_model,
                "openrouter_provider_routing": "google_gemini_latency",
                "openrouter_selected_source": "byok",
                "openrouter_selection_alias": openrouter_alias_for_fields(
                    model=openrouter_model,
                    source="byok",
                ),
            }
        else:
            updates = {
                "openrouter_provider_routing": "default",
                "gemini": replace(translation.gemini, llm_model="gemini-3.1-flash-lite"),
            }
    elif model == "local_llm":
        updates = {"openrouter_provider_routing": "default"}
    elif model in {"managed_gemma", "managed_gemma_12b"}:
        updates = {"openrouter_provider_routing": "default"}
    else:
        updates = {
            "openrouter_provider_routing": "default",
            "qwen": replace(translation.qwen, llm_model="qwen3.5-plus"),
        }
    return replace(
        settings,
        intent=replace(settings.intent, translation=replace(translation, **updates)),
    )


def compose_settings_owner(path: Path) -> SettingsOwner:
    return SettingsOwner(path=path, persistence=compose_canonical_settings_persistence())


__all__ = [
    "CanonicalSettingsPatchRepository",
    "SettingsOwner",
    "SettingsOwnerStartResult",
    "compose_canonical_settings_persistence",
    "compose_settings_owner",
    "canonical_snapshot_values",
    "materialize_canonical_translation_settings",
]
