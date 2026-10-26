from __future__ import annotations

import asyncio
import copy
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.application_runtime_logging import ApplicationRuntimeLoggingPort
from puripuly_heart.app.ports.canonical_settings_persistence import (
    CanonicalSettingsPersistenceError,
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
from puripuly_heart.config.settings_vnext import serialization
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
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.owned_thread import run_owned_thread_call
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
class _SettingsCommit:
    settings: AppSettingsVNext
    diagnostics: tuple[ErrorDiagnostics, ...] = ()
    ready: bool = False


@dataclass(slots=True)
class CanonicalSettingsPatchRepository:
    owner: SettingsOwner
    committed_settings: AppSettingsVNext
    base_settings: AppSettingsVNext | None = None
    surface: str = "translation_provider"
    provider_verification_binding: ProviderVerificationBinding | None = None
    commit_sink: Callable[[], None] | None = None
    commit_guard: Callable[[], bool] | None = None
    commit_succeeded: bool = False
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    secondary_omitted_count: int = 0
    _publication: _SettingsCommit | None = None

    async def load(self) -> SettingsSnapshot:
        settings = self.owner.canonical or self.committed_settings
        return SettingsSnapshot(
            values=canonical_snapshot_values(settings),
            revision=None,
        )

    async def save(self, request: SettingsCommitRequest) -> SettingsCommitResult:
        self.commit_succeeded = False
        self._publication = None
        self.secondary_diagnostics = ()
        self.secondary_omitted_count = 0
        try:
            result = await run_owned_thread_call(partial(self._save, request))
        except asyncio.CancelledError:
            if self.commit_succeeded:
                self._notify_commit()
            raise
        if self.commit_succeeded:
            self._notify_commit()
        return result

    def _notify_commit(self) -> None:
        if self._publication is not None:
            self.secondary_diagnostics += self._publication.diagnostics
        if self.commit_sink is not None:
            try:
                self.commit_sink()
            except Exception as exc:
                self.secondary_diagnostics += (
                    _settings_mutation_diagnostics(
                        component="settings_repository",
                        operation="commit_notification",
                        code="settings_commit_notification_failed",
                        surface=self.surface,
                        exception=exc,
                    ),
                )

    def _save(self, request: SettingsCommitRequest) -> SettingsCommitResult:
        publication = None
        try:
            with self.owner._commit_lock:
                if self.commit_guard is not None and not self.commit_guard():
                    return SettingsCommitResult(
                        succeeded=False,
                        snapshot=None,
                        message=None,
                        diagnostics=ErrorDiagnostics(
                            component="settings_repository",
                            operation="save",
                            code="settings_selection_superseded",
                            category="lifecycle",
                            visibility="basic",
                            content_policy="metadata_only",
                            status_code=None,
                            retry_after_ms=None,
                            fields={"surface": self.surface},
                        ),
                    )
                try:
                    source = self.base_settings or self.committed_settings
                    if request.values and all(
                        isinstance(path, str) and "." in path for path in request.values
                    ):
                        source = self.owner.canonical or source
                        next_settings = apply_settings_path_patch(source, request.values)
                    elif "state" in request.values or "intent" in request.values:
                        next_settings = merge_canonical_payload(source, request.values)
                    else:
                        next_settings = copy.deepcopy(self.committed_settings)
                    if not isinstance(next_settings, AppSettingsVNext):
                        raise TypeError("canonical patch repository requires AppSettingsVNext")
                    publication = self.owner._commit_delta_locked(
                        source,
                        next_settings,
                        provider_verification_binding=self.provider_verification_binding,
                    )
                except Exception as exc:
                    diagnostics = self.owner.last_failure or _settings_mutation_diagnostics(
                        component="settings_repository",
                        operation="save",
                        code="settings_save_failed",
                        surface=self.surface,
                        exception=exc,
                    )
                    diagnostics = replace(
                        diagnostics, fields={**diagnostics.fields, "surface": self.surface}
                    )
                    self.secondary_diagnostics = self.owner.persistence_secondary_diagnostics
                    self.secondary_omitted_count = self.owner.persistence_secondary_omitted_count
                    return SettingsCommitResult(
                        succeeded=False,
                        snapshot=None,
                        message=None,
                        diagnostics=diagnostics,
                    )
                self._publication = publication
                self.committed_settings = publication.settings
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
        finally:
            self.owner._release_publication(publication)


def compose_canonical_settings_persistence(
    *,
    retired_asset_cleanup_logging: ApplicationRuntimeLoggingPort | None = None,
) -> CanonicalSettingsPersistencePort:
    return SettingsVNextCanonicalPersistenceAdapter(
        retired_asset_cleanup_logging=retired_asset_cleanup_logging
    )


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
    committed_settings: AppSettingsVNext | None = None
    commit_listener: Callable[[AppSettingsVNext], None] | None = None
    _commit_loop: asyncio.AbstractEventLoop | None = None
    _commit_thread: int | None = None
    last_failure: ErrorDiagnostics | None = None
    notification_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    persistence_secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    persistence_secondary_omitted_count: int = 0
    _commit_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _publication_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _publications: deque[_SettingsCommit] = field(default_factory=deque, repr=False)
    _publishing: bool = False

    def require_canonical(self) -> AppSettingsVNext:
        if self.canonical is None:
            raise RuntimeError("settings owner has no canonical settings")
        return self.canonical

    def require_committed(self) -> AppSettingsVNext:
        if self.committed_settings is None:
            raise RuntimeError("settings owner has no committed settings")
        return self.committed_settings

    def observe_commits(self, listener: Callable[[AppSettingsVNext], None]) -> None:
        try:
            self._commit_loop = asyncio.get_running_loop()
            self._commit_thread = threading.get_ident()
        except RuntimeError:
            pass
        self.commit_listener = listener
        if self.committed_settings is not None:
            listener(self.committed_settings)

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
        if self.commit_listener is not None and self._commit_loop is None:
            try:
                self._commit_loop = asyncio.get_running_loop()
                self._commit_thread = threading.get_ident()
            except RuntimeError:
                pass
        if not self.path.exists():
            self.canonical = new_settings_for_first_run()
            self.persistence.persist(self.path, self.canonical)
            loaded = self.persistence.load_active(self.path)
            self.canonical = loaded.canonical_settings
            self._publish_commit(copy.deepcopy(self.canonical))
            return SettingsOwnerStartResult(
                settings=self.canonical,
                migrated=False,
                backup_path=None,
            )
        loaded = self.persistence.load_active(self.path)
        self.canonical = loaded.canonical_settings
        self._publish_commit(copy.deepcopy(self.canonical))
        return SettingsOwnerStartResult(
            settings=self.canonical,
            migrated=loaded.migrated,
            backup_path=loaded.backup_path,
        )

    def _publish_commit(self, snapshot: AppSettingsVNext) -> None:
        with self._commit_lock:
            publication = self._enqueue_commit_locked(snapshot)
        self._release_publication(publication)

    def _enqueue_commit_locked(self, snapshot: AppSettingsVNext) -> _SettingsCommit:
        publication = _SettingsCommit(snapshot)
        self.committed_settings = snapshot
        with self._publication_lock:
            self._publications.append(publication)
        return publication

    def _release_publication(self, publication: _SettingsCommit | None) -> None:
        if publication is not None:
            with self._publication_lock:
                publication.ready = True
            self._dispatch_publications()

    def _dispatch_publications(self) -> None:
        with self._publication_lock:
            if self._publishing or not self._publications:
                return
            self._publishing = True
        loop = self._commit_loop
        if loop is not None and threading.get_ident() != self._commit_thread:
            try:
                loop.call_soon_threadsafe(self._drain_publications)
            except RuntimeError as exc:
                self._fail_publications(exc)
        else:
            self._drain_publications()

    def _fail_publications(self, exc: Exception) -> None:
        diagnostics = (
            _settings_mutation_diagnostics(
                component="settings_repository",
                operation="commit_notification",
                code="settings_commit_notification_failed",
                exception=exc,
            ),
        )
        with self._publication_lock:
            while self._publications:
                self._publications.popleft().diagnostics = diagnostics
            self._publishing = False
        self.notification_diagnostics = diagnostics

    def _drain_publications(self) -> None:
        while True:
            with self._publication_lock:
                if not self._publications or not self._publications[0].ready:
                    self._publishing = False
                    return
                publication = self._publications.popleft()
            if self.commit_listener is not None:
                try:
                    self.commit_listener(publication.settings)
                except Exception as exc:
                    publication.diagnostics = (
                        _settings_mutation_diagnostics(
                            component="settings_repository",
                            operation="commit_notification",
                            code="settings_commit_notification_failed",
                            exception=exc,
                        ),
                    )
                    self.notification_diagnostics = publication.diagnostics

    def _persist_locked(self) -> _SettingsCommit:
        self.last_failure = None
        self.notification_diagnostics = ()
        self.persistence_secondary_diagnostics = ()
        self.persistence_secondary_omitted_count = 0
        if (
            self.commit_listener is not None
            and threading.get_ident() != self._commit_thread
            and (self._commit_loop is None or not self._commit_loop.is_running())
        ):
            raise RuntimeError("settings commit observer loop is unavailable")
        snapshot = copy.deepcopy(self.require_canonical())
        try:
            self.persistence.persist(self.path, snapshot)
        except Exception as exc:
            self.last_failure = (
                exc.diagnostics if isinstance(exc, CanonicalSettingsPersistenceError) else None
            ) or _settings_mutation_diagnostics(
                component="settings_repository",
                operation="save",
                code="settings_save_failed",
                exception=exc,
            )
            self.persistence_secondary_diagnostics = (
                exc.secondary_diagnostics
                if isinstance(exc, CanonicalSettingsPersistenceError)
                else ()
            )
            self.persistence_secondary_omitted_count = (
                exc.secondary_omitted_count
                if isinstance(exc, CanonicalSettingsPersistenceError)
                else 0
            )
            raise
        return self._enqueue_commit_locked(snapshot)

    def _commit_delta_locked(
        self,
        base_settings: AppSettingsVNext | None,
        next_settings: AppSettingsVNext,
        *,
        provider_verification_binding: ProviderVerificationBinding | None = None,
        rollback_snapshot: AppSettingsVNext | None = None,
    ) -> _SettingsCommit:
        self.begin(snapshot=rollback_snapshot)
        try:
            self.apply_canonical_delta(base_settings, next_settings)
            if provider_verification_binding is not None:
                self.bind_provider_verification(provider_verification_binding)
            publication = self._persist_locked()
        except BaseException:
            try:
                self.rollback()
            except Exception as exc:
                self.persistence_secondary_diagnostics += (
                    replace(
                        _settings_mutation_diagnostics(
                            component="settings_repository",
                            operation="rollback",
                            code="settings_rollback_failed",
                            exception=exc,
                        ),
                        cause_id=self.last_failure.diagnostic_id if self.last_failure else None,
                    ),
                )
            raise
        self.canonical = publication.settings
        self.remember_projection(publication.settings)
        self.complete()
        return publication

    def commit_delta(
        self,
        base_settings: AppSettingsVNext | None,
        next_settings: AppSettingsVNext,
    ) -> AppSettingsVNext:
        publication = None
        try:
            with self._commit_lock:
                publication = self._commit_delta_locked(base_settings, next_settings)
                return publication.settings
        finally:
            self._release_publication(publication)

    async def commit_delta_async(
        self,
        base_settings: AppSettingsVNext | None,
        next_settings: AppSettingsVNext,
    ) -> AppSettingsVNext:
        return await run_owned_thread_call(partial(self.commit_delta, base_settings, next_settings))

    def persist(self) -> None:
        publication = None
        try:
            with self._commit_lock:
                publication = self._persist_locked()
        finally:
            self._release_publication(publication)

    def save_current(
        self,
        *,
        failure_sink: Callable[[BaseException], None] | None = None,
    ) -> bool:
        try:
            self.persist_current()
        except Exception as exc:
            if failure_sink is not None:
                try:
                    failure_sink(exc)
                except Exception as notification_exc:
                    self.notification_diagnostics = (
                        _settings_mutation_diagnostics(
                            component="settings_repository",
                            operation="failure_notification",
                            code="settings_failure_notification_failed",
                            exception=notification_exc,
                        ),
                    )
            return False
        return True

    def persist_current(self) -> None:
        publication = None
        try:
            with self._commit_lock:
                live = self.require_canonical()
                baseline = self.projection_snapshot or live
                publication = self._commit_delta_locked(
                    baseline,
                    live,
                    rollback_snapshot=baseline if self.mutation_depth == 0 else None,
                )
        finally:
            self._release_publication(publication)

    def with_telemetry_enabled(
        self,
        settings: AppSettingsVNext,
        enabled: bool,
    ) -> AppSettingsVNext:
        return with_telemetry_enabled(settings, enabled)

    def persist_telemetry_preference(self, enabled: bool) -> AppSettingsVNext:
        publication = None
        try:
            with self._commit_lock:
                settings = self.canonical
                if settings is None:
                    settings = (
                        self.persistence.load_active(self.path).canonical_settings
                        if self.path.exists()
                        else new_settings_for_first_run()
                    )
                publication = self._commit_delta_locked(
                    settings, self.with_telemetry_enabled(settings, enabled)
                )
                verified = self.persistence.load_active(self.path).canonical_settings
                if verified != publication.settings:
                    raise RuntimeError("persisted telemetry preference did not verify")
                return verified
        finally:
            self._release_publication(publication)

    def build_managed_openrouter_byok_target(
        self,
        current_settings: AppSettingsVNext | None = None,
    ) -> AppSettingsVNext | None:
        from puripuly_heart.config.llm_profiles import (
            OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
            OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK,
            get_openrouter_llm_profile,
            get_openrouter_selection_alias_for_model_and_source,
        )
        from puripuly_heart.config.translation_values import (
            provider_llm_for_translation,
        )

        settings = current_settings if current_settings is not None else self.canonical
        if settings is None:
            return None
        translation = settings.intent.translation
        if provider_llm_for_translation(translation.model, translation.connection) != "openrouter":
            return None
        if translation.openrouter_selected_source != "managed":
            return None
        combined_gemma = translation.model in {"gemma4_26b_31b", "gemma4", "gemma4_31b"}
        if combined_gemma:
            openrouter_model = OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT
            alias_value = OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK
        else:
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
        history["gemma4_26b_31b" if combined_gemma else translation.model] = "openrouter"
        if combined_gemma:
            history.pop("gemma4", None)
            history.pop("gemma4_31b", None)
        return replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(
                    translation,
                    model="gemma4_26b_31b" if combined_gemma else translation.model,
                    connection="openrouter",
                    connection_history=history,
                    openrouter_selection_alias=alias_value,
                    openrouter_selected_source="byok",
                    openrouter_model=openrouter_model,
                    openrouter_provider_routing=(
                        "gemma4_26b_31b_latency" if combined_gemma else "default"
                    ),
                ),
            ),
        )

    def persist_managed_identity(
        self,
        settings: AppSettingsVNext,
        *,
        bound_managed_snapshot: object | None = None,
    ) -> None:
        publication = None
        try:
            with self._commit_lock:
                active_settings = self.canonical or settings
                managed_baseline = (
                    bound_managed_snapshot
                    if bound_managed_snapshot is not None
                    else (self.projection_snapshot or active_settings).state.managed_connection
                )
                managed_delta = _managed_connection_delta(
                    managed_baseline,
                    settings.state.managed_connection,
                )
                next_settings = replace(
                    active_settings,
                    state=replace(
                        active_settings.state,
                        managed_connection=replace(
                            active_settings.state.managed_connection,
                            **managed_delta,
                        ),
                    ),
                )
                publication = self._commit_delta_locked(active_settings, next_settings)
        finally:
            self._release_publication(publication)

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
        if success:
            if binding is None:
                raise RuntimeError("verified provider requires evidence binding")
            if active_secret != key:
                raise RuntimeError(
                    "verified credential does not match the active SecretStore value"
                )
        publication = None
        try:
            with self._commit_lock:
                live = self.require_canonical()
                next_settings = replace(
                    live,
                    state=replace(
                        live.state,
                        provider_verification=replace(
                            live.state.provider_verification,
                            **{provider: ProviderVerificationEntry(status="unknown")},
                        ),
                    ),
                )
                publication = self._commit_delta_locked(
                    live,
                    next_settings,
                    provider_verification_binding=binding if success else None,
                )
        finally:
            self._release_publication(publication)

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
        commit_sink: Callable[[], None] | None = None,
        commit_guard: Callable[[], bool] | None = None,
    ) -> CanonicalSettingsPatchRepository:
        return CanonicalSettingsPatchRepository(
            owner=self,
            committed_settings=committed_settings,
            base_settings=base_settings,
            surface=surface,
            provider_verification_binding=provider_verification_binding,
            commit_sink=commit_sink,
            commit_guard=commit_guard,
        )

    def apply_capture_target(self, capture_target: CaptureTargetIntent) -> AppSettingsVNext:
        publication = None
        try:
            with self._commit_lock:
                live = self.canonical or new_settings_for_first_run()
                publication = self._commit_delta_locked(
                    live, with_capture_target(live, capture_target)
                )
                return publication.settings
        finally:
            self._release_publication(publication)

    def update_capture_target(
        self,
        settings: AppSettingsVNext,
        capture_target: CaptureTargetIntent,
    ) -> AppSettingsVNext:
        publication = None
        try:
            with self._commit_lock:
                live = self.canonical or settings
                next_settings = with_capture_target(
                    replace(
                        live,
                        intent=replace(
                            live.intent,
                            desktop_audio=replace(
                                live.intent.desktop_audio,
                                vad_speech_threshold=settings.intent.desktop_audio.vad_speech_threshold,
                                vad_hangover_ms=settings.intent.desktop_audio.vad_hangover_ms,
                                vad_pre_roll_ms=settings.intent.desktop_audio.vad_pre_roll_ms,
                            ),
                        ),
                    ),
                    capture_target,
                )
                publication = self._commit_delta_locked(live, next_settings)
                return publication.settings
        finally:
            self._release_publication(publication)

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

    def project_canonical_delta(
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

    def remember_projection(self, settings: AppSettingsVNext) -> None:
        self.projection_snapshot = copy.deepcopy(settings)

    def begin(self, *, snapshot: AppSettingsVNext | None = None) -> None:
        if self._mutation_depth == 0:
            self._rollback_snapshot = self.persistence.snapshot(
                snapshot if snapshot is not None else self.canonical
            )
            self._rollback_overlay_enabled = self._overlay_enabled
            self._rollback_peer_translation_enabled = self._peer_translation_enabled
            self._rollback_overlay_desktop_locked = self._overlay_desktop_locked
            self._rollback_authoritative = self.authoritative
            self._rollback_pending = True
            if snapshot is not None and self.projection_snapshot is None:
                self.projection_snapshot = copy.deepcopy(snapshot)
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
        OPENROUTER_MODEL_DEEPSEEK_V4_FLASH_41,
        OPENROUTER_MODEL_GPT_6_LUNA,
        OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK,
        OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED,
        openrouter_alias_for_fields,
    )

    translation = settings.intent.translation
    if translation.model == "index_translate_35b":
        return replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(
                    translation,
                    connection="public_api",
                    openrouter_selected_source="none",
                    openrouter_selection_alias=None,
                    openrouter_provider_routing="default",
                ),
            ),
        )
    model = "gemma4_26b_31b" if translation.model in {"gemma4", "gemma4_31b"} else translation.model
    if model != translation.model:
        translation = replace(translation, model=model)
    if model in {"gemini31_flash_lite", "gemini37_flash"}:
        translation = replace(translation, model="gemini_flash")
        model = "gemini_flash"
    elif model == "qwen35_plus":
        translation = replace(translation, model="qwen38_flash")
        model = "qwen38_flash"
    connection = translation.connection
    if model == "gpt_6_luna" and connection not in {"chatgpt", "openrouter", "official_byok"}:
        connection = "chatgpt"
        translation = replace(translation, connection=connection)
    retired_managed_connections = {"managed", "managed_china"}
    history = dict(translation.connection_history)
    if model == "deepseek_v4_flash" and connection in retired_managed_connections:
        history["deepseek_v4_flash_41"] = connection
        if history.get("deepseek_v4_flash") in retired_managed_connections:
            history.pop("deepseek_v4_flash", None)
        previous_llm_model = translation.previous_llm_model
        if previous_llm_model == "deepseek_v4_flash":
            previous_llm_model = "deepseek_v4_flash_41"
        translation = replace(
            translation,
            model="deepseek_v4_flash_41",
            connection_history=history,
            previous_llm_model=previous_llm_model,
        )
        model = "deepseek_v4_flash_41"
    elif history.get("deepseek_v4_flash") in retired_managed_connections:
        saved_connection = history.pop("deepseek_v4_flash")
        history.setdefault("deepseek_v4_flash_41", saved_connection)
        previous_llm_model = translation.previous_llm_model
        if previous_llm_model == "deepseek_v4_flash":
            previous_llm_model = "deepseek_v4_flash_41"
        translation = replace(
            translation,
            connection_history=history,
            previous_llm_model=previous_llm_model,
        )
    if model == "deepseek_v4_flash" and connection == "official_byok":
        translation = replace(translation, model="deepseek_v4_flash_41")
        model = "deepseek_v4_flash_41"
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
    elif model == "deepseek_v4_flash":
        openrouter_model = OPENROUTER_MODEL_DEEPSEEK_V4_FLASH
        updates = {
            "openrouter_model": openrouter_model,
            "openrouter_provider_routing": "deepseek_v4_flash_latency",
            "openrouter_selected_source": "byok",
            "openrouter_selection_alias": openrouter_alias_for_fields(
                model=openrouter_model,
                source="byok",
            ),
        }
    elif model == "deepseek_v4_flash_41":
        if connection == "official_byok":
            updates = {
                "openrouter_model": OPENROUTER_MODEL_DEEPSEEK_V4_FLASH_41,
                "openrouter_provider_routing": "default",
                "deepseek": replace(translation.deepseek, llm_model="deepseek-flash"),
            }
        else:
            selected_source = "managed" if connection in {"managed", "managed_china"} else "byok"
            openrouter_model = OPENROUTER_MODEL_DEEPSEEK_V4_FLASH_41
            updates = {
                "openrouter_model": openrouter_model,
                "openrouter_provider_routing": "deepseek_v4_flash_41_strict",
                "openrouter_selected_source": selected_source,
                "openrouter_selection_alias": openrouter_alias_for_fields(
                    model=openrouter_model,
                    source=selected_source,
                ),
            }
    elif model == "gpt_6_luna":
        if connection == "openrouter":
            updates = {
                "openrouter_model": OPENROUTER_MODEL_GPT_6_LUNA,
                "openrouter_selected_source": "byok",
                "openrouter_provider_routing": "default",
                "openrouter_selection_alias": openrouter_alias_for_fields(
                    model=OPENROUTER_MODEL_GPT_6_LUNA,
                    source="byok",
                ),
            }
        else:
            updates = {
                "openrouter_selected_source": "none",
                "openrouter_selection_alias": None,
                "openrouter_provider_routing": "default",
            }
    elif model == "gemini_flash":
        if connection == "openrouter":
            openrouter_model = "google/gemini-3.8-flash"
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
                "gemini": replace(translation.gemini, llm_model="gemini-3.8-flash"),
            }
    elif model == "qwen38_flash":
        updates = {
            "openrouter_provider_routing": "default",
            "qwen": replace(translation.qwen, llm_model="qwen3.8-flash"),
        }
    else:
        updates = {"openrouter_provider_routing": "default"}
    return replace(
        settings,
        intent=replace(settings.intent, translation=replace(translation, **updates)),
    )


def compose_settings_owner(
    path: Path,
    *,
    retired_asset_cleanup_logging: ApplicationRuntimeLoggingPort | None = None,
) -> SettingsOwner:
    return SettingsOwner(
        path=path,
        persistence=compose_canonical_settings_persistence(
            retired_asset_cleanup_logging=retired_asset_cleanup_logging
        ),
    )


__all__ = [
    "CanonicalSettingsPatchRepository",
    "SettingsOwner",
    "SettingsOwnerStartResult",
    "compose_canonical_settings_persistence",
    "compose_settings_owner",
    "canonical_snapshot_values",
    "materialize_canonical_translation_settings",
]
