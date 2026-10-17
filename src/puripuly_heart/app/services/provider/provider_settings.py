from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

from puripuly_heart.app.ports.canonical_settings_persistence import (
    ProviderVerificationBinding,
)
from puripuly_heart.app.ports.runtime_apply import RuntimeApplyPort, RuntimeApplyRequest
from puripuly_heart.app.ports.secret_store import SecretStorePort
from puripuly_heart.app.ports.settings_repository import (
    SettingsCommitRequest,
    SettingsCommitResult,
    SettingsSnapshot,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.secret_settings_transaction import (
    SecretSettingsTransaction,
)
from puripuly_heart.app.services.settings_mutation import (
    SettingsMutationInterrupted,
    SettingsMutationService,
    SttLanguageAudioSettingsMutation,
    TranslationProviderSettingsMutation,
)
from puripuly_heart.app.services.settings_mutation_legacy import (
    apply_settings_path_patch,
    build_stt_language_audio_settings_path_patch,
    build_translation_provider_settings_path_patch,
    settings_path_mutation_validator_for_command,
)
from puripuly_heart.app.services.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.config.alibaba_connection import resolve_alibaba_connection
from puripuly_heart.config.settings_vnext.schema import ProviderVerificationEntry
from puripuly_heart.core.messages import (
    RUNTIME_APPLY_STATUS_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    ErrorDiagnostics,
    TransactionResult,
)

from .provider_runtime_apply import (
    NoopRuntimeApply,
    ProviderRuntimeApplyAdapter,
    ProviderRuntimeApplyPlan,
    ProviderRuntimeOwner,
    _provider_runtime_apply_unavailable_result,
    _runtime_apply_failed_result,
    _runtime_apply_result_as_degraded_transaction,
    _settings_mutation_diagnostics,
    _settings_mutation_secondary_diagnostics,
    _stt_language_audio_save_failed_transaction_result,
    _translation_provider_save_failed_transaction_result,
)
from .provider_secret_change import (
    ProviderSecretChangeExecution,
    ProviderSecretChangeOwner,
    ProviderSecretChangeRequest,
)
from .provider_verification_binding import (
    ProviderVerificationBindingOwner,
)


class ProviderCompatibilityProjection(Protocol):
    intent: object
    state: object


ProviderSecretStoreFactory = Callable[[ProviderCompatibilityProjection], SecretStorePort]
ProviderActiveSecretProvider = Callable[[ProviderCompatibilityProjection, str], str | None]
ProviderSettingsMerge = Callable[[ProviderCompatibilityProjection], ProviderCompatibilityProjection]
ProviderSettingsAsyncEffect = Callable[[ProviderCompatibilityProjection], Awaitable[None]]
ProviderSettingsRoute = Callable[[ProviderCompatibilityProjection], Awaitable[bool]]
ProviderSettingsSync = Callable[[ProviderCompatibilityProjection], None]
ProviderSettingsPredicate = Callable[
    [ProviderCompatibilityProjection, ProviderCompatibilityProjection], bool
]
ProviderSettingsCompensation = Callable[..., Awaitable[None]]
ProviderSettingsMutationServiceProvider = Callable[[], SettingsMutationService | None]
ProviderOrder24PatchProvider = Callable[
    [ProviderCompatibilityProjection],
    tuple[ProviderCompatibilityProjection, dict[str, object]] | None,
]
ProviderSupersededSettingsConsumer = Callable[[ProviderCompatibilityProjection], bool]


class ProviderStrictSettingsSaveFailed(Exception):
    def __init__(self, diagnostics: ErrorDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__("Provider settings persistence failed")


def _noop_rolling_stt_secret_rebind(_secret_key: str, _value: str) -> None:
    return None


class _HttpExtensionSecretSettingsRepository:
    async def load(self) -> SettingsSnapshot:
        return SettingsSnapshot(values={}, revision=None)

    async def save(self, request: SettingsCommitRequest) -> SettingsCommitResult:
        return SettingsCommitResult(
            succeeded=True,
            snapshot=SettingsSnapshot(values=request.values, revision=None),
            message=None,
            diagnostics=None,
        )


@dataclass(slots=True)
class ProviderApplicationOwner:
    settings: SettingsOwner
    runtime: ProviderRuntimeOwner
    merge_settings: ProviderSettingsMerge
    preserve_before_replace: ProviderSettingsAsyncEffect
    sync_ui: Callable[[], None]
    order24_patch_provider: ProviderOrder24PatchProvider
    apply_order24: ProviderSettingsRoute
    remember_order22: ProviderSettingsSync
    mutation_service_provider: ProviderSettingsMutationServiceProvider
    results: SettingsTransactionResultOwner
    sync_memory: ProviderSettingsSync
    capture_runtime_signatures: Callable[[], None]
    sync_stt_signatures: ProviderSettingsSync
    consume_superseded_settings: ProviderSupersededSettingsConsumer
    active_local_asr_change: ProviderSettingsPredicate
    compensate_local_asr: ProviderSettingsCompensation
    llm_retry_pending: Callable[[], bool]
    mark_llm_retry: Callable[[], None]
    rolling_stt_secret_rebind: Callable[[str, str], None] = _noop_rolling_stt_secret_rebind

    def rebind_rolling_stt_secret(self, secret_key: str, value: str) -> None:
        self.rolling_stt_secret_rebind(secret_key, value)

    async def apply(
        self,
        pending: ProviderCompatibilityProjection | None = None,
        *,
        force_rebuild_llm: bool = False,
        persist_settings: bool = True,
        refresh_ui: bool = True,
    ) -> bool:
        next_settings = self.settings.canonical if pending is None else self.merge_settings(pending)
        if next_settings is None:
            return False
        preserved = await self.preserve_before_replace(next_settings)
        if preserved is not None:
            next_settings = preserved
        try:
            if persist_settings and pending is not None and not force_rebuild_llm:
                if await self._apply_combined(next_settings):
                    return (
                        self.results.current is not None
                        and self.results.current.status
                        == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
                    )
                if await self._apply_translation(next_settings):
                    return (
                        self.results.current is not None
                        and self.results.current.status
                        == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
                    )
                if await self.apply_order24(next_settings):
                    return (
                        self.results.current is not None
                        and self.results.current.status
                        == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
                    )
            applied = await self._apply_direct(
                next_settings,
                force_rebuild_llm=force_rebuild_llm,
                persist_settings=persist_settings,
            )
            return applied and (
                not persist_settings
                or (
                    self.results.current is not None
                    and self.results.current.status
                    == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
                )
            )
        finally:
            if refresh_ui:
                self.sync_ui()

    async def _apply_combined(self, next_settings: ProviderCompatibilityProjection) -> bool:
        base_settings = self.settings.canonical
        if base_settings is None:
            return False
        order21_values = build_translation_provider_settings_path_patch(
            base_settings,
            next_settings,
        )
        order22_values = build_stt_language_audio_settings_path_patch(
            base_settings,
            next_settings,
        )
        order24_base_and_patch = self.order24_patch_provider(next_settings)
        if order24_base_and_patch is None:
            return False
        _, order24_values = order24_base_and_patch
        if sum(bool(values) for values in (order21_values, order22_values, order24_values)) < 2:
            return False
        committed_results: list[TransactionResult] = []

        async def route_patch(
            values: dict[str, object],
            route: ProviderSettingsRoute,
        ) -> bool:
            current = self.settings.canonical
            if current is None:
                return False
            patch_settings = apply_settings_path_patch(current, values)
            if not await route(patch_settings):
                return False
            result = self.results.current
            if result is not None and _settings_mutation_committed(result):
                committed_results.append(result)
            return True

        if order21_values:
            if not await route_patch(order21_values, self._apply_translation):
                return False
            if not self._last_result_committed():
                return True
        if order22_values:
            if not await route_patch(order22_values, self._apply_stt_language_audio):
                return False
            if not self._last_result_committed():
                return True
        if order24_values:
            current = self.settings.canonical
            if current is None:
                return True
            order24_settings = apply_settings_path_patch(current, order24_values)
            if not await self.apply_order24(order24_settings):
                return False
            if not self._last_result_committed():
                return True

        committed_before_full_draft = (
            copy.deepcopy(self.settings.canonical) if self.settings.canonical is not None else None
        )
        if self.settings.canonical is not None and self.settings.snapshot_values(
            self.settings.canonical
        ) != self.settings.snapshot_values(next_settings):
            try:
                await self._apply_direct(
                    next_settings,
                    force_rebuild_llm=False,
                    route_order22=False,
                    strict_persistence_errors=True,
                )
            except ProviderStrictSettingsSaveFailed as exc:
                if committed_before_full_draft is not None:
                    self.sync_memory(committed_before_full_draft)
                self._set_result(
                    _translation_provider_save_failed_transaction_result(
                        operation="apply_order21_order22_order24_provider_full_draft_save",
                        exception=exc.__cause__,
                    )
                )
            except Exception as exc:
                self._set_result(
                    _runtime_apply_result_as_degraded_transaction(
                        _runtime_apply_failed_result(
                            operation="apply_order21_order22_order24_provider_runtime",
                            code="provider_runtime_apply_exception",
                            surface="translation_provider",
                            exception=exc,
                        )
                    )
                )
        if (
            self.results.current is not None
            and self.results.current.status
            == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
            and committed_results
        ):
            degraded = next(
                (
                    result
                    for result in committed_results
                    if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
                ),
                None,
            )
            if degraded is not None:
                self._set_result(degraded)
        return True

    async def apply_translation_selection(
        self,
        next_settings: ProviderCompatibilityProjection,
        *,
        commit_guard: Callable[[], bool],
        runtime_guard: Callable[[], bool] | None = None,
    ) -> TransactionResult | None:
        if not commit_guard():
            return None
        try:
            if not await self._apply_translation(
                next_settings, commit_guard=commit_guard, runtime_guard=runtime_guard
            ):
                return None
            return self.results.current
        finally:
            self.sync_ui()

    async def _apply_translation(
        self,
        next_settings: ProviderCompatibilityProjection,
        *,
        commit_guard: Callable[[], bool] | None = None,
        runtime_guard: Callable[[], bool] | None = None,
    ) -> bool:
        base_settings = self.settings.canonical
        if base_settings is None:
            return False
        patch_values = build_translation_provider_settings_path_patch(
            base_settings,
            next_settings,
        )
        if not patch_values:
            return False
        committed_settings = copy.deepcopy(base_settings)
        committed_settings = apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self.settings.snapshot_values(
            committed_settings
        ) != self.settings.snapshot_values(next_settings)
        plan = self.runtime.build_plan(
            committed_settings,
            force_rebuild_llm=False,
            canonical_settings=self.settings.project_canonical_delta(
                base_settings,
                committed_settings,
            ),
        )
        repository = self.settings.create_canonical_patch_repository(
            base_settings=base_settings,
            committed_settings=committed_settings,
            commit_guard=commit_guard,
        )
        runtime_apply = ProviderRuntimeApplyAdapter(
            owner=self.runtime,
            settings=committed_settings,
            plan=plan,
            authority_guard=runtime_guard,
        )
        command = TranslationProviderSettingsMutation(values=patch_values)
        service = self.mutation_service_provider() or SettingsMutationService(
            settings_repository=repository,
            runtime_apply=runtime_apply,
            validator=settings_path_mutation_validator_for_command(command),
        )
        result: TransactionResult | None = None
        try:
            result = await service.mutate(
                command.to_mutation_request(
                    expected_revision=None,
                    correlation_id=None,
                )
            )
        except SettingsMutationInterrupted as exc:
            result = exc.transaction_result
            self._set_result(result)
            raise
        finally:
            if getattr(repository, "commit_succeeded", False) or (
                result is not None and _settings_mutation_committed(result)
            ):
                self.settings.complete()
            else:
                self.settings.rollback()
        if result is None:
            raise RuntimeError("provider settings mutation completed without a result")
        self._set_result(result)
        if not _settings_mutation_committed(result):
            return True
        self.settings.canonical = committed_settings
        if has_out_of_scope_draft:
            fallback_plan = self.runtime.build_plan(
                next_settings,
                force_rebuild_llm=False,
                canonical_settings=self.settings.project_canonical_delta(
                    committed_settings,
                    next_settings,
                ),
            )
            try:
                await self._apply_direct(
                    next_settings,
                    force_rebuild_llm=False,
                    plan=fallback_plan,
                    route_order22=False,
                    strict_persistence_errors=True,
                )
            except ProviderStrictSettingsSaveFailed as exc:
                preserve_retry = self.llm_retry_pending()
                self.sync_memory(committed_settings)
                if preserve_retry:
                    self.mark_llm_retry()
                self._set_result(
                    _translation_provider_save_failed_transaction_result(
                        operation="apply_translation_provider_full_draft_save",
                        exception=exc.__cause__,
                    )
                )
            except Exception as exc:
                self._set_result(
                    _runtime_apply_result_as_degraded_transaction(
                        _runtime_apply_failed_result(
                            operation="apply_translation_provider_runtime",
                            code="provider_runtime_apply_exception",
                            surface="translation_provider",
                            exception=exc,
                        )
                    )
                )
            else:
                unavailable = _provider_runtime_apply_unavailable_result(
                    owner=self.runtime,
                    settings=next_settings,
                    plan=fallback_plan,
                    operation="apply_translation_provider_runtime",
                    surface="translation_provider",
                )
                if unavailable is not None:
                    self._set_result(_runtime_apply_result_as_degraded_transaction(unavailable))
        elif result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
            self.sync_stt_signatures(committed_settings)
        self.remember_order22(self.settings.canonical)
        return True

    async def _apply_stt_language_audio(
        self, next_settings: ProviderCompatibilityProjection
    ) -> bool:
        base_settings = self.settings.canonical
        if base_settings is None:
            return False
        patch_values = build_stt_language_audio_settings_path_patch(
            base_settings,
            next_settings,
        )
        if not patch_values:
            return False
        committed_settings = copy.deepcopy(base_settings)
        committed_settings = apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self.settings.snapshot_values(
            committed_settings
        ) != self.settings.snapshot_values(next_settings)
        plan = self.runtime.build_plan(
            committed_settings,
            force_rebuild_llm=False,
            canonical_settings=self.settings.project_canonical_delta(
                base_settings,
                committed_settings,
            ),
        )
        repository = self.settings.create_canonical_patch_repository(
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface="stt_language_audio",
        )
        runtime_apply: RuntimeApplyPort = ProviderRuntimeApplyAdapter(
            owner=self.runtime,
            settings=committed_settings,
            plan=plan,
            surface="stt_language_audio",
            operation="apply_stt_language_audio_provider_runtime",
        )
        if has_out_of_scope_draft:
            runtime_apply = NoopRuntimeApply()
        command = SttLanguageAudioSettingsMutation(values=patch_values)
        service = self.mutation_service_provider() or SettingsMutationService(
            settings_repository=repository,
            runtime_apply=runtime_apply,
            validator=settings_path_mutation_validator_for_command(command),
        )
        result: TransactionResult | None = None
        try:
            result = await service.mutate(
                command.to_mutation_request(
                    expected_revision=None,
                    correlation_id=None,
                )
            )
        except SettingsMutationInterrupted as exc:
            result = exc.transaction_result
            self._set_result(result)
            raise
        finally:
            if getattr(repository, "commit_succeeded", False) or (
                result is not None and _settings_mutation_committed(result)
            ):
                self.settings.complete()
            else:
                self.settings.rollback()
        if result is None:
            raise RuntimeError("provider settings mutation completed without a result")
        self._set_result(result)
        if not _settings_mutation_committed(result):
            self.settings.canonical = copy.deepcopy(base_settings)
            self.remember_order22(self.settings.canonical)
            return True
        if self.consume_superseded_settings(committed_settings):
            self.remember_order22(self.settings.canonical)
            return True
        if (
            not has_out_of_scope_draft
            and result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
            and self.active_local_asr_change(
                self.settings.project(base_settings, authoritative=True),
                self.settings.project(committed_settings, authoritative=True),
            )
        ):
            try:
                await self.compensate_local_asr(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                )
            except Exception as exc:
                secondary = _settings_mutation_diagnostics(
                    component="provider_application",
                    operation="compensate_local_asr",
                    code="settings_compensation_failed",
                    exception=exc,
                )
                self._set_result(
                    replace(
                        result,
                        secondary_diagnostics=(
                            *result.secondary_diagnostics,
                            replace(
                                secondary,
                                cause_id=(
                                    result.diagnostics.diagnostic_id if result.diagnostics else None
                                ),
                            ),
                        ),
                    )
                )
            self.remember_order22(self.settings.canonical)
            return True
        if has_out_of_scope_draft:
            fallback_plan = self.runtime.build_plan(
                next_settings,
                force_rebuild_llm=False,
                canonical_settings=self.settings.project_canonical_delta(
                    committed_settings,
                    next_settings,
                ),
            )
            try:
                await self._apply_direct(
                    next_settings,
                    force_rebuild_llm=False,
                    plan=fallback_plan,
                    route_order22=False,
                    strict_persistence_errors=True,
                )
            except ProviderStrictSettingsSaveFailed as exc:
                await self._resync_committed_provider_runtime(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                    plan=plan,
                    failure_result=_stt_language_audio_save_failed_transaction_result(
                        operation="apply_stt_language_audio_provider_full_draft_save",
                        exception=exc,
                    ),
                )
            except Exception as exc:
                self._set_result(
                    _runtime_apply_result_as_degraded_transaction(
                        _runtime_apply_failed_result(
                            operation="apply_stt_language_audio_provider_runtime",
                            code="provider_runtime_apply_exception",
                            surface="stt_language_audio",
                            exception=exc,
                        )
                    )
                )
            else:
                unavailable = _provider_runtime_apply_unavailable_result(
                    owner=self.runtime,
                    settings=next_settings,
                    plan=fallback_plan,
                    operation="apply_stt_language_audio_provider_runtime",
                    surface="stt_language_audio",
                )
                if unavailable is not None:
                    self._set_result(_runtime_apply_result_as_degraded_transaction(unavailable))
        else:
            self.settings.canonical = committed_settings
            if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                self.sync_stt_signatures(committed_settings)
        self.remember_order22(self.settings.canonical)
        return True

    async def _apply_direct(
        self,
        next_settings: ProviderCompatibilityProjection,
        *,
        force_rebuild_llm: bool,
        plan: ProviderRuntimeApplyPlan | None = None,
        route_order22: bool = True,
        strict_persistence_errors: bool = False,
        persist_settings: bool = True,
    ) -> bool:
        if persist_settings and route_order22 and not force_rebuild_llm and plan is None:
            if await self._apply_stt_language_audio(next_settings):
                return self._last_result_committed()
        if not persist_settings:
            self.capture_runtime_signatures()
            if plan is None:
                plan = self.runtime.build_plan(
                    next_settings,
                    force_rebuild_llm=force_rebuild_llm,
                )
            runtime_result = await ProviderRuntimeApplyAdapter(
                owner=self.runtime,
                settings=next_settings,
                plan=plan,
            ).apply_runtime(
                RuntimeApplyRequest(
                    settings_values=self.settings.snapshot_values(next_settings),
                    reason="provider_runtime_only",
                    correlation_id=None,
                )
            )
            if runtime_result.status != RUNTIME_APPLY_STATUS_APPLIED:
                self._set_result(_runtime_apply_result_as_degraded_transaction(runtime_result))
            return runtime_result.status == RUNTIME_APPLY_STATUS_APPLIED
        self.settings.begin(snapshot=self.settings.projection_snapshot or self.settings.canonical)
        committed = False
        try:
            self.capture_runtime_signatures()
            self.settings.apply_canonical_delta(
                self.settings.projection_snapshot or self.settings.canonical,
                next_settings,
            )
            if plan is None:
                plan = self.runtime.build_plan(
                    next_settings,
                    force_rebuild_llm=force_rebuild_llm,
                )
            self.settings.canonical = next_settings
            if strict_persistence_errors:
                try:
                    self.settings.persist()
                except Exception as exc:
                    self.settings.rollback()
                    raise ProviderStrictSettingsSaveFailed(
                        self.settings.last_failure
                        or _settings_mutation_diagnostics(
                            component="settings_repository",
                            operation="save",
                            code="settings_save_failed",
                            exception=exc,
                        )
                    ) from exc
                self.settings.remember_projection(next_settings)
            elif self.settings.save_current() is False:
                self.settings.rollback()
                self._set_result(
                    TransactionResult(
                        status="settings_commit_failed",
                        message=None,
                        diagnostics=self.settings.last_failure,
                    )
                )
                return False
            committed = True
            runtime_result = await ProviderRuntimeApplyAdapter(
                owner=self.runtime,
                settings=next_settings,
                plan=plan,
            ).apply_runtime(
                RuntimeApplyRequest(
                    settings_values=self.settings.snapshot_values(next_settings),
                    reason="provider_direct",
                    correlation_id=None,
                )
            )
            if runtime_result.status == RUNTIME_APPLY_STATUS_APPLIED:
                self._set_result(
                    TransactionResult(
                        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                        message=runtime_result.message,
                        diagnostics=runtime_result.diagnostics,
                    )
                )
            else:
                self._set_result(_runtime_apply_result_as_degraded_transaction(runtime_result))
            self.remember_order22(self.settings.canonical)
            return True
        except ProviderStrictSettingsSaveFailed:
            raise
        except BaseException:
            if not committed:
                self.settings.rollback()
            raise
        finally:
            if committed:
                self.settings.complete()

    async def _resync_committed_provider_runtime(
        self,
        *,
        base_settings: ProviderCompatibilityProjection,
        committed_settings: ProviderCompatibilityProjection,
        plan: ProviderRuntimeApplyPlan,
        failure_result: TransactionResult,
    ) -> None:
        phase = "restore_memory"
        try:
            self.sync_memory(base_settings)
            phase = "resync_committed_runtime"
            await self.runtime.apply(copy.deepcopy(committed_settings), plan)
        except Exception as exc:
            secondary = _settings_mutation_diagnostics(
                component="provider_application",
                operation=phase,
                code="settings_runtime_resync_failed",
                exception=exc,
            )
            additional_diagnostics = _settings_mutation_secondary_diagnostics(
                primary=secondary,
                component="provider_application",
                operation=phase,
                code="settings_runtime_resync_failed",
                exception=exc,
            )
            restoration_failure = None
            preserve_retry = self.llm_retry_pending()
            try:
                self.sync_memory(committed_settings)
            except Exception as restore_exc:
                restoration_failure = replace(
                    _settings_mutation_diagnostics(
                        component="provider_application",
                        operation="restore_memory",
                        code="settings_memory_restore_failed",
                        exception=restore_exc,
                    ),
                    cause_id=secondary.diagnostic_id,
                )
            finally:
                if preserve_retry:
                    self.mark_llm_retry()
            failure_result = replace(
                failure_result,
                secondary_diagnostics=(
                    *failure_result.secondary_diagnostics,
                    replace(
                        secondary,
                        cause_id=(
                            failure_result.diagnostics.diagnostic_id
                            if failure_result.diagnostics is not None
                            else None
                        ),
                    ),
                    *additional_diagnostics,
                    *((restoration_failure,) if restoration_failure is not None else ()),
                ),
            )
        self._set_result(failure_result)

    def _set_result(self, result: TransactionResult) -> None:
        self.results.set(result)

    def _last_result_committed(self) -> bool:
        return self.results.committed()


def _settings_mutation_committed(result: TransactionResult) -> bool:
    return result.status in {
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    }


def provider_verification_context(
    settings: ProviderCompatibilityProjection | None,
    provider: str,
    *,
    low_latency: bool,
) -> dict[str, object]:
    if settings is None:
        return {}
    translation = settings.intent.translation
    if provider == "google":
        return {"model": translation.gemini.llm_model}
    if provider == "openai":
        return {"model": "gpt-6-luna"}
    if provider in {"alibaba_beijing", "alibaba_singapore"}:
        region = "beijing" if provider == "alibaba_beijing" else "singapore"
        connection = resolve_alibaba_connection(region, getattr(translation.qwen, region))
        return {
            "base_url": connection.native_url,
            "region": connection.region,
            "endpoint_mode": connection.endpoint_mode,
            "host": connection.host,
            "capability": "translation",
            "revision": connection.revision,
            "model": translation.qwen.llm_model,
            "low_latency": low_latency,
        }
    return {}


@dataclass(slots=True)
class ProviderSettingsOwner:
    settings: SettingsOwner
    binding: ProviderVerificationBindingOwner
    secret_store_factory: ProviderSecretStoreFactory
    active_secret_provider: ProviderActiveSecretProvider
    secret_change: ProviderSecretChangeOwner = field(default_factory=ProviderSecretChangeOwner)
    results: SettingsTransactionResultOwner = field(default_factory=SettingsTransactionResultOwner)

    def verification_binding(
        self,
        provider: str,
        key: str,
        *,
        flow: str,
        context_values: dict[str, object] | None = None,
    ) -> ProviderVerificationBinding:
        return self.binding.binding(
            provider,
            key,
            flow=flow,
            context_values=context_values,
        )

    def persist_verification(self, provider: str, key: str, success: bool) -> None:
        current = self._current()
        binding = (
            self.verification_binding(
                provider,
                key,
                flow="settings_api_key_verification",
            )
            if success
            else None
        )
        active_secret = (
            self.active_secret_provider(current, binding.secret_key)
            if binding is not None
            else None
        )
        self.settings.persist_provider_verification(
            provider=provider,
            key=key,
            success=success,
            binding=binding,
            active_secret=active_secret,
        )

    async def change_secret(self, secret_key: str, value: str) -> bool:
        if secret_key.startswith("http_extension."):
            return await self._change_http_extension_secret(secret_key, value)
        return await self.secret_change.change(
            lambda: self._secret_change_execution(secret_key, value)
        )

    async def _change_http_extension_secret(self, secret_key: str, value: str) -> bool:
        return await self.secret_change.change(
            lambda: self._http_extension_secret_change_execution(secret_key, value)
        )

    def _http_extension_secret_change_execution(
        self,
        secret_key: str,
        value: str,
    ) -> ProviderSecretChangeExecution:
        current = self._current()
        transaction = SecretSettingsTransaction(
            secret_store=self.secret_store_factory(current),
            settings_repository=_HttpExtensionSecretSettingsRepository(),
        )
        return ProviderSecretChangeExecution(
            transaction=transaction,
            request=ProviderSecretChangeRequest(
                provider="http_extension",
                secret_key=secret_key,
                secret_value=value,
                settings_values=self.settings.snapshot_values(current),
            ),
            result_handler=lambda result, _succeeded: self.results.set(result),
        )

    def _secret_change_execution(
        self,
        secret_key: str,
        value: str,
    ) -> ProviderSecretChangeExecution:
        current = self._current()
        provider = self.binding.provider_for_secret_key(secret_key)
        updated = replace(
            current,
            state=replace(
                current.state,
                provider_verification=replace(
                    current.state.provider_verification,
                    **{provider: ProviderVerificationEntry(status="unknown")},
                ),
            ),
        )
        if provider in {"alibaba_beijing", "alibaba_singapore"}:
            region = "beijing" if provider == "alibaba_beijing" else "singapore"
            translation = updated.intent.translation
            qwen = translation.qwen
            regional = getattr(qwen, region)
            updated = replace(
                updated,
                intent=replace(
                    updated.intent,
                    translation=replace(
                        translation,
                        qwen=replace(
                            qwen, **{region: replace(regional, revision=regional.revision + 1)}
                        ),
                    ),
                ),
            )
        repository = self.settings.create_canonical_patch_repository(
            base_settings=current,
            committed_settings=updated,
            surface="provider_secret_change",
        )
        transaction = SecretSettingsTransaction(
            secret_store=self.secret_store_factory(current),
            settings_repository=repository,
        )
        return ProviderSecretChangeExecution(
            transaction=transaction,
            request=ProviderSecretChangeRequest(
                provider=provider,
                secret_key=secret_key,
                secret_value=value,
                settings_values=self.settings.snapshot_values(updated),
            ),
            result_handler=lambda result, succeeded: self._apply_secret_change_result(
                repository.committed_settings,
                result,
                succeeded,
            ),
        )

    def _apply_secret_change_result(
        self,
        committed_settings: ProviderCompatibilityProjection,
        result: TransactionResult,
        succeeded: bool,
    ) -> None:
        self.results.set(result)
        if not succeeded:
            return
        self.settings.canonical = committed_settings
        self.settings.remember_projection(committed_settings)
        self.settings.complete()

    def _current(self) -> ProviderCompatibilityProjection:
        if self.settings.canonical is None:
            raise RuntimeError("settings owner has no canonical settings")
        return self.settings.canonical


__all__ = [
    "ProviderApplicationOwner",
    "ProviderActiveSecretProvider",
    "ProviderStrictSettingsSaveFailed",
    "ProviderSecretStoreFactory",
    "ProviderSettingsOwner",
    "provider_verification_context",
]
