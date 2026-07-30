from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from puripuly_heart.app.ports.canonical_settings_persistence import (
    ProviderVerificationBinding,
)
from puripuly_heart.app.ports.runtime_apply import RuntimeApplyPort, RuntimeApplyRequest
from puripuly_heart.app.ports.secret_store import SecretStorePort
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.secret_settings_transaction import (
    SecretSettingsTransaction,
)
from puripuly_heart.app.services.settings_mutation import (
    SettingsMutationService,
    SttLanguageAudioSettingsMutation,
    TranslationProviderSettingsMutation,
)
from puripuly_heart.app.services.settings_mutation_legacy import (
    _apply_settings_path_patch,
    build_stt_language_audio_settings_path_patch,
    build_translation_provider_settings_path_patch,
    settings_path_mutation_validator_for_command,
)
from puripuly_heart.app.services.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.core.messages import (
    RUNTIME_APPLY_STATUS_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
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

ProviderSecretStoreFactory = Callable[[AppSettings], SecretStorePort]
ProviderActiveSecretProvider = Callable[[AppSettings, str], str | None]
ProviderSettingsSaveFailureSink = Callable[[str], None]
ProviderSettingsMerge = Callable[[AppSettings], AppSettings]
ProviderSettingsAsyncEffect = Callable[[AppSettings], Awaitable[None]]
ProviderSettingsRoute = Callable[[AppSettings], Awaitable[bool]]
ProviderSettingsSync = Callable[[AppSettings], None]
ProviderSettingsPredicate = Callable[[AppSettings, AppSettings], bool]
ProviderSettingsCompensation = Callable[..., Awaitable[None]]
ProviderSettingsMutationServiceProvider = Callable[[], SettingsMutationService | None]
ProviderOrder24PatchProvider = Callable[
    [AppSettings],
    tuple[AppSettings, dict[str, object]] | None,
]
ProviderSupersededSettingsConsumer = Callable[[AppSettings], bool]


class ProviderStrictSettingsSaveFailed(Exception):
    pass


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
    save_failure_sink: ProviderSettingsSaveFailureSink
    results: SettingsTransactionResultOwner
    sync_memory: ProviderSettingsSync
    capture_runtime_signatures: Callable[[], None]
    sync_signatures: ProviderSettingsSync
    consume_superseded_settings: ProviderSupersededSettingsConsumer
    active_local_asr_change: ProviderSettingsPredicate
    compensate_local_asr: ProviderSettingsCompensation
    llm_retry_pending: Callable[[], bool]
    mark_llm_retry: Callable[[], None]

    async def apply(
        self,
        pending: AppSettings | None = None,
        *,
        force_rebuild_llm: bool = False,
    ) -> bool:
        next_settings = self.settings.current if pending is None else self.merge_settings(pending)
        if next_settings is None:
            return False
        await self.preserve_before_replace(next_settings)
        try:
            if pending is not None and not force_rebuild_llm:
                if await self._apply_combined(next_settings):
                    return self._last_result_committed()
                if await self._apply_translation(next_settings):
                    return self._last_result_committed()
                if await self.apply_order24(next_settings):
                    return self._last_result_committed()
            return await self._apply_direct(
                next_settings,
                force_rebuild_llm=force_rebuild_llm,
            )
        finally:
            self.sync_ui()

    async def _apply_combined(self, next_settings: AppSettings) -> bool:
        base_settings = self.settings.current
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
            current = self.settings.current
            if current is None:
                return False
            patch_settings = copy.deepcopy(current)
            _apply_settings_path_patch(patch_settings, values)
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
            current = self.settings.current
            if current is None:
                return True
            order24_settings = copy.deepcopy(current)
            _apply_settings_path_patch(order24_settings, order24_values)
            order24_settings.ui.overlay_enabled = bool(next_settings.ui.overlay_enabled)
            order24_settings.ui.peer_translation_enabled = bool(
                next_settings.ui.peer_translation_enabled
            )
            if not await self.apply_order24(order24_settings):
                return False
            if not self._last_result_committed():
                return True

        committed_before_full_draft = (
            copy.deepcopy(self.settings.current) if self.settings.current is not None else None
        )
        if self.settings.current is not None and self.settings.legacy_snapshot_values(
            self.settings.current
        ) != self.settings.legacy_snapshot_values(next_settings):
            try:
                await self._apply_direct(
                    next_settings,
                    force_rebuild_llm=False,
                    route_order22=False,
                    strict_persistence_errors=True,
                )
            except ProviderStrictSettingsSaveFailed:
                if committed_before_full_draft is not None:
                    self.sync_memory(committed_before_full_draft)
                self._set_result(
                    _translation_provider_save_failed_transaction_result(
                        operation="apply_order21_order22_order24_provider_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(
                    _runtime_apply_result_as_degraded_transaction(
                        _runtime_apply_failed_result(
                            operation="apply_order21_order22_order24_provider_runtime",
                            code="provider_runtime_apply_exception",
                            surface="translation_provider",
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

    async def _apply_translation(self, next_settings: AppSettings) -> bool:
        base_settings = self.settings.current
        if base_settings is None:
            return False
        patch_values = build_translation_provider_settings_path_patch(
            base_settings,
            next_settings,
        )
        if not patch_values:
            return False
        committed_settings = copy.deepcopy(base_settings)
        _apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self.settings.legacy_snapshot_values(
            committed_settings
        ) != self.settings.legacy_snapshot_values(next_settings)
        plan = self.runtime.build_plan(
            committed_settings,
            force_rebuild_llm=False,
            canonical_settings=self.settings.project_legacy_delta(
                base_settings,
                committed_settings,
            ),
        )
        repository = self.settings.create_legacy_patch_repository(
            base_settings=base_settings,
            committed_settings=committed_settings,
            save_failure_sink=self.save_failure_sink,
        )
        runtime_apply = ProviderRuntimeApplyAdapter(
            owner=self.runtime,
            settings=committed_settings,
            plan=plan,
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
        self.settings.current = committed_settings
        if has_out_of_scope_draft:
            fallback_plan = self.runtime.build_plan(
                next_settings,
                force_rebuild_llm=False,
                canonical_settings=self.settings.project_legacy_delta(
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
            except ProviderStrictSettingsSaveFailed:
                preserve_retry = self.llm_retry_pending()
                self.sync_memory(committed_settings)
                if preserve_retry:
                    self.mark_llm_retry()
                self._set_result(
                    _translation_provider_save_failed_transaction_result(
                        operation="apply_translation_provider_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(
                    _runtime_apply_result_as_degraded_transaction(
                        _runtime_apply_failed_result(
                            operation="apply_translation_provider_runtime",
                            code="provider_runtime_apply_exception",
                            surface="translation_provider",
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
            self.sync_signatures(committed_settings)
        self.remember_order22(self.settings.current)
        return True

    async def _apply_stt_language_audio(self, next_settings: AppSettings) -> bool:
        base_settings = self.settings.current
        if base_settings is None:
            return False
        patch_values = build_stt_language_audio_settings_path_patch(
            base_settings,
            next_settings,
        )
        if not patch_values:
            return False
        committed_settings = copy.deepcopy(base_settings)
        _apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self.settings.legacy_snapshot_values(
            committed_settings
        ) != self.settings.legacy_snapshot_values(next_settings)
        plan = self.runtime.build_plan(
            committed_settings,
            force_rebuild_llm=False,
            canonical_settings=self.settings.project_legacy_delta(
                base_settings,
                committed_settings,
            ),
        )
        repository = self.settings.create_legacy_patch_repository(
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface="stt_language_audio",
            save_failure_sink=self.save_failure_sink,
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
            self.settings.current = copy.deepcopy(base_settings)
            self.remember_order22(self.settings.current)
            return True
        if self.consume_superseded_settings(committed_settings):
            self.remember_order22(self.settings.current)
            return True
        if (
            not has_out_of_scope_draft
            and result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
            and self.active_local_asr_change(base_settings, committed_settings)
        ):
            try:
                await self.compensate_local_asr(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                )
            except Exception:
                self.save_failure_sink("Failed to compensate local ASR provider settings apply")
            self.remember_order22(self.settings.current)
            return True
        if has_out_of_scope_draft:
            fallback_plan = self.runtime.build_plan(
                next_settings,
                force_rebuild_llm=False,
                canonical_settings=self.settings.project_legacy_delta(
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
            except ProviderStrictSettingsSaveFailed:
                await self._resync_committed_provider_runtime(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                    plan=plan,
                )
                self._set_result(
                    _stt_language_audio_save_failed_transaction_result(
                        operation="apply_stt_language_audio_provider_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(
                    _runtime_apply_result_as_degraded_transaction(
                        _runtime_apply_failed_result(
                            operation="apply_stt_language_audio_provider_runtime",
                            code="provider_runtime_apply_exception",
                            surface="stt_language_audio",
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
            self.settings.current = committed_settings
            if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                self.sync_signatures(committed_settings)
        self.remember_order22(self.settings.current)
        return True

    async def _apply_direct(
        self,
        next_settings: AppSettings,
        *,
        force_rebuild_llm: bool,
        plan: ProviderRuntimeApplyPlan | None = None,
        route_order22: bool = True,
        strict_persistence_errors: bool = False,
    ) -> bool:
        if route_order22 and not force_rebuild_llm and plan is None:
            if await self._apply_stt_language_audio(next_settings):
                return self._last_result_committed()
        self.settings.begin(
            legacy_snapshot=self.settings.projection_snapshot or self.settings.current
        )
        committed = False
        try:
            self.capture_runtime_signatures()
            self.settings.apply_legacy_delta(
                self.settings.projection_snapshot or self.settings.current,
                next_settings,
            )
            if plan is None:
                plan = self.runtime.build_plan(
                    next_settings,
                    force_rebuild_llm=force_rebuild_llm,
                )
            self.settings.current = next_settings
            if strict_persistence_errors:
                try:
                    self.settings.persist()
                except Exception:
                    self.settings.rollback()
                    raise ProviderStrictSettingsSaveFailed from None
                self.settings.remember_projection(next_settings)
            elif (
                self.settings.save_current(
                    failure_sink=lambda exc: self.save_failure_sink(
                        f"Failed to save settings: {exc}"
                    )
                )
                is False
            ):
                self.settings.rollback()
                return False
            committed = True
            runtime_result = await ProviderRuntimeApplyAdapter(
                owner=self.runtime,
                settings=next_settings,
                plan=plan,
            ).apply_runtime(
                RuntimeApplyRequest(
                    settings_values=self.settings.legacy_snapshot_values(next_settings),
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
            self.remember_order22(self.settings.current)
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
        base_settings: AppSettings,
        committed_settings: AppSettings,
        plan: ProviderRuntimeApplyPlan,
    ) -> None:
        self.sync_memory(base_settings)
        try:
            await self.runtime.apply(copy.deepcopy(committed_settings), plan)
        except Exception:
            self.save_failure_sink("Failed to resync committed order22 provider runtime")
            self.sync_memory(committed_settings)

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
    settings: AppSettings | None,
    provider: str,
    *,
    low_latency: bool,
) -> dict[str, object]:
    if settings is None:
        return {}
    if provider == "google":
        return {"model": settings.gemini.llm_model.value}
    if provider == "cerebras":
        return {"model": settings.cerebras.llm_model.value}
    if provider in {"alibaba_beijing", "alibaba_singapore"}:
        return {
            "base_url": (
                "https://dashscope.aliyuncs.com/api/v1"
                if provider == "alibaba_beijing"
                else "https://dashscope-intl.aliyuncs.com/api/v1"
            ),
            "model": settings.qwen.llm_model.value,
            "low_latency": low_latency,
        }
    return {}


@dataclass(slots=True)
class ProviderSettingsOwner:
    settings: SettingsOwner
    binding: ProviderVerificationBindingOwner
    secret_store_factory: ProviderSecretStoreFactory
    active_secret_provider: ProviderActiveSecretProvider
    save_failure_sink: ProviderSettingsSaveFailureSink | None = None
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
        return await self.secret_change.change(
            lambda: self._secret_change_execution(secret_key, value)
        )

    def _secret_change_execution(
        self,
        secret_key: str,
        value: str,
    ) -> ProviderSecretChangeExecution:
        current = self._current()
        provider = self.binding.provider_for_secret_key(secret_key)
        updated = copy.deepcopy(current)
        setattr(updated.api_key_verified, provider, False)
        repository = self.settings.create_legacy_patch_repository(
            base_settings=current,
            committed_settings=updated,
            surface="provider_secret_change",
            save_failure_sink=self.save_failure_sink,
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
                settings_values=self.settings.legacy_snapshot_values(updated),
            ),
            result_handler=lambda result, succeeded: self._apply_secret_change_result(
                repository.committed_settings,
                result,
                succeeded,
            ),
        )

    def _apply_secret_change_result(
        self,
        committed_settings: AppSettings,
        result: TransactionResult,
        succeeded: bool,
    ) -> None:
        self.results.set(result)
        if not succeeded:
            return
        self.settings.current = committed_settings
        self.settings.remember_projection(committed_settings)
        self.settings.complete()

    def _current(self) -> AppSettings:
        if self.settings.current is None:
            raise RuntimeError("settings owner has no compatibility settings")
        return self.settings.current


__all__ = [
    "ProviderApplicationOwner",
    "ProviderActiveSecretProvider",
    "ProviderStrictSettingsSaveFailed",
    "ProviderSecretStoreFactory",
    "ProviderSettingsSaveFailureSink",
    "ProviderSettingsOwner",
    "provider_verification_context",
]
