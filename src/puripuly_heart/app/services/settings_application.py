from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.runtime_apply import RuntimeApplyPort
from puripuly_heart.app.ports.settings_runtime_effects import SettingsRuntimeEffectsPort
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.manual_local_asr_fallback import (
    ManualLocalASRFallbackOwner,
    ManualLocalASRFallbackPlan,
)
from puripuly_heart.app.services.provider_runtime_apply import (
    NoopRuntimeApply,
    OverlayOscOutputRuntimeApplyAdapter,
    SttLanguageAudioRuntimeApplyAdapter,
    UiPromptClipboardStateRuntimeApplyAdapter,
    _overlay_osc_output_runtime_degraded_transaction_result,
    _overlay_osc_output_save_failed_transaction_result,
    _runtime_apply_result_as_degraded_transaction,
    _stt_language_audio_runtime_degraded_transaction_result,
    _stt_language_audio_runtime_unavailable_result,
    _stt_language_audio_save_failed_transaction_result,
    _ui_prompt_clipboard_state_runtime_degraded_transaction_result,
    _ui_prompt_clipboard_state_save_failed_transaction_result,
)
from puripuly_heart.app.services.settings_mutation import (
    OverlayOscOutputSettingsMutation,
    SettingsMutationCommand,
    SettingsMutationService,
    SttLanguageAudioSettingsMutation,
    UiPromptClipboardStateSettingsMutation,
)
from puripuly_heart.app.services.settings_mutation_legacy import (
    _apply_settings_path_patch,
    build_overlay_osc_output_settings_path_patch,
    settings_path_mutation_validator_for_command,
)
from puripuly_heart.app.services.settings_projection import SettingsProjectionOwner
from puripuly_heart.app.services.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TransactionResult,
)


class StrictSettingsSaveFailed(Exception):
    pass


SettingsMutationServiceProvider = Callable[[], SettingsMutationService | None]
SettingsPredicate = Callable[[AppSettings, AppSettings], bool]
SettingsFailureSink = Callable[[str], None]
SettingsAsyncEffect = Callable[[], Awaitable[None]]
SettingsFallbackSink = Callable[[tuple[str, ...], bool], None]
SettingsFallbackLogSink = Callable[
    [AppSettings, AppSettings, tuple[str, ...]],
    None,
]


def _settings_mutation_committed(result: TransactionResult) -> bool:
    return result.status in {
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    }


def _copy_runtime_only_ui_state(source: AppSettings, target: AppSettings) -> None:
    target.ui.overlay_enabled = bool(source.ui.overlay_enabled)
    target.ui.peer_translation_enabled = bool(source.ui.peer_translation_enabled)


@dataclass(slots=True)
class SettingsApplicationOwner:
    settings: SettingsOwner
    projection: SettingsProjectionOwner
    runtime_effects: SettingsRuntimeEffectsPort[AppSettings]
    manual_fallback: ManualLocalASRFallbackOwner
    cpu_auto_available: Callable[[], bool]
    inspect_cpu: SettingsAsyncEffect
    fallback_sink: SettingsFallbackSink
    sync_ui: Callable[[], None]
    fallback_log_sink: SettingsFallbackLogSink
    mutation_service_provider: SettingsMutationServiceProvider
    consume_superseded_settings: Callable[[AppSettings], bool]
    active_local_asr_change: SettingsPredicate
    failure_sink: SettingsFailureSink
    results: SettingsTransactionResultOwner = field(default_factory=SettingsTransactionResultOwner)

    async def apply(self, next_settings: AppSettings) -> bool:
        current = self.settings.current
        if current is not None:
            self.settings.normalize_compatibility(current)
        self.settings.normalize_compatibility(next_settings)
        fallback_channels: tuple[str, ...] = ()
        installation_fallback = False
        normalization_channels = self.manual_fallback.normalization_channels(
            current=(
                self.manual_fallback.state(
                    current,
                    cpu_auto_available=self.cpu_auto_available(),
                )
                if current is not None
                else None
            ),
            pending=self.manual_fallback.state(
                next_settings,
                cpu_auto_available=self.cpu_auto_available(),
            ),
        )
        if normalization_channels:
            await self.inspect_cpu()
            fallback_plan = self.manual_fallback.plan(
                self.manual_fallback.state(
                    next_settings,
                    cpu_auto_available=self.cpu_auto_available(),
                )
            )
            fallback_channels = tuple(
                channel
                for channel in fallback_plan.fallback_channels
                if channel in normalization_channels
            )
            if fallback_channels:
                next_settings = self.manual_fallback.apply(
                    next_settings,
                    ManualLocalASRFallbackPlan(
                        self_provider=(
                            fallback_plan.self_provider
                            if "self" in fallback_channels
                            else next_settings.provider.stt.value
                        ),
                        peer_provider=(
                            fallback_plan.peer_provider
                            if "peer" in fallback_channels
                            else next_settings.provider.peer_stt.value
                        ),
                        fallback_channels=fallback_channels,
                        installation_fallback=bool(
                            fallback_plan.installation_fallback and fallback_channels
                        ),
                    ),
                )
            installation_fallback = bool(fallback_plan.installation_fallback and fallback_channels)
        if next_settings is not self.settings.current:
            if await self._route(next_settings):
                self.fallback_sink(fallback_channels, installation_fallback)
                return True
        await self.apply_direct(next_settings)
        self.fallback_sink(fallback_channels, installation_fallback)
        return True

    async def _route(self, next_settings: AppSettings) -> bool:
        if await self._apply_combined(next_settings):
            return True
        if await self._apply_stt_language_audio(next_settings):
            return True
        if await self._apply_overlay_osc_output(next_settings):
            return True
        return await self._apply_ui_prompt_clipboard_state(next_settings)

    def notify_fallback(
        self,
        channels: tuple[str, ...],
        installation_fallback: bool,
    ) -> None:
        self.fallback_sink(channels, installation_fallback)

    def persist_manual_fallback(self, *, channel: str | None = None) -> bool:
        previous = self.settings.current
        if previous is None:
            return False
        plan = self.manual_fallback.plan(
            self.manual_fallback.state(
                previous,
                cpu_auto_available=self.cpu_auto_available(),
            ),
            channel=channel,
        )
        if not plan.changed:
            return True
        normalized = self.manual_fallback.apply(previous, plan)
        self.settings.current = normalized
        if not self.settings.save_current(
            failure_sink=lambda exc: self.failure_sink(f"Failed to save settings: {exc}")
        ):
            self.settings.current = previous
            return False
        self.sync_ui()
        self.fallback_sink(
            plan.fallback_channels,
            plan.installation_fallback,
        )
        self.fallback_log_sink(
            previous,
            normalized,
            plan.fallback_channels,
        )
        return True

    async def apply_direct(
        self,
        next_settings: AppSettings,
        *,
        persist: bool = True,
        strict_runtime_errors: bool = False,
        strict_persistence_errors: bool = False,
        reload_settings_view: bool = True,
    ) -> None:
        await self.runtime_effects.preserve_before_replace(next_settings)
        if persist:
            baseline = self.settings.projection_snapshot or self.settings.current
            self.settings.begin(legacy_snapshot=baseline)
        committed = not persist
        try:
            if persist:
                self.runtime_effects.capture_runtime_signatures()
                self.settings.apply_legacy_delta(baseline, next_settings)
            transition = await self.runtime_effects.prepare(
                self.settings.current,
                next_settings,
            )
            self.settings.current = transition.settings
            self.runtime_effects.activate_before_persist(transition)
            if persist:
                if strict_persistence_errors:
                    try:
                        self.settings.persist()
                    except Exception:
                        raise StrictSettingsSaveFailed from None
                    self.settings.remember_projection(transition.settings)
                elif not self.settings.save_current(
                    failure_sink=lambda exc: self.failure_sink(f"Failed to save settings: {exc}")
                ):
                    return
                committed = True
            await self.runtime_effects.apply_after_persist(
                transition,
                strict_runtime_errors=strict_runtime_errors,
                reload_settings_view=reload_settings_view,
            )
            self.projection.remember_all(self.settings.current)
        finally:
            if persist:
                if committed:
                    self.settings.complete()
                else:
                    self.settings.rollback()

    async def _apply_runtime_effect(
        self,
        settings: object,
        reload_settings_view: bool,
    ) -> None:
        if not isinstance(settings, AppSettings):
            raise TypeError("settings runtime effect requires AppSettings")
        await self.apply_direct(
            settings,
            persist=False,
            strict_runtime_errors=True,
            reload_settings_view=reload_settings_view,
        )

    async def apply_ui_prompt_clipboard_state(
        self,
        next_settings: AppSettings,
    ) -> bool:
        return await self._apply_ui_prompt_clipboard_state(next_settings)

    async def apply_overlay_osc_output(
        self,
        next_settings: AppSettings,
    ) -> bool:
        return await self._apply_overlay_osc_output(next_settings)

    async def apply_language_selection(
        self,
        change: LanguageSelectionChange,
    ) -> None:
        current = self.settings.current
        if current is None:
            return
        updated = copy.deepcopy(current)
        updated.languages.source_language = change.source_code
        updated.languages.target_language = change.target_code
        updated.languages.peer_source_mode = change.peer_source_mode
        updated.languages.peer_source_language = change.peer_source_code
        updated.languages.peer_target_language = change.peer_target_code
        updated.languages.recent_source_languages = list(change.recent_source_codes)
        updated.languages.recent_target_languages = list(change.recent_target_codes)
        await self.apply(updated)
        if self.settings.current is not None:
            self.projection.render(
                self.settings.current,
                preserve_custom_vocab_draft=True,
            )

    async def compensate_failed_local_asr_settings_apply(
        self,
        *,
        base_settings: AppSettings,
        committed_settings: AppSettings,
    ) -> None:
        self.settings.begin(legacy_snapshot=committed_settings)
        committed = False
        try:
            self.settings.apply_legacy_delta(
                committed_settings,
                base_settings,
            )
            await asyncio.to_thread(self.settings.persist)
            self.settings.remember_projection(base_settings)
            committed = True
        finally:
            if committed:
                self.settings.complete()
            else:
                self.settings.rollback()
        await self.apply_direct(
            copy.deepcopy(base_settings),
            persist=False,
            strict_runtime_errors=False,
        )
        self.settings.current = copy.deepcopy(base_settings)
        self.projection.render(
            self.settings.current,
            preserve_custom_vocab_draft=True,
        )

    async def _apply_combined(self, next_settings: AppSettings) -> bool:
        order22 = self.projection.order22_patch_base_and_values(next_settings)
        order23 = self.projection.order23_patch_base_and_values(next_settings)
        order24 = self.projection.order24_patch_base_and_values(next_settings)
        if order22 is None or order23 is None or order24 is None:
            return False
        patches = (order22[1], order23[1], order24[1])
        if sum(bool(values) for values in patches) < 2:
            return False
        committed_results: list[TransactionResult] = []

        async def route(
            values: dict[str, object],
            apply_patch: Callable[[AppSettings], Awaitable[bool]],
            *,
            runtime_source: AppSettings | None = None,
        ) -> bool:
            current = self.settings.current
            if current is None:
                return False
            patch_settings = copy.deepcopy(current)
            _apply_settings_path_patch(patch_settings, values)
            if runtime_source is not None:
                _copy_runtime_only_ui_state(runtime_source, patch_settings)
            if not await apply_patch(patch_settings):
                return False
            if self.results.current is not None and _settings_mutation_committed(
                self.results.current
            ):
                committed_results.append(self.results.current)
            return True

        if patches[0]:
            if not await route(
                patches[0],
                lambda settings: self._apply_stt_language_audio(
                    settings,
                    reload_settings_view=False,
                ),
            ):
                return False
            if not self._last_result_committed():
                return True
        if patches[1]:
            if not await route(patches[1], self._apply_overlay_osc_output):
                return False
            if not self._last_result_committed():
                return True
        if patches[2]:
            if not await route(
                patches[2],
                self._apply_ui_prompt_clipboard_state,
                runtime_source=next_settings,
            ):
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
                await self.apply_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                )
            except StrictSettingsSaveFailed:
                if committed_before_full_draft is not None:
                    await self._resync_committed_runtime(
                        base_settings=committed_before_full_draft,
                        committed_settings=committed_before_full_draft,
                        failure_message="Failed to resync committed order24 settings runtime",
                    )
                self._set_result(
                    _ui_prompt_clipboard_state_save_failed_transaction_result(
                        operation="apply_order22_order23_order24_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(_ui_prompt_clipboard_state_runtime_degraded_transaction_result())

        if self.settings.current is not None:
            self.projection.render(
                self.settings.current,
                preserve_custom_vocab_draft=True,
            )
        if (
            self.results.current is not None
            and self.results.current.status
            == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
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

    async def _apply_stt_language_audio(
        self,
        next_settings: AppSettings,
        *,
        reload_settings_view: bool = True,
    ) -> bool:
        base_and_patch = self.projection.order22_patch_base_and_values(next_settings)
        if base_and_patch is None:
            return False
        base_settings, patch_values = base_and_patch
        if not patch_values:
            return False
        committed_settings = copy.deepcopy(base_settings)
        _apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self.settings.legacy_snapshot_values(
            committed_settings
        ) != self.settings.legacy_snapshot_values(next_settings)
        runtime_apply = (
            NoopRuntimeApply()
            if has_out_of_scope_draft
            else SttLanguageAudioRuntimeApplyAdapter(
                apply_settings=self._apply_runtime_effect,
                state_provider=self.runtime_effects.state,
                settings=committed_settings,
                reload_settings_view=reload_settings_view,
            )
        )
        result = await self._mutate(
            command=SttLanguageAudioSettingsMutation(values=patch_values),
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface="stt_language_audio",
            runtime_apply=runtime_apply,
        )
        if not _settings_mutation_committed(result):
            self.settings.current = copy.deepcopy(base_settings)
            self.projection.remember_order22(self.settings.current)
            return True
        if self.consume_superseded_settings(committed_settings):
            self.projection.remember_order22(self.settings.current)
            return True
        if (
            not has_out_of_scope_draft
            and result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
            and self.active_local_asr_change(base_settings, committed_settings)
        ):
            try:
                await self.compensate_failed_local_asr_settings_apply(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                )
            except Exception:
                self.failure_sink("Failed to compensate local ASR settings apply")
            self.projection.remember_order22(self.settings.current)
            return True
        if has_out_of_scope_draft:
            try:
                await self.apply_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                    reload_settings_view=reload_settings_view,
                )
            except StrictSettingsSaveFailed:
                await self._resync_committed_runtime(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                    failure_message="Failed to resync committed order22 settings runtime",
                )
                self._set_result(
                    _stt_language_audio_save_failed_transaction_result(
                        operation="apply_stt_language_audio_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(_stt_language_audio_runtime_degraded_transaction_result())
            else:
                unavailable = _stt_language_audio_runtime_unavailable_result(
                    state=self.runtime_effects.state(next_settings),
                    settings=next_settings,
                )
                if unavailable is not None:
                    self._set_result(_runtime_apply_result_as_degraded_transaction(unavailable))
        else:
            self.settings.current = committed_settings
            if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                self.runtime_effects.sync_signatures(committed_settings)
        self.projection.remember_order22(self.settings.current)
        return True

    async def _apply_overlay_osc_output(self, next_settings: AppSettings) -> bool:
        base_and_patch = self.projection.order23_patch_base_and_values(next_settings)
        if base_and_patch is None:
            return False
        base_settings, patch_values = base_and_patch
        if not patch_values:
            return False
        next_settings = copy.deepcopy(next_settings)
        await self.runtime_effects.prepare_overlay_persistence(
            base_settings,
            next_settings,
        )
        patch_values = build_overlay_osc_output_settings_path_patch(
            base_settings,
            next_settings,
        )
        committed_settings = copy.deepcopy(base_settings)
        _apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self.settings.legacy_snapshot_values(
            committed_settings
        ) != self.settings.legacy_snapshot_values(next_settings)
        runtime_apply = (
            NoopRuntimeApply()
            if has_out_of_scope_draft
            else OverlayOscOutputRuntimeApplyAdapter(
                apply_settings=self._apply_runtime_effect,
                settings=committed_settings,
            )
        )
        result = await self._mutate(
            command=OverlayOscOutputSettingsMutation(values=patch_values),
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface="overlay_osc_output",
            runtime_apply=runtime_apply,
        )
        if not _settings_mutation_committed(result):
            self.settings.current = copy.deepcopy(base_settings)
            self.projection.remember_order23(self.settings.current)
            return True
        if has_out_of_scope_draft:
            try:
                await self.apply_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                )
            except StrictSettingsSaveFailed:
                await self._resync_committed_runtime(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                    failure_message="Failed to resync committed order23 settings runtime",
                )
                self._set_result(
                    _overlay_osc_output_save_failed_transaction_result(
                        operation="apply_overlay_osc_output_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(_overlay_osc_output_runtime_degraded_transaction_result())
        elif self.settings.current is None or self.settings.current is base_settings:
            self.settings.current = committed_settings
        self.projection.remember_order23(self.settings.current)
        return True

    async def _apply_ui_prompt_clipboard_state(
        self,
        next_settings: AppSettings,
    ) -> bool:
        base_and_patch = self.projection.order24_patch_base_and_values(next_settings)
        if base_and_patch is None:
            return False
        base_settings, patch_values = base_and_patch
        if not patch_values:
            return False
        committed_settings = copy.deepcopy(base_settings)
        _apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self.settings.legacy_snapshot_values(
            committed_settings
        ) != self.settings.legacy_snapshot_values(next_settings)
        runtime_settings = copy.deepcopy(next_settings)
        runtime_apply = (
            NoopRuntimeApply()
            if has_out_of_scope_draft
            else UiPromptClipboardStateRuntimeApplyAdapter(
                apply_settings=self._apply_runtime_effect,
                settings=runtime_settings,
            )
        )
        result = await self._mutate(
            command=UiPromptClipboardStateSettingsMutation(values=patch_values),
            base_settings=self.settings.current or committed_settings,
            committed_settings=committed_settings,
            surface="ui_prompt_clipboard_state",
            runtime_apply=runtime_apply,
        )
        if not _settings_mutation_committed(result):
            self.settings.current = copy.deepcopy(base_settings)
            self.projection.remember_order24(self.settings.current)
            return True
        if has_out_of_scope_draft:
            try:
                await self.apply_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                )
            except StrictSettingsSaveFailed:
                await self._resync_committed_runtime(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                    failure_message="Failed to resync committed order24 settings runtime",
                )
                self._set_result(
                    _ui_prompt_clipboard_state_save_failed_transaction_result(
                        operation="apply_ui_prompt_clipboard_state_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(_ui_prompt_clipboard_state_runtime_degraded_transaction_result())
        else:
            self.settings.current = runtime_settings
            if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                self.runtime_effects.sync_signatures(runtime_settings)
        self.projection.remember_order24(self.settings.current)
        return True

    async def _mutate(
        self,
        *,
        command: SettingsMutationCommand,
        base_settings: AppSettings,
        committed_settings: AppSettings,
        surface: str,
        runtime_apply: RuntimeApplyPort,
    ) -> TransactionResult:
        repository = self.settings.create_legacy_patch_repository(
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface=surface,
            save_failure_sink=self.failure_sink,
        )
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
            raise RuntimeError("settings mutation completed without a result")
        self._set_result(result)
        return result

    async def _resync_committed_runtime(
        self,
        *,
        base_settings: AppSettings,
        committed_settings: AppSettings,
        failure_message: str,
    ) -> None:
        self.runtime_effects.restore_memory(base_settings)
        try:
            await self.apply_direct(
                copy.deepcopy(committed_settings),
                persist=False,
                strict_runtime_errors=True,
            )
        except Exception:
            self.failure_sink(failure_message)
            self.runtime_effects.restore_memory(committed_settings)

    def _set_result(self, result: TransactionResult) -> None:
        self.results.set(result)

    def _last_result_committed(self) -> bool:
        return self.results.committed()


__all__ = [
    "SettingsApplicationOwner",
    "StrictSettingsSaveFailed",
]
