from __future__ import annotations

import logging
from pathlib import Path

from puripuly_heart.app.ports.ui_application import UiApplicationPort
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresencePort
from puripuly_heart.app.services.canonical_settings_persistence import (
    compose_settings_owner,
)
from puripuly_heart.app.services.provider_runtime_apply import ProviderRuntimeApplyPlan
from puripuly_heart.app.services.provider_settings import (
    ProviderApplicationOwner,
    ProviderSettingsOwner,
    provider_verification_context,
)
from puripuly_heart.app.services.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)
from puripuly_heart.app.services.ui_application import UiApplicationBoundary
from puripuly_heart.app.services.ui_application_state import UiApplicationStateOwner
from puripuly_heart.app.wiring import (
    create_secret_store,
    create_self_capture_admission_adapter,
    create_sync_secret_store_adapter,
)
from puripuly_heart.app.wiring_application_runtime_logging import (
    compose_application_runtime_logging,
)
from puripuly_heart.app.wiring_capture_runtime import CaptureOwnerFactory
from puripuly_heart.app.wiring_managed_account import (
    ManagedAccountComponents,
    ManagedOpenRouterReleaseRuntime,
    ManagedTranslationRuntimeAccess,
    compose_managed_account,
)
from puripuly_heart.app.wiring_provider_runtime import compose_provider_runtime
from puripuly_heart.app.wiring_runtime_composition import RuntimeCompositionComponents
from puripuly_heart.app.wiring_runtime_pipeline import RuntimePipelineLauncher
from puripuly_heart.composition.controller_application_startup import (
    compose_controller_application_startup,
)
from puripuly_heart.composition.controller_ui_application_state import (
    ControllerUiApplicationStateAdapter,
)
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY
from puripuly_heart.ui.controller import GuiController


def compose_gui_runtime_components(
    backend: GuiController,
) -> RuntimeCompositionComponents:
    capture_factory = CaptureOwnerFactory(
        settings_provider=lambda: backend.settings,
        self_admission=create_self_capture_admission_adapter(
            state_provider=(
                backend._get_local_asr_application_runtime().adapters.state.self_admission
            ),
            validate_gpu_activation=backend._validate_gpu_activation,
            effect_sink=(
                backend._get_local_asr_application_runtime().adapters.effects.apply_self_admission
            ),
        ),
        ensure_peer_local_ready=lambda generation: (
            backend._get_local_asr_application_runtime().ensure_peer_ready(
                activation_generation=generation,
            )
        ),
        clock=backend.clock,
        log_detailed=backend.log_detailed,
        detailed_enabled=backend._detailed_audio_diag_enabled,
        source_wrapper=lambda source, channel: (
            backend._get_capture_diagnostics_adapter().wrap_source(
                source,
                channel_label=channel,
            )
        ),
        self_state_sink=backend._on_self_capture_state_changed,
        self_diagnostic_sink=backend._get_capture_diagnostics_adapter().self_capture,
        peer_state_sink=backend._on_peer_capture_state_changed,
        peer_diagnostic_sink=(backend._get_peer_application_runtime().owner.on_runtime_diagnostic),
        local_asr_diagnostic_sink=(
            backend._get_local_asr_diagnostics_owner().transition_diagnostic
        ),
    )

    def self_capture_owner() -> SelfCaptureSessionOwner:
        owner = backend._self_capture_owner
        if owner is None:
            owner = capture_factory.compose_self(
                backend.hub,
                backend.vrc_mic_audio_gate,
            )
            backend._self_capture_owner = owner
        return owner

    managed_account: ManagedAccountComponents | None = None

    def managed_release() -> ManagedOpenRouterReleaseRuntime:
        if managed_account is None:
            raise RuntimeError("managed-account composition is incomplete")
        return managed_account.release

    async def recover_gpu(
        settings: object,
        plan: ProviderRuntimeApplyPlan,
    ) -> None:
        await backend._get_gpu_provider_recovery_owner().recover(
            lambda: backend._gpu_provider_recovery_request(
                settings,
                reason="settings_restart",
                plan=plan,
            )
        )

    provider_runtime = compose_provider_runtime(
        config_path=backend.config_path,
        settings=backend._get_settings_owner(),
        hub_provider=lambda: backend.hub,
        self_capture_provider=lambda: backend._self_capture_owner,
        self_capture_owner=self_capture_owner,
        peer=lambda: backend._get_peer_application_runtime().owner,
        peer_desired=backend._peer_runtime_should_be_active,
        canonical_settings=backend._canonical_vnext_settings_for,
        clear_local_pending=lambda: (
            backend._get_local_asr_application_runtime().cpu_repair.clear_if_provider_switched_away()
        ),
        sync_local_notice=lambda: (
            backend._get_local_asr_application_runtime().adapters.notice.sync()
        ),
        managed_pending_sink=backend._set_managed_trial_pending_auth,
        managed_pending_provider=lambda: backend.managed_auth_pending,
        dashboard_managed_pending_sink=(backend.app.set_dashboard_managed_auth_pending),
        sync_effective_flags=backend._sync_effective_hub_flags,
        refresh_overlay=backend._refresh_overlay_peer_consumers,
        refresh_peer_runtime=lambda: (
            backend._get_peer_application_runtime().owner.refresh_runtime()
        ),
        replace_self_stt=lambda smooth: (
            backend._get_self_capture_application_owner().replace_provider(smooth_local=smooth)
        ),
        self_state_sink=backend._on_self_capture_state_changed,
        self_availability=(backend._get_self_capture_application_owner().project_availability),
        gpu_recovery=recover_gpu,
        managed_release=managed_release,
        managed_delegate_ready=backend._on_managed_trial_delegate_ready,
        runtime_logging=backend.runtime_logging,
        translation_needs_key_sink=(backend.app.set_dashboard_translation_needs_key),
        usage_refresh=backend._refresh_managed_trial_usage_state_best_effort,
        failure_sink=backend._log_error,
        success_sink=backend.log_basic,
        additional_signature_sink=backend._sync_non_provider_runtime_signatures,
        signatures=backend._provider_runtime_signatures,
    )
    managed_account = compose_managed_account(
        config_path=backend.config_path,
        settings=backend._get_settings_owner(),
        provider_settings=backend._get_provider_settings_owner(),
        provider_runtime=provider_runtime.runtime,
        verifier=backend._get_provider_verifier(),
        results=backend._get_settings_application_owner().results,
        runtime=ManagedTranslationRuntimeAccess(
            runtime_provider=lambda: backend.hub,
            rebuild_llm=provider_runtime.llm_rebuild.rebuild,
        ),
        ingress_provider=lambda: backend._shutdown_ingress_frozen,
        pending_sink=backend.app.set_dashboard_managed_auth_pending,
        usage_view_sink=backend._apply_managed_usage_view_state,
        dashboard_sink=backend.app.set_dashboard_translation_enabled,
        message_sink=lambda key, values: backend._show_short_message(
            key,
            **dict(values),
        ),
        qq_dialog_sink=lambda: (
            backend.app.show_qq_managed_auth_dialog()
            if callable(getattr(backend.app, "show_qq_managed_auth_dialog", None))
            else None
        ),
        founder_dialog=backend.app.show_founder_letter_dialog,
        failure_route=backend._maybe_show_founder_letter_after_pkce_failure,
        log_basic=backend.log_basic,
        log_detailed=backend.log_detailed,
        log_error=backend._log_error,
        basic_warning_sink=lambda message: backend.log_basic(
            message,
            level=logging.WARNING,
        ),
        detailed_warning_sink=lambda message, exception: backend.log_detailed(
            message,
            level=logging.WARNING,
            exception=exception,
        ),
    )
    provider_application = ProviderApplicationOwner(
        settings=backend._get_settings_owner(),
        runtime=provider_runtime.runtime,
        merge_settings=backend.merge_settings_tab_apply_with_current_languages,
        preserve_before_replace=(
            backend._preserve_github_star_prompt_observation_before_settings_replace
        ),
        sync_ui=backend._sync_ui_from_settings,
        order24_patch_provider=(backend._settings_projection().order24_patch_base_and_values),
        apply_order24=(backend._get_settings_application_owner().apply_ui_prompt_clipboard_state),
        remember_order22=backend._settings_projection().remember_order22,
        mutation_service_provider=lambda: backend.settings_mutation_service,
        save_failure_sink=backend._log_error,
        results=backend._get_settings_application_owner().results,
        sync_memory=(backend._get_settings_application_owner().runtime_effects.restore_memory),
        capture_runtime_signatures=(backend._capture_runtime_signatures_before_canonical_mutation),
        sync_signatures=backend._sync_signature_caches,
        consume_superseded_settings=backend._consume_superseded_local_asr_settings,
        active_local_asr_change=backend._active_local_asr_change,
        compensate_local_asr=(
            backend._get_settings_application_owner().compensate_failed_local_asr_settings_apply
        ),
        llm_retry_pending=lambda: (backend._provider_runtime_signatures.last_llm_provider == ()),
        mark_llm_retry=backend._provider_runtime_signatures.mark_llm_retry,
    )
    pipeline_launcher = RuntimePipelineLauncher(
        config_path=backend.config_path,
        clock=backend.clock,
        runtime_logging=backend.runtime_logging,
        managed_release=managed_account.release,
        managed_delegate_ready=backend._on_managed_trial_delegate_ready,
        local_asr_factory=lambda secrets: (
            backend._build_local_asr_provider_runtime_factory(secrets=secrets)
        ),
        self_capture_factory=capture_factory.compose_self,
        peer_capture_factory=capture_factory.compose_peer,
        previous_self_capture=lambda: backend._self_capture_owner,
        component_sink=backend._apply_runtime_pipeline_components,
        peer_application=lambda: backend._get_peer_application_runtime().owner,
        configure_vrc_mic=backend._configure_vrc_mic_receiver,
        stt_failure_sink=backend._log_error,
        cleanup_failure_sink=lambda message, exc: backend._log_error(f"{message}: {exc}"),
    )
    return RuntimeCompositionComponents(
        self_capture_owner=self_capture_owner,
        provider_runtime=provider_runtime,
        managed_account=managed_account,
        provider_application=provider_application,
        pipeline_launcher=pipeline_launcher,
    )


def compose_gui_controller(
    *,
    presentation: UiPresentationPort,
    config_path: Path,
    allow_stable_settings_import: bool = False,
    runtime_logging_sinks: RuntimeLoggingSinks | None = None,
    vrchat_osc_presence: VrchatOscPresencePort | None = None,
) -> GuiController:
    settings_owner = compose_settings_owner(config_path)
    provider_settings_owner = ProviderSettingsOwner(
        settings=settings_owner,
        binding=ProviderVerificationBindingOwner(
            context_provider=lambda provider: provider_verification_context(
                settings_owner.current,
                provider,
                low_latency=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
            ),
        ),
        secret_store_factory=lambda settings: create_sync_secret_store_adapter(
            create_secret_store(settings.secrets, config_path=config_path)
        ),
        active_secret_provider=lambda settings, secret_key: create_secret_store(
            settings.secrets,
            config_path=config_path,
        ).get(secret_key),
    )
    backend = GuiController(
        page=None,
        app=presentation,
        config_path=config_path,
        allow_stable_settings_import=allow_stable_settings_import,
        runtime_logging_sinks=runtime_logging_sinks,
        vrchat_osc_presence=vrchat_osc_presence,
        settings_owner=settings_owner,
        provider_settings_owner=provider_settings_owner,
    )
    backend.install_runtime_logging_owner(
        compose_application_runtime_logging(
            presentation=presentation,
            sinks=runtime_logging_sinks,
            overlay_logging_mode_update=backend._emit_overlay_runtime_logging_mode_update,
            overlay_logging_mode_update_available=lambda: (
                backend._get_overlay_application_owner().current_bridge() is not None
            ),
        )
    )
    backend.install_runtime_composition(compose_gui_runtime_components(backend))
    backend.install_startup_owner(compose_controller_application_startup(backend))
    provider_settings_owner.save_failure_sink = backend._log_error
    return backend


def compose_ui_application(
    *,
    presentation: UiPresentationPort,
    config_path: Path,
    allow_stable_settings_import: bool = False,
    runtime_logging_sinks: RuntimeLoggingSinks | None = None,
    vrchat_osc_presence: VrchatOscPresencePort | None = None,
) -> UiApplicationPort:
    backend = compose_gui_controller(
        presentation=presentation,
        config_path=config_path,
        allow_stable_settings_import=allow_stable_settings_import,
        runtime_logging_sinks=runtime_logging_sinks,
        vrchat_osc_presence=vrchat_osc_presence,
    )
    return compose_gui_application_boundary(backend)


def compose_gui_application_boundary(
    backend: GuiController,
) -> UiApplicationBoundary:
    runtime_logging = backend.runtime_logging_owner
    return UiApplicationBoundary(
        backend,
        state=UiApplicationStateOwner(
            ControllerUiApplicationStateAdapter(backend),
            runtime_logging=runtime_logging,
        ),
        runtime_shutdown=backend,
        runtime_logging=runtime_logging,
    )
