from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.app.adapters.peer_application_state import (
    PeerApplicationSettings,
    PeerApplicationStateAdapter,
)
from puripuly_heart.app.adapters.peer_capture_inventory import (
    PeerCaptureTargetRuntimeEffectsAdapter,
    WindowsLoopbackDeviceInventoryAdapter,
    WindowsProcessCaptureInventoryAdapter,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.services.peer_application import (
    PeerApplicationOwner,
    PeerApplicationState,
)
from puripuly_heart.app.services.peer_capture_target_application import (
    PeerCaptureTargetApplicationOwner,
)
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.core.local_asr_provider_runtime import LocalASRProviderRuntimePort
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfigurationPort,
)

from .wiring_stt_factory import build_peer_capture_session_config
from .wiring_translation_runtime_configuration import (
    replace_translation_runtime_effective_flags,
)


@dataclass(frozen=True, slots=True)
class PeerApplicationRuntime:
    state: PeerApplicationStateAdapter
    state_for: Callable[[AppSettings | None], PeerApplicationState]
    owner: PeerApplicationOwner
    target: PeerCaptureTargetApplicationOwner


def compose_peer_application(
    *,
    settings_provider: Callable[[], AppSettings | None],
    settings_owner: SettingsOwner,
    canonical_settings: Callable[[AppSettings], object],
    runtime_provider: Callable[[], LocalASRProviderRuntimePort | None],
    translation_runtime_configuration_provider: Callable[
        [],
        TranslationRuntimeConfigurationPort | None,
    ],
    overlay_provider: Callable[[], OverlayApplicationOwner],
    ingress_frozen: Callable[[], bool],
    persist_manual_fallback: Callable[[], bool],
    ensure_local_ready: Callable[[int], Awaitable[bool]],
    clear_cpu_pending: Callable[[], None],
    clear_gpu_pending: Callable[[], None],
    clear_switched_pending: Callable[[], None],
    sync_local_notice: Callable[[], None],
    presentation_changed: Callable[[], None],
    disclosure_sink: Callable[[], None],
    superseded_sink: Callable[[], None],
    localize: Callable[[str], str],
    settings_presentation_sink: Callable[[AppSettings], None],
    log_basic: Callable[..., object],
    log_detailed: Callable[..., object],
    translation_demand_sink: Callable[[], Awaitable[None]] | None = None,
) -> PeerApplicationRuntime:
    def application_settings(
        settings: AppSettings | None,
    ) -> PeerApplicationSettings | None:
        if settings is None:
            return None
        return PeerApplicationSettings(
            intent_enabled=settings.ui.peer_translation_enabled,
            eula_accepted=settings.ui.peer_translation_eula_accepted,
            overlay_intent_enabled=settings.ui.overlay_enabled,
            provider_id=settings.provider.peer_stt.value,
        )

    state = PeerApplicationStateAdapter(
        settings_provider=lambda: application_settings(settings_provider()),
        runtime_provider=runtime_provider,
        overlay_owner_provider=overlay_provider,
        ingress_frozen_provider=ingress_frozen,
    )

    def current_settings() -> AppSettings:
        settings = settings_provider()
        if settings is None:
            raise RuntimeError("Peer application requires settings")
        return settings

    def effective_sink(
        peer_translation_enabled: bool,
        integrated_context_enabled: bool,
    ) -> None:
        owner = translation_runtime_configuration_provider()
        if owner is None:
            return
        replace_translation_runtime_effective_flags(
            owner,
            peer_translation_enabled=peer_translation_enabled,
            integrated_context_enabled=integrated_context_enabled,
        )

    owner = PeerApplicationOwner(
        state_provider=state.state,
        config_factory=lambda: build_peer_capture_session_config(
            current_settings(),
            canonical_settings=canonical_settings(current_settings()),
        ),
        peer_intent_sink=lambda enabled: setattr(
            current_settings().ui,
            "peer_translation_enabled",
            enabled,
        ),
        overlay_intent_sink=lambda enabled: setattr(
            current_settings().ui,
            "overlay_enabled",
            enabled,
        ),
        persist_manual_fallback=persist_manual_fallback,
        ensure_local_ready=ensure_local_ready,
        clear_cpu_pending=clear_cpu_pending,
        clear_gpu_pending=clear_gpu_pending,
        clear_switched_pending=clear_switched_pending,
        sync_local_notice=sync_local_notice,
        presentation_changed=presentation_changed,
        begin_overlay_start=lambda: overlay_provider().begin_start(),
        effective_sink=effective_sink,
        disclosure_sink=disclosure_sink,
        superseded_sink=superseded_sink,
        log_basic=log_basic,
        log_detailed=log_detailed,
        log_failure=lambda message: log_basic(message, level=logging.ERROR),
        lifecycle_trace_sink=lambda event, fields: overlay_provider().record_lifecycle_trace(
            event,
            **fields,
        ),
        translation_demand_sink=translation_demand_sink,
    )
    target = PeerCaptureTargetApplicationOwner(
        settings=settings_owner,
        localize=localize,
        processes=WindowsProcessCaptureInventoryAdapter(),
        devices=WindowsLoopbackDeviceInventoryAdapter(),
        runtime_effects=PeerCaptureTargetRuntimeEffectsAdapter(
            refresh_peer=owner.refresh_runtime,
            sync_effective_flags=lambda settings: owner.sync_effective_flags(
                state.state(application_settings(settings))
            ),
            refresh_presentation=presentation_changed,
        ),
        settings_presentation_sink=settings_presentation_sink,
        warning_reset=lambda: setattr(owner, "process_warning_reason", None),
    )
    return PeerApplicationRuntime(
        state=state,
        state_for=lambda settings: state.state(application_settings(settings)),
        owner=owner,
        target=target,
    )


__all__ = [
    "PeerApplicationRuntime",
    "compose_peer_application",
]
