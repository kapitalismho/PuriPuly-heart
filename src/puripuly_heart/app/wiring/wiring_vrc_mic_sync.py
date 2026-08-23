from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping

from puripuly_heart.app.ports.oscquery import OscQueryServicePort
from puripuly_heart.app.ports.ui_models import (
    OscControlPresentationName,
    OscControlPresentationState,
)
from puripuly_heart.app.services.osc.control_runtime import OscControlIntegrationOwner
from puripuly_heart.app.services.osc.state_publisher import OscCanonicalState
from puripuly_heart.app.services.vrc_mic_sync import (
    VrcMicAudioGatePort,
    VrcMicSyncOwner,
)
from puripuly_heart.core.osc.oscquery import ZeroconfOscQueryService
from puripuly_heart.core.osc.receiver import VrcOscReceiver
from puripuly_heart.core.osc.receiver_contract import (
    VRC_OSC_RECEIVER_HOST,
    VRC_OSC_RECEIVER_PORT,
    VrcMicState,
)


def compose_vrc_mic_sync(
    *,
    state_provider: Callable[[], VrcMicState | None],
    gate_provider: Callable[[], VrcMicAudioGatePort | None],
    log_detailed: Callable[[str, int], None],
    error_sink: Callable[[str], None],
    settings_provider: Callable[[], object | None],
    apply_settings: Callable[[object], Awaitable[object]],
    application_provider: Callable[[], object | None],
    sender_provider: Callable[[], object | None],
    osc_state_provider: Callable[[], OscCanonicalState],
    language_state_provider: Callable[[], tuple[str, str, str, str]],
    translation_model_normalizer: Callable[[object], object],
    ui_state_provider: (
        Callable[[OscControlPresentationName], OscControlPresentationState] | None
    ) = None,
    ui_state_sink: Callable[[OscControlPresentationState], None] | None = None,
    query_service: OscQueryServicePort | None = None,
) -> OscControlIntegrationOwner:
    def diagnostics_sink(event: str, metadata: Mapping[str, object]) -> None:
        log_detailed(
            f"[Lifecycle][VrcMicReceiverRuntime] event={event} metadata={dict(metadata)}",
            logging.WARNING,
        )

    receiver_owner = VrcMicSyncOwner(
        state_provider=state_provider,
        gate_provider=gate_provider,
        receiver_factory=VrcOscReceiver,
        diagnostics_sink=diagnostics_sink,
        error_sink=error_sink,
        host=VRC_OSC_RECEIVER_HOST,
        port=VRC_OSC_RECEIVER_PORT,
    )
    return OscControlIntegrationOwner(
        receiver_owner=receiver_owner,
        settings_provider=settings_provider,
        apply_settings=apply_settings,
        application_provider=application_provider,
        sender_provider=sender_provider,
        state_provider=osc_state_provider,
        ui_state_provider=ui_state_provider,
        ui_state_sink=ui_state_sink,
        language_state_provider=language_state_provider,
        translation_model_normalizer=translation_model_normalizer,
        query_service=query_service or ZeroconfOscQueryService(),
        error_sink=error_sink,
    )


__all__ = ["compose_vrc_mic_sync"]
