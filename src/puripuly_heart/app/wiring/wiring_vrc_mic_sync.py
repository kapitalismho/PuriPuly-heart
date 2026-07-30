from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from puripuly_heart.app.services.vrc_mic_sync import (
    VrcMicAudioGatePort,
    VrcMicSyncOwner,
)
from puripuly_heart.core.osc.receiver import (
    VRC_OSC_RECEIVER_HOST,
    VRC_OSC_RECEIVER_PORT,
    VrcOscReceiver,
)


def compose_vrc_mic_sync(
    *,
    state_provider: Callable[[], object | None],
    gate_provider: Callable[[], VrcMicAudioGatePort | None],
    log_detailed: Callable[[str, int], None],
    error_sink: Callable[[str], None],
) -> VrcMicSyncOwner:
    def diagnostics_sink(event: str, metadata: Mapping[str, object]) -> None:
        log_detailed(
            f"[Lifecycle][VrcMicReceiverRuntime] event={event} metadata={dict(metadata)}",
            logging.WARNING,
        )

    return VrcMicSyncOwner(
        state_provider=state_provider,
        gate_provider=gate_provider,
        receiver_factory=lambda **kwargs: VrcOscReceiver(**kwargs),
        diagnostics_sink=diagnostics_sink,
        error_sink=error_sink,
        host=VRC_OSC_RECEIVER_HOST,
        port=VRC_OSC_RECEIVER_PORT,
    )


__all__ = ["compose_vrc_mic_sync"]
