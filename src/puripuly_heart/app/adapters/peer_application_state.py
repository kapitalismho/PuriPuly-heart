from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.app.services.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.services.peer_application import (
    PeerApplicationOwner,
    PeerApplicationState,
)
from puripuly_heart.core.orchestrator.hub import ClientHub


@dataclass(frozen=True, slots=True)
class PeerApplicationSettings:
    intent_enabled: bool
    eula_accepted: bool
    overlay_intent_enabled: bool
    provider_id: str


SettingsProvider = Callable[[], PeerApplicationSettings | None]
HubProvider = Callable[[], ClientHub | None]
OverlayOwnerProvider = Callable[[], OverlayApplicationOwner]
IngressFrozenProvider = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class PeerApplicationStateAdapter:
    settings_provider: SettingsProvider
    hub_provider: HubProvider
    overlay_owner_provider: OverlayOwnerProvider
    ingress_frozen_provider: IngressFrozenProvider

    def state(
        self,
        settings: PeerApplicationSettings | None = None,
    ) -> PeerApplicationState:
        resolved_settings = settings or self.settings_provider()
        hub = self.hub_provider()
        activation_requested = bool(
            resolved_settings is not None
            and PeerApplicationOwner.activation_requested(
                intent_enabled=resolved_settings.intent_enabled,
                eula_accepted=resolved_settings.eula_accepted,
            )
        )
        overlay = self.overlay_owner_provider()
        return PeerApplicationState(
            settings_available=resolved_settings is not None,
            peer_intent_enabled=bool(resolved_settings and resolved_settings.intent_enabled),
            eula_accepted=bool(resolved_settings and resolved_settings.eula_accepted),
            overlay_intent_enabled=bool(
                resolved_settings and resolved_settings.overlay_intent_enabled
            ),
            peer_provider_id=(
                resolved_settings.provider_id if resolved_settings is not None else None
            ),
            runtime_available=hub is not None,
            peer_provider_available=bool(
                activation_requested and hub is not None and hub.has_stt_provider("peer")
            ),
            overlay_state=overlay.snapshot.state,
            overlay_command_available=overlay.current_bridge() is not None,
            ingress_frozen=self.ingress_frozen_provider(),
        )


__all__ = [
    "PeerApplicationSettings",
    "PeerApplicationStateAdapter",
]
