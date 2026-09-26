from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from puripuly_heart.app.ports.ui_models import OverlayPeerPresentationState
from puripuly_heart.ui.i18n import t

OverlayPeerSurfaceState = Literal["off", "starting", "on", "warning"]


@dataclass(slots=True, frozen=True)
class OverlayPeerToggleContract:
    intent_enabled: bool
    effective_enabled: bool
    action_enabled: bool
    state: OverlayPeerSurfaceState
    helper_text: str = ""
    warning_reason: str | None = None
    failure_reason: str | None = None


@dataclass(slots=True, frozen=True)
class OverlayPeerConsumerContract:
    overlay: OverlayPeerToggleContract
    peer: OverlayPeerToggleContract
    desktop_first_visible: bool = False


def build_overlay_peer_consumer_contract(
    *,
    overlay_intent_enabled: bool,
    overlay_state: str,
    overlay_failure_reason: str | None,
    peer_intent_enabled: bool,
    peer_effective_enabled: bool,
    peer_warning_reason: str | None = None,
    peer_activation_starting: bool = False,
    desktop_first_visible: bool = False,
    overlay_activation_pending: bool = False,
) -> OverlayPeerConsumerContract:
    overlay_contract = OverlayPeerToggleContract(
        intent_enabled=overlay_intent_enabled,
        effective_enabled=overlay_state == "connected",
        action_enabled=True,
        state=(
            "starting"
            if overlay_intent_enabled and overlay_activation_pending
            else _overlay_surface_state(overlay_intent_enabled, overlay_state)
        ),
        warning_reason=(
            None
            if overlay_activation_pending
            else _overlay_warning_reason(overlay_intent_enabled, overlay_state)
        ),
        failure_reason=overlay_failure_reason,
    )
    resolved_peer_warning_reason = _resolve_peer_warning_reason(
        peer_intent_enabled=peer_intent_enabled,
        peer_effective_enabled=peer_effective_enabled,
        overlay_state=overlay_state,
        peer_warning_reason=peer_warning_reason,
    )
    peer_state = _peer_surface_state(
        peer_intent_enabled,
        peer_effective_enabled,
        peer_activation_starting,
        resolved_peer_warning_reason,
    )
    peer_contract = OverlayPeerToggleContract(
        intent_enabled=peer_intent_enabled,
        effective_enabled=peer_effective_enabled,
        action_enabled=overlay_state == "connected" or peer_intent_enabled,
        state=peer_state,
        helper_text=_peer_helper_text(
            peer_state=peer_state,
            peer_warning_reason=resolved_peer_warning_reason,
        ),
        warning_reason=resolved_peer_warning_reason,
        failure_reason=(
            overlay_failure_reason if resolved_peer_warning_reason == "overlay_failed" else None
        ),
    )
    return OverlayPeerConsumerContract(
        overlay=overlay_contract,
        peer=peer_contract,
        desktop_first_visible=bool(desktop_first_visible),
    )


def build_overlay_peer_consumer_contract_from_state(
    state: OverlayPeerPresentationState,
) -> OverlayPeerConsumerContract:
    return build_overlay_peer_consumer_contract(
        overlay_intent_enabled=state.overlay_intent_enabled,
        overlay_state=state.overlay_state,
        overlay_failure_reason=state.overlay_failure_reason,
        peer_intent_enabled=state.peer_intent_enabled,
        peer_effective_enabled=state.peer_effective_enabled,
        peer_warning_reason=state.peer_warning_reason,
        peer_activation_starting=state.peer_activation_starting,
        desktop_first_visible=bool(getattr(state, "desktop_first_visible", False)),
        overlay_activation_pending=state.overlay_activation_pending,
    )


def _overlay_surface_state(
    overlay_intent_enabled: bool,
    overlay_state: str,
) -> OverlayPeerSurfaceState:
    if not overlay_intent_enabled:
        return "off"
    if overlay_state in {"starting", "recovering", "connected"}:
        return "on"
    return "warning"


def _overlay_warning_reason(
    overlay_intent_enabled: bool,
    overlay_state: str,
) -> str | None:
    if not overlay_intent_enabled or overlay_state in {"starting", "recovering", "connected"}:
        return None
    if overlay_state == "failed":
        return "overlay_failed"
    if overlay_state == "stopping":
        return "overlay_stopping"
    return "overlay_required"


def _peer_surface_state(
    peer_intent_enabled: bool,
    peer_effective_enabled: bool,
    peer_activation_starting: bool,
    peer_warning_reason: str | None,
) -> OverlayPeerSurfaceState:
    if not peer_intent_enabled:
        return "off"
    if peer_activation_starting or peer_warning_reason == "overlay_starting":
        return "starting"
    if peer_effective_enabled:
        return "on"
    return "warning"


def _resolve_peer_warning_reason(
    *,
    peer_intent_enabled: bool,
    peer_effective_enabled: bool,
    overlay_state: str,
    peer_warning_reason: str | None,
) -> str | None:
    if not peer_intent_enabled or peer_effective_enabled:
        return None
    if peer_warning_reason is not None:
        return peer_warning_reason
    if overlay_state in {"starting", "recovering"}:
        return "overlay_starting"
    if overlay_state == "stopping":
        return "overlay_stopping"
    if overlay_state == "failed":
        return "overlay_failed"
    if overlay_state != "connected":
        return "overlay_required"
    return "runtime_unavailable"


def _peer_helper_text(
    *,
    peer_state: OverlayPeerSurfaceState,
    peer_warning_reason: str | None,
) -> str:
    if peer_state != "warning":
        return ""
    if peer_warning_reason == "process_unavailable_no_process":
        return t("settings.peer_translation.warning.process_unavailable_no_process")
    if peer_warning_reason == "process_unavailable_ambiguous":
        return t("settings.peer_translation.warning.process_unavailable_ambiguous")
    if peer_warning_reason == "process_unavailable_ineligible":
        return t("settings.peer_translation.warning.process_unavailable_ineligible")
    if peer_warning_reason == "process_unavailable_unsupported_platform":
        return t("settings.peer_translation.warning.process_unavailable_unsupported_platform")
    return ""


def is_process_capture_warning_reason(reason: str | None) -> bool:
    if reason is None:
        return False
    return reason.startswith("process_")
