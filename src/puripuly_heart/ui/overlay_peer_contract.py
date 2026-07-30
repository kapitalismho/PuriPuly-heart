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
    status_text: str
    helper_text: str = ""
    warning_reason: str | None = None
    failure_reason: str | None = None


@dataclass(slots=True, frozen=True)
class OverlayPeerConsumerContract:
    overlay: OverlayPeerToggleContract
    peer: OverlayPeerToggleContract


def build_overlay_peer_consumer_contract(
    *,
    overlay_intent_enabled: bool,
    overlay_state: str,
    overlay_failure_reason: str | None,
    peer_intent_enabled: bool,
    peer_effective_enabled: bool,
    peer_warning_reason: str | None = None,
    peer_activation_starting: bool = False,
) -> OverlayPeerConsumerContract:
    overlay_contract = OverlayPeerToggleContract(
        intent_enabled=overlay_intent_enabled,
        effective_enabled=overlay_state == "connected",
        action_enabled=True,
        state=_overlay_surface_state(overlay_intent_enabled, overlay_state),
        status_text=_overlay_status_text(overlay_state, overlay_failure_reason),
        warning_reason=_overlay_warning_reason(overlay_intent_enabled, overlay_state),
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
        status_text=t(f"settings.peer_translation.status.{peer_state}"),
        helper_text=_peer_helper_text(
            peer_state=peer_state,
            overlay_state=overlay_state,
            overlay_failure_reason=overlay_failure_reason,
            peer_warning_reason=resolved_peer_warning_reason,
        ),
        warning_reason=resolved_peer_warning_reason,
        failure_reason=(
            overlay_failure_reason if resolved_peer_warning_reason == "overlay_failed" else None
        ),
    )
    return OverlayPeerConsumerContract(overlay=overlay_contract, peer=peer_contract)


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
    )


def _overlay_surface_state(
    overlay_intent_enabled: bool,
    overlay_state: str,
) -> OverlayPeerSurfaceState:
    if not overlay_intent_enabled:
        return "off"
    if overlay_state in {"starting", "connected"}:
        return "on"
    return "warning"


def _overlay_warning_reason(
    overlay_intent_enabled: bool,
    overlay_state: str,
) -> str | None:
    if not overlay_intent_enabled or overlay_state in {"starting", "connected"}:
        return None
    if overlay_state == "failed":
        return "overlay_failed"
    if overlay_state == "stopping":
        return "overlay_stopping"
    return "overlay_required"


def _overlay_status_text(
    overlay_state: str,
    overlay_failure_reason: str | None,
) -> str:
    state_label = t(f"settings.overlay.status.{overlay_state}", default=overlay_state)
    if overlay_state == "failed" and overlay_failure_reason:
        return t(
            "settings.overlay.status.failed_with_reason",
            status=state_label,
            reason=_overlay_failure_text(overlay_failure_reason),
            default=f"{state_label}: {_overlay_failure_text(overlay_failure_reason)}",
        )
    return state_label


def _overlay_failure_text(overlay_failure_reason: str | None) -> str:
    return t(
        f"settings.overlay.failure.{overlay_failure_reason or 'unknown'}",
        default=overlay_failure_reason or "unknown",
    )


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
    if overlay_state == "starting":
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
    overlay_state: str,
    overlay_failure_reason: str | None,
    peer_warning_reason: str | None,
) -> str:
    if peer_state == "off":
        if overlay_state == "connected":
            return ""
        return t("settings.peer_translation.disabled.overlay_required")
    if peer_state in {"starting", "on"}:
        return ""
    if peer_warning_reason == "overlay_starting":
        return t("settings.peer_translation.warning.overlay_starting")
    if peer_warning_reason == "overlay_stopping":
        return t("settings.peer_translation.warning.overlay_stopping")
    if peer_warning_reason == "overlay_failed":
        return t(
            "settings.peer_translation.warning.overlay_failed",
            reason=_overlay_failure_text(overlay_failure_reason),
        )
    if peer_warning_reason == "runtime_unavailable":
        return t("settings.peer_translation.warning.runtime_unavailable")
    if peer_warning_reason == "process_unavailable_no_process":
        return t("settings.peer_translation.warning.process_unavailable_no_process")
    if peer_warning_reason == "process_unavailable_ambiguous":
        return t("settings.peer_translation.warning.process_unavailable_ambiguous")
    if peer_warning_reason == "process_unavailable_ineligible":
        return t("settings.peer_translation.warning.process_unavailable_ineligible")
    if peer_warning_reason == "process_unavailable_unsupported_platform":
        return t("settings.peer_translation.warning.process_unavailable_unsupported_platform")
    if peer_warning_reason == "process_setup_failed":
        return t("settings.peer_translation.warning.process_setup_failed")
    if peer_warning_reason == "process_target_exited":
        return t("settings.peer_translation.warning.process_target_exited")
    if peer_warning_reason == "process_source_failed":
        return t("settings.peer_translation.warning.process_source_failed")
    if peer_warning_reason == "process_provider_failed":
        return t("settings.peer_translation.warning.process_provider_failed")
    if peer_warning_reason is not None and peer_warning_reason.startswith("process_"):
        return t("settings.peer_translation.warning.process_capture_failed")
    return t("settings.peer_translation.disabled.overlay_required")


def is_process_capture_warning_reason(reason: str | None) -> bool:
    if reason is None:
        return False
    return reason.startswith("process_")
