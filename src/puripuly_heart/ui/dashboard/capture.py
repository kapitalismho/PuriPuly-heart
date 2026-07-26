from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import flet as ft

from puripuly_heart.ui.components.power_button import PowerButton
from puripuly_heart.ui.dashboard.renderer import (
    DASHBOARD_POWER_BUTTON_ICON_SIZE,
    DASHBOARD_POWER_BUTTON_LABEL_SIZE,
)
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.overlay_peer_contract import (
    OverlayPeerConsumerContract,
    is_process_capture_warning_reason,
)


@dataclass(frozen=True, slots=True)
class CaptureChannelPresentation:
    enabled: bool
    starting: bool = False
    warning: bool = False


@dataclass(frozen=True, slots=True)
class DashboardCapturePresentation:
    peer: CaptureChannelPresentation
    overlay: CaptureChannelPresentation
    process_capture_warning_active: bool = False
    process_capture_warning_reason: str | None = None
    process_capture_warning_text: str = ""


CAPTURE_PRESENTATION_IDLE = DashboardCapturePresentation(
    peer=CaptureChannelPresentation(enabled=False),
    overlay=CaptureChannelPresentation(enabled=False),
)


def capture_presentation_from_contract(
    contract: OverlayPeerConsumerContract | None,
) -> DashboardCapturePresentation:
    if contract is None:
        return CAPTURE_PRESENTATION_IDLE
    peer = contract.peer
    overlay = contract.overlay
    process_warning = (
        peer.state == "warning"
        and is_process_capture_warning_reason(peer.warning_reason)
        and bool(peer.helper_text)
    )
    return DashboardCapturePresentation(
        peer=CaptureChannelPresentation(
            enabled=peer.state == "on",
            starting=peer.state == "starting",
            warning=peer.state == "warning",
        ),
        overlay=CaptureChannelPresentation(
            enabled=overlay.state == "on",
            warning=overlay.state == "warning",
        ),
        process_capture_warning_active=process_warning,
        process_capture_warning_reason=peer.warning_reason if process_warning else None,
        process_capture_warning_text=peer.helper_text if process_warning else "",
    )


class DashboardCaptureControls:
    def __init__(
        self,
        *,
        on_self_capture_click: Callable[[], None],
        on_peer_capture_click: Callable[[], None],
        on_overlay_click: Callable[[], None],
    ) -> None:
        self._self_button = PowerButton(
            label=t("dashboard.stt_label"),
            icon=ft.Icons.MIC,
            on_click=on_self_capture_click,
            icon_size=DASHBOARD_POWER_BUTTON_ICON_SIZE,
            label_size=DASHBOARD_POWER_BUTTON_LABEL_SIZE,
        )
        self._peer_button = PowerButton(
            label=t("dashboard.peer_label"),
            icon=ft.Icons.RECORD_VOICE_OVER,
            on_click=on_peer_capture_click,
            icon_size=DASHBOARD_POWER_BUTTON_ICON_SIZE,
            label_size=DASHBOARD_POWER_BUTTON_LABEL_SIZE,
        )
        self._overlay_button = PowerButton(
            label=t("dashboard.overlay_label"),
            icon=ft.Icons.SUBTITLES,
            on_click=on_overlay_click,
            icon_size=DASHBOARD_POWER_BUTTON_ICON_SIZE,
            label_size=DASHBOARD_POWER_BUTTON_LABEL_SIZE,
        )

    def self_capture_control(self) -> ft.Control:
        return self._self_button

    def peer_capture_control(self) -> ft.Control:
        return self._peer_button

    def overlay_control(self) -> ft.Control:
        return self._overlay_button

    def apply_self_capture_state(
        self,
        *,
        enabled: bool,
        starting: bool,
        warning: bool,
    ) -> None:
        self._self_button.set_state(enabled, needs_key=warning, is_starting=starting)

    def apply_peer_capture_state(
        self,
        *,
        enabled: bool,
        starting: bool = False,
        warning: bool = False,
    ) -> None:
        self._peer_button.set_state(enabled, needs_key=warning, is_starting=starting)

    def apply_overlay_state(self, *, enabled: bool, warning: bool = False) -> None:
        self._overlay_button.set_state(enabled, needs_key=warning)

    def apply_presentation(self, presentation: DashboardCapturePresentation) -> None:
        self.apply_peer_capture_state(
            enabled=presentation.peer.enabled,
            starting=presentation.peer.starting,
            warning=presentation.peer.warning,
        )
        self.apply_overlay_state(
            enabled=presentation.overlay.enabled,
            warning=presentation.overlay.warning,
        )

    def apply_locale(self) -> None:
        self._self_button.set_label(t("dashboard.stt_label"))
        self._peer_button.set_label(t("dashboard.peer_label"))
        self._overlay_button.set_label(t("dashboard.overlay_label"))


__all__ = [
    "CAPTURE_PRESENTATION_IDLE",
    "CaptureChannelPresentation",
    "DashboardCaptureControls",
    "DashboardCapturePresentation",
    "capture_presentation_from_contract",
]
