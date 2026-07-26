from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.components.power_button import PowerButton
from puripuly_heart.ui.dashboard.renderer import (
    DASHBOARD_POWER_BUTTON_ICON_SIZE,
    DASHBOARD_POWER_BUTTON_LABEL_SIZE,
)
from puripuly_heart.ui.i18n import t


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

    def apply_locale(self) -> None:
        self._self_button.set_label(t("dashboard.stt_label"))
        self._peer_button.set_label(t("dashboard.peer_label"))
        self._overlay_button.set_label(t("dashboard.overlay_label"))


__all__ = ["DashboardCaptureControls"]
