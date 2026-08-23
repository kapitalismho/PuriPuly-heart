from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import flet as ft

from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_NEUTRAL_DARK,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SURFACE,
)

DEBUG_PREVIEW_PANEL_DATA_KEY = "debug-preview-panel"


def _make_text_button(label: str, **kwargs) -> ft.TextButton:
    return ft.TextButton(content=label, **kwargs)


def _set_text_button_label(button: ft.TextButton, label: str) -> None:
    button.content = label


@dataclass(frozen=True)
class _PreviewAction:
    key: str
    label_key: str
    callback: Callable[[], None]


class DebugPreviewPanel(ft.Container):
    def __init__(
        self,
        *,
        on_brake_notice: Callable[[], None],
        on_revoked_notice: Callable[[], None],
        on_founder_letter: Callable[[], None],
        on_pkce_failure: Callable[[], None],
        on_discord_auth: Callable[[], None],
        on_qq_auth: Callable[[], None],
        on_qq_auth_recoverable_error: Callable[[], None],
        on_qq_auth_translation_gated: Callable[[], None],
        on_discord_callback_page: Callable[[], None],
        on_peer_translation_eula: Callable[[], None],
        on_local_qwen_hallucination_modal: Callable[[], None],
        on_talk_together_pass_invite_progress: Callable[[], None],
        on_capture_fault_cycle: Callable[[], None],
        on_stt_fault_cycle: Callable[[], None],
        on_audio_fault_clear: Callable[[], None],
        on_gpu_state_cycle: Callable[[], None],
        on_github_star_snackbar: Callable[[], None],
        on_foundation_primitives: Callable[[], None],
        on_stt_loading_button_cycle: Callable[[], None] | None = None,
        on_http_extension_form: Callable[[], None] | None = None,
    ) -> None:
        actions = [
            _PreviewAction("brake_notice", "debug_preview.brake_notice", on_brake_notice),
            _PreviewAction("revoked_notice", "debug_preview.revoked_notice", on_revoked_notice),
            _PreviewAction(
                "github_star_snackbar",
                "debug_preview.github_star_snackbar",
                on_github_star_snackbar,
            ),
            _PreviewAction("founder_letter", "debug_preview.founder_letter", on_founder_letter),
            _PreviewAction("pkce_failure", "debug_preview.pkce_failure", on_pkce_failure),
            _PreviewAction("discord_auth", "debug_preview.discord_auth", on_discord_auth),
            _PreviewAction("qq_auth", "debug_preview.qq_auth", on_qq_auth),
            _PreviewAction(
                "qq_auth_recoverable_error",
                "debug_preview.qq_auth_recoverable_error",
                on_qq_auth_recoverable_error,
            ),
            _PreviewAction(
                "qq_auth_translation_gated",
                "debug_preview.qq_auth_translation_gated",
                on_qq_auth_translation_gated,
            ),
            _PreviewAction(
                "discord_callback_page",
                "debug_preview.discord_callback_page",
                on_discord_callback_page,
            ),
            _PreviewAction(
                "peer_translation_eula",
                "debug_preview.peer_translation_eula",
                on_peer_translation_eula,
            ),
            _PreviewAction(
                "local_qwen_hallucination_modal",
                "debug_preview.local_qwen_hallucination_modal",
                on_local_qwen_hallucination_modal,
            ),
            _PreviewAction(
                "talk_together_pass_invite_progress",
                "debug_preview.talk_together_pass_invite_progress",
                on_talk_together_pass_invite_progress,
            ),
            _PreviewAction(
                "capture_fault_cycle",
                "debug_preview.capture_fault_cycle",
                on_capture_fault_cycle,
            ),
            _PreviewAction(
                "stt_fault_cycle",
                "debug_preview.stt_fault_cycle",
                on_stt_fault_cycle,
            ),
            _PreviewAction(
                "audio_fault_clear",
                "debug_preview.audio_fault_clear",
                on_audio_fault_clear,
            ),
            _PreviewAction(
                "gpu_state_cycle",
                "debug_preview.gpu_state_cycle",
                on_gpu_state_cycle,
            ),
        ]
        if on_stt_loading_button_cycle is not None:
            actions.append(
                _PreviewAction(
                    "stt_loading_button_cycle",
                    "debug_preview.stt_loading_button_cycle",
                    on_stt_loading_button_cycle,
                )
            )
        actions.append(
            _PreviewAction(
                "foundation_primitives",
                "debug_preview.foundation_primitives",
                on_foundation_primitives,
            )
        )
        actions.append(
            _PreviewAction(
                "http_extension_form",
                "debug_preview.http_extension_form",
                on_http_extension_form,
            )
        )
        self._actions = tuple(actions)
        self._toggle_button = _make_text_button(
            t("debug_preview.button"),
            tooltip=t("debug_preview.tooltip"),
            on_click=self._toggle,
            style=self._toggle_style(),
        )
        self._action_buttons = {
            action.key: self._build_action_button(action) for action in self._actions
        }
        self._popover = ft.Container(
            visible=False,
            bgcolor=COLOR_SURFACE,
            border_radius=14,
            padding=ft.Padding.symmetric(horizontal=8, vertical=8),
            content=ft.Column(
                controls=list(self._action_buttons.values()),
                spacing=2,
                tight=True,
            ),
        )

        super().__init__(
            data=DEBUG_PREVIEW_PANEL_DATA_KEY,
            top=64,
            right=18,
            content=ft.Column(
                controls=[self._toggle_button, self._popover],
                spacing=6,
                horizontal_alignment=ft.CrossAxisAlignment.END,
                tight=True,
            ),
        )

    def _toggle_style(self) -> ft.ButtonStyle:
        return ft.ButtonStyle(
            color={
                ft.ControlState.DEFAULT: COLOR_NEUTRAL_DARK,
                ft.ControlState.HOVERED: COLOR_PRIMARY,
            },
            bgcolor=COLOR_SURFACE,
            text_style=ft.TextStyle(size=12, weight=ft.FontWeight.BOLD),
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            shape=ft.RoundedRectangleBorder(radius=12),
            overlay_color=ft.Colors.TRANSPARENT,
            animation_duration=0,
        )

    def _action_style(self) -> ft.ButtonStyle:
        return ft.ButtonStyle(
            color={
                ft.ControlState.DEFAULT: COLOR_ON_BACKGROUND,
                ft.ControlState.HOVERED: COLOR_PRIMARY,
            },
            bgcolor=ft.Colors.TRANSPARENT,
            text_style=ft.TextStyle(size=12, weight=ft.FontWeight.W_600),
            padding=ft.Padding.symmetric(horizontal=10, vertical=7),
            shape=ft.RoundedRectangleBorder(radius=10),
            overlay_color=ft.Colors.with_opacity(0.08, COLOR_PRIMARY),
            animation_duration=0,
        )

    def _build_action_button(self, action: _PreviewAction) -> ft.TextButton:
        return _make_text_button(
            t(action.label_key),
            on_click=lambda _e, callback=action.callback: self._invoke(callback),
            style=self._action_style(),
        )

    def _toggle(self, _event) -> None:
        self._popover.visible = not self._popover.visible
        self._update_if_mounted()

    def _invoke(self, callback: Callable[[], None]) -> None:
        callback()

    def apply_locale(self) -> None:
        _set_text_button_label(self._toggle_button, t("debug_preview.button"))
        self._toggle_button.tooltip = t("debug_preview.tooltip")
        for action in self._actions:
            _set_text_button_label(self._action_buttons[action.key], t(action.label_key))
        self._update_if_mounted()

    def _is_mounted(self) -> bool:
        try:
            return self.page is not None
        except RuntimeError as exc:
            if "Control must be added to the page first" not in str(exc):
                raise
            return False

    def _update_if_mounted(self) -> None:
        if not self._is_mounted():
            return
        try:
            self.update()
        except RuntimeError as exc:
            if "Control must be added to the page first" not in str(exc):
                raise
            return
