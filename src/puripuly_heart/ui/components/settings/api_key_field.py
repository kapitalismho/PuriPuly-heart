"""API key input field with auto-verification on blur.

The blur -> save -> verify coordination lives in the UI-independent
``ApiKeyVerificationController``; this widget binds it to Flet controls.
"""

from __future__ import annotations

from typing import Callable

import flet as ft
from flet import Colors as colors
from flet import Icons as icons

from puripuly_heart.ui.components.settings.api_key_verification_controller import (
    ApiKeyVerificationController,
)
from puripuly_heart.ui.flet_runtime import control_page, update_control_if_mounted
from puripuly_heart.ui.i18n import provider_label, t
from puripuly_heart.ui.theme import (
    COLOR_DIVIDER,
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_WARNING,
)

_STATUS_MESSAGES = {
    "success": ("snackbar.verification_ok", colors.GREEN_400),
    "error": ("snackbar.verification_failed", colors.RED_400),
}


_STATUS_KEYS = {
    "idle": "api_key.status.idle",
    "verifying": "api_key.status.verifying",
    "success": "api_key.status.success",
    "error": "api_key.status.error",
}

_STATUS_ICONS = {
    "idle": (icons.HELP_OUTLINE_ROUNDED, COLOR_SECONDARY),
    "verifying": (icons.HOURGLASS_TOP_ROUNDED, COLOR_SECONDARY),
    "success": (icons.CHECK_CIRCLE_ROUNDED, COLOR_PRIMARY),
    "error": (icons.WARNING_ROUNDED, COLOR_WARNING),
}


class ApiKeyField(ft.Row):
    """API key input field with auto-verification on blur and status indicator."""

    def __init__(
        self,
        label_key: str,
        secret_key: str,
        provider: str,
        on_verify: Callable[[str, str], object] | None = None,
        on_save: Callable[[str, str], object] | None = None,
        show_snackbar: Callable[[str, str], None] | None = None,
        show_status: bool = True,
    ):
        self._label_key = label_key
        self._secret_key = secret_key
        self._provider = provider
        self._show_snackbar_cb = show_snackbar
        self._show_status = show_status

        self._reveal_button = ft.IconButton(
            icon=icons.VISIBILITY_OFF_ROUNDED,
            icon_color=COLOR_DIVIDER,
            icon_size=24,
            on_click=self._toggle_password_visibility,
        )

        self._text_field = ft.TextField(
            label=t(label_key),
            password=True,
            can_reveal_password=False,
            on_blur=self._handle_blur,
            on_change=self._handle_change,
            border_radius=12,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            expand=True,
            text_size=28,
            color=COLOR_NEUTRAL_DARK,
            label_style=ft.TextStyle(size=20, weight=ft.FontWeight.BOLD, color=COLOR_NEUTRAL_DARK),
            suffix=self._reveal_button,
        )

        self._current_status = "idle"
        self._status_icon = ft.Icon(
            icon=icons.HELP_OUTLINE_ROUNDED,
            color=COLOR_SECONDARY,
            size=36,
            tooltip=t("api_key.status.idle"),
        )

        controls: list[ft.Control] = [self._text_field]
        if self._show_status:
            controls.append(self._status_icon)

        super().__init__(
            controls=controls,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._controller = ApiKeyVerificationController(
            secret_key=secret_key,
            provider=provider,
            on_verify=on_verify,
            on_save=on_save,
            on_status=self._apply_status,
            on_message=self._show_controller_message,
            show_status=show_status,
        )
        self._controller.set_value_getter(lambda: self._text_field.value or "")

    @property
    def value(self) -> str:
        """Get current field value."""
        return self._text_field.value or ""

    @value.setter
    def value(self, val: str) -> None:
        """Set field value."""
        self._text_field.value = val
        self._controller.replace_value()
        update_control_if_mounted(self._text_field)

    @property
    def controller(self) -> ApiKeyVerificationController:
        return self._controller

    @property
    def _last_verified_hash(self) -> str:
        return self._controller.last_verified_hash

    @_last_verified_hash.setter
    def _last_verified_hash(self, value: str) -> None:
        self._controller.last_verified_hash = value

    def _get_key_hash(self, key: str) -> str:
        return ApiKeyVerificationController.key_hash(key)

    def _set_status(self, status: str) -> None:
        self._controller.force_status(status)

    def _toggle_password_visibility(self, e) -> None:
        """Toggle password visibility and update eye icon."""
        self._text_field.password = not self._text_field.password
        self._reveal_button.icon = (
            icons.VISIBILITY_OFF_ROUNDED if self._text_field.password else icons.VISIBILITY_ROUNDED
        )
        update_control_if_mounted(self._text_field)
        update_control_if_mounted(self._reveal_button)

    def _handle_change(self, e) -> None:
        """Mark the field dirty after user edits."""
        _ = e
        self._controller.notify_edit()

    def _apply_status(self, status: str) -> None:
        self._current_status = status
        if not self._show_status:
            return
        icon, color = _STATUS_ICONS.get(status, _STATUS_ICONS["idle"])
        self._status_icon.icon = icon
        self._status_icon.color = color
        self._status_icon.tooltip = t(_STATUS_KEYS.get(status, "api_key.status.idle"))
        update_control_if_mounted(self._status_icon)

    def _handle_blur(self, e) -> None:
        """Handle blur event - save and verify via the controller."""
        _ = e
        self._controller.handle_blur(self.value)
        drain = self._controller.run_pending()
        if drain is None:
            return
        page = control_page(self)
        if page is None:
            self._pending_drain = drain
            return
        page.run_task(self._drain_pending_verification, drain)

    async def _drain_pending_verification(self, drain) -> None:
        """Schedule a controller drain coroutine on the page event loop."""
        await drain

    async def _verify_async(self, key: str, key_hash: str) -> None:
        """Verify one key/hash pair directly, bypassing blur gating."""
        _ = key_hash
        await self._controller.verify_direct(key)

    async def _run_verification(self) -> None:
        """Test/compatibility entry point draining pending verification."""
        drain = getattr(self, "_pending_drain", None)
        if drain is not None:
            self._pending_drain = None
            await drain
            return
        task = self._controller.run_pending()
        if task is not None:
            await task

    def _show_controller_message(self, message_key: str, message: str) -> None:
        kwargs: dict[str, object] = {}
        if message_key == "snackbar.verification_ok":
            kwargs["provider"] = provider_label(self._provider)
            bgcolor = colors.GREEN_400
        elif message_key == "snackbar.verification_error":
            kwargs["message"] = self._friendly_message(message)
            bgcolor = colors.RED_400
        else:
            kwargs["message"] = self._friendly_message(message)
            bgcolor = colors.RED_400
        self._show_snackbar(t(message_key, **kwargs), bgcolor)

    def _friendly_message(self, raw: str) -> str:
        if raw.startswith("error.qwen_model_unavailable:"):
            model = raw.split(":", 1)[1] if ":" in raw else "unknown"
            return t("error.qwen_model_unavailable", model=model or "unknown")
        if raw.startswith("error."):
            return t(raw)
        return raw

    def _show_snackbar(self, message: str, bgcolor) -> None:
        """Show a toast via App-level callback or fallback to page."""
        if self._show_snackbar_cb:
            self._show_snackbar_cb(message, bgcolor)
        else:
            page = control_page(self)
            if page is None:
                return
            page.show_dialog(
                ft.SnackBar(
                    ft.Text(message, size=18, color=ft.Colors.WHITE),
                    bgcolor=bgcolor,
                    duration=4000,
                    behavior=ft.SnackBarBehavior.FLOATING,
                    elevation=0,
                    margin=ft.Margin.only(bottom=90),
                    padding=20,
                )
            )

    def apply_locale(self) -> None:
        """Update labels and tooltips when locale changes."""
        self._text_field.label = t(self._label_key)
        if self._show_status:
            tooltip_key = _STATUS_KEYS.get(self._current_status, "api_key.status.idle")
            self._status_icon.tooltip = t(tooltip_key)
        update_control_if_mounted(self._text_field)
        if self._show_status:
            update_control_if_mounted(self._status_icon)
