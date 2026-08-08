from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import update_control_if_mounted
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_NEUTRAL_DARK,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)


class OscConnectionModal:
    def __init__(
        self,
        page: ft.Page,
        on_select: Callable[[str, int, int], None],
        *,
        effective_ports_provider: Callable[[], tuple[int | None, int | None]] | None = None,
    ) -> None:
        self._page = page
        self._on_select = on_select
        self._effective_ports_provider = effective_ports_provider
        self._dialog: ft.AlertDialog | None = None
        self._stored_send_port = 9000
        self._stored_receive_port = 9001
        self._effective_send_port: int | None = None
        self._effective_receive_port: int | None = None
        self._send_port_field: ft.TextField | None = None
        self._receive_port_field: ft.TextField | None = None
        self._send_effective_text: ft.Text | None = None
        self._receive_effective_text: ft.Text | None = None
        self._mode_group: ft.RadioGroup | None = None
        self._apply_button: ft.TextButton | None = None

    @property
    def dialog(self) -> ft.AlertDialog | None:
        return self._dialog

    @property
    def send_port_field(self) -> ft.TextField | None:
        return self._send_port_field

    @property
    def receive_port_field(self) -> ft.TextField | None:
        return self._receive_port_field

    @property
    def mode_group(self) -> ft.RadioGroup | None:
        return self._mode_group

    def open(
        self,
        mode: str,
        send_port: int,
        receive_port: int,
        *,
        effective_send_port: int | None = None,
        effective_receive_port: int | None = None,
    ) -> None:
        self._stored_send_port = int(send_port)
        self._stored_receive_port = int(receive_port)
        provider_ports = self._read_effective_ports()
        self._effective_send_port = effective_send_port or provider_ports[0]
        self._effective_receive_port = effective_receive_port or provider_ports[1]
        if mode not in {"automatic", "manual", "off"}:
            mode = "automatic"

        self._send_port_field = ft.TextField(
            label=t("settings.osc.send_port"),
            value=str(self._stored_send_port),
            keyboard_type=ft.KeyboardType.NUMBER,
            dense=True,
            expand=True,
            disabled=mode != "manual",
        )
        self._receive_port_field = ft.TextField(
            label=t("settings.osc.receive_port"),
            value=str(self._stored_receive_port),
            keyboard_type=ft.KeyboardType.NUMBER,
            dense=True,
            expand=True,
            disabled=mode != "manual",
        )
        self._send_effective_text = ft.Text(size=14, color=COLOR_NEUTRAL_DARK)
        self._receive_effective_text = ft.Text(size=14, color=COLOR_NEUTRAL_DARK)
        self._mode_group = ft.RadioGroup(
            value=mode,
            on_change=self._on_mode_change,
            content=ft.Column(
                controls=[
                    ft.Radio(value="automatic", label=t("settings.osc.mode.automatic")),
                    ft.Radio(value="manual", label=t("settings.osc.mode.manual")),
                    ft.Radio(value="off", label=t("settings.osc.mode.off")),
                ],
                spacing=16,
            ),
        )
        self._apply_button = ft.TextButton(
            content=t("settings.osc.action.apply"),
            on_click=self._apply,
        )
        cancel_button = ft.TextButton(
            content=t("settings.osc.action.cancel"),
            on_click=self._cancel,
        )

        left_column = ft.Column(
            controls=[
                ft.Text(
                    t("settings.osc.ports"),
                    size=18,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_SECONDARY,
                ),
                self._send_port_field,
                self._send_effective_text,
                self._receive_port_field,
                self._receive_effective_text,
            ],
            spacing=10,
            expand=True,
        )
        right_column = ft.Column(
            controls=[
                ft.Text(
                    t("settings.osc.connection"),
                    size=18,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_SECONDARY,
                ),
                self._mode_group,
            ],
            spacing=10,
            expand=True,
        )
        columns = ft.Row(
            controls=[left_column, right_column],
            spacing=32,
            vertical_alignment=ft.CrossAxisAlignment.START,
            expand=True,
        )
        actions = ft.Row(
            controls=[cancel_button, self._apply_button],
            alignment=ft.MainAxisAlignment.END,
        )
        modal_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        t("settings.osc.connection.title"),
                        size=24,
                        weight=ft.FontWeight.BOLD,
                        color=COLOR_SECONDARY,
                    ),
                    columns,
                    actions,
                ],
                spacing=20,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            width=880,
            height=480,
            padding=ft.Padding.symmetric(horizontal=32, vertical=32),
            bgcolor=COLOR_SURFACE,
            border_radius=28,
        )
        self._dialog = ft.AlertDialog(
            modal=False,
            content=modal_content,
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
        )
        self._refresh_mode_display()
        self._page.show_dialog(self._dialog)

    def _read_effective_ports(self) -> tuple[int | None, int | None]:
        provider = self._effective_ports_provider
        if provider is None:
            return (None, None)
        try:
            value = provider()
        except Exception:
            return (None, None)
        if not isinstance(value, tuple) or len(value) != 2:
            return (None, None)
        return (
            value[0] if isinstance(value[0], int) and value[0] > 0 else None,
            value[1] if isinstance(value[1], int) and value[1] > 0 else None,
        )

    def _on_mode_change(self, _event: ft.ControlEvent) -> None:
        self._refresh_mode_display()
        if self._dialog is not None:
            update_control_if_mounted(self._dialog)

    def _refresh_mode_display(self) -> None:
        group = self._mode_group
        send_field = self._send_port_field
        receive_field = self._receive_port_field
        send_effective = self._send_effective_text
        receive_effective = self._receive_effective_text
        if group is None or send_field is None or receive_field is None:
            return
        mode = group.value or "automatic"
        manual = mode == "manual"
        send_field.disabled = not manual
        receive_field.disabled = not manual
        if mode == "automatic":
            send_value = self._effective_send_port or self._stored_send_port
            receive_value = self._effective_receive_port or self._stored_receive_port
            send_field.value = str(send_value)
            receive_field.value = str(receive_value)
            if send_effective is not None:
                send_effective.value = t("settings.osc.effective_port", port=send_value)
                send_effective.visible = True
            if receive_effective is not None:
                receive_effective.value = t("settings.osc.effective_port", port=receive_value)
                receive_effective.visible = True
        else:
            send_field.value = str(self._stored_send_port)
            receive_field.value = str(self._stored_receive_port)
            if send_effective is not None:
                send_effective.visible = False
            if receive_effective is not None:
                receive_effective.visible = False

    def _apply(self, _event: ft.ControlEvent) -> None:
        group = self._mode_group
        send_field = self._send_port_field
        receive_field = self._receive_port_field
        if group is None or send_field is None or receive_field is None:
            return
        mode = group.value or "automatic"
        if mode == "manual":
            try:
                send_port = int((send_field.value or "").strip())
                receive_port = int((receive_field.value or "").strip())
            except (TypeError, ValueError):
                self._show_port_error()
                return
            if not (1 <= send_port <= 65535 and 1 <= receive_port <= 65535):
                self._show_port_error()
                return
        else:
            send_port = self._stored_send_port
            receive_port = self._stored_receive_port
        self._close()
        self._on_select(mode, send_port, receive_port)

    def _show_port_error(self) -> None:
        message = t("settings.osc.invalid_port")
        if self._send_port_field is not None:
            self._send_port_field.error_text = message
        if self._receive_port_field is not None:
            self._receive_port_field.error_text = message
        if self._dialog is not None:
            try:
                self._dialog.update()
            except Exception:
                pass

    def _cancel(self, _event: ft.ControlEvent) -> None:
        self._close()

    def _close(self) -> None:
        try:
            self._page.pop_dialog()
        except Exception:
            return


__all__ = ["OscConnectionModal"]
