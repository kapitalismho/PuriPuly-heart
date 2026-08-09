from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import (
    FILL_PARENT_WIDTH,
    is_hover_active,
    update_control_if_mounted,
)
from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.i18n import get_locale, t
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_NEUTRAL_DARK,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)

_OPTION_CARD_RADIUS = 16
_OPTION_CARD_PADDING = 24
_OPTION_CARD_HEIGHT = 110
_PORT_LABEL_SIZE = 20
_PORT_TEXT_SIZE = 72
_PORT_LABEL_HEIGHT = 30
_PORT_FIELD_HEIGHT = _OPTION_CARD_HEIGHT
_PORT_BOTTOM_SPACER_HEIGHT = (
    _OPTION_CARD_HEIGHT * 3 + 12 * 3 - _PORT_LABEL_HEIGHT * 2 - _PORT_FIELD_HEIGHT * 2 - 8 * 5
)
_MODE_LABEL_KEYS = {
    "automatic": "settings.osc.mode.automatic",
    "manual": "settings.osc.mode.manual",
    "off": "settings.osc.mode.off",
}


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
        self._selected_mode = "automatic"
        self._mode_cards: dict[str, ft.Container] = {}
        self._mode_texts: dict[str, ft.Text] = {}
        self._last_committed: tuple[str, int, int] | None = None

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
    def selected_mode(self) -> str:
        return self._selected_mode

    @property
    def mode_cards(self) -> dict[str, ft.Container]:
        return self._mode_cards

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
        self._selected_mode = mode
        self._last_committed = None

        self._send_port_field = self._build_port_field(
            value=str(self._stored_send_port),
        )
        self._receive_port_field = self._build_port_field(
            value=str(self._stored_receive_port),
        )
        self._mode_cards = self._build_mode_cards()

        ports_column = ft.Column(
            controls=[
                self._build_section_header(t("settings.osc.ports")),
                self._build_port_label(t("settings.osc.send_port")),
                self._send_port_field,
                ft.Container(height=_PORT_BOTTOM_SPACER_HEIGHT),
                self._build_port_label(t("settings.osc.receive_port")),
                self._receive_port_field,
            ],
            spacing=8,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        mode_column = ft.Column(
            controls=[
                self._build_section_header(t("settings.osc.mode")),
                self._mode_cards["automatic"],
                self._mode_cards["manual"],
                self._mode_cards["off"],
            ],
            spacing=12,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        columns = ft.Row(
            controls=[ports_column, mode_column],
            spacing=32,
            vertical_alignment=ft.CrossAxisAlignment.START,
            expand=True,
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
                    ft.Container(height=16),
                    columns,
                ],
                spacing=20,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            width=880,
            height=700,
            padding=ft.Padding.symmetric(horizontal=32, vertical=32),
            bgcolor=COLOR_SURFACE,
            border_radius=28,
        )
        self._dialog = ft.AlertDialog(
            modal=False,
            content=modal_content,
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
            on_dismiss=self._commit,
        )
        self._refresh_mode_display()
        self._repaint_mode_cards()
        self._page.show_dialog(self._dialog)

    def _build_port_field(self, *, value: str) -> ft.TextField:
        return ft.TextField(
            value=value,
            keyboard_type=ft.KeyboardType.NUMBER,
            dense=True,
            height=_PORT_FIELD_HEIGHT,
            width=FILL_PARENT_WIDTH,
            border=ft.InputBorder.NONE,
            filled=False,
            bgcolor=ft.Colors.TRANSPARENT,
            color=COLOR_PRIMARY,
            text_size=_PORT_TEXT_SIZE,
            text_style=ft.TextStyle(
                weight=ft.FontWeight.BOLD,
                font_family=font_for_language(get_locale()),
            ),
            text_align=ft.TextAlign.CENTER,
            content_padding=ft.Padding.symmetric(horizontal=8, vertical=4),
            on_change=self._on_port_change,
            on_blur=self._on_port_change_end,
            on_submit=self._on_port_change_end,
        )

    def _build_port_label(self, label: str) -> ft.Container:
        return ft.Container(
            content=ft.Text(
                label,
                size=_PORT_LABEL_SIZE,
                weight=ft.FontWeight.BOLD,
                color=COLOR_NEUTRAL_DARK,
                text_align=ft.TextAlign.CENTER,
            ),
            height=_PORT_LABEL_HEIGHT,
            alignment=ft.Alignment(0, 0),
        )

    def _build_section_header(self, label: str) -> ft.Text:
        return ft.Text(
            label,
            size=18,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )

    def _build_mode_cards(self) -> dict[str, ft.Container]:
        cards: dict[str, ft.Container] = {}
        for value in ("automatic", "manual", "off"):
            text = ft.Text(
                t(_MODE_LABEL_KEYS[value]),
                size=20,
                weight=ft.FontWeight.BOLD,
                color=COLOR_ON_BACKGROUND,
                text_align=ft.TextAlign.CENTER,
            )
            card = ft.Container(
                content=text,
                bgcolor=COLOR_BACKGROUND,
                border_radius=_OPTION_CARD_RADIUS,
                padding=ft.Padding.all(_OPTION_CARD_PADDING),
                alignment=ft.Alignment.CENTER,
                on_click=lambda e, mode=value: self._select_mode(mode),
                on_hover=self._on_mode_hover,
                animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
                height=_OPTION_CARD_HEIGHT,
            )
            cards[value] = card
            self._mode_texts[value] = text
        return cards

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

    def _select_mode(self, mode: str) -> None:
        if mode not in {"automatic", "manual", "off"}:
            return
        if self._selected_mode == "manual" and not self._store_manual_ports():
            return
        self._selected_mode = mode
        self._repaint_mode_cards()
        self._refresh_mode_display()
        if self._dialog is not None:
            update_control_if_mounted(self._dialog)
        self._commit()
        if mode != "manual":
            self._close()

    def _repaint_mode_cards(self) -> None:
        for value, card in self._mode_cards.items():
            text = self._mode_texts.get(value)
            if text is None:
                continue
            selected = value == self._selected_mode
            card.bgcolor = COLOR_PRIMARY if selected else COLOR_BACKGROUND
            text.color = ft.Colors.WHITE if selected else COLOR_ON_BACKGROUND

    def _on_mode_hover(self, e: ft.ControlEvent) -> None:
        container = e.control
        text_control = container.content
        if not isinstance(text_control, ft.Text):
            return
        if text_control.color == ft.Colors.WHITE:
            return
        next_color = COLOR_PRIMARY if is_hover_active(e) else COLOR_ON_BACKGROUND
        if text_control.color == next_color:
            return
        text_control.color = next_color
        container.update()

    def _refresh_mode_display(self) -> None:
        send_field = self._send_port_field
        receive_field = self._receive_port_field
        if send_field is None or receive_field is None:
            return
        manual = self._selected_mode == "manual"
        send_field.read_only = not manual
        receive_field.read_only = not manual
        muted = self._selected_mode == "off"
        port_color = COLOR_NEUTRAL_DARK if muted else COLOR_PRIMARY
        if self._selected_mode == "automatic":
            send_field.value = str(self._effective_send_port or self._stored_send_port)
            receive_field.value = str(self._effective_receive_port or self._stored_receive_port)
        else:
            send_field.value = str(self._stored_send_port)
            receive_field.value = str(self._stored_receive_port)
        send_field.color = port_color
        receive_field.color = port_color

    def _on_port_change_end(self, _event: ft.ControlEvent | None = None) -> None:
        self._commit()

    def _on_port_change(self, _event: ft.ControlEvent | None = None) -> None:
        if self._send_port_field is not None:
            self._send_port_field.error_text = None
        if self._receive_port_field is not None:
            self._receive_port_field.error_text = None

    def _store_manual_ports(self) -> bool:
        send_field = self._send_port_field
        receive_field = self._receive_port_field
        if send_field is None or receive_field is None:
            return False
        try:
            send_port = int((send_field.value or "").strip())
            receive_port = int((receive_field.value or "").strip())
        except (TypeError, ValueError):
            self._show_port_error()
            return False
        if not (1 <= send_port <= 65535 and 1 <= receive_port <= 65535):
            self._show_port_error()
            return False
        self._stored_send_port = send_port
        self._stored_receive_port = receive_port
        return True

    def _commit(self, _event: ft.ControlEvent | None = None) -> None:
        send_field = self._send_port_field
        receive_field = self._receive_port_field
        if send_field is None or receive_field is None:
            return
        send_field.error_text = None
        receive_field.error_text = None
        mode = self._selected_mode
        if mode == "manual":
            if not self._store_manual_ports():
                return
            send_port = self._stored_send_port
            receive_port = self._stored_receive_port
        else:
            send_port = self._stored_send_port
            receive_port = self._stored_receive_port
        commit = (mode, send_port, receive_port)
        if commit == self._last_committed:
            return
        self._last_committed = commit
        self._on_select(*commit)

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

    def _close(self) -> None:
        try:
            self._page.pop_dialog()
        except Exception:
            return


__all__ = ["OscConnectionModal"]
