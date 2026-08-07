from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.components.warm_document_dialog import (
    BODY_TEXT_SIZE,
    WarmDocumentDialogAction,
    WarmDocumentDialogResult,
    join_body_paragraphs,
    open_warm_document_dialog,
    split_body_paragraphs,
)
from puripuly_heart.ui.flet_runtime import FILL_PARENT_WIDTH
from puripuly_heart.ui.fonts import default_font_family
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import COLOR_DIVIDER, COLOR_ON_BACKGROUND, COLOR_PRIMARY


class DiscordManagedAuthDialog:
    action_labels = [
        "discord_auth.close",
        "discord_auth.continue",
    ]
    waiting_action_labels = [
        "discord_auth.cancel",
        "discord_auth.reopen_browser",
    ]

    def __init__(
        self,
        page: ft.Page,
        *,
        on_continue: Callable[[], None],
        on_byok: Callable[[], None],
        on_close: Callable[[], None],
        on_reopen_browser: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self._page = page
        self._on_continue = on_continue
        self._on_byok = on_byok
        self._on_close = on_close
        self._on_reopen_browser = on_reopen_browser
        self._on_cancel = on_cancel
        self._dialog: ft.AlertDialog | None = None
        self._is_open = False
        self._is_waiting = False
        self._referral_expanded = False

        self._dialog_result: WarmDocumentDialogResult | None = None
        self._body_text: ft.Text | None = None
        self._actions: ft.Row | None = None
        self._continue_button: ft.TextButton | None = None
        self._byok_button: ft.TextButton | None = None
        self._close_button: ft.TextButton | None = None
        self._reopen_browser_button: ft.TextButton | None = None
        self._cancel_button: ft.TextButton | None = None
        self._referral_toggle: ft.TextButton | None = None
        self._referral_toggle_label: ft.Text | None = None
        self._referral_id_field: ft.TextField | None = None
        self._referral_section: ft.Column | None = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_waiting(self) -> bool:
        return self._is_waiting

    @property
    def referral_id(self) -> str:
        if self._referral_id_field is None:
            return ""
        value = self._referral_id_field.value
        return value if isinstance(value, str) else ""

    def open(self) -> None:
        if self._dialog is not None and self._is_open:
            return

        self._is_waiting = False
        self._referral_expanded = False
        self._referral_id_field = self._build_referral_id_field()
        self._referral_toggle = self._build_referral_toggle()
        self._referral_section = ft.Column(
            controls=[self._referral_toggle, self._referral_id_field],
            spacing=12,
            tight=True,
        )
        self._dialog_result = open_warm_document_dialog(
            self._page,
            body_paragraphs=split_body_paragraphs(t("discord_auth.body")),
            extra_body_controls=[self._referral_section],
            body_spacing=44,
            action_top_margin=24,
            actions=[
                WarmDocumentDialogAction(
                    label=t("discord_auth.close"),
                    on_select=lambda: self._close_then(self._on_close),
                    close_before_action=False,
                ),
                WarmDocumentDialogAction(
                    label=t("discord_auth.continue"),
                    on_select=self._on_continue,
                    close_before_action=False,
                ),
            ],
        )
        self._dialog = self._dialog_result.dialog
        self._body_text = self._dialog_result.body_text
        self._actions = self._dialog_result.action_row
        (
            self._close_button,
            self._continue_button,
        ) = self._dialog_result.initial_action_buttons[0:2]
        self._byok_button = None
        self._reopen_browser_button = None
        self._cancel_button = None
        self._is_open = True

    def _build_referral_id_field(self) -> ft.TextField:
        return ft.TextField(
            label=t("discord_auth.referral_id.label"),
            value="",
            helper=t("discord_auth.referral_id.helper"),
            dense=False,
            border_radius=14,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            content_padding=ft.Padding.symmetric(horizontal=16, vertical=20),
            text_size=22,
            color=COLOR_ON_BACKGROUND,
            width=FILL_PARENT_WIDTH,
            visible=False,
            on_submit=lambda _: self._on_continue(),
        )

    def _build_referral_toggle(self) -> ft.TextButton:
        label = ft.Text(
            t("discord_auth.referral_id.expand"),
            size=BODY_TEXT_SIZE,
            color=COLOR_ON_BACKGROUND,
            font_family=default_font_family(),
            weight=ft.FontWeight.W_400,
        )
        self._referral_toggle_label = label
        return ft.TextButton(
            content=label,
            style=ft.ButtonStyle(
                color={
                    ft.ControlState.DEFAULT: COLOR_ON_BACKGROUND,
                    ft.ControlState.HOVERED: COLOR_PRIMARY,
                },
                bgcolor=ft.Colors.TRANSPARENT,
                padding=ft.Padding.symmetric(horizontal=0, vertical=4),
                overlay_color=ft.Colors.TRANSPARENT,
                animation_duration=0,
            ),
            on_click=lambda _: self._toggle_referral_section(),
        )

    def _toggle_referral_section(self) -> None:
        if self._referral_id_field is None or self._referral_toggle_label is None:
            return
        self._referral_expanded = not self._referral_expanded
        self._referral_id_field.visible = self._referral_expanded
        self._referral_toggle_label.value = t(
            "discord_auth.referral_id.collapse"
            if self._referral_expanded
            else "discord_auth.referral_id.expand"
        )
        self._update_page_if_possible()

    def set_waiting(self) -> None:
        self._is_waiting = True
        if self._dialog_result is None or self._body_text is None:
            return

        if self._referral_section is not None:
            referral_section = self._referral_section
            if referral_section in self._dialog_result.body_column.controls:
                self._dialog_result.body_column.controls.remove(referral_section)
            self._referral_section = None
        self._referral_id_field = None
        self._referral_toggle = None
        self._referral_toggle_label = None
        self._referral_expanded = False

        self._body_text.value = join_body_paragraphs(
            split_body_paragraphs(t("discord_auth.waiting_body"))
        )
        waiting_buttons = self._dialog_result.set_actions(self._build_waiting_actions())
        self._reopen_browser_button = None
        self._cancel_button = None
        if self._on_reopen_browser is not None:
            self._cancel_button = waiting_buttons[0]
            self._reopen_browser_button = waiting_buttons[1]
        else:
            self._cancel_button = waiting_buttons[0]
        self._update_page_if_possible()

    def set_callback_received(self) -> None:
        if not self._is_open or not self._is_waiting:
            return
        if self._dialog_result is None or self._body_text is None:
            return
        self._body_text.value = join_body_paragraphs(
            split_body_paragraphs(t("discord_auth.callback_received_body"))
        )
        self._update_page_if_possible()

    def close(self) -> None:
        if self._dialog is None or not self._is_open:
            return
        self._page.pop_dialog()
        self._is_open = False

    def _build_waiting_actions(self) -> list[WarmDocumentDialogAction]:
        actions: list[WarmDocumentDialogAction] = []
        actions.append(
            WarmDocumentDialogAction(
                label=t("discord_auth.cancel"),
                on_select=self._cancel_waiting,
                close_before_action=False,
            )
        )
        if self._on_reopen_browser is not None:
            actions.append(
                WarmDocumentDialogAction(
                    label=t("discord_auth.reopen_browser"),
                    on_select=self._reopen_browser,
                    close_before_action=False,
                )
            )
        return actions

    def _close_then(self, action: Callable[[], None]) -> None:
        self.close()
        action()

    def _reopen_browser(self) -> None:
        if self._on_reopen_browser is not None:
            self._on_reopen_browser()

    def _cancel_waiting(self) -> None:
        self.close()
        if self._on_cancel is not None:
            self._on_cancel()
        else:
            self._on_close()

    def _update_page_if_possible(self) -> None:
        update = getattr(self._page, "update", None)
        if callable(update):
            update()
