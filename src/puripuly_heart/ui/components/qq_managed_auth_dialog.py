from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.components.glow import create_glow_stack
from puripuly_heart.ui.components.warm_document_dialog import (
    WarmDocumentDialogAction,
    WarmDocumentDialogResult,
    join_body_paragraphs,
    open_warm_document_dialog,
    split_body_paragraphs,
)
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import COLOR_DIVIDER, COLOR_ON_BACKGROUND, COLOR_PRIMARY


class QqManagedAuthDialog:
    action_labels = [
        "qq_auth.close",
        "qq_auth.submit",
    ]

    def __init__(
        self,
        page: ft.Page,
        *,
        on_submit: Callable[[], None],
        on_close: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self._page = page
        self._on_submit = on_submit
        self._on_close = on_close
        self._on_cancel = on_cancel
        self._dialog: ft.AlertDialog | None = None
        self._is_open = False
        self._is_waiting = False

        self._dialog_result: WarmDocumentDialogResult | None = None
        self._body_text: ft.Text | None = None
        self._actions: ft.Row | None = None
        self._submit_button: ft.TextButton | None = None
        self._close_button: ft.TextButton | None = None
        self._cancel_button: ft.TextButton | None = None
        self._qq_identity_field: ft.TextField | None = None
        self._credential_field: ft.TextField | None = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_waiting(self) -> bool:
        return self._is_waiting

    @property
    def qq_identity(self) -> str:
        if self._qq_identity_field is None:
            return ""
        value = self._qq_identity_field.value
        return value.strip() if isinstance(value, str) else ""

    @property
    def credential(self) -> str:
        if self._credential_field is None:
            return ""
        value = self._credential_field.value
        return value.strip() if isinstance(value, str) else ""

    def open(self) -> None:
        if self._dialog is not None and self._is_open:
            return

        self._is_waiting = False
        self._qq_identity_field = self._build_qq_identity_field()
        self._credential_field = self._build_credential_field()
        self._dialog_result = open_warm_document_dialog(
            self._page,
            body_paragraphs=split_body_paragraphs(t("qq_auth.body")),
            extra_body_controls=[self._qq_identity_field, self._credential_field],
            body_spacing=44,
            action_top_margin=24,
            actions=[
                WarmDocumentDialogAction(
                    label=t("qq_auth.close"),
                    on_select=lambda: self._close_then(self._on_close),
                    close_before_action=False,
                ),
                WarmDocumentDialogAction(
                    label=t("qq_auth.submit"),
                    on_select=self._on_submit,
                    close_before_action=False,
                ),
            ],
            glow_factory=create_glow_stack,
        )
        self._dialog = self._dialog_result.dialog
        self._body_text = self._dialog_result.body_text
        self._actions = self._dialog_result.action_row
        (
            self._close_button,
            self._submit_button,
        ) = self._dialog_result.initial_action_buttons[0:2]
        self._cancel_button = None
        self._is_open = True

    def _build_qq_identity_field(self) -> ft.TextField:
        return ft.TextField(
            label=t("qq_auth.qq_identity.label"),
            value="",
            helper_text=t("qq_auth.qq_identity.helper"),
            dense=False,
            border_radius=14,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            content_padding=ft.padding.symmetric(horizontal=16, vertical=20),
            text_size=22,
            color=COLOR_ON_BACKGROUND,
            on_submit=lambda _: self._on_submit(),
        )

    def _build_credential_field(self) -> ft.TextField:
        return ft.TextField(
            label=t("qq_auth.credential.label"),
            value="",
            helper_text=t("qq_auth.credential.helper"),
            dense=False,
            border_radius=14,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            content_padding=ft.padding.symmetric(horizontal=16, vertical=20),
            text_size=22,
            color=COLOR_ON_BACKGROUND,
            multiline=True,
            max_lines=3,
            min_lines=1,
            on_submit=lambda _: self._on_submit(),
        )

    def set_waiting(self) -> None:
        self._is_waiting = True
        if self._dialog_result is None or self._body_text is None:
            return

        if self._qq_identity_field is not None:
            self._qq_identity_field.disabled = True
        if self._credential_field is not None:
            self._credential_field.disabled = True

        self._body_text.value = join_body_paragraphs(
            split_body_paragraphs(t("qq_auth.waiting_body"))
        )
        self._update_page_if_possible()

    def close(self) -> None:
        if self._dialog is None or not self._is_open:
            return
        self._page.close(self._dialog)
        self._is_open = False

    def _close_then(self, action: Callable[[], None]) -> None:
        self.close()
        action()

    def _update_page_if_possible(self) -> None:
        update = getattr(self._page, "update", None)
        if callable(update):
            update()
