from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.components.warm_document_dialog import (
    WarmDocumentDialogAction,
    WarmDocumentDialogResult,
    join_body_paragraphs,
    open_warm_document_dialog,
    split_body_paragraphs,
)
from puripuly_heart.ui.flet_runtime import FILL_PARENT_WIDTH
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import COLOR_DIVIDER, COLOR_ON_BACKGROUND, COLOR_PRIMARY

_QQ_CREDENTIAL_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
_QQ_IDENTITY_MAX_LENGTH = 128


class QqManagedAuthDialog:
    action_labels = [
        "qq_auth.close",
        "qq_auth.submit",
    ]
    waiting_action_labels = ["qq_auth.cancel"]

    def __init__(
        self,
        page: ft.Page,
        *,
        on_continue: Callable[[], None],
        on_close: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self._page = page
        self._on_continue = on_continue
        self._on_close = on_close
        self._on_cancel = on_cancel
        self._dialog: ft.AlertDialog | None = None
        self._is_open = False
        self._is_waiting = False
        self._dialog_result: WarmDocumentDialogResult | None = None
        self._body_text: ft.Text | None = None
        self._actions: ft.Row | None = None
        self._continue_button: ft.TextButton | None = None
        self._close_button: ft.TextButton | None = None
        self._cancel_button: ft.TextButton | None = None
        self._qq_identity_field: ft.TextField | None = None
        self._credential_field: ft.TextField | None = None
        self._error_text: ft.Text | None = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_waiting(self) -> bool:
        return self._is_waiting

    @property
    def qq_identity(self) -> str:
        return self._field_value(self._qq_identity_field).strip()

    @property
    def credential(self) -> str:
        return self._field_value(self._credential_field)

    def open(self) -> None:
        if self._dialog is not None and self._is_open:
            return

        self._is_waiting = False
        self._qq_identity_field = self._build_text_field(
            "qq_auth.qq_identity.label",
            helper_key="qq_auth.qq_identity.helper",
        )
        self._credential_field = self._build_text_field(
            "qq_auth.credential.label",
            helper_key="qq_auth.credential.helper",
            password=True,
        )
        self._error_text = ft.Text(
            "",
            size=18,
            color=ft.Colors.ORANGE_700,
            selectable=True,
            visible=False,
        )
        self._dialog_result = open_warm_document_dialog(
            self._page,
            body_paragraphs=split_body_paragraphs(t("qq_auth.body")),
            extra_body_controls=[
                self._qq_identity_field,
                self._credential_field,
                self._error_text,
            ],
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
                    on_select=self._submit,
                    close_before_action=False,
                ),
            ],
        )
        self._dialog = self._dialog_result.dialog
        self._body_text = self._dialog_result.body_text
        self._actions = self._dialog_result.action_row
        self._close_button, self._continue_button = self._dialog_result.initial_action_buttons[0:2]
        self._cancel_button = None
        self._is_open = True

    def set_waiting(self) -> None:
        self._is_waiting = True
        self._set_fields_disabled(True)
        self._set_error_key(None)
        if self._dialog_result is None or self._body_text is None:
            return
        self._body_text.value = join_body_paragraphs(
            split_body_paragraphs(t("qq_auth.waiting_body"))
        )
        waiting_buttons = self._dialog_result.set_actions(
            [
                WarmDocumentDialogAction(
                    label=t("qq_auth.cancel"),
                    on_select=self._cancel_waiting,
                    close_before_action=False,
                )
            ]
        )
        self._continue_button = None
        self._close_button = None
        self._cancel_button = waiting_buttons[0]
        self._update_page_if_possible()

    def set_error(self, message_key: str, **message_kwargs: object) -> None:
        self._is_waiting = False
        self._set_fields_disabled(False)
        if self._body_text is not None:
            self._body_text.value = join_body_paragraphs(split_body_paragraphs(t("qq_auth.body")))
        if self._dialog_result is not None:
            buttons = self._dialog_result.set_actions(
                [
                    WarmDocumentDialogAction(
                        label=t("qq_auth.close"),
                        on_select=lambda: self._close_then(self._on_close),
                        close_before_action=False,
                    ),
                    WarmDocumentDialogAction(
                        label=t("qq_auth.submit"),
                        on_select=self._submit,
                        close_before_action=False,
                    ),
                ]
            )
            self._close_button, self._continue_button = buttons[0:2]
            self._cancel_button = None
        self._set_error_key(message_key, **message_kwargs)
        self._update_page_if_possible()

    def close(self) -> None:
        if self._dialog is None or not self._is_open:
            return
        self._page.pop_dialog()
        self._is_open = False

    def _build_text_field(
        self,
        label_key: str,
        *,
        helper_key: str,
        password: bool = False,
    ) -> ft.TextField:
        return ft.TextField(
            label=t(label_key),
            value="",
            helper=t(helper_key),
            dense=False,
            border_radius=14,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            content_padding=ft.Padding.symmetric(horizontal=16, vertical=20),
            text_size=22,
            color=COLOR_ON_BACKGROUND,
            width=FILL_PARENT_WIDTH,
            password=password,
            on_submit=lambda _: self._submit(),
        )

    def _submit(self) -> None:
        if not self._validate_inputs():
            self._set_error_key("qq_auth.error.invalid_input")
            self._update_page_if_possible()
            return
        self._on_continue()

    def _validate_inputs(self) -> bool:
        qq_identity = self.qq_identity.strip()
        if not qq_identity or not 1 <= len(qq_identity) <= _QQ_IDENTITY_MAX_LENGTH:
            return False
        if any(character.isspace() for character in qq_identity):
            return False
        if any(unicodedata.category(character).startswith("C") for character in qq_identity):
            return False
        if _QQ_CREDENTIAL_PATTERN.fullmatch(self.credential) is None:
            return False
        return True

    def _cancel_waiting(self) -> None:
        self.close()
        if self._on_cancel is not None:
            self._on_cancel()
        else:
            self._on_close()

    def _close_then(self, action: Callable[[], None]) -> None:
        self.close()
        action()

    def _set_fields_disabled(self, disabled: bool) -> None:
        for field in (self._qq_identity_field, self._credential_field):
            if field is not None:
                field.disabled = disabled

    def _set_error_key(self, message_key: str | None, **message_kwargs: object) -> None:
        if self._error_text is None:
            return
        self._error_text.value = "" if message_key is None else t(message_key, **message_kwargs)
        self._error_text.visible = message_key is not None

    def _update_page_if_possible(self) -> None:
        update = getattr(self._page, "update", None)
        if callable(update):
            update()

    @staticmethod
    def _field_value(field: ft.TextField | None) -> str:
        if field is None or not isinstance(field.value, str):
            return ""
        return field.value
