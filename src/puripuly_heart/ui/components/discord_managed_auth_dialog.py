from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.components.warm_document_dialog import (
    WarmDocumentDialogAction,
    WarmDocumentDialogResult,
    join_body_paragraphs,
    open_warm_document_dialog,
    split_body_paragraphs,
)
from puripuly_heart.ui.i18n import t


class DiscordManagedAuthDialog:
    action_labels = [
        "openrouter.handoff.chatgpt",
        "discord_auth.continue",
    ]
    waiting_action_labels = [
        "discord_auth.cancel",
    ]

    def __init__(
        self,
        page: ft.Page,
        *,
        on_continue: Callable[[], None],
        on_byok: Callable[[], None],
        on_close: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
        on_chatgpt: Callable[[], None] | None = None,
    ) -> None:
        self._page = page
        self._on_continue = on_continue
        self._on_byok = on_byok
        self._on_close = on_close
        self._on_cancel = on_cancel
        self._on_chatgpt = on_chatgpt
        self._dialog: ft.AlertDialog | None = None
        self._is_open = False
        self._is_waiting = False

        self._dialog_result: WarmDocumentDialogResult | None = None
        self._body_text: ft.Text | None = None
        self._actions: ft.Row | None = None
        self._continue_button: ft.TextButton | None = None
        self._byok_button: ft.TextButton | None = None
        self._close_button: ft.TextButton | None = None
        self._chatgpt_button: ft.TextButton | None = None
        self._cancel_button: ft.TextButton | None = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_waiting(self) -> bool:
        return self._is_waiting

    def open(self) -> None:
        if self._dialog is not None and self._is_open:
            return

        self._is_waiting = False
        self._dialog_result = open_warm_document_dialog(
            self._page,
            body_paragraphs=split_body_paragraphs(t("discord_auth.body")),
            action_top_margin=24,
            modal=False,
            on_dismiss=self._handle_dismiss,
            actions=[
                WarmDocumentDialogAction(
                    label=t("openrouter.handoff.chatgpt"),
                    on_select=lambda: self._close_then(self._on_chatgpt or self._on_close),
                    close_before_action=False,
                ),
                WarmDocumentDialogAction(
                    label=t("discord_auth.continue"),
                    on_select=self._submit,
                    close_before_action=False,
                ),
            ],
        )
        self._dialog = self._dialog_result.dialog
        self._body_text = self._dialog_result.body_text
        self._actions = self._dialog_result.action_row
        (
            self._chatgpt_button,
            self._continue_button,
        ) = self._dialog_result.initial_action_buttons[0:2]
        self._close_button = None
        self._byok_button = None
        self._cancel_button = None
        self._is_open = True

    def set_waiting(self) -> None:
        self._is_waiting = True
        if self._dialog is not None:
            self._dialog.modal = True
        if self._dialog_result is None or self._body_text is None:
            return

        self._body_text.value = join_body_paragraphs(
            split_body_paragraphs(t("discord_auth.waiting_body"))
        )
        waiting_buttons = self._dialog_result.set_actions(self._build_waiting_actions())
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

    def set_recovering(self) -> None:
        if not self._is_open or not self._is_waiting:
            return
        if self._dialog_result is None or self._body_text is None:
            return
        self._body_text.value = join_body_paragraphs(
            split_body_paragraphs(t("discord_auth.recovering_body"))
        )
        self._update_page_if_possible()

    def set_action_required(self) -> None:
        self.set_error("discord_auth.action_required_body")

    def set_error(
        self, message_key: str = "discord_auth.error.retry", *, resume: bool = False
    ) -> None:
        if not self._is_open:
            return
        self._is_waiting = False
        if self._dialog is not None:
            self._dialog.modal = False
        if self._dialog_result is None or self._body_text is None:
            return
        self._body_text.value = join_body_paragraphs(split_body_paragraphs(t(message_key)))
        self._close_button, self._continue_button = self._dialog_result.set_actions(
            [
                WarmDocumentDialogAction(
                    label=t("discord_auth.close"),
                    on_select=lambda: self._close_then(self._on_close),
                    close_before_action=False,
                ),
                WarmDocumentDialogAction(
                    label=t("discord_auth.resume" if resume else "discord_auth.retry"),
                    on_select=self._submit,
                    close_before_action=False,
                ),
            ]
        )
        self._cancel_button = None
        self._update_page_if_possible()

    def close(self) -> None:
        if self._dialog is None or not self._is_open:
            return
        self._page.pop_dialog()
        self._is_open = False
        self._is_waiting = False

    def _build_waiting_actions(self) -> list[WarmDocumentDialogAction]:
        actions: list[WarmDocumentDialogAction] = []
        actions.append(
            WarmDocumentDialogAction(
                label=t("discord_auth.cancel"),
                on_select=self._cancel_waiting,
                close_before_action=False,
            )
        )
        return actions

    def _submit(self) -> None:
        if self._is_open and not self._is_waiting:
            self._on_continue()

    def _handle_dismiss(self) -> None:
        if not self._is_open or self._is_waiting:
            return
        self._is_open = False
        self._on_close()

    def _close_then(self, action: Callable[[], None]) -> None:
        self.close()
        action()

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
