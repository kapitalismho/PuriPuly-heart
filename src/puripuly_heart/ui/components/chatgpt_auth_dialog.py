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
from puripuly_heart.ui.motion import close_modal


class ChatGptAuthDialog:
    action_labels = [
        "chatgpt_auth.close",
        "chatgpt_auth.continue",
    ]
    waiting_action_labels = [
        "chatgpt_auth.cancel",
    ]

    def __init__(
        self,
        page: ft.Page,
        *,
        on_continue: Callable[[], None],
        on_close: Callable[[], None],
        on_cancel: Callable[[], None],
    ) -> None:
        self._page = page
        self._on_continue = on_continue
        self._on_close = on_close
        self._on_cancel = on_cancel
        self._dialog_result: WarmDocumentDialogResult | None = None
        self._is_open = False
        self._is_waiting = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_waiting(self) -> bool:
        return self._is_waiting

    def open(self) -> None:
        if self._is_open:
            return
        self._is_waiting = False
        self._dialog_result = open_warm_document_dialog(
            self._page,
            body_paragraphs=split_body_paragraphs(t("chatgpt_auth.body")),
            actions=[
                WarmDocumentDialogAction(
                    label=t("chatgpt_auth.close"),
                    on_select=self._close_then_notify,
                    close_before_action=False,
                ),
                WarmDocumentDialogAction(
                    label=t("chatgpt_auth.continue"),
                    on_select=self._on_continue,
                    close_before_action=False,
                ),
            ],
        )
        self._is_open = True

    def set_waiting(self) -> None:
        self._is_waiting = True
        result = self._dialog_result
        if result is None or result.body_text is None:
            return
        result.body_text.value = join_body_paragraphs(
            split_body_paragraphs(t("chatgpt_auth.waiting_body"))
        )
        result.set_actions(
            [
                WarmDocumentDialogAction(
                    label=t("chatgpt_auth.cancel"),
                    on_select=self._cancel_waiting,
                    close_before_action=False,
                )
            ]
        )
        self._update_page()

    def close(self) -> None:
        if not self._is_open:
            return
        close_modal(self._page, self._dialog_result.dialog)
        self._is_open = False
        self._is_waiting = False

    def _close_then_notify(self) -> None:
        self.close()
        self._on_close()

    def _cancel_waiting(self) -> None:
        self.close()
        self._on_cancel()

    def _update_page(self) -> None:
        update = getattr(self._page, "update", None)
        if callable(update):
            update()


__all__ = ["ChatGptAuthDialog"]
