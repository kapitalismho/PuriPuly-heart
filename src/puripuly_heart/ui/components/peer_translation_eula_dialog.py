from __future__ import annotations

from typing import Callable

import flet as ft

from puripuly_heart.ui.components.warm_document_dialog import (
    open_warm_document_dialog,
    split_body_paragraphs,
)
from puripuly_heart.ui.i18n import t


class PeerTranslationEulaDialog:
    def __init__(
        self,
        page: ft.Page,
        *,
        on_accept: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self._page = page
        self._on_accept = on_accept
        self._on_cancel = on_cancel
        self._dialog: ft.AlertDialog | None = None

    def open(self) -> None:
        result = open_warm_document_dialog(
            self._page,
            body_paragraphs=split_body_paragraphs(t("peer_translation_eula.body")),
            primary_label=t("peer_translation_eula.accept"),
            primary_action=self._on_accept,
            secondary_label=t("peer_translation_eula.cancel"),
            secondary_action=self._on_cancel,
        )
        self._dialog = result.dialog
