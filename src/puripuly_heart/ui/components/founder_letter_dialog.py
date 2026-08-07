from __future__ import annotations

from typing import Callable

import flet as ft

from puripuly_heart.ui.components.warm_document_dialog import open_warm_document_dialog
from puripuly_heart.ui.i18n import t

FOUNDER_LETTER_PARAGRAPH_KEYS = (
    "openrouter.handoff.letter.p1",
    "openrouter.handoff.letter.p2",
    "openrouter.handoff.letter.p3",
    "openrouter.handoff.letter.p4",
    "openrouter.handoff.letter.p5",
    "openrouter.handoff.letter.p6",
)


class FounderLetterDialog:
    def __init__(
        self,
        page: ft.Page,
        *,
        on_readme: Callable[[], None] | None = None,
        on_connect: Callable[[], None] | None = None,
        on_contact: Callable[[], None] | None = None,
    ) -> None:
        # Legacy callbacks are accepted for older call sites but intentionally ignored.
        del on_connect, on_contact
        self._page = page
        self._on_readme = on_readme
        self._dialog: ft.AlertDialog | None = None
        self._acknowledge_button: ft.TextButton | None = None
        self._cancel_button: ft.TextButton | None = None
        self._connect_button: ft.TextButton | None = None
        self._contact_button: ft.TextButton | None = None

    def open(self) -> None:
        paragraphs = [t(key) for key in FOUNDER_LETTER_PARAGRAPH_KEYS]
        result = open_warm_document_dialog(
            self._page,
            body_paragraphs=paragraphs,
            primary_label=t("openrouter.handoff.readme"),
            primary_action=self._on_readme,
            secondary_label=t("openrouter.handoff.close"),
        )
        self._dialog = result.dialog
        self._acknowledge_button = result.primary_button
        self._cancel_button = result.secondary_button
        self._connect_button = self._acknowledge_button
        self._contact_button = self._cancel_button
