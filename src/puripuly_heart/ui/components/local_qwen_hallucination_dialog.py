from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.components.glow import create_glow_stack
from puripuly_heart.ui.components.warm_document_dialog import (
    WarmDocumentDialogResult,
    open_warm_document_dialog,
    split_body_paragraphs,
)
from puripuly_heart.ui.i18n import t


class LocalQwenHallucinationDialog:
    action_labels = [
        "local_qwen_hallucination.close",
        "local_qwen_hallucination.open_guide",
    ]

    def __init__(
        self,
        page: ft.Page,
        *,
        on_open_guide: Callable[[], None],
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._page = page
        self._on_open_guide_callback = on_open_guide
        self._on_close_callback = on_close
        self._dialog: ft.AlertDialog | None = None
        self._dialog_result: WarmDocumentDialogResult | None = None

    def open(self) -> None:
        result = open_warm_document_dialog(
            self._page,
            body_paragraphs=split_body_paragraphs(t("local_qwen_hallucination.body")),
            primary_label=t("local_qwen_hallucination.open_guide"),
            primary_action=self._on_open_guide,
            secondary_label=t("local_qwen_hallucination.close"),
            secondary_action=self._on_close,
            glow_factory=create_glow_stack,
        )
        self._dialog = result.dialog
        self._dialog_result = result

    def close(self) -> None:
        if self._dialog is None:
            return
        self._page.pop_dialog()

    def _on_open_guide(self) -> None:
        self._on_open_guide_callback()

    def _on_close(self) -> None:
        if self._on_close_callback is not None:
            self._on_close_callback()
