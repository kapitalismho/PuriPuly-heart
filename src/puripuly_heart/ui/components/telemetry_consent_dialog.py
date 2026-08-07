from __future__ import annotations

from typing import Callable

import flet as ft

from puripuly_heart.ui.components.warm_document_dialog import (
    WarmDocumentDialogResult,
    open_warm_document_dialog,
)
from puripuly_heart.ui.i18n import t


class TelemetryConsentDialog:
    def __init__(
        self, page: ft.Page, *, on_allow: Callable[[], None], on_decline: Callable[[], None]
    ):
        self._page = page
        self._on_allow = on_allow
        self._on_decline = on_decline
        self._dialog: ft.AlertDialog | None = None
        self._dialog_result: WarmDocumentDialogResult | None = None

    def open(self) -> None:
        result = open_warm_document_dialog(
            self._page,
            title=t("telemetry.consent.title"),
            body_paragraphs=(
                t("telemetry.consent.body"),
                t("telemetry.consent.excludes"),
            ),
            primary_label=t("telemetry.consent.allow"),
            primary_action=self._on_allow,
            secondary_label=t("telemetry.consent.decline"),
            secondary_action=self._on_decline,
        )
        self._dialog = result.dialog
        self._dialog_result = result
