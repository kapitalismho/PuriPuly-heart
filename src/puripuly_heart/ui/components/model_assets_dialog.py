from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import flet as ft

from puripuly_heart.ui.components.modal_kit import (
    CARD_HEIGHT,
    CARD_LABEL_SIZE,
    CARD_PADDING,
    CARD_RADIUS,
    CARD_SPACING,
    MODAL_COLUMN_SPACING,
    ModalCard,
    build_modal_dialog,
    build_modal_surface,
    build_section_header,
)
from puripuly_heart.ui.flet_runtime import update_control_if_mounted
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import COLOR_BACKGROUND, COLOR_PRIMARY, COLOR_WARNING

logger = logging.getLogger(__name__)

RENDER_INTERVAL_S = 0.25
_NOTICE_TEXT_SIZE = 18
_ERROR_KEYS = frozenset(
    {
        "connection_failed",
        "proxy_failed",
        "download_timeout",
        "disk_full",
        "checksum_failed",
        "file_size_mismatch",
        "missing_file",
        "model_in_use",
        "task_busy",
    }
)
_BUSY_STATUSES = frozenset({"downloading", "importing", "deleting"})

ModelAssetsRunTask = Callable[[Callable[[], Awaitable[None]]], object]


def _status_label(entry: object) -> str:
    status = str(getattr(entry, "status", "missing"))
    percent = getattr(entry, "progress_percent", None) or 0
    if status in {"downloading", "importing"}:
        return t(f"model_assets.status.{status}", percent=percent)
    return t(f"model_assets.status.{status}")


def _size_label(entry: object) -> str:
    size = int(getattr(entry, "size_bytes", 0) or 0)
    return f"{size / 1024**3:.1f} GB"


def _failure_label(entry: object) -> str | None:
    code = getattr(entry, "failure_code", None)
    if not code:
        return None
    return t("model_assets.error." + (code if code in _ERROR_KEYS else "download_failed"))


class ModelAssetsDialog:
    """Two-column model manager: models on the left, actions for the selection on the right."""

    def __init__(
        self,
        page: ft.Page,
        application,
        run_task: ModelAssetsRunTask,
        *,
        picker_factory: Callable[[], object] = ft.FilePicker,
    ) -> None:
        self.page = page
        self.application = application
        self.run_task = run_task
        self._picker_factory = picker_factory
        self._picker = None
        self._selected: str | None = None
        self._confirm_delete: str | None = None
        self._is_open = False
        self._dialog: ft.AlertDialog | None = None
        self._content: ft.Container | None = None
        self._model_list: ft.Column | None = None
        self._cards: dict[str, ModalCard] = {}
        self._actions = ft.Column(
            spacing=CARD_SPACING,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        self._failure = ft.Text(
            "",
            size=_NOTICE_TEXT_SIZE,
            weight=ft.FontWeight.BOLD,
            color=COLOR_WARNING,
            text_align=ft.TextAlign.CENTER,
            visible=False,
        )
        self._progress_text: ft.Text | None = None
        self._action_signature: tuple | None = None

    @property
    def dialog(self) -> ft.AlertDialog | None:
        return self._dialog

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def selected_model_id(self) -> str | None:
        return self._selected

    def open(self) -> None:
        if self._is_open:
            return
        entries = self.application.model_asset_entries()
        if self._selected is None and entries:
            self._selected = entries[0].model_id
        self._confirm_delete = None
        self._action_signature = None
        self._dialog = self._build_dialog(entries)
        self._is_open = True
        self.render()
        self.page.show_dialog(self._dialog)
        self.run_task(self._refresh_and_watch)

    def close(self) -> None:
        if not self._is_open:
            return
        self._is_open = False
        self.page.pop_dialog()

    def select(self, model_id: str) -> None:
        if self._selected != model_id:
            self._confirm_delete = None
        self._selected = model_id
        self.render()

    def render(self) -> None:
        entries = {entry.model_id: entry for entry in self.application.model_asset_entries()}
        busy = any(entry.status in _BUSY_STATUSES for entry in entries.values())
        for model_id, card in self._cards.items():
            entry = entries.get(model_id)
            if entry is None:
                continue
            card.set_detail(f"{_size_label(entry)} ({_status_label(entry)})")
            card.set_state(selected=model_id == self._selected)
        entry = entries.get(self._selected) if self._selected else None
        if entry is not None:
            self._render_actions(entry, busy)
        if self._content is not None:
            update_control_if_mounted(self._content)

    def _render_actions(self, entry: object, busy: bool) -> None:
        status = entry.status
        failure = _failure_label(entry)
        self._failure.value = failure or ""
        self._failure.visible = failure is not None and status not in _BUSY_STATUSES
        signature = (
            entry.model_id,
            status,
            busy,
            self._confirm_delete == entry.model_id,
        )
        if signature == self._action_signature:
            if self._progress_text is not None:
                self._progress_text.value = _status_label(entry)
            return
        self._action_signature = signature
        self._progress_text = None
        controls: list[ft.Control] = []
        if status in _BUSY_STATUSES:
            self._progress_text = ft.Text(
                _status_label(entry),
                size=CARD_LABEL_SIZE,
                weight=ft.FontWeight.BOLD,
                color=COLOR_PRIMARY,
                text_align=ft.TextAlign.CENTER,
                max_lines=1,
            )
            controls.append(
                ft.Container(
                    content=self._progress_text,
                    height=CARD_HEIGHT,
                    padding=ft.Padding.symmetric(horizontal=CARD_PADDING),
                    alignment=ft.Alignment.CENTER,
                    bgcolor=COLOR_BACKGROUND,
                    border_radius=CARD_RADIUS,
                )
            )
            if status != "deleting":
                controls.append(
                    ModalCard(t("model_assets.action.cancel"), on_click=self._on_cancel)
                )
        else:
            enabled = not busy
            controls.append(
                ModalCard(
                    t(
                        "model_assets.action.redownload"
                        if status == "installed"
                        else "model_assets.action.download"
                    ),
                    on_click=self._on_download,
                    enabled=enabled,
                )
            )
            if entry.importable:
                controls.append(
                    ModalCard(
                        t("model_assets.action.import"),
                        on_click=self._on_import,
                        enabled=enabled,
                    )
                )
            if status in {"installed", "invalid"}:
                confirming = self._confirm_delete == entry.model_id
                controls.append(
                    ModalCard(
                        t(
                            "model_assets.action.delete_confirm"
                            if confirming
                            else "model_assets.action.delete"
                        ),
                        on_click=self._on_delete,
                        selected=confirming,
                        enabled=enabled,
                    )
                )
        self._actions.controls = controls

    def _build_dialog(self, entries) -> ft.AlertDialog:
        self._cards = {}
        items: list[ft.Control] = []
        for purpose in ("asr", "translation"):
            items.append(build_section_header(t("model_assets.group." + purpose)))
            for entry in (item for item in entries if item.purpose == purpose):
                card = ModalCard(
                    entry.name,
                    on_click=lambda model_id=entry.model_id: self.select(model_id),
                    detail="",
                )
                card.key = ft.ScrollKey(entry.model_id)
                self._cards[entry.model_id] = card
                items.append(card)
        model_list = ft.Column(
            controls=[
                ft.Container(
                    content=ft.Column(
                        controls=items,
                        spacing=CARD_SPACING,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                    padding=ft.Padding.only(right=8, bottom=12),
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        self._model_list = model_list
        actions = ft.Column(
            controls=[build_section_header(" "), self._actions, self._failure],
            spacing=CARD_SPACING,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        self._content = build_modal_surface(
            t("settings.model_assets"),
            ft.Row(
                controls=[model_list, actions],
                spacing=MODAL_COLUMN_SPACING,
                vertical_alignment=ft.CrossAxisAlignment.START,
                expand=True,
            ),
        )
        return build_modal_dialog(
            self._content,
            semantics_label=t("settings.model_assets"),
            on_dismiss=self._handle_dismiss,
        )

    def _handle_dismiss(self, _event) -> None:  # noqa: ANN001
        self._is_open = False

    async def _refresh_and_watch(self) -> None:
        await self._reveal_selected()
        try:
            await self.application.refresh_model_assets()
        except Exception:
            logger.debug("Model asset refresh failed", exc_info=True)
        while self._is_open:
            self.render()
            await asyncio.sleep(RENDER_INTERVAL_S)

    async def _reveal_selected(self) -> None:
        if self._model_list is None or self._selected not in self._cards:
            return
        try:
            await self._model_list.scroll_to(scroll_key=ft.ScrollKey(self._selected))
        except Exception:
            logger.debug("Model list scroll failed", exc_info=True)

    def _on_download(self) -> None:
        model_id = self._selected
        if model_id is None:
            return
        self._confirm_delete = None
        self._run(lambda: self.application.download_model_asset(model_id))

    def _on_cancel(self) -> None:
        model_id = self._selected
        if model_id is not None:
            self._run(lambda: self.application.cancel_model_asset(model_id))

    def _on_delete(self) -> None:
        model_id = self._selected
        if model_id is None:
            return
        if self._confirm_delete != model_id:
            self._confirm_delete = model_id
            self.render()
            return
        self._confirm_delete = None
        self._run(lambda: self.application.delete_model_asset(model_id))

    def _on_import(self) -> None:
        model_id = self._selected
        if model_id is None:
            return
        self._confirm_delete = None

        async def choose() -> None:
            if self._picker is None:
                self._picker = self._picker_factory()
            directory = await self._picker.get_directory_path()
            if directory:
                await self.application.import_model_asset(model_id, Path(directory))

        self._run(choose)

    def _run(self, operation: Callable[[], Awaitable[object]]) -> None:
        async def task() -> None:
            try:
                await operation()
            except Exception:
                logger.debug("Model asset operation ended with a failure", exc_info=True)
            finally:
                if self._is_open:
                    self.render()

        self.run_task(task)
        self.render()


__all__ = ["ModelAssetsDialog"]
