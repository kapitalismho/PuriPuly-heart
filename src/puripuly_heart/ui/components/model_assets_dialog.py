from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import flet as ft

from puripuly_heart.ui.flet_runtime import is_hover_active, update_control_if_mounted
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_NEUTRAL_DARK,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    COLOR_WARNING,
)

logger = logging.getLogger(__name__)

RENDER_INTERVAL_S = 0.25
_MODEL_CARD_HEIGHT = 84
_ACTION_CARD_HEIGHT = 96
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
        self._cards: dict[str, tuple[ft.Container, ft.Text, ft.Text]] = {}
        self._actions = ft.Column(spacing=12, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
        self._description = ft.Text("", size=16, color=COLOR_NEUTRAL_DARK)
        self._failure = ft.Text("", size=14, color=COLOR_WARNING, visible=False)
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
        for model_id, (card, name, meta) in self._cards.items():
            entry = entries.get(model_id)
            if entry is None:
                continue
            selected = model_id == self._selected
            card.bgcolor = COLOR_PRIMARY if selected else COLOR_BACKGROUND
            name.color = ft.Colors.WHITE if selected else COLOR_ON_BACKGROUND
            meta.color = ft.Colors.WHITE if selected else COLOR_NEUTRAL_DARK
            meta.value = f"{_size_label(entry)} · {_status_label(entry)}"
        entry = entries.get(self._selected) if self._selected else None
        if entry is not None:
            self._render_actions(entry, busy)
        if self._content is not None:
            update_control_if_mounted(self._content)

    def _render_actions(self, entry: object, busy: bool) -> None:
        status = entry.status
        failure = _failure_label(entry)
        self._description.value = t(entry.description_key)
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
                size=22,
                weight=ft.FontWeight.BOLD,
                color=COLOR_PRIMARY,
                text_align=ft.TextAlign.CENTER,
            )
            controls.append(
                ft.Container(
                    content=self._progress_text,
                    height=_ACTION_CARD_HEIGHT,
                    alignment=ft.Alignment.CENTER,
                    bgcolor=COLOR_BACKGROUND,
                    border_radius=16,
                )
            )
            if status != "deleting":
                controls.append(self._action_card(t("model_assets.action.cancel"), self._on_cancel))
        else:
            enabled = not busy
            controls.append(
                self._action_card(
                    t(
                        "model_assets.action.redownload"
                        if status == "installed"
                        else "model_assets.action.download"
                    ),
                    self._on_download,
                    enabled=enabled,
                )
            )
            if entry.importable:
                controls.append(
                    self._action_card(
                        t("model_assets.action.import"),
                        self._on_import,
                        enabled=enabled,
                        detail=t(
                            "model_assets.import_files",
                            files=", ".join(entry.required_files),
                        ),
                    )
                )
            if status in {"installed", "invalid"}:
                confirming = self._confirm_delete == entry.model_id
                controls.append(
                    self._action_card(
                        t(
                            "model_assets.action.delete_confirm"
                            if confirming
                            else "model_assets.action.delete"
                        ),
                        self._on_delete,
                        enabled=enabled,
                        emphasis=confirming,
                    )
                )
            if busy:
                controls.append(
                    ft.Text(t("model_assets.busy_other"), size=14, color=COLOR_NEUTRAL_DARK)
                )
        self._actions.controls = controls

    def _action_card(
        self,
        label: str,
        handler: Callable[[], None],
        *,
        enabled: bool = True,
        detail: str | None = None,
        emphasis: bool = False,
    ) -> ft.Container:
        text = ft.Text(
            label,
            size=20,
            weight=ft.FontWeight.BOLD,
            color=(
                ft.Colors.WHITE
                if emphasis
                else COLOR_ON_BACKGROUND if enabled else COLOR_NEUTRAL_DARK
            ),
            text_align=ft.TextAlign.CENTER,
        )
        content: ft.Control = text
        if detail is not None:
            content = ft.Column(
                controls=[
                    text,
                    ft.Text(
                        detail,
                        size=12,
                        color=ft.Colors.WHITE if emphasis else COLOR_NEUTRAL_DARK,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                spacing=4,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
        return ft.Container(
            content=content,
            height=_ACTION_CARD_HEIGHT,
            padding=ft.Padding.symmetric(horizontal=16, vertical=8),
            alignment=ft.Alignment.CENTER,
            bgcolor=COLOR_PRIMARY if emphasis else COLOR_BACKGROUND,
            border_radius=16,
            opacity=1.0 if enabled else 0.5,
            disabled=not enabled,
            on_click=lambda _event: handler(),
            on_hover=None if emphasis else lambda event: self._on_hover(event, text),
            animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
        )

    @staticmethod
    def _on_hover(event, text: ft.Text) -> None:
        text.color = COLOR_PRIMARY if is_hover_active(event) else COLOR_ON_BACKGROUND
        update_control_if_mounted(event.control)

    def _build_dialog(self, entries) -> ft.AlertDialog:
        self._cards = {}
        model_list = ft.ListView(spacing=10, expand=True)
        for purpose in ("asr", "translation"):
            model_list.controls.append(
                ft.Text(
                    t("model_assets.group." + purpose),
                    size=18,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_SECONDARY,
                )
            )
            for entry in (item for item in entries if item.purpose == purpose):
                name = ft.Text(entry.name, size=18, weight=ft.FontWeight.BOLD)
                meta = ft.Text("", size=14)
                card = ft.Container(
                    content=ft.Column(
                        controls=[name, meta],
                        spacing=4,
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    height=_MODEL_CARD_HEIGHT,
                    padding=ft.Padding.symmetric(horizontal=20, vertical=10),
                    border_radius=16,
                    bgcolor=COLOR_BACKGROUND,
                    on_click=lambda _event, model_id=entry.model_id: self.select(model_id),
                    animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
                )
                self._cards[entry.model_id] = (card, name, meta)
                model_list.controls.append(card)
        actions = ft.Column(
            controls=[self._description, self._failure, self._actions],
            spacing=16,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        self._content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        t("settings.model_assets"),
                        size=24,
                        weight=ft.FontWeight.BOLD,
                        color=COLOR_SECONDARY,
                    ),
                    ft.Container(height=16),
                    ft.Row(
                        controls=[model_list, actions],
                        spacing=32,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                        expand=True,
                    ),
                ],
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            width=880,
            height=700,
            padding=ft.Padding.symmetric(horizontal=32, vertical=32),
            bgcolor=COLOR_SURFACE,
            border_radius=28,
        )
        return ft.AlertDialog(
            modal=False,
            content=self._content,
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
            semantics_label=t("settings.model_assets"),
            on_dismiss=self._handle_dismiss,
        )

    def _handle_dismiss(self, _event) -> None:  # noqa: ANN001
        self._is_open = False

    async def _refresh_and_watch(self) -> None:
        try:
            await self.application.refresh_model_assets()
        except Exception:
            logger.debug("Model asset refresh failed", exc_info=True)
        while self._is_open:
            self.render()
            await asyncio.sleep(RENDER_INTERVAL_S)

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
