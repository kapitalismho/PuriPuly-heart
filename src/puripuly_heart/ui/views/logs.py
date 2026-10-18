"""System logs view with real-time log display and folder access."""

import asyncio
import inspect
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import flet as ft

from puripuly_heart.domain.events import SelfTurnPresentation
from puripuly_heart.ui.flet_runtime import control_page, update_control_if_mounted
from puripuly_heart.ui.fonts import (
    LOG_TYPEFACE,
    TypeFace,
    font_for_language,
    locale_body_letter_spacing,
)
from puripuly_heart.ui.i18n import get_locale, language_name, source_label, t
from puripuly_heart.ui.logs.contract import LogsSurfaceSlots
from puripuly_heart.ui.logs.renderer import compose_logs_surface
from puripuly_heart.ui.theme import (
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    text_style,
)

MAX_LOG_ENTRIES = 4000
CLEANUP_BATCH = 500
MAX_CONVERSATION_RECORDS = 1000
_UPDATE_INTERVAL = 0.2  # 200ms throttling


def _get_log_dir() -> Path:
    """Get the directory where log files are stored."""
    from puripuly_heart.config.paths import user_config_dir

    return user_config_dir()


def _format_conversation_timestamp(origin_wall_clock_ms: int | None) -> str:
    timestamp_s = origin_wall_clock_ms / 1000.0 if origin_wall_clock_ms is not None else time.time()
    return datetime.fromtimestamp(timestamp_s).strftime("%H:%M:%S")


class LiveLogViewModel:
    """Keeps the merged live-log buffer shared by app and overlay sources."""

    def __init__(
        self,
        *,
        max_entries: int = MAX_LOG_ENTRIES,
        cleanup_batch: int = CLEANUP_BATCH,
    ) -> None:
        self._max_entries = max_entries
        self._cleanup_batch = cleanup_batch
        self._visible_lines: list[str] = []
        self._cleanup_count = 0

    @property
    def visible_lines(self) -> list[str]:
        return self._visible_lines

    @property
    def cleanup_count(self) -> int:
        return self._cleanup_count

    def append(self, record: str) -> None:
        self._visible_lines.append(record)
        if len(self._visible_lines) > self._max_entries + self._cleanup_batch:
            del self._visible_lines[: self._cleanup_batch]
            self._cleanup_count += 1


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    timestamp_label: str
    source: str
    channel: str
    source_text: str | None
    translated_text: str | None
    utterance_id: str = "legacy"
    source_language: str | None = None
    target_language: str | None = None
    target_index: int | None = None
    disposition: str = "translated"
    turn_kind: str | None = None
    turn_generation: int | None = None
    presentation_revision: int = 0
    source_revision: int = 0
    translation_source_revision: int | None = None
    turn_status: str | None = None
    accepted: bool = True
    parent_utterance_id: str | None = None


class ConversationViewModel:
    """Keeps session-only original/translation pairs for the Logs conversation view."""

    def __init__(
        self,
        *,
        max_entries: int = MAX_CONVERSATION_RECORDS,
        cleanup_batch: int = CLEANUP_BATCH,
    ) -> None:
        self._max_entries = max_entries
        self._cleanup_batch = cleanup_batch
        self._records: list[ConversationRecord] = []
        self._cleanup_count = 0

    @property
    def records(self) -> list[ConversationRecord]:
        return self._records

    @property
    def cleanup_count(self) -> int:
        return self._cleanup_count

    def append(self, record: ConversationRecord) -> None:
        if record.utterance_id != "legacy":
            parent_id = record.parent_utterance_id or record.utterance_id
            matches = [
                index
                for index, previous in enumerate(self._records)
                if previous.channel == record.channel
                and (previous.parent_utterance_id or previous.utterance_id) == parent_id
                and previous.turn_generation is not None
                and (
                    (record.target_index is None and record.target_language is None)
                    or (
                        record.target_index is not None
                        and previous.target_index == record.target_index
                    )
                    or (
                        record.target_index is None
                        and previous.target_language == record.target_language
                    )
                )
            ]
            if matches:
                for index in matches:
                    previous = self._records[index]
                    self._records[index] = replace(
                        previous,
                        accepted=True,
                        source_text=record.source_text or previous.source_text,
                        translated_text=record.translated_text or previous.translated_text,
                    )
                return
        self._records.append(record)
        if len(self._records) > self._max_entries + self._cleanup_batch:
            del self._records[: self._cleanup_batch]
            self._cleanup_count += 1

    def update_self_turn(self, snapshot: SelfTurnPresentation, *, source: str) -> None:
        utterance_id = str(snapshot.utterance_id)
        existing = [
            record
            for record in self._records
            if record.channel == "self"
            and (record.parent_utterance_id or record.utterance_id) == utterance_id
        ]
        managed = next((record for record in existing if record.turn_generation is not None), None)
        if managed is not None and (
            snapshot.turn_generation != managed.turn_generation
            or snapshot.revision <= managed.presentation_revision
            or snapshot.source_revision < managed.source_revision
            or (managed.turn_status != "active" and snapshot.status != managed.turn_status)
        ):
            return
        timestamp = (
            existing[0].timestamp_label if existing else _format_conversation_timestamp(None)
        )
        replacements = []
        for index, target in enumerate(snapshot.targets or (None,)):
            language = target.target_language if target is not None else None
            previous = next(
                (
                    record
                    for record in existing
                    if (target is not None and record.target_index == index)
                    or record.target_language == language
                    or (
                        target is not None
                        and target.status == "source_only"
                        and record.target_language is None
                        and record.target_index is None
                    )
                ),
                None,
            )
            translation = target.translation if target is not None else None
            replacements.append(
                ConversationRecord(
                    timestamp_label=timestamp,
                    source=source,
                    channel="self",
                    utterance_id=utterance_id,
                    source_text=snapshot.source_text,
                    translated_text=translation.text if translation is not None else None,
                    source_language=snapshot.source_language,
                    target_language=language,
                    target_index=index if target is not None else None,
                    disposition=target.status if target is not None else "source_only",
                    turn_kind="speech",
                    turn_generation=snapshot.turn_generation,
                    presentation_revision=snapshot.revision,
                    source_revision=snapshot.source_revision,
                    translation_source_revision=(
                        target.source_revision if translation is not None else None
                    ),
                    turn_status=snapshot.status,
                    accepted=previous.accepted if previous is not None else False,
                    parent_utterance_id=utterance_id,
                )
            )
        indices = [
            index
            for index, record in enumerate(self._records)
            if record.channel == "self"
            and (record.parent_utterance_id or record.utterance_id) == utterance_id
        ]
        insertion = indices[0] if indices else len(self._records)
        for index in reversed(indices):
            del self._records[index]
        self._records[insertion:insertion] = replacements
        if len(self._records) > self._max_entries + self._cleanup_batch:
            del self._records[: self._cleanup_batch]
            self._cleanup_count += 1

    def render(self) -> str:
        if not self._records:
            return t("logs.conversation.empty")

        rendered: list[str] = []
        for record in self._records:
            header = f"[{record.timestamp_label}] {source_label(record.source)}"
            if record.target_language and (
                record.translated_text
                or (
                    record.turn_status is not None
                    and record.target_index is not None
                    and record.disposition != "source_only"
                )
            ):
                header = f"{header} → {language_name(record.target_language)}"
            elif record.source_text and record.source_language:
                header = f"{header} · {language_name(record.source_language)}"
            body = [record.source_text] if record.source_text else []
            if record.turn_status is not None:
                status = record.turn_status
                if status == "active":
                    status = "preview"
                elif (
                    status == "failed"
                    and record.disposition == "ready"
                    and record.translated_text
                    and record.translation_source_revision == record.source_revision
                ):
                    status = "complete"
                header = f"{header} · {t(f'display.turn.{status}')}"
                if record.disposition != "ready":
                    body.append(t(f"logs.conversation.target.{record.disposition}"))
            if record.translated_text:
                if (
                    record.translation_source_revision is not None
                    and record.translation_source_revision != record.source_revision
                ):
                    body.append(
                        t(
                            "logs.conversation.earlier_source",
                            revision=record.translation_source_revision,
                            current_revision=record.source_revision,
                        )
                    )
                body.append(record.translated_text)
            rendered.append("\n".join((header, *body)))
        return "\n\n".join(rendered)


class _LogListProxy:
    """Compatibility proxy for tests expecting a list-style log view."""

    def __init__(self, view: "LogsView") -> None:
        self._view = view

    @property
    def controls(self) -> list[ft.Text]:
        return [ft.Text(entry) for entry in self._view._model.visible_lines]


class LogsView(ft.Column):
    """System logs view with VR-optimized display and folder access."""

    def __init__(self):
        super().__init__(expand=True, spacing=16)

        self._title_text: ft.Text | None = None
        self._log_text: ft.Text | None = None
        self._log_scroll: ft.Column | None = None
        self._folder_button: ft.TextButton | None = None
        self._header_button_row: ft.Row | None = None
        self._conversation_button: ft.TextButton | None = None
        self._showing_conversation = False
        self._conversation_model = ConversationViewModel()

        # Log buffer and throttling state
        self._model = LiveLogViewModel()
        self._log_buffer = self._model.visible_lines
        self._last_update: float = 0.0
        self._pending_update: bool = False
        self._trailing_flush: asyncio.TimerHandle | None = None
        self._rendered_line_count: int = 0
        self._last_cleanup_count: int = 0
        self.log_list = _LogListProxy(self)

        self._build_ui()

    def _get_button_style(self, font_family: str) -> ft.ButtonStyle:
        """Create a complete ButtonStyle with the specified font."""
        return ft.ButtonStyle(
            color={
                ft.ControlState.HOVERED: COLOR_PRIMARY,
                ft.ControlState.DEFAULT: COLOR_SECONDARY,
            },
            icon_color={
                ft.ControlState.HOVERED: COLOR_PRIMARY,
                ft.ControlState.DEFAULT: COLOR_SECONDARY,
            },
            text_style=text_style(
                TypeFace(font_family, locale_body_letter_spacing(get_locale())),
                size=20,
            ),
            overlay_color=ft.Colors.TRANSPARENT,
            animation_duration=0,
        )

    def _build_ui(self):
        """Build the logs view UI."""
        font_family = font_for_language(get_locale())

        # Title (styled like About page section headers)
        self._title_text = ft.Text(
            t("logs.title"),
            size=28,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )

        # Folder open button (brown, hover -> primary)
        self._folder_button = ft.TextButton(
            content=t("logs.open_folder"),
            icon=ft.Icons.FOLDER_OPEN,
            style=self._get_button_style(font_family),
            on_click=self._open_log_folder,
        )
        self._conversation_button = ft.TextButton(
            content=self._conversation_button_label(),
            icon=ft.Icons.CHAT_BUBBLE_OUTLINE,
            style=self._get_button_style(font_family),
            on_click=self._on_conversation_button_click,
        )

        # Single selectable text for all logs (enables multi-line drag selection)
        self._log_text = ft.Text(
            "",
            size=16,
            font_family=LOG_TYPEFACE.font_family,
            color=COLOR_ON_BACKGROUND,
            style=text_style(LOG_TYPEFACE),
            selectable=True,
        )

        # Header rows
        regions = compose_logs_surface(
            LogsSurfaceSlots(
                title=self._title_text,
                folder_button=self._folder_button,
                conversation_button=self._conversation_button,
                log_text=self._log_text,
            )
        )
        self._header_button_row = regions.header_button_row
        self._log_scroll = regions.log_scroll
        self.controls = [regions.root]

    def append_log(self, record: str):
        """Append a log entry with throttled updates."""
        self._model.append(record)

        # Throttled update
        now = time.time()
        if now - self._last_update >= _UPDATE_INTERVAL:
            self._flush_logs()
        else:
            self._pending_update = True
            self._schedule_trailing_flush()

    def append_log_threadsafe(self, record: str) -> None:
        """Append a log entry from any thread without mutating Flet state off-loop."""
        page = control_page(self)
        loop = getattr(page, "loop", None) if page is not None else None
        if loop is None:
            self.append_log(record)
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            self.append_log(record)
            return

        if self._schedule_log_append(record):
            return
        self._buffer_pending_log(record)

    @property
    def conversation_records(self) -> tuple[ConversationRecord, ...]:
        return tuple(self._conversation_model.records)

    def append_conversation_record(
        self,
        *,
        source: str,
        channel: str,
        utterance_id: str = "legacy",
        source_text: str | None,
        translated_text: str | None,
        source_language: str | None = None,
        target_language: str | None = None,
        target_index: int | None = None,
        disposition: str = "translated",
        origin_wall_clock_ms: int | None = None,
        turn_kind: str | None = None,
        parent_utterance_id: str | None = None,
    ) -> None:
        cleaned_source = source_text.strip() if isinstance(source_text, str) else None
        cleaned_translation = translated_text.strip() if isinstance(translated_text, str) else None
        if utterance_id == "legacy" and (not cleaned_source or not cleaned_translation):
            return
        if not cleaned_source and not cleaned_translation:
            return

        self._conversation_model.append(
            ConversationRecord(
                timestamp_label=_format_conversation_timestamp(origin_wall_clock_ms),
                source=source.strip() or ("Listen" if channel == "peer" else "Mic"),
                channel=channel,
                utterance_id=utterance_id,
                source_text=cleaned_source or None,
                translated_text=cleaned_translation or None,
                source_language=source_language,
                target_language=target_language,
                target_index=target_index,
                disposition=disposition,
                turn_kind=turn_kind,
                parent_utterance_id=parent_utterance_id,
            )
        )
        if self._showing_conversation:
            self._render_conversation_text()
            if self._log_text is not None:
                update_control_if_mounted(self._log_text)

    def update_self_turn(self, snapshot: SelfTurnPresentation, *, source: str) -> None:
        self._conversation_model.update_self_turn(snapshot, source=source)
        if self._showing_conversation:
            self._render_conversation_text()
            if self._log_text is not None:
                update_control_if_mounted(self._log_text)

    def _schedule_log_append(self, record: str) -> bool:
        page = control_page(self)
        loop = getattr(page, "loop", None) if page is not None else None
        if loop is None:
            return False

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            return False

        coroutine = self._append_log_on_page_loop(record)
        try:
            asyncio.run_coroutine_threadsafe(coroutine, loop)
        except RuntimeError:
            coroutine.close()
            return False
        return True

    async def _append_log_on_page_loop(self, record: str) -> None:
        self.append_log(record)

    def _buffer_pending_log(self, record: str) -> None:
        self._model.append(record)
        self._pending_update = True

    def _schedule_trailing_flush(self) -> None:
        if self._trailing_flush is not None and not self._trailing_flush.cancelled():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        delay = max(0.0, _UPDATE_INTERVAL - (time.time() - self._last_update))
        self._trailing_flush = loop.call_later(delay, self._flush_logs)

    def _cancel_trailing_flush(self) -> None:
        handle = self._trailing_flush
        self._trailing_flush = None
        if handle is not None:
            handle.cancel()

    def will_unmount(self) -> None:
        self._cancel_trailing_flush()

    def _flush_logs(self):
        """Flush pending logs to the UI."""
        self._cancel_trailing_flush()
        if self._log_text is None:
            return

        if self._showing_conversation:
            self._rendered_line_count = len(self._log_buffer)
            self._last_cleanup_count = self._model.cleanup_count
            self._last_update = time.time()
            self._pending_update = False
            return

        cleanup_changed = self._model.cleanup_count != self._last_cleanup_count
        rendered_ahead = self._rendered_line_count > len(self._log_buffer)

        if cleanup_changed or rendered_ahead:
            self._rebuild_visible_text()
        else:
            new_lines = self._log_buffer[self._rendered_line_count :]
            self._append_visible_text(new_lines)

        self._last_update = time.time()
        self._pending_update = False

        update_control_if_mounted(self._log_text)

    def _rebuild_visible_text(self) -> None:
        assert self._log_text is not None
        self._log_text.value = "\n".join(self._log_buffer)
        self._rendered_line_count = len(self._log_buffer)
        self._last_cleanup_count = self._model.cleanup_count

    def _append_visible_text(self, new_lines: list[str]) -> None:
        assert self._log_text is not None
        if not new_lines:
            return
        addition = "\n".join(new_lines)
        if self._log_text.value:
            self._log_text.value = f"{self._log_text.value}\n{addition}"
        else:
            self._log_text.value = addition
        self._rendered_line_count = len(self._log_buffer)
        self._last_cleanup_count = self._model.cleanup_count

    def _render_conversation_text(self) -> None:
        assert self._log_text is not None
        self._log_text.value = self._conversation_model.render()

    def _conversation_button_label(self) -> str:
        key = "logs.conversation.hide" if self._showing_conversation else "logs.conversation.show"
        return t(key)

    def _on_conversation_button_click(self, _e: ft.ControlEvent | object) -> None:
        self._showing_conversation = not self._showing_conversation
        if self._conversation_button is not None:
            self._conversation_button.content = self._conversation_button_label()
        if self._showing_conversation:
            self._render_conversation_text()
        else:
            self._rebuild_visible_text()
        update_control_if_mounted(self)

    def apply_locale(self) -> None:
        """Refresh UI text when locale changes."""
        font_family = font_for_language(get_locale())
        if self._title_text:
            self._title_text.value = t("logs.title")
        if self._folder_button:
            self._folder_button.content = t("logs.open_folder")
            self._folder_button.style = self._get_button_style(font_family)
        if self._conversation_button:
            self._conversation_button.content = self._conversation_button_label()
            self._conversation_button.style = self._get_button_style(font_family)
        if self._showing_conversation and self._log_text is not None:
            self._render_conversation_text()
        # Only update if added to page
        update_control_if_mounted(self)

    async def scroll_to_bottom(self) -> None:
        """Scroll to the latest log entry."""
        if self._log_scroll and control_page(self) is not None:
            if self._pending_update:
                self._flush_logs()
            result = self._log_scroll.scroll_to(offset=-1, duration=0)
            if inspect.isawaitable(result):
                await result

    def _open_log_folder(self, _):
        """Open the log folder in the system file explorer."""
        log_dir = _get_log_dir()
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(log_dir)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(log_dir)])
        else:
            subprocess.Popen(["xdg-open", str(log_dir)])
