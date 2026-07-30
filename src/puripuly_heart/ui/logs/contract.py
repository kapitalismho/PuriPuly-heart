from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import flet as ft


@dataclass(frozen=True, slots=True)
class LogsIntents:
    runtime_logging_mode_change: Callable[[str], None]


class LogsIntentConsumer(Protocol):
    def bind_logs_intents(self, intents: LogsIntents) -> None: ...


class LogsStateSink(Protocol):
    @property
    def runtime_logging_mode(self) -> str: ...

    def set_runtime_logging_mode(self, mode: str) -> None: ...

    def attach_log_handler(self) -> None: ...

    def append_log(self, record: str) -> None: ...

    def append_conversation_record(
        self,
        *,
        source: str,
        channel: str,
        source_text: str,
        translated_text: str,
        origin_wall_clock_ms: int | None = None,
    ) -> None: ...

    def apply_locale(self) -> None: ...


@dataclass(frozen=True, slots=True)
class LogsSurfaceSlots:
    title: ft.Control
    folder_button: ft.Control
    mode_button: ft.Control
    conversation_button: ft.Control
    log_text: ft.Control


@dataclass(frozen=True, slots=True)
class LogsSurfaceRegions:
    root: ft.Control
    card: ft.Container
    header: ft.Container
    header_button_row: ft.Row
    log_scroll: ft.Column


__all__ = [
    "LogsIntentConsumer",
    "LogsIntents",
    "LogsStateSink",
    "LogsSurfaceRegions",
    "LogsSurfaceSlots",
]
