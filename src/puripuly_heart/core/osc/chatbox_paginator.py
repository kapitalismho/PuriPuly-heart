from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field

from puripuly_heart.core.clock import Clock
from puripuly_heart.core.osc.sender import OscSender
from puripuly_heart.core.runtime_logging import SessionRuntimeLoggingService
from puripuly_heart.domain.models import OSCMessage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChatboxPaginator:
    sender: OscSender
    clock: Clock
    max_chars: int = 144
    page_interval_s: float = 3.0
    runtime_logging: SessionRuntimeLoggingService | None = None
    _pending_pages: list[str] | None = None
    _pending_messages: list[OSCMessage] | None = None
    _typing_reasons: set[str] = field(init=False, default_factory=set)
    _next_page_at: float = 0.0
    _typing_visible: bool = False

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        if self.page_interval_s <= 0:
            raise ValueError("page_interval_s must be > 0")
        self._pending_pages = []
        self._pending_messages = []

    def enqueue(self, message: OSCMessage) -> None:
        if self._is_paginating():
            self._pending_messages.append(message)
            return
        self._start_message(message)

    def process_due(self) -> None:
        if not self._is_paginating():
            self._drain_pending_messages()
            return

        now = self.clock.now()
        if now < self._next_page_at:
            return

        page = self._pending_pages.pop(0)
        remaining_parts = len(self._pending_pages)
        self._send_page(mode="queued", text=page, remaining_parts=remaining_parts)

        if self._pending_pages:
            self._next_page_at = now + self.page_interval_s
            return

        self._next_page_at = 0.0
        self._drain_pending_messages()

    def send_immediate(self, text: str) -> bool:
        """Send a single chatbox packet immediately without changing pagination state."""
        text = text.strip()
        if not text:
            return False
        return self._send_page(mode="immediate", text=text, remaining_parts=0)

    def send_typing(self, is_typing: bool) -> None:
        if self._send_typing_state(is_typing):
            self._typing_visible = is_typing

    def set_typing_reason(self, reason: str, active: bool) -> None:
        reason = reason.strip()
        if not reason:
            raise ValueError("reason must be non-empty")

        if active:
            self._typing_reasons.add(reason)
        else:
            self._typing_reasons.discard(reason)
        self._sync_typing_visibility()

    def clear_typing_reasons(self) -> None:
        self._typing_reasons.clear()
        self._sync_typing_visibility()

    def _sync_typing_visibility(self) -> None:
        visible = bool(self._typing_reasons)
        if visible == self._typing_visible:
            return
        if self._send_typing_state(visible):
            self._typing_visible = visible

    def _send_typing_state(self, is_typing: bool) -> bool:
        try:
            self.sender.send_typing(is_typing)
        except OSError as exc:
            self._emit_basic(
                f"[Basic][OSC] typing status=failed error={exc}", level=logging.WARNING
            )
            return False
        return True

    def _is_paginating(self) -> bool:
        return bool(self._pending_pages)

    def _start_message(self, message: OSCMessage) -> None:
        text = message.text.strip()
        if not text:
            return

        parts = self._split_text(text)
        head = parts[0]
        tail = parts[1:]
        self._send_page(mode="queued", text=head, remaining_parts=len(tail))

        if tail:
            self._pending_pages.extend(tail)
            self._next_page_at = self.clock.now() + self.page_interval_s

    def _drain_pending_messages(self) -> None:
        while self._pending_messages and not self._is_paginating():
            next_message = self._pending_messages.pop(0)
            self._start_message(next_message)

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        return textwrap.wrap(
            text,
            width=self.max_chars,
            break_long_words=True,
            break_on_hyphens=False,
        )

    def _send_page(self, *, mode: str, text: str, remaining_parts: int) -> bool:
        self._emit_send_attempt(mode=mode, text=text, remaining_parts=remaining_parts)
        try:
            self.sender.send_chatbox(text)
        except OSError as exc:
            self._emit_send_failure(mode=mode, exc=exc)
            return False
        self._emit_send_delivered(mode=mode, text=text, remaining_parts=remaining_parts)
        return True

    def _emit_send_attempt(self, *, mode: str, text: str, remaining_parts: int) -> None:
        self._emit_detailed(
            f"[Detailed][OSC] send mode={mode} status=attempt chars={len(text)} "
            f"remaining_parts={remaining_parts} text={text!r}"
        )

    def _emit_send_delivered(self, *, mode: str, text: str, remaining_parts: int) -> None:
        self._emit_basic(
            f"[Basic][OSC] send mode={mode} status=delivered chars={len(text)} "
            f"remaining_parts={remaining_parts}"
        )

    def _emit_send_failure(self, *, mode: str, exc: OSError) -> None:
        self._emit_basic(
            f"[Basic][OSC] send mode={mode} status=failed error={exc}",
            level=logging.WARNING,
        )

    def _emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        if self.runtime_logging is not None:
            self.runtime_logging.emit_basic(message, level=level)
            return
        logger.log(level, message)

    def _emit_detailed(self, message: str, *, level: int = logging.INFO) -> None:
        if self.runtime_logging is not None:
            self.runtime_logging.emit_detailed(message, level=level)
            return
        logger.debug(message)
