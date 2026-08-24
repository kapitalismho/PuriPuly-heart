from __future__ import annotations

import logging
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from puripuly_heart.core.clock import Clock
from puripuly_heart.core.diagnostic_validation import (
    DIAGNOSTIC_REDACTION_MARKER,
    DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE,
    DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
    redact_text_for_sink,
)
from puripuly_heart.core.osc.sender import OscSender
from puripuly_heart.core.output.chatbox import SelfUtterancePublication, SystemDisclosurePublication
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
    stage_recorder: Callable[..., object] | None = None
    _pending_pages: list[str] | None = None
    _pending_messages: list[OSCMessage] | None = None
    _next_page_at: float = 0.0
    _active_message: OSCMessage | None = field(default=None, init=False, repr=False)
    _active_page_index: int = field(default=0, init=False, repr=False)
    _active_page_count: int = field(default=0, init=False, repr=False)
    _typing_reasons: set[str] = field(default_factory=set, init=False, repr=False)
    _legacy_typing_active: bool = field(default=False, init=False, repr=False)
    _last_typing_state: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        if self.page_interval_s <= 0:
            raise ValueError("page_interval_s must be > 0")
        self._pending_pages = []
        self._pending_messages = []

    def enqueue(self, message: OSCMessage) -> None:
        page_count = len(self._split_text(message.text.strip())) if message.text.strip() else 0
        if self._is_paginating():
            self._pending_messages.append(message)
            self._record_stage(
                "message_enqueue",
                utterance_id=str(message.utterance_id),
                text_len=len(message.text),
                page_count=page_count,
                started=False,
            )
            return
        self._record_stage(
            "message_enqueue",
            utterance_id=str(message.utterance_id),
            text_len=len(message.text),
            page_count=page_count,
            started=True,
        )
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
        self._active_page_index += 1
        self._send_page(mode="queued", text=page, remaining_parts=remaining_parts)

        if self._pending_pages:
            self._next_page_at = now + self.page_interval_s
            return

        self._next_page_at = 0.0
        self._active_message = None
        self._active_page_index = 0
        self._active_page_count = 0
        self._drain_pending_messages()

    def send_immediate(self, text: str) -> bool:
        """Send a single chatbox packet immediately without changing pagination state."""
        text = text.strip()
        if not text:
            return False
        return self._send_page(mode="immediate", text=text, remaining_parts=0)

    def send_typing(self, is_typing: bool) -> None:
        """Forward typing indicator to the OSC sender."""
        self._legacy_typing_active = bool(is_typing)
        if not is_typing and self._typing_reasons:
            self._apply_typing_state()
            return
        self._send_typing_state(self._is_typing_active())

    def set_typing_reason(self, reason: str, active: bool) -> None:
        reason = reason.strip()
        if not reason:
            raise ValueError("reason must be non-empty")
        was_typing = self._is_typing_active()
        if active:
            self._typing_reasons.add(reason)
        else:
            self._typing_reasons.discard(reason)
        self._apply_typing_state(was_typing=was_typing)

    def clear_typing_reasons(self) -> None:
        if not self._typing_reasons and self._last_typing_state == self._is_typing_active():
            return
        self._typing_reasons.clear()
        self._apply_typing_state()

    def _is_typing_active(self) -> bool:
        return self._legacy_typing_active or bool(self._typing_reasons)

    def _apply_typing_state(self, *, was_typing: bool | None = None) -> None:
        is_typing = self._is_typing_active()
        previous = self._last_typing_state if was_typing is None else was_typing
        if is_typing != previous or is_typing != self._last_typing_state:
            self._send_typing_state(is_typing)

    def _send_typing_state(self, is_typing: bool) -> None:
        try:
            self.sender.send_typing(is_typing)
            self._last_typing_state = bool(is_typing)
        except OSError as exc:
            self._emit_basic(
                f"[Basic][OSC] typing status=failed error={exc}", level=logging.WARNING
            )

    def drop_pending(self) -> None:
        """Drop queued chatbox pages/messages during output runtime shutdown."""
        assert self._pending_pages is not None
        assert self._pending_messages is not None
        dropped_pages = len(self._pending_pages)
        dropped_messages = len(self._pending_messages)
        self._pending_pages.clear()
        self._pending_messages.clear()
        self._next_page_at = 0.0
        self._active_message = None
        self._active_page_index = 0
        self._active_page_count = 0
        self._emit_basic(
            "[Basic][OSC] chatbox backlog dropped on output shutdown "
            f"pages={dropped_pages} messages={dropped_messages}"
        )

    def _is_paginating(self) -> bool:
        return bool(self._pending_pages)

    def _start_message(self, message: OSCMessage) -> None:
        text = message.text.strip()
        if not text:
            return

        parts = self._split_text(text)
        head = parts[0]
        tail = parts[1:]
        self._active_message = message
        self._active_page_index = 0
        self._active_page_count = len(parts)
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
        if mode == "queued":
            active = self._active_message
            self._record_stage(
                "page_send",
                utterance_id=None if active is None else str(active.utterance_id),
                page_index=self._active_page_index,
                page_count=self._active_page_count,
                remaining_parts=remaining_parts,
                text_len=len(text),
            )
        return True

    def _record_stage(self, event: str, **fields: object) -> None:
        if self.stage_recorder is None:
            return
        now = self.clock.now()
        pending = self._pending_messages or []
        oldest_created = min(
            (message.created_at for message in pending),
            default=None,
        )
        if self._active_message is not None:
            oldest_created = (
                self._active_message.created_at
                if oldest_created is None
                else min(oldest_created, self._active_message.created_at)
            )
        self.stage_recorder(
            event,
            pending_messages=len(pending),
            pending_pages=0 if self._pending_pages is None else len(self._pending_pages),
            oldest_age_s=(
                None if oldest_created is None else round(max(0.0, now - oldest_created), 3)
            ),
            **fields,
        )

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


@dataclass(slots=True)
class ChatboxPaginatorOutputAdapter:
    paginator: ChatboxPaginator
    render_system_disclosure: Callable[[SystemDisclosurePublication], str]
    include_source: bool = True

    async def publish_self_utterance(self, publication: SelfUtterancePublication) -> None:
        message = OSCMessage(
            utterance_id=UUID(publication.utterance_id),
            text=self._merge_chatbox_text(publication),
            created_at=self.paginator.clock.now(),
        )
        self.paginator.enqueue(message)
        self.paginator.send_typing(False)

    async def publish_system_disclosure(self, publication: SystemDisclosurePublication) -> None:
        message = OSCMessage(
            utterance_id=UUID(publication.disclosure_id),
            text=_redact_chatbox_disclosure_text(self.render_system_disclosure(publication)),
            created_at=self.paginator.clock.now(),
        )
        self.paginator.enqueue(message)

    def _merge_chatbox_text(self, publication: SelfUtterancePublication) -> str:
        transcript_text = publication.transcript_text or ""
        translation_text = publication.translation_text
        if translation_text is None:
            return transcript_text
        if self.include_source and transcript_text:
            return f"{transcript_text} ({translation_text})"
        return translation_text


def _redact_chatbox_disclosure_text(text: str) -> str:
    result = redact_text_for_sink(text, DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE)
    if result.status == DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED and result.text is not None:
        return result.text
    return DIAGNOSTIC_REDACTION_MARKER
