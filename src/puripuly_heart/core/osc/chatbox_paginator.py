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
    _latest_self_turn_key: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _latest_self_utterance_id: UUID | None = field(default=None, init=False, repr=False)
    _latest_self_revision: int = field(default=-1, init=False, repr=False)
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
        page_count = (
            len(
                self._split_text(
                    message.text.strip(),
                    preserve_newlines=message.self_turn_key is not None,
                )
            )
            if message.text.strip()
            else 0
        )
        if message.self_turn_key is not None and self._prepare_self_turn_enqueue(
            message,
            page_count=page_count,
        ):
            return
        if self._is_paginating():
            self._pending_messages.append(message)
            self._record_stage(
                "message_enqueue",
                utterance_id=str(message.utterance_id),
                text_len=len(message.text),
                page_count=page_count,
                started=False,
                turn_generation=message.turn_generation,
                turn_order=message.turn_order,
                presentation_revision=message.presentation_revision,
            )
            return
        self._record_stage(
            "message_enqueue",
            utterance_id=str(message.utterance_id),
            text_len=len(message.text),
            page_count=page_count,
            started=True,
            turn_generation=message.turn_generation,
            turn_order=message.turn_order,
            presentation_revision=message.presentation_revision,
        )
        self._start_message(message)
        if not self._is_paginating():
            self._drain_pending_messages()

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

    def _prepare_self_turn_enqueue(
        self,
        message: OSCMessage,
        *,
        page_count: int,
    ) -> bool:
        assert self._pending_messages is not None
        incoming_key = message.self_turn_key
        assert incoming_key is not None
        if self._latest_self_turn_key is not None:
            if self._latest_self_turn_key > incoming_key:
                return True
            if self._latest_self_turn_key == incoming_key:
                if self._latest_self_utterance_id != message.utterance_id:
                    raise ValueError("self turn key cannot identify multiple parent utterances")
                if self._latest_self_revision >= message.presentation_revision:
                    return True
        active = self._active_message if self._is_paginating() else None
        active_key = None if active is None else active.self_turn_key
        existing_self_messages = tuple(
            candidate
            for candidate in (active, *self._pending_messages)
            if candidate is not None and candidate.self_turn_key is not None
        )
        if any(candidate.self_turn_key > incoming_key for candidate in existing_self_messages):
            return True
        for candidate in existing_self_messages:
            if (
                candidate.self_turn_key == incoming_key
                and candidate.utterance_id != message.utterance_id
            ):
                raise ValueError("self turn key cannot identify multiple parent utterances")
        if any(
            candidate.self_turn_key == incoming_key
            and candidate.presentation_revision >= message.presentation_revision
            for candidate in existing_self_messages
        ):
            return True
        self._latest_self_turn_key = incoming_key
        self._latest_self_utterance_id = message.utterance_id
        self._latest_self_revision = message.presentation_revision

        active_pruned = False
        dropped_pages = 0
        if active is not None and active_key == incoming_key:
            dropped_pages = self._clear_active_message()
            self._record_stage(
                "chatbox_revision_replaced",
                utterance_id=str(message.utterance_id),
                turn_generation=message.turn_generation,
                turn_order=message.turn_order,
                previous_revision=active.presentation_revision,
                presentation_revision=message.presentation_revision,
                location="active",
                dropped_pages=dropped_pages,
            )
        elif active_key is not None and active_key < incoming_key:
            dropped_pages = self._clear_active_message()
            active_pruned = True

        replaced_pending_revision: int | None = None
        replacement_index: int | None = None
        pruned_messages = 0
        next_pending: list[OSCMessage] = []
        for candidate in self._pending_messages:
            candidate_key = candidate.self_turn_key
            if candidate_key is None:
                next_pending.append(candidate)
                continue
            if candidate_key < incoming_key:
                if replacement_index is None:
                    replacement_index = len(next_pending)
                pruned_messages += 1
                continue
            if candidate_key == incoming_key:
                if replacement_index is None:
                    replacement_index = len(next_pending)
                replaced_pending_revision = max(
                    candidate.presentation_revision,
                    replaced_pending_revision or 0,
                )
                continue
            next_pending.append(candidate)
        self._pending_messages[:] = next_pending

        if replaced_pending_revision is not None:
            self._record_stage(
                "chatbox_revision_replaced",
                utterance_id=str(message.utterance_id),
                turn_generation=message.turn_generation,
                turn_order=message.turn_order,
                previous_revision=replaced_pending_revision,
                presentation_revision=message.presentation_revision,
                location="pending",
                dropped_pages=0,
            )

        if active_pruned or pruned_messages:
            self._record_stage(
                "chatbox_older_turn_pruned",
                utterance_id=str(message.utterance_id),
                turn_generation=message.turn_generation,
                turn_order=message.turn_order,
                presentation_revision=message.presentation_revision,
                pruned_messages=pruned_messages,
                pruned_pages=dropped_pages,
            )

        if self._is_paginating() and replacement_index is not None:
            self._pending_messages.insert(replacement_index, message)
            self._record_stage(
                "message_enqueue",
                utterance_id=str(message.utterance_id),
                text_len=len(message.text),
                page_count=page_count,
                started=False,
                replaced=True,
                turn_generation=message.turn_generation,
                turn_order=message.turn_order,
                presentation_revision=message.presentation_revision,
            )
            return True
        return False

    def _clear_active_message(self) -> int:
        assert self._pending_pages is not None
        dropped_pages = len(self._pending_pages)
        self._pending_pages.clear()
        self._next_page_at = 0.0
        self._active_message = None
        self._active_page_index = 0
        self._active_page_count = 0
        return dropped_pages

    def _start_message(self, message: OSCMessage) -> None:
        text = message.text.strip()
        if not text:
            return

        parts = self._split_text(
            text,
            preserve_newlines=message.self_turn_key is not None,
        )
        head = parts[0]
        tail = parts[1:]
        self._active_message = message
        self._active_page_index = 0
        self._active_page_count = len(parts)
        self._send_page(mode="queued", text=head, remaining_parts=len(tail))

        if tail:
            self._pending_pages.extend(tail)
            self._next_page_at = self.clock.now() + self.page_interval_s
            return
        self._active_message = None
        self._active_page_index = 0
        self._active_page_count = 0

    def _drain_pending_messages(self) -> None:
        while self._pending_messages and not self._is_paginating():
            next_message = self._pending_messages.pop(0)
            self._start_message(next_message)

    def _split_text(self, text: str, *, preserve_newlines: bool = False) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        if preserve_newlines and "\n" in text:
            return [
                text[index : index + self.max_chars]
                for index in range(0, len(text), self.max_chars)
            ]
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
                turn_generation=None if active is None else active.turn_generation,
                turn_order=None if active is None else active.turn_order,
                presentation_revision=(
                    None if active is None else active.presentation_revision
                ),
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
        metadata = publication.metadata
        generation = self._non_negative_metadata_int(publication, "turn_generation")
        order = self._non_negative_metadata_int(publication, "turn_order")
        revision = self._non_negative_metadata_int(publication, "presentation_revision")
        if "turn_generation" in metadata and generation is None:
            raise TypeError("turn_generation must be a non-negative integer")
        if "turn_order" in metadata and order is None:
            raise TypeError("turn_order must be a non-negative integer")
        if "presentation_revision" in metadata and revision is None:
            raise TypeError("presentation_revision must be a non-negative integer")
        if (generation is None) != (order is None):
            raise ValueError("turn generation and order must be provided together")
        revision = revision or 0
        if generation is None and revision != 0:
            raise ValueError("presentation_revision requires self turn identity")
        message = OSCMessage(
            utterance_id=UUID(publication.utterance_id),
            text=self._merge_chatbox_text(publication),
            created_at=self.paginator.clock.now(),
            turn_generation=generation,
            turn_order=order,
            presentation_revision=revision,
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

    @staticmethod
    def _non_negative_metadata_int(
        publication: SelfUtterancePublication,
        key: str,
    ) -> int | None:
        value = publication.metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value


def _redact_chatbox_disclosure_text(text: str) -> str:
    result = redact_text_for_sink(text, DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE)
    if result.status == DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED and result.text is not None:
        return result.text
    return DIAGNOSTIC_REDACTION_MARKER
