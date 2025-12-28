from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass

from puripuly_heart.core.clock import Clock
from puripuly_heart.core.osc.sender import OscSender
from puripuly_heart.domain.models import OSCMessage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SmartOscQueue:
    sender: OscSender
    clock: Clock
    max_chars: int = 144
    cooldown_s: float = 1.5
    ttl_s: float = 7.0
    _next_send_at: float = 0.0
    _pending: list[OSCMessage] | None = None

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        if self.cooldown_s <= 0:
            raise ValueError("cooldown_s must be > 0")
        if self.ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        self._pending = []

    def enqueue(self, message: OSCMessage) -> None:
        self._pending.append(message)
        self.process_due()

    def process_due(self) -> None:
        now = self.clock.now()
        if now < self._next_send_at:
            return

        self._drop_expired(now)
        if not self._pending:
            return

        head_utterance_id = self._pending[0].utterance_id
        combined_text, created_at = self._combine_pending()
        if not combined_text:
            self._pending.clear()
            return

        parts = self._split_text(combined_text)
        head = parts[0]
        tail = parts[1:]

        logger.info(f"[OSC] Sending: '{head}'")
        try:
            self.sender.send_chatbox(head)
        except OSError as exc:
            logger.warning(f"[OSC] Send failed: {exc}")
            return
        self._next_send_at = now + self.cooldown_s

        self._pending.clear()
        if tail:
            self._pending.append(
                OSCMessage(
                    utterance_id=head_utterance_id,
                    text=" ".join(tail),
                    created_at=created_at,
                )
            )

    def _drop_expired(self, now: float) -> None:
        self._pending[:] = [m for m in self._pending if (now - m.created_at) <= self.ttl_s]

    def _combine_pending(self) -> tuple[str, float]:
        created_at = min(m.created_at for m in self._pending)
        combined = " ".join(m.text for m in self._pending if m.text)
        combined = combined.strip()
        return combined, created_at

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        return textwrap.wrap(
            text,
            width=self.max_chars,
            break_long_words=True,
            break_on_hyphens=False,
        )

    def send_typing(self, is_typing: bool) -> None:
        """Forward typing indicator to the OSC sender."""
        try:
            self.sender.send_typing(is_typing)
        except OSError as exc:
            logger.warning(f"[OSC] Typing send failed: {exc}")
