"""UI-independent coordination state machine for API key blur/save/verify."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Awaitable
from typing import Callable

SaveHandler = Callable[[str, str], object]
VerifyHandler = Callable[[str, str], object]
StatusHandler = Callable[[str], None]
MessageHandler = Callable[[str, str], None]


class ApiKeyVerificationController:
    """Owns the blur -> save -> verify flow for one API key field.

    The controller is UI-independent: it reports status transitions through
    ``on_status`` and user messages through ``on_message``. The field widget
    binds those to Flet icons and snackbars.
    """

    def __init__(
        self,
        *,
        secret_key: str,
        provider: str,
        on_verify: VerifyHandler,
        on_save: SaveHandler | None = None,
        on_status: StatusHandler | None = None,
        on_message: MessageHandler | None = None,
        show_status: bool = True,
    ) -> None:
        self._secret_key = secret_key
        self._provider = provider
        self._on_verify = on_verify
        self._on_save = on_save
        self._on_status = on_status
        self._on_message = on_message
        self._show_status = show_status

        self._value_dirty = False
        self._last_verified_hash = ""
        self._is_verifying = False
        self._pending: tuple[str, str, bool] | None = None
        self._current_status = "idle"
        self._value_getter: Callable[[], str] = lambda: ""

    @staticmethod
    def key_hash(key: str) -> str:
        if not key:
            return ""
        return hashlib.sha256(key.encode()).hexdigest()

    @property
    def last_verified_hash(self) -> str:
        return self._last_verified_hash

    @last_verified_hash.setter
    def last_verified_hash(self, value: str) -> None:
        self._last_verified_hash = value

    def force_status(self, status: str) -> None:
        """Set status without verification (used for projection refresh)."""
        self._set_status(status)

    @property
    def status(self) -> str:
        return self._current_status

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def notify_edit(self) -> None:
        """Mark the current field value dirty after a user edit."""
        self._value_dirty = True

    def replace_value(self) -> None:
        """Reset dirty tracking after the field value is replaced programmatically."""
        self._value_dirty = False

    def handle_blur(self, key: str) -> None:
        """Coordinate save and verification for the blurred value."""
        needs_save = self._value_dirty
        if needs_save:
            self._value_dirty = False
        elif not self._show_status:
            return

        if not self._show_status:
            if needs_save and self._on_save:
                self._on_save(self._secret_key, key)
            return
        if self._on_verify is None:
            return

        key_hash = self.key_hash(key)
        if not needs_save and key_hash == self._last_verified_hash:
            return

        self._pending = (key, key_hash, needs_save)

    def run_pending(self) -> Awaitable[None] | None:
        """Start draining pending blur requests; returns the drain coroutine.

        The caller is responsible for scheduling the returned awaitable (for
        example via ``page.run_task``). When a drain is already active, the
        pending request is queued and ``None`` is returned.
        """
        if self._is_verifying:
            return None
        self._is_verifying = True

        async def drain() -> None:
            try:
                await self._drain()
            finally:
                self._is_verifying = False

        return drain()

    async def verify_direct(self, key: str) -> None:
        """Verify one key directly, bypassing blur save/skip gating."""
        key_hash = self.key_hash(key)
        await self._verify(key, key_hash)

    async def _drain(self) -> None:
        while self._pending is not None:
            key, key_hash, needs_save = self._pending
            self._pending = None
            if not key_hash and not needs_save:
                return

            if needs_save and self._on_save:
                save_result = self._on_save(self._secret_key, key)
                if inspect.isawaitable(save_result):
                    save_result = await save_result
                if save_result is False:
                    self._set_status("error")
                    self._last_verified_hash = ""
                    continue
            if not key:
                self._set_status("idle")
                self._last_verified_hash = ""
                continue
            await self._verify(key, key_hash)

    async def _verify(self, key: str, key_hash: str) -> None:
        self._set_status("verifying")
        try:
            result = self._on_verify(self._provider, key)
            if inspect.isawaitable(result):
                result = await result
            success, msg = result
        except Exception as exc:
            if self.current_value() != key:
                return
            self._set_status("error")
            self._last_verified_hash = ""
            self._emit_message(
                "snackbar.verification_error",
                message=self.translate_error(str(exc)),
            )
            return

        if self.current_value() != key:
            return

        if success:
            self._set_status("success")
            self._last_verified_hash = key_hash
            self._emit_message("snackbar.verification_ok")
        else:
            self._set_status("error")
            self._last_verified_hash = ""
            self._emit_message(
                "snackbar.verification_failed",
                message=self.translate_error(msg),
            )

    def current_value(self) -> str:
        return self._value_getter()

    def set_value_getter(self, getter: Callable[[], str]) -> None:
        self._value_getter = getter

    def _set_status(self, status: str) -> None:
        self._current_status = status
        if self._show_status and self._on_status:
            self._on_status(status)

    def _emit_message(self, message_key: str, *, message: str = "") -> None:
        if self._on_message:
            self._on_message(message_key, message)

    def translate_error(self, msg: str) -> str:
        """Translate common error messages to translatable message keys."""
        msg_lower = msg.lower()
        if msg_lower.startswith("qwen_model_unavailable:"):
            model = msg.split(":", 1)[1].strip() if ":" in msg else ""
            return f"error.qwen_model_unavailable:{model or 'unknown'}"
        if "401" in msg or "unauthorized" in msg_lower:
            return "error.api_key_invalid"
        if "403" in msg or "forbidden" in msg_lower:
            return "error.api_key_invalid"
        if "timeout" in msg_lower or "timed out" in msg_lower:
            return "error.network_timeout"
        if "connection" in msg_lower or "network" in msg_lower:
            return "error.network_error"
        return msg
