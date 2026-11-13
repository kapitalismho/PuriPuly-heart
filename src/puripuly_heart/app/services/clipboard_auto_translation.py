from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.runtime.clipboard import ClipboardRuntime

ClipboardWatcherFactory = Callable[[Callable[[str], None]], object]
ClipboardSubmitHandler = Callable[[str], Awaitable[None]]
ClipboardFailureSink = Callable[[str], None]


@dataclass(slots=True)
class ClipboardAutoTranslationOwner:
    watcher_factory: ClipboardWatcherFactory
    submit_text: ClipboardSubmitHandler
    failure_sink: ClipboardFailureSink
    platform_provider: Callable[[], str] = lambda: sys.platform
    strict_runtime_errors: bool = False
    _runtime: ClipboardRuntime | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _lock: asyncio.Lock | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def runtime(self) -> ClipboardRuntime | None:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: ClipboardRuntime | None) -> None:
        self._runtime = runtime

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def failure_diagnostics(self) -> ErrorDiagnostics | None:
        return self._runtime.failure_diagnostics if self._runtime is not None else None

    def get_runtime(self) -> ClipboardRuntime:
        if self._runtime is None:
            self._runtime = ClipboardRuntime(
                watcher_factory=self.watcher_factory,
                submit_handler=self.submit_text,
                failure_sink=self._on_failure,
            )
        return self._runtime

    async def sync(
        self,
        *,
        enabled: bool,
        strict_runtime_errors: bool | None = None,
    ) -> None:
        strict = self._resolve_strict_runtime_errors(strict_runtime_errors)
        if not enabled or self.platform_provider() != "win32":
            await self.stop(strict_runtime_errors=strict)
            return
        async with self.lock:
            try:
                await self.get_runtime().sync(
                    enabled=True,
                    strict_runtime_errors=strict,
                )
            except Exception:
                if strict:
                    raise

    async def stop(self, *, strict_runtime_errors: bool | None = None) -> None:
        strict = self._resolve_strict_runtime_errors(strict_runtime_errors)
        async with self.lock:
            runtime = self._runtime
            if runtime is None:
                return
            try:
                await runtime.stop(strict_runtime_errors=strict)
            except Exception:
                if strict:
                    raise

    async def close(self, *, strict_runtime_errors: bool | None = None) -> None:
        strict = self._resolve_strict_runtime_errors(strict_runtime_errors)
        async with self.lock:
            runtime = self._runtime
            if runtime is None:
                return
            try:
                await runtime.close()
            except Exception:
                if strict:
                    raise

    def on_text_from_thread(self, text: str) -> None:
        self.get_runtime().on_text_from_thread(text)

    def submit_from_loop(self, text: str) -> None:
        try:
            self.get_runtime().submit_from_loop(text)
        except RuntimeError as exc:
            self.get_runtime().record_failure(exc, "clipboard_submit")

    async def submit_now(self, text: str) -> None:
        await self._submit_text(text)

    async def _submit_text(self, text: str) -> None:
        try:
            await self.submit_text(text)
        except Exception as exc:
            self.get_runtime().record_failure(exc, "clipboard_submit")

    def _on_failure(self, diagnostics: ErrorDiagnostics) -> None:
        message = {
            "watcher_start": "Clipboard watcher failed to start",
            "watcher_stop": "Clipboard watcher failed to stop",
            "clipboard_read": "Clipboard could not be read",
            "clipboard_callback": "Clipboard callback failed",
            "clipboard_loop": "Clipboard watcher stopped after an internal failure",
            "clipboard_cleanup": "Clipboard watcher cleanup failed",
            "submit_drain": "Clipboard submissions did not settle before shutdown",
        }.get(diagnostics.operation, "Clipboard submit failed")
        try:
            self.failure_sink(message)
        except Exception:
            pass

    def _resolve_strict_runtime_errors(self, value: bool | None) -> bool:
        return self.strict_runtime_errors if value is None else bool(value)


__all__ = ["ClipboardAutoTranslationOwner"]
