from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

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

    def get_runtime(self) -> ClipboardRuntime:
        if self._runtime is None:
            self._runtime = ClipboardRuntime(
                watcher_factory=self.watcher_factory,
                submit_handler=self._submit_text,
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
                self.failure_sink("Clipboard watcher failed to start")
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
                self.failure_sink("Clipboard watcher failed to stop")
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
                self.failure_sink("Clipboard runtime failed to close")
                if strict:
                    raise

    def on_text_from_thread(self, text: str) -> None:
        self.get_runtime().on_text_from_thread(text)

    def submit_from_loop(self, text: str) -> None:
        try:
            self.get_runtime().submit_from_loop(text)
        except RuntimeError as exc:
            self.failure_sink(f"Clipboard submit scheduling failed: {exc}")

    async def submit_now(self, text: str) -> None:
        await self._submit_text(text)

    async def _submit_text(self, text: str) -> None:
        try:
            await self.submit_text(text)
        except Exception as exc:
            self.failure_sink(f"Clipboard submit failed: {exc}")

    def _resolve_strict_runtime_errors(self, value: bool | None) -> bool:
        return self.strict_runtime_errors if value is None else bool(value)


__all__ = ["ClipboardAutoTranslationOwner"]
