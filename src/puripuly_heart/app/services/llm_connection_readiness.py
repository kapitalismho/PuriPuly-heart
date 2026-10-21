from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

_WRAPPER_ATTRIBUTES = ("provider", "inner", "primary")
_MAX_WRAPPER_DEPTH = 6


def _connection_mode_provider(llm: object | None) -> object | None:
    current = llm
    for _ in range(_MAX_WRAPPER_DEPTH):
        if current is None:
            return None
        if callable(getattr(current, "set_connection_mode", None)):
            return current
        current = next(
            (
                getattr(current, name)
                for name in _WRAPPER_ATTRIBUTES
                if getattr(current, name, None) is not None
            ),
            None,
        )
    return None


async def apply_llm_connection_mode(
    llm: object | None, mode: Literal["inactive", "standby", "active"]
) -> bool:
    provider = _connection_mode_provider(llm)
    if provider is None:
        return False
    result = getattr(provider, "set_connection_mode")(mode)
    if inspect.isawaitable(result):
        await result
    return True


@dataclass(slots=True)
class LlmConnectionReadinessOwner:
    llm_provider: Callable[[], object | None]
    translation_enabled: Callable[[], bool]
    _task: asyncio.Task[bool] | None = field(init=False, default=None, repr=False)
    _task_provider: object | None = field(init=False, default=None, repr=False)
    _task_mode: Literal["inactive", "standby", "active"] | None = field(
        init=False, default=None, repr=False
    )
    _tasks: set[asyncio.Task[bool]] = field(init=False, default_factory=set, repr=False)
    _revision: int = field(init=False, default=0, repr=False)
    _suspended: bool = field(init=False, default=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)

    def sync(self) -> None:
        if self._closed:
            return
        provider = _connection_mode_provider(self.llm_provider())
        mode: Literal["inactive", "standby", "active"] = (
            "inactive" if self._suspended else "active" if self.translation_enabled() else "standby"
        )
        if self._task_provider is provider and self._task_mode == mode:
            return
        previous = self._task_provider
        self._task_provider = provider
        self._task_mode = mode
        self._revision += 1
        if previous is not None and previous is not provider:
            self._start_mode(previous, "inactive")
        self._task = self._start_mode(provider, mode) if provider is not None else None

    async def suspend(self) -> None:
        if self._closed:
            return
        self._suspended = True
        self.sync()
        if self._task is not None:
            await asyncio.shield(self._task)

    def resume(self) -> None:
        if self._closed:
            return
        self._suspended = False
        self.sync()

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._revision += 1
            self._task_mode = "inactive"
            provider = self._task_provider
            self._task = self._start_mode(provider, "inactive") if provider is not None else None
        if self._tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tuple(self._tasks)), return_exceptions=True
            )

    def _start_mode(
        self, provider: object, mode: Literal["inactive", "standby", "active"]
    ) -> asyncio.Task[bool]:
        task = asyncio.get_running_loop().create_task(
            self._run(provider, mode, self._revision), name=f"llm-connection-{mode}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(
        self, provider: object, mode: Literal["inactive", "standby", "active"], revision: int
    ) -> bool:
        if provider is self._task_provider:
            if revision != self._revision:
                return False
        elif mode != "inactive":
            return False
        try:
            return await apply_llm_connection_mode(provider, mode)
        except Exception:
            return False


__all__ = [
    "LlmConnectionReadinessOwner",
    "apply_llm_connection_mode",
]
