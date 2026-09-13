from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

TransportTaskFactory = Callable[
    [Coroutine[Any, Any, Any], str],
    asyncio.Task[Any],
]


@dataclass(frozen=True, slots=True)
class TransportWriteResult:
    outcome: str
    send_task: asyncio.Task[Any]
    cause: str | None = None


class TransportWriteCancelled(asyncio.CancelledError):
    def __init__(self, send_task: asyncio.Task[Any]) -> None:
        super().__init__()
        self.send_task = send_task


@dataclass(frozen=True, slots=True)
class TransportCloseResult:
    close_task: asyncio.Task[Any]
    failure: Exception | None


class OverlayTransportExecutor:
    async def write(
        self,
        connection: Any,
        message: str,
        *,
        timeout: float,
        task_factory: TransportTaskFactory,
    ) -> TransportWriteResult:
        send_task = task_factory(connection.send(message), "send")
        try:
            await asyncio.wait_for(asyncio.shield(send_task), timeout=timeout)
        except asyncio.CancelledError as exc:
            raise TransportWriteCancelled(send_task) from exc
        except TimeoutError:
            return TransportWriteResult(
                outcome="ambiguous",
                send_task=send_task,
                cause="write_timeout",
            )
        except Exception as exc:
            return TransportWriteResult(
                outcome="failed",
                send_task=send_task,
                cause=type(exc).__name__,
            )
        return TransportWriteResult(outcome="written", send_task=send_task)

    async def close(
        self,
        connection: Any,
        *,
        timeout: float,
        existing_task: asyncio.Task[Any] | None,
        task_factory: TransportTaskFactory,
    ) -> TransportCloseResult:
        close_task = existing_task
        if close_task is None:
            close_task = task_factory(connection.close(), "close-connection")
        try:
            await asyncio.wait_for(asyncio.shield(close_task), timeout=timeout)
        except TimeoutError as exc:
            self.abort(connection)
            return TransportCloseResult(close_task=close_task, failure=exc)
        except Exception as exc:
            self.abort(connection)
            return TransportCloseResult(close_task=close_task, failure=exc)
        return TransportCloseResult(close_task=close_task, failure=None)

    def abort(self, connection: Any) -> None:
        transport = getattr(connection, "transport", None)
        abort = getattr(transport, "abort", None)
        if callable(abort):
            with contextlib.suppress(Exception):
                abort()
