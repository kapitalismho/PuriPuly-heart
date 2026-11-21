from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import TypeVar

_ThreadResultT = TypeVar("_ThreadResultT")


async def run_owned_thread_call(
    operation: Callable[[], _ThreadResultT],
) -> _ThreadResultT:
    loop = asyncio.get_running_loop()
    return await wait_for_owned_future(loop.run_in_executor(None, operation))


async def wait_for_owned_future(
    future: asyncio.Future[_ThreadResultT],
    *,
    on_cancel: Callable[[], None] | None = None,
) -> _ThreadResultT:
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        try:
            if on_cancel is not None:
                on_cancel()
        finally:
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is None or not current_task.cancelling():
                        raise
                except BaseException:
                    break
            with contextlib.suppress(BaseException):
                future.result()
        raise asyncio.CancelledError
