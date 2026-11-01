import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def settings_rendezvous(operation, entered, release, *, context):
    task = asyncio.create_task(operation, name=context)
    waiter = asyncio.create_task(entered.wait(), name=f"{context}: callback")
    deadline = asyncio.timeout(3)
    stage = "reach the expected callback"
    try:
        try:
            async with deadline:
                await asyncio.wait((task, waiter), return_when=asyncio.FIRST_COMPLETED)
                if task.done():
                    task.result()
                if not entered.is_set():
                    raise AssertionError(
                        f"{context}: operation completed before the expected callback"
                    )
                stage = "finish the controlled operation"
                yield task
                await task
        except TimeoutError as error:
            if deadline.expired():
                raise AssertionError(f"{context}: timed out after 3s waiting to {stage}") from error
            raise
    finally:
        release.set()
        for owned in (task, waiter):
            if not owned.done():
                owned.cancel()
        await asyncio.gather(task, waiter, return_exceptions=True)
