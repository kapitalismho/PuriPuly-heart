from __future__ import annotations

import asyncio
from typing import Any


async def run_configured_provider_status_verification(controller: Any) -> None:
    failures: list[BaseException] = []
    owner = controller._get_provider_status_verification_owner()

    def capture_failure(
        _event: str,
        _metadata: object,
        exception: BaseException | None,
    ) -> None:
        if exception is not None:
            failures.append(exception)

    owner.diagnostics_sink = capture_failure
    existing_tasks = frozenset(owner.active_task_names)
    controller._schedule_provider_status_verification()
    scheduled_tasks = frozenset(owner.active_task_names) - existing_tasks
    assert len(scheduled_tasks) == 1
    for _ in range(100):
        if not owner.active_task_names:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("provider status verification did not finish")
    if failures:
        raise failures[0]


__all__ = ["run_configured_provider_status_verification"]
