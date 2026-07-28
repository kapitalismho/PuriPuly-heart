from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task

ManagedStatusRefreshWork = Callable[[], Awaitable[None]]
ManagedStatusRefreshDiagnosticsSink = Callable[
    [str, Mapping[str, object], BaseException | None],
    None,
]


def _managed_status_refresh_scope() -> LifecycleScope:
    return LifecycleScope("ManagedStatusRefreshOwner")


@dataclass(slots=True)
class ManagedStatusRefreshOwner:
    diagnostics_sink: ManagedStatusRefreshDiagnosticsSink | None = None
    _task_scope: LifecycleScope = field(
        init=False,
        default_factory=_managed_status_refresh_scope,
        repr=False,
    )
    _task_sequence: int = field(init=False, default=0, repr=False)
    _ingress_stopped: bool = field(init=False, default=False, repr=False)

    @property
    def owner_name(self) -> str:
        return "ManagedStatusRefreshOwner"

    @property
    def active_task_names(self) -> tuple[str, ...]:
        return self._task_scope.active_task_names

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": ("_task_scope", "_task_sequence"),
            "stop_ingress": "reject new managed status and usage refresh work",
            "shutdown_policy": "cancel and await every managed refresh task",
            "late_callback_rule": "controller refresh callbacks recheck current identity and shutdown",
        }

    def schedule_status_refresh(self, work: ManagedStatusRefreshWork) -> bool:
        return self._schedule(work, kind="status")

    def schedule_trial_usage_refresh(self, work: ManagedStatusRefreshWork) -> bool:
        return self._schedule(work, kind="trial_usage")

    def stop_ingress(self) -> None:
        self._ingress_stopped = True

    async def close(self) -> None:
        self.stop_ingress()
        await self._task_scope.close()

    def _schedule(self, work: ManagedStatusRefreshWork, *, kind: str) -> bool:
        if self._ingress_stopped or self._task_scope.is_closed:
            return False

        async def run() -> None:
            try:
                await work()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._emit(
                    "managed_status_refresh_failed",
                    {
                        "kind": kind,
                        "error_type": type(exc).__name__,
                    },
                    exc,
                )

        self._task_sequence += 1
        coroutine = run()
        try:
            start_lifecycle_task(
                self._task_scope,
                coroutine,
                name=f"{kind}-{self._task_sequence}",
            )
        except Exception as exc:
            coroutine.close()
            self._emit(
                "managed_status_refresh_schedule_failed",
                {
                    "kind": kind,
                    "error_type": type(exc).__name__,
                },
                exc,
            )
            return False
        return True

    def _emit(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None = None,
    ) -> None:
        if self.diagnostics_sink is None:
            return
        try:
            self.diagnostics_sink(event, metadata, exception)
        except Exception:
            return


__all__ = ["ManagedStatusRefreshOwner"]
