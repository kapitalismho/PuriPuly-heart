from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.core.diagnostics import (
    current_diagnostic_operation,
    emit_failure,
    safe_exception_fields,
)
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.messages import ErrorDiagnostics

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

        identity = current_diagnostic_operation()
        generation = self._task_sequence + 1

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
                    generation=generation,
                    **identity,
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
                generation=generation,
                **identity,
            )
            return False
        return True

    def _emit(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None = None,
        *,
        generation: int | None = None,
        operation_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        diagnostics = getattr(exception, "diagnostics", None)
        if not isinstance(diagnostics, ErrorDiagnostics):
            diagnostics = ErrorDiagnostics(
                component="managed_status_refresh",
                operation=event,
                code=event,
                category="lifecycle",
                visibility="basic",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                fields={
                    "phase": event,
                    "refresh_kind": metadata.get("kind"),
                    **(safe_exception_fields(exception) if exception is not None else {}),
                },
                generation=generation,
                operation_id=operation_id,
                request_id=request_id,
            )
            if exception is not None and type(exception).__module__ == "builtins":
                exception.diagnostics = diagnostics
        emit_failure(
            logging.getLogger(__name__), diagnostics, severity="warning", disposition="degraded"
        )
        if self.diagnostics_sink is None:
            return
        try:
            self.diagnostics_sink(event, metadata, exception)
        except Exception as exc:
            secondary = ErrorDiagnostics(
                component="managed_status_refresh",
                operation="auth_notification",
                code="managed_status_notification_failed",
                category="lifecycle",
                visibility="basic",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                fields={"phase": "auth_notification", **safe_exception_fields(exc)},
                cause_id=diagnostics.diagnostic_id,
                generation=generation,
                operation_id=operation_id,
                request_id=request_id,
            )
            emit_failure(
                logging.getLogger(__name__), secondary, severity="warning", disposition="degraded"
            )


__all__ = ["ManagedStatusRefreshOwner"]
