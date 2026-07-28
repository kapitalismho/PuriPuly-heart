from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task

ProviderStatusVerificationWork = Callable[[], Awaitable[None]]
ProviderStatusVerificationDiagnosticsSink = Callable[
    [str, Mapping[str, object], BaseException | None],
    None,
]


def _provider_status_verification_scope() -> LifecycleScope:
    return LifecycleScope("ProviderStatusVerificationOwner")


@dataclass(slots=True)
class ProviderStatusVerificationOwner:
    diagnostics_sink: ProviderStatusVerificationDiagnosticsSink | None = None
    _task_scope: LifecycleScope = field(
        init=False,
        default_factory=_provider_status_verification_scope,
        repr=False,
    )
    _task_sequence: int = field(init=False, default=0, repr=False)
    _ingress_stopped: bool = field(init=False, default=False, repr=False)

    @property
    def owner_name(self) -> str:
        return "ProviderStatusVerificationOwner"

    @property
    def active_task_names(self) -> tuple[str, ...]:
        return self._task_scope.active_task_names

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": ("_task_scope", "_task_sequence"),
            "stop_ingress": "reject new provider-status verification work",
            "shutdown_policy": "cancel and await every provider-status verification task",
            "late_callback_rule": "verification callback rechecks shutdown before publication",
        }

    def schedule(self, work: ProviderStatusVerificationWork) -> bool:
        if self._ingress_stopped or self._task_scope.is_closed:
            return False

        async def run() -> None:
            try:
                await work()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._emit(
                    "provider_status_verification_failed",
                    {"error_type": type(exc).__name__},
                    exc,
                )

        self._task_sequence += 1
        coroutine = run()
        try:
            start_lifecycle_task(
                self._task_scope,
                coroutine,
                name=f"verification-{self._task_sequence}",
            )
        except Exception as exc:
            coroutine.close()
            self._emit(
                "provider_status_verification_schedule_failed",
                {"error_type": type(exc).__name__},
                exc,
            )
            return False
        return True

    def stop_ingress(self) -> None:
        self._ingress_stopped = True

    async def close(self) -> None:
        self.stop_ingress()
        await self._task_scope.close()

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


__all__ = ["ProviderStatusVerificationOwner"]
