from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any

from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresencePort

VrchatOscProbeTaskFactory = Callable[
    [Coroutine[Any, Any, None], str],
    asyncio.Task[None],
]
VrchatOscProbeDiagnosticsSink = Callable[
    [str, Mapping[str, object], BaseException | None],
    None,
]


def _create_vrchat_osc_probe_task(
    coroutine: Coroutine[Any, Any, None],
    name: str,
) -> asyncio.Task[None]:
    return asyncio.create_task(
        coroutine,
        name=f"VrchatOscPresenceProbeOwner:{name}",
    )


@dataclass(slots=True)
class VrchatOscPresenceProbeOwner:
    presence_provider: Callable[[], VrchatOscPresencePort | None]
    port_provider: Callable[[], int]
    publish_notice: Callable[[bool], None]
    task_factory: VrchatOscProbeTaskFactory = _create_vrchat_osc_probe_task
    diagnostics_sink: VrchatOscProbeDiagnosticsSink | None = None
    interval_seconds: float = 30.0
    _notice_active: bool = field(init=False, default=False, repr=False)
    _task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _generation: int = field(init=False, default=0, repr=False)
    _accepting_ingress: bool = field(init=False, default=True, repr=False)

    @property
    def owner_name(self) -> str:
        return "VrchatOscPresenceProbeOwner"

    @property
    def notice_active(self) -> bool:
        return self._notice_active

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def accepting_ingress(self) -> bool:
        return self._accepting_ingress

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": (
                "_task",
                "_generation",
                "_notice_active",
                "_accepting_ingress",
            ),
            "stop_ingress": "reject new probes and invalidate the active generation",
            "shutdown_policy": "cancel and await presence probe task",
            "late_callback_rule": "generation check drops stale probe results",
        }

    def publish(self, active: bool) -> None:
        active = bool(active)
        if self._notice_active == active:
            return
        self._notice_active = active
        try:
            self.publish_notice(active)
        except Exception:
            return

    def schedule(self, *, force: bool = False) -> None:
        _ = force
        if not self._accepting_ingress:
            return
        if self.presence_provider() is None:
            self.publish(False)
            return
        self._generation += 1
        generation = self._generation
        existing = self._task
        if existing is not None and not existing.done():
            existing.cancel()
        coroutine = self.run(generation)
        try:
            self._task = self.task_factory(
                coroutine,
                f"vrchat-osc-presence-{generation}",
            )
        except RuntimeError:
            coroutine.close()
            self._task = None

    async def run(self, generation: int) -> None:
        while generation == self._generation:
            try:
                presence = self.presence_provider()
                should_prompt = (
                    None
                    if presence is None
                    else await presence.should_prompt_enable_osc(port=self.port_provider())
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._emit(
                    "vrchat_osc_presence_probe_failed",
                    {"error_type": type(exc).__name__},
                    exc,
                )
                should_prompt = None
            if generation != self._generation:
                return
            self.publish(bool(should_prompt))
            try:
                await asyncio.sleep(max(0.0, float(self.interval_seconds)))
            except asyncio.CancelledError:
                raise

    def stop_ingress(self) -> None:
        self._accepting_ingress = False
        self._generation += 1

    async def cancel(self) -> None:
        self.stop_ingress()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.publish(False)

    async def close(self) -> None:
        await self.cancel()

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


__all__ = ["VrchatOscPresenceProbeOwner"]
