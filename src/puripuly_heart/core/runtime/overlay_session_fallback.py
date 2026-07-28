from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any

OverlaySessionFallbackTaskFactory = Callable[
    [Coroutine[Any, Any, None], str],
    asyncio.Task[None],
]
OverlaySessionFallbackDiagnosticsSink = Callable[
    [str, Mapping[str, object], BaseException | None],
    None,
]


def _create_overlay_session_fallback_task(
    coroutine: Coroutine[Any, Any, None],
    name: str,
) -> asyncio.Task[None]:
    return asyncio.create_task(
        coroutine,
        name=f"OverlaySessionFallbackOwner:{name}",
    )


@dataclass(slots=True)
class OverlaySessionFallbackOwner:
    can_start: Callable[[], bool]
    start_overlay: Callable[[], Awaitable[None]]
    publish_notice: Callable[[bool], None]
    task_factory: OverlaySessionFallbackTaskFactory = _create_overlay_session_fallback_task
    diagnostics_sink: OverlaySessionFallbackDiagnosticsSink | None = None
    _active: bool = field(init=False, default=False, repr=False)
    _notice_active: bool = field(init=False, default=False, repr=False)
    _generation: int = field(init=False, default=0, repr=False)
    _task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _accepting_ingress: bool = field(init=False, default=True, repr=False)

    @property
    def owner_name(self) -> str:
        return "OverlaySessionFallbackOwner"

    @property
    def active(self) -> bool:
        return self._active

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def accepting_ingress(self) -> bool:
        return self._accepting_ingress

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": (
                "_active",
                "_generation",
                "_task",
                "_accepting_ingress",
            ),
            "stop_ingress": "reject new work, invalidate and cancel deferred fallback start",
            "shutdown_policy": "cancel and await deferred fallback start",
            "late_callback_rule": "generation and active-state checks reject stale starts",
        }

    def should_fallback(
        self,
        *,
        reason: str,
        active_target: str | None,
        configured_enabled: bool,
        configured_target: str,
        desktop_target: str,
        steamvr_target: str,
    ) -> bool:
        return bool(
            reason in {"steamvr_not_running", "steamvr_not_installed"}
            and self._accepting_ingress
            and not self._active
            and active_target != desktop_target
            and configured_enabled
            and configured_target == steamvr_target
        )

    def activate(self) -> None:
        if not self._accepting_ingress:
            return
        self._active = True

    def publish(self, active: bool) -> None:
        active = bool(active)
        if self._notice_active == active:
            return
        self._notice_active = active
        try:
            self.publish_notice(active)
        except Exception:
            return

    def schedule(self) -> None:
        if not self._accepting_ingress:
            return
        self._generation += 1
        generation = self._generation
        task = self._task
        if task is not None and not task.done():
            task.cancel()
        coroutine = self._run(generation)
        try:
            task = self.task_factory(
                coroutine,
                f"overlay-session-desktop-fallback-{generation}",
            )
        except RuntimeError as exc:
            coroutine.close()
            self._task = None
            self._emit(
                "overlay_session_fallback_schedule_failed",
                {"error_type": type(exc).__name__},
                exc,
            )
            return
        self._task = task
        task.add_done_callback(self._on_task_done)

    def clear(self) -> None:
        self._active = False
        self._invalidate_pending()
        self.publish(False)

    def stop_ingress(self) -> None:
        self._accepting_ingress = False
        self._invalidate_pending()

    def _invalidate_pending(self) -> None:
        self._generation += 1
        task = self._task
        if task is not None and not task.done():
            task.cancel()

    async def close(self) -> None:
        self.stop_ingress()
        task = self._task
        self._task = None
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        self._active = False
        self.publish(False)

    async def _run(self, generation: int) -> None:
        await asyncio.sleep(0)
        if generation != self._generation or not self._active or not self.can_start():
            return
        try:
            await self.start_overlay()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit(
                "overlay_session_fallback_start_failed",
                {"error_type": type(exc).__name__},
                exc,
            )

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        if self._task is task:
            self._task = None

    def _emit(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None,
    ) -> None:
        if self.diagnostics_sink is None:
            return
        try:
            self.diagnostics_sink(event, metadata, exception)
        except Exception:
            return


__all__ = ["OverlaySessionFallbackOwner"]
