from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

_TaskResultT = TypeVar("_TaskResultT")
_PROCESS_EVENT_READER_TASK_PREFIXES = ("process-read-",)


class OverlayRuntimeHandle:
    """Owns one overlay runtime generation and its background work."""

    resource_fields = (
        "_presenter",
        "_bridge",
        "_process_manager",
        "_start_task",
        "_monitor_task",
        "_renderer_events",
        "_renderer_event_task",
        "OverlayBridge._heartbeat_task",
        "_AsyncioOverlayProcess._reader_tasks",
        "OverlayProcessManager._monitor_task",
        "OverlayProcessManager startup event_task/bridge_task/exit_task/timeout_task",
        "OverlayProcessManager connected event_task/bridge_task/exit_task",
        "OverlayPresenter._expiration_tasks",
        "OverlayPresenter._peer_presentation_refresh_burst_task",
        "OverlayPresenter._self_presentation_refresh_burst_task",
    )
    stop_ingress = "broadcast shutdown and reject new overlay commands"
    shutdown_policy = (
        "owner-specific grace, cancel/gather start/monitor/renderer tasks, "
        "async presenter close, process stop, bridge stop, and kill escalation"
    )
    late_callback_rule = "old overlay instance events ignored after instance id changes"

    def __init__(
        self,
        *,
        overlay_instance_id: str | None = None,
        shutdown_grace_s: float = 0.05,
    ) -> None:
        self._overlay_instance_id = overlay_instance_id
        self._shutdown_grace_s = shutdown_grace_s
        self._presenter: object | None = None
        self._bridge: object | None = None
        self._process_manager: object | None = None
        self._diagnostics: object | None = None
        self._renderer_events: asyncio.Queue[dict[str, object]] | None = None
        self._start_task: asyncio.Task[Any] | None = None
        self._monitor_task: asyncio.Task[Any] | None = None
        self._renderer_event_task: asyncio.Task[Any] | None = None
        self._child_task_names: dict[asyncio.Task[Any], str] = {}
        self._completed_task_failures: dict[asyncio.Task[Any], Exception] = {}
        self._closing = False
        self._close_completed = False
        self._close_lock = asyncio.Lock()

    @property
    def owner_name(self) -> str:
        return "OverlayRuntimeHandle"

    @property
    def overlay_instance_id(self) -> str | None:
        return self._overlay_instance_id

    @property
    def presenter(self) -> object | None:
        return self._presenter

    @property
    def bridge(self) -> object | None:
        return self._bridge

    @property
    def process_manager(self) -> object | None:
        return self._process_manager

    @property
    def diagnostics(self) -> object | None:
        return self._diagnostics

    @property
    def renderer_events(self) -> asyncio.Queue[dict[str, object]] | None:
        return self._renderer_events

    @property
    def start_task(self) -> asyncio.Task[Any] | None:
        return self._start_task

    @property
    def monitor_task(self) -> asyncio.Task[Any] | None:
        return self._monitor_task

    @property
    def renderer_event_task(self) -> asyncio.Task[Any] | None:
        return self._renderer_event_task

    @property
    def child_task_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._child_task_names.values()))

    @property
    def is_closing(self) -> bool:
        return self._closing

    @property
    def is_closed(self) -> bool:
        return self._close_completed

    def has_resources(self) -> bool:
        return self._has_resources()

    def current_presenter_for_ingress(self) -> object | None:
        if self._closing or self._close_completed:
            return None
        return self._presenter

    def current_bridge_for_runtime_command(self) -> object | None:
        if self._closing or self._close_completed:
            return None
        return self._bridge

    def renderer_events_or_none(self) -> asyncio.Queue[dict[str, object]] | None:
        if self._closing or self._close_completed:
            return None
        return self._renderer_events

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    def is_current_instance_id(self, overlay_instance_id: str | None) -> bool:
        return (
            not self._closing
            and not self._close_completed
            and overlay_instance_id is not None
            and overlay_instance_id == self._overlay_instance_id
        )

    def set_overlay_instance_id(self, overlay_instance_id: str) -> None:
        self._overlay_instance_id = overlay_instance_id
        self._close_completed = False

    def attach_presenter(self, presenter: object | None) -> None:
        self._presenter = presenter

    def adopt_presenter(self, presenter: object | None) -> object | None:
        if presenter is None:
            self._presenter = None
            return None
        self._detach_presenter_runtime_resources(presenter)
        self._attach_presenter_runtime_resources(presenter)
        self._presenter = presenter
        return presenter

    def detach_preserved_presenter(self) -> object | None:
        presenter = self._presenter
        if presenter is None:
            return None
        if self._closing or not self._close_completed or self._has_owned_task_handles():
            raise RuntimeError(
                "Cannot detach preserved presenter before "
                "OverlayRuntimeHandle.close(preserve_presenter_state=True) completes"
            )
        self._detach_presenter_runtime_resources(presenter)
        self._presenter = None
        return presenter

    def attach_bridge(self, bridge: object | None) -> None:
        self._bridge = bridge

    def attach_process_manager(self, manager: object | None) -> None:
        self._process_manager = manager

    def attach_diagnostics(self, diagnostics: object | None) -> None:
        self._diagnostics = diagnostics

    def attach_renderer_events(
        self,
        renderer_events: asyncio.Queue[dict[str, object]] | None,
    ) -> None:
        self._renderer_events = renderer_events

    def _attach_presenter_runtime_resources(self, presenter: object) -> None:
        if hasattr(presenter, "diagnostics"):
            setattr(presenter, "diagnostics", self._diagnostics)
        if hasattr(presenter, "task_factory"):
            setattr(presenter, "task_factory", self.create_child_task)

    def _detach_presenter_runtime_resources(self, presenter: object) -> None:
        detach_bridge = getattr(presenter, "detach_bridge", None)
        if callable(detach_bridge):
            detach_bridge()
        if hasattr(presenter, "diagnostics"):
            setattr(presenter, "diagnostics", None)
        if hasattr(presenter, "task_factory"):
            setattr(presenter, "task_factory", None)

    def _has_owned_task_handles(self) -> bool:
        return any(
            task is not None
            for task in (
                self._start_task,
                self._monitor_task,
                self._renderer_event_task,
            )
        ) or bool(self._child_task_names)

    def create_start_task(
        self,
        coroutine: Coroutine[Any, Any, _TaskResultT],
    ) -> asyncio.Task[_TaskResultT]:
        task = self._create_task(coroutine, task_name="start")
        self._start_task = task
        task.add_done_callback(lambda completed: self._clear_task("start", completed))
        return task

    def create_monitor_task(
        self,
        coroutine: Coroutine[Any, Any, _TaskResultT],
    ) -> asyncio.Task[_TaskResultT]:
        task = self._create_task(coroutine, task_name="monitor")
        self._monitor_task = task
        task.add_done_callback(lambda completed: self._clear_task("monitor", completed))
        return task

    def create_renderer_event_task(
        self,
        coroutine: Coroutine[Any, Any, _TaskResultT],
    ) -> asyncio.Task[_TaskResultT]:
        task = self._create_task(coroutine, task_name="renderer-events")
        self._renderer_event_task = task
        task.add_done_callback(lambda completed: self._clear_task("renderer", completed))
        return task

    def create_child_task(
        self,
        coroutine: Coroutine[Any, Any, _TaskResultT],
        *,
        task_name: str,
    ) -> asyncio.Task[_TaskResultT]:
        task = self._create_task(coroutine, task_name=task_name)
        self._child_task_names[task] = task_name
        task.add_done_callback(self._clear_child_task)
        return task

    async def close(
        self,
        *,
        preserve_presenter_state: bool,
        overlay_sink_detach: Callable[[object | None], Awaitable[bool]] | None = None,
        preview_reset: Callable[[], Awaitable[None]] | None = None,
        diagnostics_detach: Callable[[object | None], object] | None = None,
        emit_shutdown: bool = True,
    ) -> None:
        if self._close_completed and not self._has_resources():
            return

        async with self._close_lock:
            if self._close_completed and not self._has_resources():
                return
            self._closing = True
            failures: list[Exception] = []
            output_ingress_detached = False
            output_ingress_detach_failed = False
            try:
                output_ingress_detached = await self._detach_overlay_ingress(
                    self._presenter,
                    overlay_sink_detach,
                )
            except Exception as exc:
                output_ingress_detach_failed = True
                failures.append(exc)
            try:
                if emit_shutdown:
                    await self._attempt(failures, self._mark_process_shutdown_requested)
                    await self._attempt(failures, self._broadcast_shutdown_with_grace)
                await self._cancel_owned_tasks(
                    failures,
                    preserve_child_task_prefixes=_PROCESS_EVENT_READER_TASK_PREFIXES,
                )
                await self._close_presenter(
                    preserve_presenter_state,
                    failures,
                    output_ingress_detached=output_ingress_detached,
                    output_ingress_detach_failed=output_ingress_detach_failed,
                    preview_reset=preview_reset,
                    diagnostics_detach=diagnostics_detach,
                )
                await self._stop_process_manager(failures)
                await self._cancel_owned_tasks(failures)
                await self._stop_bridge(failures)
                self._renderer_events = None
                if not preserve_presenter_state:
                    self._diagnostics = None
                if failures:
                    _raise_close_failures(failures, "overlay runtime close failed")
                self._close_completed = True
            finally:
                if failures:
                    self._closing = False
                else:
                    self._closing = False

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, _TaskResultT],
        *,
        task_name: str,
    ) -> asyncio.Task[_TaskResultT]:
        if self._closing or self._close_completed:
            coroutine.close()
            state = "closing" if self._closing else "closed"
            raise RuntimeError(f"OverlayRuntimeHandle is {state} to new tasks")
        self._close_completed = False
        return asyncio.create_task(coroutine, name=f"{self.owner_name}:{task_name}")

    def _clear_task(self, task_kind: str, completed: asyncio.Task[Any]) -> None:
        self._record_completed_task_failure(completed)
        if task_kind == "start" and self._start_task is completed:
            self._start_task = None
        elif task_kind == "monitor" and self._monitor_task is completed:
            self._monitor_task = None
        elif task_kind == "renderer" and self._renderer_event_task is completed:
            self._renderer_event_task = None

    def _clear_child_task(self, completed: asyncio.Task[Any]) -> None:
        self._record_completed_task_failure(completed)
        self._child_task_names.pop(completed, None)

    def _record_completed_task_failure(self, completed: asyncio.Task[Any]) -> None:
        if completed.cancelled():
            return
        try:
            failure = completed.exception()
        except asyncio.CancelledError:
            return
        if isinstance(failure, Exception):
            self._completed_task_failures[completed] = failure

    def _has_resources(self) -> bool:
        return any(
            resource is not None
            for resource in (
                self._presenter,
                self._bridge,
                self._process_manager,
                self._diagnostics,
                self._renderer_events,
                self._start_task,
                self._monitor_task,
                self._renderer_event_task,
                self._child_task_names or None,
            )
        )

    async def _attempt(
        self,
        failures: list[Exception],
        operation: object,
    ) -> None:
        try:
            if callable(operation):
                result = operation()
            else:
                result = operation
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            failures.append(exc)

    async def _broadcast_shutdown_with_grace(self) -> None:
        presenter = self._presenter
        if presenter is None:
            bridge = self._bridge
            if bridge is None:
                return
            broadcast_shutdown = getattr(bridge, "broadcast_shutdown", None)
        else:
            broadcast_shutdown = getattr(presenter, "broadcast_shutdown", None)
        if not callable(broadcast_shutdown):
            return
        result = broadcast_shutdown()
        if inspect.isawaitable(result):
            await result
        if self._shutdown_grace_s > 0:
            await asyncio.sleep(self._shutdown_grace_s)

    def _mark_process_shutdown_requested(self) -> None:
        manager = self._process_manager
        if manager is None:
            return
        mark_shutdown_requested = getattr(manager, "mark_shutdown_requested", None)
        if callable(mark_shutdown_requested):
            mark_shutdown_requested()

    async def _detach_overlay_ingress(
        self,
        presenter: object | None,
        overlay_sink_detach: Callable[[object | None], Awaitable[bool]] | None,
    ) -> bool:
        if presenter is None or overlay_sink_detach is None:
            return False
        return await overlay_sink_detach(presenter)

    async def _cancel_owned_tasks(
        self,
        failures: list[Exception],
        *,
        preserve_child_task_prefixes: tuple[str, ...] = (),
    ) -> None:
        current_task = asyncio.current_task()
        primary_tasks = tuple(
            task
            for task in (self._start_task, self._monitor_task, self._renderer_event_task)
            if task is not None and task is not current_task
        )
        child_tasks = tuple(
            task
            for task, task_name in self._child_task_names.items()
            if task is not current_task and not task_name.startswith(preserve_child_task_prefixes)
        )
        tasks = primary_tasks + child_tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for task, result in zip(tasks, results):
                self._completed_task_failures.pop(task, None)
                if isinstance(result, asyncio.CancelledError):
                    continue
                if isinstance(result, Exception):
                    failures.append(result)
        failures.extend(self._completed_task_failures.values())
        self._completed_task_failures.clear()
        if self._start_task is not current_task:
            self._start_task = None
        if self._monitor_task is not current_task:
            self._monitor_task = None
        if self._renderer_event_task is not current_task:
            self._renderer_event_task = None
        for task in child_tasks:
            self._child_task_names.pop(task, None)

    async def _close_presenter(
        self,
        preserve_presenter_state: bool,
        failures: list[Exception],
        *,
        output_ingress_detached: bool,
        output_ingress_detach_failed: bool,
        preview_reset: Callable[[], Awaitable[None]] | None,
        diagnostics_detach: Callable[[object | None], object] | None,
    ) -> None:
        presenter = self._presenter
        if presenter is None:
            return

        before = len(failures)
        if not preserve_presenter_state:
            presenter_close = getattr(presenter, "close", None)
            if callable(presenter_close):
                await self._attempt(failures, presenter_close)
            else:
                await self._attempt(failures, getattr(presenter, "clear_for_runtime_detach", None))

        detach_bridge = getattr(presenter, "detach_bridge", None)
        if callable(detach_bridge):
            try:
                detach_bridge()
            except Exception as exc:
                failures.append(exc)

        if not preserve_presenter_state and diagnostics_detach is not None:
            try:
                diagnostics_detach(self._diagnostics)
            except Exception as exc:
                failures.append(exc)
        if not preserve_presenter_state and output_ingress_detached:
            await self._attempt(failures, preview_reset)

        if preserve_presenter_state:
            return

        reset_scene = getattr(presenter, "reset_scene", None)
        if callable(reset_scene):
            try:
                reset_scene()
            except Exception as exc:
                failures.append(exc)
        if len(failures) == before and not output_ingress_detach_failed:
            self._presenter = None

    async def _stop_process_manager(self, failures: list[Exception]) -> None:
        manager = self._process_manager
        if manager is None:
            return
        before = len(failures)
        await self._attempt(failures, getattr(manager, "stop", None))
        if len(failures) == before:
            self._process_manager = None

    async def _stop_bridge(self, failures: list[Exception]) -> None:
        bridge = self._bridge
        if bridge is None:
            return
        before = len(failures)
        await self._attempt(failures, getattr(bridge, "stop", None))
        if len(failures) == before:
            self._bridge = None


def _raise_close_failures(failures: list[Exception], message: str) -> None:
    if len(failures) == 1:
        raise failures[0]
    raise ExceptionGroup(message, failures)


__all__ = ["OverlayRuntimeHandle"]
