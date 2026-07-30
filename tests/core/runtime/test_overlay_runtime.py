from __future__ import annotations

import asyncio
import inspect

import pytest

from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle


class FakePresenter:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.bridge: object | None = None
        self.diagnostics: object | None = None
        self.task_factory: object | None = None
        self.broadcast_shutdown_calls = 0
        self.clear_for_runtime_detach_calls = 0
        self.detach_bridge_calls = 0
        self.reset_scene_calls = 0

    async def broadcast_shutdown(self) -> None:
        self.broadcast_shutdown_calls += 1
        self.events.append("presenter.broadcast_shutdown")

    async def clear_for_runtime_detach(self) -> None:
        self.clear_for_runtime_detach_calls += 1
        self.events.append("presenter.clear_for_runtime_detach")

    def detach_bridge(self) -> None:
        self.detach_bridge_calls += 1
        self.bridge = None
        self.events.append("presenter.detach_bridge")

    def reset_scene(self) -> None:
        self.reset_scene_calls += 1
        self.events.append("presenter.reset_scene")


class IngressObservingPresenter(FakePresenter):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.output_projection: FakeOutputProjection | None = None
        self.ingress_detached_at_broadcast: bool | None = None

    async def broadcast_shutdown(self) -> None:
        self.broadcast_shutdown_calls += 1
        self.ingress_detached_at_broadcast = (
            self.output_projection is not None and self.output_projection.overlay_sink is None
        )
        self.events.append("presenter.broadcast_shutdown")


class FakeBridge:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1
        self.events.append("bridge.stop")


class FakeManager:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.stop_calls = 0
        self.mark_shutdown_requested_calls = 0

    def mark_shutdown_requested(self) -> None:
        self.mark_shutdown_requested_calls += 1
        self.events.append("manager.mark_shutdown_requested")

    async def stop(self) -> None:
        self.stop_calls += 1
        self.events.append("manager.stop")


class FailingOnceManager(FakeManager):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.fail_next_stop = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.events.append("manager.stop")
        if self.fail_next_stop:
            self.fail_next_stop = False
            raise RuntimeError("manager stop failed")


class FakeOutputProjection:
    def __init__(self, presenter: FakePresenter, diagnostics: object) -> None:
        _ = diagnostics
        self.overlay_sink = presenter
        self.reset_overlay_preview_calls = 0

    async def detach_overlay_sink(self, expected_current: object | None) -> bool:
        if self.overlay_sink is not expected_current:
            return False
        self.overlay_sink = None
        return True

    async def reset_overlay_preview(self) -> None:
        self.reset_overlay_preview_calls += 1


class FailingOnceOutputProjection(FakeOutputProjection):
    def __init__(self, presenter: FakePresenter, diagnostics: object) -> None:
        super().__init__(presenter, diagnostics)
        self.fail_next_detach = True

    async def detach_overlay_sink(self, expected_current: object | None) -> bool:
        if self.fail_next_detach:
            self.fail_next_detach = False
            raise RuntimeError("output detach failed")
        return await super().detach_overlay_sink(expected_current)


class DiagnosticsDetach:
    def __init__(self) -> None:
        self.calls: list[object | None] = []

    def __call__(self, expected_current: object | None) -> None:
        self.calls.append(expected_current)


async def _blocked_until_cancel(label: str, events: list[str]) -> None:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        events.append(f"{label}.cancelled")
        raise


async def _fail_cleanup_when_cancelled(label: str, events: list[str]) -> None:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError as exc:
        events.append(f"{label}.cleanup_failed")
        raise RuntimeError("task cleanup failed") from exc


async def _complete_work(label: str) -> str:
    return label


async def _fail_after_one_tick() -> None:
    await asyncio.sleep(0)
    raise RuntimeError("completed task failed")


def test_overlay_runtime_handle_exposes_lifecycle_inventory_and_policy() -> None:
    handle = OverlayRuntimeHandle()

    snapshot = handle.lifecycle_owner_snapshot()

    assert snapshot["owner"] == "OverlayRuntimeHandle"
    assert snapshot["resource_fields"] == OverlayRuntimeHandle.resource_fields
    for field_name in (
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
    ):
        assert field_name in snapshot["resource_fields"]
    assert snapshot["stop_ingress"] == "broadcast shutdown and reject new overlay commands"
    assert "async presenter close" in snapshot["shutdown_policy"]
    assert "kill escalation" in snapshot["shutdown_policy"]
    assert snapshot["late_callback_rule"] == (
        "old overlay instance events ignored after instance id changes"
    )


def test_overlay_runtime_handle_exposes_current_runtime_resources() -> None:
    events: list[str] = []
    presenter = FakePresenter(events)
    bridge = FakeBridge(events)
    manager = FakeManager(events)
    diagnostics = object()
    renderer_events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    handle = OverlayRuntimeHandle()

    handle.attach_presenter(presenter)
    handle.attach_bridge(bridge)
    handle.attach_process_manager(manager)
    handle.attach_diagnostics(diagnostics)
    handle.attach_renderer_events(renderer_events)

    assert handle.has_resources()
    assert handle.current_presenter_for_ingress() is presenter
    assert handle.current_bridge_for_runtime_command() is bridge
    assert handle.process_manager is manager
    assert handle.diagnostics is diagnostics
    assert handle.renderer_events_or_none() is renderer_events
    assert handle.start_task is None
    assert handle.monitor_task is None
    assert handle.renderer_event_task is None


@pytest.mark.asyncio
async def test_overlay_runtime_handle_rejects_preserved_presenter_detach_before_close() -> None:
    events: list[str] = []
    presenter = FakePresenter(events)
    bridge = FakeBridge(events)
    output_projection = FakeOutputProjection(presenter, object())
    handle = OverlayRuntimeHandle(shutdown_grace_s=0)
    handle.attach_presenter(presenter)
    handle.attach_bridge(bridge)
    presenter.bridge = bridge
    presenter.task_factory = handle.create_child_task
    child_task = handle.create_child_task(
        _blocked_until_cancel("presenter-refresh", events),
        task_name="presenter-refresh",
    )
    await asyncio.sleep(0)

    try:
        with pytest.raises(RuntimeError, match="close.*preserve_presenter_state=True"):
            handle.detach_preserved_presenter()
    finally:
        await handle.close(
            preserve_presenter_state=True,
            overlay_sink_detach=output_projection.detach_overlay_sink,
            preview_reset=output_projection.reset_overlay_preview,
            emit_shutdown=False,
        )

    assert child_task.done()
    assert handle.presenter is presenter
    assert presenter.bridge is None
    assert getattr(presenter.task_factory, "__self__", None) is handle
    assert (
        getattr(presenter.task_factory, "__func__", None) is OverlayRuntimeHandle.create_child_task
    )


@pytest.mark.asyncio
async def test_overlay_runtime_handle_detaches_and_adopts_preserved_presenter_after_close() -> None:
    events: list[str] = []
    old_diagnostics = object()
    new_diagnostics = object()
    stale_bridge = FakeBridge(events)
    presenter = FakePresenter(events)
    presenter.bridge = stale_bridge
    presenter.diagnostics = old_diagnostics
    old_runtime = OverlayRuntimeHandle()
    old_runtime.attach_presenter(presenter)
    old_runtime.attach_bridge(stale_bridge)
    old_runtime.attach_diagnostics(old_diagnostics)
    presenter.task_factory = old_runtime.create_child_task

    await old_runtime.close(
        preserve_presenter_state=True,
        emit_shutdown=False,
    )

    assert old_runtime.current_presenter_for_ingress() is None
    assert old_runtime.current_bridge_for_runtime_command() is None
    assert old_runtime.renderer_events_or_none() is None
    assert old_runtime.presenter is presenter

    preserved = old_runtime.detach_preserved_presenter()

    assert preserved is presenter
    assert old_runtime.presenter is None
    assert presenter.bridge is None
    assert presenter.diagnostics is None
    assert presenter.task_factory is None
    assert presenter.detach_bridge_calls >= 1
    assert old_runtime.bridge is None

    new_runtime = OverlayRuntimeHandle()
    new_runtime.attach_diagnostics(new_diagnostics)

    adopted = new_runtime.adopt_presenter(preserved)

    assert adopted is presenter
    assert new_runtime.presenter is presenter
    assert presenter.bridge is None
    assert presenter.diagnostics is new_diagnostics
    assert getattr(presenter.task_factory, "__self__", None) is new_runtime
    assert (
        getattr(presenter.task_factory, "__func__", None) is OverlayRuntimeHandle.create_child_task
    )


@pytest.mark.asyncio
async def test_overlay_runtime_handle_close_controls_tasks_and_resources() -> None:
    events: list[str] = []
    diagnostics = object()
    presenter = FakePresenter(events)
    bridge = FakeBridge(events)
    manager = FakeManager(events)
    output_projection = FakeOutputProjection(presenter, diagnostics)
    diagnostics_detach = DiagnosticsDetach()
    handle = OverlayRuntimeHandle(shutdown_grace_s=0)
    handle.attach_presenter(presenter)
    handle.attach_bridge(bridge)
    handle.attach_process_manager(manager)
    handle.attach_diagnostics(diagnostics)
    handle.attach_renderer_events(asyncio.Queue())

    start_task = handle.create_start_task(_blocked_until_cancel("start", events))
    monitor_task = handle.create_monitor_task(_blocked_until_cancel("monitor", events))
    renderer_task = handle.create_renderer_event_task(_blocked_until_cancel("renderer", events))
    await asyncio.sleep(0)

    await handle.close(
        preserve_presenter_state=False,
        overlay_sink_detach=output_projection.detach_overlay_sink,
        preview_reset=output_projection.reset_overlay_preview,
        diagnostics_detach=diagnostics_detach,
    )

    assert start_task.done()
    assert monitor_task.done()
    assert renderer_task.done()
    assert events == [
        "manager.mark_shutdown_requested",
        "presenter.broadcast_shutdown",
        "start.cancelled",
        "monitor.cancelled",
        "renderer.cancelled",
        "presenter.clear_for_runtime_detach",
        "presenter.detach_bridge",
        "presenter.reset_scene",
        "manager.stop",
        "bridge.stop",
    ]
    assert presenter.broadcast_shutdown_calls == 1
    assert presenter.clear_for_runtime_detach_calls == 1
    assert presenter.detach_bridge_calls == 1
    assert presenter.reset_scene_calls == 1
    assert manager.stop_calls == 1
    assert manager.mark_shutdown_requested_calls == 1
    assert bridge.stop_calls == 1
    assert output_projection.overlay_sink is None
    assert diagnostics_detach.calls == [diagnostics]
    assert output_projection.reset_overlay_preview_calls == 1
    assert handle.presenter is None
    assert handle.bridge is None
    assert handle.process_manager is None
    assert handle.renderer_events is None
    assert handle.start_task is None
    assert handle.monitor_task is None
    assert handle.renderer_event_task is None

    await handle.close(
        preserve_presenter_state=False,
        overlay_sink_detach=output_projection.detach_overlay_sink,
        preview_reset=output_projection.reset_overlay_preview,
        diagnostics_detach=diagnostics_detach,
    )
    assert presenter.broadcast_shutdown_calls == 1
    assert manager.stop_calls == 1
    assert bridge.stop_calls == 1


@pytest.mark.asyncio
async def test_overlay_runtime_handle_close_detaches_output_ingress_before_shutdown_broadcast() -> (
    None
):
    events: list[str] = []
    diagnostics = object()
    presenter = IngressObservingPresenter(events)
    output_projection = FakeOutputProjection(presenter, diagnostics)
    diagnostics_detach = DiagnosticsDetach()
    presenter.output_projection = output_projection
    handle = OverlayRuntimeHandle(shutdown_grace_s=0)
    handle.attach_presenter(presenter)
    handle.attach_diagnostics(diagnostics)

    await handle.close(
        preserve_presenter_state=False,
        overlay_sink_detach=output_projection.detach_overlay_sink,
        preview_reset=output_projection.reset_overlay_preview,
        diagnostics_detach=diagnostics_detach,
    )

    assert presenter.ingress_detached_at_broadcast is True
    assert presenter.broadcast_shutdown_calls == 1
    assert output_projection.overlay_sink is None
    assert diagnostics_detach.calls == [diagnostics]
    assert output_projection.reset_overlay_preview_calls == 1


@pytest.mark.asyncio
async def test_overlay_runtime_handle_retains_presenter_for_output_detach_retry() -> None:
    events: list[str] = []
    presenter = FakePresenter(events)
    manager = FakeManager(events)
    output_projection = FailingOnceOutputProjection(presenter, object())
    handle = OverlayRuntimeHandle(shutdown_grace_s=0)
    handle.attach_presenter(presenter)
    handle.attach_process_manager(manager)

    with pytest.raises(RuntimeError, match="output detach failed"):
        await handle.close(
            preserve_presenter_state=False,
            overlay_sink_detach=output_projection.detach_overlay_sink,
            preview_reset=output_projection.reset_overlay_preview,
        )

    assert handle.presenter is presenter
    assert output_projection.overlay_sink is presenter
    assert manager.stop_calls == 1

    await handle.close(
        preserve_presenter_state=False,
        overlay_sink_detach=output_projection.detach_overlay_sink,
        preview_reset=output_projection.reset_overlay_preview,
    )

    assert handle.presenter is None
    assert output_projection.overlay_sink is None
    assert output_projection.reset_overlay_preview_calls == 1


@pytest.mark.asyncio
async def test_overlay_runtime_marks_shutdown_before_broadcast_grace_exit_and_stop() -> None:
    events: list[str] = []

    class ExitingPresenter(FakePresenter):
        async def broadcast_shutdown(self) -> None:
            self.broadcast_shutdown_calls += 1
            self.events.append("presenter.broadcast_shutdown")
            await asyncio.sleep(0)
            self.events.append("native.exit:0")

    presenter = ExitingPresenter(events)
    manager = FakeManager(events)
    handle = OverlayRuntimeHandle(shutdown_grace_s=0.001)
    handle.attach_presenter(presenter)
    handle.attach_process_manager(manager)

    await handle.close(preserve_presenter_state=True)

    assert events == [
        "manager.mark_shutdown_requested",
        "presenter.broadcast_shutdown",
        "native.exit:0",
        "presenter.detach_bridge",
        "manager.stop",
    ]


@pytest.mark.asyncio
async def test_overlay_runtime_handle_close_cancels_child_tasks() -> None:
    events: list[str] = []
    handle = OverlayRuntimeHandle(shutdown_grace_s=0)

    child_task = handle.create_child_task(
        _blocked_until_cancel("presenter-expiration", events),
        task_name="presenter-expiration",
    )
    await asyncio.sleep(0)

    await handle.close(preserve_presenter_state=True)

    assert child_task.done()
    assert events == ["presenter-expiration.cancelled"]
    assert handle.child_task_names == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_task_kind", ["start", "child"])
async def test_overlay_runtime_handle_close_surfaces_owned_task_cleanup_failures(
    owned_task_kind: str,
) -> None:
    events: list[str] = []
    presenter = FakePresenter(events)
    bridge = FakeBridge(events)
    manager = FakeManager(events)
    output_projection = FakeOutputProjection(presenter, object())
    handle = OverlayRuntimeHandle(shutdown_grace_s=0)
    handle.attach_presenter(presenter)
    handle.attach_bridge(bridge)
    handle.attach_process_manager(manager)

    if owned_task_kind == "start":
        task = handle.create_start_task(
            _fail_cleanup_when_cancelled("start", events),
        )
        expected_failure_event = "start.cleanup_failed"
    else:
        task = handle.create_child_task(
            _fail_cleanup_when_cancelled("presenter-expiration", events),
            task_name="presenter-expiration",
        )
        expected_failure_event = "presenter-expiration.cleanup_failed"
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="task cleanup failed"):
        await handle.close(
            preserve_presenter_state=False,
            overlay_sink_detach=output_projection.detach_overlay_sink,
            preview_reset=output_projection.reset_overlay_preview,
        )

    assert task.done()
    assert events == [
        "manager.mark_shutdown_requested",
        "presenter.broadcast_shutdown",
        expected_failure_event,
        "presenter.clear_for_runtime_detach",
        "presenter.detach_bridge",
        "presenter.reset_scene",
        "manager.stop",
        "bridge.stop",
    ]
    assert presenter.clear_for_runtime_detach_calls == 1
    assert manager.stop_calls == 1
    assert bridge.stop_calls == 1
    assert handle.presenter is None
    assert handle.process_manager is None
    assert handle.bridge is None
    assert handle.start_task is None
    assert handle.child_task_names == ()
    assert not handle.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_task_kind", ["start", "child"])
async def test_overlay_runtime_handle_close_surfaces_owned_task_failures_completed_before_close(
    owned_task_kind: str,
) -> None:
    handle = OverlayRuntimeHandle(shutdown_grace_s=0)

    if owned_task_kind == "start":
        task = handle.create_start_task(_fail_after_one_tick())
    else:
        task = handle.create_child_task(
            _fail_after_one_tick(),
            task_name="completed-child",
        )
    for _ in range(3):
        await asyncio.sleep(0)
    assert task.done()

    with pytest.raises(RuntimeError, match="completed task failed"):
        try:
            await handle.close(preserve_presenter_state=True)
        finally:
            if task.done() and not task.cancelled():
                task.exception()


@pytest.mark.asyncio
async def test_overlay_runtime_handle_close_keeps_failed_resources_for_retry() -> None:
    events: list[str] = []
    presenter = FakePresenter(events)
    bridge = FakeBridge(events)
    manager = FailingOnceManager(events)
    handle = OverlayRuntimeHandle(shutdown_grace_s=0)
    handle.attach_presenter(presenter)
    handle.attach_bridge(bridge)
    handle.attach_process_manager(manager)

    with pytest.raises(RuntimeError, match="manager stop failed"):
        await handle.close(preserve_presenter_state=True)

    assert handle.process_manager is manager
    assert handle.bridge is None
    assert bridge.stop_calls == 1

    await handle.close(preserve_presenter_state=True)

    assert handle.process_manager is None
    assert manager.stop_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("preserve_presenter_state", [False, True])
async def test_overlay_runtime_handle_rejects_new_work_after_successful_close(
    preserve_presenter_state: bool,
) -> None:
    events: list[str] = []
    handle = OverlayRuntimeHandle(shutdown_grace_s=0)
    handle.attach_presenter(FakePresenter(events))

    await handle.close(preserve_presenter_state=preserve_presenter_state)

    child_coroutine = _complete_work("stale-child")
    with pytest.raises(RuntimeError, match="closed to new tasks"):
        handle.create_child_task(child_coroutine, task_name="stale-child")
    assert inspect.getcoroutinestate(child_coroutine) is inspect.CORO_CLOSED
    assert handle.child_task_names == ()

    for factory_name, task_attr in (
        ("create_start_task", "start_task"),
        ("create_monitor_task", "monitor_task"),
        ("create_renderer_event_task", "renderer_event_task"),
    ):
        primary_coroutine = _complete_work(factory_name)
        with pytest.raises(RuntimeError, match="closed to new tasks"):
            getattr(handle, factory_name)(primary_coroutine)
        assert inspect.getcoroutinestate(primary_coroutine) is inspect.CORO_CLOSED
        assert getattr(handle, task_attr) is None


@pytest.mark.asyncio
async def test_overlay_runtime_handle_allows_work_after_explicit_generation_activation() -> None:
    handle = OverlayRuntimeHandle(
        overlay_instance_id="overlay-old",
        shutdown_grace_s=0,
    )

    await handle.close(preserve_presenter_state=False)

    stale_coroutine = _complete_work("stale")
    with pytest.raises(RuntimeError, match="closed to new tasks"):
        handle.create_child_task(stale_coroutine, task_name="stale")
    assert inspect.getcoroutinestate(stale_coroutine) is inspect.CORO_CLOSED

    handle.set_overlay_instance_id("overlay-new")
    task = handle.create_child_task(_complete_work("new"), task_name="new")

    assert await task == "new"
    assert handle.is_current_instance_id("overlay-new")
    assert not handle.is_current_instance_id("overlay-old")


@pytest.mark.asyncio
async def test_overlay_runtime_handle_closed_generation_is_not_current_until_reactivated() -> None:
    handle = OverlayRuntimeHandle(
        overlay_instance_id="overlay-old",
        shutdown_grace_s=0,
    )
    assert handle.is_current_instance_id("overlay-old")

    await handle.close(preserve_presenter_state=False)

    assert not handle.is_current_instance_id("overlay-old")

    handle.set_overlay_instance_id("overlay-new")

    assert handle.is_current_instance_id("overlay-new")


@pytest.mark.asyncio
async def test_overlay_runtime_handle_attach_after_close_does_not_reopen_without_generation() -> (
    None
):
    handle = OverlayRuntimeHandle(shutdown_grace_s=0)

    await handle.close(preserve_presenter_state=False)

    handle.attach_diagnostics(object())
    coroutine = _complete_work("attached-without-generation")
    with pytest.raises(RuntimeError, match="closed to new tasks"):
        handle.create_child_task(coroutine, task_name="attached-without-generation")

    assert inspect.getcoroutinestate(coroutine) is inspect.CORO_CLOSED
