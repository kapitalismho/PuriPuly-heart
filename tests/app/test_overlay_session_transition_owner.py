from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from puripuly_heart.app.services.overlay_session_transition import (
    OverlaySessionShutdownExecution,
    OverlaySessionStartExecution,
    OverlaySessionTransitionDiagnostic,
    OverlaySessionTransitionOwner,
)
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle


@dataclass
class StartHarness:
    state: str = "off"
    teardown_result: bool = True
    events: list[str] = field(default_factory=list)
    teardown_started: asyncio.Event = field(default_factory=asyncio.Event)
    teardown_release: asyncio.Event = field(default_factory=asyncio.Event)
    start_started: asyncio.Event = field(default_factory=asyncio.Event)
    start_release: asyncio.Event = field(default_factory=asyncio.Event)
    runtime: OverlayRuntimeHandle = field(
        default_factory=lambda: OverlayRuntimeHandle(shutdown_grace_s=0)
    )
    previous_runtime: OverlayRuntimeHandle | None = None

    async def teardown(self) -> bool:
        self.events.append("teardown")
        self.teardown_started.set()
        await self.teardown_release.wait()
        return self.teardown_result

    def create_runtime(self) -> OverlayRuntimeHandle:
        self.events.append("create")
        return self.runtime

    def on_starting(self, runtime: OverlayRuntimeHandle, target: str) -> None:
        assert runtime is self.runtime
        self.events.append(f"starting:{target}")

    async def run_start(self, runtime: OverlayRuntimeHandle) -> None:
        assert runtime is self.runtime
        self.events.append("run")
        self.start_started.set()
        await self.start_release.wait()

    def execution(self) -> OverlaySessionStartExecution:
        return OverlaySessionStartExecution(
            state=self.state,
            previous_runtime=self.previous_runtime,
            teardown=self.teardown,
            create_runtime=self.create_runtime,
            resolve_target=lambda: "desktop",
            on_starting=self.on_starting,
            run_start=self.run_start,
        )


@dataclass
class ShutdownHarness:
    state: str = "connected"
    has_resources: bool = True
    teardown_result: bool = True
    retained_resources: bool = False
    events: list[str] = field(default_factory=list)
    teardown_started: asyncio.Event = field(default_factory=asyncio.Event)
    teardown_release: asyncio.Event = field(default_factory=asyncio.Event)

    async def teardown(self) -> bool:
        self.events.append("teardown")
        self.teardown_started.set()
        await self.teardown_release.wait()
        return self.teardown_result

    async def on_failed(self) -> None:
        self.events.append("failed")

    async def on_stopped(self) -> None:
        self.events.append("stopped")

    def execution(self) -> OverlaySessionShutdownExecution:
        return OverlaySessionShutdownExecution(
            state=self.state,
            has_resources=self.has_resources,
            teardown=self.teardown,
            has_resources_after_teardown=lambda: self.retained_resources,
            on_stopping=lambda: self.events.append("stopping"),
            on_failed=self.on_failed,
            on_stopped=self.on_stopped,
        )


@pytest.mark.asyncio
async def test_start_skips_an_already_active_session() -> None:
    harness = StartHarness(state="connected")
    owner = OverlaySessionTransitionOwner()

    assert await owner.begin_start(harness.execution) == "already_active"
    assert harness.events == []


@pytest.mark.asyncio
async def test_start_preserves_presenter_before_new_generation_task() -> None:
    presenter = object()
    previous = OverlayRuntimeHandle(shutdown_grace_s=0)
    previous.attach_presenter(presenter)
    await previous.close(
        preserve_presenter_state=True,
        emit_shutdown=False,
    )
    harness = StartHarness(previous_runtime=previous)
    harness.teardown_release.set()
    owner = OverlaySessionTransitionOwner()

    assert await owner.begin_start(harness.execution) == "started"
    await harness.start_started.wait()

    assert harness.runtime.presenter is presenter
    assert harness.events == ["teardown", "create", "starting:desktop", "run"]

    harness.start_release.set()
    task = harness.runtime.start_task
    if task is not None:
        await task


@pytest.mark.asyncio
async def test_start_stops_when_previous_teardown_fails() -> None:
    harness = StartHarness(teardown_result=False)
    harness.teardown_release.set()

    assert await OverlaySessionTransitionOwner().begin_start(harness.execution) == "teardown_failed"
    assert harness.events == ["teardown"]
    assert harness.runtime.start_task is None


@pytest.mark.asyncio
async def test_shutdown_skips_empty_off_session() -> None:
    harness = ShutdownHarness(state="off", has_resources=False)

    assert await OverlaySessionTransitionOwner().shutdown(harness.execution) == "already_off"
    assert harness.events == []


@pytest.mark.asyncio
async def test_shutdown_publishes_failure_when_teardown_retains_resources() -> None:
    harness = ShutdownHarness(
        teardown_result=False,
        retained_resources=True,
    )
    harness.teardown_release.set()

    assert await OverlaySessionTransitionOwner().shutdown(harness.execution) == "failed"
    assert harness.events == ["stopping", "teardown", "failed"]


@pytest.mark.asyncio
async def test_shutdown_completes_when_teardown_releases_resources() -> None:
    harness = ShutdownHarness(
        teardown_result=False,
        retained_resources=False,
    )
    harness.teardown_release.set()

    assert await OverlaySessionTransitionOwner().shutdown(harness.execution) == "stopped"
    assert harness.events == ["stopping", "teardown", "stopped"]


@pytest.mark.asyncio
async def test_start_and_shutdown_share_one_transition_admission_lock() -> None:
    owner = OverlaySessionTransitionOwner()
    start = StartHarness()
    shutdown = ShutdownHarness()
    shutdown.teardown_release.set()
    factory_events: list[str] = []

    def start_factory() -> OverlaySessionStartExecution:
        factory_events.append("start")
        return start.execution()

    def shutdown_factory() -> OverlaySessionShutdownExecution:
        factory_events.append("shutdown")
        return shutdown.execution()

    start_task = asyncio.create_task(owner.begin_start(start_factory))
    await start.teardown_started.wait()
    shutdown_task = asyncio.create_task(owner.shutdown(shutdown_factory))
    await asyncio.sleep(0)

    assert factory_events == ["start"]

    start.teardown_result = False
    start.teardown_release.set()
    await asyncio.gather(start_task, shutdown_task)
    assert factory_events == ["start", "shutdown"]


@pytest.mark.asyncio
async def test_cancelled_transition_releases_admission_lock_and_reports_metadata() -> None:
    diagnostics: list[OverlaySessionTransitionDiagnostic] = []
    owner = OverlaySessionTransitionOwner(diagnostic_sink=diagnostics.append)
    start = StartHarness()
    task = asyncio.create_task(owner.begin_start(start.execution))
    await start.teardown_started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    shutdown = ShutdownHarness(state="off", has_resources=False)
    assert await owner.shutdown(shutdown.execution) == "already_off"
    assert diagnostics[0].outcome == "cancelled"
    assert diagnostics[1].outcome == "already_off"


def test_owner_declares_cross_generation_transition_policy() -> None:
    assert OverlaySessionTransitionOwner().lifecycle_owner_snapshot() == {
        "owner": "OverlaySessionTransitionOwner",
        "resource_fields": ("_lock",),
        "operation_policy": ("serialize cross-generation overlay start and shutdown transitions"),
        "cancellation_policy": "propagate cancellation without admitting another transition",
        "shutdown_policy": (
            "delegate generation teardown to OverlayRuntimeHandle before publishing completion"
        ),
    }
