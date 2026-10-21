from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from puripuly_heart.cli.host import GuiControlBoundary, HostedApplication
from puripuly_heart.cli.transport import request


@pytest.mark.asyncio
async def test_remote_gui_stop_closes_presentation_after_ordered_runtime_shutdown():
    events = []
    shutdown_started = asyncio.Event()
    release_shutdown = asyncio.Event()
    presentation_closed = asyncio.Event()

    class Control:
        def __init__(self):
            self.stop_requested = asyncio.Event()
            self.frozen = False

        def bind_instance(self, instance_id):
            self.instance_id = instance_id

        async def submit(self, name, arguments, *, request_id, expected_revision=None):
            assert name == "app.stop"
            assert arguments == {}
            events.append("stop-requested")
            self.stop_requested.set()
            return {
                "instance_id": self.instance_id,
                "operation_id": request_id,
                "status": "accepted",
                "terminal": False,
            }

        async def wait_for_stop_request(self):
            await self.stop_requested.wait()

        def freeze_ingress(self):
            self.frozen = True

    class Lease:
        closed = False
        record = None

        def publish(self, **endpoint):
            self.record = SimpleNamespace(**endpoint)

        def close(self):
            self.closed = True

    control = Control()
    lease = Lease()

    class Boundary:
        def control(self):
            return control

        def register_application_shutdown_callbacks(self, callbacks):
            self.callbacks = callbacks

        async def start(self):
            events.append("runtime-started")

        async def stop(self):
            assert control.frozen
            events.append("runtime-stopping")
            shutdown_started.set()
            await release_shutdown.wait()
            events.append("runtime-closed")

        async def close_presentation(self):
            assert events[-1] == "runtime-closed"
            assert lease.closed
            events.append("presentation-closed")
            presentation_closed.set()

    gui = GuiControlBoundary(Boundary(), lease)
    await gui.start()
    try:
        receipt = await request(
            lease.record,
            "submit",
            {"name": "app.stop", "values": {}, "request_id": "stop-operation"},
        )
        assert receipt == {
            "instance_id": lease.record.instance_id,
            "operation_id": "stop-operation",
            "status": "accepted",
            "terminal": False,
        }
        await asyncio.wait_for(shutdown_started.wait(), 2)
        assert not presentation_closed.is_set()
        assert not lease.closed
        release_shutdown.set()
        await asyncio.wait_for(presentation_closed.wait(), 2)
    finally:
        release_shutdown.set()
        await gui.stop()

    assert events == [
        "runtime-started",
        "stop-requested",
        "runtime-stopping",
        "runtime-closed",
        "presentation-closed",
    ]


@pytest.fixture
def controlled_host():
    control = SimpleNamespace(bind_instance=Mock(), freeze_ingress=Mock())
    boundary = SimpleNamespace(
        control=lambda: control,
        register_application_shutdown_callbacks=Mock(),
        start=AsyncMock(),
        stop=AsyncMock(),
    )
    lease = SimpleNamespace(publish=Mock(), close=Mock())
    host = HostedApplication(boundary, lease)
    host.server = SimpleNamespace(
        start=AsyncMock(return_value=32123),
        close=AsyncMock(),
        freeze=Mock(),
    )
    return host


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["ready", "cancel", "failure", "close"])
async def test_control_publication_keeps_loop_responsive_and_owns_lease_until_drained(
    controlled_host, outcome
):
    host = controlled_host
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    events = []
    threads = []

    def publish(**endpoint):
        threads.append(threading.get_ident())
        assert endpoint == {
            "port": 32123,
            "token": host.token,
            "instance_id": host.instance_id,
        }
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(3)
        host.lease.close.assert_not_called()
        events.append("publication-finished")
        if outcome == "failure":
            raise OSError("synthetic publication failure")

    host.lease.publish.side_effect = publish
    host.server.close.side_effect = lambda: events.append("server-closed")
    host.lease.close.side_effect = lambda: events.append("lease-closed")
    start = asyncio.create_task(host.start())
    close = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0)
        assert not start.done()
        assert threads != [loop_thread]
        if outcome == "cancel":
            start.cancel()
            await asyncio.sleep(0)
            start.cancel()
        elif outcome == "close":
            close = asyncio.create_task(host.close())
        await asyncio.sleep(0)
        if outcome != "ready":
            assert not start.done()
        host.lease.close.assert_not_called()
        release.set()
        if outcome in {"cancel", "close"}:
            with pytest.raises(asyncio.CancelledError):
                await start
        elif outcome == "failure":
            with pytest.raises(OSError, match="publication failure"):
                await start
        else:
            await start
            host.server.close.assert_not_awaited()
            host.lease.close.assert_not_called()
        if close is not None:
            await close
        if outcome != "ready":
            assert events.index("publication-finished") < events.index("lease-closed")
            host.server.close.assert_awaited()
            host.lease.close.assert_called()
    finally:
        release.set()
        await asyncio.gather(start, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await host.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_start", "boundary_start", "server_start"])
async def test_close_before_publication_never_advertises_or_resurrects_endpoint(
    controlled_host, phase
):
    host = controlled_host
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_start():
        entered.set()
        await release.wait()
        return 32123

    if phase == "before_start":
        await host.close()
    elif phase == "boundary_start":
        host.boundary.start.side_effect = gated_start
    else:
        host.server.start.side_effect = gated_start
    start = asyncio.create_task(host.start())
    close = None
    try:
        if phase != "before_start":
            await asyncio.wait_for(entered.wait(), 1)
            close = asyncio.create_task(host.close())
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await start
        if close is not None:
            await close
        host.lease.publish.assert_not_called()
        host.lease.close.assert_called()
        if phase == "before_start":
            host.boundary.start.assert_not_awaited()
        if phase != "server_start":
            host.server.start.assert_not_awaited()
    finally:
        release.set()
        await asyncio.gather(start, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await host.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_interrupted_server_start_closes_endpoint_without_publication(
    controlled_host, failure
):
    host = controlled_host
    entered = asyncio.Event()

    async def interrupted():
        entered.set()
        if failure:
            raise OSError("synthetic server startup failure")
        await asyncio.Event().wait()

    host.server.start.side_effect = interrupted
    start = asyncio.create_task(host.start())
    await asyncio.wait_for(entered.wait(), 1)
    if not failure:
        start.cancel()
    with pytest.raises(OSError if failure else asyncio.CancelledError):
        await start
    host.lease.publish.assert_not_called()
    host.server.close.assert_awaited_once()
    host.lease.close.assert_called_once()
