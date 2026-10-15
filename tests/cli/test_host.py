from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from puripuly_heart.cli.host import GuiControlBoundary
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
