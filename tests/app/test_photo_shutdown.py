from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from websockets.asyncio.client import connect

from puripuly_heart.app.services.application_shutdown import (
    DEFAULT_APPLICATION_SHUTDOWN_CALLBACK_TIMEOUT_SECONDS,
    ApplicationShutdownCoordinator,
    application_shutdown_callback,
)
from puripuly_heart.app.services.photo_application import PhotoApplicationOwner
from puripuly_heart.core import network_clients
from puripuly_heart.core.lifecycle import (
    SHUTDOWN_PHASE_FREEZE_INGRESS,
    SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
    SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
    LifecycleDiagnosticsUnavailableError,
)
from puripuly_heart.core.overlay.process_adapter import _AsyncioOverlayProcess
from puripuly_heart.core.photo.lifecycle import PhotoRoute
from puripuly_heart.core.photo.process import PhotoProcessRunner
from puripuly_heart.providers.photo import LunaPhotoConnection
from tests.app.test_photo_application import Watcher, captured, eventually
from tests.providers.test_photo import response_delta, response_item

_RENDERER = r"""
import asyncio
import json
import sys
from pathlib import Path

from websockets.asyncio.client import connect


async def wait_file(path):
    while not path.exists():
        await asyncio.sleep(0.01)


async def run():
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    signals = Path(sys.argv[2])
    generation = manifest["generation"]
    await wait_file(signals / "authenticate")
    async with connect(manifest["bridge_url"], proxy=None) as socket:
        async def send(kind, **fields):
            await socket.send(json.dumps({"type": kind, **fields}))

        await send("photo_auth", **{
            key: manifest[key] for key in (
                "protocol_version", "role", "session_token", "instance_id", "generation"
            )
        })
        await send("photo_ready")
        pulled = False
        preparation = None
        async for raw in socket:
            message = json.loads(raw)
            kind = message["type"]
            if kind == "photo_snapshot":
                photo = message["photo"] or {}
                await send(
                    "photo_state_received", photo_id=photo.get("photo_id"),
                    attempt_id=photo.get("attempt_id", 0), revision=message["revision"],
                    block_count=len(photo.get("blocks", [])),
                )
                if photo.get("offer_open") and not pulled and generation == 1:
                    pulled = True
                    await send("photo_pull", photo_id=photo["photo_id"], event_id="controlled")
            elif kind == "photo_prepare":
                preparation = message
                (signals / "preparing").touch()
                if not (signals / "hold-preparation").exists():
                    await prepared(send, preparation)
                    preparation = None
            elif kind == "shutdown":
                (signals / f"shutdown-{generation}").touch()
                await wait_file(signals / "settle")
                if (signals / "fail-reader").exists():
                    print("x" * 70000, file=sys.stderr, flush=True)
                    return
                if preparation is not None:
                    await prepared(send, preparation)
                await send("photo_shutdown_complete")
                return


async def prepared(send, preparation):
    directory = Path(preparation["output_dir"])
    source = directory / "source.png"
    jpeg = directory / "send.jpg"
    source.write_bytes(b"synthetic source")
    jpeg.write_bytes(b"\xff\xd8synthetic jpeg\xff\xd9")
    await send(
        "photo_prepared", photo_id=preparation["photo_id"], attempt_id=preparation["attempt_id"],
        source_path=str(source), jpeg_path=str(jpeg), width=1280, height=720, scale=1,
        offset_x=0, offset_y=0, preparation_ms=1,
    )


asyncio.run(run())
"""


class ControlledRenderer(PhotoProcessRunner):
    def __init__(self, signals):
        self.signals = signals
        self.children = []
        self.spawn_release = asyncio.Event()
        self.spawn_release.set()

    def prepare(self, manifest):
        self.url = manifest["bridge_url"]
        return Path(sys.executable)

    async def spawn(self, executable, manifest_path, *, task_factory):
        child = _AsyncioOverlayProcess(
            process=await asyncio.create_subprocess_exec(
                str(executable),
                "-c",
                _RENDERER,
                str(manifest_path),
                str(self.signals),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            task_factory=task_factory,
        )
        self.children.append(child)
        await self.spawn_release.wait()
        return child


class ControlledProvider:
    def __init__(self):
        self.sent = []
        self.responses = asyncio.Queue()
        self.closing = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.allow_close.set()
        self.closed = 0

    async def access_token(self):
        return "synthetic-token"

    async def connect(self, url, *, additional_headers, **kwargs):
        assert url == "wss://api.openai.com/v1/responses"
        assert additional_headers == {"Authorization": "Bearer synthetic-token"}
        return self

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        return json.dumps(await self.responses.get())

    async def close(self):
        self.closing.set()
        await self.allow_close.wait()
        self.closed += 1


@pytest.fixture
async def photo(tmp_path, monkeypatch, request):
    owned = tmp_path / "owned"
    owned.mkdir()
    renderer = ControlledRenderer(tmp_path)
    provider = ControlledProvider()
    observations = []
    monkeypatch.setattr(network_clients, "external_websocket_connect", provider.connect)

    async def route():
        return PhotoRoute("luna", lambda model: LunaPhotoConnection(provider))

    watcher = Watcher()
    owner = PhotoApplicationOwner(
        resolve_route=route,
        target_language=lambda: "ko",
        diagnostic=lambda stage, fields: observations.append((stage, fields)),
        watcher=watcher,
        runner=renderer,
        temporary_root=owned,
    )
    allow_drain = asyncio.Event()
    allow_drain.set()

    async def stop_external_producers():
        await allow_drain.wait()

    coordinator = ApplicationShutdownCoordinator(
        (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
                owner_name="PhotoApplicationOwner",
                callback_name="stop_ingress",
                callback=owner.stop_ingress,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="ControlledProducer",
                callback_name="stop",
                callback=stop_external_producers,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="PhotoApplicationOwner",
                callback_name="close",
                callback=owner.close,
                timeout_seconds=getattr(
                    request, "param", DEFAULT_APPLICATION_SHUTDOWN_CALLBACK_TIMEOUT_SECONDS
                ),
            ),
        )
    )
    await owner.start()
    try:
        yield SimpleNamespace(
            owner=owner,
            provider=provider,
            renderer=renderer,
            coordinator=coordinator,
            owned=owned,
            signals=tmp_path,
            observations=observations,
            allow_drain=allow_drain,
            offer=lambda: watcher.emit(captured(tmp_path)),
        )
    finally:
        (tmp_path / "authenticate").touch()
        (tmp_path / "settle").touch()
        provider.allow_close.set()
        renderer.spawn_release.set()
        allow_drain.set()
        for child in renderer.children:
            if child.returncode is None:
                await child.terminate()
        if not coordinator.snapshot.terminal:
            await asyncio.wait_for(coordinator.shutdown(), 5)


async def assert_settled(photo):
    assert photo.owner.lifecycle.snapshot["photo"] is None
    assert photo.provider.closed == 1
    assert not photo.owner.bridge.connected
    assert photo.owner.supervisor.process is None
    assert all(child.returncode is not None for child in photo.renderer.children)
    assert all(task.done() for child in photo.renderer.children for task in child._reader_tasks)
    assert not photo.owner.bridge._preparations
    assert not photo.owner.lifecycle._retired
    for scope in (
        photo.owner._scope,
        photo.owner.bridge._scope,
        photo.owner.supervisor._scope,
        photo.owner.lifecycle._scope,
        photo.owner.lifecycle._release_scope,
    ):
        assert not scope.active_task_names
    assert not list(photo.owned.iterdir())
    with pytest.raises(OSError):
        await connect(photo.renderer.url, proxy=None)


@pytest.mark.parametrize("acknowledge", [True, False])
async def test_shutdown_reaches_pending_preparation_without_an_upload(photo, acknowledge):
    (photo.signals / "authenticate").touch()
    (photo.signals / "hold-preparation").touch()
    if not acknowledge:
        photo.owner.supervisor._shutdown_timeout = 0.75
    photo.offer()
    await eventually((photo.signals / "preparing").exists)
    (directory,) = (
        path
        for path in photo.owned.glob("puripuly-photo-*")
        if not path.name.startswith("puripuly-photo-child-")
    )
    shutdown = asyncio.create_task(photo.coordinator.shutdown())
    await eventually((photo.signals / "shutdown-1").exists)
    assert directory.is_dir()
    assert photo.renderer.children[0].returncode is None
    assert not shutdown.done()
    assert not photo.provider.sent
    if acknowledge:
        await eventually(lambda: photo.owner.lifecycle.snapshot["photo"] is None)
        (photo.signals / "settle").touch()
    async with asyncio.timeout(3):
        assert (await asyncio.shield(shutdown)).state == "completed"
    assert photo.owner.supervisor.last_shutdown_acknowledged is acknowledge
    assert (photo.renderer.children[0].returncode == 0) is acknowledge
    assert not photo.provider.sent
    assert all(stage != "result" for stage, _ in photo.observations)
    await assert_settled(photo)


@pytest.mark.parametrize("complete_before_freeze", [False, True])
async def test_renderer_loss_recovers_only_before_freeze_while_provider_drains(
    photo, complete_before_freeze
):
    (photo.signals / "authenticate").touch()
    photo.provider.allow_close.clear()
    photo.offer()
    await eventually(lambda: len(photo.provider.sent) == 1)
    await photo.renderer.children[0].terminate()
    await eventually(
        lambda: photo.owner.supervisor.generation == 2
        and any(receipt["generation"] == 2 for receipt in photo.owner.bridge.receipts)
    )
    photo.provider.responses.put_nowait(response_item())
    photo.provider.responses.put_nowait(
        response_delta(
            '{"blocks":[{"bbox":[0,0,100,100],"source":"source","translation":"translated"}]}'
        )
    )
    await eventually(
        lambda: any(
            receipt["generation"] == 2 and receipt["block_count"] == 1
            for receipt in photo.owner.bridge.receipts
        )
    )
    if complete_before_freeze:
        photo.provider.responses.put_nowait({"type": "response.completed"})
        await eventually(photo.provider.closing.is_set)
    photo.allow_drain.clear()
    shutdown = asyncio.create_task(photo.coordinator.shutdown())
    await eventually((photo.signals / "shutdown-2").exists)
    photo.allow_drain.set()
    await eventually(photo.provider.closing.is_set)
    await photo.renderer.children[1].terminate()
    await asyncio.sleep(0.6)
    async with asyncio.timeout(2):
        await asyncio.shield(photo.owner.supervisor._task)
    assert photo.owner.supervisor.generation == 2
    assert len(photo.renderer.children) == 2
    assert len(photo.provider.sent) == 1
    assert not shutdown.done()
    photo.provider.allow_close.set()
    async with asyncio.timeout(3):
        assert (await asyncio.shield(shutdown)).state == "completed"
    assert len(photo.renderer.children) == 2
    assert len(photo.provider.sent) == 1
    assert all(stage != "result" for stage, _ in photo.observations)
    await assert_settled(photo)


@pytest.mark.parametrize("authenticate_before_handoff", [True, False])
async def test_shutdown_owns_an_admitted_child_until_it_exits(photo, authenticate_before_handoff):
    photo.renderer.spawn_release.clear()
    if authenticate_before_handoff:
        (photo.signals / "authenticate").touch()
        (photo.signals / "hold-preparation").touch()
    photo.offer()
    await eventually(lambda: len(photo.renderer.children) == 1)
    if authenticate_before_handoff:
        await eventually((photo.signals / "preparing").exists)
    shutdown = asyncio.create_task(photo.coordinator.shutdown())
    if authenticate_before_handoff:
        await eventually((photo.signals / "shutdown-1").exists)
        assert not photo.provider.closing.is_set()
    else:
        await eventually(photo.provider.closing.is_set)
    assert not shutdown.done()
    assert photo.renderer.children[0].returncode is None
    assert next(photo.owned.glob("puripuly-photo-child-*"), None) is not None
    photo.renderer.spawn_release.set()
    await eventually(lambda: photo.owner.supervisor.pid is not None)
    if not authenticate_before_handoff:
        (photo.signals / "authenticate").touch()
    await eventually((photo.signals / "shutdown-1").exists)
    assert not shutdown.done()
    (photo.signals / "settle").touch()
    async with asyncio.timeout(3):
        assert (await asyncio.shield(shutdown)).state == "completed"
    assert photo.owner.supervisor.last_shutdown_acknowledged
    assert [child.returncode for child in photo.renderer.children] == [0]
    assert not photo.provider.sent
    await assert_settled(photo)


async def test_confirmed_child_exit_drains_pending_preparation_despite_reader_failure(photo):
    for signal in ("authenticate", "hold-preparation", "fail-reader"):
        (photo.signals / signal).touch()
    photo.offer()
    await eventually((photo.signals / "preparing").exists)
    shutdown = asyncio.create_task(photo.coordinator.shutdown())
    await eventually((photo.signals / "shutdown-1").exists)
    (photo.signals / "settle").touch()
    async with asyncio.timeout(3):
        with pytest.raises(LifecycleDiagnosticsUnavailableError, match="PhotoProcessSupervisor"):
            await asyncio.shield(shutdown)
    assert photo.coordinator.snapshot.state == "completed_with_failures"
    assert [child.returncode for child in photo.renderer.children] == [0]
    assert not photo.provider.sent
    await assert_settled(photo)


@pytest.mark.parametrize("photo", [0.75], indirect=True)
async def test_unconfirmed_termination_reports_failure_and_owns_late_exit(photo, monkeypatch):
    for signal in ("authenticate", "hold-preparation"):
        (photo.signals / signal).touch()
    photo.owner.supervisor._shutdown_timeout = 0.05
    terminate = _AsyncioOverlayProcess.terminate
    attempts = []

    async def unconfirmed(child):
        attempts.append(child)
        if len(attempts) == 1:
            raise RuntimeError("overlay child termination was not confirmed")
        await terminate(child)

    monkeypatch.setattr(_AsyncioOverlayProcess, "terminate", unconfirmed)
    photo.offer()
    await eventually((photo.signals / "preparing").exists)
    (directory,) = (
        path
        for path in photo.owned.glob("puripuly-photo-*")
        if not path.name.startswith("puripuly-photo-child-")
    )
    shutdown = asyncio.create_task(photo.coordinator.shutdown())
    await eventually(
        lambda: (
            "process_failed",
            {"generation": 1, "failure_code": "photo_child_termination_unconfirmed"},
        )
        in photo.observations
    )
    assert attempts == photo.renderer.children
    assert attempts[0].returncode is None
    assert directory.is_dir()
    assert photo.provider.closed == 0
    async with asyncio.timeout(2):
        with pytest.raises(TimeoutError, match="coordinator deadline"):
            await asyncio.shield(shutdown)
    assert not photo.coordinator.snapshot.terminal
    assert directory.is_dir()
    assert photo.owner.bridge.connected
    (photo.signals / "settle").touch()
    await eventually(lambda: photo.coordinator.snapshot.terminal)
    assert photo.coordinator.snapshot.state == "completed_with_failures"
    assert [failure.timed_out for failure in photo.coordinator.snapshot.failures] == [True, False]
    assert [child.returncode for child in photo.renderer.children] == [0]
    assert not photo.provider.sent
    await assert_settled(photo)
