from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.app.services.osc.control_router import OscControlRouter
from puripuly_heart.app.services.overlay.overlay_application import OverlayApplicationOwner
from puripuly_heart.core.diagnostics import diagnostic_operation, exception_diagnostics, serialize_diagnostics
from puripuly_heart.core.osc.oscquery import ZeroconfOscQueryService
from puripuly_heart.core.osc.oscquery_contract import OscQueryServiceInfo
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.child_diagnostics import project_child_failure
from puripuly_heart.core.overlay.process import OverlayProcessManager
from puripuly_heart.core.overlay.protocol import OverlayPresentationSnapshot
from puripuly_heart.core.runtime.oscquery import OscQueryRuntime
from puripuly_heart.ui.desktop_overlay import DesktopOverlayRenderer


async def _noop() -> None:
    pass


def _raise_sink(*_args, **_kwargs):
    raise RuntimeError("private sink detail")


def _walk(value):
    yield value
    for cause in value.get("causes", ()):
        yield from _walk(cause)


@pytest.mark.asyncio
async def test_query_failure_is_unavailable_not_absence_and_empty_avatar_recovers(monkeypatch):
    status = 503
    client_type = httpx.AsyncClient

    def respond(request):
        return httpx.Response(status, json={} if status == 200 else {"secret": "private body"})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs)
    )
    service = ZeroconfOscQueryService()
    remote = OscQueryServiceInfo("vrchat", "127.0.0.1", query_port=12345, is_vrchat=True)
    service._services[remote.service_id] = remote
    assert await service.discover_vrchat() is None
    assert service.discovery_observation == "unavailable"
    assert service.discovery_failure.status_code == 503
    assert await service.query_avatar(remote) == {}
    assert service.avatar_observation == "unavailable"
    assert service.avatar_failure.status_code == 503
    retained = serialize_diagnostics(service.avatar_failure)
    assert "private body" not in json.dumps(retained)
    status = 200
    assert await service.discover_vrchat() is not None
    assert service.discovery_observation == "available"
    assert await service.query_avatar(remote) == {}
    assert service.avatar_observation == "available"
    assert service.avatar_failure is None
    service._services.clear()
    assert await service.discover_vrchat() is None
    assert service.discovery_observation == "absent"
    assert service.discovery_failure is None
    assert service.failures


@pytest.mark.asyncio
async def test_unavailable_avatar_does_not_assert_missing_parameters():
    inspected = []
    remote = OscQueryServiceInfo("vrchat", "127.0.0.1")

    class Service:
        avatar_observation = "unavailable"

        async def query_avatar(self, _remote):
            return {}

    runtime = OscQueryRuntime(Service(), _noop, _noop, lambda: 1, avatar_inspector=inspected.append)
    await runtime._apply_service_info(remote, fallback_send_port=9000)
    assert inspected == []
    runtime.service.avatar_observation = "available"
    await runtime._apply_service_info(remote, fallback_send_port=9000)
    assert inspected == [{}]


@pytest.mark.asyncio
async def test_monitor_and_cleanup_retain_safe_failures_without_changing_polling(monkeypatch):
    attempts = 0

    class Service:
        async def discover_vrchat(self):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError(10061, "private endpoint")
            completed.set()
            return None

        async def stop(self):
            raise PermissionError(13, "private path")

        async def unadvertise_receiver(self):
            raise OSError(5, "private advertisement")

    completed = asyncio.Event()
    runtime = OscQueryRuntime(Service(), _noop, _noop, lambda: 1, discovery_poll_interval_seconds=0.1)
    runtime.mode = "automatic"
    runtime._started = True
    runtime._service_started = True
    runtime.advertised_port = 1
    runtime._start_discovery_monitor()
    await asyncio.wait_for(completed.wait(), 2)
    assert attempts >= 2
    await runtime.stop()
    assert not runtime.started
    assert [failure.operation for failure in runtime.failures] == [
        "discovery_monitor", "unadvertise_receiver", "service_stop"
    ]
    assert runtime.failures[-1].fields["errno"] == 13
    assert "private" not in json.dumps([serialize_diagnostics(item) for item in runtime.failures])


@pytest.mark.asyncio
async def test_dispatch_preserves_primary_despite_failed_error_sink():
    class Application:
        async def set_translation(self, _enabled):
            raise PermissionError(13, "private user path")

    router = OscControlRouter(Application(), error_sink=_raise_sink)
    operation_id, request_id = str(uuid4()), str(uuid4())
    with diagnostic_operation("test", operation_id, request_id):
        result = await router.dispatch_packet("/avatar/parameters/PuriPuly_Trans", True)
    assert not result.applied
    assert result.diagnostics.fields["errno"] == 13
    assert result.diagnostics.operation_id == operation_id
    assert result.diagnostics.request_id == request_id
    assert "private" not in json.dumps(serialize_diagnostics(result.diagnostics))
    await router.close()


@pytest.mark.asyncio
async def test_rejected_application_dispatch_preserves_owned_failure_and_identity():
    failure = exception_diagnostics(
        OSError(5, "private"), component="osc_control", operation="control_dispatch", code="osc_dispatch_failed"
    )
    operation_id, request_id = str(uuid4()), str(uuid4())

    class Application:
        async def set_translation(self, _enabled):
            return SimpleNamespace(
                status="failed", failure_diagnostics=failure,
                operation_id=operation_id, request_id=request_id,
            )

    router = OscControlRouter(Application(), error_sink=_raise_sink)
    result = await router.dispatch_packet("/avatar/parameters/PuriPuly_Trans", True)
    assert result.error == "application_rejected"
    assert result.diagnostics is failure
    assert (result.operation_id, result.request_id) == (operation_id, request_id)
    await router.close()


@pytest.mark.asyncio
async def test_bridge_delivery_receipt_retains_original_when_recorder_fails():
    class Connection:
        async def send(self, _message):
            raise OSError(10054, "private child output")

        async def close(self):
            pass

    connection = Connection()
    bridge = OverlayBridge("synthetic", overlay_instance_id=f"overlay-{uuid4()}")
    bridge.diagnostics = SimpleNamespace(record_bridge=_raise_sink)
    bridge._connection_epochs[connection] = 7
    bridge._authenticated_connections.add(connection)
    result = await bridge._write_message(
        connection, 7, "{}", timeout=0.2, scene_revision=19, payload_type="snapshot"
    )
    assert not result
    receipt = bridge.delivery_receipts[-1]
    assert (receipt.stage, receipt.outcome) == ("ambiguous_receipt", "ambiguous")
    assert (receipt.scene_revision, receipt.connection_epoch) == (19, 7)
    assert receipt.diagnostics.fields["errno"] == 10054
    assert receipt.diagnostics.fields["overlay_instance_id"] == bridge.overlay_instance_id
    assert "private" not in json.dumps(serialize_diagnostics(receipt.diagnostics))


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_late_transport_failure_retains_ambiguous_receipt_identity(cancelled):
    gate = asyncio.Event()
    started = asyncio.Event()

    class Connection:
        async def send(self, _message):
            started.set()
            await gate.wait()
            raise OSError(10054, "private late transport")

        async def close(self):
            pass

    connection = Connection()
    bridge = OverlayBridge("synthetic", overlay_instance_id=f"overlay-{uuid4()}")
    bridge._connection_epochs[connection] = 3
    bridge._authenticated_connections.add(connection)
    operation_id, request_id = str(uuid4()), str(uuid4())
    with diagnostic_operation("test", operation_id, request_id):
        bridge._mailbox.admit_scene(OverlayPresentationSnapshot(revision=29), {})
        identity = bridge._mailbox.current_scene.operation_identity
        if cancelled:
            write = asyncio.create_task(bridge._write_message(
                connection, 3, "{}", timeout=1, scene_revision=29, payload_type="snapshot",
                operation_identity=identity,
            ))
            await started.wait()
            write.cancel()
            with pytest.raises(asyncio.CancelledError):
                await write
        else:
            assert not await bridge._write_message(
                connection, 3, "{}", timeout=0.01, scene_revision=29, payload_type="snapshot",
                operation_identity=identity,
            )
    initial = bridge.delivery_receipts[-1]
    assert initial.diagnostics is None if cancelled else initial.diagnostics.code == "destination_write_timeout"
    tasks = tuple(bridge._unresolved_transport_tasks)
    gate.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)
    settled = bridge.delivery_receipts[-1]
    assert settled is not initial
    assert (settled.scene_revision, settled.connection_epoch) == (29, 3)
    assert settled.outcome == "ambiguous"
    assert settled.diagnostics.fields["errno"] == 10054
    if not cancelled:
        assert settled.diagnostics.causes[0].diagnostic_id == initial.diagnostics.diagnostic_id
    assert (settled.diagnostics.operation_id, settled.diagnostics.request_id) == (operation_id, request_id)


def test_native_ingress_preserves_stage_code_cleanup_and_rejects_untrusted_values(tmp_path):
    identity = f"overlay-{uuid4()}"
    operation_id, request_id = str(uuid4()), str(uuid4())
    projected = project_child_failure({
        "type": "startup_error", "overlay_instance_id": identity,
        "failure_reason": "render_failed", "primary_failure_reason": "render_failed",
        "cleanup_failure_reason": "openvr_failed", "runtime_generation": 1,
        "native_failure": {"native_stage": "render_frame", "native_code": -2005270523, "raw": "private body"},
        "cleanup_native_failure": {
            "native_stage": "cleanup_hide", "native_code": 10,
            "secondary": {"native_stage": "cleanup_release", "native_code": 11},
        },
    })
    assert projected["rejected_records"] == 1
    assert project_child_failure(projected) is projected
    manager = OverlayProcessManager(
        overlay_instance_id=identity, diagnostics_dir=tmp_path,
        operation_id=operation_id, request_id=request_id,
    )
    manager._current_phase = "startup"
    manager._record_native_status(projected)
    receipt = manager.shutdown_receipt()
    primary = receipt["failure_diagnostics"]
    cleanup = receipt["cleanup_diagnostics"]
    assert (primary["operation_id"], primary["request_id"]) == (operation_id, request_id)
    assert any(item["fields"].get("native_code") == -2005270523 for item in _walk(primary))
    stages = {item["fields"].get("native_stage") for item in _walk(cleanup)}
    assert {"cleanup_hide", "cleanup_release"} <= stages
    assert all(item["fields"].get("overlay_instance_id", identity) == identity for item in _walk(cleanup))
    assert "private body" not in json.dumps(receipt)
    malformed = project_child_failure({
        "type": "runtime_error", "failure_reason": "openvr_failed",
        "native_failure": {"native_stage": "private path", "native_code": 1 << 40},
    })
    assert malformed["native_failure"] == {}
    assert malformed["rejected_records"] == 2


@pytest.mark.asyncio
async def test_process_termination_cleanup_retains_primary_and_cleanup_cause(tmp_path):
    class Process:
        returncode = None
        pid = 123

        async def terminate(self):
            raise PermissionError(13, "private process path")

    class Manager(OverlayProcessManager):
        async def _request_graceful_shutdown_before_terminate(self, _process):
            return False

    manager = Manager(diagnostics_dir=tmp_path)
    manager._process = Process()
    manager._set_shutdown_failure("render_failed")
    await manager._complete_failure(terminate_process=True, cleanup_manifest=False)
    receipt = manager.shutdown_receipt()
    assert manager._process is not None
    assert receipt["failure_diagnostics"]["code"] == "render_failed"
    cleanup = receipt["failures"][-1]
    assert cleanup["code"] == "termination_unconfirmed"
    assert cleanup["fields"]["errno"] == 13
    assert cleanup["causes"][0]["code"] == "render_failed"


@pytest.mark.asyncio
async def test_desktop_cleanup_retains_multiple_safe_causes_when_delivery_fails():
    class FailedClose:
        async def close(self):
            raise OSError(5, "private resource")

    class FailedSink:
        async def emit(self, _event):
            raise RuntimeError("private sink")

    renderer = DesktopOverlayRenderer(
        SimpleNamespace(overlay_instance_id=f"overlay-{uuid4()}", parent_pid=123),
        window=FailedClose(), parent_monitor=FailedClose(), diagnostic_port=FailedClose(),
        lifecycle_sink=FailedSink(),
    )
    await renderer.shutdown()
    receipt = renderer.diagnostic_delivery_receipt()
    cleanup = receipt["cleanup_fields"]["cleanup_diagnostics"]
    stages = {item["fields"].get("cleanup_stage") for item in _walk(cleanup)}
    assert {"window_close", "diagnostic_close", "parent_monitor_close"} <= stages
    assert not receipt["close_complete"]
    assert receipt["reliable_delivery_failures"] > 0
    assert "private" not in json.dumps(receipt)


def test_overlay_replacement_retains_manager_primary_when_log_sink_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("logging.Logger.log", _raise_sink)
    manager = OverlayProcessManager(diagnostics_dir=tmp_path)
    manager._set_shutdown_failure("render_failed")
    owner = object.__new__(OverlayApplicationOwner)
    owner._runtime = SimpleNamespace(process_manager=manager, overlay_instance_id=manager.overlay_instance_id)
    owner._failure_diagnostics = None
    owner._retain_failure(PermissionError(13, "private replacement"), "replacement_start")
    retained = serialize_diagnostics(owner._failure_diagnostics)
    assert retained["fields"]["errno"] == 13
    assert retained["causes"][0]["code"] == "render_failed"
    assert "private replacement" not in json.dumps(retained)
