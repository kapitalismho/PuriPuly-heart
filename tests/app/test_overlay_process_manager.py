from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.core.overlay import openvr_vendor as openvr_vendor_module
from puripuly_heart.core.overlay import process as process_module
from puripuly_heart.core.overlay.manifest import OVERLAY_CONTRACT_VERSION, OverlayLaunchManifest
from puripuly_heart.core.overlay.openvr_vendor import VendoredOpenVrBundle
from puripuly_heart.core.overlay.process import (
    DefaultOverlayProcessRunner,
    OverlayManagedProcess,
    OverlayPreparationError,
    OverlayProcessManager,
)


@pytest.mark.asyncio
async def test_default_runner_passes_p05_product_default_in_child_env_without_mutating_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    async def fake_spawn(*command: str, **kwargs: object) -> object:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(stdout=None, stderr=None)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.delenv(process_module.QUIET_TAIL_PROFILE_ENV, raising=False)
    runner = DefaultOverlayProcessRunner()
    await runner.spawn(tmp_path / "overlay.exe", tmp_path / "manifest.json")
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env[process_module.QUIET_TAIL_PROFILE_ENV] == "p05"
    assert process_module.QUIET_TAIL_PROFILE_ENV not in os.environ


@pytest.mark.asyncio
async def test_default_runner_passes_explicit_p20_in_child_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    async def fake_spawn(*command: str, **kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(stdout=None, stderr=None)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    runner = DefaultOverlayProcessRunner(quiet_tail_profile="p20")
    await runner.spawn(tmp_path / "overlay.exe", tmp_path / "manifest.json")
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env[process_module.QUIET_TAIL_PROFILE_ENV] == "p20"


def test_new_manifest_serialization_omits_runtime_profile() -> None:
    payload = _overlay_manifest().to_dict()
    assert "quiet_tail_profile" not in payload


@dataclass(slots=True)
class FakeOverlayManagedProcess(OverlayManagedProcess):
    ready_event_delay_ms: int | None = None
    startup_error: str | None = None
    exit_code: int | None = None
    exit_after_ready_code: int | None = None
    runtime_error_after_ready: str | None = None
    terminated: bool = False

    def __post_init__(self) -> None:
        self._events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._exit_future: asyncio.Future[int | None] = asyncio.get_running_loop().create_future()
        self._schedule_transitions()

    async def next_event(self) -> dict[str, object]:
        return await self._events.get()

    async def wait(self) -> int | None:
        return await asyncio.shield(self._exit_future)

    async def terminate(self) -> None:
        self.terminated = True
        if not self._exit_future.done():
            self._exit_future.set_result(0)

    @property
    def returncode(self) -> int | None:
        if not self._exit_future.done() or self._exit_future.cancelled():
            return None
        return self._exit_future.result()

    def _schedule_transitions(self) -> None:
        async def runner() -> None:
            if self.startup_error is not None:
                await self._events.put(
                    {
                        "type": "startup_error",
                        "failure_reason": self.startup_error,
                    }
                )
                if self.exit_code is not None and not self._exit_future.done():
                    await asyncio.sleep(0)
                    self._exit_future.set_result(self.exit_code)
                return

            if self.ready_event_delay_ms is not None:
                await asyncio.sleep(self.ready_event_delay_ms / 1000.0)
                await self._events.put({"type": "overlay_ready"})
                if self.runtime_error_after_ready is not None:
                    await asyncio.sleep(0)
                    await self._events.put(
                        {
                            "type": "runtime_error",
                            "failure_reason": self.runtime_error_after_ready,
                        }
                    )
                    return
                if self.exit_after_ready_code is not None and not self._exit_future.done():
                    await asyncio.sleep(0)
                    self._exit_future.set_result(self.exit_after_ready_code)
                    return

            if self.exit_code is not None and not self._exit_future.done():
                await asyncio.sleep(0)
                self._exit_future.set_result(self.exit_code)

        asyncio.create_task(runner())


@pytest.mark.asyncio
async def test_asyncio_overlay_process_terminate_escalates_to_kill_after_grace() -> None:
    class HangingProcess:
        stdout = None
        stderr = None
        returncode = None
        pid = 4321

        def __init__(self) -> None:
            self.terminate_calls = 0
            self.kill_calls = 0
            self._wait_future: asyncio.Future[int | None] = (
                asyncio.get_running_loop().create_future()
            )

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9
            if not self._wait_future.done():
                self._wait_future.set_result(self.returncode)

        async def wait(self) -> int | None:
            return await self._wait_future

    process = HangingProcess()
    managed = process_module._AsyncioOverlayProcess(
        process=process,
        terminate_grace_s=0.0,
    )

    await managed.terminate()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1


@pytest.mark.asyncio
async def test_overlay_manager_traces_asyncio_process_kill_escalation() -> None:
    class HangingProcess:
        stdout = None
        stderr = None
        returncode = None
        pid = 4321

        def __init__(self) -> None:
            self._wait_future: asyncio.Future[int | None] = (
                asyncio.get_running_loop().create_future()
            )

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            self.returncode = -9
            self._wait_future.set_result(self.returncode)

        async def wait(self) -> int | None:
            return await self._wait_future

    managed = process_module._AsyncioOverlayProcess(
        process=HangingProcess(),
        terminate_grace_s=0.0,
    )
    manager = OverlayProcessManager(selected_target="desktop", geometry_authority="flet")
    manager.state = "connected"
    manager._process = managed
    manager._attach_process_diagnostics(managed)

    await manager.stop()

    assert manager.diagnostics is not None
    events = [event["event"] for event in manager.diagnostics.process_events]
    assert events.index("terminate_requested") < events.index("kill_requested")
    assert events.index("kill_requested") < events.index("process_exited")


@dataclass(slots=True)
class FakeProcessRunner:
    ready_event_delay_ms: int | None = None
    startup_error: str | None = None
    exit_code: int | None = None
    exit_after_ready_code: int | None = None
    runtime_error_after_ready: str | None = None
    spawn_error: Exception | None = None
    manifest_error: Exception | None = None
    last_process: FakeOverlayManagedProcess | None = None

    def prepare(self, manifest: OverlayLaunchManifest) -> Path:
        _ = manifest
        if self.manifest_error is not None:
            raise self.manifest_error
        return Path("C:/fake/PuriPulyHeartOverlay.exe")

    async def spawn(
        self,
        executable_path: Path,
        manifest_path: Path,
    ) -> OverlayManagedProcess:
        _ = (executable_path, manifest_path)
        if self.spawn_error is not None:
            raise self.spawn_error
        self.last_process = FakeOverlayManagedProcess(
            ready_event_delay_ms=self.ready_event_delay_ms,
            startup_error=self.startup_error,
            exit_code=self.exit_code,
            exit_after_ready_code=self.exit_after_ready_code,
            runtime_error_after_ready=self.runtime_error_after_ready,
        )
        return self.last_process


@dataclass(slots=True)
class MissingExecutableRunner:
    def prepare(self, manifest: OverlayLaunchManifest) -> Path:
        _ = manifest
        raise FileNotFoundError("missing")

    async def spawn(
        self,
        executable_path: Path,
        manifest_path: Path,
    ) -> OverlayManagedProcess:
        _ = (executable_path, manifest_path)
        raise AssertionError("spawn should not be called")


def _overlay_manifest() -> OverlayLaunchManifest:
    return OverlayLaunchManifest(
        contract_version=OVERLAY_CONTRACT_VERSION,
        app_version="test",
        overlay_instance_id="overlay-test",
        bridge_url="ws://127.0.0.1:8765",
        session_token="session-token",
        parent_pid=1234,
        startup_deadline_ms=3000,
        log_dir="logs",
        log_level="INFO",
        locale="en",
    )


def _write_vendored_openvr_bundle(
    repo_root: Path,
    *,
    dll_bytes: bytes = b"vendored-openvr-runtime",
) -> VendoredOpenVrBundle:
    bundle_dir = repo_root / "third_party" / "openvr"
    dll_path = bundle_dir / "win64" / "openvr_api.dll"
    sha256_path = bundle_dir / "win64" / "openvr_api.dll.sha256"
    license_path = bundle_dir / "LICENSE"
    readme_path = bundle_dir / "README.md"

    dll_path.parent.mkdir(parents=True, exist_ok=True)
    dll_path.write_bytes(dll_bytes)
    sha256_path.write_text(
        f"{hashlib.sha256(dll_bytes).hexdigest()} *openvr_api.dll\n",
        encoding="utf-8",
    )
    license_path.write_text("OpenVR license", encoding="utf-8")
    readme_path.write_text("Vendored OpenVR runtime", encoding="utf-8")

    return VendoredOpenVrBundle(
        bundle_dir=bundle_dir,
        dll_path=dll_path,
        sha256_path=sha256_path,
        license_path=license_path,
        readme_path=readme_path,
        dll_sha256=hashlib.sha256(dll_bytes).hexdigest(),
    )


def _make_local_dev_overlay_fixture(
    tmp_path: Path,
    *,
    staged_dll_bytes: bytes | None,
    vendored_dll_bytes: bytes = b"vendored-openvr-runtime",
) -> tuple[Path, VendoredOpenVrBundle]:
    repo_root = tmp_path / "repo"
    staged = repo_root / "build" / "overlay" / "PuriPulyHeartOverlay.exe"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("staged-overlay", encoding="utf-8")

    overlay_source = repo_root / "native" / "overlay" / "src" / "state.rs"
    overlay_source.parent.mkdir(parents=True, exist_ok=True)
    overlay_source.write_text("// overlay source", encoding="utf-8")

    if staged_dll_bytes is not None:
        staged.with_name("openvr_api.dll").write_bytes(staged_dll_bytes)

    base_mtime = 1_700_000_000
    os.utime(overlay_source, (base_mtime, base_mtime))
    os.utime(staged, (base_mtime + 60, base_mtime + 60))

    return staged, _write_vendored_openvr_bundle(repo_root, dll_bytes=vendored_dll_bytes)


def _make_packaged_overlay_fixture(
    tmp_path: Path,
    *,
    packaged_dll_bytes: bytes | None,
    vendored_dll_bytes: bytes = b"vendored-openvr-runtime",
) -> tuple[Path, VendoredOpenVrBundle]:
    overlay_executable = tmp_path / "installed" / "PuriPulyHeartOverlay.exe"
    overlay_executable.parent.mkdir(parents=True, exist_ok=True)
    overlay_executable.write_text("packaged-overlay", encoding="utf-8")

    if packaged_dll_bytes is not None:
        overlay_executable.with_name("openvr_api.dll").write_bytes(packaged_dll_bytes)

    return overlay_executable, _write_vendored_openvr_bundle(
        tmp_path / "repo",
        dll_bytes=vendored_dll_bytes,
    )


def _repo_vendored_openvr_dll_path() -> Path:
    return (
        Path(__file__).resolve().parents[2] / "third_party" / "openvr" / "win64" / "openvr_api.dll"
    )


def _patch_vendored_openvr_bundle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundle: VendoredOpenVrBundle | None = None,
    error: Exception | None = None,
) -> None:
    def validate_vendored_openvr_bundle() -> VendoredOpenVrBundle:
        if error is not None:
            raise error
        assert bundle is not None
        return bundle

    monkeypatch.setattr(
        process_module,
        "openvr_vendor",
        SimpleNamespace(
            validate_openvr_runtime_dll=openvr_vendor_module.validate_openvr_runtime_dll,
            validate_vendored_openvr_bundle=validate_vendored_openvr_bundle,
        ),
        raising=False,
    )


@pytest.mark.asyncio
async def test_overlay_process_manager_stop_preserves_process_when_terminate_fails() -> None:
    class FailingOnceTerminateProcess:
        def __init__(self) -> None:
            self.terminate_calls = 0

        async def next_event(self) -> dict[str, object]:
            raise AssertionError("next_event should not be called")

        async def wait(self) -> int | None:
            return None

        async def terminate(self) -> None:
            self.terminate_calls += 1
            if self.terminate_calls == 1:
                raise RuntimeError("terminate failed")

        def set_logging_mode(self, mode: str) -> None:
            _ = mode

    manager = OverlayProcessManager()
    process = FailingOnceTerminateProcess()
    manager._process = process

    with pytest.raises(RuntimeError, match="terminate failed"):
        await manager.stop()

    assert manager._process is process

    await manager.stop()

    assert process.terminate_calls == 2
    assert manager._process is None


@pytest.mark.asyncio
async def test_overlay_process_manager_stop_awaits_pre_requested_graceful_cleanup() -> None:
    process = FakeOverlayManagedProcess()
    shutdown_request_calls = 0

    def reject_new_runtime_task(
        coroutine: object,
        *,
        task_name: str,
    ) -> asyncio.Task[object]:
        _ = task_name
        close = getattr(coroutine, "close", None)
        if callable(close):
            close()
        raise RuntimeError("runtime is closing to new tasks")

    async def request_shutdown() -> None:
        nonlocal shutdown_request_calls
        shutdown_request_calls += 1

    manager = OverlayProcessManager(
        graceful_shutdown_request=request_shutdown,
        graceful_shutdown_timeout_s=0.2,
        selected_target="desktop",
        geometry_authority="flet",
        task_factory=reject_new_runtime_task,
    )
    manager.state = "connected"
    manager._process = process
    manager._attach_process_diagnostics(process)
    manager.mark_shutdown_requested()
    for event in (
        "stop_requested",
        "graceful_close_requested",
        "process_exited",
        "pid_file_removed",
    ):
        await process._events.put(
            {
                "type": "overlay_trace",
                "component": "flet_desktop_view_process",
                "event": event,
                "generation": 1,
                "monotonic_ms": 1,
            }
        )
    await process._events.put(
        {
            "type": "shutdown_complete",
            "overlay_instance_id": manager.overlay_instance_id,
        }
    )
    process._exit_future.set_result(0)

    await manager.stop()

    assert shutdown_request_calls == 0
    assert process.terminated is False
    assert manager.state == "off"
    assert manager._process is None
    assert manager.diagnostics is not None
    process_events = manager.diagnostics.process_events
    child_events = [
        event["trace_event"]
        for event in process_events
        if event.get("trace_component") == "flet_desktop_view_process"
    ]
    assert child_events == [
        "stop_requested",
        "graceful_close_requested",
        "process_exited",
        "pid_file_removed",
    ]
    manager_events = [event["event"] for event in process_events]
    assert "graceful_shutdown_acknowledged" in manager_events
    assert "terminate_requested" not in manager_events


@pytest.mark.asyncio
async def test_overlay_process_manager_latches_ack_consumed_by_connected_monitor() -> None:
    process = FakeOverlayManagedProcess()
    shutdown_request_calls = 0

    async def request_shutdown() -> None:
        nonlocal shutdown_request_calls
        shutdown_request_calls += 1

    manager = OverlayProcessManager(
        graceful_shutdown_request=request_shutdown,
        graceful_shutdown_timeout_s=0.2,
        selected_target="desktop",
        geometry_authority="flet",
    )
    manager.state = "connected"
    manager._process = process
    manager._attach_process_diagnostics(process)
    manager._monitor_task = manager._create_task(
        manager._monitor_connected_process(),
        task_name="connected-process-monitor",
    )
    await asyncio.sleep(0)
    manager.mark_shutdown_requested()
    await process._events.put(
        {
            "type": "shutdown_complete",
            "overlay_instance_id": manager.overlay_instance_id,
        }
    )
    for _ in range(10):
        if manager._shutdown_acknowledged:
            break
        await asyncio.sleep(0)
    assert manager._shutdown_acknowledged is True

    async def exit_after_stop_begins() -> None:
        await asyncio.sleep(0.01)
        process._exit_future.set_result(0)

    exit_task = asyncio.create_task(exit_after_stop_begins())
    await manager.stop()
    await exit_task

    assert shutdown_request_calls == 0
    assert process.terminated is False
    assert manager.state == "off"
    assert manager.diagnostics is not None
    manager_events = manager.diagnostics.process_events
    assert sum(event["event"] == "graceful_shutdown_acknowledged" for event in manager_events) == 1
    assert not any(event["event"] == "graceful_shutdown_timeout" for event in manager_events)
    assert not any(event["event"] == "terminate_requested" for event in manager_events)


@pytest.mark.asyncio
async def test_overlay_process_manager_reconciles_ack_read_during_monitor_cancellation() -> None:
    class AckHandoffManager(OverlayProcessManager):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.ack_handoff_reached = asyncio.Event()

        async def _handle_lifecycle_event(
            self,
            event: object,
            *,
            allow_ready: bool,
            trusted_process_event: bool = True,
        ) -> str:
            if (
                isinstance(event, dict)
                and event.get("type") == "shutdown_complete"
                and not self.ack_handoff_reached.is_set()
            ):
                self.ack_handoff_reached.set()
                await asyncio.Event().wait()
            return await super()._handle_lifecycle_event(
                event,
                allow_ready=allow_ready,
                trusted_process_event=trusted_process_event,
            )

    process = FakeOverlayManagedProcess()
    manager = AckHandoffManager(
        graceful_shutdown_request=lambda: asyncio.sleep(0),
        graceful_shutdown_timeout_s=0.2,
        selected_target="desktop",
        geometry_authority="flet",
    )
    manager.state = "connected"
    manager._process = process
    manager._attach_process_diagnostics(process)
    manager._monitor_task = manager._create_task(
        manager._monitor_connected_process(),
        task_name="connected-process-monitor",
    )
    await asyncio.sleep(0)
    manager.mark_shutdown_requested()
    await process._events.put(
        {
            "type": "shutdown_complete",
            "overlay_instance_id": manager.overlay_instance_id,
        }
    )
    await asyncio.wait_for(manager.ack_handoff_reached.wait(), timeout=0.2)
    assert manager._shutdown_acknowledged is False

    async def exit_after_stop_begins() -> None:
        await asyncio.sleep(0.01)
        process._exit_future.set_result(0)

    exit_task = asyncio.create_task(exit_after_stop_begins())
    await manager.stop()
    await exit_task

    assert process.terminated is False
    assert manager.diagnostics is not None
    manager_events = manager.diagnostics.process_events
    assert sum(event["event"] == "graceful_shutdown_acknowledged" for event in manager_events) == 1
    assert not any(event["event"] == "graceful_shutdown_timeout" for event in manager_events)
    assert not any(event["event"] == "terminate_requested" for event in manager_events)


@pytest.mark.asyncio
async def test_overlay_process_manager_timeout_does_not_claim_cancelled_waiter_exited() -> None:
    process = FakeOverlayManagedProcess()

    async def request_shutdown() -> None:
        return None

    manager = OverlayProcessManager(
        graceful_shutdown_request=request_shutdown,
        graceful_shutdown_timeout_s=0.01,
        selected_target="desktop",
        geometry_authority="flet",
    )
    manager.state = "connected"
    manager._process = process
    manager._attach_process_diagnostics(process)

    await manager.stop()

    assert process.terminated is True
    assert manager.diagnostics is not None
    timeout_event = next(
        event
        for event in manager.diagnostics.process_events
        if event["event"] == "graceful_shutdown_timeout"
    )
    assert timeout_event["acknowledged"] is False
    assert timeout_event["process_exited"] is False


def test_overlay_process_manager_cleanup_manifest_preserves_path_when_unlink_fails() -> None:
    class FailingOnceManifestPath:
        def __init__(self) -> None:
            self.unlink_calls = 0

        def unlink(self) -> None:
            self.unlink_calls += 1
            if self.unlink_calls == 1:
                raise PermissionError("manifest locked")

    manager = OverlayProcessManager()
    manifest_path = FailingOnceManifestPath()
    manager._manifest_path = manifest_path

    with pytest.raises(PermissionError, match="manifest locked"):
        manager._cleanup_manifest()

    assert manager._manifest_path is manifest_path

    manager._cleanup_manifest()

    assert manifest_path.unlink_calls == 2
    assert manager._manifest_path is None


@pytest.mark.asyncio
async def test_overlay_process_manager_waits_for_overlay_ready_before_connected() -> None:
    manager = OverlayProcessManager(process_runner=FakeProcessRunner(ready_event_delay_ms=50))

    try:
        await manager.start()

        assert manager.state == "connected"
        assert manager.failure_reason is None
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_maps_missing_executable_to_failure_reason() -> None:
    manager = OverlayProcessManager(process_runner=MissingExecutableRunner())

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == "missing_executable"


@pytest.mark.asyncio
async def test_overlay_process_manager_maps_spawn_failure_to_failure_reason() -> None:
    manager = OverlayProcessManager(process_runner=FakeProcessRunner(spawn_error=OSError("boom")))

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == "spawn_failed"


@pytest.mark.asyncio
async def test_overlay_process_manager_maps_invalid_manifest_to_failure_reason() -> None:
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(manifest_error=ValueError("bad manifest"))
    )

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == "manifest_invalid"


@pytest.mark.asyncio
async def test_overlay_process_manager_prefers_explicit_startup_error_event_over_exit_code() -> (
    None
):
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(startup_error="bridge_auth_failed", exit_code=21)
    )

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == "bridge_auth_failed"


@pytest.mark.asyncio
async def test_overlay_process_manager_maps_profile_startup_event_to_manifest_invalid() -> None:
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(startup_error="manifest_invalid", exit_code=1)
    )

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == "manifest_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_reason",
    [
        "steamvr_not_installed",
        "steamvr_not_running",
        "hmd_not_found",
    ],
)
async def test_overlay_process_manager_preserves_specific_preflight_failure_reasons(
    failure_reason: str,
) -> None:
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(startup_error=failure_reason, exit_code=20)
    )

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == failure_reason


@pytest.mark.asyncio
async def test_overlay_process_manager_maps_pre_ready_exit_code_to_standard_failure_reason() -> (
    None
):
    manager = OverlayProcessManager(process_runner=FakeProcessRunner(exit_code=21))

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == "renderer_init_failed"


@pytest.mark.asyncio
async def test_overlay_process_manager_schedules_restart_after_connected_crash() -> None:
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(ready_event_delay_ms=0, exit_after_ready_code=1)
    )

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == "runtime_crashed"
    assert manager.restart_scheduled is True
    assert manager._manifest_path is None


@pytest.mark.asyncio
async def test_overlay_process_manager_treats_expected_connected_zero_exit_as_graceful(
    tmp_path: Path,
) -> None:
    runner = FakeProcessRunner(ready_event_delay_ms=0)
    manager = OverlayProcessManager(
        process_runner=runner,
        diagnostics_dir=tmp_path,
    )

    await manager.start()
    assert manager.state == "connected"
    manager.mark_shutdown_requested()
    assert runner.last_process is not None
    runner.last_process._exit_future.set_result(0)
    assert manager._monitor_task is not None
    await manager._monitor_task

    assert manager.state == "stopping"
    assert manager.failure_reason is None
    assert manager._failure_dumped is False
    assert manager._process is None
    assert manager._manifest_path is None

    await manager.stop()

    assert manager.state == "off"


@pytest.mark.asyncio
async def test_overlay_process_manager_does_not_restart_when_shutdown_requested() -> None:
    runner = FakeProcessRunner(ready_event_delay_ms=0)
    manager = OverlayProcessManager(process_runner=runner)

    await manager.start()
    assert manager.state == "connected"
    manager.mark_shutdown_requested()
    assert runner.last_process is not None
    runner.last_process._exit_future.set_result(1)
    assert manager._monitor_task is not None
    await manager._monitor_task

    assert manager.state == "failed"
    assert manager.failure_reason == "runtime_crashed"
    assert manager.restart_scheduled is False


@pytest.mark.asyncio
async def test_overlay_process_manager_treats_unexpected_connected_zero_exit_as_crash(
    tmp_path: Path,
) -> None:
    runner = FakeProcessRunner(ready_event_delay_ms=0)
    manager = OverlayProcessManager(
        process_runner=runner,
        diagnostics_dir=tmp_path,
    )

    await manager.start()
    assert manager.state == "connected"
    assert runner.last_process is not None
    runner.last_process._exit_future.set_result(0)
    assert manager._monitor_task is not None
    await manager._monitor_task

    assert manager.state == "failed"
    assert manager.failure_reason == "runtime_crashed"
    assert manager._failure_dumped is True
    assert manager._process is None
    assert manager._manifest_path is None


@pytest.mark.asyncio
async def test_overlay_process_manager_terminates_child_on_startup_timeout() -> None:
    runner = FakeProcessRunner()
    manager = OverlayProcessManager(process_runner=runner, startup_timeout_ms=10)

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == "startup_timeout"
    assert runner.last_process is not None
    assert runner.last_process.terminated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("startup_error", "expected_failure"),
    [(None, "startup_timeout"), ("renderer_init_failed", "renderer_init_failed")],
)
async def test_overlay_process_manager_waits_for_renderer_cleanup_ack_before_escalation(
    startup_error: str | None,
    expected_failure: str,
) -> None:
    runner = FakeProcessRunner(startup_error=startup_error)
    shutdown_requests: list[str] = []
    manager: OverlayProcessManager

    async def request_shutdown() -> None:
        shutdown_requests.append("requested")
        process = runner.last_process
        assert process is not None
        await process._events.put(
            {
                "type": "shutdown_complete",
                "overlay_instance_id": manager.overlay_instance_id,
            }
        )
        if not process._exit_future.done():
            process._exit_future.set_result(0)

    manager = OverlayProcessManager(
        process_runner=runner,
        startup_timeout_ms=10,
        graceful_shutdown_request=request_shutdown,
        graceful_shutdown_timeout_s=0.2,
    )

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == expected_failure
    assert shutdown_requests == ["requested"]
    assert runner.last_process is not None
    assert runner.last_process.terminated is False


@pytest.mark.asyncio
async def test_overlay_process_manager_native_failure_has_no_desktop_ack_delay() -> None:
    runner = FakeProcessRunner(startup_error="steamvr_not_running")
    manager = OverlayProcessManager(
        process_runner=runner,
        graceful_shutdown_request=None,
        graceful_shutdown_timeout_s=60.0,
    )

    await asyncio.wait_for(manager.start(), timeout=0.2)

    assert manager.state == "failed"
    assert manager.failure_reason == "steamvr_not_running"
    assert runner.last_process is not None
    assert runner.last_process.terminated is True


@pytest.mark.asyncio
async def test_overlay_process_manager_consumes_structured_stdout_events_from_default_runner(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "overlay_stub.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import time",
                "assert sys.argv[1] == '--config'",
                'print(\'{"type": "overlay_ready"}\', flush=True)',
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    manager = OverlayProcessManager(
        process_runner=DefaultOverlayProcessRunner(executable_path=script_path),
        startup_timeout_ms=500,
    )

    try:
        await manager.start()

        assert manager.state == "connected"
        assert manager.failure_reason is None
    finally:
        await manager.stop()


def test_default_overlay_process_runner_prefers_newer_packaged_sibling_over_staged_overlay(
    tmp_path: Path,
) -> None:
    installed_dir = tmp_path / "installed"
    installed_dir.mkdir()
    app_executable = installed_dir / "PuriPulyHeart.exe"
    app_executable.write_text("", encoding="utf-8")

    packaged_sibling = installed_dir / "PuriPulyHeartOverlay.exe"
    packaged_sibling.write_text("packaged", encoding="utf-8")

    repo_root = tmp_path / "repo"
    staged = repo_root / "build" / "overlay" / "PuriPulyHeartOverlay.exe"
    staged.parent.mkdir(parents=True)
    staged.write_text("staged", encoding="utf-8")

    base_mtime = 1_700_000_000
    os.utime(staged, (base_mtime, base_mtime))
    os.utime(packaged_sibling, (base_mtime + 60, base_mtime + 60))

    resolved = DefaultOverlayProcessRunner.resolve_default_executable(
        sys_executable=app_executable,
        repo_root=repo_root,
    )

    assert resolved == packaged_sibling


def test_default_overlay_process_runner_prefers_newer_staged_overlay_over_packaged_sibling(
    tmp_path: Path,
) -> None:
    installed_dir = tmp_path / "installed"
    installed_dir.mkdir()
    app_executable = installed_dir / "PuriPulyHeart.exe"
    app_executable.write_text("", encoding="utf-8")

    packaged_sibling = installed_dir / "PuriPulyHeartOverlay.exe"
    packaged_sibling.write_text("packaged", encoding="utf-8")

    repo_root = tmp_path / "repo"
    staged = repo_root / "build" / "overlay" / "PuriPulyHeartOverlay.exe"
    staged.parent.mkdir(parents=True)
    staged.write_text("staged", encoding="utf-8")

    base_mtime = 1_700_000_000
    os.utime(packaged_sibling, (base_mtime, base_mtime))
    os.utime(staged, (base_mtime + 60, base_mtime + 60))

    resolved = DefaultOverlayProcessRunner.resolve_default_executable(
        sys_executable=app_executable,
        repo_root=repo_root,
    )

    assert resolved == staged


def test_default_overlay_process_runner_uses_staged_overlay_for_local_dev_when_sibling_missing(
    tmp_path: Path,
) -> None:
    installed_dir = tmp_path / "installed"
    installed_dir.mkdir()
    app_executable = installed_dir / "PuriPulyHeart.exe"
    app_executable.write_text("", encoding="utf-8")

    repo_root = tmp_path / "repo"
    staged = repo_root / "build" / "overlay" / "PuriPulyHeartOverlay.exe"
    staged.parent.mkdir(parents=True)
    staged.write_text("staged", encoding="utf-8")

    resolved = DefaultOverlayProcessRunner.resolve_default_executable(
        sys_executable=app_executable,
        repo_root=repo_root,
    )

    assert resolved == staged


@pytest.mark.asyncio
async def test_overlay_process_manager_rejects_stale_staged_overlay_build(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    staged = repo_root / "build" / "overlay" / "PuriPulyHeartOverlay.exe"
    staged.parent.mkdir(parents=True)
    staged.write_text("staged", encoding="utf-8")

    overlay_source = repo_root / "native" / "overlay" / "src" / "state.rs"
    overlay_source.parent.mkdir(parents=True)
    overlay_source.write_text("// changed overlay source", encoding="utf-8")

    staged_mtime = 1_700_000_000
    source_mtime = staged_mtime + 60
    os.utime(staged, (staged_mtime, staged_mtime))
    os.utime(overlay_source, (source_mtime, source_mtime))

    manager = OverlayProcessManager(
        process_runner=DefaultOverlayProcessRunner(executable_path=staged),
    )

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == "stale_overlay_build"


@pytest.mark.parametrize("staged_dll_bytes", [None, b"stale-openvr-runtime"])
def test_default_overlay_process_runner_refreshes_staged_openvr_runtime_dll_from_vendored_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    staged_dll_bytes: bytes | None,
) -> None:
    overlay_executable, bundle = _make_local_dev_overlay_fixture(
        tmp_path,
        staged_dll_bytes=staged_dll_bytes,
    )
    _patch_vendored_openvr_bundle(monkeypatch, bundle=bundle)

    resolved = DefaultOverlayProcessRunner(executable_path=overlay_executable).prepare(
        _overlay_manifest()
    )
    bundled_path = overlay_executable.with_name("openvr_api.dll")

    assert resolved == overlay_executable
    assert bundled_path.read_bytes() == bundle.dll_path.read_bytes()


def test_default_overlay_process_runner_rejects_missing_vendored_openvr_bundle_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay_executable, bundle = _make_local_dev_overlay_fixture(
        tmp_path,
        staged_dll_bytes=b"stale-openvr-runtime",
    )
    _ = bundle
    _patch_vendored_openvr_bundle(
        monkeypatch,
        error=FileNotFoundError("vendored bundle contract missing"),
    )

    with pytest.raises(OverlayPreparationError) as exc_info:
        DefaultOverlayProcessRunner(executable_path=overlay_executable).prepare(_overlay_manifest())

    assert exc_info.value.failure_reason == "vendored_openvr_dll_missing"


def test_default_overlay_process_runner_rejects_missing_packaged_openvr_runtime_dll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay_executable, bundle = _make_packaged_overlay_fixture(tmp_path, packaged_dll_bytes=None)
    _patch_vendored_openvr_bundle(monkeypatch, bundle=bundle)

    with pytest.raises(OverlayPreparationError) as exc_info:
        DefaultOverlayProcessRunner(executable_path=overlay_executable).prepare(_overlay_manifest())

    assert exc_info.value.failure_reason == "packaged_openvr_dll_missing"


def test_default_overlay_process_runner_rejects_packaged_openvr_runtime_dll_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay_executable, bundle = _make_packaged_overlay_fixture(
        tmp_path,
        packaged_dll_bytes=b"mismatched-openvr-runtime",
    )
    _patch_vendored_openvr_bundle(monkeypatch, bundle=bundle)

    with pytest.raises(OverlayPreparationError) as exc_info:
        DefaultOverlayProcessRunner(executable_path=overlay_executable).prepare(_overlay_manifest())

    assert exc_info.value.failure_reason == "openvr_dll_hash_mismatch"


def test_default_overlay_process_runner_accepts_packaged_sibling_openvr_runtime_without_repo_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay_executable = tmp_path / "installed" / "PuriPulyHeartOverlay.exe"
    overlay_executable.parent.mkdir(parents=True, exist_ok=True)
    overlay_executable.write_text("packaged-overlay", encoding="utf-8")

    packaged_dll = overlay_executable.with_name("openvr_api.dll")
    shutil.copy2(_repo_vendored_openvr_dll_path(), packaged_dll)

    fake_repo_module = (
        tmp_path / "fake-repo" / "src" / "puripuly_heart" / "core" / "overlay" / "openvr_vendor.py"
    )
    fake_repo_module.parent.mkdir(parents=True, exist_ok=True)
    fake_repo_module.write_text("# fake packaged layout module path\n", encoding="utf-8")
    monkeypatch.setattr(openvr_vendor_module, "__file__", str(fake_repo_module))

    resolved = DefaultOverlayProcessRunner(executable_path=overlay_executable).prepare(
        _overlay_manifest()
    )

    assert resolved == overlay_executable
    assert packaged_dll.read_bytes() == _repo_vendored_openvr_dll_path().read_bytes()


@pytest.mark.asyncio
async def test_overlay_process_manager_logs_tagged_overlay_child_lines_in_detailed_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_path = tmp_path / "overlay_stub_logs.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import time",
                "assert sys.argv[1] == '--config'",
                'print("[overlay][INFO] child line", flush=True)',
                'print(\'{"type": "overlay_ready"}\', flush=True)',
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    manager = OverlayProcessManager(
        process_runner=DefaultOverlayProcessRunner(executable_path=script_path),
        startup_timeout_ms=500,
        logging_mode="detailed",
    )

    try:
        with caplog.at_level("INFO", logger="puripuly_heart.core.overlay.process"):
            await manager.start()

        assert manager.state == "connected"
        assert any("[overlay][INFO] child line" in message for message in caplog.messages)
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_peer_first_render_trace_passthrough_is_visible_in_detailed_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_path = tmp_path / "overlay_stub_peer_first_render_trace_detailed.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import time",
                "assert sys.argv[1] == '--config'",
                'print("[overlay][INFO] latency_trace stage=peer_overlay_first_render utterance_id=utterance-1 block_id=peer:utterance-1", flush=True)',
                'print(\'{"type": "overlay_ready"}\', flush=True)',
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    manager = OverlayProcessManager(
        process_runner=DefaultOverlayProcessRunner(executable_path=script_path),
        startup_timeout_ms=500,
        logging_mode="detailed",
    )

    try:
        with caplog.at_level("INFO", logger="puripuly_heart.core.overlay.process"):
            await manager.start()

        assert manager.state == "connected"
        assert any(
            "stage=peer_overlay_first_render" in message and "block_id=peer:utterance-1" in message
            for message in caplog.messages
        )
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_basic_mode_hides_info_passthrough_but_keeps_warning_and_stderr(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_path = tmp_path / "overlay_stub_basic_logs.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import time",
                "assert sys.argv[1] == '--config'",
                'print("[overlay][INFO] hidden info", flush=True)',
                'print("[overlay][WARN] visible warning", flush=True)',
                'print("stderr-visible", file=sys.stderr, flush=True)',
                'print(\'{"type": "overlay_ready"}\', flush=True)',
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    manager = OverlayProcessManager(
        process_runner=DefaultOverlayProcessRunner(executable_path=script_path),
        startup_timeout_ms=500,
        logging_mode="basic",
    )

    try:
        with caplog.at_level("INFO", logger="puripuly_heart.core.overlay.process"):
            await manager.start()

        assert manager.state == "connected"
        assert not any("hidden info" in message for message in caplog.messages)
        assert any("visible warning" in message for message in caplog.messages)
        assert any("stderr-visible" in message for message in caplog.messages)
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_peer_first_render_trace_passthrough_is_hidden_in_basic_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_path = tmp_path / "overlay_stub_peer_first_render_trace_basic.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import time",
                "assert sys.argv[1] == '--config'",
                'print("[overlay][INFO] latency_trace stage=peer_overlay_first_render utterance_id=utterance-2 block_id=peer:utterance-2", flush=True)',
                'print(\'{"type": "overlay_ready"}\', flush=True)',
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    manager = OverlayProcessManager(
        process_runner=DefaultOverlayProcessRunner(executable_path=script_path),
        startup_timeout_ms=500,
        logging_mode="basic",
    )

    try:
        with caplog.at_level("INFO", logger="puripuly_heart.core.overlay.process"):
            await manager.start()

        assert manager.state == "connected"
        assert not any("stage=peer_overlay_first_render" in message for message in caplog.messages)
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_overlay_visible_update_rendered_passthrough_is_visible_in_detailed_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_path = tmp_path / "overlay_stub_visible_update_rendered_detailed.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import time",
                "assert sys.argv[1] == '--config'",
                'print("[overlay][INFO] overlay_visible_update_rendered revision=7 block_id=self:1 update_id=upd-self-2 slot_index=0", flush=True)',
                'print(\'{"type": "overlay_ready"}\', flush=True)',
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    manager = OverlayProcessManager(
        process_runner=DefaultOverlayProcessRunner(executable_path=script_path),
        startup_timeout_ms=500,
        logging_mode="detailed",
    )

    try:
        with caplog.at_level("INFO", logger="puripuly_heart.core.overlay.process"):
            await manager.start()

        assert manager.state == "connected"
        assert any(
            "overlay_visible_update_rendered" in message and "update_id=upd-self-2" in message
            for message in caplog.messages
        )
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_overlay_visible_update_rendered_passthrough_is_hidden_in_basic_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_path = tmp_path / "overlay_stub_visible_update_rendered_basic.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import time",
                "assert sys.argv[1] == '--config'",
                'print("[overlay][INFO] overlay_visible_update_rendered revision=8 block_id=self:1 update_id=upd-self-3 slot_index=0", flush=True)',
                'print(\'{"type": "overlay_ready"}\', flush=True)',
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    manager = OverlayProcessManager(
        process_runner=DefaultOverlayProcessRunner(executable_path=script_path),
        startup_timeout_ms=500,
        logging_mode="basic",
    )

    try:
        with caplog.at_level("INFO", logger="puripuly_heart.core.overlay.process"):
            await manager.start()

        assert manager.state == "connected"
        assert not any("overlay_visible_update_rendered" in message for message in caplog.messages)
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_peer_first_render_visibility_checkpoint_passthrough_is_visible_in_detailed_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_path = tmp_path / "overlay_stub_peer_visibility_checkpoint_detailed.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import time",
                "assert sys.argv[1] == '--config'",
                'print("[overlay][INFO] peer_first_render_visibility_checkpoint revision=11 peer_ids=[peer:utterance-3] has_drawable_text=true overlay_visible_before=true should_show_after_submit=false hide_deadline_active=false first_texture_submitted=true redraw_requested=true visible_block_count=1 self_block_count=0 fully_transparent=false", flush=True)',
                'print(\'{"type": "overlay_ready"}\', flush=True)',
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    manager = OverlayProcessManager(
        process_runner=DefaultOverlayProcessRunner(executable_path=script_path),
        startup_timeout_ms=500,
        logging_mode="detailed",
    )

    try:
        with caplog.at_level("INFO", logger="puripuly_heart.core.overlay.process"):
            await manager.start()

        assert manager.state == "connected"
        assert any(
            "peer_first_render_visibility_checkpoint" in message
            and "peer_ids=[peer:utterance-3]" in message
            for message in caplog.messages
        )
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_maps_post_ready_runtime_error_to_failure_reason() -> None:
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(
            ready_event_delay_ms=0,
            runtime_error_after_ready="runtime_disconnected",
        )
    )

    await manager.start()
    await asyncio.sleep(0)

    assert manager.state == "failed"
    assert manager.failure_reason == "runtime_disconnected"
    assert manager._manifest_path is None


@pytest.mark.asyncio
async def test_overlay_process_manager_does_not_accept_overlay_ready_from_bridge_messages() -> None:
    bridge_messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(),
        bridge_messages=bridge_messages,
        startup_timeout_ms=100,
    )

    async def publish_ready() -> None:
        await asyncio.sleep(0)
        await bridge_messages.put({"type": "overlay_ready"})

    publisher = asyncio.create_task(publish_ready())
    try:
        await manager.start()
        assert manager.state == "failed"
        assert manager.failure_reason == "startup_timeout"
        assert manager.native_retry_owner_confirmed is False
    finally:
        publisher.cancel()
        await asyncio.gather(publisher, return_exceptions=True)
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_maps_bridge_runtime_error_after_ready() -> None:
    bridge_messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(ready_event_delay_ms=0),
        bridge_messages=bridge_messages,
    )

    await manager.start()
    await bridge_messages.put(
        {
            "type": "runtime_error",
            "failure_reason": "runtime_disconnected",
        }
    )

    for _ in range(10):
        if manager.state == "failed":
            break
        await asyncio.sleep(0)

    assert manager.state == "failed"
    assert manager.failure_reason == "runtime_disconnected"


@pytest.mark.asyncio
async def test_overlay_process_manager_renderer_events_forwards_valid_window_bounds_from_bridge() -> (
    None
):
    bridge_messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    renderer_events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(ready_event_delay_ms=0),
        bridge_messages=bridge_messages,
        renderer_events=renderer_events,
    )
    event: dict[str, object] = {
        "type": "overlay_event",
        "payload": {
            "event": "window_bounds_changed",
            "source": "user",
            "persist": True,
            "x": 320,
            "y": 720,
            "width": 1280,
            "height": 330,
        },
    }

    try:
        await manager.start()
        await bridge_messages.put(event)

        forwarded = await asyncio.wait_for(renderer_events.get(), timeout=0.5)

        assert forwarded == event
        assert manager.state == "connected"
        assert manager.failure_reason is None
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_renderer_events_strips_legacy_window_bounds_epoch() -> None:
    bridge_messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    renderer_events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(ready_event_delay_ms=0),
        bridge_messages=bridge_messages,
        renderer_events=renderer_events,
    )
    event: dict[str, object] = {
        "type": "overlay_event",
        "payload": {
            "event": "window_bounds_changed",
            "source": "user",
            "persist": True,
            "x": 320,
            "y": 720,
            "width": 1280,
            "height": 330,
            "bounds_epoch": 2,
        },
    }

    try:
        await manager.start()
        await bridge_messages.put(event)

        forwarded = await asyncio.wait_for(renderer_events.get(), timeout=0.5)

        expected = copy.deepcopy(event)
        assert isinstance(expected["payload"], dict)
        expected["payload"].pop("bounds_epoch")
        assert forwarded == expected
        assert manager.state == "connected"
        assert manager.failure_reason is None
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_renderer_events_forwards_valid_mode_and_reset_events() -> (
    None
):
    bridge_messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    renderer_events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(ready_event_delay_ms=0),
        bridge_messages=bridge_messages,
        renderer_events=renderer_events,
    )
    mode_event: dict[str, object] = {
        "type": "overlay_event",
        "payload": {
            "event": "interaction_mode_changed",
            "mode": "pass_through",
        },
    }
    reset_event: dict[str, object] = {
        "type": "overlay_event",
        "payload": {"event": "reset_to_bottom_center_requested"},
    }

    try:
        await manager.start()
        await bridge_messages.put(mode_event)
        await bridge_messages.put(reset_event)

        forwarded_mode = await asyncio.wait_for(renderer_events.get(), timeout=0.5)
        forwarded_reset = await asyncio.wait_for(renderer_events.get(), timeout=0.5)

        assert forwarded_mode == mode_event
        assert forwarded_reset == reset_event
        assert manager.state == "connected"
        assert manager.failure_reason is None
    finally:
        await manager.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_message",
    [
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "programmatic",
                "persist": True,
                "x": 320,
                "y": 720,
                "width": 1280,
                "height": 330,
            },
        },
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 320,
                "y": 720,
                "width": 0,
                "height": 330,
            },
        },
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 320,
                "y": 720,
                "width": 1280,
                "height": 330,
                "bounds_epoch": -1,
            },
        },
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 320,
                "y": 720,
                "width": 1280,
                "height": 330,
                "bounds_epoch": True,
            },
        },
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 320,
                "y": 720,
                "width": 1280,
                "height": 330,
                "bounds_epoch": "2",
            },
        },
        {
            "type": "overlay_event",
            "payload": {
                "event": "interaction_mode_changed",
                "mode": "dragging",
            },
        },
        {
            "type": "overlay_event",
            "payload": {
                "event": "interaction_mode_changed",
                "mode": ["edit"],
            },
        },
        {
            "type": "overlay_event",
            "payload": {
                "event": "interaction_mode_changed",
                "mode": "edit",
                "persist": True,
            },
        },
        {
            "type": "overlay_event",
            "payload": {
                "event": "reset_to_bottom_center_requested",
                "source": "user",
                "persist": True,
            },
        },
        "not-a-renderer-message",
    ],
)
async def test_overlay_process_manager_renderer_events_ignores_invalid_messages_without_failure(
    invalid_message: object,
) -> None:
    bridge_messages: asyncio.Queue[object] = asyncio.Queue()
    renderer_events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(ready_event_delay_ms=0),
        bridge_messages=bridge_messages,  # type: ignore[arg-type]
        renderer_events=renderer_events,
    )

    try:
        await manager.start()
        await bridge_messages.put(invalid_message)
        await asyncio.sleep(0.05)

        assert renderer_events.empty()
        assert manager.state == "connected"
        assert manager.failure_reason is None
        assert manager._monitor_task is not None
        assert not manager._monitor_task.done()
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_renderer_events_without_queue_are_diagnostic_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bridge_messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(ready_event_delay_ms=0),
        bridge_messages=bridge_messages,
    )

    try:
        await manager.start()
        with caplog.at_level("INFO", logger="puripuly_heart.core.overlay.process"):
            await bridge_messages.put(
                {
                    "type": "overlay_event",
                    "payload": {
                        "event": "interaction_mode_changed",
                        "mode": "edit",
                    },
                }
            )
            await asyncio.sleep(0.05)

        assert manager.state == "connected"
        assert manager.failure_reason is None
        assert any(
            "Renderer event ignored without controller queue" in message
            for message in caplog.messages
        )
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_overlay_process_manager_writes_runtime_crash_dump_with_recent_child_output(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "overlay_stub_runtime_crash.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "assert sys.argv[1] == '--config'",
                'print("[overlay][WARN] warning-line-0", flush=True)',
                'print("[overlay][WARN] warning-line-1", flush=True)',
                "for index in range(105):",
                '    print(f"stdout-line-{index}", flush=True)',
                "for index in range(3):",
                '    print(f"stderr-line-{index}", file=sys.stderr, flush=True)',
                'print(\'{"type": "overlay_ready"}\', flush=True)',
                "raise SystemExit(1)",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    manager = OverlayProcessManager(
        process_runner=DefaultOverlayProcessRunner(executable_path=script_path),
        startup_timeout_ms=500,
        diagnostics_dir=tmp_path,
    )

    await manager.start()
    for _ in range(50):
        if manager.state == "failed":
            break
        await asyncio.sleep(0.01)

    assert manager.state == "failed"
    assert manager.failure_reason == "runtime_crashed"

    dump_files = sorted(tmp_path.glob("overlay-diagnostics-*.jsonl"))
    assert len(dump_files) == 1
    dump_rows = [
        json.loads(line) for line in dump_files[0].read_text(encoding="utf-8").splitlines()
    ]

    summary = dump_rows[0]
    assert summary["event"] == "failure_summary"
    assert summary["overlay_instance_id"] == manager.overlay_instance_id
    assert summary["phase"] == "connected"
    assert summary["exit_code"] == 1
    assert summary["failure_reason"] == "runtime_crashed"

    stdout_rows = [row for row in dump_rows if row.get("stream") == "stdout"]
    stderr_rows = [row for row in dump_rows if row.get("stream") == "stderr"]
    assert {row["line"] for row in stdout_rows} == {
        "[overlay][WARN] warning-line-0",
        "[overlay][WARN] warning-line-1",
    }
    assert {row["line"] for row in stderr_rows} == {
        "stderr-line-0",
        "stderr-line-1",
        "stderr-line-2",
    }


@pytest.mark.asyncio
async def test_overlay_process_manager_dump_marks_startup_phase_for_pre_ready_exit(
    tmp_path: Path,
) -> None:
    manager = OverlayProcessManager(
        process_runner=FakeProcessRunner(exit_code=21),
        diagnostics_dir=tmp_path,
    )

    await manager.start()

    assert manager.state == "failed"
    assert manager.failure_reason == "renderer_init_failed"

    dump_files = sorted(tmp_path.glob("overlay-diagnostics-*.jsonl"))
    assert len(dump_files) == 1
    summary = json.loads(dump_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert summary["event"] == "failure_summary"
    assert summary["phase"] == "startup"
    assert summary["exit_code"] == 21
    assert summary["failure_reason"] == "renderer_init_failed"


@pytest.mark.asyncio
async def test_retry_ownership_capability_is_conservative_and_renegotiable() -> None:
    changes: list[bool] = []

    async def ownership_changed(confirmed: bool) -> None:
        changes.append(confirmed)

    manager = OverlayProcessManager(retry_ownership_changed=ownership_changed)
    supported = {
        "type": "overlay_ready",
        "capabilities": {"native_presentation_retry": {"version": 1, "ownership": "exclusive"}},
    }
    assert await manager._handle_lifecycle_event(supported, allow_ready=True) == "ready"
    assert manager.native_retry_owner_confirmed is True
    assert await manager._handle_lifecycle_event(supported, allow_ready=False) == "ignored"
    assert changes == [True]
    assert manager.native_retry_owner_confirmed is True

    for payload in (
        {"type": "overlay_ready"},
        {"type": "overlay_ready", "capabilities": []},
        {
            "type": "overlay_ready",
            "capabilities": {"native_presentation_retry": {"version": 2, "ownership": "exclusive"}},
        },
        {
            "type": "overlay_ready",
            "capabilities": {
                "native_presentation_retry": {"version": True, "ownership": "exclusive"}
            },
        },
        {
            "type": "overlay_ready",
            "capabilities": {
                "native_presentation_retry": {"version": 1.0, "ownership": "exclusive"}
            },
        },
    ):
        await manager._handle_lifecycle_event(payload, allow_ready=True)
        assert manager.native_retry_owner_confirmed is False

    await manager._handle_lifecycle_event(supported, allow_ready=True)
    assert manager.native_retry_owner_confirmed is True
    await manager._fail("runtime_crashed", terminate_process=False)
    assert manager.native_retry_owner_confirmed is False
    assert changes == [True, False, True, False]


@pytest.mark.asyncio
async def test_overlay_ready_rejects_stale_instance_and_duplicate_generation() -> None:
    manager = OverlayProcessManager(overlay_instance_id="overlay-current")

    stale = await manager._handle_lifecycle_event(
        {
            "type": "overlay_ready",
            "overlay_instance_id": "overlay-retired",
            "generation": 1,
        },
        allow_ready=True,
    )
    accepted = await manager._handle_lifecycle_event(
        {
            "type": "overlay_ready",
            "overlay_instance_id": "overlay-current",
            "generation": 2,
        },
        allow_ready=True,
    )
    duplicate = await manager._handle_lifecycle_event(
        {
            "type": "overlay_ready",
            "overlay_instance_id": "overlay-current",
            "generation": 2,
        },
        allow_ready=True,
    )

    assert (stale, accepted, duplicate) == ("ignored", "ready", "ignored")
    assert manager._accepted_ready_generation == 2
    assert manager.diagnostics is not None
    rejected = [
        event
        for event in manager.diagnostics.process_events
        if event.get("event") == "renderer_message_ignored"
    ]
    assert {event.get("reason") for event in rejected} == {
        "stale_overlay_instance",
        "duplicate_ready_generation",
    }
    assert all(event["accepted"] is False for event in rejected)


@pytest.mark.asyncio
async def test_overlay_trace_records_complete_sanitized_generation_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = OverlayProcessManager(
        overlay_instance_id="overlay-trace",
        logging_mode="detailed",
        selected_target="desktop",
        fallback_reason="steamvr_not_running",
        geometry_authority="flet",
    )

    with caplog.at_level("INFO", logger="puripuly_heart.core.overlay.process"):
        outcome = await manager._handle_lifecycle_event(
            {
                "type": "overlay_trace",
                "component": "desktop_window_startup",
                "event": "bounds_confirmed",
                "generation": 7,
                "phase": "bounds_confirmed",
                "monotonic_ms": 18.25,
                "accepted": True,
                "parent_pid": 2468,
                "canonical_bounds": {"x": 20, "y": 30, "width": 900, "height": 240},
                "observed_bounds": [20, 30, 900, 240],
            },
            allow_ready=False,
        )

    assert outcome == "ignored"
    assert manager.diagnostics is not None
    event = manager.diagnostics.process_events[-1]
    assert event["trace_component"] == "desktop_window_startup"
    assert event["trace_event"] == "bounds_confirmed"
    assert event["generation"] == 7
    assert event["source_monotonic_ms"] == 18.25
    assert event["selected_target"] == "desktop"
    assert event["fallback_reason"] == "steamvr_not_running"
    assert event["geometry_authority"] == "flet"
    assert event["parent_pid"] == 2468
    assert event["canonical_bounds"] == {"x": 20, "y": 30, "width": 900, "height": 240}
    assert event["observed_bounds"] == [20, 30, 900, 240]
    assert any('"trace_event": "bounds_confirmed"' in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_overlay_stop_drains_terminal_child_lifecycle_trace() -> None:
    class DrainingProcess:
        pid = 4321
        returncode = 0

        async def next_event(self) -> dict[str, object]:
            raise AssertionError("next_event should not be called")

        async def wait(self) -> int | None:
            return self.returncode

        async def terminate(self) -> None:
            return None

        def set_logging_mode(self, mode: str) -> None:
            _ = mode

        def drain_events(self) -> list[dict[str, object]]:
            return [
                {
                    "type": "overlay_trace",
                    "component": "flet_view_process",
                    "event": "pid_file_removed",
                    "generation": 1,
                    "monotonic_ms": 44.0,
                }
            ]

    manager = OverlayProcessManager(selected_target="desktop", geometry_authority="flet")
    manager._process = DrainingProcess()

    await manager.stop()

    assert manager.diagnostics is not None
    assert any(
        event.get("trace_event") == "pid_file_removed"
        for event in manager.diagnostics.process_events
    )


@pytest.mark.asyncio
async def test_connected_expected_exit_drains_all_terminal_child_traces() -> None:
    class ExitedProcess:
        pid = 4321
        returncode = 0

        def __init__(self) -> None:
            self.events = [
                {
                    "type": "overlay_trace",
                    "component": "flet_view_process",
                    "event": event,
                    "generation": 3,
                    "parent_pid": 9876,
                    "monotonic_ms": monotonic_ms,
                }
                for event, monotonic_ms in (
                    ("stop_requested", 41.0),
                    ("process_exited", 42.0),
                    ("pid_file_removed", 43.0),
                )
            ]

        async def next_event(self) -> dict[str, object]:
            return self.events.pop(0)

        async def wait(self) -> int | None:
            return self.returncode

        async def terminate(self) -> None:
            return None

        def set_logging_mode(self, mode: str) -> None:
            _ = mode

        def drain_events(self) -> list[dict[str, object]]:
            events = list(self.events)
            self.events.clear()
            return events

    process = ExitedProcess()
    manager = OverlayProcessManager(selected_target="desktop", geometry_authority="flet")
    manager.state = "connected"
    manager._process = process
    manager.mark_shutdown_requested()

    await manager._monitor_connected_process()
    await manager.stop()

    assert manager.diagnostics is not None
    child_events = [
        event.get("trace_event")
        for event in manager.diagnostics.process_events
        if event.get("trace_component") == "flet_view_process"
    ]
    assert child_events == ["stop_requested", "process_exited", "pid_file_removed"]
    assert process.events == []


@pytest.mark.asyncio
async def test_window_bounds_event_rejects_generation_other_than_ready_generation() -> None:
    events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    manager = OverlayProcessManager(
        overlay_instance_id="overlay-current",
        renderer_events=events,
    )
    await manager._handle_lifecycle_event(
        {
            "type": "overlay_ready",
            "overlay_instance_id": "overlay-current",
            "generation": 4,
        },
        allow_ready=True,
    )

    def bounds_event(generation: int) -> dict[str, object]:
        return {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "generation": generation,
                "x": 100,
                "y": 200,
                "width": 900,
                "height": 240,
            },
        }

    await manager._handle_lifecycle_event(bounds_event(3), allow_ready=False)
    assert events.empty()

    current = bounds_event(4)
    await manager._handle_lifecycle_event(current, allow_ready=False)
    assert await events.get() == current


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "fail"])
async def test_retry_fallback_waits_for_process_termination(operation: str) -> None:
    termination_started = asyncio.Event()
    release_termination = asyncio.Event()
    ownership_changes: list[bool] = []

    class BlockingProcess:
        async def terminate(self) -> None:
            termination_started.set()
            await release_termination.wait()

    async def ownership_changed(confirmed: bool) -> None:
        ownership_changes.append(confirmed)

    manager = OverlayProcessManager(retry_ownership_changed=ownership_changed)
    manager._process = BlockingProcess()  # type: ignore[assignment]
    manager.native_retry_owner_confirmed = True
    manager.state = "connected"
    task = asyncio.create_task(
        manager.stop() if operation == "stop" else manager._fail("runtime_crashed")
    )
    await termination_started.wait()
    assert manager.native_retry_owner_confirmed is True
    assert ownership_changes == []
    release_termination.set()
    await task
    assert manager.native_retry_owner_confirmed is False
    assert ownership_changes == [False]
    assert manager._process is None


@pytest.mark.asyncio
async def test_start_force_notifies_new_listener_when_manager_state_is_already_false() -> None:
    changes: list[bool] = []

    async def ownership_changed(confirmed: bool) -> None:
        changes.append(confirmed)

    runner = FakeProcessRunner(ready_event_delay_ms=0)
    manager = OverlayProcessManager(
        process_runner=runner,
        retry_ownership_changed=ownership_changed,
        startup_timeout_ms=100,
    )
    assert manager.native_retry_owner_confirmed is False
    await manager.start()
    assert changes[0] is False
    assert manager.state == "connected"
    await manager.stop()
