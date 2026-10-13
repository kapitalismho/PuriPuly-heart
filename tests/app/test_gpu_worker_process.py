from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import pytest

import puripuly_heart.app.adapters.gpu_worker_process as gpu_worker_process_module
from puripuly_heart.app.adapters.gpu_worker_process import (
    DefaultGpuWorkerProcessFactory,
)
from puripuly_heart.app.ports.gpu_worker import (
    GpuWorkerClosedError,
    GpuWorkerRequestError,
)

FAKE_WORKER = Path(__file__).parents[1] / "fixtures" / "fake_gpu_worker.py"


def _factory(**overrides: object) -> DefaultGpuWorkerProcessFactory:
    values = {
        "executable_path": FAKE_WORKER,
        "command_prefix": (sys.executable,),
        "startup_timeout_s": 1.0,
        "request_timeout_s": 1.0,
        "heartbeat_interval_ms": 100,
        "heartbeat_timeout_s": 0.5,
        "cooperative_shutdown_s": 0.2,
        "terminate_grace_s": 0.5,
        **overrides,
    }
    return DefaultGpuWorkerProcessFactory(**values)


async def test_authenticated_local_process_discovers_and_shuts_down() -> None:
    client = await asyncio.wait_for(_factory().start(mode="discovery"), timeout=2.0)
    temporary_root = client.temporary_directory_path

    event = await asyncio.wait_for(client.next_event(), timeout=1.0)
    devices = await asyncio.wait_for(client.discover(), timeout=1.0)

    assert event.name == "startup"
    assert len(devices) == 1
    assert devices[0].device_id == "vulkan:0"
    assert client.pid is not None
    assert temporary_root.is_dir()

    await asyncio.wait_for(client.close(), timeout=2.0)
    assert client.is_closed
    assert not temporary_root.exists()


async def test_cancel_response_preserves_decode_only_failure_fields(
    tmp_path: Path,
) -> None:
    client = await _factory().start(mode="persistent")
    await client.next_event()
    await client.activate(model_path=tmp_path / "model.gguf", device_id="vulkan:0")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    transcribe = asyncio.create_task(
        client.transcribe(
            request_id="decode-1",
            channel="peer",
            audio_path=audio_path,
        )
    )
    started = await client.next_event()
    assert started.name == "transcribe_started"
    assert started.fields["audio_seconds"] == 1.0
    await client.cancel("decode-1")

    with pytest.raises(GpuWorkerRequestError) as error:
        await transcribe
    assert error.value.code == "cancelled"
    assert error.value.attempt_started is True
    assert error.value.fields == {
        "audio_seconds": 1.0,
        "decode_seconds": 0.25,
        "rtf": 0.25,
    }
    await client.close()


async def test_prestart_invalid_audio_has_no_started_event_or_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GPU_WORKER_PRESTART_AUDIO_INVALID", "1")
    client = await _factory().start(mode="persistent")
    await client.next_event()
    await client.activate(model_path=tmp_path / "model.gguf", device_id="vulkan:0")
    audio_path = tmp_path / "invalid.wav"
    audio_path.write_bytes(b"invalid")

    with pytest.raises(GpuWorkerRequestError) as error:
        await client.transcribe(
            request_id="invalid-audio",
            channel="self",
            audio_path=audio_path,
        )

    assert error.value.code == "audio_invalid"
    assert error.value.attempt_started is False
    assert error.value.fields == {"channel": "self", "backend": "Vulkan"}
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(client.next_event(), timeout=0.05)
    await client.close()


async def test_started_decode_failure_has_event_and_finite_exact_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GPU_WORKER_STARTED_FAILURE", "1")
    client = await _factory().start(mode="persistent")
    await client.next_event()
    await client.activate(model_path=tmp_path / "model.gguf", device_id="vulkan:0")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fixture")

    transcribe = asyncio.create_task(
        client.transcribe(
            request_id="decode-failure",
            channel="peer",
            audio_path=audio_path,
        )
    )
    started = await client.next_event()
    with pytest.raises(GpuWorkerRequestError) as error:
        await transcribe

    assert started.name == "transcribe_started"
    assert started.request_id == "decode-failure"
    assert started.fields["audio_seconds"] == 1.0
    assert error.value.attempt_started is True
    assert error.value.fields["audio_seconds"] == 1.0
    assert error.value.fields["decode_seconds"] == 0.25
    assert error.value.fields["rtf"] == (
        error.value.fields["decode_seconds"] / error.value.fields["audio_seconds"]
    )
    await client.close()


async def test_started_decode_failure_does_not_log_worker_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("FAKE_GPU_WORKER_STARTED_FAILURE", "1")
    monkeypatch.setenv("FAKE_GPU_WORKER_STDERR_ON_FAILURE", "1")
    client = await _factory().start(mode="persistent")
    try:
        await client.next_event()
        await client.activate(model_path=tmp_path / "model.gguf", device_id="vulkan:0")
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fixture")

        with caplog.at_level(logging.ERROR, logger=gpu_worker_process_module.logger.name):
            with pytest.raises(GpuWorkerRequestError) as error:
                await client.transcribe(
                    request_id="stderr-decode-failure",
                    channel="peer",
                    audio_path=audio_path,
                )

        assert error.value.code == "decode_failure"
        assert "stderr-decode-failure" not in caplog.text
        assert "invalid token state" not in caplog.text
    finally:
        await client.close()


async def test_started_decode_success_has_event_and_finite_exact_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GPU_WORKER_STARTED_SUCCESS", "1")
    client = await _factory().start(mode="persistent")
    await client.next_event()
    await client.activate(model_path=tmp_path / "model.gguf", device_id="vulkan:0")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fixture")

    transcribe = asyncio.create_task(
        client.transcribe(
            request_id="decode-success",
            channel="self",
            audio_path=audio_path,
        )
    )
    started = await client.next_event()
    result = await transcribe

    assert started.name == "transcribe_started"
    assert started.request_id == "decode-success"
    assert started.fields["audio_seconds"] == 1.0
    assert result.audio_seconds == 1.0
    assert result.decode_seconds == 0.2
    assert result.rtf == result.decode_seconds / result.audio_seconds
    await client.close()


async def test_transcribe_sends_optional_language_hint_in_contract_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GPU_WORKER_STARTED_SUCCESS", "1")
    client = await _factory().start(mode="persistent")
    await client.next_event()
    await client.activate(model_path=tmp_path / "model.gguf", device_id="vulkan:0")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fixture")

    transcribe = asyncio.create_task(
        client.transcribe(
            request_id="hinted-decode",
            channel="peer",
            audio_path=audio_path,
            language_hint="ja",
        )
    )
    await client.next_event()
    result = await transcribe

    assert result.text == "ja"
    await client.close()


async def test_close_forces_process_termination_after_cooperative_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GPU_WORKER_IGNORE_SHUTDOWN", "1")
    client = await _factory(cooperative_shutdown_s=0.05).start(mode="persistent")
    await client.next_event()

    await asyncio.wait_for(client.close(), timeout=2.0)

    assert client.returncode is not None


async def test_invalid_authentication_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GPU_WORKER_BAD_AUTH", "1")

    with pytest.raises(GpuWorkerClosedError, match="authentication"):
        await _factory(startup_timeout_s=0.25).start(mode="discovery")


async def test_process_exit_before_authentication_fails_without_waiting_for_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GPU_WORKER_EXIT_BEFORE_AUTH", "1")
    monkeypatch.setattr(gpu_worker_process_module.tempfile, "tempdir", str(tmp_path))
    started_at = asyncio.get_running_loop().time()

    with pytest.raises(GpuWorkerClosedError, match="code 23"):
        await _factory(startup_timeout_s=5.0).start(mode="discovery")

    assert asyncio.get_running_loop().time() - started_at < 2.0
    assert tuple(tmp_path.iterdir()) == ()


async def test_failed_termination_keeps_incomplete_client_and_independent_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GPU_WORKER_IGNORE_SHUTDOWN", "1")
    client = await _factory(cooperative_shutdown_s=0.01).start(mode="persistent")
    await client.next_event()
    terminate = gpu_worker_process_module._terminate_process

    async def fail_terminate(*args: object, **kwargs: object) -> None:
        raise OSError(5, "forbidden-termination-path-token")

    monkeypatch.setattr(gpu_worker_process_module, "_terminate_process", fail_terminate)
    try:
        with pytest.raises(GpuWorkerClosedError) as caught:
            await asyncio.wait_for(client.close(), timeout=2.0)
        receipt = client.lifecycle_receipt()
        assert caught.value.code == "worker_cleanup_failed"
        assert caught.value.cleanup_complete is False
        assert client.is_closed is False
        assert receipt["close_complete"] is False
        assert receipt["writer_closed"] is True
        assert receipt["server_closed"] is True
        assert receipt["tasks_closed"] is True
        assert caught.value.cleanup_failures[0]["errno"] == 5
        assert "forbidden-termination-path-token" not in repr(receipt)
    finally:
        monkeypatch.setattr(gpu_worker_process_module, "_terminate_process", terminate)
        await asyncio.wait_for(client.force_close(), timeout=2.0)
    assert client.is_closed
    assert client.returncode is not None


async def test_startup_cause_survives_failed_cleanup_and_factory_retries_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GPU_WORKER_SKIP_AUTH", "1")
    factory = _factory(startup_timeout_s=0.1)
    terminate = gpu_worker_process_module._terminate_process

    async def fail_terminate(*args: object, **kwargs: object) -> None:
        raise PermissionError(5, "forbidden-startup-cleanup-body")

    monkeypatch.setattr(gpu_worker_process_module, "_terminate_process", fail_terminate)
    try:
        with pytest.raises(GpuWorkerClosedError) as caught:
            await factory.start(mode="discovery")
        assert caught.value.code == "authentication_timeout"
        assert caught.value.cleanup_complete is False
        assert caught.value.cleanup_failures[0]["cleanup_stage"] == "process_terminate"
        assert caught.value.cleanup_failures[0]["errno"] == 5
        assert factory.startup_cleanup_snapshot()["pending_startups"] == 1
        with pytest.raises(GpuWorkerClosedError):
            await factory.close_pending_startups()
        assert factory.startup_cleanup_snapshot()["close_complete"] is False
        with pytest.raises(GpuWorkerClosedError) as rejected:
            await factory.start(mode="discovery")
        assert rejected.value.code == "startup_cleanup_failed"
        assert factory.startup_cleanup_snapshot()["pending_startups"] == 1
        assert "forbidden-startup-cleanup-body" not in repr(factory.startup_cleanup_snapshot())
    finally:
        monkeypatch.setattr(gpu_worker_process_module, "_terminate_process", terminate)
        await asyncio.wait_for(factory.close_pending_startups(), timeout=2.0)
    await factory.close_pending_startups()
    assert factory.startup_cleanup_snapshot()["close_complete"] is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file-sharing cleanup contract")
@pytest.mark.skipif(os.getenv("INTEGRATION") != "1", reason="requires real owned child")
async def test_real_worker_activation_failure_retains_locked_audio_directory_until_close(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import ctypes
    from ctypes import wintypes
    from uuid import uuid4

    import psutil

    import puripuly_heart.core.runtime.gpu_asr as gpu_asr_module
    from puripuly_heart.core.diagnostics import DIAGNOSTIC_RECORD_ATTRIBUTE
    from puripuly_heart.core.runtime.gpu_asr import (
        GpuASRManualRetryRequired,
        GpuASRRuntimeState,
        SharedGpuASRRuntime,
    )

    for name in tuple(os.environ):
        if any(
            part in name.upper() for part in ("API_KEY", "AUTH_TOKEN", "ACCESS_TOKEN", "CREDENTIAL")
        ):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FAKE_GPU_WORKER_ACTIVATION_FAILURE", "1")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    original_directory = gpu_asr_module.tempfile.TemporaryDirectory
    locks: list[object] = []
    owned_paths: list[Path] = []

    def create_owned_directory(*args: object, **kwargs: object) -> object:
        directory = original_directory(*args, **kwargs)
        if kwargs.get("prefix") == "puripuly-gpu-asr-audio-":
            path = Path(directory.name) / "owned-cleanup-lock"
            path.write_bytes(b"")
            handle = create_file(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
            if handle in (None, ctypes.c_void_p(-1).value):
                raise ctypes.WinError(ctypes.get_last_error())
            locks.append(handle)
            owned_paths.append(Path(directory.name))
        return directory

    monkeypatch.setattr(gpu_asr_module.tempfile, "TemporaryDirectory", create_owned_directory)
    original_start = DefaultGpuWorkerProcessFactory.start
    children: list[tuple[object, float]] = []

    async def track_start(
        factory: DefaultGpuWorkerProcessFactory,
        *,
        mode: str,
        operation_id: str | None = None,
        request_id: str | None = None,
    ) -> object:
        child = await original_start(
            factory,
            mode=mode,
            operation_id=operation_id,
            request_id=request_id,
        )
        children.append((child, psutil.Process(child.pid).create_time()))
        return child

    monkeypatch.setattr(DefaultGpuWorkerProcessFactory, "start", track_start)
    factory = _factory()
    runtime = SharedGpuASRRuntime(process_factory=factory)
    operation_id, request_id = uuid4().hex, uuid4().hex
    try:
        with pytest.raises(GpuASRManualRetryRequired) as caught:
            await asyncio.wait_for(
                runtime.activate_channel(
                    "self",
                    owner=object(),
                    model_path=Path("synthetic-unused-model.gguf"),
                    model_id="qwen-gpu",
                    device_id="vulkan:0",
                    operation_id=operation_id,
                    request_id=request_id,
                ),
                timeout=5,
            )
        assert caught.value.code == "out_of_memory"
        assert isinstance(caught.value.__cause__, GpuWorkerRequestError)
        primary = caught.value.failure_diagnostics
        assert primary is not None
        assert primary.operation_id == operation_id and primary.request_id == request_id
        assert primary.fields["native_classification"] == "out_of_memory"
        assert runtime.state == GpuASRRuntimeState.FAILED
        assert runtime.startup_cleanup_snapshot["pending_temporary_directories"] == 1
        assert owned_paths[0].exists()
        secondary = next(
            getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)[0]
            for record in caplog.records
            if hasattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
            and getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)[0].diagnostic_id
            == runtime.startup_cleanup_snapshot["temporary_cleanup_diagnostic_id"]
        )
        assert secondary.cause_id == primary.diagnostic_id
        assert secondary.fields["exception_class"] == "PermissionError"
        assert secondary.fields["errno"] == 13
        assert str(owned_paths[0]) not in caplog.text
        assert len(children) == 1 and children[0][0].is_closed
        child, create_time = children[0]
        try:
            assert psutil.Process(child.pid).create_time() != create_time
        except psutil.NoSuchProcess:
            pass
        for handle in locks:
            assert close_handle(handle)
        locks.clear()
        await asyncio.wait_for(runtime.close(), timeout=5)
        assert runtime.state == GpuASRRuntimeState.CLOSED
        assert runtime.startup_cleanup_snapshot["pending_temporary_directories"] == 0
        assert not owned_paths[0].exists()
    finally:
        for handle in locks:
            close_handle(handle)
        await asyncio.wait_for(runtime.close(), timeout=5)


async def test_first_real_startup_failure_is_retained_with_explicit_admission_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import uuid4

    from puripuly_heart.core.runtime.gpu_asr import GpuASRManualRetryRequired, SharedGpuASRRuntime
    from tests.core.test_diagnostic_delivery import finish, logging_service

    monkeypatch.setenv("FAKE_GPU_WORKER_EXIT_BEFORE_AUTH", "1")
    root, sinks, service, _basic = logging_service(tmp_path)
    monkeypatch.setattr(gpu_worker_process_module, "logger", root)
    operation_id, request_id = uuid4().hex, uuid4().hex
    runtime = SharedGpuASRRuntime(process_factory=_factory())
    try:
        with pytest.raises(GpuASRManualRetryRequired) as caught:
            await asyncio.wait_for(
                runtime.activate_channel(
                    "self",
                    owner=object(),
                    model_path=tmp_path / "unused-model.gguf",
                    model_id="qwen-gpu",
                    device_id="vulkan:0",
                    operation_id=operation_id,
                    request_id=request_id,
                ),
                timeout=5,
            )
        primary = caught.value.failure_diagnostics
        assert primary is not None
        retained = service.diagnostic_history({"operation_id": operation_id})["records"]
        assert len(retained) == 1
        assert retained[0]["diagnostic_id"] == primary.diagnostic_id
        assert retained[0]["request_id"] == request_id
        assert retained[0]["code"] == primary.code
        assert retained[0]["fields"]["exit_code"] == 23
        assert runtime.startup_cleanup_snapshot["pending_temporary_directories"] == 0
    finally:
        await runtime.close()
        finish(root, sinks, service)
