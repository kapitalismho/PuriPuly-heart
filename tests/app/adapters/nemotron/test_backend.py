from __future__ import annotations

import asyncio
import io
import subprocess
from pathlib import Path

import numpy as np
import pytest

from puripuly_heart.app.adapters.nemotron import NemotronBackend
from puripuly_heart.app.adapters.nemotron.protocol import decode_header, encode, read_message


def readonly(size: int, value: float = 0.0):
    result = np.full(size, value, dtype=np.float32)
    result.flags.writeable = False
    return result


@pytest.fixture
def worker(monkeypatch):
    original = asyncio.create_subprocess_exec
    processes = []

    async def spawn(executable, *args, **kwargs):
        child = await original(
            executable, str(Path(__file__).with_name("fake_worker.py")), **kwargs
        )
        processes.append(child)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    return processes


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["1.04", "0.64"])
async def test_persistent_pause_resume_and_repeated_reset(worker, tmp_path, profile):
    backend = NemotronBackend(
        artifact_path=tmp_path / "model.gguf", runtime_dir=tmp_path, device=0, profile=profile
    )
    ready = await backend.start()
    assert ready.max_revision_samples == ready.tail_padding_samples
    process = worker[0]
    cursor = 0
    observed = 0
    for _ in range(200):
        update = await backend.feed(readonly(1600), cursor)
        cursor += 1600
        assert update.processed_through_sample <= cursor
        assert cursor - update.immutable_through_sample <= ready.max_revision_samples
        assert len(update.frames) <= 1024
        assert all(
            len(frame.probabilities) == 8 and frame.active_slots == () for frame in update.frames
        )
        observed += len(update.frames)
    assert observed > 1000
    before = update
    pause = await backend.pause()
    assert pause.processed_through_sample == before.processed_through_sample
    assert pause.immutable_through_sample == before.immutable_through_sample
    assert pause.frames == ()
    await backend.feed(readonly(1600), cursor)
    for epoch in range(1, 11):
        await backend.reset(epoch)
        update = await backend.feed(readonly(1600), 0)
        assert update.immutable_through_sample == 0
        assert backend._process is process
    assert len(backend._stderr) <= 16384
    await backend.stop()
    assert process.returncode is not None
    await backend.stop()


@pytest.mark.asyncio
async def test_unavailable_device_no_fallback(worker, tmp_path):
    backend = NemotronBackend(
        artifact_path=tmp_path / "m", runtime_dir=tmp_path, device=99, profile="0.64"
    )
    with pytest.raises(RuntimeError, match="no selected Vulkan GPU"):
        await backend.start()
    assert worker[0].returncode is not None
    assert backend._ready is None


@pytest.mark.asyncio
async def test_stop_and_cancel_inflight_own_child_only(worker, tmp_path):
    for action in ("stop", "cancel"):
        backend = NemotronBackend(
            artifact_path=tmp_path / "m", runtime_dir=tmp_path, device=0, profile="0.64"
        )
        await backend.start()
        process = worker[-1]
        task = asyncio.create_task(backend.feed(readonly(1600, 99), 0))
        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError, match="already in flight"):
            await backend.pause()
        if action == "stop":
            await backend.stop()
        else:
            task.cancel()
        with pytest.raises((asyncio.CancelledError, asyncio.IncompleteReadError)):
            await task
        assert process.returncode is not None
        assert backend._process is None
        await backend.stop()


@pytest.mark.asyncio
async def test_stop_during_process_creation(worker, monkeypatch, tmp_path):
    original = asyncio.create_subprocess_exec
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)
    backend = NemotronBackend(
        artifact_path=tmp_path / "m", runtime_dir=tmp_path, device=0, profile="1.04"
    )
    startup = asyncio.create_task(backend.start())
    await entered.wait()
    stopped = asyncio.create_task(backend.stop())
    release.set()
    await stopped
    with pytest.raises(asyncio.CancelledError):
        await startup
    assert worker[0].returncode is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["kill", "wait"])
async def test_failed_stop_requires_real_cleanup_before_replacement(
    worker, monkeypatch, tmp_path, failure
):
    original_spawn = asyncio.create_subprocess_exec
    stderr_settled = asyncio.Event()
    attempts = 0

    async def spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        if len(worker) != 1:
            return process
        original_read = process.stderr.read

        async def read(*args):
            block = await original_read(*args)
            if not block:
                stderr_settled.set()
            return block

        monkeypatch.setattr(process.stderr, "read", read)
        if failure == "kill":
            original_kill = process.kill

            def kill():
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("controlled kill failure")
                original_kill()

            monkeypatch.setattr(process, "kill", kill)
        else:
            original_wait = process.wait

            async def wait():
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("controlled wait failure")
                return await original_wait()

            monkeypatch.setattr(process, "wait", wait)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    backend = NemotronBackend(
        artifact_path=tmp_path / "m", runtime_dir=tmp_path, device=0, profile="0.64"
    )
    try:
        await backend.start()
        process = worker[0]
        with pytest.raises(OSError, match=f"controlled {failure} failure"):
            await backend.stop()
        if failure == "kill":
            assert process.returncode is None
        with pytest.raises(RuntimeError, match="cleanup pending"):
            await backend.start()
        assert len(worker) == 1
        await asyncio.wait_for(backend.stop(), timeout=5)
        assert attempts == 2
        assert process.returncode is not None
        assert stderr_settled.is_set()
        await backend.start()
        assert len(worker) == 2
        await backend.stop()
        assert worker[1].returncode is not None
    finally:
        for process in worker:
            if process.returncode is None:
                process.kill()
            await process.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["kill", "wait"])
async def test_cancelled_spawn_cleanup_failure_retries_same_child(
    worker, monkeypatch, tmp_path, failure
):
    original_spawn = asyncio.create_subprocess_exec
    entered = asyncio.Event()
    release = asyncio.Event()
    stderr_settled = asyncio.Event()
    attempts = 0

    async def spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        process = await original_spawn(*args, **kwargs)
        original_read = process.stderr.read

        async def read(*args):
            block = await original_read(*args)
            if not block:
                stderr_settled.set()
            return block

        monkeypatch.setattr(process.stderr, "read", read)
        if failure == "kill":
            original_kill = process.kill

            def kill():
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("controlled kill failure")
                original_kill()

            monkeypatch.setattr(process, "kill", kill)
        else:
            original_wait = process.wait

            async def wait():
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("controlled wait failure")
                return await original_wait()

            monkeypatch.setattr(process, "wait", wait)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    backend = NemotronBackend(
        artifact_path=tmp_path / "m", runtime_dir=tmp_path, device=0, profile="0.64"
    )
    startup = asyncio.create_task(backend.start())
    try:
        await entered.wait()
        startup.cancel()
        release.set()
        with pytest.raises(OSError, match=f"controlled {failure} failure"):
            await startup
        process = worker[0]
        if failure == "kill":
            assert process.returncode is None
        with pytest.raises(RuntimeError, match="cleanup pending"):
            await backend.start()
        assert len(worker) == 1
        await asyncio.wait_for(backend.stop(), timeout=5)
        assert attempts == 2
        assert process.returncode is not None
        assert stderr_settled.is_set()
        await backend.stop()
    finally:
        release.set()
        await asyncio.gather(startup, return_exceptions=True)
        for process in worker:
            if process.returncode is None:
                process.kill()
            await process.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["wait", "stderr"])
async def test_cancelled_stop_keeps_cleanup_owned_for_retry(
    worker, monkeypatch, tmp_path, boundary
):
    original_spawn = asyncio.create_subprocess_exec
    entered = asyncio.Event()
    release = asyncio.Event()
    stderr_settled = asyncio.Event()
    wait_attempts = 0

    async def spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        original_read = process.stderr.read
        original_wait = process.wait

        async def read(*args):
            block = await original_read(*args)
            if not block:
                if boundary == "stderr":
                    entered.set()
                    await release.wait()
                stderr_settled.set()
            return block

        async def wait():
            nonlocal wait_attempts
            wait_attempts += 1
            if boundary == "wait":
                entered.set()
                await release.wait()
            return await original_wait()

        monkeypatch.setattr(process.stderr, "read", read)
        monkeypatch.setattr(process, "wait", wait)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    backend = NemotronBackend(
        artifact_path=tmp_path / "m", runtime_dir=tmp_path, device=0, profile="0.64"
    )
    stopped = None
    retry = None
    try:
        await backend.start()
        stopped = asyncio.create_task(backend.stop())
        await asyncio.wait_for(entered.wait(), timeout=5)
        retry = asyncio.create_task(backend.stop())
        await asyncio.sleep(0)
        assert wait_attempts == 1
        stopped.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopped
        with pytest.raises(RuntimeError, match="cleanup pending"):
            await backend.start()
        assert len(worker) == 1
        if boundary == "stderr":
            await asyncio.sleep(0)
            assert not retry.done()
        release.set()
        await asyncio.wait_for(retry, timeout=5)
        assert wait_attempts == 2
        assert worker[0].returncode is not None
        assert stderr_settled.is_set()
        await backend.stop()
    finally:
        release.set()
        for task in (stopped, retry):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        for process in worker:
            if process.returncode is None:
                process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_repeated_start_cancellation_retains_pending_spawn(worker, monkeypatch, tmp_path):
    original_spawn = asyncio.create_subprocess_exec
    entered = asyncio.Event()
    release = asyncio.Event()
    stderr_settled = asyncio.Event()

    async def spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        process = await original_spawn(*args, **kwargs)
        original_read = process.stderr.read

        async def read(*args):
            block = await original_read(*args)
            if not block:
                stderr_settled.set()
            return block

        monkeypatch.setattr(process.stderr, "read", read)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    backend = NemotronBackend(
        artifact_path=tmp_path / "m", runtime_dir=tmp_path, device=0, profile="0.64"
    )
    startup = asyncio.create_task(backend.start())
    try:
        await entered.wait()
        startup.cancel()
        await asyncio.sleep(0)
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup
        with pytest.raises(RuntimeError, match="cleanup pending"):
            await backend.start()
        assert worker == []
        release.set()
        await asyncio.wait_for(backend.stop(), timeout=5)
        assert len(worker) == 1
        assert worker[0].returncode is not None
        assert stderr_settled.is_set()
    finally:
        release.set()
        startup.cancel()
        await asyncio.gather(startup, return_exceptions=True)
        for process in worker:
            if process.returncode is None:
                process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_reject_invalid_input_before_ipc(worker, tmp_path):
    backend = NemotronBackend(
        artifact_path=tmp_path / "m", runtime_dir=tmp_path, device=0, profile="0.64"
    )
    await backend.start()
    for pcm, start in (
        (np.zeros(160, dtype=np.float32), 0),
        (readonly(16001), 0),
        (readonly(160), 1),
        (readonly(160, float("nan")), 0),
    ):
        with pytest.raises(ValueError):
            await backend.feed(pcm, start)
    await backend.feed(readonly(160), 0)
    await backend.stop()


def test_bounded_protocol_rejects_truncation_and_oversize():
    assert read_message(io.BytesIO(encode({"id": 7}, b"abcd"))) == ({"id": 7}, b"abcd")
    with pytest.raises(ValueError):
        encode({}, bytes(65537))
    with pytest.raises(ValueError):
        encode({"value": "x" * 8193})
    with pytest.raises(ValueError):
        decode_header(b"\xff" * 8)
    with pytest.raises(ValueError):
        read_message(io.BytesIO(encode({}, b"abcd")[:-1]))


@pytest.mark.parametrize(
    "device,profile", [(-1, "1.04"), (True, "0.64"), (0, "default"), (0, "v3-streaming")]
)
def test_no_implicit_device_or_profile(device, profile, tmp_path):
    with pytest.raises(ValueError):
        NemotronBackend(
            artifact_path=tmp_path / "m", runtime_dir=tmp_path, device=device, profile=profile
        )


def test_native_worker_entrypoint_is_not_eagerly_imported():
    from puripuly_heart.runtime_layout import current_runtime_layout

    layout = current_runtime_layout()
    environment = layout.python_child_environment()
    if layout.host_kind == "source":
        environment["PYTHONPATH"] = str(layout.app_resource_root / "src")
    result = subprocess.run(
        [str(layout.python_executable), "-m", "puripuly_heart.app.adapters.nemotron.worker"],
        input=b"",
        capture_output=True,
        env=environment,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert b"RuntimeWarning" not in result.stderr
    assert result.stdout == b""
