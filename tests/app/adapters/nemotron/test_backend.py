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
